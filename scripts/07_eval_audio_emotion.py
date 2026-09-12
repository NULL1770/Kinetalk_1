"""Read-only Stage2 teacher / Stage3 audio classification diagnostic.

Evaluate deterministic centre crops on the manifest's actual splits.  Load the
Stage2 teacher independently: the nested teacher in Stage3 must never overwrite
the checkpoint being used as the Stage2 reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.data import B0ResidualDataset
from kinetalk_b0.models import Stage2Model, Stage3Model
from kinetalk_b0.utils import load_checkpoint, load_yaml, move_to_device, seed_everything


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QueryNeutralDataset(Dataset):
    """Only read the two branches actually used by this evaluation."""

    def __init__(self, base: B0ResidualDataset, indices: list[int]):
        self.base, self.indices = base, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | int]:
        index = self.indices[position]
        query = self.base._read(self.base.items[index])
        neutral_index, neutral_valid = self.base._neutral_index(index)
        neutral = self.base._read(self.base.items[neutral_index], start=query["crop_start"])
        self.base._attach_targets(query, neutral, neutral_valid)
        return {key: query[key] for key in (
            "audio_emotion", "mask", "residual_gt", "residual_mask", "emotion_id", "intensity_id"
        )}


def classification_metrics(target: torch.Tensor, prediction: torch.Tensor, names: list[str]) -> dict[str, Any]:
    """Rows are true labels, columns predicted labels; macro excludes absent labels."""
    classes = len(names)
    confusion = torch.bincount(target * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    support = confusion.sum(1)
    present = support > 0
    recall = confusion.diag().double() / support.clamp_min(1)
    n = int(support.sum())
    majority_id = int(support.argmax()) if n else None
    balanced = float(recall[present].mean()) if present.any() else None
    return {
        "n": n,
        "accuracy": float(confusion.diag().sum() / n) if n else None,
        "balanced_accuracy": balanced,
        "macro_recall": balanced,
        "macro_recall_classes": int(present.sum()),
        "majority_baseline_accuracy": float(support.max() / n) if n else None,
        "majority_class": names[majority_id] if majority_id is not None else None,
        "uniform_guess_accuracy": 1.0 / classes,
        "confusion_orientation": "rows=true, columns=predicted",
        "class_order": names,
        "confusion": confusion.tolist(),
        "per_class": {name: {
            "n": int(support[i]),
            "predicted_n": int(confusion[:, i].sum()),
            "recall": float(recall[i]) if present[i] else None,
        } for i, name in enumerate(names)},
    }


def compare_state(first: torch.nn.Module, second: torch.nn.Module) -> dict[str, Any]:
    left, right = first.state_dict(), second.state_dict()
    common = sorted(set(left) & set(right))
    different = [key for key in common if not torch.equal(left[key], right[key])]
    return {
        "exact_equal": set(left) == set(right) and not different,
        "tensor_count": len(common),
        "differing_tensor_names": different,
        "only_stage2": sorted(set(left) - set(right)),
        "only_stage3_teacher": sorted(set(right) - set(left)),
    }


def checkpoint_info(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    digest_before = sha256(path)
    payload = load_checkpoint(path, model, map_location="cpu", strict=True)
    digest_after = sha256(path)
    if digest_before != digest_after:
        raise RuntimeError(f"Checkpoint changed during load: {path}")
    invalid = [name for name, tensor in model.state_dict().items() if not torch.isfinite(tensor).all()]
    if invalid:
        raise RuntimeError(f"Non-finite checkpoint tensors in {path}: {invalid}")
    return {"path": str(path.resolve()), "epoch": payload.get("epoch"),
            "stage": payload.get("stage"), "sha256": digest_after, "strict_load": True,
            "all_state_tensors_finite": True}


def select_indices(dataset: B0ResidualDataset, per_emotion: int, seed: int) -> list[int]:
    if per_emotion <= 0:
        return list(range(len(dataset)))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.items):
        grouped[dataset._emotion_name(record)].append(index)
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for name in dataset.emotion_classes:
        indices = grouped[name]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[i] for i in order[:per_emotion])
    return sorted(selected)


def dataset_metadata(dataset: B0ResidualDataset, indices: list[int]) -> dict[str, Any]:
    return {
        "available_n": len(dataset), "selected_n": len(indices),
        "speakers": sorted({dataset._group_key(record)[0] for record in dataset.items}),
        "sentences": sorted({dataset._group_key(record)[1] for record in dataset.items}),
        "selected_emotion_counts": {name: sum(dataset._emotion_name(dataset.items[i]) == name for i in indices)
                                    for name in dataset.emotion_classes},
    }


@torch.inference_mode()
def evaluate(loader: DataLoader, stage2: Stage2Model, stage3: Stage3Model,
             names: list[str], levels: int, device: torch.device, split: str) -> dict[str, Any]:
    saved: dict[str, list[torch.Tensor]] = defaultdict(list)
    seen, excluded, last_print = 0, 0, 0
    for batch in loader:
        batch = move_to_device(batch, device)
        valid = batch["residual_mask"].any(1) & batch["mask"].any(1)
        excluded += int((~valid).sum())
        if not valid.any():
            continue
        batch = {key: value[valid] for key, value in batch.items()}
        audio = stage3(batch["audio_emotion"], batch["mask"])
        teacher = stage2.emotion(batch["residual_gt"], batch["residual_mask"])
        for prefix, factors in (("stage3", audio), ("stage2", teacher)):
            for label in ("emotion", "intensity"):
                logits = factors[f"{label}_logits"]
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite {split} {prefix} {label} logits")
                saved[f"{prefix}_{label}_pred"].append(logits.argmax(-1).cpu())
                saved[f"{prefix}_{label}_ce"].append(F.cross_entropy(logits, batch[f"{label}_id"], reduction="none").cpu())
            saved[f"{prefix}_intensity_value"].append(factors["intensity_value"].flatten().cpu())
        cosine = F.cosine_similarity(audio["global"], teacher["global"], dim=-1)
        if not torch.isfinite(cosine).all():
            raise RuntimeError(f"Non-finite global cosine in {split}")
        saved["global_cosine"].append(cosine.cpu())
        for key in ("emotion_id", "intensity_id"):
            saved[key].append(batch[key].cpu())
        seen += len(batch["emotion_id"])
        if seen - last_print >= 512:
            print(f"{split}: evaluated {seen}/{len(loader.dataset)} clips", flush=True)
            last_print = seen
    if not seen:
        raise RuntimeError(f"No valid evaluation examples for {split}")
    values = {key: torch.cat(chunks) for key, chunks in saved.items()}
    result: dict[str, Any] = {"n": seen, "excluded_empty_masks": excluded, "stage3": {}, "stage2_teacher": {}}
    intensity_names = [str(i) for i in range(levels)]
    for prefix, output_key in (("stage3", "stage3"), ("stage2", "stage2_teacher")):
        for label, class_names in (("emotion", names), ("intensity", intensity_names)):
            metrics = classification_metrics(values[f"{label}_id"], values[f"{prefix}_{label}_pred"], class_names)
            metrics["cross_entropy"] = float(values[f"{prefix}_{label}_ce"].mean())
            result[output_key][label] = metrics
        result[output_key]["intensity"]["expected_scalar_mae"] = float(
            (values[f"{prefix}_intensity_value"] - values["intensity_id"]).abs().mean())
        result[output_key]["intensity"]["expected_scalar_units"] = f"ordinal label units, 0..{levels - 1}"
    result["stage3"]["global_cosine_to_separately_loaded_stage2"] = float(values["global_cosine"].mean())
    for label, name in enumerate(names):
        selection = values["emotion_id"] == label
        if selection.any():
            result["stage3"]["emotion"]["per_class"][name].update({
                "global_cosine_to_stage2": float(values["global_cosine"][selection].mean()),
                "intensity_accuracy": float((values["stage3_intensity_pred"][selection] == values["intensity_id"][selection]).float().mean()),
            })
    print(f"{split}: complete n={seen}, stage3 emotion={result['stage3']['emotion']['accuracy']:.6f}, "
          f"macro={result['stage3']['emotion']['macro_recall']:.6f}, "
          f"intensity={result['stage3']['intensity']['accuracy']:.6f}", flush=True)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="Combined JSON; sibling .<split>.json files are also saved")
    parser.add_argument("--splits", default="val,test,train")
    parser.add_argument("--per-emotion", type=int, default=0, help="0 evaluates every accepted clip")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.per_emotion < 0:
        parser.error("batch-size must be positive; workers and per-emotion must be nonnegative")
    splits = list(dict.fromkeys(name.strip() for name in args.splits.split(",") if name.strip()))
    if not splits or set(splits) - {"train", "val", "test"}:
        parser.error("splits must be a comma-separated subset of train,val,test")
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage2 = Stage2Model(cfg)
    stage2_info = checkpoint_info(Path(cfg["paths"]["stage2_ckpt"]), stage2)
    # This separate instance ensures loading Stage3 cannot mutate `stage2`.
    stage3 = Stage3Model(cfg, Stage2Model(cfg))
    stage3_info = checkpoint_info(Path(cfg["paths"]["stage3_ckpt"]), stage3)
    teacher_check = compare_state(stage2.emotion, stage3.teacher_emotion)
    stage2.to(device).eval()
    stage3.to(device).eval()
    report: dict[str, Any] = {
        "evaluation_mode": "internal_teacher_and_audio_prior_diagnostic",
        "independent_motion_recognizer": False,
        "generator_heldout_generalization_established": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config), "values": cfg},
        "checkpoints": {"stage2": stage2_info, "stage3": stage3_info},
        "stage3_embedded_teacher_vs_stage2": teacher_check,
        "protocol": {"device": str(device), "dtype": "float32, no autocast", "seed": seed,
                     "crop": "deterministic center crop", "window": cfg["data"]["window"],
                     "per_emotion_limit": args.per_emotion, "batch_size": args.batch_size,
                     "teacher_target": "query residual_gt = motion - masked neutral GT; intersected valid masks",
                     "notes": ["Training split is a fit diagnostic, not generalization evidence.",
                               "Macro recall excludes classes absent in the evaluated split.",
                               "Global cosine alone does not establish emotion/style disentanglement."]},
        "dataset_splits": {}, "split_overlap": {}, "results": {},
    }
    datasets: dict[str, B0ResidualDataset] = {}
    selected: dict[str, list[int]] = {}
    # Inspect all splits even when a smaller subset is selected for inference.
    for split in ("train", "val", "test"):
        try:
            dataset = B0ResidualDataset(cfg, split=split, random_crop=False)
        except ValueError as error:
            report["dataset_splits"][split] = {"error": str(error)}
            if split in splits:
                raise
            continue
        indices = select_indices(dataset, args.per_emotion, seed)
        report["dataset_splits"][split] = dataset_metadata(dataset, indices)
        datasets[split], selected[split] = dataset, indices
    for left in datasets:
        for right in datasets:
            if left >= right:
                continue
            left_meta, right_meta = report["dataset_splits"][left], report["dataset_splits"][right]
            speakers = sorted(set(left_meta["speakers"]) & set(right_meta["speakers"]))
            sentences = sorted(set(left_meta["sentences"]) & set(right_meta["sentences"]))
            clip_overlap = sorted({str(x["clip_id"]) for x in datasets[left].items} & {str(x["clip_id"]) for x in datasets[right].items})
            report["split_overlap"][f"{left}__{right}"] = {
                "speaker_disjoint": not speakers, "shared_speakers": speakers,
                "shared_sentence_ids": sentences, "clip_disjoint": not clip_overlap,
                "shared_clip_ids": clip_overlap,
            }
    print(f"Loaded Stage2 epoch={stage2_info['epoch']}, Stage3 epoch={stage3_info['epoch']}; "
          f"embedded teacher exact match={teacher_check['exact_equal']}", flush=True)
    for split in splits:
        loader = DataLoader(QueryNeutralDataset(datasets[split], selected[split]),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", persistent_workers=False)
        result = evaluate(loader, stage2, stage3, list(cfg["data"]["emotion_classes"]),
                          int(cfg["data"]["num_intensity_levels"]), device, split)
        report["results"][split] = result
        split_report = {key: value for key, value in report.items() if key != "results"}
        split_report.update({"split": split, "result": result})
        write_json(args.output.with_name(f"{args.output.stem}.{split}.json"), split_report)
        write_json(args.output, report)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
