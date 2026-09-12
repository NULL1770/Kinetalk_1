"""Independent, sentence-grouped linear leakage probes for frozen Stage-2 style.

This does not train or alter the experiment. Probe-held-out training clips are
NOT model-unseen evaluation. Stage-3/4 modules are never constructed or loaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.data import B0ResidualDataset
from kinetalk_b0.models import Stage2Model
from kinetalk_b0.utils import load_checkpoint, load_yaml, seed_everything


class QueryNeutralDataset(Dataset):
    """Load two branches instead of the training dataset's twelve reads."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        query = self.source._read(self.source.items[source_index])
        neutral_index, valid = self.source._neutral_index(source_index)
        neutral = self.source._read(
            self.source.items[neutral_index], start=query["crop_start"]
        )
        self.source._attach_targets(query, neutral, valid)
        return {
            key: query[key]
            for key in (
                "residual_gt", "residual_mask", "audio_emotion", "emotion_id",
                "intensity_id", "speaker", "sentence_id", "clip_id",
            )
        }


def sample_indices(dataset, limit, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for emotion in dataset.emotion_classes:
        indices = [
            i for i, record in enumerate(dataset.items)
            if dataset._emotion_name(record) == emotion
        ]
        if limit > 0 and len(indices) > limit:
            indices = rng.choice(indices, size=limit, replace=False).tolist()
        selected.extend(indices)
    return sorted(selected)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_style(cfg, checkpoint, device, audio_hint=None):
    model_cfg = copy.deepcopy(cfg)
    if audio_hint is not None:
        model_cfg["model"]["style_use_audio_hint"] = bool(audio_hint)
    model = Stage2Model(model_cfg)
    payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
        raise ValueError(f"Non-finite Stage-2 state: {checkpoint}")
    style = model.style.to(device).eval()
    style.requires_grad_(False)
    return style, {
        "checkpoint": str(Path(checkpoint).resolve()),
        "sha256": file_hash(checkpoint),
        "epoch": payload.get("epoch"),
        "strict_load": True,
        "style_use_audio_hint": style.use_reference_audio,
    }


@torch.inference_mode()
def extract(dataset, models, args, device, split):
    indices = sample_indices(dataset, args.per_emotion, args.seed)
    loader = DataLoader(
        QueryNeutralDataset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    collected = {name: [] for name in models}
    labels = {key: [] for key in ("emotion", "intensity", "speaker", "sentence", "clip")}
    skipped = 0
    for step, batch in enumerate(loader):
        mask = batch["residual_mask"].to(device)
        keep = mask.any(dim=1)
        skipped += int((~keep).sum().item())
        if not keep.any():
            continue
        residual = batch["residual_gt"].to(device)[keep]
        audio = batch["audio_emotion"].to(device)[keep]
        valid = keep.cpu().numpy().astype(bool)
        for name, model in models.items():
            style = model(residual, mask[keep], audio).float().cpu().numpy()
            if not np.isfinite(style).all():
                raise ValueError(f"Non-finite style features in {name}/{split}")
            collected[name].append(style)
        for target, source in (
            ("emotion", "emotion_id"), ("intensity", "intensity_id"),
            ("speaker", "speaker"), ("sentence", "sentence_id"), ("clip", "clip_id"),
        ):
            labels[target].extend(np.asarray(batch[source])[valid].tolist())
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == len(loader):
            print(f"extract {split}: {step + 1}/{len(loader)} batches", flush=True)
    if not labels["clip"]:
        raise ValueError(f"No valid neutral/query intersections in {split}")
    result = {key: np.asarray(value) for key, value in labels.items()}
    result.update({f"style_{key}": np.concatenate(value) for key, value in collected.items()})
    info = {
        "available_records": len(dataset), "selected_records": len(indices),
        "valid_records": len(result["clip"]), "skipped_empty_mask": skipped,
        "emotion_counts": dict(Counter(dataset._emotion_name(dataset.items[i]) for i in indices)),
        "sentence_groups": len(np.unique(result["sentence"])),
        "speakers": len(np.unique(result["speaker"])),
    }
    return result, info


def classification_metrics(target, prediction, classes):
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (target, prediction), 1)
    counts = confusion.sum(axis=1)
    supported = counts > 0
    recall = np.divide(confusion.diagonal(), counts, out=np.zeros(len(classes), float), where=supported)
    return {
        "n": len(target), "accuracy": float((target == prediction).mean()),
        "balanced_accuracy": float(recall[supported].mean()),
        "macro_recall": float(recall[supported].mean()),
        "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
        "per_class_recall": {str(c): float(r) if n else None for c, r, n in zip(classes, recall, counts)},
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "classes": list(map(str, classes)),
    }


def fit_linear(train_x, train_y, test_x, n_classes, args):
    """Deterministic CPU multinomial logistic regression, balanced CE + ridge."""
    x = torch.as_tensor(train_x, dtype=torch.float64)
    xt = torch.as_tensor(test_x, dtype=torch.float64)
    y = torch.as_tensor(train_y, dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-4)
    x, xt = (x - mean) / std, (xt - mean) / std
    counts = torch.bincount(y, minlength=n_classes).double()
    weights = counts.clamp_min(1).reciprocal()
    weights[counts == 0] = 0
    weights = weights / weights[weights > 0].mean()
    w = torch.zeros((x.shape[1], n_classes), dtype=x.dtype, requires_grad=True)
    bias = torch.zeros(n_classes, dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [w, bias], lr=1, max_iter=args.probe_iterations,
        tolerance_grad=1e-7, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ w + bias, y, weight=weights) + args.ridge * w.square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    if not torch.isfinite(w).all() or not torch.isfinite(bias).all():
        raise ValueError("Probe optimizer produced non-finite weights")
    with torch.no_grad():
        # Never predict a class missing from the probe fitting labels.
        train_logits, test_logits = x @ w + bias, xt @ w + bias
        train_logits[:, counts == 0] = -torch.inf
        test_logits[:, counts == 0] = -torch.inf
        return train_logits.argmax(1).numpy(), test_logits.argmax(1).numpy()


def probe(train_x, train_y, test_x, test_y, classes, args, *, shuffled_control=False):
    if len(np.unique(train_y)) < 2 or not len(test_y):
        return {"unavailable": "Need at least two fitting classes and a nonempty evaluation set"}
    train_prediction, test_prediction = fit_linear(train_x, train_y, test_x, len(classes), args)
    train_counts = np.bincount(train_y, minlength=len(classes))
    majority = int(train_counts.argmax())
    majority_prediction = np.full(len(test_y), majority, dtype=np.int64)
    result = {
        "train": classification_metrics(train_y, train_prediction, classes),
        "test": classification_metrics(test_y, test_prediction, classes),
        "majority_baseline": {
            "class_selected_from_probe_train": str(classes[majority]),
            **classification_metrics(test_y, majority_prediction, classes),
        },
        "uniform_baseline_expected_accuracy": 1 / len(classes),
        "uniform_baseline_expected_balanced_accuracy": 1 / len(classes),
        "test_labels_not_in_probe_train": {
            str(classes[i]): int((test_y == i).sum())
            for i in range(len(classes)) if not train_counts[i] and (test_y == i).any()
        },
    }
    if shuffled_control:
        controls = []
        for repetition in range(args.shuffle_repeats):
            shuffled = np.random.default_rng(args.seed + 1000 + repetition).permutation(train_y)
            _, prediction = fit_linear(train_x, shuffled, test_x, len(classes), args)
            controls.append(classification_metrics(test_y, prediction, classes))
        result["shuffled_train_labels_control"] = {
            "repetitions": controls,
            "mean_accuracy": float(np.mean([x["accuracy"] for x in controls])) if controls else None,
            "mean_balanced_accuracy": float(np.mean([x["balanced_accuracy"] for x in controls])) if controls else None,
        }
    return result


def evaluate_protocol(train, test, model_names, emotion_classes, args):
    shared_clips = sorted(set(train["clip"].tolist()) & set(test["clip"].tolist()))
    if shared_clips:
        raise ValueError(f"Probe fitting/evaluation clip overlap: {shared_clips[:5]}")
    speakers = sorted(set(train["speaker"].tolist()))
    speaker_to_id = {speaker: i for i, speaker in enumerate(speakers)}
    known_speaker = np.asarray([speaker in speaker_to_id for speaker in test["speaker"]])
    speaker_train = np.asarray([speaker_to_id[x] for x in train["speaker"]])
    speaker_test = np.asarray([speaker_to_id[x] for x in test["speaker"][known_speaker]], dtype=np.int64)
    result = {
        "probe_train_n": len(train["clip"]), "probe_test_n": len(test["clip"]),
        "shared_clip_ids": 0,
        "shared_sentence_ids": len(set(train["sentence"]) & set(test["sentence"])),
        "speaker_probe_unseen_speaker_samples_excluded": int((~known_speaker).sum()),
        "models": {},
    }
    for model_name in model_names:
        print(f"fitting independent probes: {model_name}", flush=True)
        train_x, test_x = train[f"style_{model_name}"], test[f"style_{model_name}"]
        result["models"][model_name] = {
            "emotion": probe(train_x, train["emotion"], test_x, test["emotion"], emotion_classes, args, shuffled_control=True),
            "speaker": probe(train_x, speaker_train, test_x[known_speaker], speaker_test, speakers, args),
            "feature_diagnostics_not_a_disentanglement_test": {
                "probe_train_mean_coordinate_std": float(train_x.std(axis=0).mean()),
                "probe_test_mean_coordinate_std": float(test_x.std(axis=0).mean()),
                "probe_train_mean_l2_norm": float(np.linalg.norm(train_x, axis=1).mean()),
            },
        }
    return result


def subset(features, indices):
    return {key: value[indices] for key, value in features.items()}


def grouped_indices(features, seed, fraction):
    groups = np.unique(features["sentence"])
    if len(groups) < 2 or "unknown" in groups:
        raise ValueError("Sentence-group probe requires at least two known sentence_id groups")
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_test = max(1, min(len(groups) - 1, round(fraction * len(groups))))
    test_groups = shuffled[:n_test]
    is_test = np.isin(features["sentence"], test_groups)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), test_groups.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--baseline-audio-hint", action="store_true")
    parser.add_argument("--per-emotion", type=int, default=256, help="Cap per split/class; 0 uses all accepted records")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--probe-iterations", type=int, default=120)
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--feature-cache", help="Optional NPZ cache output, not reused implicitly")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or args.ridge <= 0:
        parser.error("test-fraction must be in (0,1) and ridge must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    cfg = load_yaml(args.config)
    models, model_info = {}, {}
    models["current"], model_info["current"] = load_style(cfg, args.checkpoint, device)
    if args.baseline_checkpoint:
        models["baseline"], model_info["baseline"] = load_style(
            cfg, args.baseline_checkpoint, device, audio_hint=args.baseline_audio_hint
        )
    features, splits, unavailable = {}, {}, {}
    for split in ("train", "val", "test"):
        try:
            dataset = B0ResidualDataset(cfg, split=split, random_crop=False)
        except ValueError as error:
            if split == "train":
                raise
            unavailable[split] = str(error)
            print(f"split {split} unavailable: {error}", flush=True)
            continue
        features[split], splits[split] = extract(dataset, models, args, device, split)
    if args.feature_cache:
        cache_path = Path(args.feature_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{
            f"{split}__{key}": value
            for split, values in features.items() for key, value in values.items()
        })
    report = {
        "protocol_version": 1, "model_checkpoints": model_info,
        "arguments": vars(args), "splits": splits, "unavailable_splits": unavailable,
        "canonical_style_input": "query_motion - channel_mask * neutral_GT; intersect query/neutral valid masks; center crop",
        "probe": "fresh independent CPU linear multinomial logistic regression; train-only standardization; inverse-count CE; fixed ridge; no tuning on evaluation labels",
        "caveats": [
            "High emotion predictability establishes linearly accessible leakage; a low linear probe does not establish full independence.",
            "Grouped probe held-out training clips were potentially seen by the encoder and are not model-unseen validation.",
            "Speaker predictability measures retained identity information, not whether the renderer produces perceptible style differences.",
            "Official split exclusion follows manifest labels and neutral-partner filtering; cross-split near-duplicate audio is not audited.",
        ],
        "protocols": {},
    }
    train = features["train"]
    fitting, heldout, test_groups = grouped_indices(train, args.seed, args.test_fraction)
    grouped = evaluate_protocol(subset(train, fitting), subset(train, heldout), models, cfg["data"]["emotion_classes"], args)
    grouped.update({"encoder_data_status": "model-train clips; probe-held-out sentence groups", "test_sentence_groups": test_groups})
    report["protocols"]["sentence_group_holdout_within_model_train"] = grouped
    for split in ("val", "test"):
        if split in features:
            official = evaluate_protocol(train, features[split], models, cfg["data"]["emotion_classes"], args)
            official["encoder_data_status"] = f"probe fit on manifest train; evaluated on official {split}"
            report["protocols"][f"official_{split}"] = official
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for protocol_name, protocol in report["protocols"].items():
        for model_name, scores in protocol["models"].items():
            print(json.dumps({"protocol": protocol_name, "model": model_name,
                              "emotion": scores["emotion"].get("test"),
                              "speaker": scores["speaker"].get("test")}, ensure_ascii=False), flush=True)
    print(f"saved {destination}", flush=True)


if __name__ == "__main__":
    main()
