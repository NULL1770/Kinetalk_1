"""Measure actual deployment output against the query's emotional-motion GT.

Uses a different-sentence reference from the same speaker. This is a paired
fidelity diagnostic, not a claim that one stochastic sample must equal GT.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.data import B0ResidualDataset
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model
from kinetalk_b0.utils import load_checkpoint, load_yaml, move_to_device, seed_everything

_spec = importlib.util.spec_from_file_location("stage1_fidelity", Path(__file__).with_name("05_eval_stage1_fidelity.py"))
_fidelity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fidelity)
Metrics, ARKIT_NAMES, KEY_CHANNELS = _fidelity.Metrics, _fidelity.ARKIT_NAMES, _fidelity.KEY_CHANNELS


def choose_reference(base, query_index, seed):
    """Stable per-query selection; never use another speaker or same sentence."""
    query = base.items[query_index]
    speaker, sentence = base._group_key(query)
    candidates = [i for i in base.by_speaker[speaker] if base._group_key(base.items[i])[1] != sentence]
    preferred = [i for i in candidates if base._emotion_name(base.items[i]) != base._emotion_name(query)]
    pool = preferred or candidates
    if not pool:
        return None
    digest = hashlib.sha256(f"{seed}:{query['clip_id']}".encode()).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


class QueryReferenceDataset(Dataset):
    def __init__(self, base, pairs):
        self.base, self.pairs = base, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, row):
        query_index, reference_index = self.pairs[row]
        # _read is a center crop here and does not load any neutral targets.
        query = self.base._read(self.base.items[query_index])
        reference = self.base._read(self.base.items[reference_index])
        keys = ("content", "audio_emotion", "motion", "mask", "emotion_id", "clip_id")
        return {"query": {key: query[key] for key in keys},
                "style_reference": {key: reference[key] for key in keys}}


def summarize_subset(metric, indices):
    subset = Metrics(len(indices), metric.std_epsilon)
    subset.clips = metric.clips
    subset.skipped_empty_clips = metric.skipped_empty_clips
    for key, values in metric.sums.items():
        subset.sums[key] = values[indices]
    return subset.summarize()


def metric_report(metric, mouth_indices):
    return {
        "mouth": summarize_subset(metric, mouth_indices),
        "all52": metric.summarize(),
        "key_channels": {name: {"source_index": ARKIT_NAMES.index(name),
                                  **metric.summarize(ARKIT_NAMES.index(name))} for name in KEY_CHANNELS},
        "mouth_channels": {ARKIT_NAMES[index]: {"source_index": index,
                                                **metric.summarize(index)} for index in mouth_indices},
    }


def load_deployment(cfg, device):
    metadata = {}

    def load(stage_name, model):
        path = cfg["paths"][f"{stage_name}_ckpt"]
        payload = load_checkpoint(path, model, map_location="cpu", strict=True)
        metadata[stage_name] = {"path": str(Path(path).resolve()), "epoch": payload.get("epoch"),
                                "strict_load": True}
        del payload
        for name, tensor in model.state_dict().items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite {stage_name} checkpoint tensor: {name}")
        print(f"loaded {stage_name} epoch={metadata[stage_name]['epoch']} strict=True", flush=True)

    stage1 = Stage1Model(cfg)
    load("stage1", stage1)
    stage2 = Stage2Model(cfg)
    load("stage2", stage2)
    stage3 = Stage3Model(cfg, stage2)
    load("stage3", stage3)
    stage4 = Stage4Model(cfg, stage1, stage2, stage3)
    load("stage4", stage4)
    # Later-stage checkpoints intentionally overwrite shared registered modules.
    # The final loaded Stage4 state is authoritative for this deployment test.
    return stage4.to(device).eval(), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--split", choices=("val", "test", "train"), default="val")
    parser.add_argument("--per-emotion", type=int, default=0, help="0 evaluates all eligible samples")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--std-epsilon", type=float, default=1e-5)
    parser.add_argument("--stochastic", action="store_true", help="also evaluate one fixed-seed stochastic sample per query")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.per_emotion < 0 or args.steps < 1 or args.std_epsilon <= 0:
        parser.error("per-emotion >= 0, steps >= 1 and std-epsilon > 0 required")
    seed_everything(args.seed)
    cfg = load_yaml(args.config)
    if int(cfg["data"]["motion_dim"]) != 52:
        raise ValueError("This evaluator requires the raw ARKit-52 model contract")
    base = B0ResidualDataset(cfg, split=args.split, random_crop=False)
    names = base.emotion_classes
    grouped, references = defaultdict(list), {}
    for index, record in enumerate(base.items):
        reference = choose_reference(base, index, args.seed)
        if reference is not None:
            grouped[base._emotion_name(record)].append(index)
            references[index] = reference
    rng = random.Random(args.seed)
    selected = []
    for name in names:
        candidates = grouped[name]
        chosen = rng.sample(candidates, args.per_emotion) if args.per_emotion and len(candidates) > args.per_emotion else candidates
        selected.extend(sorted(chosen))
    if not selected:
        raise ValueError(f"No eligible same-speaker different-sentence reference pairs for split={args.split}; no fallback")
    pairs = [(index, references[index]) for index in selected]
    pair_info = [{"query": str(base.items[q]["clip_id"]), "reference": str(base.items[r]["clip_id"]),
                  "speaker": base._group_key(base.items[q])[0], "query_emotion": base._emotion_name(base.items[q]),
                  "reference_emotion": base._emotion_name(base.items[r])} for q, r in pairs]
    loader = DataLoader(QueryReferenceDataset(base, pairs), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                        persistent_workers=args.workers > 0)
    device = torch.device(args.device)
    model, metadata = load_deployment(cfg, device)
    mouth_indices = list(cfg["data"]["neutral_output_indices"])
    variants = ["stage1_b0_vs_query_gt", "stage4_deterministic_vs_query_gt"]
    if args.stochastic:
        variants.append("stage4_fixed_seed_stochastic_vs_query_gt")
    totals = {name: Metrics(52, args.std_epsilon) for name in variants}
    emotions = {variant: {name: Metrics(52, args.std_epsilon) for name in names} for variant in variants}
    out_of_range = {variant: {"below_zero": 0, "above_one": 0, "valid_coefficients": 0} for variant in variants}
    # Reset after all checkpoint/model setup so stochastic sampling has a known state.
    seed_everything(args.seed)
    started = time.monotonic()
    print(f"split={args.split} selected={len(pairs)} counts={dict(Counter(p['query_emotion'] for p in pair_info))} "
          f"steps={args.steps} stochastic_variant={args.stochastic}", flush=True)
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            batch = move_to_device(cpu_batch, device)
            conditions = model.conditions(batch)
            deterministic, _ = model.render(batch, conditions, steps=args.steps, stochastic=False)
            outputs = {"stage1_b0_vs_query_gt": conditions["b0_pred"],
                       "stage4_deterministic_vs_query_gt": deterministic}
            if args.stochastic:
                stochastic, _ = model.render(batch, conditions, steps=args.steps, stochastic=True)
                outputs["stage4_fixed_seed_stochastic_vs_query_gt"] = stochastic
            # Compare final raw coefficients with QUERY emotional motion, never neutral B0.
            target = cpu_batch["query"]["motion"].numpy()
            masks = cpu_batch["query"]["mask"].numpy()
            labels = cpu_batch["query"]["emotion_id"].tolist()
            for variant, tensor in outputs.items():
                prediction = tensor.float().cpu().numpy()
                for row, label in enumerate(labels):
                    totals[variant].add(prediction[row], target[row], masks[row])
                    emotions[variant][names[label]].add(prediction[row], target[row], masks[row])
                valid_values = prediction[masks]
                out_of_range[variant]["below_zero"] += int((valid_values < 0).sum())
                out_of_range[variant]["above_one"] += int((valid_values > 1).sum())
                out_of_range[variant]["valid_coefficients"] += int(valid_values.size)
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
                print(f"batches={batch_index+1}/{len(loader)} clips={totals[variants[0]].clips} "
                      f"elapsed_s={time.monotonic()-started:.1f}", flush=True)
    result = {
        "schema_version": 1, "split": args.split, "seed": args.seed,
        "evaluation_mode": "train_set_diagnostic_not_held_out" if args.split == "train" else "manifest_split_evaluation",
        "checkpoints": metadata,
        "authoritative_state": "Stage4 checkpoint loaded last strictly; its registered Stage1, style, audio prior and renderer are the evaluated deployment modules",
        "selected_samples": len(pairs), "available_samples_after_config_filters": len(base),
        "eligible_reference_pairs": len(references),
        "class_counts": dict(Counter(p["query_emotion"] for p in pair_info)),
        "eligible_class_counts": {name: len(grouped[name]) for name in names},
        "reference_different_emotion_count": sum(p["query_emotion"] != p["reference_emotion"] for p in pair_info),
        "pairs": pair_info, "pairs_sha256": hashlib.sha256(json.dumps(pair_info, sort_keys=True).encode()).hexdigest(),
        "mouth_channel_indices": mouth_indices, "elapsed_seconds": time.monotonic() - started,
        "protocol": {
            "target": "query's original emotional-motion raw ARKit-52 aligned GT, NOT the neutral target",
            "reference": "same speaker, different sentence; prefer different emotion; deterministic per-query selection; no cross-speaker paired reconstruction",
            "crop_and_filter": "center windows; original config data filters retained, including neutral-partner eligibility if enabled; no neutral arrays read",
            "inference": "actual Stage4 conditions/render, steps=" + str(args.steps) + "; raw un-clamped FP32 output",
            "default_api_note": "Stage4.generate defaults stochastic=True whereas Stage4.render defaults False; main diagnostic uses False explicitly and optional variant uses True",
            "stochastic_note": "one reproducible sample per query at the fixed seed and batch size; not distribution evaluation or an expectation of exact GT matching",
            "batch_size": args.batch_size, "std_epsilon": args.std_epsilon,
            "metrics": "same 05_eval_stage1_fidelity Metrics: per-channel temporal std/P95-P05 on variable GT clips; mean per-clip Pearson; constant predictions count zero; adjacent valid-frame velocities; amplitude ratios are summed prediction/GT values",
            "baseline": "Stage4-embedded Stage1 B0 against the SAME emotional GT; GT temporal-mean oracle is target-aware, not a deployable predictor",
            "limits": "same-speaker references can still differ in expression; GT is one possible realization; held-out aligned feature fidelity is not independent audiovisual lip-sync validation",
        },
        "results": {},
    }
    for variant in variants:
        result["results"][variant] = {**metric_report(totals[variant], mouth_indices),
                                      "per_emotion_mouth": {name: summarize_subset(metric, mouth_indices)
                                                            for name, metric in emotions[variant].items()},
                                      "raw_coefficient_range_counts": out_of_range[variant]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "mouth": {name: result["results"][name]["mouth"] for name in variants}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
