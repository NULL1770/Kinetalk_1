"""Read-only Stage1 diagnostics against aligned neutral articulation targets.

Reports motion *variation*, not just mean coefficient magnitude. The temporal
mean baseline is explicitly an oracle that sees the target, not a usable model.
"""
from __future__ import annotations

import argparse
import hashlib
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
from kinetalk_b0.models import Stage1Model
from kinetalk_b0.utils import load_checkpoint, load_yaml, seed_everything


ARKIT_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]
KEY_CHANNELS = ["jawOpen", "mouthFunnel", "mouthPucker", "mouthLowerDownLeft",
                "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight"]


class QueryNeutralDataset(Dataset):
    """Loads only query + its neutral partner, without other training branches."""

    def __init__(self, base: B0ResidualDataset, indices: list[int]):
        self.base, self.indices = base, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        index = self.indices[row]
        query = self.base._read(self.base.items[index])
        neutral_index, valid = self.base._neutral_index(index)
        if not valid:
            raise ValueError(f"Missing true neutral partner: {query['clip_id']}")
        neutral = self.base._read(self.base.items[neutral_index], start=query["crop_start"])
        return {
            "content": query["content"], "mask": query["mask"],
            "valid_mask": query["mask"] & neutral["mask"],
            "target": neutral["motion"] * self.base._channel_mask[None],
            "emotion_id": query["emotion_id"], "clip_id": query["clip_id"],
        }


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else None


class Metrics:
    def __init__(self, channels: int, std_epsilon: float):
        self.sums = defaultdict(lambda: np.zeros(channels, dtype=np.float64))
        self.std_epsilon = std_epsilon
        self.clips = 0
        self.skipped_empty_clips = 0

    def add(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
        if not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite prediction or target in Stage1 evaluation")
        p, t = pred[mask].astype(np.float64), target[mask].astype(np.float64)
        if not len(t):
            self.skipped_empty_clips += 1
            return
        s = self.sums
        self.clips += 1
        s["frame_count"] += len(t)
        s["absolute_error"] += np.abs(p - t).sum(0)
        s["squared_error"] += np.square(p - t).sum(0)
        s["pred_abs"] += np.abs(p).sum(0)
        s["target_abs"] += np.abs(t).sum(0)
        s["oracle_mean_absolute_error"] += np.abs(t - t.mean(0)).sum(0)
        if len(t) >= 3:
            pc, tc = p - p.mean(0), t - t.mean(0)
            ps, ts = p.std(0), t.std(0)
            variable = ts > self.std_epsilon
            s["moving_target_clip_count"] += variable
            s["pred_std"] += ps * variable
            s["target_std"] += ts * variable
            s["pred_p95_p05"] += (np.percentile(p, 95, axis=0) - np.percentile(p, 5, axis=0)) * variable
            s["target_p95_p05"] += (np.percentile(t, 95, axis=0) - np.percentile(t, 5, axis=0)) * variable
            # A constant prediction scores zero instead of disappearing from the metric.
            denom = np.sqrt(np.square(pc).sum(0) * np.square(tc).sum(0))
            corr = np.divide((pc * tc).sum(0), denom, out=np.zeros_like(denom), where=denom > 1e-12)
            s["temporal_corr"] += np.clip(corr, -1, 1) * variable
        adjacent = mask[1:] & mask[:-1]
        vp = np.diff(pred, axis=0)[adjacent].astype(np.float64)
        vt = np.diff(target, axis=0)[adjacent].astype(np.float64)
        if len(vt):
            s["velocity_frame_count"] += len(vt)
            s["pred_velocity_abs"] += np.abs(vp).sum(0)
            s["target_velocity_abs"] += np.abs(vt).sum(0)
            s["velocity_absolute_error"] += np.abs(vp - vt).sum(0)
        if len(vt) >= 3:
            pc, tc = vp - vp.mean(0), vt - vt.mean(0)
            variable = vt.std(0) > self.std_epsilon
            denom = np.sqrt(np.square(pc).sum(0) * np.square(tc).sum(0))
            corr = np.divide((pc * tc).sum(0), denom, out=np.zeros_like(denom), where=denom > 1e-12)
            s["velocity_corr"] += np.clip(corr, -1, 1) * variable
            s["variable_velocity_clip_count"] += variable

    def summarize(self, channel: int | None = None):
        def v(key):
            arr = self.sums[key]
            return float(arr.sum() if channel is None else arr[channel])

        frames, velocities = v("frame_count"), v("velocity_frame_count")
        moving, variable_velocity = v("moving_target_clip_count"), v("variable_velocity_clip_count")
        return {
            "clips": self.clips, "skipped_empty_clips": self.skipped_empty_clips,
            "valid_frame_channel_count": int(frames),
            "mae": ratio(v("absolute_error"), frames),
            "rmse": np.sqrt(v("squared_error") / frames).item() if frames else None,
            "zero_baseline_mae": ratio(v("target_abs"), frames),
            "oracle_gt_temporal_mean_baseline_mae": ratio(v("oracle_mean_absolute_error"), frames),
            "pred_mean_abs": ratio(v("pred_abs"), frames),
            "target_mean_abs": ratio(v("target_abs"), frames),
            "mean_abs_ratio": ratio(v("pred_abs"), v("target_abs")),
            "moving_target_clip_channel_count": int(moving),
            "pred_temporal_std": ratio(v("pred_std"), moving),
            "target_temporal_std": ratio(v("target_std"), moving),
            "temporal_std_ratio": ratio(v("pred_std"), v("target_std")),
            "pred_p95_p05_amplitude": ratio(v("pred_p95_p05"), moving),
            "target_p95_p05_amplitude": ratio(v("target_p95_p05"), moving),
            "p95_p05_amplitude_ratio": ratio(v("pred_p95_p05"), v("target_p95_p05")),
            "temporal_corr": ratio(v("temporal_corr"), moving),
            "velocity_mae": ratio(v("velocity_absolute_error"), velocities),
            "pred_velocity_mean_abs": ratio(v("pred_velocity_abs"), velocities),
            "target_velocity_mean_abs": ratio(v("target_velocity_abs"), velocities),
            "velocity_mean_abs_ratio": ratio(v("pred_velocity_abs"), v("target_velocity_abs")),
            "variable_velocity_clip_channel_count": int(variable_velocity),
            "velocity_corr": ratio(v("velocity_corr"), variable_velocity),
        }


def evaluate(cfg, loader, checkpoint, device, active, names, args):
    model = Stage1Model(cfg).to(device)
    payload = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
    epoch = payload.get("epoch")
    del payload
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    model.eval()
    totals = Metrics(len(active), args.std_epsilon)
    per_emotion = {name: Metrics(len(active), args.std_epsilon) for name in names}
    started = time.monotonic()
    inactive = [i for i in range(int(cfg["data"]["motion_dim"])) if i not in active]
    inactive_max = 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            content = batch["content"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            output = model(content, mask)["b0"].float().cpu().numpy()
            if inactive:
                inactive_max = max(inactive_max, float(np.abs(output[..., inactive]).max()))
            pred = output[..., active]
            target = batch["target"].numpy()[..., active]
            valid = batch["valid_mask"].numpy()
            for row, label in enumerate(batch["emotion_id"].tolist()):
                totals.add(pred[row], target[row], valid[row])
                per_emotion[names[label]].add(pred[row], target[row], valid[row])
            if (batch_index + 1) % 20 == 0 or batch_index + 1 == len(loader):
                print(f"checkpoint={checkpoint} epoch={epoch} batches={batch_index+1}/{len(loader)} "
                      f"clips={totals.clips} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    channels = {ARKIT_NAMES[idx] if idx < len(ARKIT_NAMES) else str(idx):
                {"source_index": idx, **totals.summarize(pos)} for pos, idx in enumerate(active)}
    return {
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_epoch": epoch,
        "strict_load": True, "finite_checkpoint_and_predictions": True,
        "elapsed_seconds": time.monotonic() - started,
        "aggregate": totals.summarize(), "channels": channels,
        "key_channels": {key: channels[key] for key in KEY_CHANNELS if key in channels},
        "per_emotion": {name: metric.summarize() for name, metric in per_emotion.items()},
        "inactive_channel_abs_max": inactive_max,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--per-emotion", type=int, default=0, help="0 uses all accepted samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--std-epsilon", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.per_emotion < 0 or args.std_epsilon <= 0:
        parser.error("per-emotion must be >= 0 and std-epsilon must be > 0")
    seed_everything(args.seed)
    cfg = load_yaml(args.config)
    base = B0ResidualDataset(cfg, split=args.split, random_crop=False)
    names = base.emotion_classes
    grouped = defaultdict(list)
    for i, record in enumerate(base.items):
        grouped[base._emotion_name(record)].append(i)
    rng = random.Random(args.seed)
    selected = []
    for name in names:
        candidates = grouped[name]
        chosen = rng.sample(candidates, args.per_emotion) if args.per_emotion and len(candidates) > args.per_emotion else candidates
        selected.extend(sorted(chosen))
    if not selected:
        raise RuntimeError(f"No selected samples for explicit split={args.split}; no automatic fallback")
    ids = [str(base.items[i]["clip_id"]) for i in selected]
    count = Counter(base._emotion_name(base.items[i]) for i in selected)
    loader = DataLoader(QueryNeutralDataset(base, selected), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=args.device.startswith("cuda"),
                        persistent_workers=args.num_workers > 0)
    active = list(cfg["data"]["neutral_output_indices"])
    print(f"split={args.split} selected={len(selected)} available={len(base)} "
          f"class_counts={dict(count)} device={args.device}", flush=True)
    result = {
        "schema_version": 1, "split": args.split, "seed": args.seed,
        "evaluation_mode": "train_set_diagnostic_not_held_out" if args.split == "train" else "manifest_split_evaluation",
        "selected_samples": len(selected), "available_samples": len(base),
        "class_counts": dict(count), "available_class_counts": {name: len(grouped[name]) for name in names},
        "selected_clip_ids": ids, "selected_clip_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "window": base.window, "active_channel_indices": active,
        "protocol": {
            "target": "same-speaker/same-sentence aligned neutral GT, masked to active B0 channels",
            "crop": "deterministic query center crop; same requested start for neutral, matching training loader",
            "validity": "query mask AND neutral mask; velocity requires both adjacent frames valid",
            "absolute_metrics": "frame-channel weighted, active channels only; no channel flattening for correlations",
            "amplitude_metrics": "per-clip per-channel temporal std and P95-P05, only target std > epsilon; reported ratio is sum(pred)/sum(target)",
            "correlations": "mean per-clip per-channel Pearson correlation; target std > epsilon, >=3 observations; constant prediction counts as zero",
            "std_epsilon": args.std_epsilon,
            "velocity": "consecutive aligned-frame differences, not physical units per second",
            "oracle_baseline": "per-clip per-channel temporal GT mean repeated over time; uses target and is not an inference baseline",
            "limits": "aligned neutral target and center windows only; not original emotional-motion amplitude or independent audiovisual lip-sync validation; clips sharing a neutral partner are dependent",
        },
    }
    result["current"] = evaluate(cfg, loader, args.checkpoint, torch.device(args.device), active, names, args)
    if args.baseline_checkpoint:
        result["baseline_same_samples"] = evaluate(cfg, loader, args.baseline_checkpoint, torch.device(args.device), active, names, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "current": result["current"]["aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
