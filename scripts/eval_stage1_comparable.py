"""Evaluate Stage1 on one fixed manifest with mouth-only fidelity metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinetalk_b0.data import CanonicalStage1Dataset
from kinetalk_b0.models import Stage1Model
from kinetalk_b0.utils import load_checkpoint, load_yaml


PERCENTILES = (5, 25, 50, 75, 95)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-10 else 0.0


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {str(p): float("nan") for p in PERCENTILES}
    return {str(p): float(np.percentile(values, p)) for p in PERCENTILES}


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    keys = rows[0].keys() if rows else ()
    return {key: quantiles([float(row[key]) for row in rows]) for key in keys}


def evaluate(cfg: dict, checkpoint: str) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage1Model(cfg).to(device)
    payload = load_checkpoint(checkpoint, model, map_location=device, strict=True)
    model.eval()
    dataset = CanonicalStage1Dataset(cfg, split=str(cfg.get("_eval_split", "train")), random_crop=False)
    mouth = np.asarray(cfg["data"]["neutral_output_indices"], dtype=np.int64)
    jaw_index = 17
    all_rows: list[dict[str, float]] = []
    by_emotion: dict[str, list[dict[str, float]]] = defaultdict(list)
    manifest_rows = dataset.items

    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            pred_full = model(
                item["content"].unsqueeze(0).to(device),
                item["mask"].unsqueeze(0).to(device),
            )["b0"][0].float().cpu().numpy()
            target_full = item["target"].numpy()
            valid = item["mask"].numpy().astype(bool)
            if int(valid.sum()) < 4:
                continue
            pred = pred_full[valid][:, mouth]
            target = target_full[valid][:, mouth]
            target_amp = np.ptp(target, axis=0)
            active = target_amp > 1e-5
            if not active.any():
                continue
            pred_active = pred[:, active]
            target_active = target[:, active]
            row = {
                "mouth_mae": float(np.abs(pred_active - target_active).mean()),
                "zero_mouth_mae": float(np.abs(target_active).mean()),
                "mean_mouth_mae": float(
                    np.abs(target_active - target_active.mean(axis=0, keepdims=True)).mean()
                ),
                "amplitude_ratio": float(
                    (
                        np.percentile(pred_active, 95, axis=0)
                        - np.percentile(pred_active, 5, axis=0)
                    ).sum()
                    / max(
                        (
                            np.percentile(target_active, 95, axis=0)
                            - np.percentile(target_active, 5, axis=0)
                        ).sum(),
                        1e-9,
                    )
                ),
                "temporal_corr": float(
                    np.mean([corr(pred[:, j], target[:, j]) for j in np.flatnonzero(active)])
                ),
            }
            if len(pred) > 2:
                adjacent = valid[1:] & valid[:-1]
                pred_velocity = np.diff(pred_full[:, mouth], axis=0)[adjacent]
                target_velocity = np.diff(target_full[:, mouth], axis=0)[adjacent]
                velocity_active = np.std(target_velocity, axis=0) > 1e-6
                row["velocity_corr"] = float(
                    np.mean(
                        [
                            corr(pred_velocity[:, j], target_velocity[:, j])
                            for j in np.flatnonzero(velocity_active)
                        ]
                    )
                    if velocity_active.any()
                    else 0.0
                )
            if jaw_index in mouth.tolist():
                jaw_local = int(np.flatnonzero(mouth == jaw_index)[0])
                jaw_target_range = float(
                    np.percentile(target[:, jaw_local], 95)
                    - np.percentile(target[:, jaw_local], 5)
                )
                if jaw_target_range > 1e-5:
                    row["jaw_open_ratio"] = float(
                        (
                            np.percentile(pred[:, jaw_local], 95)
                            - np.percentile(pred[:, jaw_local], 5)
                        )
                        / jaw_target_range
                    )
            all_rows.append(row)
            emotion = str(manifest_rows[index].get("emotion", "unknown"))
            by_emotion[emotion].append(row)

    manifest_identity = "\n".join(
        f"{row.get('source_clip_id', '')}\t{row.get('reference_clip_id', '')}"
        for row in manifest_rows
    ).encode()
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "pairs": len(dataset),
        "evaluated": len(all_rows),
        "manifest_sha256": hashlib.sha256(manifest_identity).hexdigest(),
        "split": str(cfg.get("_eval_split", "train")),
        "validation_is_training_data": str(cfg.get("_eval_split", "train")) == "train",
        "target_kind": "canonical_neutral_target",
        "overall": summarize(all_rows),
        "by_emotion": {key: summarize(value) for key, value in sorted(by_emotion.items())},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg["_eval_split"] = args.split
    result = evaluate(cfg, args.ckpt)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
