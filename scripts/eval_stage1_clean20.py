"""Evaluate Stage1 against the materialized canonical neutral targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinetalk_b0.data import CanonicalStage1Dataset
from kinetalk_b0.models import Stage1Model
from kinetalk_b0.utils import load_checkpoint, load_yaml


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-10 else 0.0


def percentile(values: list[float]) -> dict[str, float]:
    return {str(k): float(np.percentile(values, k)) for k in (5, 25, 50, 75, 95)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage1Model(cfg).to(device)
    payload = load_checkpoint(args.ckpt, model, map_location=device, strict=True)
    model.eval()
    dataset = CanonicalStage1Dataset(cfg, split="train", random_crop=False)
    mouth = list(cfg["data"]["neutral_output_indices"])
    amp_ratios: list[float] = []
    jaw_ratios: list[float] = []
    corrs: list[float] = []
    vel_corrs: list[float] = []
    pred_mae: list[float] = []
    zero_mae: list[float] = []
    mean_mae: list[float] = []
    peak_errors: list[float] = []
    with torch.inference_mode():
        for i in range(len(dataset)):
            item = dataset[i]
            x = item["content"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).to(device)
            pred = model(x, mask)["b0"][0].float().cpu().numpy()
            target = item["target"].numpy()
            valid = item["mask"].numpy().astype(bool)
            if valid.sum() < 4:
                continue
            pr, gt = pred[valid], target[valid]
            gt_mean = gt.mean(axis=0, keepdims=True)
            pred_mae.append(float(np.abs(pr - gt).mean()))
            zero_mae.append(float(np.abs(gt).mean()))
            mean_mae.append(float(np.abs(gt - gt_mean).mean()))
            p_amp = np.percentile(pr[:, mouth], 95, axis=0) - np.percentile(pr[:, mouth], 5, axis=0)
            t_amp = np.percentile(gt[:, mouth], 95, axis=0) - np.percentile(gt[:, mouth], 5, axis=0)
            keep = t_amp > 1e-5
            if keep.any():
                amp_ratios.append(float(p_amp[keep].sum() / max(t_amp[keep].sum(), 1e-9)))
                corrs.append(float(np.mean([corr(pr[:, j], gt[:, j]) for j in np.asarray(mouth)[keep]])))
            jaw = 17
            if jaw in mouth and np.ptp(gt[:, jaw]) > 1e-5:
                jaw_ratios.append(float(np.ptp(pr[:, jaw]) / np.ptp(gt[:, jaw])))
            if len(pr) > 2:
                pv, gv = np.diff(pr, axis=0), np.diff(gt, axis=0)
                vkeep = np.std(gv[:, mouth], axis=0) > 1e-6
                if vkeep.any():
                    vel_corrs.append(float(np.mean([corr(pv[:, j], gv[:, j]) for j in np.asarray(mouth)[vkeep]])))
            gt_peak = int(np.argmax(np.mean(np.abs(gt[:, mouth]), axis=1)))
            pr_peak = int(np.argmax(np.mean(np.abs(pr[:, mouth]), axis=1)))
            peak_errors.append(float(abs(pr_peak - gt_peak)))
            if (i + 1) % 500 == 0:
                print(i + 1, flush=True)
    out = {
        "checkpoint_epoch": payload.get("epoch"),
        "pairs": len(dataset),
        "pred_mae": percentile(pred_mae),
        "zero_baseline_mae": percentile(zero_mae),
        "mean_baseline_mae": percentile(mean_mae),
        "amplitude_ratio": percentile(amp_ratios),
        "jaw_open_ratio": percentile(jaw_ratios),
        "mouth_temporal_corr": percentile(corrs),
        "mouth_velocity_corr": percentile(vel_corrs),
        "peak_timing_error_frames": percentile(peak_errors),
        "pred_vs_zero_mae_ratio": float(np.mean(pred_mae) / max(np.mean(zero_mae), 1e-9)),
        "pred_vs_mean_mae_ratio": float(np.mean(pred_mae) / max(np.mean(mean_mae), 1e-9)),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
