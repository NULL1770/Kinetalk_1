"""Native-scale dynamic metrics with sentence-cluster uncertainty.

Zero-baseline R2 measures accuracy, whereas energy_ratio only measures motion
amount. A nonzero prediction or a positive correlation does not establish an
improvement over the zero field. No scale is fitted against evaluation targets.
"""
from __future__ import annotations

import numpy as np
import torch


def field_metrics(prediction: torch.Tensor, target: torch.Tensor,
                  weight: torch.Tensor, sentence_ids: list,
                  bootstrap_seed: int = 45,
                  bootstrap_samples: int = 1000) -> dict:
    """Evaluate [B,K,C] fields with nonnegative [B,K] frame/bin weights.

    ``native_mse`` and ``zero_mse`` average over valid weighted channel values;
    R2 is ``1 - SSE / sum(weight * target**2)``. ``energy_ratio`` is the ratio
    of squared prediction to target energy (not a root-amplitude ratio).
    Correlation pools weighted covariance after centering each clip/channel
    independently. ``sst`` likewise sums this within-clip target variation.

    ``per_sentence`` aggregates all clips sharing a sentence ID. Its ``r2``
    uses the same zero baseline; ``r2_against_centered_sst`` additionally reports
    1 - SSE/SST, and is not substituted for the zero-baseline result. Bootstrap
    resamples entire sentence groups and recomputes the ratio of aggregate
    sums, never individual frames. Fewer than two observed sentences produce
    no confidence interval. Invalid padded values are ignored; nonfinite
    observed values and invalid weights are rejected. Outputs contain only
    JSON-compatible Python values (undefined metrics are None).
    """
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must share [batch,time,channels]")
    if prediction.shape[-1] < 1 or weight.shape != prediction.shape[:2]:
        raise ValueError("weight must be [batch,time] and channels must be nonempty")
    if len(sentence_ids) != prediction.shape[0]:
        raise ValueError("sentence_ids must have one ID per clip")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    p = prediction.detach().to(device="cpu", dtype=torch.float64)
    y = target.detach().to(device="cpu", dtype=torch.float64)
    w = weight.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("weight must be finite and nonnegative")
    observed = w > 0
    if not torch.isfinite(p[observed]).all() or not torch.isfinite(y[observed]).all():
        raise ValueError("observed prediction and target must be finite")
    p = torch.where(observed[..., None], p, 0)
    y = torch.where(observed[..., None], y, 0)
    ww = w[..., None]
    clip_weight = w.sum(1)
    denominator = torch.where(clip_weight > 0, clip_weight, 1)[:, None, None]
    pc = torch.where(observed[..., None], p - (p * ww).sum(1, keepdim=True) / denominator, 0)
    yc = torch.where(observed[..., None], y - (y * ww).sum(1, keepdim=True) / denominator, 0)
    sse = ((p - y).square() * ww).sum((1, 2))
    zero = (y.square() * ww).sum((1, 2))
    pred_energy = (p.square() * ww).sum((1, 2))
    sst = (yc.square() * ww).sum((1, 2))
    pred_centered_energy = float((pc.square() * ww).sum())
    target_centered_energy = float(sst.sum())
    covariance = float((pc * yc * ww).sum())
    total_sse, total_zero = float(sse.sum()), float(zero.sum())
    total_prediction = float(pred_energy.sum())
    weighted_values = float(clip_weight.sum()) * p.shape[-1]

    def ratio(numerator: float, denominator: float):
        return numerator / denominator if denominator > 0 else None

    def r2(error: float, baseline: float):
        return 1 - error / baseline if baseline > 0 else None

    correlation = None
    if pred_centered_energy > 0 and target_centered_energy > 0:
        correlation = float(np.clip(covariance / np.sqrt(
            pred_centered_energy * target_centered_energy), -1, 1))

    groups = {}
    for i, sentence_id in enumerate(sentence_ids):
        if clip_weight[i] <= 0:
            continue
        sid = str(sentence_id)
        entry = groups.setdefault(sid, {"clips": 0, "weighted_values": 0.,
                                       "sse": 0., "zero": 0., "sst": 0.})
        entry["clips"] += 1
        entry["weighted_values"] += float(clip_weight[i]) * p.shape[-1]
        for key, values in (("sse", sse), ("zero", zero), ("sst", sst)):
            entry[key] += float(values[i])
    for entry in groups.values():
        entry["r2"] = r2(entry["sse"], entry["zero"])
        entry["r2_against_zero"] = entry["r2"]
        entry["r2_against_centered_sst"] = r2(entry["sse"], entry["sst"])

    ci = None
    valid_bootstraps = 0
    if len(groups) >= 2 and bootstrap_samples > 0:
        totals = np.array([[row["sse"], row["zero"]] for row in groups.values()])
        generator = np.random.default_rng(bootstrap_seed)
        sampled = generator.integers(len(totals), size=(bootstrap_samples, len(totals)))
        sums = totals[sampled].sum(1)
        usable = sums[:, 1] > 0
        valid_bootstraps = int(usable.sum())
        if valid_bootstraps:
            scores = 1 - sums[usable, 0] / sums[usable, 1]
            ci = [float(v) for v in np.quantile(scores, [.025, .975])]

    return {
        "native_mse": ratio(total_sse, weighted_values),
        "zero_mse": ratio(total_zero, weighted_values),
        "r2_against_zero": r2(total_sse, total_zero),
        "pooled_centered_correlation": correlation,
        "energy_ratio": ratio(total_prediction, total_zero),
        "prediction_energy": total_prediction,
        "target_energy": total_zero,
        "sse": total_sse,
        "sst": target_centered_energy,
        "weighted_values": weighted_values,
        "clips": int((clip_weight > 0).sum()),
        "sentences": len(groups),
        "per_sentence": groups,
        "bootstrap_r2_ci95": ci,
        "bootstrap_valid_samples": valid_bootstraps,
        "bootstrap_requested_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
