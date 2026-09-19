"""Literature metric interface for ARKit coefficient sequences.

This module keeps coefficient-space adaptations distinct from vertex-space
benchmarks. Unavailable audiovisual or learned-feature evaluations are explicit
pending records, never numerical zeroes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LIP23 = tuple(range(18, 41))
UPPER9 = (41, 42, 43, 44, 45, 5, 6, 12, 13)
BEAT_LIP12 = (15, 16, 27, 28, 19, 20, 31, 32, 33, 34, 39, 40)
BEAT_UPPER16 = tuple(range(14)) + (49, 50)
SOURCE = {
    "repository": "https://github.com/uuembodiedsocialai/FaceDiffuser",
    "commit": "e15f3500fdae0eda962f5d018488dfa0a1a9d552",
    "path": "evaluation/compute_objective_metrics_blendshape.py",
    "function": "main_beat",
    "sha256": "ea909e1346f78e5b1c930b794020654945cb224c08fb1f4c18efcb4b5eaeb10f",
}


def validate_inputs(samples, target, mask, times, fps=25.0):
    """Validate observed coefficients; never treat missing values as targets.

    Returns S,T,52 samples, T,52 target/mask, time, and an adjacency flag for
    each original consecutive frame pair. No time-gap compaction is performed.
    Non-finite values outside the observation mask are permitted.
    """
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    m = np.asarray(mask)
    clock = np.asarray(times, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    if y.ndim != 2 or y.shape[1] != 52 or len(y) == 0:
        raise ValueError("Expected nonempty target[T,52]")
    if x.ndim != 3 or x.shape[1:] != y.shape or len(x) == 0:
        raise ValueError("Expected samples[S,T,52] with at least one sample")
    if m.shape != y.shape or m.dtype != np.bool_:
        raise ValueError("A boolean channel mask[T,52] is required")
    if clock.shape != (len(y),) or not np.isfinite(clock).all():
        raise ValueError("Finite times[T] are required")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    dt = np.diff(clock)
    if (dt <= 0).any():
        raise ValueError("Time must be strictly increasing")
    if not np.isfinite(y[m]).all() or not np.isfinite(x[:, m]).all():
        raise ValueError("Observed target and predicted coefficients must be finite")
    adjacent = np.isclose(dt, 1.0 / fps, rtol=1e-4, atol=1e-7)
    return x, y, m, clock, adjacent


def validate_region(channels, name):
    values = tuple(channels)
    if not values or any(not isinstance(c, (int, np.integer)) for c in values):
        raise ValueError(f"{name} must contain integer channel indices")
    if len(set(values)) != len(values) or min(values) < 0 or max(values) >= 52:
        raise ValueError(f"{name} must contain unique channel indices in [0,52)")
    return values


def region_coverage(mask, channels):
    """Expose support so missing-channel adaptations cannot look like full52."""
    region = validate_region(channels, "region")
    support = np.asarray(mask)[:, region]
    counts = support.sum(1)
    nonempty = counts > 0
    return {
        "requested_channels": list(region),
        "requested_channel_count": len(region),
        "valid_frames": int(nonempty.sum()),
        "complete_region_frames": int(support.all(1).sum()),
        "observed_values": int(support.sum()),
        "observed_channel_count_min": int(counts[nonempty].min()) if nonempty.any() else 0,
        "observed_channel_count_max": int(counts.max()),
        "partial_support": bool(((counts > 0) & (counts < len(region))).any()),
    }


def _frame_region_l2(x, y, mask, channels):
    support = mask[:, channels]
    frames = support.any(1)
    if not frames.any():
        return pending_metric("No observed coefficients in the requested region.",
                              ["observed regional target coefficients"])
    error = np.where(support[None], x[:, :, channels] - y[None, :, channels], 0.0)
    per_sample = np.linalg.norm(error[:, frames], axis=2).mean(1)
    return {"status": "computed", "value": float(per_sample.mean()),
            "per_sample": per_sample.tolist(), "unit": "coefficient",
            "definition": "mean_draw mean_observed_frame sqrt(sum_observed_region (pred-gt)^2)",
            "coverage": region_coverage(mask, channels)}


def literature_coefficient_metrics(samples, target, mask, times, *,
                                   lip_channels=BEAT_LIP12, upper_channels=BEAT_UPPER16, fps=25.0):
    """FaceDiffuser BEAT evaluator formulas in this project's ARKit order.

    MBE/LBE are frame-level region L2 norms, averaged over time and then draws.
    FDD is std over time of *summed squared raw coefficient magnitude*, not
    velocity or adjacent-frame differences, with population standard deviation.
    The signed result is GT minus prediction; ABS is taken per draw before mean.

    Mask adaptation: MBE/LBE omit unobserved elements from each norm without
    rescaling and omit completely unobserved frames. FDD uses frames on which
    every requested upper channel is observed, so its energy always has the
    same channels. Missing frames never become zero-valued reference frames.
    At least two complete-region frames are required for the temporal std.
    Defaults map the official FaceDiffuser BEAT column names to this project's
    ARKit order. Its upper mask covers eyes and nose, and excludes eyebrows.
    Regions are configurable; project Lip23/Upper9 are supplementary variants.
    """
    x, y, m, clock, adjacent = validate_inputs(samples, target, mask, times, fps)
    lips = validate_region(lip_channels, "lip_channels")
    upper = validate_region(upper_channels, "upper_channels")
    metrics = {
        "arkit_mbe": _frame_region_l2(x, y, m, tuple(range(52))),
        "arkit_lbe": _frame_region_l2(x, y, m, lips),
    }
    complete = m[:, upper].all(1)
    if complete.sum() < 2:
        for name in ("arkit_fdd_signed", "arkit_fdd_absolute"):
            metrics[name] = pending_metric(
                "Fewer than two frames have the entire requested upper region observed.",
                ["at least two fully observed upper-region frames"])
    else:
        gt_std = np.square(y[complete][:, upper]).sum(1).std(ddof=0)
        pred_std = np.square(x[:, complete][:, :, upper]).sum(2).std(axis=1, ddof=0)
        difference = gt_std - pred_std
        common = {"status": "computed", "unit": "coefficient_squared",
                  "gt_energy_std": float(gt_std), "pred_energy_std": pred_std.tolist(),
                  "coverage": region_coverage(m, upper),
                  "temporal_order_invariant": True, "ddof": 0}
        metrics["arkit_fdd_signed"] = {
            **common, "value": float(difference.mean()), "per_sample": difference.tolist(),
            "definition": "mean_draw [std_t(sum_upper gt^2) - std_t(sum_upper pred^2)]",
            "interpretation": "zero-centered; not monotonically lower-is-better",
        }
        metrics["arkit_fdd_absolute"] = {
            **common, "value": float(np.abs(difference).mean()),
            "per_sample": np.abs(difference).tolist(),
            "definition": "mean_draw abs(std_t(sum_upper gt^2) - std_t(sum_upper pred^2))",
            "interpretation": "lower-is-better magnitude discrepancy; not timing agreement",
        }
    return {
        "schema": "arkit_literature_metrics_v1",
        "representation": "raw ARKit52 coefficients; no clipping or standardization",
        "official_benchmark_reproduced": False,
        "definition_source": SOURCE,
        "scope": "coefficient-space formulas adapted to declared regions and masks",
        "region_adaptation": lips != BEAT_LIP12 or upper != BEAT_UPPER16,
        "region_protocol": (
            "FaceDiffuser BEAT official semantic masks mapped by channel name to project ARKit order"
            if lips == BEAT_LIP12 and upper == BEAT_UPPER16 else
            "custom declared coefficient regions; not official BEAT regional metric"
        ),
        "fdd_includes_eyebrows": bool(set(upper) & set(range(41, 46))),
        "samples": int(len(x)), "frames": int(len(y)), "fps": float(fps),
        "clock_gaps": int((~adjacent).sum()),
        "region_indices": {"lips": list(lips), "upper": list(upper)},
        "aggregation": "per-draw metric then mean; no best-of-k or averaged trajectory",
        "mask_protocol": (
            "MBE/LBE: observed channel norm per nonempty frame, no count normalization. "
            "FDD: whole-region complete frames only, population std; no time differences."
        ),
        "metrics": {**metrics, **pending_external_metrics()},
    }


def pending_metric(reason, required_inputs):
    return {"status": "pending", "value": None, "reason": reason,
            "required_inputs": list(required_inputs)}


def pending_external_metrics():
    """Declared dependencies for metrics not computable from coefficients alone.

    Future pipeline stages may replace these records after their own evaluator
    executes, supplying evaluator identity, weights hash, and input provenance.
    A proxy is deliberately not emitted under a literature metric's name.
    """
    av = ("Synchronized audio and a validated common-avatar render/SyncNet "
          "protocol are required; coefficient arrays alone are insufficient.")
    learned = ("A frozen train-only learned motion evaluator and its validated "
               "literature preprocessing/protocol have not been supplied.")
    return {
        "av_offset": pending_metric(av, ["audio", "common-avatar video",
                                          "validated SyncNet weights/protocol"]),
        "av_confidence": pending_metric(av, ["audio", "common-avatar video",
                                              "validated SyncNet weights/protocol"]),
        "multimodality": pending_metric(
            learned + " Repeated samples of each identical conditioning input are required.",
            ["fixed train-only motion evaluator", "repeated conditional samples",
             "literature sample/feature distance protocol"]),
        "fd": pending_metric(learned, ["fixed train-only motion evaluator",
                                       "reference and generated features",
                                       "literature sample pooling protocol"]),
        "wind": pending_metric(learned, ["fixed train-only motion evaluator",
                                         "reference and generated motion windows",
                                         "literature window/transport protocol"]),
    }


def _parse_channels(value):
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="NPZ with samples[S,T,52], target[T,52], boolean mask[T,52], times[T]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lip-channels", type=_parse_channels, default=BEAT_LIP12)
    parser.add_argument("--upper-channels", type=_parse_channels, default=BEAT_UPPER16)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        report = literature_coefficient_metrics(
            data["samples"], data["target"], data["mask"], data["times"],
            lip_channels=args.lip_channels, upper_channels=args.upper_channels, fps=args.fps)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "computed": [k for k, v in report["metrics"].items() if v["status"] == "computed"],
                      "pending": [k for k, v in report["metrics"].items() if v["status"] == "pending"]}))


if __name__ == "__main__":
    main()
