"""Compare two native-time DTW releases and audit teacher attenuation."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MOUTH = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
         28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 51]
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
    "noseSneerLeft", "noseSneerRight", "tongueOut"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def percentile(values: list[float]) -> dict[str, float] | None:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not len(finite):
        return None
    return {str(p): round(float(np.percentile(finite, p)), 6)
            for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def load_mapping(row: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(row["artifact"], allow_pickle=False) as z:
        times = z["reference_feature_times"].astype(np.float64)
        mapping = z["source_time_on_reference"].astype(np.float64)
        arrays = {key: z[key].copy() for key in (
            "neutral_teacher_on_source", "native_teacher_mask",
            "reference_time_on_source", "source_motion_times",
            "source_feature_times")}
    return times, mapping, arrays


def path_disagreement(a: dict, b: dict) -> tuple[float, float]:
    at, am, _ = load_mapping(a)
    bt, bm, _ = load_mapping(b)
    start, end = max(at[0], bt[0]), min(at[-1], bt[-1])
    if end <= start:
        return float("nan"), float("nan")
    grid = np.arange(start, end + 1e-9, 0.01)
    error = np.abs(np.interp(grid, at, am) - np.interp(grid, bt, bm))
    return float(np.median(error)), float(np.percentile(error, 95))


def motion_path(record: dict, bs_root: Path) -> Path:
    if record.get("bs_rel"):
        candidate = bs_root / record["bs_rel"]
        if candidate.exists():
            return candidate
    for key in ("npz_final", "bs"):
        candidate = Path(record.get(key, ""))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(record["clip_id"])


def robust_range(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0)


def attenuation(row: dict, records: dict[str, dict], bs_root: Path) -> tuple[dict[int, float], dict[int, float]]:
    _, _, arrays = load_mapping(row)
    mask = arrays["native_teacher_mask"].astype(bool)
    if mask.sum() < 10:
        return {}, {}
    ref = np.load(motion_path(records[row["reference_clip_id"]], bs_root),
                  allow_pickle=False)["coeffs"].astype(np.float32)
    teacher = arrays["neutral_teacher_on_source"].astype(np.float32)
    source_times = arrays["source_motion_times"].astype(np.float64)
    source_feature_times = arrays["source_feature_times"].astype(np.float64)
    ref_on_source = arrays["reference_time_on_source"].astype(np.float64)
    mapped_times = np.interp(source_times, source_feature_times, ref_on_source)
    nearest = ref[np.clip(np.rint(mapped_times * 25).astype(int), 0, len(ref) - 1)]
    full_range = robust_range(ref[:, MOUTH])
    linear_range = robust_range(teacher[mask][:, MOUTH])
    nearest_range = robust_range(nearest[mask][:, MOUTH])
    # Tiny-range channels produce meaningless ratios.  A 0.01 P95-P05 floor
    # keeps only visibly active coefficients on the normalized ARKit scale.
    effective = {index: float(linear_range[pos] / full_range[pos])
                 for pos, index in enumerate(MOUTH) if full_range[pos] > .01}
    interpolation = {index: float(linear_range[pos] / nearest_range[pos])
                     for pos, index in enumerate(MOUTH) if nearest_range[pos] > .01}
    return effective, interpolation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-a", type=Path, required=True)
    ap.add_argument("--quality-b", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--bs-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--teacher-output", type=Path, default=None)
    args = ap.parse_args()

    rows_a = {r["source_clip_id"]: r for r in read_jsonl(args.quality_a)}
    rows_b = {r["source_clip_id"]: r for r in read_jsonl(args.quality_b)}
    records = {r["clip_id"]: r for r in read_jsonl(args.manifest)}
    if rows_a.keys() != rows_b.keys():
        raise ValueError("quality manifests do not contain the same source clips")

    pair_rows, median_errors, p95_errors = [], [], []
    effective_by_channel = defaultdict(list)
    interpolation_by_channel = defaultdict(list)
    by_emotion = defaultdict(lambda: Counter())
    for clip_id in sorted(rows_a):
        a, b = rows_a[clip_id], rows_b[clip_id]
        identity = a["source_clip_id"] == a["reference_clip_id"]
        a_ok, b_ok = bool(a.get("teacher_eligible")), bool(b.get("teacher_eligible"))
        emotion = str(records[clip_id].get("emotion"))
        category = "both" if a_ok and b_ok else "a_only" if a_ok else "b_only" if b_ok else "neither"
        by_emotion[emotion][category] += 1
        median_ms = p95_ms = None
        if not identity:
            median_s, p95_s = path_disagreement(a, b)
            median_ms, p95_ms = median_s * 1000, p95_s * 1000
            median_errors.append(median_ms)
            p95_errors.append(p95_ms)
        release_teacher = bool(
            a_ok and b_ok and (identity or (median_ms <= 20 and p95_ms <= 80)))
        if a_ok and b_ok and not identity:
            effective, interpolation = attenuation(a, records, args.bs_root)
            for index, value in effective.items():
                effective_by_channel[index].append(value)
            for index, value in interpolation.items():
                interpolation_by_channel[index].append(value)
        pair_rows.append({"source_clip_id": clip_id,
                          "reference_clip_id": a["reference_clip_id"],
                          "emotion": emotion, "category": category,
                          "path_median_error_ms": median_ms,
                          "path_p95_error_ms": p95_ms,
                          "teacher_eligible": release_teacher,
                          "teacher_artifact": a["artifact"],
                          "audit_artifact_b": b["artifact"]})

    cross = [r for r in pair_rows if r["path_median_error_ms"] is not None]
    consensus = [r for r in cross if r["category"] == "both"]
    consensus_rules = {
        "both_feature_gates": len(consensus),
        "both_and_median_le_20ms": sum(r["path_median_error_ms"] <= 20 for r in consensus),
        "both_and_p95_le_80ms": sum(r["path_p95_error_ms"] <= 80 for r in consensus),
        "both_and_median_le_20ms_and_p95_le_80ms": sum(
            r["path_median_error_ms"] <= 20 and r["path_p95_error_ms"] <= 80
            for r in consensus),
    }
    summary = {
        "quality_a": str(args.quality_a), "quality_b": str(args.quality_b),
        "pairs": len(pair_rows), "cross_pairs": len(cross),
        "agreement": dict(Counter(r["category"] for r in pair_rows)),
        "by_emotion": {key: dict(value) for key, value in sorted(by_emotion.items())},
        "path_median_error_ms": percentile(median_errors),
        "path_p95_error_ms": percentile(p95_errors),
        "cross_pairs_below_error_threshold": {
            str(ms): sum(r["path_median_error_ms"] <= ms for r in cross)
            for ms in (20, 40, 60, 80, 100)},
        "consensus_teacher_rules": consensus_rules,
        "teacher_effective_p95_p05_ratio": percentile(
            [value for values in effective_by_channel.values() for value in values]),
        "linear_vs_nearest_interpolation_p95_p05_ratio": percentile(
            [value for values in interpolation_by_channel.values() for value in values]),
        "amplitude_by_channel": {
            ARKIT_NAMES[index]: {
                "index": index,
                "effective": percentile(effective_by_channel[index]),
                "interpolation": percentile(interpolation_by_channel[index]),
                "samples": len(effective_by_channel[index]),
            } for index in MOUTH},
        "amplitude_note": (
            "Ratios use mouth-channel P95-P05. Effective compares the warped teacher on valid "
            "source frames with the full neutral reference; interpolation compares linear and "
            "nearest-neighbor sampling at the same DTW-mapped times. Ratios require the "
            "denominator P95-P05 > 0.01. Only cross pairs accepted by both releases are included."),
        "worst_path_disagreements": sorted(
            cross, key=lambda r: r["path_p95_error_ms"], reverse=True)[:50],
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.teacher_output:
        eligible = [row for row in pair_rows if row["teacher_eligible"]]
        args.teacher_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eligible),
            encoding="utf-8")
        summary["teacher_output"] = str(args.teacher_output)
        summary["release_teacher_count"] = len(eligible)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
