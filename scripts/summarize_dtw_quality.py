"""Independent summary for a v3 DTW quality.jsonl."""
from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def percentiles(rows, key):
    values = [float(r[key]) for r in rows
              if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not values:
        return None
    return {str(p): round(float(np.percentile(values, p)), 6)
            for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("quality", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.quality.read_text(encoding="utf-8").splitlines() if line.strip()]
    emotion_by_clip = {}
    if args.manifest:
        manifest_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        emotion_by_clip = {r["clip_id"]: r.get("emotion") for r in manifest_rows}
        for row in rows:
            row["emotion"] = emotion_by_clip.get(row.get("source_clip_id"))
    cross = [r for r in rows if r.get("source_clip_id") != r.get("reference_clip_id")]
    summary = {
        "rows": len(rows),
        "status": dict(Counter(r.get("status") for r in rows)),
        "teacher_eligible": sum(bool(r.get("teacher_eligible")) for r in rows),
        "identity": len(rows) - len(cross),
        "cross": len(cross),
        "all_rows": {k: percentiles(rows, k) for k in (
            "repeat_ratio", "excess_repeat_ratio", "max_same_direction_run",
            "max_hold_seconds", "local_slope_p01", "local_slope_p99",
            "cycle_p95_seconds", "local_valid_ratio", "mean_cost",
            "source_duration_difference", "reference_duration_difference",
            "layer9_cost_diagnostic")},
        "cross_rows": {k: percentiles(cross, k) for k in (
            "repeat_ratio", "excess_repeat_ratio", "max_same_direction_run",
            "max_hold_seconds", "local_slope_p01", "local_slope_p99",
            "cycle_p95_seconds", "local_valid_ratio", "mean_cost")},
    }
    bad = [r for r in cross if r.get("repeat_ratio", 0) > .35
           or r.get("max_same_direction_run", 0) > 12
           or r.get("local_slope_p01", 0) < .25
           or r.get("local_slope_p99", 0) > 4
           or r.get("cycle_p95_seconds", 0) > .08
           or r.get("local_valid_ratio", 0) < .8]
    strict_teacher = [r for r in rows if r.get("source_clip_id") == r.get("reference_clip_id") or (
        r.get("repeat_ratio", 1) <= .35
        and r.get("max_same_direction_run", 99) <= 3
        and r.get("local_slope_p01", 0) >= .25
        and r.get("local_slope_p99", 99) <= 4
        and r.get("cycle_p95_seconds", 99) <= .08
        and r.get("local_valid_ratio", 0) >= .8)]
    summary["strict_teacher_eligible"] = len(strict_teacher)
    summary["strict_teacher_fraction"] = len(strict_teacher) / max(len(rows), 1)
    summary["strict_cross_teacher_eligible"] = sum(r in strict_teacher for r in cross)
    summary["cross_triage_bad"] = len(bad)
    summary["cross_triage_bad_fraction"] = len(bad) / max(len(cross), 1)
    by_emotion = defaultdict(list)
    for row in cross:
        by_emotion[str(row.get("emotion"))].append(row)
    summary["by_emotion"] = {
        emotion: {
            "rows": len(group),
            "repeat_ratio": percentiles(group, "repeat_ratio"),
            "max_same_direction_run": percentiles(group, "max_same_direction_run"),
            "local_slope_p01": percentiles(group, "local_slope_p01"),
            "local_slope_p99": percentiles(group, "local_slope_p99"),
            "cycle_p95_seconds": percentiles(group, "cycle_p95_seconds"),
            "mean_cost": percentiles(group, "mean_cost"),
            "strict_teacher_eligible": sum(r in strict_teacher for r in group),
        } for emotion, group in sorted(by_emotion.items())
    }
    summary["worst_examples"] = [
        {k: row.get(k) for k in ("source_clip_id", "reference_clip_id", "emotion",
                                  "repeat_ratio", "max_same_direction_run",
                                  "local_slope_p01", "local_slope_p99",
                                  "cycle_p95_seconds", "local_valid_ratio", "mean_cost")}
        for row in sorted(bad, key=lambda r: (-r.get("repeat_ratio", 0),
                                              -r.get("cycle_p95_seconds", 0)))[:30]
    ]
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
