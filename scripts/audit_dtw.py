"""Audit existing aligned_dtw_v2 outputs without changing them.

The audit is deliberately independent of the DTW builder's status field.  It
recomputes path geometry from saved paths and checks whether warped audio
activity still agrees with mouth motion on the common timeline.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MOUTH = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 51]


def percentile(x: list[float], p: float) -> float | None:
    return float(np.percentile(np.asarray(x, np.float64), p)) if x else None


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) < 4 or len(b) != len(a):
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / den) if den > 1e-10 else 0.0


def max_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 6) -> tuple[float, int]:
    best = (-1.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            score = corr(a[:-lag], b[lag:])
        elif lag < 0:
            score = corr(a[-lag:], b[:lag])
        else:
            score = corr(a, b)
        if score > best[0]:
            best = (score, lag)
    return best


def path_stats(path: np.ndarray) -> dict[str, float | int]:
    path = np.asarray(path, np.int64)
    if len(path) < 2:
        return {"path_len": int(len(path)), "repeat_ratio": 0.0, "horizontal_ratio": 0.0,
                "vertical_ratio": 0.0, "diagonal_ratio": 0.0, "max_repeat_run": 0,
                "local_slope_p01": 0.0, "local_slope_p99": 0.0}
    d = np.diff(path, axis=0)
    horizontal = (d[:, 0] == 0) & (d[:, 1] > 0)
    vertical = (d[:, 1] == 0) & (d[:, 0] > 0)
    diagonal = (d[:, 0] > 0) & (d[:, 1] > 0)
    repeat = horizontal | vertical
    # A run is measured in consecutive repeated steps, independent of direction.
    max_run = run = 0
    for value in repeat:
        run = run + 1 if value else 0
        max_run = max(max_run, run)
    # Local path slope: source frames consumed per reference frame.  Values far
    # from 1 mean strong local time compression/expansion.
    slopes: list[float] = []
    for i in range(len(path) - 1):
        j = min(len(path) - 1, i + 8)
        ds = path[j, 0] - path[i, 0]
        dr = path[j, 1] - path[i, 1]
        if dr > 0:
            slopes.append(float(ds / dr))
    return {
        "path_len": int(len(path)),
        "repeat_ratio": float(repeat.mean()),
        "horizontal_ratio": float(horizontal.mean()),
        "vertical_ratio": float(vertical.mean()),
        "diagonal_ratio": float(diagonal.mean()),
        "max_repeat_run": int(max_run),
        "local_slope_p01": percentile(slopes, 1) or 0.0,
        "local_slope_p99": percentile(slopes, 99) or 0.0,
    }


def motion_activity(bs: np.ndarray) -> np.ndarray:
    idx = [i for i in MOUTH if i < bs.shape[-1]]
    mouth = bs[:, idx]
    return np.linalg.norm(np.diff(mouth, axis=0, prepend=mouth[:1]), axis=-1)


def audio_activity(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    dx = np.diff(x, axis=0, prepend=x[:1])
    # The saved two-channel stream is an audio/prosody stream.  Use both level
    # and change so the metric does not depend on undocumented channel names.
    return np.linalg.norm(x - np.median(x, axis=0, keepdims=True), axis=-1) + 0.5 * np.linalg.norm(dx, axis=-1)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "p01": percentile(values, 1), "p10": percentile(values, 10),
            "median": percentile(values, 50), "p90": percentile(values, 90),
            "p99": percentile(values, 99), "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--quality", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-lag", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()
    root = args.root
    quality_path = args.quality or root / "quality.jsonl"
    rows = [json.loads(line) for line in quality_path.open(encoding="utf-8") if line.strip()]
    path_root = root / "paths"
    bs_root = root / "bs"
    audio_root = root / "audio"
    by_status: Counter[str] = Counter(str(r.get("status")) for r in rows)
    selected = [r for r in rows if r.get("status") in {"ok", "cached", "review"}]
    if args.max_samples:
        selected = selected[: args.max_samples]

    metric_values: dict[str, list[float]] = defaultdict(list)
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    anomalies: list[dict[str, object]] = []
    missing = Counter()
    for row in selected:
        cid = str(row.get("clip_id"))
        path_file = path_root / f"{cid}.npz"
        bs_file = bs_root / f"{cid}.npz"
        dataset = str(row.get("dataset", "mead"))
        audio_candidates = [audio_root / dataset / f"{cid}.npz", audio_root / "mead" / f"{cid}.npz", audio_root / f"{cid}.npz"]
        audio_file = next((p for p in audio_candidates if p.exists()), None)
        if not path_file.exists():
            missing["path"] += 1
            continue
        if not bs_file.exists():
            missing["bs"] += 1
            continue
        if audio_file is None:
            missing["audio"] += 1
            continue
        with np.load(path_file, allow_pickle=False) as z:
            path = np.asarray(z["path"])
        with np.load(bs_file, allow_pickle=False) as z:
            bs = np.asarray(z["coeffs"], np.float32)
        with np.load(audio_file, allow_pickle=False) as z:
            audio = np.asarray(z["feat"], np.float32)
        ps = path_stats(path)
        act_a = audio_activity(audio)
        act_m = motion_activity(bs)
        n = min(len(act_a), len(act_m))
        aligned_corr, aligned_lag = max_lag_corr(act_a[:n], act_m[:n], args.max_lag)
        for k in ("repeat_ratio", "horizontal_ratio", "vertical_ratio", "max_repeat_run", "local_slope_p01", "local_slope_p99"):
            metric_values[k].append(float(ps[k]))
        metric_values["event_corr"].append(float(aligned_corr))
        metric_values["event_lag"].append(float(aligned_lag))
        group = str(row.get("emotion") or "unknown").lower()
        for k in ("repeat_ratio", "max_repeat_run", "local_slope_p01", "local_slope_p99", "event_corr"):
            groups[group][k].append(float(ps[k]) if k != "event_corr" else float(aligned_corr))
        bad = (
            ps["repeat_ratio"] > 0.50 or ps["max_repeat_run"] > 20 or
            ps["local_slope_p01"] < 0.25 or ps["local_slope_p99"] > 4.0 or
            aligned_corr < 0.10
        )
        if bad:
            anomalies.append({"clip_id": cid, "status": row.get("status"), "emotion": row.get("emotion"),
                              **ps, "event_corr": aligned_corr, "event_lag": aligned_lag,
                              "builder_mean_cost": row.get("mean_cost"), "builder_av_corr": row.get("av_corr")})

    report = {
        "schema_version": 1,
        "root": str(root),
        "quality_rows": len(rows),
        "selected_rows": len(selected),
        "builder_status": dict(by_status),
        "missing_artifacts": dict(missing),
        "metrics": {k: summarize(v) for k, v in metric_values.items()},
        "by_emotion": {g: {k: summarize(v) for k, v in values.items()} for g, values in sorted(groups.items())},
        "audit_gate_proposal": {
            "hard_reject": ["repeat_ratio > 0.50", "max_repeat_run > 20", "local_slope_p01 < 0.25", "local_slope_p99 > 4.0", "event_corr < 0.10"],
            "note": "Path thresholds are geometry triage. event_corr uses the saved two-channel audio proxy and is not a phoneme or raw-audio lip-sync metric; replace it with raw energy/F0/Hubert or phoneme-boundary checks before final training gate.",
        },
        "anomaly_count": len(anomalies),
        "anomaly_examples": sorted(anomalies, key=lambda x: (float(x.get("event_corr", 0)), -float(x.get("repeat_ratio", 0))))[:200],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"], "anomaly_count": len(anomalies), "missing": dict(missing)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
