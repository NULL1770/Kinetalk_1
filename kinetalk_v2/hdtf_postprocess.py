"""Conservative, run-wise HDTF motion post-processing.

The MediaPipe source artefacts are immutable.  This module evaluates SG3/5/7
without crossing invalid-frame runs and writes an explicit processed artefact
only after a policy has been audited.  It does not claim that smoothing recovers
physical 3-D truth; short/faulty/event-distorting runs fall back to raw values.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .hdtf import SCHEMA as SOURCE_SCHEMA, atomic_json, atomic_jsonl, sha256

SCHEMA = "hdtf-motion-postprocess-v2.1"
FPS = 25
JAW = 17  # fixed only for the current rig contract; channel name is verified below
BLINK = (0, 7)


@dataclass(frozen=True)
class Policy:
    candidate_window: int = 5
    polyorder: int = 2
    closure_threshold: float = .05
    min_run_frames: int = 9
    min_peak_for_guard: float = .08
    min_peak_retention: float = .80
    max_peak_retention: float = 1.25
    max_closure_rate_delta: float = .20
    blink_min_duration_ratio: float = .50
    blink_max_duration_ratio: float = 1.50
    blink_event_height: float = .30
    blink_event_prominence: float = .12
    min_event_area_retention: float = .80
    max_event_area_retention: float = 1.25
    preserve_closure_events: bool = True

    def __post_init__(self) -> None:
        if self.candidate_window not in (3, 5, 7) or self.candidate_window <= self.polyorder:
            raise ValueError("candidate window must be 3, 5, or 7 and exceed polyorder")


def _array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _runs(mask: np.ndarray, times: np.ndarray | None = None) -> Iterable[tuple[int, int]]:
    mask = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    runs = []
    for start, end in zip(edges[::2].tolist(), edges[1::2].tolist()):
        if times is None:
            runs.append((start, end))
        else:
            # No derivative or convolution can bridge a timestamp discontinuity.
            dt = np.diff(np.asarray(times[start:end], np.float64))
            cuts = start + np.flatnonzero(np.abs(dt - 1 / FPS) > 1e-6) + 1
            points = [start, *cuts.tolist(), end]
            runs.extend(zip(points[:-1], points[1:]))
    return runs


def blink_events(raw: np.ndarray, candidate: np.ndarray, policy: Policy) -> list[dict[str, float]]:
    """Evaluate actual local peaks, not just a whole-recording maximum."""
    raw, candidate = np.asarray(raw, float), np.asarray(candidate, float)
    peaks, _ = find_peaks(raw, height=policy.blink_event_height,
                          prominence=policy.blink_event_prominence, distance=2)
    events = []
    for peak in peaks:
        height = float(raw[peak])
        lo, hi = int(peak), int(peak + 1)
        while lo > 0 and raw[lo - 1] >= .5 * height: lo -= 1
        while hi < len(raw) and raw[hi] >= .5 * height: hi += 1
        # Include one filter radius on both sides to measure redistributed area.
        start, end = max(0, lo - policy.candidate_window // 2), min(len(raw), hi + policy.candidate_window // 2)
        r, c = raw[start:end], candidate[start:end]
        threshold = .5 * height
        events.append({"peak_frame": int(peak), "start_frame": start, "end_frame": end,
                       # Same event neighborhood on both sides ensures raw
                       # identity is exactly one even for nearby blink peaks.
                       "peak_retention": _ratio(float(c.max(initial=0)), float(r.max(initial=0))),
                       "duration_ratio": _ratio(float((c >= threshold).sum()), float((r >= threshold).sum())),
                       "area_ratio": _ratio(float(c.sum()), float(r.sum()))})
    return events


def _channel_indices(channels: np.ndarray | list[str]) -> tuple[int, tuple[int, int]]:
    names = [str(s).lower() for s in channels]
    if len(names) != 52 or "jawopen" not in names:
        raise ValueError("postprocess requires the named fixed 52-channel rig")
    jaw = names.index("jawopen")
    blink = (names.index("eyeblinkleft"), names.index("eyeblinkright"))
    return jaw, blink


def _event_stats(x: np.ndarray, threshold: float = .10) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    active = x > threshold
    return {"peak": float(np.max(x, initial=0.)), "duration_frames": float(active.sum()),
            "area": float(np.maximum(x, 0.).sum()),
            "closure_rate": float((x <= threshold).mean())}


def _ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-8 else 1.


def candidate_metrics(raw: np.ndarray, candidate: np.ndarray, jaw: int, blink: tuple[int, int],
                      threshold: float = .05, *, motion_valid: np.ndarray | None = None,
                      times: np.ndarray | None = None, channel_mask: np.ndarray | None = None,
                      policy: Policy | None = None) -> dict[str, Any]:
    raw = np.asarray(raw, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if raw.shape != candidate.shape:
        raise ValueError("raw/candidate shape mismatch")
    valid = np.asarray(motion_valid, bool) if motion_valid is not None else np.ones(len(raw), bool)
    cm = np.asarray(channel_mask, bool) if channel_mask is not None else np.ones(raw.shape[1], bool)
    if not valid.any():
        raise ValueError("no valid motion for postprocessing metrics")
    pairs = valid[1:] & valid[:-1]
    triples = valid[2:] & valid[1:-1] & valid[:-2]
    if times is not None:
        clock_pairs = np.abs(np.diff(np.asarray(times, float)) - 1 / FPS) <= 1e-6
        pairs &= clock_pairs
        triples &= clock_pairs[1:] & clock_pairs[:-1]
    d1_raw, d1_c = np.diff(raw, axis=0)[pairs][:, cm], np.diff(candidate, axis=0)[pairs][:, cm]
    d2_raw, d2_c = np.diff(raw, n=2, axis=0)[triples][:, cm], np.diff(candidate, n=2, axis=0)[triples][:, cm]
    average = lambda x: float(np.abs(x).mean()) if x.size else 0.
    result: dict[str, Any] = {"n_frames": int(len(raw)), "channels": int(raw.shape[1]),
        "mean_abs_velocity_ratio": _ratio(average(d1_c), average(d1_raw)),
        "mean_abs_acceleration_ratio": _ratio(average(d2_c), average(d2_raw)),
        "jaw": {}, "blink": {}, "channel_mean_abs_delta": float(np.abs(candidate[valid][:,cm] - raw[valid][:,cm]).mean()),
        "valid_frames": int(valid.sum()), "valid_adjacent_pairs": int(pairs.sum()),
        "valid_adjacent_triples": int(triples.sum())}
    jaw_raw = _event_stats(raw[valid, jaw], threshold)
    jaw_c = _event_stats(candidate[valid, jaw], threshold)
    result["jaw"] = {"raw": jaw_raw, "candidate": jaw_c,
                      "peak_retention": _ratio(jaw_c["peak"], jaw_raw["peak"]),
                      "closure_rate_delta": jaw_c["closure_rate"] - jaw_raw["closure_rate"],
                      "duration_ratio": _ratio(jaw_c["duration_frames"], jaw_raw["duration_frames"]),
                      "area_ratio": _ratio(jaw_c["area"], jaw_raw["area"])}
    for i in blink:
        r, c = _event_stats(raw[valid, i], threshold), _event_stats(candidate[valid, i], threshold)
        events = []
        for start, end in _runs(valid, times):
            events.extend(blink_events(raw[start:end, i], candidate[start:end, i], policy or Policy()))
        result["blink"][str(i)] = {"raw": r, "candidate": c,
            "peak_retention": _ratio(c["peak"], r["peak"]),
            "duration_ratio": _ratio(c["duration_frames"], r["duration_frames"]),
            "area_ratio": _ratio(c["area"], r["area"]),
            "local_events": events, "local_event_count": len(events),
            "local_peak_retention_p05": float(np.percentile([e["peak_retention"] for e in events],5)) if events else 1.,
            "local_peak_retention_median": float(np.median([e["peak_retention"] for e in events])) if events else 1.,
            "local_duration_ratio_median": float(np.median([e["duration_ratio"] for e in events])) if events else 1.,
            "local_area_ratio_median": float(np.median([e["area_ratio"] for e in events])) if events else 1.}
    result["closure_threshold"] = threshold
    return result


def smooth_runs(raw: np.ndarray, motion_valid: np.ndarray, policy: Policy, *, channels: np.ndarray | None = None, times: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (processed, smoothed_mask[T,C], fallback_mask[T,C])."""
    x = np.asarray(raw, dtype=np.float32)
    valid = np.asarray(motion_valid, bool)
    if x.ndim != 2 or x.shape[0] != valid.size or x.shape[1] != 52:
        raise ValueError("expected motion (T,52) and motion_valid (T,)")
    jaw, blink = _channel_indices(channels) if channels is not None else (JAW, BLINK)
    out, smoothed, fallback = x.copy(), np.zeros_like(x, bool), np.zeros_like(x, bool)
    # Invalid source frames are never filled/interpolated. Mark them as raw
    # fallback so downstream provenance can distinguish them from smoothed data.
    fallback[~valid] = True
    for start, end in _runs(valid, times):
        if end - start < max(policy.min_run_frames, policy.candidate_window):
            fallback[start:end] = True
            continue
        segment = x[start:end].astype(np.float64)
        candidate = savgol_filter(segment, window_length=policy.candidate_window,
                                  polyorder=policy.polyorder, axis=0, mode="interp")
        candidate = np.clip(candidate, 0., 1.).astype(np.float32)
        # Event-preservation guard.  A peak which disappears or a grossly
        # changed jaw closure proxy is retained raw for that channel/run.
        guard = np.zeros(52, bool)
        for channel in (jaw, *blink):
            r = _event_stats(segment[:, channel], policy.closure_threshold)
            c = _event_stats(candidate[:, channel], policy.closure_threshold)
            peak_ratio = _ratio(c["peak"], r["peak"])
            if r["peak"] >= policy.min_peak_for_guard and not (policy.min_peak_retention <= peak_ratio <= policy.max_peak_retention):
                guard[channel] = True
            if channel == jaw and abs(c["closure_rate"] - r["closure_rate"]) > policy.max_closure_rate_delta:
                guard[channel] = True
            if channel in blink and r["duration_frames"] >= 2:
                ratio = _ratio(c["duration_frames"], r["duration_frames"])
                if not (policy.blink_min_duration_ratio <= ratio <= policy.blink_max_duration_ratio):
                    guard[channel] = True
        for channel in blink:
            for event in blink_events(segment[:,channel], candidate[:,channel], policy):
                if not (policy.min_peak_retention <= event["peak_retention"] <= policy.max_peak_retention
                        and policy.blink_min_duration_ratio <= event["duration_ratio"] <= policy.blink_max_duration_ratio
                        and policy.min_event_area_retention <= event["area_ratio"] <= policy.max_event_area_retention):
                    guard[channel] = True
        out[start:end] = candidate
        for channel in np.flatnonzero(guard):
            out[start:end, channel] = segment[:, channel]
            fallback[start:end, channel] = True
        smoothed[start:end] = ~guard
        if policy.preserve_closure_events:
            # Preserve every observed jaw closure episode, including its filter
            # neighborhood. Never call a short closure high-frequency noise.
            jaw_closure = segment[:,jaw] <= policy.closure_threshold
            protected = np.zeros(end-start,bool)
            for lo, hi in _runs(jaw_closure):
                protected[max(0,lo-policy.candidate_window//2):min(end-start,hi+policy.candidate_window//2)] = True
            out[start:end,jaw][protected] = segment[protected,jaw]
            fallback[start:end,jaw][protected] = True
            smoothed[start:end,jaw][protected] = False
    return out, smoothed, fallback


def load_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        required = ("motion", "times", "motion_valid", "channel_mask", "channels")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"source NPZ missing {missing}: {path}")
        result = {key: np.asarray(d[key]).copy() for key in required}
        result["valid"] = np.asarray(d["valid"], bool).copy() if "valid" in d else result["motion_valid"].copy()
    if result["motion"].shape[1] != 52 or not np.all(np.diff(result["times"]) > 0):
        raise ValueError(f"invalid source clock/motion shape: {path}")
    return result


def process_file(source: Path, output: Path, policy: Policy, *, source_record: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_hash = sha256(source)
    binding = {"schema":SCHEMA,"policy":asdict(policy),"raw_motion_sha256":raw_hash,
               "implementation_sha256":sha256(Path(__file__))}
    signature = hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    receipt_path = output.parent.parent / "receipts" / (source.stem + ".json")
    if receipt_path.exists() and output.exists():
        try:
            cached = json.loads(receipt_path.read_text(encoding="utf-8"))
            if cached.get("postprocess_signature")==signature and cached["motion_processed_sha256"]==sha256(output):
                return dict(cached,processing_status="cached")
        except (ValueError,KeyError,OSError):
            pass
    src = load_motion(source)
    names = np.asarray(src["channels"])
    jaw, blink = _channel_indices(names)
    if names[jaw].lower() != "jawopen":
        raise ValueError("JawOpen channel contract failed")
    candidate, smoothed, fallback = smooth_runs(src["motion"], src["motion_valid"], policy,
                                                channels=src["channels"], times=src["times"])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(src)
    payload.update(motion=candidate, motion_processed=candidate,
                   motion_raw=src["motion"], smoothed_mask=smoothed,
                   fallback_mask=fallback, source_schema=np.asarray(SOURCE_SCHEMA),
                   postprocess_schema=np.asarray(SCHEMA), candidate_window=np.int64(policy.candidate_window),
                   postprocess_signature=np.asarray(signature),
                   policy_json=np.asarray(json.dumps(asdict(policy), sort_keys=True)),
                   raw_motion_sha256=np.asarray(raw_hash))
    tmp = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, output)
    metrics = candidate_metrics(src["motion"], candidate, jaw, blink, policy.closure_threshold,
                                motion_valid=src["motion_valid"], times=src["times"],
                                channel_mask=src["channel_mask"], policy=policy)
    rec = dict(source_record or {}, source_id=source.stem, source_motion_path=str(source.resolve()),
               motion_processed_path=str(output.resolve()), raw_motion_sha256=raw_hash,
               motion_processed_sha256=sha256(output), postprocess_schema=SCHEMA,
               motion_processed=str(output.resolve()),postprocess_signature=signature,postprocess_binding=binding,
               source_schema=SOURCE_SCHEMA, candidate_window=policy.candidate_window,
               fallback_fraction=float(fallback.mean()), smoothed_fraction=float(smoothed.mean()),
               n_frames=int(len(candidate)), quality_valid_fraction=float(src["motion_valid"].mean()),
               metrics=metrics, processing_status="ok")
    rec.update(emotion_id=-1,emotion_valid=False,intensity_id=-1,intensity_valid=False)
    atomic_json(receipt_path,rec)
    return rec


def aggregate_audit(rows: list[dict[str, Any]], candidate: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty postprocess audit")
    def med(path: tuple[str, ...]) -> float:
        vals = []
        keys = path[1:] if path and path[0] == "metrics" else path
        for row in rows:
            val: Any = row["metrics"]
            try:
                for key in keys:
                    val = val[key]
            except (KeyError, TypeError):
                continue
            if np.isfinite(val): vals.append(float(val))
        return float(np.median(vals)) if vals else float("nan")
    result = {"schema": SCHEMA, "candidate_window": candidate, "n_sources": len(rows),
            "n_frames": int(sum(r["n_frames"] for r in rows)),
            "mean_fallback_fraction": float(np.mean([r["fallback_fraction"] for r in rows])),
            "mean_smoothed_fraction": float(np.mean([r["smoothed_fraction"] for r in rows])),
            "median_velocity_ratio": med(("metrics", "mean_abs_velocity_ratio")),
            "median_acceleration_ratio": med(("metrics", "mean_abs_acceleration_ratio")),
            "median_jaw_peak_retention": med(("metrics", "jaw", "peak_retention")),
            "median_jaw_closure_rate_delta": med(("metrics", "jaw", "closure_rate_delta")),
            "median_jaw_duration_ratio": med(("metrics", "jaw", "duration_ratio")),
            "median_blink_left_peak_retention": med(("metrics", "blink", "0", "peak_retention")),
            "median_blink_right_peak_retention": med(("metrics", "blink", "7", "peak_retention")),
            "median_blink_left_duration_ratio": med(("metrics", "blink", "0", "duration_ratio")),
            "median_blink_right_duration_ratio": med(("metrics", "blink", "7", "duration_ratio")),
            "median_blink_left_area_ratio": med(("metrics", "blink", "0", "area_ratio")),
            "median_blink_right_area_ratio": med(("metrics", "blink", "7", "area_ratio"))}
    for side,key in (("left","0"),("right","7")):
        events = [e for r in rows for e in r["metrics"]["blink"][key].get("local_events", [])]
        result[f"blink_{side}_event_count"] = len(events)
        for metric in ("peak_retention", "duration_ratio", "area_ratio"):
            values = [e[metric] for e in events]
            result[f"blink_{side}_local_{metric}_median"] = float(np.median(values)) if values else 1.
            result[f"blink_{side}_local_{metric}_p05"] = float(np.percentile(values,5)) if values else 1.
            result[f"blink_{side}_local_{metric}_p95"] = float(np.percentile(values,95)) if values else 1.
    return result
