"""Coarse positive-excursion event teacher with fit-only noise thresholds.

Events are weak motion labels, not semantic annotations. No query anchor,
mean, scale or amplitude is supplied to the generator. Native gaps and missing
motion channels split runs; incomplete edge excursions remain unknown.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
from scipy.signal import find_peaks

FPS = 25.0
GROUPS = {"up": (2, 3, 4), "down": (0, 1), "squint": (5, 7), "wide": (6, 8)}
GROUP_NAMES = tuple(GROUPS)
SCHEMA = "event_excursion_teacher_v2"


@dataclass(frozen=True)
class EventThresholds:
    schema: str
    fps: float
    threshold: dict[str, float]
    min_frames: int = 4
    floor: float = .02
    noise_mad: dict[str, float] | None = None
    noise_samples: dict[str, int] | None = None
    fit_clips: int = 0

    def to_dict(self):
        return asdict(self)


def as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def valid_runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def validate_motion(motion, valid, motion_mask=None):
    x, m = as_numpy(motion).astype(np.float64), as_numpy(valid)
    if x.ndim != 2 or x.shape[1] != 9 or m.shape != (len(x),) or m.dtype != bool:
        raise ValueError("motion [T,9] and Boolean native valid [T] required")
    observed = np.ones_like(x, dtype=bool) if motion_mask is None else as_numpy(motion_mask)
    if observed.shape == (9,):
        observed = np.broadcast_to(observed, x.shape)
    if observed.shape != x.shape or observed.dtype != bool:
        raise ValueError("motion_mask must be Boolean [T,9] or [9]")
    observed = observed & m[:, None]
    if not np.isfinite(x[observed]).all():
        raise ValueError("observed motion values must be finite")
    return x, m, observed


def clip_arrays(row):
    motion = row.get("target", row.get("motion9"))
    valid = row.get("valid", row.get("native_valid"))
    if motion is None or valid is None:
        raise ValueError("clip requires target/motion9 and valid/native_valid")
    return validate_motion(motion, valid, row.get("motion_mask", row.get("observed9")))


def smooth_run(values):
    # Fixed triangular filter is offset invariant and non-overshooting. Edge
    # padding is only local to this run; edge evidence is excluded below.
    return np.convolve(np.pad(values, (2, 2), mode="edge"), [1/9, 2/9, 3/9, 2/9, 1/9], mode="valid")


def fit_event_thresholds(clips: Iterable[Mapping], *, fps=FPS, min_frames=4, floor=.02):
    rows = list(clips)  # generator iterables must survive all passes
    if not rows or fps <= 0 or min_frames < 2 or floor <= 0:
        raise ValueError("nonempty fit clips, positive fps/floor and min_frames >= 2 required")
    residuals = {g: [] for g in GROUPS}
    for row in rows:
        x, _, observed = clip_arrays(row)
        for group, idx in GROUPS.items():
            for a, b in valid_runs(observed[:, idx].all(1)):
                if b-a < 7:
                    continue
                raw = x[a:b][:, idx].mean(1)
                residuals[group].append((raw-smooth_run(raw))[2:-2])
    threshold, mads, counts = {}, {}, {}
    for group, arrays in residuals.items():
        values = np.concatenate(arrays) if arrays else np.empty(0)
        mad = float(np.median(np.abs(values-np.median(values)))) if len(values) else 0.
        mads[group], counts[group] = mad, len(values)
        threshold[group] = max(float(floor), 6.*mad)
    if not any(counts.values()):
        raise ValueError("fit split has no observed interior samples for noise thresholds")
    return EventThresholds(SCHEMA, float(fps), threshold, int(min_frames), float(floor), mads, counts, len(rows))


def _support(z, peak, low, left, bound):
    # Two neighboring low samples are enough; unlike the old 32-clip teacher,
    # this does not require four-frame flat plateaus on both sides.
    if left:
        for stop in range(peak, bound, -1):
            support = z[stop-1:stop+1]
            if len(support) == 2 and np.all(support <= low + 1e-12):
                return stop
    else:
        for start in range(peak+1, bound):
            support = z[start:start+2]
            if len(support) == 2 and np.all(support <= low + 1e-12):
                return start
    return None


def _run_events(raw, threshold, min_frames):
    z = smooth_run(raw)
    known = np.ones(len(z), dtype=bool)
    known[:2] = False; known[-2:] = False
    candidates, rejected = [], []
    peaks, props = find_peaks(z, prominence=threshold, plateau_size=(1, None))
    for j, peak0 in enumerate(peaks):
        peak = int(peak0); prominence = float(props["prominences"][j])
        low = float(z[peak] - .9*prominence)
        left = _support(z, peak, low, True, int(props["left_bases"][j]))
        right = _support(z, peak, low, False, int(props["right_bases"][j])+1)
        # A raw one-frame spike can become a five-frame smoothed bump. Its
        # original half-prominence width must also support a coarse event.
        raw_high = raw >= (z[peak] - .5*prominence)
        raw_spans = [(a,b) for a,b in valid_runs(raw_high) if a <= peak < b or a <= int(props["left_edges"][j]) < b]
        width = max((b-a for a,b in raw_spans), default=0)
        a = left+1 if left is not None else int(props["left_bases"][j])
        b = right if right is not None else int(props["right_bases"][j])+1
        if left is None or right is None or width < min_frames or b-a < min_frames:
            known[a:b] = False
            rejected.append({"onset": a, "offset": b, "reason": "censored_or_short",
                             "left_censored": left is None, "right_censored": right is None})
            continue
        candidates.append({"onset": a, "offset": b, "peak": peak})
    # Composite overlapping candidates cannot carry a unique scalar phase.
    bad = set()
    for i, one in enumerate(candidates):
        for j, two in enumerate(candidates[i+1:], i+1):
            if max(one["onset"], two["onset"]) < min(one["offset"], two["offset"]):
                bad.update((i,j))
    events = []
    for i, event in enumerate(candidates):
        if i in bad:
            known[event["onset"]:event["offset"]] = False
            rejected.append({**event, "reason": "overlapping_composite"})
        else:
            events.append(event)
    # find_peaks omits boundary extrema. Preserve incomplete edge motion as
    # censored rather than calling it a complete event or reliable inactivity.
    left_stop = min((e["onset"] for e in events), default=len(z))
    right_start = max((e["offset"] for e in events), default=0)
    prefix, suffix = z[:left_stop], z[right_start:]
    if len(prefix) > 1 and z[0]-prefix.min() >= threshold:
        stop = int(np.flatnonzero(prefix <= z[0]-.9*(z[0]-prefix.min()))[0])+1
        known[:stop] = False
        rejected.append({"onset": 0, "offset": stop, "reason": "left_censored"})
    if len(suffix) > 1 and z[-1]-suffix.min() >= threshold:
        start = right_start+int(np.flatnonzero(suffix <= z[-1]-.9*(z[-1]-suffix.min()))[-1])
        known[start:] = False
        rejected.append({"onset": start, "offset": len(z), "reason": "right_censored"})
    safe_events = []
    for event in events:
        a, b = event["onset"], event["offset"]
        if known[a:b].all():
            safe_events.append(event)
        else:
            known[a:b] = False
            rejected.append({**event, "reason": "overlaps_unknown_support"})
    return z, known, safe_events, rejected


def extract_event_labels(motion, valid, thresholds: EventThresholds, motion_mask=None):
    """Return groups, condition[T,12] and per-group known[T,4].

    Channel blocks are 4 active, 4 linear event phase, 4 duration in seconds.
    Unknown positions are zeroed; callers MUST also retain the known mask.
    """
    x, _, observed = validate_motion(motion, valid, motion_mask)
    if thresholds.schema != SCHEMA:
        raise ValueError("unsupported event teacher schema")
    n = len(x); known_all = np.zeros((n,4), bool); condition = np.zeros((n,12), np.float32)
    groups = {}
    for gi, (group, idx) in enumerate(GROUPS.items()):
        active = np.zeros(n, bool); phase = np.zeros(n, np.float32); duration = np.zeros(n, np.float32)
        known = np.zeros(n, bool); signal = np.zeros(n, np.float64); events, rejected = [], []
        for run_id, (a,b) in enumerate(valid_runs(observed[:,idx].all(1))):
            if b-a < 7:
                continue
            z, k, local_events, local_rejected = _run_events(x[a:b][:,idx].mean(1), thresholds.threshold[group], thresholds.min_frames)
            signal[a:b] = z; known[a:b] = k
            for event in local_events:
                start, stop = a+event["onset"], a+event["offset"]
                active[start:stop] = True
                phase[start:stop] = np.linspace(0.,1.,stop-start)
                duration[start:stop] = (stop-start)/thresholds.fps
                events.append({"onset": start, "offset": stop, "peak": a+event["peak"],
                               "duration_frames": stop-start, "duration_s": (stop-start)/thresholds.fps,
                               "run": run_id, "left_censored": False, "right_censored": False})
            rejected.extend({**r, "onset": a+r["onset"], "offset": a+r["offset"], "run": run_id} for r in local_rejected)
        active &= known; phase[~known] = 0; duration[~known] = 0
        condition[:,gi] = active; condition[:,gi+4] = phase; condition[:,gi+8] = duration
        known_all[:,gi] = known
        groups[group] = {"active": active, "phase": phase, "duration": duration, "known": known,
                         "events": events, "rejected": rejected, "signal": signal}
    return {"schema": SCHEMA, "groups": groups, "condition": condition, "known": known_all}


def fit_teacher(train_clips, **kwargs):
    return fit_event_thresholds(train_clips, **kwargs).to_dict()


def extract_schedule(motion9, observed, teacher, motion_mask=None):
    thresholds = EventThresholds(**teacher) if isinstance(teacher, dict) else teacher
    result = extract_event_labels(motion9, observed, thresholds, motion_mask=motion_mask)
    onset = np.zeros_like(result["known"], dtype=bool)
    events = []
    for gi, group in enumerate(GROUPS):
        for event in result["groups"][group]["events"]:
            onset[event["onset"],gi] = True
            events.append({**event, "group": group, "group_index": gi,
                           "start": event["onset"], "end": event["offset"],
                           "duration": event["duration_s"]})
    return {**result, "schedule": result["condition"], "onset": onset, "events": events}


__all__ = ["FPS", "GROUPS", "GROUP_NAMES", "SCHEMA", "EventThresholds", "clip_arrays",
           "fit_event_thresholds", "extract_event_labels", "fit_teacher", "extract_schedule",
           "as_numpy", "validate_motion", "valid_runs"]
