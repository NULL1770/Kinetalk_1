"""CPU diagnostics for regional (brow/eye) receiver probes.

The metrics in this module intentionally treat the frame mask as a native
clock.  Invalid/padded frames are never compressed into a contiguous signal,
and therefore cannot create a correlation or a peak across a mask gap.
Thresholds are fitted once from *training target* envelopes and then reused
for every source/oracle intervention.  In particular, no threshold depends
on a prediction's own maximum; a model cannot improve its missing-event score
by merely rescaling its output.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REGION_NAMES = ("brow", "eye")


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    """Convert numpy/torch-like input to a detached CPU numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.shape}")
    return value


def _validate_envelopes(pred: Any, target: Any, valid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = _array(pred, name="pred", ndim=3)
    y = _array(target, name="target", ndim=3)
    m = _array(valid, name="valid", ndim=2).astype(bool, copy=False)
    if p.shape != y.shape or p.shape[:2] != m.shape or p.shape[-1] != 2:
        raise ValueError(f"pred/target must be [B,T,2] and valid [B,T], got {p.shape}, {y.shape}, {m.shape}")
    if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError("observed envelope values must be finite")
    if not m.any(axis=1).all():
        raise ValueError("every clip needs at least one valid frame")
    return p.astype(np.float64, copy=False), y.astype(np.float64, copy=False), m


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    pos = np.flatnonzero(mask).tolist()
    if not pos:
        return []
    result: list[tuple[int, int]] = []
    left = previous = pos[0]
    for index in pos[1:]:
        if index != previous + 1:
            result.append((left, previous + 1))
            left = index
        previous = index
    result.append((left, previous + 1))
    return result


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: Iterable[Any]) -> float | None:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def fit_activity_thresholds(train_target: Any, train_valid: Any, *, quantile: float = .75,
                            fps: float = 25.) -> dict[str, Any]:
    """Fit shared q75 and fixed near-zero thresholds from TRAIN target only."""
    y = _array(train_target, name="train_target", ndim=3)
    m = _array(train_valid, name="train_valid", ndim=2).astype(bool, copy=False)
    if y.shape[:2] != m.shape or y.shape[-1] != 2:
        raise ValueError("train_target must be [B,T,2] with matching train_valid")
    if not (0.0 < quantile < 1.0) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("quantile must be in (0,1) and fps must be positive")
    q75: list[float] = []
    for region in range(2):
        values = y[..., region][m]
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("each training region needs finite observed values")
        q75.append(float(np.quantile(values, quantile)))
    nearzero = [max(1e-12, .1 * value) for value in q75]
    return {"quantile": float(quantile), "q75": q75, "nearzero": nearzero,
            "nearzero_fraction": .1, "fps": float(fps),
            "fit_source": "training_target_only"}


def _thresholds(thresholds: Mapping[str, Any] | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(thresholds, Mapping):
        q = thresholds.get("q75")
        z = thresholds.get("nearzero")
        if q is None:
            raise ValueError("thresholds must contain q75")
        q = np.asarray(q, dtype=np.float64)
        if z is None:
            z = .1 * q
        z = np.asarray(z, dtype=np.float64)
    else:
        q = np.asarray(thresholds, dtype=np.float64)
        z = .1 * q
    if q.shape != (2,) or z.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(z).all():
        raise ValueError("thresholds q75/nearzero must be finite length-two arrays")
    return q, z


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    x = x - x.mean(); y = y - y.mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if not math.isfinite(den) or den <= 1e-12:
        return None
    return _json_number(float(np.dot(x, y) / den))


def _xcorr_lag(x: np.ndarray, y: np.ndarray, *, max_lag: int = 15,
               min_overlap: int = 8) -> tuple[int | None, float | None]:
    """Best zero-mean correlation; positive lag means prediction is delayed."""
    n = len(x)
    best_lag: int | None = None
    best_score: float | None = None
    for lag in range(-min(max_lag, n - 1), min(max_lag, n - 1) + 1):
        if lag >= 0:
            xx, yy = x[lag:], y[:n - lag]
        else:
            xx, yy = x[:n + lag], y[-lag:]
        if len(xx) < min_overlap:
            continue
        score = _pearson(xx, yy)
        if score is None:
            continue
        if (best_score is None or score > best_score + 1e-12
                or (abs(score - best_score) <= 1e-12 and abs(lag) < abs(best_lag))):
            best_lag, best_score = lag, score
    return best_lag, best_score


def _run_xcorr(x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, max_lag: int,
               min_overlap: int) -> tuple[int | None, float | None]:
    candidates: list[tuple[int, float, int]] = []
    for left, right in _runs(mask):
        lag, score = _xcorr_lag(x[left:right], y[left:right], max_lag=max_lag,
                                min_overlap=min_overlap)
        if lag is not None and score is not None:
            candidates.append((lag, score, right - left))
    if not candidates:
        return None, None
    # Average run scores, preserving the native-clock gaps.  The tie rule is
    # deterministic and prefers the smallest absolute lag.
    by_lag: dict[int, list[tuple[float, int]]] = {}
    for lag, score, length in candidates:
        by_lag.setdefault(lag, []).append((score, length))
    scored = [(lag, sum(s * n for s, n in rows) / sum(n for _, n in rows))
              for lag, rows in by_lag.items()]
    scored.sort(key=lambda item: (-item[1], abs(item[0]), item[0]))
    return scored[0]


def _dominant_peak_delay(x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                         threshold: float, nearzero: float) -> tuple[float | None, int]:
    delays: list[int] = []
    for left, right in _runs(mask):
        yy, xx = y[left:right], x[left:right]
        if yy.size == 0 or float(np.max(yy)) < threshold or float(np.max(xx)) < nearzero:
            continue
        # Earliest peak is deterministic when a plateau exists.
        delays.append(abs(int(np.argmax(xx)) - int(np.argmax(yy))))
    return (_mean(delays), len(delays))


def _active_event_coverage(x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                           threshold: float, nearzero: float) -> tuple[float | None, int, int]:
    """Return source-missing *event* coverage, separate from frame miss rate.

    A target event is one contiguous target-active run within one native-clock
    valid run.  It is considered covered when the source reaches the fixed
    train-fitted near-zero threshold at least once in that event.  Thus this
    statistic answers whether an event exists at all; ``missed_active_frame_rate``
    below answers how many of its active frames are too weak.
    """
    total = missed = 0
    active = (y >= threshold) & mask
    for left, right in _runs(mask):
        for event_left, event_right in _runs(active[left:right]):
            event_left += left; event_right += left
            total += 1
            if float(np.max(x[event_left:event_right])) < nearzero:
                missed += 1
    return (None if total == 0 else missed / total, total, missed)


def evaluate_receiver_metrics(pred: Any, target: Any, valid: Any,
                              thresholds: Mapping[str, Any] | Sequence[float], *,
                              fps: float = 25., max_lag: int = 15,
                              min_overlap: int = 8, clip_ids: Sequence[Any] | None = None,
                              sentence_ids: Sequence[Any] | None = None) -> dict[str, Any]:
    """Evaluate one receiver against a target using train-fitted thresholds."""
    p, y, m = _validate_envelopes(pred, target, valid)
    q75, nearzero = _thresholds(thresholds)
    if clip_ids is None: clip_ids = list(range(len(m)))
    if sentence_ids is None: sentence_ids = list(range(len(m)))
    if len(clip_ids) != len(m) or len(sentence_ids) != len(m):
        raise ValueError("clip_ids/sentence_ids must have one entry per clip")
    per_clip: list[dict[str, Any]] = []
    accum: dict[str, dict[str, list[Any]]] = {r: {k: [] for k in (
        "pearson", "xcorr_best_lag_frames", "xcorr_best_corr",
        "dominant_peak_abs_delay_frames", "dominant_peak_abs_delay_seconds",
        "missed_active_frame_rate", "source_missing_coverage", "source_active_coverage")}
        for r in REGION_NAMES}
    for b in range(len(m)):
        row: dict[str, Any] = {"clip_index": b, "clip_id": clip_ids[b],
                               "sentence_id": sentence_ids[b], "regions": {}}
        for region, name in enumerate(REGION_NAMES):
            mask, xx, yy = m[b], p[b, :, region], y[b, :, region]
            xv, yv = xx[mask], yy[mask]
            pearson = _pearson(xv, yv)
            lag, corr = _run_xcorr(xx, yy, mask, max_lag=max_lag, min_overlap=min_overlap)
            delay, peak_count = _dominant_peak_delay(xx, yy, mask, q75[region], nearzero[region])
            event_missing, target_events, missing_events = _active_event_coverage(
                xx, yy, mask, q75[region], nearzero[region])
            active = mask & (yy >= q75[region])
            denominator = int(active.sum())
            missing = int((active & (xx < nearzero[region])).sum())
            valid_count = int(mask.sum())
            source_low = int((mask & (xx < nearzero[region])).sum())
            values = {"pearson": pearson,
                      "xcorr_best_lag_frames": lag,
                      "xcorr_best_lag_seconds": None if lag is None else float(lag / fps),
                      "xcorr_best_corr": corr,
                      "dominant_peak_abs_delay_frames": delay,
                      "dominant_peak_abs_delay_seconds": None if delay is None else float(delay / fps),
                      "dominant_peak_runs": peak_count,
                      "missed_active_frame_rate": None if denominator == 0 else missing / denominator,
                      "source_missing_coverage": event_missing,
                      "source_active_coverage": None if denominator == 0 else 1. - missing / denominator,
                      "target_active_frames": denominator,
                      "source_missing_active_frames": missing,
                      "target_active_events": target_events,
                      "source_missing_events": missing_events}
            row["regions"][name] = values
            for key in accum[name]: accum[name][key].append(values[key])
        per_clip.append(row)
    aggregate: dict[str, Any] = {}
    for name in REGION_NAMES:
        aggregate[name] = {key: _mean(values) for key, values in accum[name].items()}
        aggregate[name]["clips"] = len(per_clip)
    return {"schema": "regional_receiver_metrics_v1", "fps": float(fps),
            "max_lag_frames": int(max_lag), "min_overlap_frames": int(min_overlap),
            "thresholds": {"q75": q75.tolist(), "nearzero": nearzero.tolist(),
                           "fit_source": "training_target_only"},
            "per_clip": per_clip, "aggregate": aggregate}


def paired_clustered_ci(delta: Sequence[float], clusters: Sequence[Any], *,
                        bootstrap: int = 2000, seed: int = 20260921) -> dict[str, Any]:
    """Sentence-cluster bootstrap CI for paired per-clip differences."""
    d = np.asarray(delta, dtype=np.float64)
    c = np.asarray(clusters, dtype=object)
    if d.ndim != 1 or len(d) != len(c) or len(d) == 0 or not np.isfinite(d).all():
        raise ValueError("delta and clusters must be nonempty finite paired vectors")
    labels, inverse = np.unique(c, return_inverse=True)
    cluster_means = np.asarray([d[inverse == i].mean() for i in range(len(labels))])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cluster_means), size=(int(bootstrap), len(cluster_means)))
    samples = cluster_means[indices].mean(axis=1)
    ci = np.quantile(samples, [.025, .975]).tolist()
    return {"delta": float(d.mean()), "sentence_cluster_95ci": [float(ci[0]), float(ci[1])],
            "clips": int(len(d)), "clusters": int(len(labels)),
            "passed": bool(ci[1] < 0)}


def paired_envelope_mse_ci(pred: Any, baseline: Any, target: Any, valid: Any,
                           sentence_ids: Sequence[Any], *, bootstrap: int = 2000,
                           seed: int = 20260921) -> dict[str, Any]:
    """Clustered CI for ``pred MSE - baseline MSE`` per clip."""
    p, b, y, m = _array(pred, name="pred", ndim=3), _array(baseline, name="baseline", ndim=3), _array(target, name="target", ndim=3), _array(valid, name="valid", ndim=2).astype(bool)
    if p.shape != b.shape or p.shape != y.shape or p.shape[:2] != m.shape or p.shape[-1] != 2:
        raise ValueError("envelopes must all be [B,T,2] with valid [B,T]")
    den = m.sum(axis=1).clip(min=1) * 2
    pm = np.where(m[..., None], (p - y) ** 2, 0.).sum(axis=(1, 2)) / den
    bm = np.where(m[..., None], (b - y) ** 2, 0.).sum(axis=(1, 2)) / den
    return paired_clustered_ci(pm - bm, sentence_ids, bootstrap=bootstrap, seed=seed)


__all__ = ["REGION_NAMES", "fit_activity_thresholds", "evaluate_receiver_metrics",
           "paired_clustered_ci", "paired_envelope_mse_ci"]
