"""Small, NumPy-only scores for freely sampled joint upper-face trajectories.

Inputs are *raw* nine-channel coefficients. ``scales`` must already have been
fitted on the caller's training partition; this module never estimates scales
or changes a prediction using its target. Centering is a scoring diagnostic,
performed separately within each observed run. Gaps retain their native clock.

All samples contribute to the fair energy score (ES), including its unbiased
finite-ensemble spread term. Covariance distances and variograms describe
temporal/joint structure; covariance distance is not a proper scoring rule.
Group scores retain every channel, rather than averaging away left/right
differences. Group order is up, down, squint, wide in ordered upper-nine space.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


SCHEMA = 'joint_motion_scores_v1'
GROUP_NAMES = ('up', 'down', 'squint', 'wide')
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
LAGS = (1, 4, 16, 32)


def _runs(valid):
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _center_runs(value, runs):
    result = np.zeros_like(value, dtype=np.float64)
    for left, right in runs:
        part = value[..., left:right, :]
        # Subtract the first frame before averaging: constant runs are exactly
        # zero, and large run offsets cause less cancellation in the mean.
        shifted = part - part[..., :1, :]
        result[..., left:right, :] = shifted - shifted.mean(axis=-2, keepdims=True)
    return result


def _fair_es(samples, target):
    """RMS Euclidean distance on a whole trajectory, not mean per-frame ES."""
    flat = samples.reshape(len(samples), -1)
    truth = target.reshape(-1)
    target_distance = np.sqrt(np.square(flat - truth).mean(axis=1)).mean()
    spread = sum(float(np.sqrt(np.square(flat[i] - flat[j]).mean()))
                 for i in range(len(flat)) for j in range(i))
    return float(target_distance - spread / (len(flat) * (len(flat) - 1)))


def _group_mean(value):
    return np.asarray([np.asarray(value)[..., list(group)].mean(axis=-1)
                       for group in GROUPS])


def _group_sum(value):
    return np.asarray([np.asarray(value)[..., list(group)].sum(axis=-1)
                       for group in GROUPS])


def _ratio(numerator, denominator, *, unchanged_zero=False):
    result = []
    for num, den in zip(numerator, denominator):
        if den > 0:
            result.append(float(math.sqrt(max(float(num), 0.) / float(den))))
        elif unchanged_zero and num == 0:
            result.append(1.)
        else:
            result.append(None)
    return result


def _speed_stats(squared_speed):
    values = np.asarray(squared_speed, dtype=np.float64).reshape(-1)
    return {'sum_squares': float(values.sum()), 'count': int(values.size),
            'rms': float(np.sqrt(values.mean())) if values.size else None,
            'p95': float(np.quantile(np.sqrt(values), .95)) if values.size else None}


def _check_boundaries(boundaries, count, frames):
    if boundaries is None:
        return None
    if not isinstance(boundaries, (Sequence, np.ndarray)) or len(boundaries) != count:
        raise ValueError('One boundary-index sequence per sample is required')
    result = []
    for row in boundaries:
        if not isinstance(row, (Sequence, np.ndarray)) or isinstance(row, (str, bytes)):
            raise ValueError('Each boundary row must be an integer sequence')
        values = list(row)
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or v < 0 or v > frames for v in values)
                or any(a >= b for a, b in zip(values, values[1:]))):
            raise ValueError('Boundary indices must be unique increasing integers in [0,T]')
        result.append(values)
    return result


def score_clip(samples, target, valid, scales, boundaries=None):
    """Score ``[K,T,9]`` raw samples against one ``[T,9]`` target.

    ``K >= 2`` is required for fair ES. Invalid frames may contain NaN and are
    ignored. A boundary is the *first frame of a new segment*: transition
    ``boundary-1 -> boundary`` is a seam only when both frames are observed.
    Boundaries 0 and T do not denote scored transitions. No boundaries means
    seam/within scores are unavailable, not that the rollout has zero seams.

    Dynamic energy is summed over normalized channels and time, then averaged
    over samples. ``summarize`` pools these numerators/denominators before
    taking RMS ratios. Raw out-of-domain rates use physical [0,1] coefficients.
    Clamp retention compares within-run energy before/after display clipping;
    clipping never modifies the primary prediction or other reported scores.
    Results contain only JSON-compatible finite values, lists, and None.
    """
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid)
    scales = np.asarray(scales, dtype=np.float64)
    if (samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 1
            or samples.shape[-1] != 9 or target.shape != samples.shape[1:]
            or valid.shape != samples.shape[1:2] or valid.dtype != np.bool_
            or not valid.any() or scales.shape != (9,)
            or not np.isfinite(scales).all() or not (scales > 0).all()
            or not np.isfinite(samples[:, valid]).all() or not np.isfinite(target[valid]).all()):
        raise ValueError('Finite observed K>=2 samples [K,T,9], target [T,9], Boolean valid and positive scales[9] required')
    count, frames, _ = samples.shape
    boundary_rows = _check_boundaries(boundaries, count, frames)
    spans = _runs(valid)
    # Invalid values cannot contaminate centering, clipping, or differencing.
    raw_x = np.where(valid[None, :, None], samples, 0.)
    raw_y = np.where(valid[:, None], target, 0.)
    x, y = raw_x / scales, raw_y / scales
    xc, yc = _center_runs(x, spans), _center_runs(y, spans)
    joint_es = {}; group_es = {}
    for name, xx, yy in (('raw', x, y), ('centered', xc, yc)):
        observed_x, observed_y = xx[:, valid], yy[valid]
        joint_es[name] = _fair_es(observed_x, observed_y)
        group_es[name] = [_fair_es(observed_x[..., list(group)], observed_y[..., list(group)])
                          for group in GROUPS]

    lag_scores = {}; variogram_sum = np.zeros(9); variogram_count = 0
    for lag in LAGS:
        channel_sum = np.zeros(9); pairs = 0
        for left, right in spans:
            if right - left <= lag:
                continue
            xd = np.abs(x[:, left + lag:right] - x[:, left:right - lag]) ** .5
            yd = np.abs(y[left + lag:right] - y[left:right - lag]) ** .5
            error = np.square(xd.mean(axis=0) - yd)
            channel_sum += error.sum(axis=0)
            pairs += len(error)
        lag_scores[str(lag)] = {'score': float(channel_sum.mean() / pairs) if pairs else None,
                                'groups': (_group_mean(channel_sum / pairs).tolist() if pairs else None),
                                'pairs': int(pairs)}
        variogram_sum += channel_sum; variogram_count += pairs
    variogram = {'aggregate': float(variogram_sum.mean() / variogram_count) if variogram_count else None,
                 'groups': (_group_mean(variogram_sum / variogram_count).tolist() if variogram_count else None),
                 'pairs': int(variogram_count), 'by_lag': lag_scores}

    observed_xc, observed_yc = xc[:, valid], yc[valid]
    covariance_x = np.einsum('kti,ktj->ij', observed_xc, observed_xc) / (count * valid.sum())
    covariance_y = observed_yc.T @ observed_yc / valid.sum()
    covariance_distance = {'centered': float(np.sqrt(np.square(covariance_x - covariance_y).mean())),
                           'velocity': None}
    transition = valid[:-1] & valid[1:]
    dx, dy = np.diff(x, axis=1), np.diff(y, axis=0)
    if transition.any():
        # Center each contiguous velocity run; no jump across a missing frame
        # enters either the velocity covariance or its moment normalization.
        vx = np.zeros((9, 9)); vy = np.zeros((9, 9)); n = 0
        for left, right in spans:
            if right - left < 2:
                continue
            a, b = dx[:, left:right - 1], dy[left:right - 1]
            a = a - a.mean(axis=1, keepdims=True); b = b - b.mean(axis=0, keepdims=True)
            # ``a`` contains K prediction draws, while ``b`` is the one
            # reference trajectory. Accumulate raw second moments and apply
            # the total transition denominators once after all runs. The
            # previous implementation divided once per run and once again at
            # the end, underestimating velocity covariance on gapped clips.
            vx += np.einsum('kti,ktj->ij', a, a)
            vy += b.T @ b; n += right - left - 1
        if n:
            covariance_distance['velocity'] = float(np.sqrt(np.square(vx / (count * n) - vy / n).mean()))

    prediction_energy = _group_sum(np.square(observed_xc).sum(axis=1).mean(axis=0))
    target_energy = _group_sum(np.square(observed_yc).sum(axis=0))
    clipped = _center_runs(np.clip(raw_x, 0., 1.) / scales, spans)[:, valid]
    clipped_energy = _group_sum(np.square(clipped).sum(axis=1).mean(axis=0))
    oob = (raw_x[:, valid] < 0.) | (raw_x[:, valid] > 1.)
    oob_count = _group_sum(oob.sum(axis=(0, 1)))
    raw_count = np.asarray([int(count * valid.sum() * len(group)) for group in GROUPS])
    pooled = {'prediction_energy': prediction_energy.tolist(), 'target_energy': target_energy.tolist(),
              'clipped_prediction_energy': clipped_energy.tolist(),
              'frame_channel_count': [int(valid.sum() * len(group)) for group in GROUPS],
              'oob_count': oob_count.tolist(), 'raw_count': raw_count.tolist()}

    squared_speed = np.square(dx).mean(axis=-1)
    reference_speed = np.square(dy).mean(axis=-1)
    speed = {'all': _speed_stats(squared_speed[:, transition]),
             'reference_all': _speed_stats(reference_speed[transition]),
             'seam': None, 'within': None, 'reference_seam': None, 'reference_within': None}
    if boundary_rows is not None:
        seam = np.zeros((count, max(frames - 1, 0)), dtype=bool)
        for k, indices in enumerate(boundary_rows):
            for boundary in indices:
                if 0 < boundary < frames and transition[boundary - 1]:
                    seam[k, boundary - 1] = True
        within = transition[None] & ~seam
        repeated_reference = np.broadcast_to(reference_speed, squared_speed.shape)
        speed.update(seam=_speed_stats(squared_speed[seam]), within=_speed_stats(squared_speed[within]),
                     reference_seam=_speed_stats(repeated_reference[seam]),
                     reference_within=_speed_stats(repeated_reference[within]))

    return {'schema': SCHEMA, 'scales': scales.tolist(), 'sample_count': int(count),
            'valid_frames': int(valid.sum()), 'valid_runs': len(spans),
            'joint_fair_es': joint_es, 'group_fair_es': group_es,
            'variogram': variogram, 'covariance_distance': covariance_distance,
            'pooled': pooled, 'rms_ratio': _ratio(prediction_energy, target_energy),
            'raw_oob': (oob_count / raw_count).tolist(),
            'clamp_rms_retention': _ratio(clipped_energy, prediction_energy, unchanged_zero=True),
            'speed': speed}


def _mean_supported(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    result = np.asarray(values, dtype=np.float64).mean(axis=0)
    return result.tolist()


def summarize(rows):
    """Clip-equal ES/descriptors, pooled energy ratios and pooled speed RMS.

    Per-clip speed P95 values are averaged and explicitly named
    ``mean_clip_p95``; no pooled quantile is fabricated from those summaries.
    Scores without temporal support remain None and have support counts.
    Mixing training scales would make both distances and pooled ratios
    incomparable, so differing scales or schemas are rejected.
    """
    rows = list(rows)
    if not rows or any(row.get('schema') != SCHEMA for row in rows):
        raise ValueError('At least one matching joint-motion score row is required')
    scales = np.asarray(rows[0]['scales'], dtype=np.float64)
    if scales.shape != (9,) or any(not np.array_equal(row['scales'], scales) for row in rows):
        raise ValueError('All rows must use identical training-fitted scales')
    pooled = {key: np.asarray([row['pooled'][key] for row in rows]).sum(axis=0)
              for key in ('prediction_energy', 'target_energy', 'clipped_prediction_energy',
                          'frame_channel_count', 'oob_count', 'raw_count')}
    speed = {}
    for kind in ('all', 'reference_all', 'seam', 'within', 'reference_seam', 'reference_within'):
        supported = [row['speed'][kind] for row in rows if row['speed'][kind] is not None]
        if not supported:
            speed[kind] = None
            continue
        ss = sum(row['sum_squares'] for row in supported); n = sum(row['count'] for row in supported)
        speed[kind] = {'sum_squares': float(ss), 'count': int(n),
                       'rms': float(math.sqrt(ss / n)) if n else None,
                       'mean_clip_p95': _mean_supported([row['p95'] for row in supported]),
                       'clips_with_boundaries' if kind not in ('all', 'reference_all') else 'clips': len(supported),
                       'clips_with_transitions': sum(row['count'] > 0 for row in supported)}
    lag_scores = {}
    for lag in LAGS:
        values = [row['variogram']['by_lag'][str(lag)] for row in rows]
        lag_scores[str(lag)] = {'score': _mean_supported([v['score'] for v in values]),
                                'groups': _mean_supported([v['groups'] for v in values]),
                                'pairs': sum(v['pairs'] for v in values),
                                'supported_clips': sum(v['pairs'] > 0 for v in values)}
    return {'schema': SCHEMA, 'clips': len(rows), 'scales': scales.tolist(),
            'aggregation': {'fair_es_and_descriptors': 'equal weight per supported clip',
                            'rms_ratio': 'sqrt(sum(mean_sample_prediction_energy)/sum(target_energy))',
                            'speed_rms': 'sqrt(sum(squared_normalized_9D_RMS_speed)/sum(transitions))',
                            'speed_quantile': 'mean of supported per-clip P95 values; not a pooled P95',
                            'group_order': list(GROUP_NAMES)},
            'joint_fair_es': {kind: _mean_supported([row['joint_fair_es'][kind] for row in rows])
                              for kind in ('raw', 'centered')},
            'group_fair_es': {kind: _mean_supported([row['group_fair_es'][kind] for row in rows])
                              for kind in ('raw', 'centered')},
            'variogram': {'aggregate': _mean_supported([row['variogram']['aggregate'] for row in rows]),
                          'groups': _mean_supported([row['variogram']['groups'] for row in rows]),
                          'pairs': sum(row['variogram']['pairs'] for row in rows),
                          'supported_clips': sum(row['variogram']['pairs'] > 0 for row in rows),
                          'by_lag': lag_scores},
            'covariance_distance': {kind: _mean_supported([row['covariance_distance'][kind] for row in rows])
                                     for kind in ('centered', 'velocity')},
            'pooled': {key: value.tolist() for key, value in pooled.items()},
            'rms_ratio': _ratio(pooled['prediction_energy'], pooled['target_energy']),
            'raw_oob': (pooled['oob_count'] / pooled['raw_count']).tolist(),
            'clamp_rms_retention': _ratio(pooled['clipped_prediction_energy'], pooled['prediction_energy'], unchanged_zero=True),
            'speed': speed}
