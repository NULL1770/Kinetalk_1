"""Observed ARKit mouth metrics and fail-closed stage protection checks.

These are paired coefficient diagnostics, not an audio/visual sync score. They
never clamp, realign, smooth, or alter predictions. Temporal centering is per
clip and channel; adjacent-frame errors do not bridge missing observations.
"""
from __future__ import annotations

import math

import numpy as np
import torch


MOUTH = tuple(range(14, 41))
SCHEMA = 'mouth_protection_v1'
THRESHOLDS = {'max_correlation_drop': .01,
              'max_velocity_ratio': 1.05,
              'max_neutral_raw_mse_ratio': 1.03}


def _inputs(prediction, target, valid, channel_mask):
    if (not torch.is_tensor(prediction) or not torch.is_tensor(target)
            or prediction.ndim != 3 or prediction.shape[-1] != 52
            or prediction.shape != target.shape
            or not prediction.is_floating_point() or not target.is_floating_point()):
        raise ValueError('prediction and target must be floating matching [B,T,52]')
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != prediction.shape[:2]):
        raise ValueError('valid must be Boolean [B,T]')
    if not torch.is_tensor(channel_mask) or channel_mask.dtype != torch.bool:
        raise ValueError('channel_mask must be Boolean [B,52] or [B,T,52]')
    if channel_mask.shape == (prediction.shape[0], 52):
        channel_mask = channel_mask[:, None].expand_as(prediction)
    elif channel_mask.shape != prediction.shape:
        raise ValueError('channel_mask must be Boolean [B,52] or [B,T,52]')
    x = prediction.detach()[..., MOUTH].double().cpu()
    y = target.detach()[..., MOUTH].double().cpu()
    observed = valid.detach().cpu()[..., None] & channel_mask.detach().cpu()[..., MOUTH]
    return x, y, observed


def _range(value):
    if not value.numel():
        return {'min': None, 'max': None, 'p01': None, 'p99': None,
                'outside_fraction': None, 'max_overshoot': None}
    # torch.quantile rejects arrays above 2**24 elements, which a complete
    # training corpus may exceed. NumPy has no such quantile-size limit.
    p01, p99 = np.quantile(value.numpy(), [.01, .99])
    return {'min': float(value.min()), 'max': float(value.max()),
            'p01': float(p01), 'p99': float(p99),
            'outside_fraction': float(((value < 0) | (value > 1)).double().mean()),
            'max_overshoot': float(torch.maximum((-value).clamp_min(0), (value - 1).clamp_min(0)).max())}


def _center(value, observed):
    # Subtract one observed value before averaging. This both avoids a large
    # static offset in the sum and makes constant decimal streams exactly
    # zero-variance rather than treating rounding residue as motion.
    first = observed.long().argmax(1, keepdim=True)
    relative = torch.where(observed, value - value.gather(1, first), 0.)
    mean = relative.sum(1, keepdim=True) / observed.sum(1, keepdim=True).clamp_min(1)
    return torch.where(observed, relative - mean, 0.)


def mouth_metrics(prediction, target, valid, channel_mask):
    """Return JSON-safe metrics over channels 14:41 using explicit observations.

    Unknown payloads may contain NaN/Inf. Observed nonfinite values invalidate
    the complete metric result instead of silently changing scoring support.
    Undefined temporal statistics are ``None`` and ``valid`` is false.
    """
    x, y, observed = _inputs(prediction, target, valid, channel_mask)
    pair = observed[:, 1:] & observed[:, :-1]
    count, pair_count = int(observed.sum()), int(pair.sum())
    result = {'schema': SCHEMA, 'mouth_channels': list(MOUTH), 'valid': False, 'reasons': [],
              'clips': len(x), 'observed_clips': int(observed.flatten(1).any(1).sum()),
              'observed_frames': int(observed.any(-1).sum()), 'observed_values': count,
              'adjacent_pairs': pair_count, 'per_channel_observations': observed.sum((0, 1)).tolist(),
              'raw_mse': None, 'centered_mse': None, 'centered_correlation': None,
              'centered_r2': None, 'rms_ratio': None, 'frame_displacement_mse': None,
              'velocity_mse': None, 'outside_fraction': None,
              'prediction_range': _range(torch.empty(0)), 'target_range': _range(torch.empty(0))}
    if count == 0:
        result['reasons'].append('no_observed_mouth_values')
        return result
    bad_x = int((~torch.isfinite(x[observed])).sum())
    bad_y = int((~torch.isfinite(y[observed])).sum())
    result['nonfinite_observed_prediction'] = bad_x
    result['nonfinite_observed_target'] = bad_y
    if bad_x or bad_y:
        result['reasons'].append('nonfinite_observed_mouth_values')
        return result
    x = torch.where(observed, x, 0.)
    y = torch.where(observed, y, 0.)
    xc, yc = _center(x, observed), _center(y, observed)
    error = (xc - yc).square().sum()
    x_energy, y_energy = xc.square().sum(), yc.square().sum()
    result.update(raw_mse=float((x[observed] - y[observed]).square().mean()),
                  centered_mse=float(error / count),
                  prediction_range=_range(x[observed]), target_range=_range(y[observed]))
    result['outside_fraction'] = result['prediction_range']['outside_fraction']
    if y_energy > 0:
        result['centered_r2'] = float(1 - error / y_energy)
        result['rms_ratio'] = float((x_energy / y_energy).sqrt())
    else:
        result['reasons'].append('undefined_target_temporal_variance')
    if x_energy > 0 and y_energy > 0:
        result['centered_correlation'] = float((xc * yc).sum() / (x_energy * y_energy).sqrt())
    elif x_energy <= 0:
        result['reasons'].append('undefined_prediction_temporal_variance')
    if pair_count:
        displacement_error = torch.diff(x, dim=1) - torch.diff(y, dim=1)
        result['frame_displacement_mse'] = float(displacement_error[pair].square().mean())
        result['velocity_mse'] = result['frame_displacement_mse']
    else:
        result['reasons'].append('no_observed_adjacent_mouth_pairs')
    # Guard overflow too: finite inputs are necessary but not sufficient for
    # finite squared statistics. JSON must not contain NaN or Infinity.
    def finite_statistics(values, prefix=''):
        for key, value in list(values.items()):
            path = prefix + key
            if isinstance(value, dict):
                finite_statistics(value, path + '/')
            elif isinstance(value, float) and not math.isfinite(value):
                values[key] = None
                result['reasons'].append('nonfinite_statistic_' + path)
    finite_statistics(result)
    result['valid'] = not result['reasons']
    return result


def protection_report(prediction, base, target, valid, channel_mask, emotion):
    """Check paired mouth preservation overall, neutral (0), and nonneutral.

    Every group must have defined finite temporal metrics; absent groups fail
    closed. Passing proves preservation relative to ``base``, not that the
    base has acceptable lip sync or that emotional expression is correct.
    """
    _inputs(base, target, valid, channel_mask)
    _inputs(prediction, target, valid, channel_mask)
    if (not torch.is_tensor(emotion) or emotion.ndim != 1 or len(emotion) != len(target)
            or emotion.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or (emotion < 0).any()):
        raise ValueError('emotion must be nonnegative integer labels [B], with neutral=0')
    # Evaluation tensors may reside on distinct devices; group indexing below
    # is deliberately CPU-only and cannot affect a training computation graph.
    values = [v.detach().cpu() for v in (prediction, base, target, valid, channel_mask)]
    emotion = emotion.detach().cpu()
    groups = {}; failures = []
    selections = {'overall': torch.ones(len(emotion), dtype=torch.bool),
                  'neutral': emotion == 0, 'nonneutral': emotion != 0}
    for name, selected in selections.items():
        p, b, y, m, cm = [v[selected] for v in values]
        predicted = mouth_metrics(p, y, m, cm)
        baseline = mouth_metrics(b, y, m, cm)
        defined = predicted['valid'] and baseline['valid']
        corr_bound = baseline['centered_correlation']
        velocity_bound = baseline['frame_displacement_mse']
        checks = {'finite_defined_metrics': bool(defined),
                  'correlation_preserved': bool(defined and predicted['centered_correlation'] >= corr_bound - THRESHOLDS['max_correlation_drop']),
                  'velocity_preserved': bool(defined and predicted['frame_displacement_mse'] <= velocity_bound * THRESHOLDS['max_velocity_ratio'])}
        if name == 'neutral':
            checks['raw_mse_preserved'] = bool(defined and predicted['raw_mse'] <= baseline['raw_mse'] * THRESHOLDS['max_neutral_raw_mse_ratio'])
        group = {'prediction': predicted, 'base': baseline, 'checks': checks, 'passed': all(checks.values())}
        groups[name] = group
        failures.extend(name + '/' + check for check, passed in checks.items() if not passed)
    return {'schema': SCHEMA, 'passed': not failures, 'thresholds': dict(THRESHOLDS),
            'groups': groups, 'failures': failures,
            'scope': 'Paired raw ARKit coefficient preservation only; base quality and AV synchronization require separate evaluation.'}
