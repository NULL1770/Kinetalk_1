"""Audio-only low-frequency event conditions for upper-face dynamics.

The frozen 1540-D audio cache already contains four native prosody channels:
``log_f0_unvoiced_zero, log_rms, periodicity, voiced``.  This module exposes
explicit, bounded event directions from those channels rather than asking a
ridge readout to discover onset/hold/release from raw values.  It never reads
motion, labels, text, or visual targets.

The output has ten dimensions, in order::

    energy_level, energy_rise, energy_fall,
    pitch_level, pitch_rise, pitch_fall,
    voicing_level, voicing_onset, voicing_offset, pause_gate

All smoothing and finite differences are performed independently inside each
continuous native-valid run.  A missing interval therefore cannot become a
spurious event.  Values are bounded to [-1, 1] (except pause_gate [0, 1]) so
the caller can concatenate this condition with VA/posterior without another
large-scale feature stream.
"""
from __future__ import annotations

import numpy as np
import torch

PROSODY = slice(1536, 1540)
EVENT_DIM = 10
FPS = 25
SMOOTH_KERNEL = np.asarray([1., 2., 3., 2., 1.], dtype=np.float64) / 9.


def _prosody(features):
    if (not torch.is_tensor(features) or not features.is_floating_point()
            or features.ndim != 2 or features.shape[1] not in (4, 1540)):
        raise ValueError("features must be floating [T,4] or [T,1540]")
    return features if features.shape[1] == 4 else features[:, PROSODY]


def _runs(valid):
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.ndim != 1):
        raise ValueError("valid must be Boolean [T]")
    m = valid.detach().cpu().numpy()
    edge = np.diff(np.r_[False, m, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def _smooth(values):
    """Centered triangular smoothing, clipped to the available run.

    The operation is intentionally non-causal: the existing native audio
    features also use centered windows, and this pilot evaluates alignment and
    event predictability rather than streaming latency.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        left, right = max(0, i - 2), min(n, i + 3)
        kernel_left, kernel_right = 2 - (i - left), 2 + (right - i)
        weights = SMOOTH_KERNEL[kernel_left:kernel_right]
        out[i] = float(np.dot(values[left:right], weights) / weights.sum())
    return out


def _derivative(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values)


def _split_direction(delta, scale):
    z = np.tanh(delta / max(float(scale), 1e-6))
    return np.maximum(z, 0.), np.maximum(-z, 0.)


def derive_event_condition(features, valid, *, return_numpy=False):
    """Derive a bounded [T,10] audio event condition.

    ``valid`` is the native acoustic mask.  Invalid rows are exactly zero.
    Robust center/scale values are fit independently from each valid run's
    audio only; no target or motion array enters this computation.  The event
    channels are designed as a condition input, not as direct eyebrow values.
    """
    local = _prosody(features)
    if local.shape[0] != len(valid):
        raise ValueError("features and valid length differ")
    if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.ndim != 1:
        raise ValueError("valid must be Boolean [T]")
    if not valid.any():
        raise ValueError("valid must contain at least one observed frame")
    if not torch.isfinite(local[valid]).all():
        raise ValueError("observed prosody must be finite")
    dtype, device = local.dtype, local.device
    output = np.zeros((len(valid), EVENT_DIM), dtype=np.float32)
    raw = local.detach().cpu().double().numpy()
    mask = valid.detach().cpu().numpy()
    for left, right in _runs(valid):
        x = raw[left:right]
        # Existing features use zero pitch for unvoiced frames.  For event
        # levels retain only voiced pitch while energy remains defined on all
        # observed frames.
        energy = x[:, 1]
        voiced_strength = np.clip(x[:, 2] * x[:, 3], 0., 1.)
        pitch = x[:, 0]
        voiced = voiced_strength > 1e-4
        e_center = float(np.median(energy))
        e_scale = float(np.quantile(energy, .75) - np.quantile(energy, .25))
        e_scale = max(e_scale / 1.349, .05)
        if voiced.any():
            p_center = float(np.median(pitch[voiced]))
            p_values = pitch.copy()
            p_values[~voiced] = p_center
            p_scale = float(np.quantile(pitch[voiced], .75) - np.quantile(pitch[voiced], .25))
            p_scale = max(p_scale / 1.349, .05)
        else:
            p_center, p_values, p_scale = 0., np.zeros_like(pitch), 1.

        e_level = np.tanh((_smooth(energy) - e_center) / e_scale)
        e_delta = _derivative(_smooth(energy))
        e_rise, e_fall = _split_direction(e_delta, e_scale / FPS)

        p_level = np.tanh((_smooth(p_values) - p_center) / p_scale) * voiced_strength
        p_delta = _derivative(_smooth(p_values))
        p_rise, p_fall = _split_direction(p_delta, p_scale / FPS)
        p_rise *= voiced_strength
        p_fall *= voiced_strength

        v_level = voiced_strength
        v_delta = _derivative(_smooth(voiced_strength))
        v_onset, v_offset = _split_direction(v_delta, .10)
        pause = 1. - v_level
        output[left:right] = np.stack((e_level, e_rise, e_fall, p_level, p_rise,
                                       p_fall, v_level, v_onset, v_offset, pause), axis=1)
    result = torch.from_numpy(output).to(device=device, dtype=dtype)
    result = torch.where(valid[:, None], result, torch.zeros_like(result))
    if not torch.isfinite(result).all():
        raise FloatingPointError("event condition became nonfinite")
    return result.detach().cpu().numpy() if return_numpy else result


__all__ = ["PROSODY", "EVENT_DIM", "derive_event_condition"]
