"""Reference-conditioned bounded upper-face centers and static recentering.

The head takes only deployment conditions. A caller supplies training-only
normalization, loss targets and an explicit old-generator fallback; a zero
head predicts the clipped neutral anchor, not the old generator's center.
The composer does not invent temporal events. It translates the valid-frame
mean and, where bounds require it, scales each channel by one constant.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _static(value, rows, width, name, reference=None):
    if (not torch.is_tensor(value) or value.ndim != 2
            or value.shape != (rows, width)
            or value.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite float32/float64 [B,{width}]')
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f'{name} must share the reference dtype and device')


class BoundedExpressionCenter(nn.Module):
    """Predict nine clip-level centers relative to an independent anchor.

    ``sigmoid(logit(clamp(anchor, eps, 1-eps)) + M*tanh(raw/M))``
    bounds both the center and the anchor-relative log-odds shift. The default
    shift limit M=6 permits stronger departures than the earlier proposed
    conservative M=2 calibrator. Neither labels nor query motion enter forward.
    Clipping defines a finite anchor logit; output is never post-hoc clipped.
    """

    def __init__(self, global_dim=64, identity_dim=128, hidden=64,
                 max_logit_delta=6., anchor_eps=.01):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (global_dim, identity_dim, hidden)):
            raise ValueError('global_dim, identity_dim and hidden must be positive integers')
        if (type(max_logit_delta) not in (int, float)
                or not math.isfinite(max_logit_delta) or max_logit_delta <= 0):
            raise ValueError('max_logit_delta must be positive and finite')
        if (type(anchor_eps) not in (int, float)
                or not math.isfinite(anchor_eps) or not 0 < anchor_eps < .5):
            raise ValueError('anchor_eps must be finite and strictly between 0 and .5')
        self.global_dim = global_dim
        self.identity_dim = identity_dim
        self.hidden = hidden
        self.max_logit_delta = float(max_logit_delta)
        self.anchor_eps = float(anchor_eps)
        self.input = nn.Linear(global_dim + identity_dim + 9, hidden)
        self.activation = nn.SiLU()
        self.head = nn.Linear(hidden, 9)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def export_config(self):
        return {'global_dim': self.global_dim, 'identity_dim': self.identity_dim,
                'hidden': self.hidden, 'max_logit_delta': self.max_logit_delta,
                'anchor_eps': self.anchor_eps}

    def forward(self, global_code, identity_code, anchor9):
        if not torch.is_tensor(global_code) or global_code.ndim != 2 or len(global_code) < 1:
            raise ValueError('global_code must have a nonempty batch dimension')
        rows = len(global_code)
        _static(global_code, rows, self.global_dim, 'global_code')
        _static(identity_code, rows, self.identity_dim, 'identity_code', global_code)
        _static(anchor9, rows, 9, 'anchor9', global_code)
        if ((anchor9 < 0) | (anchor9 > 1)).any():
            raise ValueError('anchor9 must lie in [0,1]')
        if self.input.weight.dtype != global_code.dtype or self.input.weight.device != global_code.device:
            raise ValueError('Model parameters must share the input dtype and device')
        context = torch.cat((global_code, identity_code, anchor9), dim=-1)
        raw = self.head(self.activation(self.input(context)))
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite center-head logits')
        delta = self.max_logit_delta * torch.tanh(raw / self.max_logit_delta)
        logits = torch.logit(anchor9.clamp(self.anchor_eps, 1-self.anchor_eps))
        return torch.sigmoid(logits + delta)


def compose_bounded_center(base_upper, center, valid):
    """Replace a valid-frame mean, retaining one scale per upper channel.

    Args: base_upper float32/64 [B,T,9] with observed values in [0,1]; center
    finite [B,9] in [0,1]; valid Boolean [B,T]. Every row must be observed.
    Invalid base values may be NaN and are copied bit-for-bit to the output.

    Returns: upper [B,T,9], scale [B,9], original_mean [B,9]. The residual
    mean is taken over all valid frames of a clip; gaps are excluded and no
    temporal filtering joins them. Each channel uses min(1, c/negative_peak,
    (1-c)/positive_peak). Nonzero limiting ratios include a four-epsilon
    inward numerical margin, avoiding an output clamp or roundoff overshoot.
    Peaks below sqrt(finfo.tiny) use that numerical denominator floor; this
    conservatively suppresses subnormal motion and keeps derivatives finite.
    Constant channels use scale=1; endpoint centers can collapse a moving
    channel to zero amplitude. This preserves timing and waveform up to
    the reported scale, not temporal variance or covariance magnitude.
    """
    if (not torch.is_tensor(base_upper) or base_upper.ndim != 3
            or base_upper.shape[-1] != 9 or min(base_upper.shape[:2]) < 1
            or base_upper.dtype not in (torch.float32, torch.float64)):
        raise ValueError('base_upper must be nonempty float32/float64 [B,T,9]')
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != base_upper.shape[:2] or valid.device != base_upper.device):
        raise ValueError('valid must be Boolean [B,T] on the base_upper device')
    if not valid.any(1).all():
        raise ValueError('Every clip must have at least one valid frame')
    observed = base_upper[valid]
    if not torch.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
        raise ValueError('Observed base_upper must be finite and lie in [0,1]')
    _static(center, len(base_upper), 9, 'center', base_upper)
    if ((center < 0) | (center > 1)).any():
        raise ValueError('center must lie in [0,1]')

    # Shift by an observed value before averaging: constant channels become
    # exactly zero, and NaN padding never participates in differentiable math.
    mask = valid[..., None]
    first_index = valid.long().argmax(1)[:, None, None].expand(-1, 1, 9)
    first = base_upper.gather(1, first_index)
    clean = torch.where(mask, base_upper, first)
    relative = clean - first
    offset = relative.sum(1) / valid.sum(1, keepdim=True).to(base_upper.dtype)
    original_mean = first[:, 0] + offset
    residual = torch.where(mask, relative-offset[:, None], 0.)
    positive = residual.clamp_min(0.).amax(1)
    negative = (-residual).clamp_min(0.).amax(1)

    def room_ratio(room, peak):
        active = peak > 0
        # Squared denominators in division backward must remain representable.
        # This only affects amplitudes <1.1e-19 (float32) or <1.5e-154 (float64).
        floor = math.sqrt(torch.finfo(base_upper.dtype).tiny)
        safe_peak = torch.where(active, peak.clamp_min(floor), torch.ones_like(peak))
        ratio = room / safe_peak
        # No division by zero, including branches discarded by where: this
        # matters for finite gradients on flat channels and center endpoints.
        inward = 1. - 4 * torch.finfo(base_upper.dtype).eps
        return torch.where(active, ratio * inward, torch.ones_like(ratio))

    scale = torch.minimum(torch.ones_like(center), torch.minimum(
        room_ratio(center, negative), room_ratio(1-center, positive)))
    generated = center[:, None] + scale[:, None] * residual
    return {'upper': torch.where(mask, generated, base_upper),
            'scale': scale, 'original_mean': original_mean}


__all__ = ['BoundedExpressionCenter', 'compose_bounded_center']
