"""A small, mean-preserving regional gain controller for upper-face motion.

The controller deliberately does not predict the nine upper-face coefficients.
An already trained motion prior supplies their temporal shape and direction;
this module only changes the magnitude of the brow and expression-eye regions.
The gain is evaluated at the native frame clock and is bounded.  Composition is
mean-preserving, so the frozen baseline mean and all non-upper channels remain
protected.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .slow_state_affect import UPPER_INDICES


# UPPER_INDICES is ordered as five brow controls followed by four eye controls.
REGIONAL_GROUPS = ((0, 1, 2, 3, 4), (5, 6, 7, 8))


def _validate_valid(valid: torch.Tensor, *, batch: int, frames: int, device: torch.device) -> None:
    if (not torch.is_tensor(valid) or valid.dtype is not torch.bool
            or valid.shape != (batch, frames) or valid.device != device
            or not valid.any(1).all()):
        raise ValueError("valid must be nonempty Boolean [B,T] with an observed frame")


def _validate_upper(upper: torch.Tensor, valid: torch.Tensor, name: str) -> None:
    if (not torch.is_tensor(upper) or upper.ndim != 3 or upper.shape[-1] != 9
            or not upper.is_floating_point() or upper.shape[:2] != valid.shape
            or upper.device != valid.device):
        raise ValueError(f"{name} must be floating [B,T,9] on the valid mask device")
    if not torch.isfinite(upper[valid]).all():
        raise ValueError(f"Observed {name} must be finite")


def _contiguous_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open true runs without compressing the native clock."""
    positions = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append((start, previous + 1))
            start = position
        previous = position
    runs.append((start, previous + 1))
    return runs


def _box_smooth(value: torch.Tensor, window: int) -> torch.Tensor:
    """Smooth one contiguous sequence, renormalizing at run boundaries."""
    # ``value`` is [L,G], and the caller has already split mask gaps.
    kernel = value.new_ones(len(REGIONAL_GROUPS), 1, window)
    pad = window // 2
    x = value.transpose(0, 1).unsqueeze(0)
    support = value.new_ones(1, 1, value.shape[0])
    total = F.conv1d(x, kernel, padding=pad, groups=len(REGIONAL_GROUPS))
    count = F.conv1d(support, value.new_ones(1, 1, window), padding=pad)
    return (total / count.clamp_min(1.)).squeeze(0).transpose(0, 1)


def regional_envelope(upper: torch.Tensor, valid: torch.Tensor, *, window: int = 13,
                      channel_scales: torch.Tensor | None = None) -> torch.Tensor:
    """Return a smooth brow/eye RMS activity field ``[B,T,2]``.

    ``upper`` is a nine-channel residual or centered upper-face trajectory in
    ``UPPER_INDICES`` order.  RMS is computed within each region per frame and
    then box-smoothed independently inside every contiguous valid run.  Thus a
    mask gap cannot make motion on one side of the gap influence the other side.
    Invalid positions are zero placeholders and are never used in reductions.
    ``channel_scales`` is optional train-fitted positive normalization for the
    nine channels; it is useful when raw ARKit coefficient units differ.
    """
    if (not isinstance(window, int) or window < 1 or window % 2 == 0):
        raise ValueError("window must be a positive odd integer")
    _validate_valid(valid, batch=upper.shape[0] if torch.is_tensor(upper) and upper.ndim >= 1 else 0,
                    frames=upper.shape[1] if torch.is_tensor(upper) and upper.ndim >= 2 else 0,
                    device=upper.device if torch.is_tensor(upper) else valid.device)
    _validate_upper(upper, valid, "upper")
    if channel_scales is not None:
        if (not torch.is_tensor(channel_scales) or channel_scales.shape != (9,)
                or not channel_scales.is_floating_point()
                or channel_scales.device != upper.device
                or channel_scales.dtype != upper.dtype
                or not torch.isfinite(channel_scales).all()
                or (channel_scales <= 0).any()):
            raise ValueError("channel_scales must be finite positive floating [9]")
        value = upper / channel_scales
    else:
        value = upper

    rows: list[torch.Tensor] = []
    for batch_index in range(len(upper)):
        row = value.new_zeros(value.shape[1], 2)
        for left, right in _contiguous_runs(valid[batch_index]):
            segment = value[batch_index, left:right]
            # Per-frame regional RMS, followed by a boundary-renormalized box
            # smoother. vector_norm uses a finite zero subgradient at exact
            # zero, whereas sqrt(mean(square(x))) has an undefined 0/0
            # autograd path. The forward RMS values are otherwise identical.
            energy = torch.stack((
                torch.linalg.vector_norm(segment[..., list(REGIONAL_GROUPS[0])], dim=-1)
                / math.sqrt(len(REGIONAL_GROUPS[0])),
                torch.linalg.vector_norm(segment[..., list(REGIONAL_GROUPS[1])], dim=-1)
                / math.sqrt(len(REGIONAL_GROUPS[1])),
            ), -1)
            smooth = _box_smooth(energy, window)
            indices = torch.arange(left, right, device=upper.device)
            row = row.index_copy(0, indices, smooth)
        rows.append(row)
    return torch.stack(rows, 0)


def _center_valid(upper: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Remove each clip/channel mean without compressing native mask gaps."""
    mask = valid[..., None]
    clean = torch.where(mask, upper, 0.)
    count = valid.sum(1, keepdim=True).to(upper.dtype).clamp_min(1.)
    mean = clean.sum(1, keepdim=True) / count[..., None]
    return torch.where(mask, upper - mean, 0.)


def apply_regional_gain(prior_upper: torch.Tensor, gain: torch.Tensor,
                        valid: torch.Tensor) -> torch.Tensor:
    """Scale regional temporal deviations while retaining each channel mean.

    The prior is the source of motion direction and shape.  Gain is one value
    per region and frame.  The returned trajectory has the same native-clock
    mask and the same per-channel valid-window mean as ``prior_upper``.
    """
    _validate_valid(valid, batch=prior_upper.shape[0] if torch.is_tensor(prior_upper) and prior_upper.ndim >= 1 else 0,
                    frames=prior_upper.shape[1] if torch.is_tensor(prior_upper) and prior_upper.ndim >= 2 else 0,
                    device=prior_upper.device if torch.is_tensor(prior_upper) else valid.device)
    _validate_upper(prior_upper, valid, "prior_upper")
    if (not torch.is_tensor(gain) or gain.shape != (*valid.shape, 2)
            or not gain.is_floating_point() or gain.device != prior_upper.device
            or gain.dtype != prior_upper.dtype):
        raise ValueError("gain must be floating [B,T,2] matching prior_upper")
    if not torch.isfinite(gain[valid]).all() or (gain[valid] <= 0).any():
        raise ValueError("Observed gain must be finite and positive")

    # Padding values are intentionally outside the observation contract.  Make
    # them inert before the mean-preserving reduction: multiplying a zero
    # centered value by NaN/Inf would otherwise reintroduce non-finite values
    # and contaminate the per-clip ``scaled_mean`` even though the frame is
    # masked.  The adapter already emits one on padding, but this public
    # function also accepts independently supplied gains.
    safe_gain = torch.where(valid[..., None], gain, torch.ones_like(gain))

    mask = valid[..., None]
    clean = torch.where(mask, prior_upper, 0.)
    count = valid.sum(1, keepdim=True).to(prior_upper.dtype).clamp_min(1.)
    mean = clean.sum(1, keepdim=True) / count[..., None]
    centered = torch.where(mask, prior_upper - mean, 0.)
    regional_gain = torch.cat((
        safe_gain[..., 0:1].expand(-1, -1, 5),
        safe_gain[..., 1:2].expand(-1, -1, 4),
    ), -1)
    # Express the update as a zero-centred *delta* around the original prior.
    # This is algebraically equivalent to scaling around ``mean`` but has an
    # important numerical/optimization property: at gain==1 the delta is
    # exactly zero, so the frozen-prior control is bitwise identical while
    # gradients through ``gain`` remain available for an initialized adapter.
    delta = centered * (regional_gain - 1.)
    delta_mean = delta.sum(1, keepdim=True) / count[..., None]
    output = torch.where(mask, prior_upper + delta - delta_mean, 0.)
    return torch.where(mask, output, 0.)


class RegionalIntensityGainAdapter(nn.Module):
    """Predict bounded regional gains from a target intensity field.

    The final head is zero initialized, so a freshly constructed adapter is an
    exact identity (gain one) and cannot damage a validated prior.  Inputs
    include source/prior envelope, target (audio or oracle) envelope, and their
    clipped log ratio.  The target envelope is a deployment condition; no
    target motion is read by this module.
    """

    def __init__(self, *, hidden: int = 32, max_gain: float = 3.0,
                 epsilon: float = 1e-3, ratio_clip: float = 4.0):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (hidden,)):
            raise ValueError("hidden must be a positive integer")
        if (not math.isfinite(max_gain) or max_gain <= 1.
                or not math.isfinite(epsilon) or epsilon <= 0
                or not math.isfinite(ratio_clip) or ratio_clip <= 0):
            raise ValueError("max_gain > 1, epsilon > 0 and ratio_clip > 0 are required")
        self.hidden, self.max_gain = hidden, float(max_gain)
        self.epsilon, self.ratio_clip = float(epsilon), float(ratio_clip)
        self.input = nn.Linear(6, hidden)
        self.output = nn.Linear(hidden, 2)
        # Zero output gives gain=1 exactly while retaining trainable upstream
        # features; the first optimization steps identify useful conditioning.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, prior_upper: torch.Tensor, target_envelope: torch.Tensor,
                valid: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_upper(prior_upper, valid, "prior_upper")
        if (not torch.is_tensor(target_envelope) or target_envelope.shape != (*valid.shape, 2)
                or not target_envelope.is_floating_point()
                or target_envelope.device != prior_upper.device
                or target_envelope.dtype != prior_upper.dtype):
            raise ValueError("target_envelope must be floating [B,T,2] matching prior_upper")
        if (not torch.isfinite(target_envelope[valid]).all()
                or (target_envelope[valid] < 0).any()):
            raise ValueError("Observed target_envelope must be finite and nonnegative")
        # Intensity is a temporal-motion magnitude, not a static expression
        # offset.  Remove the frozen prior's per-clip channel mean before
        # extracting the source envelope; composition restores that mean.
        source = regional_envelope(_center_valid(prior_upper, valid), valid)
        mask = valid[..., None]
        target = torch.where(mask, target_envelope, 0.)
        source_clean = torch.where(mask, source, 0.)
        log_ratio = (target + self.epsilon).log() - (source_clean + self.epsilon).log()
        log_ratio = log_ratio.clamp(-self.ratio_clip, self.ratio_clip)
        features = torch.cat((source_clean, target, log_ratio), -1)
        hidden = F.silu(self.input(features))
        raw = self.output(hidden)
        max_log = math.log(self.max_gain)
        log_gain = max_log * torch.tanh(raw)
        gain = torch.exp(log_gain)
        gain = torch.where(mask, gain, torch.ones_like(gain))
        dynamic_upper = apply_regional_gain(prior_upper, gain, valid)
        return {'source_envelope': source, 'target_envelope': target,
                'log_ratio': torch.where(mask, log_ratio, 0.),
                'log_gain': torch.where(mask, log_gain, 0.),
                'gain': gain, 'dynamic_upper': dynamic_upper}


def compose_regional_gain(baseline: torch.Tensor, prior_upper: torch.Tensor,
                          gain: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Apply a regional gain and compose it with a protected full-face base.

    ``baseline`` supplies the preserved per-channel upper mean and all other
    43 coefficients.  ``prior_upper`` supplies the stochastic temporal shape.
    When it is ``baseline[..., UPPER_INDICES]`` and gain is one, the operation
    preserves the baseline. A different supplied prior is still composed at
    gain one: identity gain means no gain modification, not ignoring the prior.
    """
    if (not torch.is_tensor(baseline) or baseline.ndim != 3 or baseline.shape[-1] != 52
            or not baseline.is_floating_point() or baseline.shape[:2] != valid.shape
            or baseline.device != valid.device):
        raise ValueError("baseline must be floating [B,T,52] matching valid")
    _validate_valid(valid, batch=baseline.shape[0], frames=baseline.shape[1], device=baseline.device)
    _validate_upper(prior_upper, valid, "prior_upper")
    if prior_upper.dtype != baseline.dtype or prior_upper.device != baseline.device:
        raise ValueError("prior_upper must match baseline dtype/device")
    if not torch.isfinite(baseline[valid]).all():
        raise ValueError("Observed baseline must be finite")
    dynamic_upper = apply_regional_gain(prior_upper, gain, valid)
    indices = list(UPPER_INDICES)
    base_upper = baseline[..., indices]
    clean_base = torch.where(valid[..., None], base_upper, 0.)
    # base + center(dynamic - base) == mean(base) + center(dynamic).
    # This delta form is exact at identity without branching on tensor values
    # or manufacturing a straight-through derivative.
    update = _center_valid(dynamic_upper - clean_base, valid)
    output = baseline.clone()
    output[..., indices] = torch.where(valid[..., None], base_upper + update, base_upper)
    return output


__all__ = [
    'REGIONAL_GROUPS', 'regional_envelope', 'apply_regional_gain',
    'RegionalIntensityGainAdapter', 'compose_regional_gain',
]

