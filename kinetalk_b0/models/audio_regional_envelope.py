"""Native-rate audio prediction of regional upper-face activity.

The module predicts only two non-negative activity envelopes (brow and eye)
from audio.  It does not predict ARKit coefficients.  A frozen motion prior
supplies coefficient directions and temporal detail; :func:`compose_prior_with_envelope`
only scales that prior around a supplied per-channel mean.

This file is intentionally an independent path.  It is not imported by the
legacy Stage4 renderer or by the signed-state runners.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .slow_state_affect import UPPER_INDICES


_REGION_WIDTHS = (5, 4)


def _check_valid(features: torch.Tensor, valid: torch.Tensor) -> None:
    if (not torch.is_tensor(features) or features.ndim != 3
            or not features.is_floating_point() or min(features.shape[:2]) < 1):
        raise ValueError("features must be nonempty floating [B,T,F]")
    if (not torch.is_tensor(valid) or valid.dtype is not torch.bool
            or valid.shape != features.shape[:2] or valid.device != features.device
            or not valid.any(1).all()):
        raise ValueError("valid must be nonempty Boolean [B,T]")
    if not torch.isfinite(features[valid]).all():
        raise ValueError("observed audio features must be finite")


def _valid_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[..., None]
    clean = torch.where(mask, value, 0.)
    count = valid.sum(1, keepdim=True).to(value.dtype).clamp_min(1.)
    return clean.sum(1, keepdim=True) / count[..., None]


def _expand_gain(gain: torch.Tensor, batch: int, frames: int, device: torch.device,
                 dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(gain) or not gain.is_floating_point() or gain.device != device:
        raise ValueError("gain must be floating on prior device")
    if gain.shape == (batch, 2):
        gain = gain[:, None].expand(batch, frames, 2)
    elif gain.shape == (frames, 2):
        gain = gain[None].expand(batch, frames, 2)
    elif gain.shape != (batch, frames, 2):
        raise ValueError("gain must have shape [B,T,2], [B,2], or [T,2]")
    if gain.dtype != dtype:
        gain = gain.to(dtype=dtype)
    if not torch.isfinite(gain).all() or (gain < 0).any():
        raise ValueError("gain must be finite and nonnegative")
    return gain


def _upper_mean(mean: torch.Tensor, *, batch: int, frames: int,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(mean) or not mean.is_floating_point() or mean.device != device:
        raise ValueError("mean must be floating on prior device")
    if mean.shape == (batch, 9):
        out = mean[:, None].expand(batch, frames, 9)
    elif mean.shape == (batch, frames, 9):
        out = mean
    elif mean.shape == (batch, 52):
        out = mean[:, None, list(UPPER_INDICES)].expand(batch, frames, 9)
    elif mean.shape == (batch, frames, 52):
        out = mean[..., list(UPPER_INDICES)]
    elif mean.shape == (9,):
        out = mean[None, None].expand(batch, frames, 9)
    else:
        raise ValueError("mean must have shape [9], [B,9], [B,T,9], [B,52], or [B,T,52]")
    if out.dtype != dtype:
        out = out.to(dtype=dtype)
    if not torch.isfinite(out).all():
        raise ValueError("mean must be finite")
    return out


def compose_prior_with_envelope(
    prior_upper: torch.Tensor,
    gain: torch.Tensor,
    mean: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale frozen prior deviations while preserving a supplied upper mean.

    ``prior_upper`` may be a nine-channel upper trajectory in ``UPPER_INDICES``
    order or a complete 52-channel face trajectory.  ``gain[...,0]`` scales
    the five brow channels and ``gain[...,1]`` scales the four eye channels.
    The source trajectory mean is removed before scaling, then ``mean`` is
    restored.  Consequently each observed channel has exactly the requested
    mean for either a static, zero, reversed, or audio-predicted gain.  For a
    52-channel input, all 43 non-upper channels are copied bit-for-bit.
    """
    if (not torch.is_tensor(prior_upper) or prior_upper.ndim != 3
            or prior_upper.shape[-1] not in (9, 52)
            or not prior_upper.is_floating_point() or min(prior_upper.shape[:2]) < 1):
        raise ValueError("prior_upper must be floating [B,T,9] or [B,T,52]")
    batch, frames = prior_upper.shape[:2]
    if valid is None:
        valid = torch.ones(batch, frames, device=prior_upper.device, dtype=torch.bool)
    if (not torch.is_tensor(valid) or valid.dtype is not torch.bool
            or valid.shape != (batch, frames) or valid.device != prior_upper.device
            or not valid.any(1).all()):
        raise ValueError("valid must be nonempty Boolean [B,T]")
    if not torch.isfinite(prior_upper[valid]).all():
        raise ValueError("observed prior values must be finite")
    gains = _expand_gain(gain, batch, frames, prior_upper.device, prior_upper.dtype)
    target_mean = _upper_mean(mean, batch=batch, frames=frames,
                              device=prior_upper.device, dtype=prior_upper.dtype)

    indices = list(UPPER_INDICES)
    selected = prior_upper[..., indices] if prior_upper.shape[-1] == 52 else prior_upper
    mask = valid[..., None]
    source_mean = _valid_mean(selected, valid)
    centered = torch.where(mask, selected - source_mean, 0.)
    region_gain = torch.cat((
        gains[..., 0:1].expand(-1, -1, _REGION_WIDTHS[0]),
        gains[..., 1:2].expand(-1, -1, _REGION_WIDTHS[1]),
    ), dim=-1)
    scaled = torch.where(mask, centered * region_gain, 0.)
    # A time-varying gain has a nonzero weighted mean in general.  Recenter
    # the scaled deviation so every observed channel keeps exactly target_mean.
    scaled_mean = _valid_mean(scaled, valid)
    generated = target_mean + scaled - scaled_mean
    generated = torch.where(mask, generated, selected)

    if prior_upper.shape[-1] == 9:
        return generated
    result = prior_upper.clone()
    result[..., indices] = generated
    return result


@torch.jit.ignore

def intervene_envelope(envelope: torch.Tensor, valid: torch.Tensor,
                       mode: Literal["audio", "zero", "static", "reverse"] = "audio") -> torch.Tensor:
    """Create deterministic controls for envelope ablations on the native clock."""
    if (not torch.is_tensor(envelope) or envelope.ndim != 3 or envelope.shape[-1] != 2
            or not envelope.is_floating_point()):
        raise ValueError("envelope must be floating [B,T,2]")
    if (not torch.is_tensor(valid) or valid.dtype is not torch.bool
            or valid.shape != envelope.shape[:2] or valid.device != envelope.device):
        raise ValueError("valid must be Boolean [B,T]")
    if not torch.isfinite(envelope[valid]).all() or (envelope[valid] < 0).any():
        raise ValueError("observed envelope must be finite and nonnegative")
    out = envelope.clone()
    if mode == "zero":
        out.zero_()
    elif mode == "static":
        out = _valid_mean(envelope, valid).expand_as(envelope).clone()
    elif mode == "reverse":
        out.zero_()
        for row in range(len(envelope)):
            ids = valid[row].nonzero(as_tuple=True)[0]
            out[row, ids] = envelope[row, ids.flip(0)]
    elif mode != "audio":
        raise ValueError(mode)
    return torch.where(valid[..., None], out, torch.zeros_like(out))


class AudioRegionalEnvelope(nn.Module):
    """Predict native-rate nonnegative brow/eye activity from audio features.

    ``stride`` controls an optional feature pooling stage and is restricted to
    small values (1 or 4); unlike the signed-state path it never uses a
    stride-16 spline target.  No clip-level centering is applied, so absolute
    prosodic energy remains available to the student.
    """

    def __init__(self, feature_mean: torch.Tensor, feature_std: torch.Tensor,
                 *, hidden: int = 64, stride: int = 1, max_envelope: float = 4.0):
        super().__init__()
        if (not torch.is_tensor(feature_mean) or feature_mean.ndim != 1
                or not torch.is_tensor(feature_std) or feature_std.shape != feature_mean.shape
                or not feature_mean.is_floating_point() or not feature_std.is_floating_point()
                or feature_mean.numel() < 1 or not torch.isfinite(feature_mean).all()
                or not torch.isfinite(feature_std).all() or (feature_std <= 0).any()):
            raise ValueError("finite feature statistics with positive std are required")
        if type(hidden) is not int or hidden < 4:
            raise ValueError("hidden must be an integer >= 4")
        if stride not in (1, 4):
            raise ValueError("stride must be 1 or 4; stride-16 is intentionally unsupported")
        if not torch.isfinite(torch.tensor(float(max_envelope))) or max_envelope <= 0:
            raise ValueError("max_envelope must be positive and finite")
        self.stride = stride
        self.hidden = hidden
        self.max_envelope = float(max_envelope)
        self.register_buffer("feature_mean", feature_mean.detach().float().clone())
        self.register_buffer("feature_std", feature_std.detach().float().clone())
        self.input = nn.Linear(feature_mean.numel(), hidden)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)

    def _temporal_runs(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolutions independently inside each valid run."""
        result = hidden.new_zeros(hidden.shape)
        for row in range(hidden.shape[0]):
            ids = torch.nonzero(valid[row], as_tuple=False).flatten().tolist()
            if not ids:
                continue
            left = previous = ids[0]
            runs = []
            for position in ids[1:]:
                if position != previous + 1:
                    runs.append((left, previous + 1))
                    left = position
                previous = position
            runs.append((left, previous + 1))
            for left, right in runs:
                segment = hidden[row, left:right].transpose(0, 1).unsqueeze(0)
                segment = self.temporal(segment).squeeze(0).transpose(0, 1)
                result[row, left:right] = segment
        return result

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        _check_valid(features, valid)
        if features.shape[-1] != self.feature_mean.numel():
            raise ValueError("feature width differs from fitted statistics")
        x = torch.where(valid[..., None],
                        (features - self.feature_mean) / self.feature_std, 0.)
        h = F.silu(self.input(x))
        if self.stride == 4 and h.shape[1] >= 4:
            # Pool independently inside each valid run, then interpolate each
            # run back to its native frame clock. This avoids crossing mask
            # gaps while retaining a short (4-frame) receptive field.
            pooled_h = h.new_zeros(h.shape)
            for row in range(h.shape[0]):
                ids = torch.nonzero(valid[row], as_tuple=False).flatten().tolist()
                if not ids:
                    continue
                left = previous = ids[0]
                runs = []
                for position in ids[1:]:
                    if position != previous + 1:
                        runs.append((left, previous + 1))
                        left = position
                    previous = position
                runs.append((left, previous + 1))
                for left, right in runs:
                    segment = h[row, left:right].transpose(0, 1).unsqueeze(0)
                    pooled = F.avg_pool1d(segment, kernel_size=4, stride=4, ceil_mode=True)
                    restored = F.interpolate(pooled, size=right - left,
                                             mode="linear", align_corners=False)
                    pooled_h[row, left:right] = restored.squeeze(0).transpose(0, 1)
            h = pooled_h
        h = self._temporal_runs(h, valid)
        h = self.norm(h)
        raw = self.head(F.silu(h))
        envelope = self.max_envelope * torch.sigmoid(raw)
        return torch.where(valid[..., None], envelope, torch.zeros_like(envelope))


__all__ = ["AudioRegionalEnvelope", "compose_prior_with_envelope", "intervene_envelope"]
