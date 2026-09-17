"""Small, single-field dynamic-expression probes.

The probe is deliberately external to the renderer: it turns the existing
low-rate controls into one interpretable upper-face intensity coordinate.  It
adds no regional routing or extra generation channel and is therefore useful
for testing whether an audio student predicts a meaningful dynamic direction.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def upper_indices(motion_dim: int, *, brow_start: int = 41, brow_end: int = 46) -> torch.Tensor:
    """Return the compact upper-face slice used by the MEAD pilot."""
    values = [*range(min(14, motion_dim)), *range(min(brow_start, motion_dim), min(brow_end, motion_dim))]
    return torch.tensor(sorted(set(values)), dtype=torch.long)


def upper_l1_field(residual: torch.Tensor, valid: torch.Tensor, stride: int,
                   channel_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a centered, stride-binned upper-face L1 field."""
    if residual.ndim != 3 or valid.shape != residual.shape[:2] or stride < 1:
        raise ValueError("residual [B,T,C], valid [B,T], and positive stride are required")
    if channel_mask is not None and channel_mask.shape != (residual.shape[0], residual.shape[-1]):
        raise ValueError("channel_mask must be [B,C]")
    selected = upper_indices(residual.shape[-1]).to(residual.device)
    clean = residual.index_select(-1, selected)
    observed = valid[..., None]
    if channel_mask is not None:
        observed = observed & channel_mask.index_select(-1, selected)[:, None]
    clean = torch.where(observed, clean, 0)
    extra = (-clean.shape[1]) % stride
    padded = F.pad(clean, (0, 0, 0, extra))
    mask = F.pad(valid, (0, extra), value=False)
    weight = mask.reshape(mask.shape[0], -1, stride).sum(-1).to(clean.dtype)
    channel_weight = observed.to(clean.dtype).sum(-1)
    channel_weight = F.pad(channel_weight, (0, extra)).reshape(clean.shape[0], -1, stride).sum(-1)
    raw = padded.abs().sum(-1).reshape(clean.shape[0], -1, stride).sum(-1)
    raw = raw / channel_weight.clamp_min(1)
    centered = raw - (raw * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp_min(1)
    return torch.where(weight.bool(), centered, torch.zeros_like(centered)).unsqueeze(-1), weight


def fit_control_probe(controls: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                      ridge: float = 1e-3) -> torch.Tensor:
    """Fit one linear control direction to a scalar field on valid bins."""
    if controls.ndim != 3 or target.shape[:2] != controls.shape[:2] or target.shape[-1] != 1:
        raise ValueError("controls [B,K,R] and scalar target [B,K,1] are required")
    if weight.shape != controls.shape[:2] or ridge <= 0:
        raise ValueError("weight shape and positive ridge are required")
    mask = weight.reshape(-1) > 0
    x, y = controls.reshape(-1, controls.shape[-1])[mask], target.reshape(-1)[mask]
    w = weight.reshape(-1)[mask].to(controls.dtype)
    gram = (x * w[:, None]).T @ x + ridge * torch.eye(controls.shape[-1], device=x.device, dtype=x.dtype)
    rhs = (x * w[:, None]).T @ y
    return torch.linalg.solve(gram, rhs)


def projected_energy(controls: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    if controls.ndim != 3 or probe.shape != (controls.shape[-1],):
        raise ValueError("probe must match the control rank")
    return controls @ probe
