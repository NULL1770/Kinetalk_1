"""Small frozen-feature temporal predictors for audio dynamic probes.

The emotion2vec cache currently stores its final 768-D frame embedding.  This
module keeps that extractor frozen and learns only a compact temporal head.  A
set of dilated depthwise temporal filters provides local and phrase-scale
context without concatenating several copies of the input (which overfit on
the small pilot split).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _clean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask[..., None], x, torch.zeros_like(x))


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # A depthwise filter keeps the number of trainable parameters small;
        # the pointwise projection mixes emotion2vec channels afterwards.
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=dilation,
                                   dilation=dilation, groups=channels)
        # Normalize channels within each frame. GroupNorm would also include
        # padded time positions, making valid outputs depend on batch padding.
        self.norm = nn.LayerNorm(channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(_clean(x.transpose(1, 2), mask).transpose(1, 2))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.pointwise(F.silu(y))
        y = torch.where(mask[:, None, :], y, torch.zeros_like(y))
        return torch.where(mask[:, None, :], x + y, torch.zeros_like(x))


class FrozenEmotion2VecTemporalPredictor(nn.Module):
    """Predict a scalar dynamic field from frozen emotion2vec frame features.

    ``features`` are [batch, time, dim] and ``mask`` marks valid frames.  The
    output is frame-rate [batch, time, 1]; callers can use the existing
    stride/binning contract afterwards.  No emotion2vec parameters are part
    of this module, so accidental fine-tuning is impossible.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 96,
                 dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or not dilations:
            raise ValueError("input_dim, hidden_dim and dilations must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dilations = tuple(int(d) for d in dilations)
        self.input = nn.Linear(self.input_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([_TemporalBlock(self.hidden_dim, d)
                                     for d in self.dilations])
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(self, features: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("features must have shape [batch,time,input_dim]")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool,
                              device=features.device)
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean [batch,time] tensor")
        x = _clean(F.silu(self.input(_clean(features, mask))), mask).transpose(1, 2)
        for block in self.blocks:
            x = block(x, mask)
        return _clean(self.output(x.transpose(1, 2)), mask)


def multiscale_feature_summary(features: torch.Tensor, mask: torch.Tensor):
    """Return finite diagnostics for a frozen feature cache."""
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("features and mask have incompatible shapes")
    observed = mask[..., None]
    values = torch.where(observed, features, torch.zeros_like(features))
    count = mask.sum().clamp_min(1)
    mean = values.sum((0, 1)) / count
    var = ((values - mean) ** 2 * observed).sum((0, 1)) / count
    return {"frames": int(mask.sum()), "dimension": int(features.shape[-1]),
            "mean_abs": float(mean.abs().mean()),
            "std_mean": float(var.clamp_min(0).sqrt().mean()),
            "temporal_std": float(values[:, 1:].sub(values[:, :-1]).abs().mean())
            if features.shape[1] > 1 else 0.0}
