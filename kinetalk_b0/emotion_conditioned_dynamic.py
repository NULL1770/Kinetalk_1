"""A compact scalar audio field conditioned on the existing global emotion.

This probe contains only a trainable temporal head over frozen audio features.
It does not consume motion, identity, or ground-truth dynamic labels. The caller
chooses global probabilities (audio predictions at inference, labels only for
an explicit oracle diagnostic), bins/centers the scalar field, and maps it into
the expression direction space. Neutral gating belongs to that direction
space, not to this predictor.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _masked(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # Multiplication would let NaNs in padded features enter temporal filters.
    return torch.where(valid[..., None], x, 0)


class _TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(width, width, 3, padding=dilation,
                                  dilation=dilation, groups=width)
        self.mix = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Per-frame normalization and repeated masking make predictions
        # independent of how much padding another clip adds to the batch.
        y = _masked(self.norm(x), valid).transpose(1, 2)
        y = self.temporal(y).transpose(1, 2)
        return _masked(x + self.mix(F.silu(y)), valid)


class EmotionConditionedDynamicPredictor(nn.Module):
    """Predict raw [B,T,1] changes from audio [B,T,D] and emotion [B,E].

    A single shared hidden state receives one static FiLM modulation. Its
    zero initialization makes every emotion condition match the same initial
    audio-only function; centered probabilities retain a shared baseline for
    the uniform condition. The final linear readout deliberately has no tanh:
    a static logit bias must not saturate and erase clip-centered dynamics.
    """

    def __init__(self, input_dim: int = 768, num_emotions: int = 8,
                 hidden_dim: int = 32):
        super().__init__()
        if min(input_dim, num_emotions, hidden_dim) < 1:
            raise ValueError("all dimensions must be positive")
        self.input_dim = input_dim
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([_TemporalBlock(hidden_dim, d) for d in (1, 4)])
        self.output = nn.Linear(hidden_dim, 1)
        self.condition = nn.Linear(num_emotions, 2 * hidden_dim, bias=False)
        nn.init.zeros_(self.condition.weight)

    def forward(self, features: torch.Tensor, probabilities: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("features must be [batch,time,input_dim]")
        if features.shape[1] < 1:
            raise ValueError("features must contain at least one time position")
        if probabilities.shape != (features.shape[0], self.num_emotions):
            raise ValueError("probabilities must be [batch,num_emotions]")
        if not probabilities.is_floating_point():
            raise ValueError("probabilities must be floating point")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch,time]")
        x = self.input_norm(self.input(_masked(features, valid)))
        x = _masked(F.silu(x), valid)
        gain, shift = self.condition(probabilities - 1 / self.num_emotions).chunk(2, -1)
        x = _masked(x * (1 + gain[:, None]) + shift[:, None], valid)
        for block in self.blocks:
            x = block(x, valid)
        return _masked(self.output(x), valid)
