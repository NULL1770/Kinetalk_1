"""Small content-conditioned dynamic probe over two frozen audio caches.

This module has no renderer, identity, global-affect or pretrained-encoder
parameters. Content modulates one hidden representation; it is not a separate
motion channel. The FiLM projection starts at zero, so its initial function is
exactly the matched audio-only head.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _masked(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid[..., None], x, 0)


class _LocalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(width, width, 3, padding=dilation,
                                  dilation=dilation, groups=width)
        self.mix = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Normalize channels per observed frame, never over padded time.
        local = _masked(self.norm(x), valid).transpose(1, 2)
        local = self.temporal(local).transpose(1, 2)
        return _masked(x + self.mix(F.silu(local)), valid)


class ContentGatedDynamicPredictor(nn.Module):
    """Frame field from emotion features, optionally modulated by content.

    ``mode`` selects a matched audio-only control, FiLM, or content-only control.
    All common layers are created before the FiLM branch, which makes their
    initialization identical under the same random seed.
    """

    def __init__(self, audio_dim: int = 768, content_dim: int = 768,
                 hidden_dim: int = 64, bottleneck_dim: int = 16,
                 output_dim: int = 1, mode: str = "film"):
        super().__init__()
        if mode not in ("audio_only", "film", "content_only"):
            raise ValueError("mode must be audio_only, film or content_only")
        if min(audio_dim, content_dim, hidden_dim, bottleneck_dim, output_dim) < 1:
            raise ValueError("all dimensions must be positive")
        self.mode = mode
        self.audio_dim, self.content_dim = audio_dim, content_dim
        self.hidden_dim, self.output_dim = hidden_dim, output_dim
        input_dim = content_dim if mode == "content_only" else audio_dim
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([_LocalBlock(hidden_dim, d) for d in (1, 4)])
        self.output = nn.Linear(hidden_dim, output_dim)
        if mode == "film":
            self.condition = nn.Sequential(
                nn.Linear(content_dim, bottleneck_dim), nn.SiLU(),
                nn.Linear(bottleneck_dim, 2 * hidden_dim),
            )
            nn.init.zeros_(self.condition[-1].weight)
            nn.init.zeros_(self.condition[-1].bias)
        else:
            self.condition = None

    def forward(self, audio: torch.Tensor, content: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[-1] != self.audio_dim:
            raise ValueError("audio must be [batch,time,audio_dim]")
        if content.ndim != 3 or content.shape != (*audio.shape[:2], self.content_dim):
            raise ValueError("content must share the audio batch/time clock")
        if valid.shape != audio.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch,time]")
        source = content if self.mode == "content_only" else audio
        hidden = _masked(F.silu(self.input(_masked(source, valid))), valid)
        if self.condition is not None:
            gain, shift = self.condition(_masked(content, valid)).chunk(2, -1)
            hidden = hidden * (1 + 0.5 * gain.tanh()) + 0.5 * shift.tanh()
            hidden = _masked(hidden, valid)
        for block in self.blocks:
            hidden = block(hidden, valid)
        return _masked(self.output(hidden), valid)
