"""Small, auditable MEAD-ARKit52 adaptations of VOCA and EmoTalk.

The original releases use incompatible vertex/3D-ETF protocols.  These
architectures retain the useful temporal design cues while sharing the paper
protocol: locked 1540-D audio cache, independent neutral 52-D identity anchor,
and 52-D blendshape output.
"""
from __future__ import annotations

import torch
from torch import nn


class VocaARKit(nn.Module):
    """VOCA-style local temporal audio convolution with an identity residual."""
    def __init__(self, audio_dim: int = 1540, hidden: int = 256, motion_dim: int = 52):
        super().__init__()
        self.audio = nn.Sequential(
            nn.Conv1d(audio_dim, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.anchor = nn.Linear(motion_dim, hidden, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, motion_dim))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, audio: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        h = self.audio(audio.transpose(1, 2)).transpose(1, 2)
        h = h + self.anchor(anchor)[:, None]
        return anchor[:, None] + self.head(h)


class EmoTalkARKit(nn.Module):
    """EmoTalk-style content/emotion branches with a shared temporal decoder."""
    def __init__(self, audio_dim: int = 1540, hidden: int = 256, motion_dim: int = 52,
                 emotions: int = 8, layers: int = 2):
        super().__init__()
        self.content = nn.Sequential(nn.Linear(audio_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.emotion = nn.Sequential(nn.Linear(audio_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=2 * hidden,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.anchor = nn.Linear(motion_dim, hidden, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, motion_dim))
        self.emotion_head = nn.Linear(hidden, emotions)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, audio: torch.Tensor, anchor: torch.Tensor,
                valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.content(audio) + self.emotion(audio) + self.anchor(anchor)[:, None]
        if valid is not None:
            h = self.decoder(h, src_key_padding_mask=~valid)
        else:
            h = self.decoder(h)
        pooled = (torch.where(valid[..., None], h, 0.).sum(1) / valid.sum(1).clamp_min(1)[:, None]
                  if valid is not None else h.mean(1))
        return anchor[:, None] + self.head(h), self.emotion_head(pooled)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    mask = valid[..., None]
    if channel_mask is not None:
        mask = mask & channel_mask[:, None] if channel_mask.ndim == 2 else mask & channel_mask[None, None]
    mask = mask.expand_as(pred)
    return ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1)


def masked_velocity_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                        channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    pair = valid[:, 1:] & valid[:, :-1]
    return masked_mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], pair, channel_mask)
