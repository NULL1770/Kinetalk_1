"""FaceFormer core adapted to 52-channel MEAD-ARKit coefficients.

The autoregressive decoder, periodic positional encoding and ALiBi-style
temporal bias follow the upstream FaceFormer implementation.  This adapter
replaces the upstream wav2vec and mesh-template inputs with the locked native
1540-D audio cache and a neutral 52-D enrollment anchor.  The anchor is the
only identity condition, so a held-out speaker can be evaluated with neutral
enrollment clips without exposing its query motion during training.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def _slopes(n: int) -> list[float]:
    def power_of_two(k: int) -> list[float]:
        start = 2 ** (-2 ** -(math.log2(k) - 3))
        ratio = start
        return [start * ratio**i for i in range(k)]

    if math.log2(n).is_integer():
        return power_of_two(n)
    closest = 2 ** math.floor(math.log2(n))
    return power_of_two(closest) + _slopes(2 * closest)[0::2][: n - closest]


def init_biased_mask(n_head: int, max_seq_len: int, period: int) -> torch.Tensor:
    """Return the causal periodic ALiBi mask used by FaceFormer."""
    slopes = torch.tensor(_slopes(n_head), dtype=torch.float32)
    bias = torch.arange(0, max_seq_len, step=period).unsqueeze(1).repeat(1, period)
    bias = bias.reshape(-1)[:max_seq_len] // period
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, : i + 1] = bias[-(i + 1) :]
    alibi = slopes[:, None, None] * alibi[None]
    causal = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    causal = causal.float().masked_fill(~causal, float("-inf")).masked_fill(causal == 1, 0.0)
    return causal[None] + alibi


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, period: int = 30, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(period, d_model)
        position = torch.arange(period, dtype=torch.float32)[:, None]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[None].repeat(1, (max_seq_len // period) + 1, 1)
        self.register_buffer("pe", pe, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1]])


def aligned_memory_mask(length: int, memory_length: int, device: torch.device) -> torch.Tensor:
    """FaceFormer/vocaset one-to-one audio-to-frame attention mask.

    A frame attends to its aligned cached-audio frame.  For unequal lengths,
    nearest-neighbour index mapping keeps the mask deterministic while the
    prepared MEAD data normally has equal frame clocks.
    """
    if length <= 0 or memory_length <= 0:
        return torch.zeros((length, memory_length), dtype=torch.bool, device=device)
    indices = torch.floor(torch.arange(length, device=device) * memory_length / length).long()
    mask = torch.ones((length, memory_length), dtype=torch.bool, device=device)
    mask[torch.arange(length, device=device), indices.clamp_max(memory_length - 1)] = False
    return mask


@dataclass(frozen=True)
class FaceFormerARKitConfig:
    audio_dim: int = 1540
    motion_dim: int = 52
    feature_dim: int = 128
    period: int = 30
    max_seq_len: int = 2048
    heads: int = 4
    dropout: float = 0.1


class FaceFormerARKit(nn.Module):
    """Official FaceFormer-style autoregressive attention in ARKit space."""

    def __init__(self, config: FaceFormerARKitConfig = FaceFormerARKitConfig()):
        super().__init__()
        self.config = config
        self.audio_feature_map = nn.Linear(config.audio_dim, config.feature_dim)
        self.motion_map = nn.Linear(config.motion_dim, config.feature_dim)
        self.anchor_map = nn.Linear(config.motion_dim, config.feature_dim, bias=False)
        self.ppe = PeriodicPositionalEncoding(config.feature_dim, config.period,
                                              config.max_seq_len, config.dropout)
        self.register_buffer("biased_mask", init_biased_mask(config.heads, config.max_seq_len,
                                                               config.period), persistent=False)
        layer = nn.TransformerDecoderLayer(
            d_model=config.feature_dim,
            nhead=config.heads,
            dim_feedforward=2 * config.feature_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.motion_map_r = nn.Linear(config.feature_dim, config.motion_dim)
        # Upstream FaceFormer starts from a neutral output and learns motion
        # as an autoregressive residual around the template/anchor.
        nn.init.zeros_(self.motion_map_r.weight)
        nn.init.zeros_(self.motion_map_r.bias)

    def _masks(self, length: int, memory_length: int, batch_size: int,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if length > self.config.max_seq_len or memory_length > self.config.max_seq_len:
            raise ValueError("Sequence exceeds configured max_seq_len")
        # PyTorch's TransformerDecoder accepts a 3-D self-attention mask in
        # [batch * heads, T, T] form.  The upstream FaceFormer code targeted
        # an older release that accepted [heads, T, T], so expand explicitly.
        tgt_mask = self.biased_mask[:, :length, :length].to(device)
        tgt_mask = tgt_mask.repeat(batch_size, 1, 1)
        memory_mask = aligned_memory_mask(length, memory_length, device)
        return tgt_mask, memory_mask

    def _encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        if audio_features.ndim != 3 or audio_features.shape[-1] != self.config.audio_dim:
            raise ValueError(f"audio_features must be [B,T,{self.config.audio_dim}]")
        return self.audio_feature_map(audio_features)

    def _decode(self, audio: torch.Tensor, anchor: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Decode a teacher-forced sequence of residual coefficients."""
        memory = self._encode_audio(audio)
        style = self.anchor_map(anchor).unsqueeze(1)
        decoder_input = self.motion_map(previous) + style
        decoder_input = self.ppe(decoder_input)
        tgt_mask, memory_mask = self._masks(previous.shape[1], memory.shape[1], previous.shape[0], previous.device)
        hidden = self.transformer_decoder(decoder_input, memory,
                                          tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.motion_map_r(hidden)

    def forward(self, audio_features: torch.Tensor, anchor: torch.Tensor,
                target: torch.Tensor, valid: torch.Tensor | None = None,
                *, teacher_forcing: bool = True) -> torch.Tensor:
        if anchor.ndim != 2 or anchor.shape[-1] != self.config.motion_dim:
            raise ValueError("anchor must be [B,52]")
        if target.ndim != 3 or target.shape[0] != anchor.shape[0] or target.shape[-1] != self.config.motion_dim:
            raise ValueError("target must be [B,T,52]")
        if audio_features.shape[:2] != target.shape[:2]:
            raise ValueError("audio and target frame clocks must have equal shape")
        if valid is None:
            valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        if valid.shape != target.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,T] mask")
        residual = target - anchor[:, None]
        if teacher_forcing:
            zero = torch.zeros_like(residual[:, :1])
            previous = torch.cat((zero, residual[:, :-1]), dim=1)
            prediction = self._decode(audio_features, anchor, previous) + anchor[:, None]
        else:
            prediction = self.predict(audio_features, anchor)
        return prediction

    def predict(self, audio_features: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        if audio_features.ndim != 3 or anchor.ndim != 2:
            raise ValueError("audio_features and anchor have invalid rank")
        memory = self._encode_audio(audio_features)
        bsz, length = audio_features.shape[:2]
        style = self.anchor_map(anchor).unsqueeze(1)
        generated = torch.zeros((bsz, 0, self.config.motion_dim), dtype=audio_features.dtype,
                                device=audio_features.device)
        for step in range(length):
            # Keep the neutral token and native aligned audio clock, matching
            # teacher forcing at every prefix length.
            zero = torch.zeros((bsz, 1, self.config.motion_dim), dtype=audio_features.dtype,
                               device=audio_features.device)
            previous = torch.cat((zero, generated - anchor[:, None]), dim=1)
            decoder_input = self.ppe(self.motion_map(previous) + style)
            prefix_memory = memory[:, :step+1]
            tgt_mask, memory_mask = self._masks(decoder_input.shape[1], prefix_memory.shape[1], bsz, audio_features.device)
            hidden = self.transformer_decoder(decoder_input, prefix_memory,
                                              tgt_mask=tgt_mask, memory_mask=memory_mask)
            delta = self.motion_map_r(hidden[:, -1:])
            generated = torch.cat((generated, delta + anchor[:, None]), dim=1)
        return generated


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute coefficient MSE without counting padding or unavailable channels."""
    mask = valid[..., None]
    if channel_mask is not None:
        if channel_mask.ndim == 2:
            mask = mask & channel_mask[:, None]
        elif channel_mask.ndim == 1:
            mask = mask & channel_mask[None, None]
        else:
            raise ValueError("channel_mask must be [B,52] or [52]")
    if not bool(mask.any()):
        raise ValueError("No valid coefficient entries")
    mask = mask.expand_as(prediction)
    return ((prediction - target).square() * mask).sum() / mask.sum()
