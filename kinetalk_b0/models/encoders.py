from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None

class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__(); self.lambd = lambd
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)

from ..utils import masked_mean


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, max_length: int = 4096):
        super().__init__()
        if dim % 2:
            raise ValueError("SinusoidalPosition requires an even feature dimension")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_length, dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds positional limit {self.encoding.shape[1]}")
        return x + self.encoding[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)


class ResidualTCN(nn.Module):
    def __init__(self, dim: int, layers: int = 4, kernel_size: int = 3):
        super().__init__()
        blocks: list[nn.Module] = []
        for index in range(layers):
            dilation = 2**index
            padding = (kernel_size - 1) * dilation // 2
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim * 2, kernel_size, padding=padding, dilation=dilation),
                    nn.GLU(dim=1),
                    nn.GroupNorm(1, dim),
                    nn.SiLU(),
                    nn.Conv1d(dim, dim, 1),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = x.transpose(1, 2)
        for block in self.blocks:
            hidden = hidden + block(hidden)
            if mask is not None:
                hidden = hidden * mask.unsqueeze(1).to(hidden.dtype)
        return hidden.transpose(1, 2)


class TemporalBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        transformer_depth: int = 2,
        heads: int = 8,
        tcn_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.input = nn.Linear(input_dim, hidden_dim)
        self.tcn = ResidualTCN(hidden_dim, tcn_layers)
        self.position = SinusoidalPosition(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_depth)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.tcn(self.input(x), mask)
        hidden = self.transformer(
            self.position(hidden),
            src_key_padding_mask=None if mask is None else ~mask,
        )
        hidden = self.norm(hidden)
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return hidden


def residual_velocity(residual: torch.Tensor) -> torch.Tensor:
    return torch.diff(residual, dim=1, prepend=residual[:, :1])


class AudioContentEncoder(nn.Module):
    """Content-only audio/phonetic encoder used by the frozen Stage-1 teacher."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.backbone = TemporalBackbone(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, content: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(self.backbone(content, mask))


class NeutralArticulationDecoder(nn.Module):
    """Maps frozen content tokens to raw BS with structural non-mouth zeros."""

    def __init__(
        self,
        content_dim: int,
        motion_dim: int,
        hidden_dim: int,
        heads: int,
        output_indices: list[int],
        dropout: float,
        raw_content_dim: int | None = None,
        output_activation: str = "linear",
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.output_indices = list(output_indices)
        self.output_activation = str(output_activation).lower()
        self.input = nn.Linear(content_dim, hidden_dim)
        self.position = SinusoidalPosition(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.output = nn.Linear(hidden_dim, len(self.output_indices))
        # A zero-start, frame-local correction preserves short phonetic events
        # that can be attenuated by the contextual decoder. It cannot change
        # the initial behavior and is trained only through the neutral target.
        raw_dim = int(raw_content_dim or content_dim)
        self.content_skip_norm = nn.LayerNorm(raw_dim)
        self.content_skip = nn.Sequential(
            nn.Conv1d(raw_dim, 256, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(256, len(self.output_indices), kernel_size=1),
        )
        self.content_skip_gate = nn.Parameter(torch.tensor(-2.197225))  # sigmoid ~= 0.10
        nn.init.zeros_(self.content_skip[-1].weight)
        nn.init.zeros_(self.content_skip[-1].bias)

    def forward(
        self,
        content_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        raw_content: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.transformer(
            self.position(self.input(content_tokens)),
            src_key_padding_mask=None if mask is None else ~mask,
        )
        selected = self.output(hidden)
        if raw_content is not None:
            local = self.content_skip(self.content_skip_norm(raw_content).transpose(1, 2)).transpose(1, 2)
            selected = selected + torch.sigmoid(self.content_skip_gate) * local
        # MEAD blendshape coefficients in this experiment are released in the
        # closed interval [0, 1]. Enforce the same contract at the Stage1
        # boundary so an occasional extrapolating logit cannot create a large
        # jaw/mouth spike that later residual stages must cancel.
        if self.output_activation == "sigmoid":
            selected = torch.sigmoid(selected)
        elif self.output_activation != "linear":
            raise ValueError(f"Unsupported Stage1 output activation: {self.output_activation}")
        full = selected.new_zeros(selected.shape[:-1] + (self.motion_dim,))
        full[..., self.output_indices] = selected
        if mask is not None:
            full = full * mask.unsqueeze(-1).to(full.dtype)
        return full


class LowRateAffectField(nn.Module):
    """Generate a temporally coherent affect trajectory from frame features.

    The old local head emitted an unconstrained vector independently at every
    frame.  This module first compresses the sequence to a small number of
    control points, applies a lightweight temporal convolution, then
    interpolates back to the original clock.  The public shape remains
    ``[batch, time, emotion_dim]`` so existing teachers, losses and DiT
    conditioning stay compatible.
    """

    def __init__(self, input_dim: int, output_dim: int, stride: int = 4):
        super().__init__()
        self.stride = max(1, int(stride))
        self.in_projection = nn.Linear(input_dim, output_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1, groups=1),
            nn.SiLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1, groups=1),
            nn.LayerNorm(output_dim),
        )
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, time, features]")
        batch, frames, _ = hidden.shape
        x = self.in_projection(hidden)
        if mask is None:
            valid = torch.ones((batch, frames), device=hidden.device, dtype=torch.bool)
        else:
            valid = mask.to(device=hidden.device, dtype=torch.bool)
            if valid.shape != (batch, frames):
                raise ValueError("mask must match hidden batch/time axes")
        weights = valid.to(x.dtype).unsqueeze(-1)
        # Average-pool to low-rate control points.  Padding is excluded from
        # the average, which keeps variable-length batches well behaved.
        pooled = F.avg_pool1d((x * weights).transpose(1, 2), kernel_size=self.stride, stride=self.stride, ceil_mode=True)
        pooled_w = F.avg_pool1d(weights.transpose(1, 2), kernel_size=self.stride, stride=self.stride, ceil_mode=True).clamp_min(1e-6)
        pooled = pooled / pooled_w
        controls = self.temporal[0](pooled)
        controls = self.temporal[1](controls)
        controls = self.temporal[2](controls).transpose(1, 2)
        controls = self.temporal[3](controls)
        # Interpolate controls back to the native frame clock.
        field = F.interpolate(controls.transpose(1, 2), size=frames, mode="linear", align_corners=True).transpose(1, 2)
        field = self.out_norm(field)
        return field * weights


class ResidualEmotionEncoder(nn.Module):
    """BS residual teacher with local and global affect coordinates."""

    def __init__(
        self,
        motion_dim: int,
        emotion_dim: int,
        hidden_dim: int,
        heads: int,
        num_emotions: int,
        num_intensities: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = TemporalBackbone(motion_dim * 2, hidden_dim, heads=heads, dropout=dropout)
        self.global_head = nn.Sequential(nn.Linear(hidden_dim, emotion_dim), nn.LayerNorm(emotion_dim))
        self.local_head = LowRateAffectField(hidden_dim, emotion_dim, stride=4)
        self.emotion_classifier = nn.Linear(emotion_dim, num_emotions)
        self.intensity_classifier = nn.Linear(emotion_dim, num_intensities)

    def forward(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.backbone(torch.cat([residual, residual_velocity(residual)], dim=-1), mask)
        global_code = self.global_head(masked_mean(hidden, mask))
        local_code = self.local_head(hidden)
        if mask is not None:
            local_code = local_code * mask.unsqueeze(-1).to(local_code.dtype)
        intensity_logits = self.intensity_classifier(global_code)
        return {
            "global": global_code,
            "local": local_code,
            "emotion_logits": self.emotion_classifier(global_code),
            "intensity_logits": intensity_logits,
            "intensity_value": torch.softmax(intensity_logits, dim=-1) @ torch.arange(intensity_logits.shape[-1], device=global_code.device, dtype=global_code.dtype).unsqueeze(-1),
            "hidden": hidden,
        }


class ResidualStyleEncoder(nn.Module):
    """Global speaker/style code from motion residuals.

    The reference audio used by the old implementation contains the teacher
    emotion and affect channels.  Feeding it into the style encoder therefore
    gives the encoder a direct shortcut for the emotion label.  Style is now
    motion-only by default; the legacy audio branch can be enabled explicitly
    for ablation, but is disabled in the production configuration.
    """

    def __init__(
        self,
        motion_dim: int,
        style_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        reference_audio_dim: int = 69,
        use_reference_audio: bool = False,
    ):
        super().__init__()
        self.backbone = TemporalBackbone(motion_dim * 2, hidden_dim, heads=heads, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, style_dim), nn.LayerNorm(style_dim))
        self.use_reference_audio = bool(use_reference_audio)
        if self.use_reference_audio:
            self.audio_hint = nn.Sequential(nn.Linear(reference_audio_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, residual: torch.Tensor, mask: torch.Tensor | None = None, reference_audio: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.backbone(torch.cat([residual, residual_velocity(residual)], dim=-1), mask)
        if self.use_reference_audio:
            if reference_audio is None:
                raise ValueError("Style audio hint is enabled but no reference audio was provided")
            if reference_audio.shape[:2] != residual.shape[:2]:
                raise ValueError("Reference audio and motion must share batch and frame axes")
            hidden = self.fusion(torch.cat([hidden, self.audio_hint(reference_audio)], dim=-1))
        return self.head(masked_mean(hidden, mask))


class AudioEmotionDistributionEncoder(nn.Module):
    """Audio affect field used by Stage 3 (local and global)."""

    def __init__(
        self,
        input_dim: int,
        emotion_dim: int,
        hidden_dim: int,
        heads: int,
        num_emotions: int,
        num_intensities: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = TemporalBackbone(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.global_head = nn.Sequential(nn.Linear(hidden_dim, emotion_dim), nn.LayerNorm(emotion_dim))
        self.local_head = LowRateAffectField(hidden_dim, emotion_dim, stride=4)
        self.emotion_classifier = nn.Linear(emotion_dim, num_emotions)
        self.intensity_classifier = nn.Linear(emotion_dim, num_intensities)

    def forward(self, audio_emotion: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.backbone(audio_emotion, mask)
        global_code = self.global_head(masked_mean(hidden, mask))
        local_code = self.local_head(hidden)
        if mask is not None:
            local_code = local_code * mask.unsqueeze(-1).to(local_code.dtype)
        emotion_logits = self.emotion_classifier(global_code)
        intensity_logits = self.intensity_classifier(global_code)
        return {
            "global": global_code,
            "local": local_code,
            "emotion_logits": emotion_logits,
            "intensity_logits": intensity_logits,
            "intensity_value": torch.softmax(intensity_logits, dim=-1) @ torch.arange(intensity_logits.shape[-1], device=global_code.device, dtype=global_code.dtype).unsqueeze(-1),
            "hidden": hidden,
        }


class IdentityCalibrator(nn.Module):
    """Identity-initialized adapter used only when Stage 4 bridges stages."""

    def __init__(self, dim: int):
        super().__init__()
        self.delta = nn.Linear(dim, dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta(x)
