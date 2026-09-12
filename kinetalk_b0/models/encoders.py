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


class MonotonicLocalCrossAttention(nn.Module):
    """Local HuBERT attention with an explicitly bounded clip lag.

    Inputs are native feature windows ordered around each motion frame.  The
    attention support is local by construction and its centre is shifted by a
    learned lag in ``[-max_lag, max_lag]`` motion frames.  Since windows are
    indexed by the monotonic motion clock, this cannot reorder phonetic frames.
    """

    def __init__(self, dim: int, context: int = 9, max_lag: float = 2.0):
        super().__init__()
        if context < 3 or context % 2 == 0:
            raise ValueError("local attention context must be an odd value >= 3")
        self.context = int(context)
        self.max_lag = float(max_lag)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.lag_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1))
        self.register_buffer("relative", torch.arange(context, dtype=torch.float32) - context // 2, persistent=False)

    def forward(self, windows: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if windows.ndim != 4 or windows.shape[2] != self.context:
            raise ValueError(f"Expected [B,T,{self.context},D] native windows")
        centre = windows[:, :, self.context // 2]
        lag = self.max_lag * torch.tanh(self.lag_head(centre.mean(1)).squeeze(-1))
        q = self.query(centre).unsqueeze(2)
        k = self.key(windows)
        v = self.value(windows)
        logits = (q * k).sum(-1) / self.temperature.abs().clamp_min(0.1)
        rel = self.relative.to(device=windows.device, dtype=windows.dtype).view(1, 1, -1)
        logits = logits - 0.5 * (rel - lag.view(-1, 1, 1)).square()
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        weights = torch.softmax(logits, dim=-1)
        output = (weights.unsqueeze(-1) * v).sum(2)
        return output, lag, weights


class NeutralArticulatoryPrototypeBank(nn.Module):
    """EMA-updated short neutral mouth trajectory prototypes."""

    def __init__(self, num_prototypes: int, patch_size: int, art_dim: int, decay: float = 0.99):
        super().__init__()
        if num_prototypes < 2 or patch_size < 3:
            raise ValueError("prototype bank requires >=2 prototypes and patch_size >=3")
        self.num_prototypes = int(num_prototypes)
        self.patch_size = int(patch_size)
        self.art_dim = int(art_dim)
        self.decay = float(decay)
        self.register_buffer("codebook", torch.zeros(num_prototypes, patch_size, art_dim))
        self.register_buffer("ema_count", torch.zeros(num_prototypes))
        self.register_buffer("ema_sum", torch.zeros(num_prototypes, patch_size, art_dim))
        self.register_buffer("initialized", torch.tensor(False))

    def _patches(self, motion: torch.Tensor) -> torch.Tensor:
        radius = self.patch_size // 2
        padded = F.pad(motion.transpose(1, 2), (radius, radius), mode="replicate").transpose(1, 2)
        return padded.unfold(1, self.patch_size, 1).permute(0, 1, 3, 2)

    @torch.no_grad()
    def update_from_targets(self, target: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if target.shape[-1] != self.art_dim:
            raise ValueError(f"Prototype target has {target.shape[-1]} channels; expected {self.art_dim}")
        patches = self._patches(target).reshape(-1, self.patch_size, self.art_dim)
        if mask is not None:
            valid = mask.reshape(-1).bool()
            patches = patches[valid]
        if patches.numel() == 0:
            return
        if not bool(self.initialized):
            indices = torch.linspace(0, patches.shape[0] - 1, self.num_prototypes, device=patches.device).long()
            chosen = patches[indices.clamp_max(patches.shape[0] - 1)]
            self.codebook.copy_(chosen)
            self.ema_sum.copy_(chosen)
            self.ema_count.fill_(1.0)
            self.initialized.fill_(True)
            return
        distances = (patches[:, None] - self.codebook[None]).square().mean((2, 3))
        labels = distances.argmin(1)
        for index in range(self.num_prototypes):
            selected = patches[labels == index]
            if selected.numel() == 0:
                continue
            count = selected.shape[0]
            self.ema_count[index].mul_(self.decay).add_(count * (1.0 - self.decay))
            self.ema_sum[index].mul_(self.decay).add_(selected.mean(0) * (1.0 - self.decay))
        self.codebook.copy_(self.ema_sum / self.ema_count.clamp_min(1e-5).view(-1, 1, 1))

    def labels(self, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = self._patches(target[..., : self.art_dim])
        distances = (patches[:, :, None] - self.codebook[None, None]).square().mean((3, 4))
        labels = distances.argmin(-1)
        if mask is not None:
            labels = labels.masked_fill(~mask.bool(), -100)
        return labels

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(logits, dim=-1)
        centre = self.codebook[:, self.patch_size // 2]
        return weights @ centre


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


class ArticulatoryStyleAdapter(nn.Module):
    """Bounded speaker articulation calibration for the canonical B0.

    It changes only mouth/jaw gain and baseline.  Timing is deliberately not
    shifted: phoneme order and event locations remain audio/content driven.
    """
    def __init__(self, motion_dim: int, style_dim: int, mouth_indices: list[int], hidden_dim: int,
                 gain_limit: float = 0.25, offset_limit: float = 0.05):
        super().__init__()
        self.mouth_indices = list(dict.fromkeys(int(index) for index in mouth_indices))
        if not self.mouth_indices:
            raise ValueError("articulatory_indices cannot be empty")
        if min(self.mouth_indices) < 0 or max(self.mouth_indices) >= motion_dim:
            raise ValueError("articulatory_indices contain an out-of-range channel")
        self.gain_limit, self.offset_limit = float(gain_limit), float(offset_limit)
        stats_dim = len(self.mouth_indices) * 3
        self.encoder = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, style_dim), nn.LayerNorm(style_dim), nn.SiLU(),
        )
        expr_dim = max(1, motion_dim - len(self.mouth_indices)) * 3
        self.expr_encoder = nn.Sequential(
            nn.Linear(expr_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, style_dim), nn.LayerNorm(style_dim), nn.SiLU(),
        )
        self.gain = nn.Linear(style_dim, len(self.mouth_indices))
        self.offset = nn.Linear(style_dim, len(self.mouth_indices))
        self.lag = nn.Linear(style_dim, 1)
        nn.init.zeros_(self.gain.weight); nn.init.zeros_(self.gain.bias)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.lag.weight); nn.init.zeros_(self.lag.bias)

    def encode_spaces(self, reference_motion: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mouth = reference_motion[..., self.mouth_indices]
        valid = torch.ones(mouth.shape[:2], device=mouth.device, dtype=mouth.dtype) if mask is None else mask.to(mouth.dtype)
        denom = valid.sum(1, keepdim=True).clamp_min(1.0)
        mean = (mouth * valid.unsqueeze(-1)).sum(1) / denom
        centered = (mouth - mean[:, None]) * valid.unsqueeze(-1)
        std = torch.sqrt(centered.square().sum(1) / denom + 1e-5)
        if mouth.shape[1] > 1:
            vv = valid[:, 1:] * valid[:, :-1]
            vm = (torch.diff(mouth, dim=1).abs() * vv.unsqueeze(-1)).sum(1) / vv.sum(1, keepdim=True).clamp_min(1.0)
        else:
            vm = torch.zeros_like(mean)
        s_art = self.encoder(torch.cat([mean, std, vm], -1))
        other = [i for i in range(reference_motion.shape[-1]) if i not in self.mouth_indices]
        body = reference_motion[..., other]
        body_mean = (body * valid.unsqueeze(-1)).sum(1) / denom
        body_centered = (body - body_mean[:, None]) * valid.unsqueeze(-1)
        body_std = torch.sqrt(body_centered.square().sum(1) / denom + 1e-5)
        body_vel = (torch.diff(body, dim=1).abs() * (valid[:, 1:] * valid[:, :-1]).unsqueeze(-1)).sum(1) / (valid[:, 1:] * valid[:, :-1]).sum(1, keepdim=True).clamp_min(1.0) if body.shape[1] > 1 else torch.zeros_like(body_mean)
        s_expr = self.expr_encoder(torch.cat([body_mean, body_std, body_vel], -1))
        lag = 2.0 * torch.tanh(self.lag(s_art)).squeeze(-1)
        return s_art, s_expr, lag, mean

    def encode(self, reference_motion: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_spaces(reference_motion, mask)[0]

    def apply(self, canonical: torch.Tensor, style: torch.Tensor, lag: torch.Tensor | None = None, baseline: torch.Tensor | None = None) -> torch.Tensor:
        mouth = canonical[..., self.mouth_indices]
        gain = torch.exp(self.gain_limit * torch.tanh(self.gain(style))).unsqueeze(1)
        offset = self.offset_limit * torch.tanh(self.offset(style)).unsqueeze(1)
        if baseline is None:
            baseline = mouth.mean(1)
        baseline = baseline.to(mouth.dtype).unsqueeze(1)
        out = canonical.clone()
        out[..., self.mouth_indices] = (baseline + gain * (mouth - baseline) + offset).clamp(0.0, 1.0)
        if lag is not None and canonical.shape[1] > 1:
            positions = torch.arange(canonical.shape[1], device=canonical.device, dtype=canonical.dtype).view(1, -1)
            positions = (positions - lag.to(canonical.dtype).view(-1, 1)).clamp(0, canonical.shape[1] - 1)
            left = positions.floor().long(); right = positions.ceil().long(); alpha = positions - left.to(positions.dtype)
            gather_left = left.unsqueeze(-1).expand(-1, -1, canonical.shape[-1])
            gather_right = right.unsqueeze(-1).expand_as(gather_left)
            shifted = torch.gather(out, 1, gather_left) * (1.0 - alpha.unsqueeze(-1)) + torch.gather(out, 1, gather_right) * alpha.unsqueeze(-1)
            out[..., self.mouth_indices] = shifted[..., self.mouth_indices]
            out = out.clamp(0.0, 1.0)
        return out


class ResidualEmotionEncoder(nn.Module):
    """BS residual teacher: global emotion and intensity only."""

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
        self.emotion_classifier = nn.Linear(emotion_dim, num_emotions)
        self.intensity_classifier = nn.Linear(emotion_dim, num_intensities)

    def forward(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.backbone(torch.cat([residual, residual_velocity(residual)], dim=-1), mask)
        global_code = self.global_head(masked_mean(hidden, mask))
        intensity_logits = self.intensity_classifier(global_code)
        return {
            "global": global_code,
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
    """Audio affect field used by Stage 3.

    Stage 3 predicts only the global coordinates consumed by the frozen
    residual renderer.  Frame-level affect is intentionally absent until it
    has a real teacher and a renderer path that uses it.
    """

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
        self.emotion_classifier = nn.Linear(emotion_dim, num_emotions)
        self.intensity_classifier = nn.Linear(emotion_dim, num_intensities)

    def forward(self, audio_emotion: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.backbone(audio_emotion, mask)
        global_code = self.global_head(masked_mean(hidden, mask))
        emotion_logits = self.emotion_classifier(global_code)
        intensity_logits = self.intensity_classifier(global_code)
        return {
            "global": global_code,
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

