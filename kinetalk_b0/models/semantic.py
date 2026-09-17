"""Version 9 factors with an explicit semantic boundary to the generator.

Motion and audio readouts expose only labelled semantics. Their hidden tokens
are deliberately absent from this module's conditioning interface. The Stage 1
interface is retained to load the pretrained articulation foundation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import freeze_module, masked_mean
from .dit import ResidualDiT
from .encoders import LowRateAffectField, ResidualStyleEncoder, TemporalBackbone, residual_velocity
from .model import Stage1Model


def _sequence_mask(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError("Expected a [batch, time, features] sequence")
    if mask is None:
        mask = torch.ones(sequence.shape[:2], device=sequence.device, dtype=torch.bool)
    if mask.shape != sequence.shape[:2]:
        raise ValueError("Sequence and mask must have matching batch/time axes")
    mask = mask.to(device=sequence.device, dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("Every sequence must contain at least one valid frame")
    if not torch.isfinite(sequence[mask]).all():
        raise ValueError("Valid sequence frames must be finite")
    return mask


class SemanticConditionProjector(nn.Module):
    """Deterministically lift category, real level and framewise VA to DiT tokens.

    Unknown intensity uses its own embedding and an explicit ``-1`` scalar;
    it is never treated as the weakest known level. Confidence gates invalid
    observations and weights pooling, but does not shrink valid VA toward zero.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        data, model = cfg["data"], cfg["model"]
        self.num_emotions = len(data["emotion_classes"])
        self.num_intensities = int(data["num_intensity_levels"])
        dimension = int(model["emotion_dim"])
        self.emotion_embedding = nn.Embedding(self.num_emotions + 1, dimension)
        self.intensity_embedding = nn.Embedding(self.num_intensities + 1, dimension)
        self.global_projection = nn.Sequential(
            nn.Linear(dimension * 2 + 3, dimension), nn.SiLU(),
            nn.Linear(dimension, dimension), nn.LayerNorm(dimension),
        )
        self.local_projection = LowRateAffectField(
            dimension * 2 + 3, dimension,
            stride=int(model.get("affect_stride", 4)),
        )
        self.register_buffer("architecture_version", torch.tensor(9))

    def forward(
        self,
        emotion_id: torch.Tensor,
        intensity_id: torch.Tensor,
        va: torch.Tensor,
        mask: torch.Tensor | None = None,
        va_valid: torch.Tensor | None = None,
        va_quality: torch.Tensor | None = None,
        intensity_valid: torch.Tensor | None = None,
        *,
        va_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if va.ndim != 3 or va.shape[-1] != 2:
            raise ValueError("VA must have shape [batch, time, 2]")
        batch, frames = va.shape[:2]
        if emotion_id.shape != (batch,) or intensity_id.shape != (batch,):
            raise ValueError("emotion_id and intensity_id must each have shape [batch]")
        if va_quality is not None and va_confidence is not None:
            raise ValueError("Pass va_quality or va_confidence, not both")
        quality = va_quality if va_quality is not None else va_confidence
        mask = torch.ones((batch, frames), dtype=torch.bool, device=va.device) if mask is None else mask.to(device=va.device, dtype=torch.bool)
        if mask.shape != (batch, frames) or not mask.any(dim=1).all():
            raise ValueError("Each semantic sequence requires a matching nonempty frame mask")
        valid = mask.clone() if va_valid is None else va_valid.to(device=va.device, dtype=torch.bool) & mask
        if valid.shape != (batch, frames):
            raise ValueError("va_valid must have shape [batch, time]")
        if quality is None:
            quality = torch.ones((batch, frames), device=va.device, dtype=va.dtype)
        else:
            quality = quality.to(device=va.device, dtype=va.dtype)
            if quality.shape != (batch, frames):
                raise ValueError("VA confidence must have shape [batch, time]")
            if not torch.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
                raise ValueError("VA confidence must be finite and lie in [0, 1]")
        valid = valid & (quality > 0)
        if not torch.isfinite(va[valid]).all() or (va[valid].abs() > 1.0001).any():
            raise ValueError("Valid visual VA must be finite and lie in [-1, 1]")
        clean_va = torch.where(valid.unsqueeze(-1), va, torch.zeros_like(va))

        emotion_id = emotion_id.to(device=va.device, dtype=torch.long)
        intensity_id = intensity_id.to(device=va.device, dtype=torch.long)
        if ((emotion_id < -1) | (emotion_id >= self.num_emotions)).any():
            raise ValueError("Emotion id is outside the configured classes (unknown is -1)")
        if ((intensity_id < -1) | (intensity_id >= self.num_intensities)).any():
            raise ValueError("Intensity id is outside the configured levels (unknown is -1)")
        level_valid = intensity_id >= 0
        if intensity_valid is not None:
            if intensity_valid.shape != (batch,):
                raise ValueError("intensity_valid must have shape [batch]")
            level_valid = level_valid & intensity_valid.to(device=va.device, dtype=torch.bool)
        emotion_index = torch.where(emotion_id >= 0, emotion_id, self.num_emotions)
        intensity_index = torch.where(level_valid, intensity_id, self.num_intensities)
        base = torch.cat([self.emotion_embedding(emotion_index), self.intensity_embedding(intensity_index)], dim=-1)
        clean_va = clean_va.to(dtype=base.dtype)
        weights = quality.to(dtype=base.dtype) * valid.to(dtype=base.dtype)
        pooled_va = (clean_va * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        any_va = valid.any(dim=1, keepdim=True).to(dtype=base.dtype)
        global_code = self.global_projection(torch.cat([base, pooled_va, any_va], dim=-1))
        local_features = torch.cat([
            base.unsqueeze(1).expand(-1, frames, -1), clean_va,
            valid.unsqueeze(-1).to(dtype=base.dtype),
        ], dim=-1)
        local_code = self.local_projection(local_features, mask)
        level_value = torch.where(level_valid, intensity_id, -1).to(dtype=base.dtype).unsqueeze(-1)
        return {
            "global": global_code,
            "local": local_code,
            "intensity_value": level_value,
            "intensity_valid": level_valid,
            "va_valid": valid,
        }


class _SemanticReadout(nn.Module):
    def __init__(self, cfg: dict[str, Any], input_dim: int):
        super().__init__()
        data, model = cfg["data"], cfg["model"]
        hidden_dim = int(model.get("semantic_hidden_dim", model["hidden_dim"]))
        self.backbone = TemporalBackbone(
            input_dim, hidden_dim,
            heads=int(model["heads"]), dropout=float(model["dropout"]),
        )
        self.emotion_classifier = nn.Linear(hidden_dim, len(data["emotion_classes"]))
        self.intensity_classifier = nn.Linear(hidden_dim, int(data["num_intensity_levels"]))
        self.va_head = nn.Linear(hidden_dim, 2)
        self.register_buffer("architecture_version", torch.tensor(9))

    def _read(self, features: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features, mask)
        pooled = masked_mean(hidden, mask)
        return {
            "emotion_logits": self.emotion_classifier(pooled),
            "intensity_logits": self.intensity_classifier(pooled),
            "va": torch.tanh(self.va_head(hidden)) * mask.unsqueeze(-1).to(hidden.dtype),
        }


class MotionSemanticReadout(_SemanticReadout):
    """A separately validated/frozen semantic critic, never a latent condition.

    Freezing parameters does not disable gradients with respect to generated
    residual input. Callers must not wrap a generated-motion critic pass in
    ``torch.no_grad()`` when using its semantic preservation loss.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg, int(cfg["data"]["motion_dim"]) * 2)

    def forward(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = _sequence_mask(residual, mask)
        residual = torch.where(mask.unsqueeze(-1), residual, torch.zeros_like(residual))
        return self._read(torch.cat([residual, residual_velocity(residual)], dim=-1), mask)


class SemanticAudioEncoder(_SemanticReadout):
    """Predict the same named semantic targets from audio, without a bypass."""

    def __init__(self, cfg: dict[str, Any]):
        data = cfg["data"]
        super().__init__(cfg, int(data["audio_emotion_dim"] if "audio_emotion_dim" in data else data["audio_dim"]))

    def forward(self, audio: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = _sequence_mask(audio, mask)
        audio = torch.where(mask.unsqueeze(-1), audio, torch.zeros_like(audio))
        return self._read(audio, mask)


class MultiReferenceStyleEncoder(nn.Module):
    """Encode references independently, then average unit style codes.

    Invalid references are skipped entirely, including the temporal encoder;
    they cannot produce all-padding transformer NaNs or bias the pooled style.
    Positive/negative sampling and training are handled outside this module.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        data, model = cfg["data"], cfg["model"]
        self.style_dim = int(model["style_dim"])
        self.encoder = ResidualStyleEncoder(
            motion_dim=int(data["motion_dim"]), style_dim=self.style_dim,
            hidden_dim=int(model["hidden_dim"]), heads=int(model["heads"]),
            dropout=float(model["dropout"]), use_reference_audio=False,
        )
        self.register_buffer("architecture_version", torch.tensor(9))

    def forward(
        self,
        residual: torch.Tensor,
        mask: torch.Tensor | None = None,
        reference_valid: torch.Tensor | None = None,
        *,
        return_per_reference: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if residual.ndim == 3:
            residual = residual.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)
        if residual.ndim != 4:
            raise ValueError("References must be [batch, refs, time, motion] or [batch, time, motion]")
        batch, references, frames = residual.shape[:3]
        if mask is None:
            mask = torch.ones((batch, references, frames), device=residual.device, dtype=torch.bool)
        if mask.shape != (batch, references, frames):
            raise ValueError("Reference mask must match [batch, refs, time]")
        mask = mask.to(device=residual.device, dtype=torch.bool)
        valid = mask.any(dim=-1)
        if reference_valid is not None:
            if reference_valid.shape != (batch, references):
                raise ValueError("reference_valid must match [batch, refs]")
            valid = valid & reference_valid.to(device=residual.device, dtype=torch.bool)
        if not valid.any(dim=1).all():
            raise ValueError("Every item needs at least one valid style reference")
        selected = residual[valid]
        selected_mask = _sequence_mask(selected, mask[valid])
        selected = torch.where(selected_mask.unsqueeze(-1), selected, torch.zeros_like(selected))
        codes = F.normalize(self.encoder(selected, selected_mask), dim=-1)
        per_reference = codes.new_zeros((batch, references, self.style_dim))
        per_reference[valid] = codes
        pooled = per_reference.sum(dim=1) / valid.sum(dim=1, keepdim=True).to(codes.dtype)
        style = F.normalize(pooled, dim=-1)
        if return_per_reference:
            return {"style": style, "per_reference": per_reference, "reference_valid": valid}
        return style


class SemanticGenerator(nn.Module):
    """Frozen audio articulation plus semantic/style-conditioned residual DiT."""

    def __init__(self, cfg: dict[str, Any], stage1: Stage1Model | None = None):
        super().__init__()
        data, model = cfg["data"], cfg["model"]
        self.stage1 = Stage1Model(cfg) if stage1 is None else stage1
        freeze_module(self.stage1)
        self.projector = SemanticConditionProjector(cfg)
        self.residual_scale = float(model.get("residual_scale", 0.25))
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if int(model.get("intensity_condition_dim", 1)) != 1:
            raise ValueError("Semantic generator uses one explicit intensity-level scalar")
        self.renderer = ResidualDiT(
            motion_dim=int(data["motion_dim"]), content_dim=int(model["content_dim"]),
            emotion_dim=int(model["emotion_dim"]), style_dim=int(model["style_dim"]),
            dim=int(model["dit_dim"]), depth=int(model["dit_depth"]),
            heads=int(model["heads"]), dropout=float(model["dropout"]), intensity_dim=1,
            global_dropout=float(model.get("global_condition_dropout", 0.0)),
            style_dropout=float(model.get("style_condition_dropout", 0.0)),
        )
        self.register_buffer("architecture_version", torch.tensor(9))

    def train(self, mode: bool = True) -> SemanticGenerator:
        super().train(mode)
        # Recursive train() would reactivate dropout in the frozen B0 teacher.
        self.stage1.eval()
        return self

    def _base(self, content: torch.Tensor, mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.stage1(content, mask)

    def forward_flow(
        self,
        motion: torch.Tensor,
        content: torch.Tensor,
        mask: torch.Tensor | None,
        emotion_id: torch.Tensor,
        intensity_id: torch.Tensor,
        va: torch.Tensor,
        style: torch.Tensor,
        *,
        va_valid: torch.Tensor | None = None,
        va_quality: torch.Tensor | None = None,
        intensity_valid: torch.Tensor | None = None,
        va_confidence: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        mask = _sequence_mask(motion, mask)
        base = self._base(content, mask)
        conditions = self.projector(
            emotion_id, intensity_id, va, mask, va_valid, va_quality,
            intensity_valid, va_confidence=va_confidence,
        )
        # The deployed generator adds its residual to predicted B0. Using a
        # separately aligned neutral here would train a different coordinate.
        target = torch.where(mask.unsqueeze(-1), motion - base["b0"], torch.zeros_like(motion))
        x_t, time, velocity_target = self.renderer.flow_inputs(target, self.residual_scale)
        prediction = self.renderer(
            x_t, time, base["h0"], conditions["global"],
            conditions["intensity_value"], style, mask,
            local_emotion=conditions["local"],
        )
        predicted_residual = (x_t + (1.0 - time[:, None, None]) * prediction) * self.residual_scale
        predicted_residual = predicted_residual * mask.unsqueeze(-1).to(predicted_residual.dtype)
        return {
            "prediction": prediction, "velocity_target": velocity_target,
            "target_residual": target, "predicted_residual": predicted_residual,
            "x_t": x_t, "time": time, "b0": base["b0"],
            "motion": base["b0"] + predicted_residual, "conditions": conditions,
        }

    def generate(
        self,
        content: torch.Tensor,
        mask: torch.Tensor | None,
        emotion_id: torch.Tensor,
        intensity_id: torch.Tensor,
        va: torch.Tensor,
        style: torch.Tensor,
        *,
        steps: int = 4,
        stochastic: bool = False,
        initial_noise: torch.Tensor | None = None,
        va_valid: torch.Tensor | None = None,
        va_quality: torch.Tensor | None = None,
        intensity_valid: torch.Tensor | None = None,
        va_confidence: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        base = self._base(content, mask)
        conditions = self.projector(
            emotion_id, intensity_id, va, mask, va_valid, va_quality,
            intensity_valid, va_confidence=va_confidence,
        )
        residual = self.renderer.decode(
            base["h0"], conditions["global"], conditions["intensity_value"], style, mask,
            residual_scale=self.residual_scale, steps=steps, stochastic=stochastic,
            local_emotion=conditions["local"], initial_noise=initial_noise,
        )
        # Raw BS is intentionally not clamped: diagnostics must distinguish
        # generator response from changes lost during export clipping.
        return {"b0": base["b0"], "residual": residual, "motion": base["b0"] + residual, "conditions": conditions}
