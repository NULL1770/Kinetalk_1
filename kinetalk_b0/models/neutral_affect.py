"""Neutral identity, motion-to-audio affect learning, and residual flow.

The low-rate affect field is a renderer condition, not a second motion decoder.
Train the motion teacher with motion reconstruction before freezing it to teach
audio. Inference uses audio affect and neutral identity references only.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import freeze_module, masked_mean
from .dit import ResidualDiT
from .model import Stage1Model


def _mask(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError("Expected [batch, time, features]")
    if mask is None:
        mask = torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
    mask = mask.to(device=sequence.device, dtype=torch.bool)
    if mask.shape != sequence.shape[:2] or not mask.any(1).all():
        raise ValueError("Each sequence requires a matching, nonempty mask")
    if not torch.isfinite(sequence[mask]).all():
        raise ValueError("Valid frames must be finite")
    return mask


def _clean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.unsqueeze(-1), sequence, torch.zeros_like(sequence))


class NeutralIdentityEncoder(nn.Module):
    """Aggregate neutral residual statistics across independent references.

    Equal reference weighting prevents a long sentence from defining identity.
    This captures expression-coefficient offsets, not identity mesh geometry.
    """

    def __init__(self, motion_dim: int, style_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, style_dim),
        )

    def forward(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if residual.ndim == 3:
            residual = residual.unsqueeze(1)
            mask = None if mask is None else mask.unsqueeze(1)
        if residual.ndim != 4:
            raise ValueError("Identity references require [batch, refs, time, motion]")
        if mask is None:
            mask = torch.ones(residual.shape[:3], dtype=torch.bool, device=residual.device)
        mask = mask.to(device=residual.device, dtype=torch.bool)
        if mask.shape != residual.shape[:3]:
            raise ValueError("Identity mask must match [batch, refs, time]")
        valid = mask.any(-1)
        if not valid.any(1).all():
            raise ValueError("Every item needs at least one valid neutral reference")
        if not torch.isfinite(residual[mask]).all():
            raise ValueError("Valid reference frames must be finite")
        clean = _clean(residual, mask)
        weights = mask.to(clean.dtype).unsqueeze(-1)
        count = weights.sum(2).clamp_min(1)
        mean = clean.sum(2) / count
        variance = (_clean(clean - mean.unsqueeze(2), mask).square()).sum(2) / count
        # An epsilon gives a finite derivative even for a constant neutral clip.
        stats = torch.cat([mean, (variance + 1e-6).sqrt()], -1)
        per_reference = self.encoder(stats) * valid.unsqueeze(-1).to(clean.dtype)
        reference_weight = valid.to(clean.dtype).unsqueeze(-1)
        code = per_reference.sum(1) / reference_weight.sum(1)
        neutral_mean = (mean * reference_weight).sum(1) / reference_weight.sum(1)
        return {"code": code, "per_reference": per_reference,
                "reference_valid": valid, "neutral_mean": neutral_mean}


class _MaskedTemporalBlock(nn.Module):
    """A TCN block with no normalization across padded frames."""

    def __init__(self, hidden_dim: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation)
        self.output = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(1).to(x.dtype)
        hidden = F.silu(self.conv(x)) * weight
        return (x + self.output(hidden)) * weight


class _AudioControlRefiner(nn.Module):
    """Optional temporal residual on the existing control logits.

    The zero-initialized output preserves a loaded student's predictions.
    Training code freezes the shared encoder when global affect must remain
    unchanged; this module adds no global, motion, or regional output path.
    """

    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.input = nn.Linear(hidden_dim, 32)
        self.temporal = nn.ModuleList([_MaskedTemporalBlock(32, dilation) for dilation in (1, 2, 4)])
        self.output = nn.Linear(32, rank)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        refined = _clean(F.silu(self.input(hidden)), mask).transpose(1, 2)
        for block in self.temporal:
            refined = block(refined, mask)
        return _clean(self.output(refined.transpose(1, 2)), mask)


class LowRateAffectEncoder(nn.Module):
    """Shared output contract for motion teacher and audio student.

    Controls live on fixed stride bins, retain amplitude, and have their
    masked temporal mean removed. A shared projection in NeutralAffectSystem
    maps both modalities' controls into precisely the same renderer space.
    Audio may change its three TCN dilations without changing any parameter
    keys or shapes; the motion teacher always uses (1, 2, 4). Dilation is part
    of the saved configuration, not the tensor state dictionary.
    """

    def __init__(self, cfg: dict[str, Any], input_dim: int, *, motion: bool):
        super().__init__()
        data, model = cfg["data"], cfg["model"]
        hidden = int(model.get("affect_hidden_dim", model["hidden_dim"]))
        self.stride = int(model.get("affect_stride", 4))
        self.rank = int(model.get("affect_rank", 8))
        self.motion = motion
        if self.stride < 1 or self.rank < 1:
            raise ValueError("affect_stride and affect_rank must be positive")
        dilations = (1, 2, 4) if motion else model.get("audio_dilations", (1, 2, 4))
        if (not isinstance(dilations, (list, tuple)) or len(dilations) != 3
                or any(type(value) is not int or value < 1 for value in dilations)):
            raise ValueError("audio_dilations must contain exactly three positive integers")
        self.dilations = tuple(dilations)
        self.input = nn.Linear(input_dim * (2 if motion else 1), hidden)
        self.temporal = nn.ModuleList([_MaskedTemporalBlock(hidden, dilation) for dilation in self.dilations])
        self.global_head = nn.Linear(hidden, int(model["emotion_dim"]))
        self.control_head = nn.Linear(hidden, self.rank)
        self.emotion_classifier = nn.Linear(int(model["emotion_dim"]), len(data["emotion_classes"]))
        self.intensity_classifier = nn.Linear(int(model["emotion_dim"]), int(data["num_intensity_levels"]))
        refiner_enabled = model.get("audio_control_refiner", False) if not motion else False
        if type(refiner_enabled) is not bool:
            raise ValueError("audio_control_refiner must be a boolean")
        # None creates no state-dict keys, preserving the default checkpoint
        # contract. Only audio receives the optional local refinement branch.
        self.control_refiner = _AudioControlRefiner(hidden, self.rank) if refiner_enabled else None

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = _mask(sequence, mask)
        sequence = _clean(sequence, mask)
        if self.motion:
            velocity = torch.diff(sequence, dim=1, prepend=sequence[:, :1])
            # Invalid-to-valid transitions are missing observations, not motion.
            adjacent = mask & F.pad(mask[:, :-1], (1, 0), value=False)
            sequence = torch.cat([sequence, _clean(velocity, adjacent)], -1)
        hidden = _clean(F.silu(self.input(sequence)), mask).transpose(1, 2)
        for block in self.temporal:
            hidden = block(hidden, mask)
        hidden = hidden.transpose(1, 2)
        global_code = self.global_head(masked_mean(hidden, mask))
        control_logits = self.control_head(hidden)
        if self.control_refiner is not None:
            control_logits = control_logits + self.control_refiner(hidden, mask)
        frame_controls = _clean(torch.tanh(control_logits), mask)
        batch, frames, rank = frame_controls.shape
        extra = (-frames) % self.stride
        binned = F.pad(frame_controls, (0, 0, 0, extra)).reshape(batch, -1, self.stride, rank)
        weight = F.pad(mask.to(hidden.dtype), (0, extra)).reshape(batch, -1, self.stride).sum(-1)
        control_mask = weight > 0
        controls = binned.sum(2) / weight.unsqueeze(-1).clamp_min(1)
        mean = (controls * weight.unsqueeze(-1)).sum(1, keepdim=True) / weight.sum(1, keepdim=True).unsqueeze(-1)
        controls = _clean(controls - mean, control_mask)
        intensity_logits = self.intensity_classifier(global_code)
        levels = torch.arange(intensity_logits.shape[-1], device=sequence.device, dtype=hidden.dtype)
        return {"global": global_code, "controls": controls,
                "control_mask": control_mask, "control_weight": weight,
                "emotion_logits": self.emotion_classifier(global_code),
                "intensity_logits": intensity_logits,
                "intensity_value": (intensity_logits.softmax(-1) * levels).sum(-1, keepdim=True)}


class NeutralAffectSystem(nn.Module):
    """M = frozen B0 + static neutral identity bias + one flow residual."""

    def __init__(self, cfg: dict[str, Any], stage1: Stage1Model | None = None):
        super().__init__()
        data, model = cfg["data"], cfg["model"]
        self.stage1 = Stage1Model(cfg) if stage1 is None else stage1
        freeze_module(self.stage1)
        self.identity_encoder = NeutralIdentityEncoder(
            int(data["motion_dim"]), int(model["style_dim"]), int(model["hidden_dim"]))
        self.identity_bias = nn.Linear(int(model["style_dim"]), int(data["motion_dim"]))
        nn.init.zeros_(self.identity_bias.weight)
        nn.init.zeros_(self.identity_bias.bias)
        self.identity_bound = float(model.get("identity_bound", 0.5))
        self.residual_scale = float(model.get("residual_scale", 0.25))
        if self.identity_bound <= 0 or self.residual_scale <= 0:
            raise ValueError("identity_bound and residual_scale must be positive")
        self.motion_teacher = LowRateAffectEncoder(cfg, int(data["motion_dim"]), motion=True)
        audio_dim = int(data.get("audio_emotion_dim", data.get("audio_dim", data["content_dim"])))
        self.audio_encoder = LowRateAffectEncoder(cfg, audio_dim, motion=False)
        self.local_projection = nn.Linear(self.motion_teacher.rank, int(model["emotion_dim"]), bias=False)
        self.renderer = ResidualDiT(
            motion_dim=int(data["motion_dim"]), content_dim=int(model["content_dim"]),
            emotion_dim=int(model["emotion_dim"]), style_dim=int(model["style_dim"]),
            dim=int(model["dit_dim"]), depth=int(model["dit_depth"]), heads=int(model["heads"]),
            dropout=float(model["dropout"]), intensity_dim=1,
            global_dropout=float(model.get("global_condition_dropout", 0.0)),
            style_dropout=float(model.get("style_condition_dropout", 0.0)))
        self.register_buffer("architecture_version", torch.tensor(10))

    def train(self, mode: bool = True) -> NeutralAffectSystem:
        super().train(mode)
        self.stage1.eval()
        # Frozen teachers remain deterministic during student training.
        if not any(parameter.requires_grad for parameter in self.motion_teacher.parameters()):
            self.motion_teacher.eval()
        return self

    def base(self, content: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # The pretrained B0 has temporal GroupNorm. Evaluate each true length
        # so padding length cannot change its valid-frame prediction.
        if content.ndim not in (3, 4):
            raise ValueError("Content requires [batch,time,dim] or native windows")
        mask = _mask(content if content.ndim == 3 else content.flatten(2), mask)
        clean = torch.where(mask.reshape(*mask.shape, *([1] * (content.ndim - 2))), content, torch.zeros_like(content))
        last = (mask * torch.arange(1, mask.shape[1] + 1, device=mask.device)).amax(1)
        output: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for length_tensor in last.unique():
                length = int(length_tensor.item())
                indices = (last == length_tensor).nonzero(as_tuple=True)[0]
                values = self.stage1(clean[indices, :length], mask[indices, :length])
                for key in ("b0", "h0"):
                    if key not in output:
                        output[key] = values[key].new_zeros((content.shape[0], content.shape[1], values[key].shape[-1]))
                    output[key][indices, :length] = values[key]
        return {key: _clean(value, mask) for key, value in output.items()}

    def encode_identity(self, reference_residual: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        identity = self.identity_encoder(reference_residual, mask)
        identity["baseline"] = self.identity_bound * torch.tanh(self.identity_bias(identity["code"]))
        return identity

    def project_affect(self, raw: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        controls, valid = raw["controls"], raw["control_mask"]
        mask = mask.to(device=controls.device, dtype=torch.bool)
        frames, stride = mask.shape[1], self.motion_teacher.stride
        # Fixed frame coordinates, unlike interpolate(size=T), do not change
        # when a batch appends padding or includes a longer companion sample.
        position = ((torch.arange(frames, device=controls.device, dtype=controls.dtype) - (stride - 1) / 2) / stride).clamp_min(0)
        last = (valid * torch.arange(valid.shape[1], device=controls.device)).amax(1)
        lower = position.floor().long().unsqueeze(0).expand(controls.shape[0], -1)
        upper = torch.minimum(lower + 1, last[:, None])
        lower = torch.minimum(lower, last[:, None])
        fraction = (position - position.floor())[None, :, None]
        left = controls.gather(1, lower.unsqueeze(-1).expand(-1, -1, controls.shape[-1]))
        right = controls.gather(1, upper.unsqueeze(-1).expand(-1, -1, controls.shape[-1]))
        local = self.local_projection(left * (1 - fraction) + right * fraction)
        return {**raw, "local": _clean(local, mask)}

    def encode_motion(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = _mask(residual, mask)
        return self.project_affect(self.motion_teacher(residual, mask), mask)

    def encode_audio(self, audio: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = _mask(audio, mask)
        return self.project_affect(self.audio_encoder(audio, mask), mask)

    def flow(self, motion: torch.Tensor, content: torch.Tensor, mask: torch.Tensor | None,
             identity: dict[str, torch.Tensor], affect: dict[str, torch.Tensor], *,
             noise: torch.Tensor | None = None, time: torch.Tensor | None = None,
             base: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        mask = _mask(motion, mask)
        base = self.base(content, mask) if base is None else base
        baseline = identity["baseline"].unsqueeze(1)
        target = _clean(motion - base["b0"] - baseline, mask)
        if noise is None:
            noise = torch.randn_like(target)
        if noise.shape != target.shape or not torch.isfinite(noise).all():
            raise ValueError("Flow noise must be finite and match motion shape")
        if time is None:
            time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        if time.shape != (target.shape[0],) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("Flow time must be finite [batch] values in [0, 1]")
        scaled_target = target / self.residual_scale
        x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * scaled_target
        velocity_target = scaled_target - noise
        prediction = self.renderer(x_t, time, base["h0"], affect["global"], affect["intensity_value"],
                                   identity["code"], mask, local_emotion=affect["local"])
        predicted = _clean((x_t + (1 - time[:, None, None]) * prediction) * self.residual_scale, mask)
        return {"prediction": prediction, "velocity_target": _clean(velocity_target, mask),
                "target_residual": target, "predicted_residual": predicted,
                "residual": predicted, "b0": base["b0"], "x_t": x_t, "time": time,
                "motion": _clean(base["b0"] + baseline + predicted, mask)}

    def generate(self, content: torch.Tensor, mask: torch.Tensor | None,
                 identity: dict[str, torch.Tensor], affect: dict[str, torch.Tensor],
                 initial_noise: torch.Tensor | None = None, steps: int = 4, *,
                 base: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        mask = _mask(content if content.ndim == 3 else content.flatten(2), mask)
        base = self.base(content, mask) if base is None else base
        residual = self.renderer.decode(base["h0"], affect["global"], affect["intensity_value"], identity["code"], mask,
                                        residual_scale=self.residual_scale, steps=steps,
                                        local_emotion=affect["local"], initial_noise=initial_noise)
        return {"b0": base["b0"], "residual": residual,
                "motion": _clean(base["b0"] + identity["baseline"].unsqueeze(1) + residual, mask)}
