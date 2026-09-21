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
from ..reference_mouth_calibration import MOUTH, calibration_parameters
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
        # Support is a train-protocol property, never inferred from a query's
        # ground-truth channel availability at inference. Keep it in config so
        # old parameter-only checkpoints still load strictly without new keys.
        self.register_buffer("motion_support", torch.ones(int(data["motion_dim"]), dtype=torch.bool), persistent=False)
        self.register_buffer("residual_support", self.motion_support.clone(), persistent=False)
        self._residual_support_explicit = False
        if "motion_support" in model:
            self.set_motion_support(model["motion_support"])
        if "residual_support" in model:
            self.set_residual_support(model["residual_support"])
        self.register_buffer("mouth_calibration_gain", torch.ones(len(MOUTH)), persistent=False)
        self.register_buffer("mouth_calibration_bias", torch.zeros(len(MOUTH)), persistent=False)
        self.mouth_reference_calibration_enabled = False
        if "mouth_reference_calibration" in model:
            self.set_mouth_reference_calibration(model["mouth_reference_calibration"])

    def set_mouth_reference_calibration(self, calibration: dict[str, Any]) -> None:
        """Install a fixed train-only map of independent neutral-reference means.

        Save the exact dictionary in model.mouth_reference_calibration. This
        adds no learnable parameter and never takes a query motion input.
        The protected mouth must be excluded from residual_support.
        """
        if self.motion_support.numel() != 52 or not self.motion_support[list(MOUTH)].all():
            raise ValueError("Mouth calibration requires 52 channels and fixed support for every mouth channel")
        if self.residual_support[list(MOUTH)].any():
            raise ValueError("Mouth calibration requires protected mouth residual_support")
        gain, bias = calibration_parameters(calibration, device=self.motion_support.device)
        self.mouth_calibration_gain.copy_(gain)
        self.mouth_calibration_bias.copy_(bias)
        self.mouth_reference_calibration_enabled = True

    def set_motion_support(self, support: torch.Tensor | list[bool]) -> None:
        """Set fixed output support fitted from training metadata only.

        Callers must persist the same Boolean vector in model.motion_support
        of their checkpoint config. Per-query observation masks belong only
        to ``flow(observation_mask=...)``, not to deployment generation.
        """
        support = torch.as_tensor(support, device=self.motion_support.device)
        if (support.dtype != torch.bool or support.shape != self.motion_support.shape
                or not support.any()):
            raise ValueError("motion_support must be a nonempty Boolean [motion_dim] vector")
        if self._residual_support_explicit and (self.residual_support & ~support).any():
            raise ValueError("motion_support cannot exclude active residual_support channels")
        if (getattr(self, 'mouth_reference_calibration_enabled', False)
                and not support[list(MOUTH)].all()):
            raise ValueError("Enabled mouth calibration requires fixed support for every mouth channel")
        self.motion_support.copy_(support)
        if not self._residual_support_explicit:
            self.residual_support.copy_(support)

    def set_residual_support(self, support: torch.Tensor | list[bool]) -> None:
        """Choose fixed train-protocol channels eligible for flow changes.

        Channels excluded here retain B0 + neutral identity at inference;
        their observations may still teach the motion emotion encoder.
        Persist this vector in model.residual_support of checkpoint config.
        """
        support = torch.as_tensor(support, device=self.motion_support.device)
        if (support.dtype != torch.bool or support.shape != self.motion_support.shape
                or not support.any() or (support & ~self.motion_support).any()):
            raise ValueError("residual_support must be a nonempty Boolean subset of motion_support")
        if getattr(self, 'mouth_reference_calibration_enabled', False) and support[list(MOUTH)].any():
            raise ValueError("Enabled mouth calibration requires protected mouth residual_support")
        self.residual_support.copy_(support)
        self._residual_support_explicit = True

    def _motion_observations(self, motion: torch.Tensor, mask: torch.Tensor | None,
                             observation_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if motion.ndim != 3 or motion.shape[-1] != self.motion_support.numel():
            raise ValueError("Motion must be [batch,time,motion_dim]")
        observed = self.motion_support[None, None].expand_as(motion)
        if observation_mask is not None:
            if (observation_mask.dtype != torch.bool or observation_mask.device != motion.device
                    or observation_mask.shape not in ((len(motion), motion.shape[-1]), motion.shape)):
                raise ValueError("observation_mask must be Boolean [B,C] or [B,T,C] on the motion device")
            observed = observed & (observation_mask[:, None] if observation_mask.ndim == 2 else observation_mask)
        # Missing values are excluded before finiteness checking/arithmetic.
        mask = _mask(torch.where(observed, motion, 0.), mask)
        observed = observed & mask[..., None]
        if not observed.flatten(1).any(1).all():
            raise ValueError("Each motion item requires observations in fixed motion_support")
        return mask, observed

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
        output = {key: _clean(value, mask) for key, value in output.items()}
        output["b0"] = torch.where(self.motion_support[None, None], output["b0"], 0.)
        return output

    def encode_identity(self, reference_residual: torch.Tensor, mask: torch.Tensor | None = None, *,
                        reference_channel_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.mouth_reference_calibration_enabled:
            if reference_residual.ndim not in (3, 4):
                raise ValueError("Identity references require [B,T,C] or [B,R,T,C]")
            shape = reference_residual.shape[:-2] + (reference_residual.shape[-1],)
            if (reference_channel_mask is None or reference_channel_mask.dtype != torch.bool
                    or reference_channel_mask.device != reference_residual.device
                    or reference_channel_mask.shape != shape):
                raise ValueError("Calibrated identity requires explicit Boolean reference_channel_mask [B,R,C] or [B,C]")
            active = torch.ones(shape[:-1], dtype=torch.bool, device=reference_residual.device) if mask is None else mask.any(-1)
            if (active[..., None] & ~reference_channel_mask[..., list(MOUTH)]).any():
                raise ValueError("Cannot calibrate absent mouth channels in independent enrollment")
        if reference_channel_mask is not None:
            shape = reference_residual.shape[:-2] + (reference_residual.shape[-1],)
            if (reference_channel_mask.dtype != torch.bool or reference_channel_mask.shape != shape
                    or reference_channel_mask.device != reference_residual.device):
                raise ValueError("reference_channel_mask must be Boolean [B,R,C] or [B,C]")
            reference_residual = torch.where(reference_channel_mask.unsqueeze(-2), reference_residual, 0.)
        reference_residual = torch.where(self.motion_support, reference_residual, 0.)
        identity = self.identity_encoder(reference_residual, mask)
        identity["baseline"] = torch.where(self.motion_support, self.identity_bound * torch.tanh(self.identity_bias(identity["code"])), 0.)
        if self.mouth_reference_calibration_enabled:
            calibrated = (identity['neutral_mean'][..., list(MOUTH)] * self.mouth_calibration_gain
                          + self.mouth_calibration_bias)
            baseline = identity['baseline'].clone()
            baseline[..., list(MOUTH)] = calibrated
            identity['baseline'] = baseline
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
        residual = torch.where(self.motion_support[None, None], residual, 0.)
        mask = _mask(residual, mask)
        return self.project_affect(self.motion_teacher(residual, mask), mask)

    def encode_audio(self, audio: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = _mask(audio, mask)
        return self.project_affect(self.audio_encoder(audio, mask), mask)

    def flow(self, motion: torch.Tensor, content: torch.Tensor, mask: torch.Tensor | None,
             identity: dict[str, torch.Tensor], affect: dict[str, torch.Tensor], *,
             noise: torch.Tensor | None = None, time: torch.Tensor | None = None,
             base: dict[str, torch.Tensor] | None = None,
             observation_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask, motion_observed = self._motion_observations(motion, mask, observation_mask)
        observed = motion_observed & self.residual_support[None, None]
        if not observed.flatten(1).any(1).all():
            raise ValueError("Each motion item requires observations in fixed residual_support")
        base = self.base(content, mask) if base is None else base
        baseline = identity["baseline"].unsqueeze(1)
        safe_base = torch.where(motion_observed, base["b0"], 0.)
        safe_baseline = torch.where(motion_observed, baseline, 0.)
        target = torch.where(observed, motion, 0.) - torch.where(observed, safe_base, 0.) - torch.where(observed, safe_baseline, 0.)
        if noise is None:
            noise = torch.randn_like(target)
        if noise.shape != target.shape or not torch.isfinite(noise[observed]).all():
            raise ValueError("Flow noise must be finite on observations and match motion shape")
        noise = torch.where(observed, noise, 0.)
        if time is None:
            time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        if time.shape != (target.shape[0],) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("Flow time must be finite [batch] values in [0, 1]")
        scaled_target = target / self.residual_scale
        x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * scaled_target
        velocity_target = scaled_target - noise
        prediction = self.renderer(x_t, time, base["h0"], affect["global"], affect["intensity_value"],
                                   identity["code"], mask, local_emotion=affect["local"])
        prediction = torch.where(observed, prediction, 0.)
        predicted = torch.where(observed, (x_t + (1 - time[:, None, None]) * prediction) * self.residual_scale, 0.)
        return {"prediction": prediction, "velocity_target": velocity_target,
                "target_residual": target, "predicted_residual": predicted,
                "residual": predicted, "b0": safe_base, "x_t": x_t, "time": time,
                "observation_mask": observed,
                "motion": safe_base + safe_baseline + predicted}

    def generate(self, content: torch.Tensor, mask: torch.Tensor | None,
                 identity: dict[str, torch.Tensor], affect: dict[str, torch.Tensor],
                 initial_noise: torch.Tensor | None = None, steps: int = 4, *,
                 base: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        mask = _mask(content if content.ndim == 3 else content.flatten(2), mask)
        base = self.base(content, mask) if base is None else base
        residual = self.renderer.decode(base["h0"], affect["global"], affect["intensity_value"], identity["code"], mask,
                                        residual_scale=self.residual_scale, steps=steps,
                                        local_emotion=affect["local"], initial_noise=initial_noise,
                                        motion_support=self.residual_support)
        support = mask[..., None] & self.motion_support[None, None]
        safe_base = torch.where(support, base["b0"], 0.)
        safe_baseline = torch.where(support, identity["baseline"].unsqueeze(1), 0.)
        return {"b0": safe_base, "residual": residual,
                "motion": safe_base + safe_baseline + residual}
