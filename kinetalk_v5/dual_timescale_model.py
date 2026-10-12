"""Dual-timescale affect distillation and factorized residual response.

The module is deliberately initialized as an exact v4 continuation: the
envelope correction and the two residual adapters start at zero.  At
deployment the motion teacher is absent.  The audio student predicts a
slowly varying expression envelope plus a high-frequency expression state; the
decoder adds separately parameterized low/high residual adapters before the
existing feasible geometry composition.

The legacy adapter names are hypotheses: a frequency split alone does not
identify emotion versus articulation, and the confidence score is not yet
calibrated uncertainty. Both adapters read the existing mixed-factor hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F

from kinetalk_b0.models.expression_response import clean, masked_pool
from kinetalk_v4.acoustic_geometry import geometry_audio_context
from kinetalk_v4.unified_model_v4 import (
    KineTalkV4Config, KineTalkV4Model, FactorInteractionDecoder,
    affine_geometry, masked_mean, semantic, v4_losses,
)


@dataclass(frozen=True)
class KineTalkV5Config(KineTalkV4Config):
    lowpass_kernel: int = 9
    teacher_advantage_threshold: float = 1.0
    envelope_weight: float = 0.35
    factor_regularizer: float = 0.05
    envelope_direct_weight: float = 0.35
    confidence_weight: float = 0.1


def temporal_lowpass(value: torch.Tensor, valid: torch.Tensor, kernel: int,
                     times: torch.Tensor | None = None) -> torch.Tensor:
    """Masked-safe same-length average pool for a native 25-fps sequence.

    Invalid frames are censored boundaries.  In particular, a valid frame on
    one side of a crop/padding gap must never average a valid frame on the
    other side of that gap.  The segment id carried through the unfolded
    window makes the operation equivalent to applying the filter separately
    to each contiguous valid run.
    """
    if value.ndim != 3 or valid.shape != value.shape[:2]:
        raise ValueError("expected value [B,T,D] and valid [B,T]")
    k = int(kernel)
    if k < 1 or k % 2 == 0:
        raise ValueError("lowpass kernel must be a positive odd integer")
    if value.shape[1] == 0:
        return value
    k = min(k, value.shape[1] if value.shape[1] % 2 else value.shape[1] - 1)
    k = max(k, 1)
    b, t, d = value.shape
    pad = k // 2
    # Every invalid frame starts a new segment.  Padding is invalid, so its
    # segment id can never contribute to a valid centre.
    boundary = ~valid
    if times is not None:
        if times.shape != valid.shape:
            raise ValueError("times must have shape [B,T]")
        connected = valid[:, 1:] & valid[:, :-1] & torch.isclose(
            times[:, 1:] - times[:, :-1], torch.full_like(times[:, 1:], .04),
            rtol=1e-4, atol=1e-7)
        boundary = boundary.clone()
        boundary[:, 1:] |= ~connected
    segment = boundary.to(torch.long).cumsum(dim=1)
    value_pad = F.pad(value, (0, 0, pad, pad))
    valid_pad = F.pad(valid, (pad, pad), value=False)
    segment_pad = F.pad(segment, (pad, pad), value=-1)
    value_windows = value_pad.unfold(1, k, 1).permute(0, 1, 3, 2)  # [B,T,k,D]
    valid_windows = valid_pad.unfold(1, k, 1)      # [B,T,k]
    segment_windows = segment_pad.unfold(1, k, 1)  # [B,T,k]
    same_segment = segment_windows == segment[:, :, None]
    support = valid_windows & same_segment
    weights = support.to(value.dtype)
    numerator = torch.where(support[..., None], value_windows, 0.).sum(dim=2)
    denominator = weights.sum(dim=2, keepdim=True)
    out = numerator / denominator.clamp_min(1.)
    return clean(out, valid)


class FactorizedResidualDecoder(FactorInteractionDecoder):
    """Existing v4 decoder plus zero-start low/high residual adapters."""

    def __init__(self, cfg, jacobian, offset, geometry_scale):
        super().__init__(cfg, jacobian, offset, geometry_scale)
        self.affect_adapter = nn.Linear(cfg.decoder_hidden, 52)
        self.phonetic_adapter = nn.Linear(cfg.decoder_hidden, 52)
        self.affect_gate = nn.Parameter(torch.tensor(-2.0))
        self.phonetic_gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.affect_adapter.weight)
        nn.init.zeros_(self.affect_adapter.bias)
        nn.init.zeros_(self.phonetic_adapter.weight)
        nn.init.zeros_(self.phonetic_adapter.bias)

    def forward(self, content, base, valid, state, emotion, intensity, style,
                events, scales, geometry_audio=None, times=None):
        t = base.shape[1]
        c = F.silu(self.content(torch.cat((content, base / scales), -1)))
        a = F.silu(self.affect(torch.cat((state,
            emotion[:, None].expand(-1, t, -1),
            intensity[:, None, None].expand(-1, t, 1)), -1)))
        s = F.silu(self.style(style))[:, None].expand(-1, t, -1)
        if not self.cfg.use_style:
            s = torch.zeros_like(s)
        pair = self.interaction(torch.cat((c * a, c * s, a * s), -1))
        if not self.cfg.use_interactions:
            pair = torch.zeros_like(pair)
        h = self.encoder(torch.cat((c, a, s, pair, events), -1), valid)
        residual_base = clean(self.output(h) * scales, valid)
        h_low = temporal_lowpass(h, valid, int(self.cfg.lowpass_kernel), times)
        h_high = clean(h - h_low, valid)
        affect_residual = clean(torch.tanh(self.affect_adapter(h_low)) * scales *
                                self.affect_gate.sigmoid(), valid)
        phonetic_residual = clean(torch.tanh(self.phonetic_adapter(h_high)) * scales *
                                  self.phonetic_gate.sigmoid(), valid)
        residual = clean(residual_base + affect_residual + phonetic_residual, valid)
        proposal = base + residual
        target = affine_geometry(base, self.geometry_jacobian, self.geometry_offset)
        target = target + self.geometry_delta(h) * self.geometry_scale
        if self.geometry_audio is not None:
            target = target + self.geometry_audio(geometry_audio, state, emotion,
                                                  intensity, style, valid) * self.geometry_scale
        motion = self._project(proposal, target, valid) if self.cfg.use_geometry else clean(proposal, valid)
        return dict(motion=motion, proposal=clean(proposal, valid), residual=residual,
                    residual_base=residual_base, affect_residual=affect_residual,
                    phonetic_residual=phonetic_residual,
                    geometry_target=clean(target, valid),
                    geometry_prediction=clean(affine_geometry(motion,
                        self.geometry_jacobian, self.geometry_offset), valid))


class DualTimescaleKineTalkV5(nn.Module):
    """Deployment model with no motion teacher or query-motion input."""

    def __init__(self, cfg, feature_mean, feature_std, scales,
                 geometry_jacobian, geometry_offset, geometry_scale=None):
        super().__init__()
        if cfg.lowpass_kernel < 1 or cfg.lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel must be odd")
        self.cfg = cfg
        self.backbone = KineTalkV4Model(cfg, feature_mean, feature_std, scales,
                                        geometry_jacobian, geometry_offset,
                                        geometry_scale)
        old = self.backbone.decoder
        self.backbone.decoder = FactorizedResidualDecoder(
            cfg, geometry_jacobian, geometry_offset,
            old.geometry_scale.detach().clone())
        self.envelope_head = nn.Linear(cfg.hidden, cfg.state_dim)
        self.confidence_head = nn.Linear(cfg.hidden, 1)
        nn.init.zeros_(self.envelope_head.weight)
        nn.init.zeros_(self.envelope_head.bias)
        nn.init.constant_(self.confidence_head.bias, 1.0)
        nn.init.zeros_(self.confidence_head.weight)

    @classmethod
    def from_v4_checkpoint(cls, checkpoint, device="cpu"):
        config = dict(checkpoint["config"])
        cfg = KineTalkV5Config(**config)
        state = checkpoint["model"]
        model = cls(cfg, state["feature_mean"], state["feature_std"], state["scales"],
                    state["decoder.geometry_jacobian"], state["decoder.geometry_offset"],
                    state["decoder.geometry_scale"]).to(device)
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        unexpected = [x for x in unexpected if not x.startswith("decoder.affect_adapter")
                      and not x.startswith("decoder.phonetic_adapter")]
        allowed_missing = ("decoder.affect_gate", "decoder.phonetic_gate",
                           "decoder.affect_adapter", "decoder.phonetic_adapter")
        if unexpected or any(not any(x.startswith(prefix) for prefix in allowed_missing)
                             for x in missing):
            raise ValueError(f"v4 state transfer mismatch: missing={missing}, unexpected={unexpected}")
        return model

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        """Load either a v5 full-state checkpoint or a v4-compatible source."""
        state = checkpoint["model"]
        if any(key.startswith("backbone.") for key in state):
            cfg = KineTalkV5Config(**checkpoint["config"])
            model = cls(cfg, state["backbone.feature_mean"], state["backbone.feature_std"],
                        state["backbone.scales"], state["backbone.decoder.geometry_jacobian"],
                        state["backbone.decoder.geometry_offset"],
                        state["backbone.decoder.geometry_scale"]).to(device)
            model.load_state_dict(state, strict=True)
            return model
        return cls.from_v4_checkpoint(checkpoint, device=device)

    @property
    def feature_mean(self): return self.backbone.feature_mean
    @property
    def feature_std(self): return self.backbone.feature_std
    @property
    def scales(self): return self.backbone.scales
    @property
    def teacher(self): return self.backbone.teacher
    @property
    def student(self): return self.backbone.student
    @property
    def style(self): return self.backbone.style
    @property
    def events(self): return self.backbone.events
    @property
    def decoder(self): return self.backbone.decoder
    @property
    def emotion_to_state(self): return self.backbone.emotion_to_state
    @property
    def composer(self): return self.backbone.composer

    def encode_style(self, refs): return self.backbone.encode_style(refs)
    def motion_teacher(self, batch): return self.backbone.motion_teacher(batch)
    def condition(self, prediction, valid, **kwargs): return self.backbone.condition(prediction, valid, **kwargs)

    def audio_state(self, audio, valid, times=None):
        raw = self.backbone.audio_state(audio, valid)
        low_raw = temporal_lowpass(raw["state"], valid, self.cfg.lowpass_kernel, times)
        delta = temporal_lowpass(self.envelope_head(raw["hidden"]), valid,
                                 self.cfg.lowpass_kernel, times)
        confidence = torch.sigmoid(self.confidence_head(masked_pool(raw["hidden"], valid)))
        envelope = clean(low_raw + confidence[:, None] * delta, valid)
        high = clean(raw["state"] - low_raw, valid)
        state = clean(envelope + high, valid)
        return {**raw, "state": state, "raw_state": raw["state"],
                "envelope": envelope, "phonetic_state": high,
                "affect_confidence": confidence}

    def decode(self, content, base, valid, conditioned, style, events,
               constraints=None, *, geometry_audio=None, times=None):
        out = self.decoder(content, base, valid, conditioned["state"],
                           conditioned["emotion"], conditioned["intensity"], style,
                           events, self.scales, geometry_audio, times)
        out["motion"] = clean(self.composer(out["motion"], constraints), valid)
        return out

    def training_forward(self, batch, refs, targets, *, teacher_state=False,
                         emotion=None, intensity=None, constraints=None, seed=47):
        style = self.encode_style(refs)
        student = self.audio_state(batch["audio_features"], batch["valid"], batch["times"])
        teacher = self.motion_teacher(batch)
        cond = self.condition(teacher if teacher_state else student, batch["valid"],
                              emotion=emotion, intensity=intensity)
        geometry_audio = geometry_audio_context(self.cfg.geometry_audio_mode,
            batch["audio_features"], batch["h0"], batch["valid"],
            self.feature_mean, self.feature_std)
        observed = self.events(student["hidden"], cond["state"], style,
                               batch["valid"], targets["events"], mode="observed",
                               times=batch["times"])
        out = self.decode(batch["h0"], batch["b0"], batch["valid"], cond, style,
                          observed["events"], constraints, geometry_audio=geometry_audio,
                          times=batch["times"])
        generator = torch.Generator(device=batch["motion"].device).manual_seed(seed)
        sampled = self.events(student["hidden"], cond["state"], style,
                              batch["valid"], generator=generator, mode="sample",
                              times=batch["times"])
        fixed = {key: value.detach() for key, value in cond.items()}
        sampled_motion = self.decode(batch["h0"].detach(), batch["b0"].detach(),
            batch["valid"], fixed, style.detach(), sampled["events"].detach(),
            constraints, geometry_audio=geometry_audio, times=batch["times"])["motion"]
        zero_events = torch.zeros_like(targets["events"])
        with torch.no_grad():
            zero_motion = self.decode(batch["h0"], batch["b0"], batch["valid"],
                fixed, style.detach(), zero_events, constraints,
                geometry_audio=geometry_audio, times=batch["times"])["motion"]
        return {**out, "student": student, "teacher": teacher, "style": style,
                "event_student": observed, "sampled_schedule": sampled,
                "sampled_motion": sampled_motion, "zero_event_motion": zero_motion,
                "conditioned": cond, "events": observed["events"],
                "geometry_audio_context": geometry_audio}

    @torch.no_grad()
    def predict(self, audio, content, base, valid, refs, *, times=None, seed=47,
                mode="sample", emotion=None, intensity=None, constraints=None,
                generator=None):
        style = self.encode_style(refs)
        student = self.audio_state(audio, valid, times)
        cond = self.condition(student, valid, emotion=emotion, intensity=intensity)
        geometry_audio = geometry_audio_context(self.cfg.geometry_audio_mode, audio,
            content, valid, self.feature_mean, self.feature_std)
        generator = torch.Generator(device=audio.device).manual_seed(seed) if generator is None else generator
        schedule = self.events(student["hidden"], cond["state"], style, valid,
            generator=generator, mode=mode, times=times)
        out = self.decode(content, base, valid, cond, style, schedule["events"],
                          constraints, geometry_audio=geometry_audio, times=times)
        return {**out, "student": student, "style": style, "schedule": schedule,
                "conditioned": cond, "events": schedule["events"],
                "geometry_audio_context": geometry_audio}

    def checkpoint_config(self): return asdict(self.cfg)


def supported_envelope(value, quality, valid, kernel, times=None):
    """Filter each coordinate within its own observed runs, including clock gaps."""
    b, t, d = value.shape
    supported = quality.bool() & valid[..., None]
    series = value.permute(0, 2, 1).reshape(b * d, t, 1)
    mask = supported.permute(0, 2, 1).reshape(b * d, t)
    clocks = None if times is None else times[:, None].expand(-1, d, -1).reshape(b * d, t)
    filtered = temporal_lowpass(series, mask, kernel, clocks)
    return filtered.reshape(b, d, t).permute(0, 2, 1)


def envelope_guidance(student, teacher, target, quality, valid, kernel,
                      threshold=1.0, times=None):
    """Detached TRAIN teacher advantage with per-axis support and fractional labels."""
    support = quality.bool() & valid[..., None]
    teacher_env = supported_envelope(teacher.detach(), quality, valid, kernel, times)
    target_env = supported_envelope(target, quality, valid, kernel, times)
    count = support.sum(-1)
    teacher_err = torch.where(support, (teacher_env - target_env).square(), 0.).sum(-1) / count.clamp_min(1)
    student_err = torch.where(support, (student.detach() - target_env).square(), 0.).sum(-1) / count.clamp_min(1)
    reliable = count.gt(0) & teacher_err.le(student_err * threshold)
    mask = support & reliable[..., None]
    clip_supported = count.gt(0).any(-1)
    reliable_fraction = reliable.sum(-1).to(student.dtype) / count.gt(0).sum(-1).clamp_min(1)
    return dict(teacher=teacher_env, target=target_env, support=support, mask=mask,
                reliable_fraction=reliable_fraction.detach(), clip_supported=clip_supported)


def v5_losses(model, out, targets, batch, *, teacher_warmup=False):
    """v4 objectives plus supported low-frequency teacher distillation."""
    losses = v4_losses(model, out, targets, batch, teacher_warmup=teacher_warmup)
    valid = batch["valid"]
    student_env = out["student"]["envelope"]
    # Teacher advantage is computed only from TRAIN labels and detached. The
    # deployment path sees the learned reliability score, never this mask.
    guide = envelope_guidance(student_env, out["teacher"]["state"], targets["state"],
        targets["quality"], valid, model.cfg.lowpass_kernel,
        model.cfg.teacher_advantage_threshold, batch.get("times"))
    losses["envelope_distill"] = masked_mean((student_env - guide["teacher"]).square(), guide["mask"])
    losses["envelope_direct"] = masked_mean((student_env - guide["target"]).square(), guide["support"])
    confidence = out["student"]["affect_confidence"].reshape(-1)
    losses["confidence"] = masked_mean(F.binary_cross_entropy(confidence,
        guide["reliable_fraction"], reduction="none"), guide["clip_supported"])
    losses["teacher_reliable_fraction"] = masked_mean(guide["reliable_fraction"], guide["clip_supported"])
    affect = out.get("affect_residual")
    phonetic = out.get("phonetic_residual")
    if affect is not None and phonetic is not None:
        affect_high = affect - temporal_lowpass(affect, valid, model.cfg.lowpass_kernel, batch.get("times"))
        phonetic_low = temporal_lowpass(phonetic, valid, model.cfg.lowpass_kernel, batch.get("times"))
        cmask = valid[..., None] & batch["channel_mask"][:, None]
        losses["factor_regularizer"] = masked_mean(affect_high.abs() + phonetic_low.abs(), cmask)
    else:
        losses["factor_regularizer"] = losses["position"] * 0.
    losses["total"] = losses["total"] + model.cfg.envelope_weight * losses["envelope_distill"] + \
        model.cfg.envelope_direct_weight * losses["envelope_direct"] + \
        model.cfg.confidence_weight * losses["confidence"] + model.cfg.factor_regularizer * losses["factor_regularizer"]
    return losses


__all__ = ["KineTalkV5Config", "DualTimescaleKineTalkV5", "FactorizedResidualDecoder",
           "temporal_lowpass", "v5_losses"]
