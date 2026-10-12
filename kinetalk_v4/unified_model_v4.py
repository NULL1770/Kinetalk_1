"""Experimental v4 core: geometry composition and relational factor interaction.

A new, isolated model. The shared linear rig is an observable coordinate system,
not physiological contact ground truth. Emotion categories do not select output
channels. Single-reference encoding and the duration-aware event prior reuse
established components; neither is claimed as a new contribution by itself.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from kinetalk_b0.models.expression_response import TemporalEncoder, clean, masked_pool
from kinetalk_v2.unified_model import (SingleReferenceStyle, ConditionalEventStudent,
    FeasibleResidualComposer, teacher_targets, anchor_matrix, relation_style_loss)
from kinetalk_v4.state_posture import AudioPostureAdapter
from kinetalk_v4.acoustic_geometry import AcousticGeometryReceiver, geometry_audio_context


@dataclass(frozen=True)
class KineTalkV4Config:
    audio_dim: int = 772
    content_dim: int = 128
    hidden: int = 128
    decoder_hidden: int = 192
    state_dim: int = 10
    style_dim: int = 64
    geometry_dim: int = 4
    event_groups: int = 4
    duration_frames: tuple = (2, 4, 8, 16, 32)
    classes: int = 8
    intensities: int = 4
    projector_strength: float = .5
    projector_ridge: float = 1e-4
    use_geometry: bool = True
    use_interactions: bool = True
    use_teacher: bool = True
    use_style: bool = True
    use_posture_adapter: bool = False
    teacher_guidance_mode: str = 'frame'
    geometry_audio_mode: str = 'none'
    state_supervision_space: str = 'raw'


def affine_geometry(motion, jacobian, offset):
    return torch.einsum("...c,gc->...g", motion, jacobian) + offset


class AffectEncoder(nn.Module):
    def __init__(self, cfg, input_dim, *, posture_adapter=False):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, cfg.hidden, 4, attention=1)
        self.state = nn.Linear(cfg.hidden, cfg.state_dim)
        self.emotion = nn.Linear(cfg.hidden, cfg.classes)
        self.intensity = nn.Linear(cfg.hidden, cfg.intensities)
        self.posture_adapter = AudioPostureAdapter(cfg.hidden, cfg.state_dim) if posture_adapter else None

    def forward(self, x, valid):
        hidden = self.encoder(x, valid)
        pooled = masked_pool(hidden, valid)
        state = clean(self.state(hidden), valid)
        if self.posture_adapter is not None:
            state = clean(state + self.posture_adapter(pooled)[:, None], valid)
        return dict(hidden=hidden, state=state,
                    emotion_logits=self.emotion(pooled), intensity_logits=self.intensity(pooled))


class FactorInteractionDecoder(nn.Module):
    """Unrestricted residual with explicit pair interactions and rig coordinates."""
    def __init__(self, cfg, jacobian, offset, geometry_scale):
        super().__init__(); self.cfg = cfg
        if jacobian.shape != (cfg.geometry_dim, 52) or offset.shape != (cfg.geometry_dim,):
            raise ValueError("rig geometry shape differs")
        if not torch.isfinite(jacobian).all() or not torch.isfinite(offset).all():
            raise ValueError("nonfinite rig")
        self.register_buffer("geometry_jacobian", jacobian.float())
        self.register_buffer("geometry_offset", offset.float())
        self.register_buffer("geometry_scale", geometry_scale.float().clamp_min(.05))
        self.content = nn.Linear(cfg.content_dim + 52, cfg.hidden)
        self.affect = nn.Linear(cfg.state_dim + cfg.classes + 1, cfg.hidden)
        self.style = nn.Linear(cfg.style_dim, cfg.hidden)
        self.interaction = nn.Linear(cfg.hidden * 3, cfg.hidden, bias=False)
        self.encoder = TemporalEncoder(cfg.hidden * 4 + cfg.event_groups * 3,
                                       cfg.decoder_hidden, 4, attention=1)
        self.output = nn.Linear(cfg.decoder_hidden, 52)
        self.geometry_delta = nn.Linear(cfg.decoder_hidden, cfg.geometry_dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.geometry_delta.weight); nn.init.zeros_(self.geometry_delta.bias)
        self.geometry_audio = AcousticGeometryReceiver(cfg) if cfg.geometry_audio_mode != 'none' else None

    def _project(self, proposal, target, valid):
        J = self.geometry_jacobian / self.geometry_scale[:, None]
        error = (target - affine_geometry(proposal, self.geometry_jacobian, self.geometry_offset)) / self.geometry_scale
        gram = J @ J.T + torch.eye(J.shape[0], device=J.device, dtype=J.dtype) * self.cfg.projector_ridge
        delta = torch.linalg.solve(gram, error.reshape(-1, J.shape[0]).T).T @ J
        return clean(proposal + self.cfg.projector_strength * delta.reshape_as(proposal), valid)

    def forward(self, content, base, valid, state, emotion, intensity, style, events, scales, geometry_audio=None):
        t = base.shape[1]
        c = F.silu(self.content(torch.cat((content, base / scales), -1)))
        a = F.silu(self.affect(torch.cat((state, emotion[:, None].expand(-1,t,-1),
                                       intensity[:, None, None].expand(-1,t,1)), -1)))
        s = F.silu(self.style(style))[:, None].expand(-1,t,-1)
        if not self.cfg.use_style:
            s = torch.zeros_like(s)
        pair = self.interaction(torch.cat((c*a, c*s, a*s), -1))
        if not self.cfg.use_interactions: pair = torch.zeros_like(pair)
        h = self.encoder(torch.cat((c, a, s, pair, events), -1), valid)
        # TRAIN scales normalize units; they do not impose a tanh amplitude cap.
        residual = clean(self.output(h) * scales, valid)
        proposal = base + residual
        target = affine_geometry(base, self.geometry_jacobian, self.geometry_offset)
        target = target + self.geometry_delta(h) * self.geometry_scale
        if self.geometry_audio is not None:
            target = target + self.geometry_audio(geometry_audio,state,emotion,intensity,style,valid) * self.geometry_scale
        motion = self._project(proposal, target, valid) if self.cfg.use_geometry else clean(proposal, valid)
        return dict(motion=motion, proposal=clean(proposal, valid), residual=residual,
                    geometry_target=clean(target, valid),
                    geometry_prediction=clean(affine_geometry(motion, self.geometry_jacobian, self.geometry_offset), valid))


class KineTalkV4Model(nn.Module):
    def __init__(self, cfg, feature_mean, feature_std, scales, geometry_jacobian, geometry_offset, geometry_scale=None):
        super().__init__(); self.cfg = cfg
        if cfg.teacher_guidance_mode not in ('frame', 'posture', 'none'):
            raise ValueError('Unknown teacher guidance mode')
        if cfg.geometry_audio_mode not in ('none','coarse','hubert'):
            raise ValueError('Unknown geometry audio mode')
        if cfg.state_supervision_space not in ('raw', 'decoder'):
            raise ValueError('Unknown state supervision coordinates')
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1e-3))
        self.register_buffer("scales", scales.float().clamp_min(.02))
        self.teacher = AffectEncoder(cfg, 156)
        self.student = AffectEncoder(cfg, cfg.audio_dim, posture_adapter=cfg.use_posture_adapter)
        self.style = SingleReferenceStyle(cfg)
        self.events = ConditionalEventStudent(cfg)
        self.decoder = FactorInteractionDecoder(cfg, geometry_jacobian, geometry_offset,
                         torch.ones(cfg.geometry_dim) if geometry_scale is None else geometry_scale)
        self.emotion_to_state = nn.Linear(cfg.classes, cfg.state_dim, bias=False)
        nn.init.zeros_(self.emotion_to_state.weight)
        self.composer = FeasibleResidualComposer()

    def encode_style(self, refs):
        code = self.style(refs["motion"], refs["valid"], refs["channel_mask"], self.scales)
        return code if self.cfg.use_style else torch.zeros_like(code)

    def audio_state(self, audio, valid):
        if audio.shape[-1] == 1540: audio = audio[...,768:]
        if audio.shape[-1] != self.cfg.audio_dim: raise ValueError("student input must be 772D/1540D")
        return self.student(clean((audio-self.feature_mean)/self.feature_std, valid), valid)

    def motion_teacher(self, batch):
        obs = batch["valid"][...,None] & batch["channel_mask"][:,None]
        motion = torch.where(obs, batch["motion"], 0.)
        residual = torch.where(obs, batch["motion"]-batch["b0"].detach(), 0.)
        mask = batch["channel_mask"][:,None].expand_as(motion).float()
        return self.teacher(torch.cat((residual/self.scales, motion/self.scales, mask),-1), batch["valid"])

    def condition(self, prediction, valid, *, emotion=None, intensity=None):
        probs = prediction["emotion_logits"].softmax(-1) if emotion is None else emotion
        # Relative intensity = 1 at the middle of the dataset ordinal scale.
        levels = torch.linspace(.5, 1.5, self.cfg.intensities, device=probs.device, dtype=probs.dtype)
        predicted = (prediction["intensity_logits"].softmax(-1)*levels).sum(-1)
        strength = predicted if intensity is None else intensity.reshape(-1).to(predicted)
        state = clean((prediction["state"] + self.emotion_to_state(probs)[:,None])*strength[:,None,None], valid)
        return dict(state=state, emotion=probs, intensity=strength)

    def decode(self, content, base, valid, conditioned, style, events, constraints=None, *, geometry_audio=None):
        out = self.decoder(content, base, valid, conditioned["state"], conditioned["emotion"],
                           conditioned["intensity"], style, events, self.scales, geometry_audio)
        out["motion"] = clean(self.composer(out["motion"], constraints), valid)
        return out

    def training_forward(self, batch, refs, targets, *, teacher_state=False,
                         emotion=None, intensity=None, constraints=None, seed=47):
        style = self.encode_style(refs)
        student = self.audio_state(batch["audio_features"], batch["valid"])
        teacher = self.motion_teacher(batch)
        cond = self.condition(teacher if teacher_state else student, batch["valid"],
                              emotion=emotion, intensity=intensity)
        geometry_audio = geometry_audio_context(self.cfg.geometry_audio_mode,
            batch['audio_features'],batch['h0'],batch['valid'],self.feature_mean,self.feature_std)
        # Use observed motion-derived events for the teacher-forced reconstruction
        # path. Pointwise motion/velocity losses must not be applied to a random
        # event sample as if it were the one uniquely correct trajectory.
        observed = self.events(student["hidden"], cond["state"], style, batch["valid"],
                               targets["events"], mode="observed", times=batch["times"])
        out = self.decode(batch["h0"], batch["b0"], batch["valid"], cond, style,
                          observed["events"], constraints, geometry_audio=geometry_audio)

        # Keep the deployable stochastic rollout available for a separate
        # counterfactual execution objective and audit. It receives no full-face
        # pointwise supervision; its hazard/duration/amplitude distributions are
        # trained by the observed-event likelihoods below.
        generator = torch.Generator(device=batch["motion"].device).manual_seed(seed)
        sampled = self.events(student["hidden"], cond["state"], style, batch["valid"],
                              generator=generator, mode="sample", times=batch["times"])
        fixed_cond = {key: value.detach() for key, value in cond.items()}
        sampled_motion = self.decode(batch["h0"].detach(), batch["b0"].detach(), batch["valid"],
                                     fixed_cond, style.detach(), sampled["events"].detach(),
                                     constraints, geometry_audio=geometry_audio)["motion"]
        zero_events = torch.zeros_like(targets["events"])
        with torch.no_grad():
            zero_motion = self.decode(batch["h0"], batch["b0"], batch["valid"], fixed_cond,
                                      style.detach(), zero_events, constraints, geometry_audio=geometry_audio)["motion"]
        return {**out, "student":student, "teacher":teacher, "style":style,
                "event_student":observed, "sampled_schedule":sampled,
                "sampled_motion":sampled_motion, "zero_event_motion":zero_motion,
                "conditioned":cond, "events":observed["events"], 'geometry_audio_context':geometry_audio}

    def predict(self, audio, content, base, valid, refs, *, times=None, seed=47,
                mode="sample", emotion=None, intensity=None, constraints=None, generator=None):
        style = self.encode_style(refs)
        student = self.audio_state(audio, valid)
        cond = self.condition(student, valid, emotion=emotion, intensity=intensity)
        geometry_audio = geometry_audio_context(self.cfg.geometry_audio_mode,
            audio,content,valid,self.feature_mean,self.feature_std)
        generator = torch.Generator(device=audio.device).manual_seed(seed) if generator is None else generator
        schedule = self.events(student["hidden"], cond["state"], style, valid,
                               generator=generator, mode=mode, times=times)
        out = self.decode(content, base, valid, cond, style, schedule["events"], constraints,
                          geometry_audio=geometry_audio)
        return {**out, "student":student, "style":style, "schedule":schedule,
                "conditioned":cond, "events":schedule["events"], 'geometry_audio_context':geometry_audio}

    def checkpoint_config(self): return asdict(self.cfg)


def masked_mean(value, mask):
    while mask.ndim < value.ndim: mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, 0.).sum()/mask.sum().clamp_min(1)


def semantic(logits, labels):
    use = labels.ge(0) & labels.lt(logits.shape[-1])
    return F.cross_entropy(logits[use], labels[use]) if use.any() else logits.sum()*0.


def v4_losses(model, out, targets, batch, *, teacher_warmup=False):
    teacher, student, event = out["teacher"],out["student"],out["event_student"]
    valid = batch["valid"]; quality = targets["quality"].bool() & valid[...,None]
    teacher_anchor = masked_mean((teacher["state"]-targets["state"]).square(),quality)
    # The privileged teacher already observes expression and intensity in
    # motion. Its anchored state remains in observed residual coordinates.
    # The new diagnostic trains the *actual decoder input* in those coordinates
    # rather than supervising a precursor that is transformed again later.
    student_state = (out['conditioned']['state'] if model.cfg.state_supervision_space == 'decoder'
                     else student['state'])
    guidance = masked_mean((student_state-teacher["state"].detach()).square(),quality)
    direct = masked_mean((student_state-targets["state"]).square(),quality)
    intensity = torch.where(batch["intensity_valid"], batch["intensity_id"],
                            torch.full_like(batch["intensity_id"], -1))
    losses = dict(teacher_anchor=teacher_anchor, guidance=guidance, state_direct=direct,
                  emotion=semantic(student["emotion_logits"],batch["emotion_id"]),
                  teacher_emotion=semantic(teacher["emotion_logits"],batch["emotion_id"]),
                  intensity=semantic(student["intensity_logits"],intensity),
                  teacher_intensity=semantic(teacher["intensity_logits"],intensity))
    losses["hazard"] = masked_mean(F.binary_cross_entropy_with_logits(event["hazard_logits"],
                         targets["onset"].float(),reduction="none"),targets["onset_valid"])
    dur = F.cross_entropy(event["duration_logits"].reshape(-1,len(model.cfg.duration_frames)),
                           targets["duration"].flatten(),reduction="none").reshape_as(targets["duration"])
    losses["duration"] = masked_mean(dur, targets["duration_valid"])
    params=event["amplitude_params"]; logstd=params[...,1].clamp(-4,0)
    nll=.5*((targets["amplitude"].clamp_min(.005).log()-params[...,0])/logstd.exp()).square()+logstd
    losses["amplitude"] = masked_mean(nll,targets["duration_valid"])
    J=model.decoder.geometry_jacobian; deps=J.abs().gt(1e-7)
    observed=(batch["channel_mask"][:,None] | ~deps[None]).all(-1)
    geom=affine_geometry(torch.where(batch["channel_mask"][:,None],batch["motion"],0.),
                          J,model.decoder.geometry_offset)
    mask=valid[...,None]&observed[:,None]
    scale=model.decoder.geometry_scale
    losses["geometry"] = masked_mean(F.smooth_l1_loss(out["geometry_prediction"]/scale,
                           geom/scale,reduction="none"),mask)
    losses["geometry_head"] = masked_mean(F.smooth_l1_loss(out["geometry_target"]/scale,
                           geom/scale,reduction="none"),mask)
    channel_observed = valid[...,None] & batch["channel_mask"][:,None]
    # These reconstruction terms operate on the teacher-forced observed-event
    # path. Never score sampled_motion against this single realized GT sequence.
    losses["position"] = masked_mean((out["motion"]-batch["motion"]).abs()/model.scales,
                                     channel_observed)
    if "times" in batch:
        dt = batch["times"][:,1:] - batch["times"][:,:-1]
        adjacent = valid[:,1:] & valid[:,:-1] & torch.isclose(
            dt, torch.full_like(dt,.04), rtol=1e-4, atol=1e-7)
        pred_delta = out["motion"][:,1:] - out["motion"][:,:-1]
        true_delta = batch["motion"][:,1:] - batch["motion"][:,:-1]
        losses["velocity"] = masked_mean((pred_delta-true_delta).abs()/model.scales,
            adjacent[...,None] & batch["channel_mask"][:,None])
    else:
        losses["velocity"] = losses["position"] * 0.
    # Keep the privileged teacher trainable after warmup.  Its state/emotion
    # heads must be anchored to observable targets; otherwise the deployment
    # path's detached guidance term silently gives the teacher zero gradient.
    teacher_term = (teacher_anchor + losses["teacher_emotion"] +
                    .2 * losses["teacher_intensity"] if teacher_warmup else
                    .2 * teacher_anchor + .2 * losses["teacher_emotion"] +
                    .05 * losses["teacher_intensity"]) if model.cfg.use_teacher else \
                   teacher_anchor * 0.
    losses["total"] = (teacher_term + losses["position"] +
        .35*losses["velocity"]+.5*direct+
        (.5*guidance if model.cfg.use_teacher and model.cfg.teacher_guidance_mode == 'frame' else 0.)+
        losses["emotion"]+.2*losses["intensity"])
    losses["total"] += losses["hazard"]+.2*losses["duration"]+.02*losses["amplitude"]
    if model.cfg.use_geometry: losses["total"] += .2*(losses["geometry"]+losses["geometry_head"])
    return losses


__all__=["KineTalkV4Config","KineTalkV4Model","affine_geometry","v4_losses",
         "teacher_targets","anchor_matrix","relation_style_loss"]

