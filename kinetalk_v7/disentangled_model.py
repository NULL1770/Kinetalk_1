"""Fast, proposal-aligned three-factor disentanglement prototype.

This module is deliberately isolated from the retained v5 control model.  It
does not call an observed motion-minus-B0 subtraction target.  A visual affect
teacher extracts a reconstructable trajectory from motion; an audio student
learns a conditional trajectory prior; and a reference encoder estimates a
speaker response code that is applied to a complete population expression.

The architecture is a candidate, not a paper result.  It exposes intervention
outputs and losses so a short remote pilot can reject it before long training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from kinetalk_b0.models.expression_response import TemporalEncoder, clean, masked_pool


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x)
    return torch.where(mask, x, 0.).sum() / mask.sum().clamp_min(1)


def _seq_mse(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return _masked_mean((x - y).square(), valid)


def _seq_l1(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return _masked_mean((x - y).abs(), valid)


def _valid_velocity(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 2:
        return x.sum() * 0.
    adjacent = valid[:, 1:] & valid[:, :-1]
    return _masked_mean(((x[:, 1:] - x[:, :-1]) - (y[:, 1:] - y[:, :-1])).abs(), adjacent)


@dataclass(frozen=True)
class V7Config:
    content_in: int = 768
    affect_in: int = 772
    motion_dim: int = 52
    content_dim: int = 128
    hidden: int = 96
    decoder_hidden: int = 128
    emotion_dim: int = 16
    event_dim: int = 8
    style_dim: int = 24
    classes: int = 8
    intensity_bins: int = 4
    flow_steps: int = 8
    teacher_depth: int = 2
    decoder_depth: int = 2
    style_depth: int = 2
    use_flow: bool = True
    intervention_scale: float = .25
    flow_noise_scale: float = .25
    flow_noise_correlation: float = .85
    posterior_ridge: float = .05
    posterior_block_frames: int = 8
    speaker_classes: int = 64
    use_response: bool = True
    use_articulation_anchor: bool = True
    teacher_emotion_weight: float = 2.0

    @property
    def trajectory_dim(self) -> int:
        return self.emotion_dim + self.event_dim


class _Head(nn.Module):
    def __init__(self, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale):
        ctx.scale=scale
        return x.view_as(x)
    @staticmethod
    def backward(ctx,grad):
        return -ctx.scale*grad,None


class VisualAffectTeacher(nn.Module):
    """Privileged motion encoder with a reconstructable semantic trajectory."""

    def __init__(self, cfg: V7Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalEncoder(cfg.motion_dim, cfg.hidden, cfg.teacher_depth,
                                       attention=1)
        self.trajectory = _Head(cfg.hidden, cfg.trajectory_dim)
        self.emotion = _Head(cfg.hidden, cfg.classes)
        self.intensity = _Head(cfg.hidden, 1)
        self.reliability = _Head(cfg.hidden, 1)

    def forward(self, motion: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(motion, valid)
        traj = clean(self.trajectory(h), valid)
        pooled = masked_pool(h, valid)
        frame_intensity = clean(self.intensity(h),valid).squeeze(-1)
        return {
            "hidden": h,
            "trajectory": traj,
            "emotion_logits": self.emotion(pooled),
            "intensity": masked_pool(frame_intensity[...,None],valid).squeeze(-1),
            "frame_intensity":frame_intensity,
            "reliability": clean(self.reliability(h).sigmoid(), valid),
            "emotion_code": masked_pool(traj[..., :self.cfg.emotion_dim], valid),
        }


class AudioAffectStudent(nn.Module):
    """Audio-only semantic/trajectory prior condition."""

    def __init__(self, cfg: V7Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalEncoder(cfg.affect_in, cfg.hidden, 2, attention=1)
        self.trajectory = _Head(cfg.hidden, cfg.trajectory_dim)
        self.logvar = _Head(cfg.hidden, cfg.trajectory_dim)
        self.emotion = _Head(cfg.hidden, cfg.classes)
        self.intensity = _Head(cfg.hidden, 1)

    def forward(self, affect: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(affect, valid)
        trajectory = clean(self.trajectory(h), valid)
        logvar = clean(self.logvar(h).clamp(-6., 2.), valid)
        frame_intensity = clean(self.intensity(h),valid).squeeze(-1)
        return {
            "hidden": h,
            "trajectory_mean": trajectory,
            "trajectory_logvar": logvar,
            "emotion_logits": self.emotion(masked_pool(h, valid)),
            "intensity": masked_pool(frame_intensity[...,None],valid).squeeze(-1),
            "frame_intensity":frame_intensity,
            "emotion_code": masked_pool(trajectory[..., :self.cfg.emotion_dim], valid),
        }


class ConditionalStyleResponse(nn.Module):
    """Reference -> posterior mean/log-variance and context coverage.

    The encoder sees reference motion only.  It never sees the query motion.
    Multiple references are encoded independently then averaged, so the same
    interface supports the deployment one-reference case and multi-reference
    training diagnostics.
    """

    def __init__(self, cfg: V7Config):
        super().__init__()
        self.encoder = TemporalEncoder(cfg.motion_dim * 3, cfg.hidden,
                                       cfg.style_depth, attention=1)
        self.mean = _Head(cfg.hidden, cfg.style_dim)
        self.logvar = _Head(cfg.hidden, cfg.style_dim)
        self.coverage = _Head(cfg.hidden, 4)

    def forward(self, refs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        motion = refs["motion"]
        valid = refs["valid"].bool()
        if motion.ndim == 3:
            motion, valid = motion[:, None], valid[:, None]
        if motion.ndim != 4 or valid.shape != motion.shape[:3]:
            raise ValueError("refs must contain motion[B,T,52] or motion[B,R,T,52]")
        b, r, t, d = motion.shape
        if d != 52:
            raise ValueError("reference motion must be 52-D ARKit")
        channel = refs.get("channel_mask")
        if channel is None:
            channel = torch.ones(b, r, d, dtype=torch.bool, device=motion.device)
        if channel.ndim == 2:
            channel = channel[:, None].expand(-1, r, -1)
        obs = valid[..., None] & channel[:, :, None]
        x = torch.where(obs, motion, 0.)
        mean = masked_pool(x.reshape(b * r, t, d), valid.reshape(b * r, t)).reshape(b, r, 1, d)
        feats = torch.cat((x, x - mean, channel[:, :, None].expand(-1, -1, t, -1).float()), -1)
        flat_valid = valid.reshape(b * r, t)
        flat_h = self.encoder(feats.reshape(b * r, t, -1), flat_valid)
        flat_pool = masked_pool(flat_h, flat_valid).reshape(b, r, -1)
        ref_ok = valid.any(-1).float()
        weights = ref_ok / ref_ok.sum(-1, keepdim=True).clamp_min(1.)
        pooled = (flat_pool * weights[..., None]).sum(1)
        mean_code = self.mean(pooled)
        logvar = self.logvar(pooled).clamp(-6., 2.)
        per_reference_mean = self.mean(flat_pool)
        per_reference_logvar = self.logvar(flat_pool).clamp(-6., 2.)
        return {"mean": mean_code, "logvar": logvar,
                "coverage": self.coverage(pooled).sigmoid(), "per_reference": flat_pool,
                "per_reference_mean": per_reference_mean,
                "per_reference_logvar": per_reference_logvar,
                "valid": ref_ok.bool()}


class ConditionalTrajectoryFlow(nn.Module):
    """Small conditional flow-matching velocity field in affect/event space."""

    def __init__(self, cfg: V7Config):
        super().__init__()
        self.cfg = cfg
        self.time = nn.Sequential(nn.Linear(4, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.condition = nn.Linear(cfg.hidden, cfg.hidden)
        self.temporal = TemporalEncoder(cfg.trajectory_dim + cfg.hidden * 2,
                                        cfg.hidden, 2, attention=1)
        self.event_temporal = TemporalEncoder(cfg.trajectory_dim + cfg.hidden * 2,
                                              cfg.hidden, 2, attention=1)
        self.emotion_head = nn.Linear(cfg.hidden, cfg.emotion_dim)
        self.event_head = nn.Sequential(nn.Linear(cfg.hidden + cfg.style_dim, cfg.hidden),
                                        nn.SiLU(), nn.Linear(cfg.hidden, cfg.event_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, hidden: torch.Tensor,
                style: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Fourier time features make short pilots stable without a large DiT.
        tau = torch.cat((t, torch.sin(2. * torch.pi * t),
                         torch.cos(2. * torch.pi * t), t.square()), -1)
        th = self.time(tau)
        cond = self.condition(hidden)
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        canonical_x = torch.cat((x[...,:self.cfg.emotion_dim],
                                 torch.zeros_like(x[...,self.cfg.emotion_dim:])), -1)
        h = self.temporal(torch.cat((canonical_x, th.expand_as(cond), cond), -1), valid)
        # The canonical E velocity never receives S. Personal habits can enter
        # the local event head, and output tests must still check E preservation.
        nh = self.event_temporal(torch.cat((x,th.expand_as(cond),cond),-1),valid)
        event = self.event_head(torch.cat((nh, style[:, None].expand(-1, x.shape[1], -1)), -1))
        return clean(torch.cat((self.emotion_head(h), event), -1), valid)


class PopulationResponseDecoder(nn.Module):
    """Complete population expression plus style-conditioned response field."""

    def __init__(self, cfg: V7Config):
        super().__init__()
        traj = cfg.trajectory_dim
        self.population = TemporalEncoder(cfg.content_dim + traj, cfg.decoder_hidden,
                                          cfg.decoder_depth, attention=1)
        self.cfg = cfg
        self.response = TemporalEncoder(cfg.content_dim + traj,
                                        cfg.decoder_hidden, cfg.decoder_depth, attention=1)
        self.population_head = nn.Linear(cfg.decoder_hidden, cfg.motion_dim)
        self.response_head = nn.Linear(cfg.decoder_hidden, cfg.motion_dim * cfg.style_dim)
        nn.init.normal_(self.response_head.weight, std=.003)
        nn.init.zeros_(self.response_head.bias)

    def basis(self, content: torch.Tensor, trajectory: torch.Tensor,
              valid: torch.Tensor) -> torch.Tensor:
        h = self.response(torch.cat((content, trajectory), -1), valid)
        return clean(self.response_head(h), valid).reshape(*content.shape[:2],
                                                            self.cfg.motion_dim, self.cfg.style_dim)

    def forward(self, content: torch.Tensor, trajectory: torch.Tensor,
                style: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        pop_h = self.population(torch.cat((content, trajectory), -1), valid)
        pop = clean(self.population_head(pop_h), valid)
        basis = self.basis(content, trajectory, valid)
        response = torch.einsum('btdk,bk->btd', basis, style)
        return {"motion": clean(pop + response, valid), "population": pop,
                "response": response, "basis": basis, "population_hidden": pop_h}


class MotionContentObserver(nn.Module):
    """Optional frozen-after-pretrain observer targetting B0's h0 content code."""

    def __init__(self, cfg: V7Config):
        super().__init__()
        self.encoder = TemporalEncoder(cfg.motion_dim, cfg.hidden, 2, attention=1)
        self.output = nn.Linear(cfg.hidden, cfg.content_dim)

    def forward(self, motion: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(motion, valid))


class V7DisentangledModel(nn.Module):
    def __init__(self, cfg: V7Config = V7Config()):
        super().__init__()
        self.cfg = cfg
        self.content = TemporalEncoder(cfg.content_in, cfg.hidden, 2, attention=1)
        self.content_projection = nn.Linear(cfg.hidden, cfg.content_dim)
        self.anchor_projection = nn.Sequential(nn.LayerNorm(cfg.content_dim),
                                               nn.Linear(cfg.content_dim,cfg.content_dim),nn.SiLU(),
                                               nn.Linear(cfg.content_dim,cfg.content_dim))
        self.teacher = VisualAffectTeacher(cfg)
        self.student = AudioAffectStudent(cfg)
        self.style = ConditionalStyleResponse(cfg)
        self.flow = ConditionalTrajectoryFlow(cfg)
        self.decoder = PopulationResponseDecoder(cfg)
        self.content_observer = MotionContentObserver(cfg)
        self.content_emotion_probe = _Head(cfg.content_dim,cfg.classes)
        self.content_speaker_probe = _Head(cfg.content_dim,cfg.speaker_classes)
        self.affect_speaker_probe = _Head(cfg.emotion_dim,cfg.speaker_classes)
        self.style_emotion_probe = _Head(cfg.style_dim,cfg.classes)
        self.style_speaker_probe = _Head(cfg.style_dim,cfg.speaker_classes)
        self.class_to_emotion = nn.Linear(cfg.classes, cfg.emotion_dim, bias=False)
        nn.init.normal_(self.class_to_emotion.weight, std=.02)
        self.register_buffer('motion_scale',torch.ones(cfg.motion_dim))

    def canonicalize(self, trajectory: torch.Tensor, logits: torch.Tensor,
                     valid: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared clip semantics plus zero-mean, multidimensional temporal E.

        Known labels anchor training. Deployment uses inferred probabilities or
        an explicit user control; the temporal state remains multidimensional.
        """
        weights=logits.softmax(-1)
        if labels is not None:
            known=labels.ge(0)&labels.lt(self.cfg.classes)
            weights=torch.where(known[:,None],F.one_hot(labels.clamp(0,self.cfg.classes-1),self.cfg.classes).float(),weights)
        prototypes=F.normalize(self.class_to_emotion.weight.T,dim=-1)*self.cfg.intervention_scale
        g=weights@prototypes
        local=trajectory[...,:self.cfg.emotion_dim]
        local=local-masked_pool(local,valid)[:,None]
        return clean(torch.cat((local+g[:,None],trajectory[...,self.cfg.emotion_dim:]),-1),valid)

    def encode_content(self, audio_content: torch.Tensor, valid: torch.Tensor,
                       h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        if audio_content.shape[-1] != self.cfg.content_in:
            raise ValueError(f"content expects {self.cfg.content_in}D")
        out=self.content_projection(self.content(audio_content, valid))
        if self.cfg.use_articulation_anchor and h0 is not None:
            if h0.shape[-1]!=self.cfg.content_dim: raise ValueError('h0 content dimension mismatch')
            out=out+self.anchor_projection(h0)
        return clean(out, valid)

    def encode_style(self, refs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "audio_features" in refs:
            result=self.response_posterior(refs)
        else:
            result=self.style(refs)
        if not self.cfg.use_response:
            result={**result,'mean':result['mean']*0.}
        return result

    def response_posterior(self, refs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Conditional linear Gaussian response estimate from independent refs.

        C/E explain each reference's own performance. Only S enters the query.
        Adjacent frames are pooled into blocks before precision accumulation;
        reported uncertainty is model-conditional, not calibrated physical truth.
        """
        y, valid, audio = refs['motion'], refs['valid'].bool(), refs['audio_features']
        if y.ndim == 3:
            y, valid, audio = y[:, None], valid[:, None], audio[:, None]
        b, r, t, d = y.shape
        vf = valid.reshape(b*r, t)
        safe_vf = vf.clone()
        safe_vf[~vf.any(1),0] = True
        audio_flat = torch.where(vf[...,None], audio.reshape(b*r,t,-1),0.)
        y_flat = torch.where(vf[...,None], y.reshape(b*r,t,d),0.)
        cf = self.encode_content(audio_flat[..., :self.cfg.content_in], safe_vf)
        rt = self.teacher(y_flat, safe_vf)
        ef = self.canonicalize(rt['trajectory'],rt['emotion_logits'],safe_vf)
        decoded = self.decoder(cf, ef, torch.zeros(b*r, self.cfg.style_dim, device=y.device), safe_vf)
        basis = decoded['basis']
        residual = (y.reshape(b*r, t, d) - decoded['population'])/self.motion_scale
        basis=basis/self.motion_scale[None,None,:,None]
        channel = refs.get('channel_mask')
        if channel is None:
            channel = torch.ones(b, r, d, dtype=torch.bool, device=y.device)
        elif channel.ndim == 2:
            channel = channel[:, None].expand(-1,r,-1)
        obs = vf[..., None] & channel.reshape(b*r,1,d)
        block = self.cfg.posterior_block_frames
        pad = (-t) % block
        bm = F.pad(obs.float(), (0,0,0,pad)).reshape(b*r,-1,block,d)
        by = F.pad(residual * obs, (0,0,0,pad)).reshape(b*r,-1,block,d)
        bb = F.pad(basis * obs[...,None], (0,0,0,0,0,pad)).reshape(b*r,-1,block,d,self.cfg.style_dim)
        count = bm.sum(2).clamp_min(1.)
        rb = by.sum(2)/count
        xb = bb.sum(2)/count[...,None]
        support = bm.sum(2).ge(block/2).float()
        xb = xb * support[...,None]
        rb = rb * support
        xtx = torch.einsum('bndk,bndl->bkl', xb, xb)
        xty = torch.einsum('bndk,bnd->bk', xb, rb)
        eye = torch.eye(self.cfg.style_dim,device=y.device)[None]
        precision = eye*self.cfg.posterior_ridge + xtx
        per_mean = torch.linalg.solve(precision, xty[...,None]).squeeze(-1)
        per_var = torch.linalg.inv(precision).diagonal(dim1=-2,dim2=-1)*self.cfg.posterior_ridge
        # Combine all references once, retaining one prior (no double-counting).
        p = eye*self.cfg.posterior_ridge + xtx.reshape(b,r,self.cfg.style_dim,self.cfg.style_dim).sum(1)
        rhs = xty.reshape(b,r,self.cfg.style_dim).sum(1)
        mean = torch.linalg.solve(p, rhs[...,None]).squeeze(-1)
        var = torch.linalg.inv(p).diagonal(dim1=-2,dim2=-1)*self.cfg.posterior_ridge
        return {'mean':mean, 'logvar':var.clamp_min(1e-6).log(),
                'per_reference_mean':per_mean.reshape(b,r,-1),
                'per_reference_logvar':per_var.clamp_min(1e-6).log().reshape(b,r,-1),
                'valid':valid.any(-1), 'coverage':support.reshape(b,r,-1).mean(-1)}

    def encode_teacher(self, motion: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.teacher(motion, valid)

    def encode_student(self, affect: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        if affect.shape[-1] != self.cfg.affect_in:
            raise ValueError(f"affect expects {self.cfg.affect_in}D (emotion2vec+prosody)")
        result=self.student(affect, valid)
        result['trajectory_mean']=self.canonicalize(result['trajectory_mean'],result['emotion_logits'],valid)
        return result

    def correlated_noise(self, target: torch.Tensor, generator=None) -> torch.Tensor:
        """AR(1) latent base with unit marginal variance, not output smoothing."""
        z=torch.randn(target.shape,device=target.device,dtype=target.dtype,generator=generator)
        rho=self.cfg.flow_noise_correlation
        chunks=[z[:,0]]
        for i in range(1,z.shape[1]):chunks.append(rho*chunks[-1]+(1-rho*rho)**.5*z[:,i])
        return torch.stack(chunks,1)*self.cfg.flow_noise_scale

    def class_control(self, labels: torch.Tensor, valid: torch.Tensor,
                      trajectory: Optional[torch.Tensor] = None) -> torch.Tensor:
        labels = labels.clamp(0, self.cfg.classes - 1)
        code = self.class_to_emotion(F.one_hot(labels, self.cfg.classes).float())
        # A fixed, bounded intervention magnitude prevents a near-zero class
        # embedding from making the advertised control path numerically inert
        # during a short pilot.  Training still learns the class directions.
        code = code / code.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        code = code * self.cfg.intervention_scale
        if trajectory is None:
            trajectory = torch.zeros(labels.shape[0], valid.shape[1],
                                     self.cfg.trajectory_dim, device=labels.device)
        out = trajectory.clone()
        out[..., :self.cfg.emotion_dim] = out[..., :self.cfg.emotion_dim] - masked_pool(out[..., :self.cfg.emotion_dim], valid)[:,None]
        out[..., :self.cfg.emotion_dim] = out[..., :self.cfg.emotion_dim] + code[:, None]
        return clean(out, valid)

    def decode(self, content: torch.Tensor, trajectory: torch.Tensor,
               style: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decoder(content, trajectory, style, valid)

    def flow_loss(self, student: dict[str, torch.Tensor], target: torch.Tensor,
                  style: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_flow:
            return _seq_mse(student["trajectory_mean"], target, valid)
        # Fit a distribution around the predictive mean. No sampled prior is
        # forced to reproduce the unique observed performance frame by frame.
        target = target.detach() - student['trajectory_mean'].detach()
        target=target.clone()
        target[...,:self.cfg.emotion_dim]-=masked_pool(target[...,:self.cfg.emotion_dim],valid)[:,None]
        x0 = self.correlated_noise(target)
        t = torch.rand(target.shape[0], 1, 1, device=target.device, dtype=target.dtype)
        xt = (1. - t) * x0 + t * target.detach()
        velocity = self.flow(xt, t.expand(target.shape[0], target.shape[1], 1),
                             student["hidden"], style, valid)
        return _masked_mean((velocity - (target.detach() - x0)).square(), valid)

    @torch.no_grad()
    def sample_trajectory(self, student: dict[str, torch.Tensor], style: torch.Tensor,
                          valid: torch.Tensor, seed: int = 47, steps: Optional[int] = None) -> torch.Tensor:
        if not self.cfg.use_flow:
            return student["trajectory_mean"]
        steps = int(steps or self.cfg.flow_steps)
        generator = torch.Generator(device=student["hidden"].device).manual_seed(seed)
        x = self.correlated_noise(student['trajectory_mean'],generator)
        for i in range(steps):
            t = torch.full((x.shape[0], x.shape[1], 1), float(i) / max(steps, 1),
                           device=x.device, dtype=x.dtype)
            x = x + self.flow(x, t, student["hidden"], style, valid) / max(steps, 1)
        x[...,:self.cfg.emotion_dim]-=masked_pool(x[...,:self.cfg.emotion_dim],valid)[:,None]
        return clean(x + student['trajectory_mean'], valid)

    def forward(self, batch: Mapping[str, torch.Tensor], refs: Mapping[str, torch.Tensor],
                *, sample_flow: bool = False, seed: int = 47) -> dict[str, object]:
        audio = batch["audio_features"]
        valid = batch["valid"].bool()
        content = self.encode_content(audio[..., :self.cfg.content_in], valid,batch.get('h0'))
        affect = audio[..., -self.cfg.affect_in:]
        teacher = self.encode_teacher(batch["motion"], valid)
        teacher['trajectory']=self.canonicalize(teacher['trajectory'],teacher['emotion_logits'],valid,batch.get('emotion_id'))
        student = self.encode_student(affect, valid)
        style = self.encode_style(refs)
        teacher_out = self.decode(content, teacher["trajectory"], style["mean"], valid)
        student_traj = (self.sample_trajectory(student, style["mean"], valid, seed=seed)
                        if sample_flow else student["trajectory_mean"])
        student_out = self.decode(content, student_traj, style["mean"], valid)
        flow_loss = self.flow_loss(student, teacher["trajectory"].detach(),
                                   style["mean"], valid)
        return {"content": content, "teacher": teacher, "student": student,
                "style": style, "teacher_out": teacher_out,
                "student_out": student_out, "student_trajectory": student_traj,
                "flow_loss": flow_loss}

    def predict(self, audio: torch.Tensor, valid: torch.Tensor,
                refs: Mapping[str,torch.Tensor], *, sample_flow: bool = True,
                h0: Optional[torch.Tensor] = None,
                seed: int = 47, emotion_id: Optional[torch.Tensor] = None) -> dict:
        """Deployment path: query audio + independent reference, no query GT."""
        content = self.encode_content(audio[...,:self.cfg.content_in],valid,h0)
        student = self.encode_student(audio[...,-self.cfg.affect_in:],valid)
        style = self.encode_style(refs)
        trajectory = (self.sample_trajectory(student,style['mean'],valid,seed)
                      if sample_flow else student['trajectory_mean'])
        if emotion_id is not None:
            trajectory = self.class_control(emotion_id,valid,trajectory)
        return {**self.decode(content,trajectory,style['mean'],valid),
                'trajectory':trajectory, 'style':style, 'student':student}

    @torch.no_grad()
    def interventions(self, batch: Mapping[str, torch.Tensor], refs: Mapping[str, torch.Tensor],
                      *, emotion_ids: Optional[torch.Tensor] = None, seed: int = 47) -> dict[str, dict[str, torch.Tensor]]:
        self.eval()
        audio = batch["audio_features"]
        valid = batch["valid"].bool()
        content = self.encode_content(audio[..., :self.cfg.content_in], valid)
        student = self.encode_student(audio[..., -self.cfg.affect_in:], valid)
        style = self.encode_style(refs)
        base = student["trajectory_mean"]
        if emotion_ids is None:
            emotion_ids = torch.arange(self.cfg.classes, device=audio.device)[:min(self.cfg.classes, audio.shape[0])]
        outputs = {}
        for label in (emotion_ids.reshape(-1).tolist()):
            labels = torch.full((audio.shape[0],), int(label), device=audio.device, dtype=torch.long)
            traj = self.class_control(labels, valid, base)
            outputs[f"emotion_{label}"] = self.decode(content, traj, style["mean"], valid)["motion"]
        outputs["style_zero"] = self.decode(content, base, torch.zeros_like(style["mean"]), valid)["motion"]
        outputs["style_ref"] = self.decode(content, base, style["mean"], valid)["motion"]
        outputs["sample_1"] = self.decode(content, self.sample_trajectory(student, style["mean"], valid, seed, 6), style["mean"], valid)["motion"]
        outputs["sample_2"] = self.decode(content, self.sample_trajectory(student, style["mean"], valid, seed + 1, 6), style["mean"], valid)["motion"]
        return {"outputs": outputs, "student": student, "style": style, "content": content}

    def checkpoint_config(self) -> dict:
        return asdict(self.cfg)


def v7_loss(model: V7DisentangledModel, out: Mapping[str, object],
            batch: Mapping[str, torch.Tensor], *, teacher_weight: float = 1.,
            student_weight: float = 1., flow_weight: float = .25,
            content_weight: float = .1, style_weight: float = .02,
            control_weight: float = .2, leakage_weight: float = .05) -> dict[str, torch.Tensor]:
    valid = batch["valid"].bool()
    channel = batch.get('channel_mask')
    observed = valid if channel is None else valid[...,None] & channel[:,None].bool()
    motion = batch["motion"]
    teacher, student = out["teacher"], out["student"]
    teacher_motion = out["teacher_out"]["motion"]
    student_motion = out["student_out"]["motion"]
    losses: dict[str, torch.Tensor] = {}
    scale=model.motion_scale[None,None]
    losses["teacher_recon"] = _seq_l1(teacher_motion/scale, motion/scale, observed) * teacher_weight
    losses["teacher_velocity"] = _valid_velocity(teacher_motion, motion, observed) * .25*teacher_weight
    losses["student_recon"] = _seq_l1(student_motion/scale, motion/scale, observed) * student_weight
    losses["student_velocity"] = _valid_velocity(student_motion, motion, observed) * .15*student_weight
    losses["flow"] = out["flow_loss"] * flow_weight
    losses['trajectory_mean_distill'] = _seq_mse(student['trajectory_mean'], teacher['trajectory'].detach(), valid)*.2
    labels = batch.get("emotion_id")
    if labels is not None:
        known = labels.ge(0) & labels.lt(model.cfg.classes)
        losses["teacher_emotion"] = (F.cross_entropy(teacher["emotion_logits"][known], labels[known])
                                      if known.any() else teacher["emotion_logits"].sum() * 0.) * model.cfg.teacher_emotion_weight
        losses["student_emotion"] = (F.cross_entropy(student["emotion_logits"][known], labels[known])
                                      if known.any() else student["emotion_logits"].sum() * 0.)
        target_code = model.class_control(labels, valid)[..., :model.cfg.emotion_dim][:,0]
        code_mask = known[:, None]
        losses["semantic_code"] = _masked_mean((teacher["emotion_code"] - target_code).square(), code_mask) * .2
        # This is a controlled intervention objective: the class code must
        # have a measurable semantic effect through the same decoder used at
        # inference. It is not a framewise emotion ground truth.
        control_traj = model.class_control(labels, valid, student["trajectory_mean"])
        control_motion = model.decode(out["content"], control_traj,
                                      out["style"]["mean"], valid)["motion"]
        control_logits = model.teacher(control_motion, valid)["emotion_logits"]
        losses["emotion_control"] = (F.cross_entropy(control_logits[known], labels[known])
                                      if known.any() else control_logits.sum() * 0.) * control_weight
        generated_logits=model.teacher(student_motion,valid)['emotion_logits']
        losses['student_output_emotion']=(F.cross_entropy(generated_logits[known],labels[known])
                                          if known.any() else generated_logits.sum()*0.)*.5
        teacher_generated_logits=model.teacher(teacher_motion,valid)['emotion_logits']
        losses['teacher_output_emotion']=(F.cross_entropy(teacher_generated_logits[known],labels[known])
                                          if known.any() else teacher_generated_logits.sum()*0.)*.5
        cp=masked_pool(out['content'],valid)
        losses['content_emotion_leakage']=(F.cross_entropy(model.content_emotion_probe(_Reverse.apply(cp,.1))[known],labels[known])
                                          if known.any() else cp.sum()*0.)*leakage_weight
        losses['style_emotion_leakage']=(F.cross_entropy(model.style_emotion_probe(_Reverse.apply(out['style']['mean'],.1))[known],labels[known])
                                        if known.any() else cp.sum()*0.)*leakage_weight
    else:
        losses["teacher_emotion"] = teacher["emotion_logits"].sum() * 0.
        losses["student_emotion"] = student["emotion_logits"].sum() * 0.
        losses["semantic_code"] = teacher["emotion_code"].sum() * 0.
        losses["emotion_control"] = teacher["emotion_logits"].sum() * 0.
    intensity = batch.get("intensity")
    if intensity is not None:
        intensity_known = batch.get('intensity_valid', torch.isfinite(intensity) & intensity.ge(0)).bool()
        intensity_known = intensity_known & torch.isfinite(intensity) & intensity.ge(0)
        losses["intensity"] = (F.smooth_l1_loss(teacher["intensity"][intensity_known], intensity.float()[intensity_known])
                               if intensity_known.any() else teacher['intensity'].sum()*0.)*.1
        losses['student_intensity'] = (F.smooth_l1_loss(student['intensity'][intensity_known], intensity.float()[intensity_known])
                                      if intensity_known.any() else student['intensity'].sum()*0.)*.1
    else:
        losses["intensity"] = teacher["intensity"].sum() * 0.
    h0 = batch.get("h0")
    if h0 is not None and h0.shape[-1] == model.cfg.content_dim:
        observed_code = model.content_observer(motion, valid)
        generated_code = model.content_observer(student_motion, valid)
        # The observer is trainable during a pilot; a later stage freezes it
        # after real motion->h0 pretraining.  It is an anchor, not proof of
        # phoneme completeness.
        losses["content_observer"] = _seq_mse(observed_code, h0.detach(), valid) * content_weight
        losses["content_preserve"] = _seq_mse(generated_code, h0.detach(), valid) * content_weight
        losses['content_anchor'] = _seq_mse(out['content'],h0.detach(),valid)*content_weight
    else:
        losses["content_observer"] = teacher_motion.sum() * 0.
        losses["content_preserve"] = teacher_motion.sum() * 0.
    style = out["style"]
    sids=batch.get('speaker_class')
    if sids is not None:
        known_s=sids.ge(0)&sids.lt(model.cfg.speaker_classes)
        def speaker_loss(head,code,reverse=False):
            logits=head(_Reverse.apply(code,.1) if reverse else code)
            return (F.cross_entropy(logits[known_s],sids[known_s]) if known_s.any() else logits.sum()*0.)
        losses['content_speaker_leakage']=speaker_loss(model.content_speaker_probe,masked_pool(out['content'],valid),True)*leakage_weight
        losses['affect_speaker_leakage']=speaker_loss(model.affect_speaker_probe,teacher['emotion_code'],True)*leakage_weight
        losses['style_identity_auxiliary']=speaker_loss(model.style_speaker_probe,style['mean'])*leakage_weight
    losses["style_prior"] = style["mean"].square().mean() * style_weight
    # Analytic posterior variance already derives from the precision. Do not
    # force it to zero log variance or advertise it as calibrated confidence.
    losses["style_logvar"] = style["logvar"].sum()*0.
    losses['style_center'] = style['mean'].mean(0).square().mean()*style_weight
    losses['event_capacity'] = _masked_mean(teacher['trajectory'][...,model.cfg.emotion_dim:].square(),valid)*.002
    if style.get("per_reference_mean") is not None and style["per_reference_mean"].shape[1] > 1:
        ref_valid = style["valid"].float()
        center = (style["per_reference_mean"] * ref_valid[..., None]).sum(1) / ref_valid.sum(1, keepdim=True).clamp_min(1.)
        losses["style_support_consistency"] = _masked_mean(
            (style["per_reference_mean"] - center[:, None]).square(),
            ref_valid.bool()[..., None]) * style_weight
    else:
        losses["style_support_consistency"] = losses["style_prior"] * 0.
    losses["total"] = sum(losses.values())
    return losses


__all__ = ["V7Config", "V7DisentangledModel", "v7_loss"]
