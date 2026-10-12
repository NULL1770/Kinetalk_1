"""Anchored factor response and TRAIN-only expression posterior.

All heads support all 52 coefficients. Roles are enforced through input
contracts and algebraic zero anchors, not anatomical/category channel masks.
The 24D content bottleneck is learned without phoneme labels: it is not a
validated viseme space. Distribution coordinates describe measured expression,
not psychological ground truth or calibrated uncertainty.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from kinetalk_b0.models.expression_response import TemporalEncoder, clean
from kinetalk_v4.unified_model_v4 import (
    KineTalkV4Config, KineTalkV4Model, FactorInteractionDecoder,
    affine_geometry, masked_mean,
)
from kinetalk_v5.dual_timescale_model import temporal_lowpass


@dataclass(frozen=True)
class KineTalkV6Config(KineTalkV4Config):
    content_bottleneck: int = 24
    distribution_kernel: int = 9
    logvar_min: float = -6.
    logvar_max: float = 1.
    posterior_weight: float = .5
    distribution_weight: float = .1
    bootstrap_weight: float = .25
    reference_weight: float = .1


class AnchoredFactorDecoder(FactorInteractionDecoder):
    """No inherited joint output, mixed temporal encoder or geometry bypass."""
    def __init__(self, cfg, jacobian, offset, geometry_scale):
        super().__init__(cfg, jacobian, offset, geometry_scale)
        # Retain only the observer buffers and projection method from v4.
        for name in ('content','affect','style','interaction','encoder','output',
                     'geometry_delta','geometry_audio'):
            delattr(self, name)
        self.content_encoder = TemporalEncoder(cfg.content_dim+52, cfg.hidden, 3, attention=1)
        self.content_code = nn.Linear(cfg.hidden, cfg.content_bottleneck)
        self.content_expand = nn.Linear(cfg.content_bottleneck, cfg.hidden, bias=False)
        self.expression_encoder = TemporalEncoder(cfg.state_dim+cfg.classes+1+
            cfg.event_groups*3, cfg.hidden, 3, attention=1)
        width = 52+cfg.geometry_dim
        self.content_response = nn.Linear(cfg.content_bottleneck, width)
        self.expression_response = nn.Linear(cfg.hidden, width, bias=False)
        self.interaction_response = nn.Linear(cfg.hidden, width, bias=False)
        self.style_posture = nn.Linear(cfg.style_dim, width, bias=False)
        self.style_content_gain = nn.Linear(cfg.style_dim, cfg.content_bottleneck, bias=False)
        self.style_expression_gain = nn.Linear(cfg.style_dim, cfg.hidden, bias=False)
        for head in (self.content_response,self.expression_response,
                     self.interaction_response,self.style_posture):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, content, base, valid, state, emotion, intensity, style,
                events, scales, geometry_audio=None, times=None):
        t = base.shape[1]
        z = torch.tanh(self.content_code(self.content_encoder(
            torch.cat((content,base/scales),-1),valid)))
        c = torch.tanh(self.content_expand(z))
        x = torch.cat((state,emotion[:,None].expand(-1,t,-1),
            intensity[:,None,None].expand(-1,t,1),events),-1)
        # Position encodings/biases cancel under exactly the same native mask.
        e = clean(self.expression_encoder(x,valid)-
                  self.expression_encoder(torch.zeros_like(x),valid),valid)
        cr = clean(self.content_response(z),valid)
        er = clean(self.expression_response(e),valid)
        ir = clean(self.interaction_response(c*e),valid) if self.cfg.use_interactions else torch.zeros_like(er)
        if self.cfg.use_style:
            cg = .25*torch.tanh(self.style_content_gain(style))[:,None]
            eg = .25*torch.tanh(self.style_expression_gain(style))[:,None]
            # Style changes posture/response, never enters the content code or
            # supplies an unrestricted second mixed-factor sequence decoder.
            sr = self.style_posture(style)[:,None].expand(-1,t,-1)
            sr = clean(sr + F.linear(z*cg,self.content_response.weight) +
                       self.expression_response(e*eg),valid)
        else:
            sr = torch.zeros_like(er)
        components = dict(content_response=cr,expression_response=er,
                          interaction_response=ir,style_response=sr)
        summed = cr+er+ir+sr
        residual = clean(summed[...,:52]*scales,valid)
        proposal = clean(base+residual,valid)
        target = clean(affine_geometry(base,self.geometry_jacobian,self.geometry_offset)+
                       summed[...,52:]*self.geometry_scale,valid)
        motion = self._project(proposal,target,valid) if self.cfg.use_geometry else proposal
        return dict(motion=motion,proposal=proposal,residual=residual,
            geometry_target=target,geometry_prediction=clean(affine_geometry(
                motion,self.geometry_jacobian,self.geometry_offset),valid),
            content_code=clean(z,valid), **components)


class FactorPathKineTalkV6(KineTalkV4Model):
    def __init__(self,cfg,feature_mean,feature_std,scales,geometry_jacobian,
                 geometry_offset,geometry_scale=None):
        if cfg.distribution_kernel<1 or cfg.distribution_kernel%2==0:
            raise ValueError('distribution_kernel must be positive odd')
        if not cfg.logvar_min<cfg.logvar_max:
            raise ValueError('invalid log variance range')
        if cfg.geometry_audio_mode!='none':
            raise ValueError('v6 has no auxiliary mixed geometry bypass')
        super().__init__(cfg,feature_mean,feature_std,scales,geometry_jacobian,
                         geometry_offset,geometry_scale)
        self.decoder = AnchoredFactorDecoder(cfg,geometry_jacobian,geometry_offset,
                                             self.decoder.geometry_scale)
        # Observable expression means have already learned intensity in their
        # coordinates; do not multiply them a second time in default inference.
        del self.emotion_to_state
        self.student_logvar = nn.Linear(cfg.hidden,cfg.state_dim)
        self.teacher_logvar = nn.Linear(cfg.hidden,cfg.state_dim)
        for head in (self.student_logvar,self.teacher_logvar):
            nn.init.zeros_(head.weight);nn.init.constant_(head.bias,-3.)

    @classmethod
    def from_parent(cls,checkpoint,device='cpu',**overrides):
        cfg = KineTalkV6Config(**{**checkpoint['config'],**overrides})
        state = checkpoint['model']
        if 'feature_mean' not in state:
            raise ValueError('v4-compatible raw-direct parent required')
        model = cls(cfg,state['feature_mean'],state['feature_std'],state['scales'],
            state['decoder.geometry_jacobian'],state['decoder.geometry_offset'],
            state['decoder.geometry_scale']).to(device)
        # Strict module migration; old joint decoder never becomes a deployment path.
        for name in ('teacher','student','style','events','composer'):
            prefix=name+'.'
            getattr(model,name).load_state_dict({k[len(prefix):]:v for k,v in state.items()
                                                if k.startswith(prefix)},strict=True)
        return model

    @classmethod
    def from_checkpoint(cls,checkpoint,device='cpu'):
        cfg=KineTalkV6Config(**checkpoint['config']);s=checkpoint['model']
        model=cls(cfg,s['feature_mean'],s['feature_std'],s['scales'],
                  s['decoder.geometry_jacobian'],s['decoder.geometry_offset'],
                  s['decoder.geometry_scale']).to(device)
        model.load_state_dict(s,strict=True)
        return model

    def audio_state(self,audio,valid):
        out=super().audio_state(audio,valid)
        return {**out,'logvar':clean(self.student_logvar(out['hidden']).clamp(
            self.cfg.logvar_min,self.cfg.logvar_max),valid)}

    def motion_teacher(self,batch):
        out=super().motion_teacher(batch)
        return {**out,'logvar':clean(self.teacher_logvar(out['hidden']).clamp(
            self.cfg.logvar_min,self.cfg.logvar_max),batch['valid'])}

    def condition(self,prediction,valid,*,emotion=None,intensity=None):
        probs=prediction['emotion_logits'].softmax(-1) if emotion is None else emotion
        levels=torch.linspace(.5,1.5,self.cfg.intensities,device=probs.device,dtype=probs.dtype)
        predicted=(prediction['intensity_logits'].softmax(-1)*levels).sum(-1)
        strength=predicted if intensity is None else intensity.reshape(-1).to(predicted)
        relative=torch.ones_like(strength) if intensity is None else strength/predicted.clamp_min(.05)
        return dict(state=clean(prediction['state']*relative[:,None,None],valid),
                    emotion=probs,intensity=strength)

    def decode(self,content,base,valid,conditioned,style,events,constraints=None,
               *,geometry_audio=None,times=None):
        out=self.decoder(content,base,valid,conditioned['state'],conditioned['emotion'],
            conditioned['intensity'],style,events,self.scales,times=times)
        out['motion']=clean(self.composer(out['motion'],constraints),valid)
        return out

    def distribution_sample(self,prediction,valid,times,generator):
        # Correlated unit-variance Gaussian innovations on each valid native run.
        noise=torch.randn(prediction['state'].shape,device=valid.device,
                          dtype=prediction['state'].dtype,generator=generator)
        smooth=temporal_lowpass(noise,valid,self.cfg.distribution_kernel,times)
        k=min(self.cfg.distribution_kernel,valid.shape[1] if valid.shape[1]%2 else valid.shape[1]-1)
        k=max(1,k);pad=k//2
        boundary=~valid
        if times is not None:
            boundary=boundary.clone()
            adjacent=valid[:,1:]&valid[:,:-1]&torch.isclose(times[:,1:]-times[:,:-1],
                torch.full_like(times[:,1:],.04),rtol=1e-4,atol=1e-7)
            boundary[:,1:]|=~adjacent
        segment=boundary.long().cumsum(1)
        windows=F.pad(segment,(pad,pad),value=-1).unfold(1,k,1)
        observed=F.pad(valid,(pad,pad),value=False).unfold(1,k,1)
        counts=(observed&(windows==segment[:,:,None])).sum(-1,keepdim=True)
        sampled=clean(prediction['state']+smooth*counts.clamp_min(1).sqrt()*
                      (.5*prediction['logvar']).exp(),valid)
        return {**prediction,'state':sampled}

    def training_forward(self,batch,refs,targets,*,teacher_state=False,
                         emotion=None,intensity=None,constraints=None,seed=47):
        out=super().training_forward(batch,refs,targets,teacher_state=teacher_state,
            emotion=emotion,intensity=intensity,constraints=constraints,seed=seed)
        generator=torch.Generator(device=batch['motion'].device).manual_seed(seed+101)
        posterior=self.distribution_sample(out['teacher'],batch['valid'],batch['times'],generator)
        pcond=self.condition(posterior,batch['valid'],emotion=emotion,intensity=intensity)
        pout=self.decode(batch['h0'],batch['b0'],batch['valid'],pcond,out['style'],
                         targets['events'],constraints,times=batch['times'])
        return {**out,'posterior_motion':pout['motion'],'posterior_conditioned':pcond}

    @torch.no_grad()
    def predict(self,audio,content,base,valid,refs,*,times=None,seed=47,mode='sample',
                emotion=None,intensity=None,constraints=None,generator=None):
        style=self.encode_style(refs);student=self.audio_state(audio,valid)
        generator=torch.Generator(device=audio.device).manual_seed(seed) if generator is None else generator
        chosen=self.distribution_sample(student,valid,times,generator) if mode=='sample' and self.cfg.distribution_weight else student
        cond=self.condition(chosen,valid,emotion=emotion,intensity=intensity)
        schedule=self.events(student['hidden'],cond['state'],style,valid,
            generator=generator,mode=mode,times=times)
        out=self.decode(content,base,valid,cond,style,schedule['events'],constraints,times=times)
        return {**out,'student':student,'style':style,'schedule':schedule,
                'conditioned':cond,'events':schedule['events'],'geometry_audio_context':None}

    def checkpoint_config(self):return asdict(self.cfg)


def distribution_losses(model,out,targets,batch):
    """q posterior learns reconstruction; p matches detached q and observed targets."""
    p,q=out['student'],out['teacher']
    mask=targets['quality'].bool()&batch['valid'][...,None]
    target=torch.where(mask,targets['state'],0.)
    lv=p['logvar'];qlv=q['logvar'].detach();qmean=q['state'].detach()
    nll=masked_mean(.5*(lv+(p['state']-target).square()*(-lv).exp()),mask)
    kl=masked_mean(.5*(lv-qlv+(qlv-lv).exp()+
        (p['state']-qmean).square()*(-lv).exp()-1.),mask)
    plv=p['logvar'].detach();pmean=p['state'].detach()
    qkl=masked_mean(.5*(plv-q['logvar']+(q['logvar']-plv).exp()+
        (q['state']-pmean).square()*(-plv).exp()-1.),mask)
    return dict(prior_nll=nll,posterior_to_prior_kl=kl,
                posterior_kl=qkl,total=nll+model.cfg.distribution_weight*(kl+qkl))
