"""Anchored expression teacher, probabilistic event student, joint residual.

No query motion is accepted by predict(). Observed events are used only by the
training reconstruction path; sampled rollouts do not receive full-face L2.
Phonetic constraints must come from a separately calibrated asset; absence of
such an asset is explicit and never replaced with a learned confidence gate.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from kinetalk_b0.models.expression_response import TemporalEncoder, clean, masked_pool


@dataclass(frozen=True)
class UnifiedConfig:
    audio_dim: int = 772
    content_dim: int = 128
    hidden: int = 128
    decoder_hidden: int = 192
    state_dim: int = 10
    style_dim: int = 64
    event_groups: int = 4
    duration_frames: tuple = (2, 4, 8, 16, 32)
    classes: int = 8
    intensities: int = 4


def anchor_matrix():
    """Ten signed/measured expression coordinates, not psychological VA."""
    matrix = torch.zeros(52, 10)
    groups = [
        ({23:.5,24:.5,25:-.5,26:-.5}),  # smile minus frown
        ({29:.25,30:.25,35:.25,36:.25}),  # mouth tension
        ({43:.5,44:.25,45:.25}),          # brow lift
        ({41:.5,42:.5}),                 # brow contraction
        ({6:.5,13:.5,5:-.5,12:-.5}),      # eye opening minus squint
        ({46:.2,47:.2,48:.2,49:.2,50:.2}), # cheek/nose
        ({27:.5,28:.5}),                 # dimples
        ({33:.5,34:.5}),                 # lip shrug
        ({37:.25,38:.25,39:.25,40:.25}),  # vertical lip deformation
        ({19:.5,20:.5}),                 # rounding
    ]
    for axis, group in enumerate(groups):
        for channel, weight in group.items():
            matrix[channel, axis] = weight
    return matrix


def adjacency(valid, times):
    return valid[:,1:] & valid[:,:-1] & torch.isclose(times[:,1:]-times[:,:-1],
        torch.full_like(times[:,1:],.04), rtol=1e-4, atol=1e-7)


@torch.no_grad()
def teacher_targets(motion, base, valid, channels, times, anchors=None,
                    event_thresholds=(.25,.055,.045,.055), duration_frames=(2,4,8,16,32)):
    """Observed states and four potentially co-occurring event streams.

    Events: eyelid closure, brow emphasis, cheek/nose emphasis, lip tension.
    Regions 1..3 use departures from a within-run median, so a sustained emotion
    is a state rather than thousands of events. Boundary/gap-cut events are
    censored; their onset/duration do not become ordinary positive labels.
    This parser is a versioned weak motion labeler, not a calibrated detector.
    """
    if motion.shape != base.shape or motion.shape[-1]!=52 or valid.shape!=motion.shape[:2]:
        raise ValueError('Expected aligned motion/base [B,T,52] and mask')
    if channels.shape!=(motion.shape[0],52) or channels.dtype!=torch.bool:
        raise ValueError('Expected observed channel mask bool[B,52]')
    obs=valid[...,None]&channels[:,None]
    residual=torch.where(obs,motion-base.detach(),0.)
    matrix=anchor_matrix().to(motion)
    state=residual@matrix
    quality=(channels.float()@matrix.abs())/matrix.abs().sum(0).clamp_min(1e-6)
    # Missing anchor parts are not silently imputed as an observed zero.
    quality=quality.eq(1).to(motion.dtype)[:,None].expand_as(state)*valid[...,None]
    if anchors is not None:
        # reference posture is separate from directional expression state.
        state=torch.where(quality.bool(),state,0.)
    levels=torch.stack(((motion[...,0]+motion[...,7])*.5,
        state[...,2:4].abs().mean(-1),state[...,5].abs(),state[...,1].abs()),-1)
    axis_quality=(channels.float()@matrix.abs()).eq(matrix.abs().sum(0)[None])
    event_quality=torch.stack((channels[:,[0,7]].all(-1),axis_quality[:,2:4].all(-1),
        axis_quality[:,5],axis_quality[:,1]),-1)
    batch,length=valid.shape; groups=4
    active=torch.zeros(batch,length,groups,device=motion.device,dtype=torch.bool)
    onset=active.clone(); onset_valid=active.clone(); event_valid=active.clone()
    duration=torch.zeros(batch,length,groups,device=motion.device,dtype=torch.long)
    duration_valid=active.clone(); amplitude=torch.zeros_like(levels); phase=torch.zeros_like(levels)
    adj=adjacency(valid,times)
    # Run boundaries are determined by observations AND clock, never compacted.
    for b in range(batch):
        t=0
        while t<length:
            if not valid[b,t]: t+=1; continue
            right=t+1
            while right<length and adj[b,right-1]:right+=1
            for g in range(groups):
                if not event_quality[b,g]:continue
                values=levels[b,t:right,g]
                centered=values if g==0 else (values-values.median()).clamp_min(0.)
                flags=centered>event_thresholds[g]
                event_valid[b,t:right,g]=True
                onset_valid[b,t+1:right,g]=True
                j=0
                while j<len(flags):
                    if not flags[j]:j+=1;continue
                    end=j+1
                    while end<len(flags) and flags[end]:end+=1
                    lo,hi=t+j,t+end; n=hi-lo
                    active[b,lo:hi,g]=True
                    onset[b,lo,g]=True
                    amp=centered[j:end].max().clamp_min(.005)
                    amplitude[b,lo:hi,g]=amp
                    phase[b,lo:hi,g]=torch.arange(n,device=motion.device,dtype=motion.dtype)/max(n-1,1)
                    if j>0 and end<len(flags):
                        duration_valid[b,lo,g]=True
                        duration[b,lo,g]=min(range(len(duration_frames)),key=lambda k:abs(duration_frames[k]-n))
                    j=end
            t=right
    events=torch.stack((active.float(),phase,amplitude),-1).flatten(-2)
    return {'state':state,'quality':quality,'events':events,'active':active,'onset':onset,
        'onset_valid':onset_valid,'event_valid':event_valid,'duration':duration,
        'duration_valid':duration_valid,'amplitude':amplitude,'phase':phase}


class SingleReferenceStyle(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder=TemporalEncoder(52*3,cfg.hidden,3,attention=1)
        self.output=nn.Linear(cfg.hidden,cfg.style_dim)
    def forward(self, motion, valid, channels, scales):
        if motion.ndim!=3 or motion.shape[-1]!=52 or valid.shape!=motion.shape[:2]:
            raise ValueError('One reference [B,T,52] required')
        obs=valid[...,None]&channels[:,None]
        x=torch.where(obs,motion,0.)/scales
        mean=masked_pool(x,valid)[:,None]
        # Centered trajectories plus posture; no reference audio required.
        h=self.encoder(torch.cat((x,x-mean,channels[:,None].expand_as(x).float()),-1),valid)
        return self.output(masked_pool(h,valid))


class MotionStateTeacher(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder=TemporalEncoder(52*3,cfg.hidden,3,attention=1)
        self.state=nn.Linear(cfg.hidden,cfg.state_dim)
        self.emotion=nn.Linear(cfg.hidden,cfg.classes)
    def forward(self,motion,base,valid,channels,scales):
        obs=valid[...,None]&channels[:,None]
        x=torch.where(obs,motion-base.detach(),0.)/scales
        h=self.encoder(torch.cat((x,torch.where(obs,motion,0.)/scales,
            channels[:,None].expand_as(x).float()),-1),valid)
        return {'state':clean(self.state(h),valid), 'emotion_logits':self.emotion(masked_pool(h,valid))}


class AudioStateStudent(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder=TemporalEncoder(cfg.audio_dim,cfg.hidden,4,attention=1)
        self.state=nn.Linear(cfg.hidden,cfg.state_dim)
        self.emotion=nn.Linear(cfg.hidden,cfg.classes)
        self.intensity=nn.Linear(cfg.hidden,cfg.intensities)
    def forward(self,audio,valid):
        h=self.encoder(audio,valid)
        return {'hidden':h,'state':clean(self.state(h),valid),
            'emotion_logits':self.emotion(masked_pool(h,valid)),
            'intensity_logits':self.intensity(masked_pool(h,valid))}


class ConditionalEventStudent(nn.Module):
    """Causal history with independent hazards and joint regional conditioning."""
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.context=nn.Linear(cfg.hidden+cfg.state_dim+cfg.style_dim,cfg.hidden)
        self.recurrent=nn.GRUCell(cfg.hidden+cfg.event_groups*3,cfg.hidden)
        self.hazard=nn.Linear(cfg.hidden,cfg.event_groups)
        self.duration=nn.Linear(cfg.hidden,cfg.event_groups*len(cfg.duration_frames))
        self.amplitude=nn.Linear(cfg.hidden,cfg.event_groups*2)
        nn.init.constant_(self.hazard.bias,-3.)
        with torch.no_grad():
            self.amplitude.bias.reshape(cfg.event_groups,2)[:,0].fill_(-2.3)
            self.amplitude.bias.reshape(cfg.event_groups,2)[:,1].fill_(-1.)
    def step(self,context,history,hidden):
        hidden=self.recurrent(torch.cat((context,history),-1),hidden)
        return hidden,self.hazard(hidden),self.duration(hidden).reshape(-1,self.cfg.event_groups,len(self.cfg.duration_frames)),self.amplitude(hidden).reshape(-1,self.cfg.event_groups,2)
    def forward(self,hidden,state,style,valid,events=None,*,generator=None,mode='sample',times=None):
        if mode not in ('sample','modal','observed'):raise ValueError('Invalid event mode')
        if mode=='observed' and events is None:raise ValueError('Observed events required')
        b,t,_=hidden.shape;g=self.cfg.event_groups
        context=F.silu(self.context(torch.cat((hidden,state,style[:,None].expand(-1,t,-1)),-1)))
        h=hidden.new_zeros(b,self.cfg.hidden); previous=hidden.new_zeros(b,g*3)
        remaining=torch.zeros(b,g,device=hidden.device,dtype=torch.long); total=remaining.clone()
        amp=hidden.new_zeros(b,g)
        hazards=[];durations=[];amplitudes=[];rollout=[]
        for i in range(t):
            connected=valid[:,i]&valid[:,i-1] if i else torch.zeros(b,device=hidden.device,dtype=torch.bool)
            if i and times is not None:connected=connected&torch.isclose(times[:,i]-times[:,i-1],torch.full_like(times[:,i],.04),rtol=1e-4,atol=1e-7)
            h=torch.where(connected[:,None],h,0.);previous=torch.where(connected[:,None],previous,0.)
            remaining=torch.where(connected[:,None],remaining,0)
            h,hazard,duration,params=self.step(context[:,i],previous,h)
            hazards.append(hazard);durations.append(duration);amplitudes.append(params)
            if mode=='observed': event=events[:,i]
            else:
                free=remaining.eq(0)&valid[:,i,None]
                probability=hazard.sigmoid()
                if mode=='sample':
                    onset=torch.rand(probability.shape,device=hidden.device,generator=generator)<probability
                    chosen=torch.multinomial(duration.softmax(-1).reshape(-1,duration.shape[-1]),1,generator=generator).reshape(b,g)
                    noise=torch.randn(amp.shape,device=hidden.device,generator=generator)
                else:
                    onset=probability>.5;chosen=duration.argmax(-1);noise=torch.zeros_like(amp)
                start=free&onset
                lengths=torch.tensor(self.cfg.duration_frames,device=hidden.device)[chosen]
                total=torch.where(start,lengths,total);remaining=torch.where(start,lengths,remaining)
                sampled=(params[...,0]+noise*params[...,1].clamp(-4,0).exp()).clamp(-6,0).exp()
                amp=torch.where(start,sampled,amp)
                active=remaining.gt(0)
                phase=torch.where(active,(total-remaining).float()/(total-1).clamp_min(1),0.)
                event=torch.stack((active.float(),phase,amp*active),-1).flatten(-2)
                remaining=(remaining-1).clamp_min(0)
            event=clean(event[:,None],valid[:,i,None])[:,0]
            rollout.append(event);previous=event.detach()
        return {'events':torch.stack(rollout,1),'hazard_logits':torch.stack(hazards,1),
            'duration_logits':torch.stack(durations,1),'amplitude_params':torch.stack(amplitudes,1)}


class FeasibleResidualComposer(nn.Module):
    """Few-step differentiable proximal composition of low-dimensional intervals.

    Each row of J measures a calibrated linear rig observation. Bounds and
    reliability come from phonetic stage/fit evidence, and are detached. Empty
    constraints mean plain residual and are reported as an ablation.
    """
    def __init__(self,steps=4,strength=8.):
        super().__init__();self.steps=steps;self.strength=strength
    def forward(self,proposal,constraints=None):
        if constraints is None:return proposal
        j=constraints['jacobian'].to(proposal).detach()
        low=constraints['lower'].to(proposal).detach();high=constraints['upper'].to(proposal).detach()
        weight=constraints['weight'].to(proposal).detach()
        if j.shape[-1]!=52 or low.shape!=high.shape or low.shape!=weight.shape or j.shape[-2]!=low.shape[-1]:
            raise ValueError('Invalid calibrated constraint contract')
        if (low>high).any() or (weight<0).any():raise ValueError('Invalid constraint intervals/weight')
        norm=j.square().sum(-1).clamp_min(1e-6)
        y=proposal
        for _ in range(self.steps):
            value=torch.einsum('...kc,...c->...k',j,y)
            error=value-value.clamp(min=low,max=high)
            factor=(self.strength*weight)/(1+self.strength*weight)
            correction=torch.einsum('...k,...kc->...c',error*factor/norm,j)/j.shape[-2]
            y=y-correction
        return y


class JointResidualDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        dim=cfg.content_dim+52+cfg.state_dim+cfg.classes+cfg.style_dim+cfg.event_groups*3
        self.encoder=TemporalEncoder(dim,cfg.decoder_hidden,4,attention=1)
        self.output=nn.Linear(cfg.decoder_hidden,52)
        nn.init.normal_(self.output.weight,std=.005);nn.init.zeros_(self.output.bias)
    def forward(self,content,base,state,emotion,style,events,valid,scales):
        z=torch.cat((content,base/scales,state,emotion[:,None].expand(-1,len(base[0]),-1),
            style[:,None].expand(-1,len(base[0]),-1),events),-1)
        return clean(self.output(self.encoder(z,valid))*scales,valid)


class UnifiedExpressionModel(nn.Module):
    def __init__(self,cfg,feature_mean,feature_std,scales):
        super().__init__();self.cfg=cfg
        if cfg.state_dim!=10 or cfg.event_groups!=4:raise ValueError('Anchored schema requires 10 states / 4 events')
        self.register_buffer('feature_mean',feature_mean.float());self.register_buffer('feature_std',feature_std.float().clamp_min(1e-3))
        self.register_buffer('scales',scales.float().clamp_min(.02))
        self.register_buffer('guidance_weights',torch.ones(cfg.state_dim))
        self.teacher=MotionStateTeacher(cfg);self.student=AudioStateStudent(cfg)
        self.style=SingleReferenceStyle(cfg);self.events=ConditionalEventStudent(cfg)
        self.decoder=JointResidualDecoder(cfg);self.composer=FeasibleResidualComposer()
    def encode_style(self,refs):
        return self.style(refs['motion'],refs['valid'],refs['channel_mask'],self.scales)
    def audio_state(self,audio,valid):
        if audio.shape[-1]==1540:audio=audio[...,768:]
        if audio.shape[-1]!=self.cfg.audio_dim:raise ValueError('Student audio width differs')
        return self.student(clean((audio-self.feature_mean)/self.feature_std,valid),valid)
    def decode(self,content,base,valid,state,emotion,style,events,constraints=None):
        residual=self.decoder(content,base,state,emotion,style,events,valid,self.scales)
        proposal=base+residual
        return {'motion':clean(self.composer(proposal,constraints),valid),'proposal':clean(proposal,valid),'residual':residual}
    def training_forward(self,batch,refs,targets,*,teacher_state=False,constraints=None):
        style=self.encode_style(refs);student=self.audio_state(batch['audio_features'],batch['valid'])
        teacher=self.teacher(batch['motion'],batch['b0'],batch['valid'],batch['channel_mask'],self.scales)
        state=teacher['state'] if teacher_state else student['state']
        logits=teacher['emotion_logits'] if teacher_state else student['emotion_logits']
        schedule=self.events(student['hidden'],student['state'],style,batch['valid'],targets['events'],mode='observed',times=batch['times'])
        decoded=self.decode(batch['h0'],batch['b0'],batch['valid'],state,logits.softmax(-1),style,targets['events'],constraints)
        return {**decoded,'student':student,'teacher':teacher,'event_student':schedule,'style':style}
    def predict(self,audio,content,base,valid,refs,*,times=None,seed=47,mode='sample',emotion=None,constraints=None):
        # No teacher/GT inputs occur here; conditional events are an actual rollout.
        style=self.encode_style(refs);student=self.audio_state(audio,valid)
        generator=torch.Generator(device=audio.device).manual_seed(seed)
        schedule=self.events(student['hidden'],student['state'],style,valid,generator=generator,mode=mode,times=times)
        probs=student['emotion_logits'].softmax(-1) if emotion is None else emotion
        decoded=self.decode(content,base,valid,student['state'],probs,style,schedule['events'],constraints)
        return {**decoded,'student':student,'style':style,'schedule':schedule}
    def checkpoint_config(self):return asdict(self.cfg)


def masked_mean(value,mask):
    return torch.where(mask,value,0.).sum()/mask.sum().clamp_min(1)


def semantic_loss(logits,labels):
    use=labels.ge(0)&labels.lt(logits.shape[-1])
    return F.cross_entropy(logits[use],labels[use]) if use.any() else logits.sum()*0.


def supervision_losses(model,output,targets,batch,*,teacher_warmup=False):
    valid=batch['valid'];student=output['student'];teacher=output['teacher'];event=output['event_student']
    quality=targets['quality'].bool()&valid[...,None]
    anchored=masked_mean((teacher['state']-targets['state']).square(),quality)
    reliability=targets['quality']*model.guidance_weights
    guidance=((student['state']-teacher['state'].detach()).square()*reliability).sum()/reliability.sum().clamp_min(1)
    # Direct anchors stop a poorly trained teacher from becoming the sole target.
    direct=((student['state']-targets['state']).square()*reliability).sum()/reliability.sum().clamp_min(1)
    onset=F.binary_cross_entropy_with_logits(event['hazard_logits'],targets['onset'].float(),reduction='none')
    hazard=masked_mean(onset,targets['onset_valid'])
    duration=F.cross_entropy(event['duration_logits'].reshape(-1,event['duration_logits'].shape[-1]),targets['duration'].flatten(),reduction='none').reshape_as(targets['duration'])
    duration=masked_mean(duration,targets['duration_valid'])
    params=event['amplitude_params'];logamp=targets['amplitude'].clamp_min(.005).log()
    logstd=params[...,1].clamp(-4,0)
    nll=.5*((logamp-params[...,0])/logstd.exp()).square()+logstd
    amplitude=masked_mean(nll,targets['duration_valid'])
    emotion=semantic_loss(student['emotion_logits'],batch['emotion_id'])
    teacher_emotion=semantic_loss(teacher['emotion_logits'],batch['emotion_id'])
    intensity_labels=torch.where(batch['intensity_valid'],batch['intensity_id'],torch.full_like(batch['intensity_id'],-1))
    intensity=semantic_loss(student['intensity_logits'],intensity_labels)
    losses={'anchor':anchored,'guidance':guidance,'state_direct':direct,'hazard':hazard,'duration':duration,
        'amplitude':amplitude,'emotion':emotion,'teacher_emotion':teacher_emotion,'intensity':intensity}
    # These coefficients are pilot defaults; validation selects no final model.
    losses['total']=anchored+teacher_emotion if teacher_warmup else .2*anchored+.2*teacher_emotion+.5*guidance+.5*direct+hazard+.2*duration+.02*amplitude+emotion+.2*intensity
    return losses


def relation_style_loss(codes,speakers,temperature=.15):
    """Supervised contrastive relation; no speaker embedding lookup at inference."""
    z=F.normalize(codes,dim=-1);similarity=z@z.T/temperature
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive=speakers[:,None].eq(speakers[None])&~eye
    eligible=positive.any(1)
    if not eligible.any():return codes.sum()*0
    logprob=similarity-similarity.masked_fill(eye,-torch.inf).logsumexp(1,keepdim=True)
    return -(torch.where(positive,logprob,0.).sum(1)/positive.sum(1).clamp_min(1))[eligible].mean()
