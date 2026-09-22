"""Audio mean plus unrestricted conditional residual flow for upper motion.

Unlike the earlier Q-projected innovation, this residual can represent slow
state uncertainty and correct the deterministic audio mean. Random outputs are
trained with a two-draw fair energy score, not independent endpoint MSE.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .slow_state_affect import UpperInnovationFlow, UPPER_INDICES, lift_slow_state


def static_audio(features, valid):
    mask=valid[...,None]
    mean=torch.where(mask,features,0.).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)
    return torch.where(mask,mean,0.)


def fair_trajectory_es(samples, target, valid, *, centered=False):
    """Unbiased finite-ensemble energy score, equal weight per input clip.

    Norms use only observed native frames. No reference enters generation.
    Centered scoring removes a per-clip/channel offset for a timing diagnostic;
    the raw score must also be kept to prevent unpenalized mean drift.
    """
    if samples.ndim!=4 or samples.shape[1:]!=target.shape or len(samples)<2:
        raise ValueError('At least two samples [S,B,T,D] matching target required')
    if valid.dtype!=torch.bool or valid.shape!=target.shape[:2] or not valid.any(1).all():
        raise ValueError('Nonempty Boolean native frame masks required')
    mask=valid[...,None];count=mask.sum((1,2))*target.shape[-1]
    x=torch.where(mask[None],samples,0.);y=torch.where(mask,target,0.)
    if centered:
        x=torch.where(mask[None],x-x.sum(2,keepdim=True)/mask.sum(1,keepdim=True)[None],0.)
        y=torch.where(mask,y-y.sum(1,keepdim=True)/mask.sum(1,keepdim=True),0.)
    def distance(delta):
        return torch.linalg.vector_norm(delta.flatten(-2),dim=-1)/count.sqrt()
    first=distance(x-y[None]).mean(0)
    spread=sum(distance(x[i]-x[j]) for i in range(len(x)) for j in range(i))
    return (first-spread/(len(x)*(len(x)-1))).mean()


class AudioResidualFlow(UpperInnovationFlow):
    """No Q projection: slow and fast conditional variation are both learned."""
    def flow_loss(self,target,q,identity,affect,local,state,noise,time):
        self._check_motion(target,q['valid'],'target')
        self._check_motion(noise,q['valid'],'noise')
        if (time.shape != (len(target),) or not time.is_floating_point() or time.device != target.device
                or not torch.isfinite(time).all() or ((time<0)|(time>1)).any()):
            raise ValueError('Flow time must be finite [B] in [0,1]')
        # This model's support is fixed to all nine coefficients. Partially
        # observed training targets are not silently reinterpreted as zero.
        if 'channel_mask' in q:
            channels=q['channel_mask']
            if (channels.dtype != torch.bool or channels.shape != (len(target),52)
                    or not channels[:,list(UPPER_INDICES)].all()):
                raise ValueError('Upper flow training requires all nine observed upper channels')
        conditions=self._conditions(q['valid'],q['h0'],identity['code'],affect,local,state)
        mask=q['valid'][...,None];a=time[:,None,None]
        target=torch.where(mask,target,0.);noise=torch.where(mask,noise,0.)
        x=(1-a)*noise+a*target
        velocity=self._velocity_unrestricted(x,time,q['valid'],conditions)
        return torch.where(mask,velocity-(target-noise),0.).square().sum()/(mask.sum()*9)

    @staticmethod
    def _check_motion(value,valid,name):
        if (not torch.is_tensor(value) or value.shape != (*valid.shape,9)
                or not value.is_floating_point() or value.device != valid.device
                or not torch.isfinite(value[valid]).all()):
            raise ValueError(f'Upper {name} must be finite observed [B,T,9]')

    def _velocity_unrestricted(self,x,time,valid,conditions):
        content,identity,global_code,intensity,local=conditions
        velocity=self.renderer(x,time,content,global_code,intensity,identity,valid,
                                local_emotion=local,condition_dropout=False)
        return torch.where(valid[...,None],velocity,0.)

    def decode(self,q,identity,affect,local,state,noise,steps,*,checkpoint_steps=True):
        """Euler rollout with recomputed activations during gradient training.

        The two-draw energy score retains two complete solver trajectories.
        Checkpoint each velocity evaluation so its transformer activations
        are recomputed in backward instead of retained for all solver steps.
        Non-reentrant checkpointing also propagates condition/parameter
        gradients when the first noise tensor itself requires no gradient.
        Inference uses the original direct path; ``checkpoint_steps=False``
        is available for numerical and gradient equivalence checks.
        """
        if type(steps) is not int or steps<1:raise ValueError('Positive integer decode budget required')
        self._check_motion(noise,q['valid'],'noise')
        conditions=self._conditions(q['valid'],q['h0'],identity['code'],affect,local,state)
        x=torch.where(q['valid'][...,None],noise,0.)
        for i in range(steps):
            t=x.new_full((len(x),),i/steps)
            if checkpoint_steps and torch.is_grad_enabled():
                velocity=checkpoint(self._velocity_unrestricted,x,t,q['valid'],conditions,
                                    use_reentrant=False,preserve_rng_state=True)
            else:
                velocity=self._velocity_unrestricted(x,t,q['valid'],conditions)
            x=x+velocity/steps
            x=torch.where(q['valid'][...,None],x,0.)
        return x


def normalized_mean(state):
    return lift_slow_state(state,state.new_ones(52))[...,list(UPPER_INDICES)]


def compose_prediction(q,scales,state,residual):
    cc=list(UPPER_INDICES)
    return q['anchors'][:,None,cc]+scales[cc]*(normalized_mean(state)+residual)
