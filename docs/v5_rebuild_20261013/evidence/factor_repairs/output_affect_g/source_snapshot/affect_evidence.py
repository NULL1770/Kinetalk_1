"""Real TRAIN motion supervision, separate from frozen evaluation probes."""
import torch
from torch import nn
from torch.nn import functional as F
from kinetalk_b0.emotion_probe import motion_features


def observed_statistics(motion, valid, channels):
    y=torch.where(valid[...,None]&channels[:,None],motion,0.)
    return torch.stack([motion_features(x[:,:51].clamp(0,1),v) for x,v in zip(y,valid)])


class AffectEvidence(nn.Module):
    """Clip-level expression recognition; no psychological frame annotations."""
    def __init__(self, classes=8, hidden=64):
        super().__init__()
        self.register_buffer('mean',torch.zeros(306))
        self.register_buffer('scale',torch.ones(306))
        self.network=nn.Sequential(nn.Linear(306,hidden),nn.SiLU(),nn.Linear(hidden,hidden),
                                   nn.SiLU(),nn.Linear(hidden,classes))

    @torch.no_grad()
    def fit_scales(self, features):
        floors=torch.full_like(self.scale,.04);floors[204:]=.015
        self.mean.copy_(features.mean(0))
        self.scale.copy_(torch.maximum(features.std(0,correction=0),floors))

    def features(self, features):
        return self.network((features-self.mean)/self.scale)

    def forward(self, motion, valid, channels):
        return self.features(observed_statistics(motion,valid,channels))


def affect_loss(evidence, motion, batch):
    known=batch['dataset_id'].eq(0)&batch['emotion_id'].ge(0)
    if not known.any():return motion.sum()*0.
    logits=evidence(motion[known],batch['valid'][known],batch['channel_mask'][known])
    return F.cross_entropy(logits,batch['emotion_id'][known])


def query_moment_loss(motion, batch, scales):
    """Actual query moments, never a swapped-performance pointwise target."""
    pred=[];real=[]
    for y,g,v in zip(motion,batch['motion'],batch['valid']):
        if not v.any():raise ValueError('empty query not supported')
        pred.append(torch.stack((y[v].mean(0),y[v].quantile(.1,dim=0),y[v].quantile(.9,dim=0))))
        real.append(torch.stack((g[v].mean(0),g[v].quantile(.1,dim=0),g[v].quantile(.9,dim=0))))
    error=F.smooth_l1_loss(torch.stack(pred)/scales.clamp_min(.04),
                         torch.stack(real).detach()/scales.clamp_min(.04),reduction='none')
    mask=batch['channel_mask'][:,None].expand_as(error)
    return error[mask].mean()
