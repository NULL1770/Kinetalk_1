"""Sustained-expression supervision for stochastic event-conditioned rollouts.

Only clip/run population moments are compared. No sampled event is treated as
having one uniquely correct GT onset. Unobserved channels and clock gaps stay
masked; this is a motion statistic, not psychological emotion ground truth.
"""
import torch
from torch.nn import functional as F


def expression_mean_loss(prediction, target, valid, channels, scales):
    if prediction.shape != target.shape or prediction.shape[:2] != valid.shape:
        raise ValueError('Aligned native motion and valid mask required')
    if channels.shape != (len(prediction), prediction.shape[-1]):
        raise ValueError('Observed channel support required')
    observed = valid[..., None] & channels[:, None].bool()
    count = observed.sum(1)
    p = torch.where(observed, prediction, 0.).sum(1) / count.clamp_min(1)
    q = torch.where(observed, target, 0.).sum(1) / count.clamp_min(1)
    loss = F.smooth_l1_loss(p / scales, q / scales, reduction='none')
    return torch.where(count > 0, loss, 0.).sum() / (count > 0).sum().clamp_min(1)


def pooled_state_loss(prediction, target, valid, quality):
    observed = valid[..., None] & quality.bool()
    count = observed.sum(1)
    p = torch.where(observed, prediction, 0.).sum(1) / count.clamp_min(1)
    q = torch.where(observed, target, 0.).sum(1) / count.clamp_min(1)
    value = (p - q).square()
    return torch.where(count > 0, value, 0.).sum() / (count > 0).sum().clamp_min(1)
