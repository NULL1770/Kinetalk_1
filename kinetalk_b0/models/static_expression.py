"""A clip-constant response correction from frozen expression and reference cues.

This receiver-side map never sees query motion at inference. Its supervised
target is a TRAIN clip's mean reconstruction error, not an emotion probe score.
It cannot alter raw centered trajectories or adjacent displacements.
"""
from __future__ import annotations

import torch
from torch import nn


def neutral_reference_mean(refs, scales):
    motion, base = refs['motion'], refs['b0']
    valid, channels = refs['valid'], refs['channel_mask']
    if motion.ndim != 4 or base.shape != motion.shape:
        raise ValueError('Expected independent reference sequences [B,R,T,C]')
    if valid.shape != motion.shape[:3] or channels.shape != (*motion.shape[:2], motion.shape[-1]):
        raise ValueError('Reference support shapes differ')
    if valid.dtype != torch.bool or channels.dtype != torch.bool:
        raise ValueError('Boolean observed reference support required')
    observed = valid[..., None] & channels[:, :, None]
    delta = torch.where(observed, motion-base, 0.) / scales
    if not torch.isfinite(delta).all():
        raise ValueError('Nonfinite observed reference')
    count = observed.sum(2)
    each = delta.sum(2)/count.clamp_min(1)
    support = count.gt(0)
    # Equal weight per observed reference, independently for each channel.
    return (each.sum(1)/support.sum(1).clamp_min(1)).detach()


def receiver_features(model, prior, style, refs, mode):
    g, style = prior['g_mean'].detach(), style.detach()
    if mode == 'latent':
        return torch.cat((g, style), -1)
    if mode != 'reference':
        raise ValueError('Expected latent or reference receiver features')
    neutral = neutral_reference_mean(refs, model.scales)
    probability = model.emotion_head(g).softmax(-1).detach()
    interaction = (probability[..., None]*neutral[:, None]).flatten(1)
    return torch.cat((g, style, neutral, interaction), -1)


class StaticExpressionCorrection(nn.Module):
    def __init__(self, mode, mean, scale, weight, bias):
        super().__init__()
        if mode not in ('latent', 'reference'):
            raise ValueError('Unknown static response features')
        mean, scale, weight, bias = [torch.as_tensor(x).float() for x in (mean, scale, weight, bias)]
        if (mean.ndim != 1 or mean.shape != scale.shape or weight.shape != (len(mean), 52)
                or bias.shape != (52,) or (scale <= 0).any()
                or not all(torch.isfinite(x).all() for x in (mean, scale, weight, bias))):
            raise ValueError('Invalid static response parameters')
        self.mode = mode
        for name, value in [('mean', mean), ('scale', scale), ('weight', weight), ('bias', bias)]:
            self.register_buffer(name, value)

    def offset(self, features):
        if features.ndim != 2 or features.shape[-1] != len(self.mean) or not torch.isfinite(features).all():
            raise ValueError('Invalid deployment features')
        return (features-self.mean)/self.scale @ self.weight + self.bias

    def forward(self, prediction, features, valid, scales):
        offset = self.offset(features).to(prediction)*scales
        return torch.where(valid[..., None], prediction+offset[:, None], 0.)

    @torch.no_grad()
    def predict(self, model, audio, base, valid, refs):
        prior = model.audio_prior(audio, valid)
        style = model.encode_style(refs)['code']
        prediction = model.decode(base, prior, style, valid)
        return self(prediction, receiver_features(model, prior, style, refs, self.mode), valid, model.scales)


def fit_static_correction(features, target, support, mode, strength=1e-3):
    """One fixed ridge solve per observed output; every TRAIN clip weighs one."""
    x, y = features.detach().double(), target.detach().double()
    if (x.ndim != 2 or y.shape != (len(x), 52) or support.shape != y.shape
            or support.dtype != torch.bool or len(x) < 2 or strength <= 0
            or not torch.isfinite(x).all() or not torch.isfinite(y[support]).all()):
        raise ValueError('Invalid fit data')
    mean = x.mean(0)
    scale = x.std(0, correction=0).clamp_min(1e-4)
    x = (x-mean)/scale
    weight = x.new_zeros(x.shape[1], 52)
    bias = x.new_zeros(52)
    # Most ARKit channels share support; solve them together, not 52 times.
    patterns, inverse = torch.unique(support.T, dim=0, return_inverse=True)
    for i, observed in enumerate(patterns):
        columns = (inverse == i).nonzero(as_tuple=True)[0]
        if not observed.any():
            continue
        xx, yy = x[observed], y[observed][:, columns]
        xm, ym = xx.mean(0), yy.mean(0)
        xx, yy = xx-xm, yy-ym
        normal = xx.T@xx/len(xx) + strength*torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        w = torch.linalg.solve(normal, xx.T@yy/len(xx))
        weight[:, columns] = w
        bias[columns] = ym-xm@w
    return StaticExpressionCorrection(mode, mean, scale, weight, bias)
