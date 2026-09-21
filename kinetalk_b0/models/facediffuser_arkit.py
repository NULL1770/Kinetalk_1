"""Masked cached-audio adaptation of the official BEAT FaceDiffuser decoder.

Adapted from FaceDiffBeat by uuembodiedsocialai/FaceDiffuser, commit
e15f3500fdae0eda962f5d018488dfa0a1a9d552, under CC BY-NC 4.0.
Source/license/provenance: third_party/facediffuser/{models.py,LICENSE,provenance.json}.
This derivative retains that license and is provided without warranties.

Changes: externally cached native-rate audio, explicit fixed ARKit support,
observation masks, and optional declared clip conditions. This is not an
official BEAT reproduction: the original fine-tunes HuBERT and pairs 50 Hz
embeddings, whereas the project's cached 768D content is already at 25 Hz.
The DDPM predicts x0 with the official cosine schedule and fixed-small variance.
No mouth bypass, target centering, coefficient normalization, or clipping occurs.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SOURCE_COMMIT = 'e15f3500fdae0eda962f5d018488dfa0a1a9d552'


def cosine_betas(steps):
    if type(steps) is not int or steps < 2:
        raise ValueError('diffusion_steps must be an integer >= 2')
    alpha_bar = lambda t: math.cos((t + .008) / 1.008 * math.pi / 2) ** 2
    return torch.tensor([min(1 - alpha_bar((i + 1) / steps) / alpha_bar(i / steps), .999)
                         for i in range(steps)], dtype=torch.float64)


class FaceDiffuserARKit(nn.Module):
    """Official BEAT GRU/x0 core, with raw ARKit52 input/output interfaces.

    The default fixed support omits tongueOut (index 51), leaving 51 trained
    channels. ``clip_condition_dim=0`` is audio-only. A positive dimension
    explicitly adds caller-supplied inference-available global/reference codes
    by concatenation; this is a separate matched-condition adaptation.
    """

    def __init__(self, *, content_dim=768, latent_dim=256, gru_hidden=256,
                 num_layers=2, diffusion_steps=1000, dropout=.3,
                 clip_condition_dim=0, support=None):
        super().__init__()
        for name, value in [('content_dim', content_dim), ('latent_dim', latent_dim),
                            ('gru_hidden', gru_hidden), ('num_layers', num_layers)]:
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if type(clip_condition_dim) is not int or clip_condition_dim < 0:
            raise ValueError('clip_condition_dim must be a nonnegative integer')
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        if support is None:
            support = torch.tensor([True] * 51 + [False])
        elif isinstance(support, (list, tuple)):
            if len(support) != 52 or any(type(v) is not bool for v in support):
                raise ValueError('support must be Boolean [52]')
            support = torch.tensor(support)
        if not torch.is_tensor(support) or support.dtype != torch.bool or support.shape != (52,) or not support.any():
            raise ValueError('support must be nonempty Boolean [52]')
        self.register_buffer('support', support.detach().cpu().clone())
        self.register_buffer('indices', self.support.nonzero().flatten())
        self.content_dim, self.latent_dim, self.gru_hidden = content_dim, latent_dim, gru_hidden
        self.num_layers, self.diffusion_steps = num_layers, diffusion_steps
        self.dropout, self.clip_condition_dim = float(dropout), clip_condition_dim
        self.motion_dim = int(self.support.sum())
        self.time_mlp = nn.Sequential(nn.Linear(diffusion_steps, latent_dim), nn.Mish())
        width = content_dim + self.motion_dim + latent_dim + clip_condition_dim
        self.norm_cond = nn.LayerNorm(width)
        self.gru = nn.GRU(width, gru_hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.)
        self.final_layer = nn.Linear(gru_hidden, self.motion_dim)
        betas = cosine_betas(diffusion_steps)
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, 0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), cumulative[:-1]))
        buffers = {'betas': betas, 'sqrt_alpha_bar': cumulative.sqrt(),
                   'sqrt_one_minus_alpha_bar': (1 - cumulative).sqrt(),
                   'posterior_variance': betas * (1 - previous) / (1 - cumulative),
                   'posterior_mean_coef1': betas * previous.sqrt() / (1 - cumulative),
                   'posterior_mean_coef2': (1 - previous) * alphas.sqrt() / (1 - cumulative)}
        for name, value in buffers.items():
            self.register_buffer(name, value)

    def export_config(self):
        return {'content_dim': self.content_dim, 'latent_dim': self.latent_dim,
                'gru_hidden': self.gru_hidden, 'num_layers': self.num_layers,
                'diffusion_steps': self.diffusion_steps, 'dropout': self.dropout,
                'clip_condition_dim': self.clip_condition_dim, 'support': self.support.tolist()}

    def _observed(self, value, valid, channel_mask=None):
        if (not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != 52
                or min(value.shape) < 1 or not value.is_floating_point()
                or value.device != self.support.device
                or value.dtype != self.final_layer.weight.dtype):
            raise ValueError('motion must be nonempty floating [B,T,52] matching model dtype/device')
        if (not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != value.shape[:2]
                or valid.device != value.device or not valid.any(1).all()):
            raise ValueError('valid must be Boolean [B,T], nonempty for every clip')
        observed = valid[..., None] & self.support[None, None]
        if channel_mask is not None:
            if (not torch.is_tensor(channel_mask) or channel_mask.dtype != torch.bool
                    or channel_mask.device != value.device
                    or channel_mask.shape not in ((len(value), 52), value.shape)):
                raise ValueError('channel_mask must be Boolean [B,52] or [B,T,52]')
            observed = observed & (channel_mask[:, None] if channel_mask.ndim == 2 else channel_mask)
        if not observed.flatten(1).any(1).all() or not torch.isfinite(value[observed]).all():
            raise ValueError('Each clip requires finite observed supported motion')
        return observed

    def _times(self, times, batch, device):
        if (not torch.is_tensor(times) or times.dtype != torch.long or times.shape != (batch,)
                or times.device != device or (times < 0).any() or (times >= self.diffusion_steps).any()):
            raise ValueError('timesteps must be long [B] in the diffusion schedule')

    def _coefficient(self, name, times, reference):
        return getattr(self, name)[times].to(reference.dtype)[:, None, None]

    def forward(self, noised_motion, timesteps, content, valid, *, clip_condition=None, channel_mask=None):
        observed = self._observed(noised_motion, valid, channel_mask)
        self._times(timesteps, len(noised_motion), noised_motion.device)
        if (not torch.is_tensor(content) or content.shape != (*valid.shape, self.content_dim)
                or content.dtype != noised_motion.dtype or content.device != noised_motion.device
                or not torch.isfinite(content[valid]).all()):
            raise ValueError('content must be finite observed native audio [B,T,content_dim]')
        if self.clip_condition_dim:
            if (not torch.is_tensor(clip_condition)
                    or clip_condition.shape != (len(content), self.clip_condition_dim)
                    or clip_condition.dtype != content.dtype or clip_condition.device != content.device
                    or not torch.isfinite(clip_condition).all()):
                raise ValueError('Declared clip_condition must be finite matching [B,clip_condition_dim]')
        elif clip_condition is not None:
            raise ValueError('Audio-only configuration cannot silently consume clip conditions')
        frames = valid.shape[1]
        end = int(valid.any(0).nonzero()[-1, 0]) + 1
        clean = torch.where(observed, noised_motion, 0.)[:, :end, self.indices]
        audio = torch.where(valid[..., None], content, 0.)[:, :end]
        time = self.time_mlp(F.one_hot(timesteps, self.diffusion_steps).to(content.dtype))
        tokens = [audio, clean, time[:, None].expand(-1, end, -1)]
        if clip_condition is not None:
            # Matched-condition comparisons use frozen source encoders. Their
            # gradients must not turn this baseline into another joint model.
            tokens.append(clip_condition.detach()[:, None].expand(-1, end, -1))
        inputs = self.norm_cond(torch.cat(tokens, -1))
        # Invalid native frames are zero tokens, not removed or joined. GRU
        # recurrence advances once at every original clock position.
        inputs = torch.where(valid[:, :end, None], inputs, 0.)
        hidden, _ = self.gru(inputs)
        prediction = self.final_layer(hidden)
        full = noised_motion.new_zeros(len(content), end, 52)
        full[..., self.indices] = prediction
        full = F.pad(full, (0, 0, 0, frames - end))
        return torch.where(observed, full, 0.)

    def q_sample(self, target, timesteps, valid, *, noise, channel_mask=None):
        observed = self._observed(target, valid, channel_mask)
        self._times(timesteps, len(target), target.device)
        if (not torch.is_tensor(noise) or noise.shape != target.shape or noise.dtype != target.dtype
                or noise.device != target.device or not torch.isfinite(noise[observed]).all()):
            raise ValueError('noise must match motion with finite observed values')
        clean = torch.where(observed, target, 0.)
        epsilon = torch.where(observed, noise, 0.)
        return self._coefficient('sqrt_alpha_bar', timesteps, target) * clean + self._coefficient(
            'sqrt_one_minus_alpha_bar', timesteps, target) * epsilon

    def training_losses(self, target, timesteps, content, valid, *, noise,
                        channel_mask=None, clip_condition=None):
        observed = self._observed(target, valid, channel_mask)
        x_t = self.q_sample(target, timesteps, valid, noise=noise, channel_mask=channel_mask)
        prediction = self(x_t, timesteps, content, valid, channel_mask=channel_mask, clip_condition=clip_condition)
        clean = torch.where(observed, target, 0.)
        squared = torch.where(observed, (prediction - clean).square(), 0.)
        # Equal-clip mean over observed native values; padding and missing
        # channels neither add targets nor dilute the loss.
        per_clip = squared.sum((1, 2)) / observed.sum((1, 2))
        return {'loss': per_clip, 'mse': per_clip, 'pred_xstart': prediction, 'x_t': x_t}

    def p_mean_variance(self, x_t, timesteps, content, valid, *, channel_mask=None, clip_condition=None):
        observed = self._observed(x_t, valid, channel_mask)
        predicted = self(x_t, timesteps, content, valid, channel_mask=channel_mask, clip_condition=clip_condition)
        clean = torch.where(observed, x_t, 0.)
        mean = self._coefficient('posterior_mean_coef1', timesteps, x_t) * predicted + self._coefficient(
            'posterior_mean_coef2', timesteps, x_t) * clean
        return {'mean': mean, 'variance': self._coefficient('posterior_variance', timesteps, x_t),
                'pred_xstart': predicted}

    def p_sample(self, x_t, timesteps, content, valid, *, noise, channel_mask=None, clip_condition=None):
        observed = self._observed(x_t, valid, channel_mask)
        if (not torch.is_tensor(noise) or noise.shape != x_t.shape or noise.dtype != x_t.dtype
                or noise.device != x_t.device or not torch.isfinite(noise[observed]).all()):
            raise ValueError('noise must match motion with finite observed values')
        result = self.p_mean_variance(x_t, timesteps, content, valid,
                                     channel_mask=channel_mask, clip_condition=clip_condition)
        sample = result['mean'] + (timesteps > 0)[:, None, None] * result['variance'].sqrt() * torch.where(observed, noise, 0.)
        return {**result, 'sample': torch.where(observed, sample, 0.)}

    @torch.no_grad()
    def sample(self, content, valid, *, initial_noise, generator, channel_mask=None, clip_condition=None):
        """Full ancestral DDPM, no timestep skipping or x0/output clipping."""
        if self.training:
            raise ValueError('Call eval() before sampling to disable GRU dropout')
        if not isinstance(generator, torch.Generator):
            raise ValueError('An explicit torch.Generator is required for ancestral noise')
        observed = self._observed(initial_noise, valid, channel_mask)
        value = torch.where(observed, initial_noise, 0.)
        for step in reversed(range(self.diffusion_steps)):
            times = torch.full((len(value),), step, dtype=torch.long, device=value.device)
            noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator) if step else torch.zeros_like(value)
            value = self.p_sample(value, times, content, valid, noise=noise,
                                  channel_mask=channel_mask, clip_condition=clip_condition)['sample']
        return value


__all__ = ['FaceDiffuserARKit', 'cosine_betas', 'SOURCE_COMMIT']
