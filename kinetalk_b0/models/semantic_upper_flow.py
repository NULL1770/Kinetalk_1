"""Small conditional rectified flow for joint native upper-face trajectories.

Semantic inputs are already expanded onto the native clock by the caller.
They are learned feature conditions, not motion coefficients or fixed action
directions. This module neither derives conditions from targets nor performs
normalization, centering, sigmoid, clipping, or full-face composition. The
runner owns train-fitted logit residual statistics and bounded reconstruction.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _valid_mask(valid):
    if (not torch.is_tensor(valid) or valid.ndim != 2 or valid.dtype != torch.bool
            or min(valid.shape) < 1 or not valid.any(1).all()):
        raise ValueError('valid must be nonempty Boolean [B,T] with observations in every row')


def _sequence(value, valid, width, name, dtype=None):
    if (not torch.is_tensor(value) or value.shape != (*valid.shape, width)
            or not value.is_floating_point() or value.device != valid.device
            or (dtype is not None and value.dtype != dtype)):
        raise ValueError(f'{name} must be matching floating [B,T,{width}]')
    if not torch.isfinite(value[valid]).all():
        raise ValueError(f'observed {name} must be finite')
    return torch.where(valid[..., None], value, 0.)


def _sinusoidal(values, width):
    frequencies = torch.exp(torch.arange(0, width, 2, device=values.device, dtype=values.dtype)
                            * (-math.log(10000.) / width))
    angles = values[..., None] * frequencies
    output = values.new_zeros(*values.shape, width)
    output[..., 0::2] = angles.sin()
    output[..., 1::2] = angles[..., :width // 2].cos()
    return output


def semantic_intervention(semantic, valid, mode='real'):
    """Real/static/reverse semantic conditions, independently within valid runs.

    Static means the observed semantic mean in that run. No condition is
    transported across a missing interval. This helper is diagnostic; a
    matched static-training control remains the runner's responsibility.
    """
    _valid_mask(valid)
    if not torch.is_tensor(semantic) or semantic.ndim != 3:
        raise ValueError('semantic must be floating [B,T,D]')
    clean = _sequence(semantic, valid, semantic.shape[-1], 'semantic')
    if mode == 'real':
        return clean
    if mode not in ('static', 'reverse'):
        raise ValueError('semantic intervention must be real, static, or reverse')
    output = clean.clone()
    for row in range(len(valid)):
        boundaries = torch.diff(F.pad(valid[row].to(torch.int8), (1, 1)))
        starts = torch.nonzero(boundaries == 1).flatten().tolist()
        stops = torch.nonzero(boundaries == -1).flatten().tolist()
        for start, stop in zip(starts, stops):
            segment = clean[row, start:stop]
            output[row, start:stop] = segment.mean(0, keepdim=True) if mode == 'static' else segment.flip(0)
    return output


class _TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(hidden)
        self.modulation = nn.Linear(hidden, 2 * hidden)
        self.semantic = nn.Linear(hidden, hidden, bias=False)
        self.temporal = nn.Linear(3 * hidden, hidden)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, x, semantic, context, valid, run_ids):
        shift, scale = self.modulation(F.silu(context)).chunk(2, -1)
        h = self.norm(x) * (1. + scale[:, None]) + shift[:, None] + self.semantic(semantic)
        h = torch.where(valid[..., None], F.silu(h), 0.)
        d, frames = self.dilation, x.shape[1]
        if d >= frames:
            left, right = torch.zeros_like(h), torch.zeros_like(h)
        else:
            # Dilated taps must not skip across a missing interval. Adjacent
            # observations on opposite sides of a gap are separate runs.
            connected = (valid[:, d:] & valid[:, :-d] & (run_ids[:, d:] == run_ids[:, :-d]))[..., None]
            left = F.pad(torch.where(connected, h[:, :-d], 0.), (0, 0, d, 0))
            right = F.pad(torch.where(connected, h[:, d:], 0.), (0, 0, 0, d))
        update = self.output(self.temporal(torch.cat((left, h, right), -1)))
        return torch.where(valid[..., None], x + update, 0.)


class SemanticUpperFlow(nn.Module):
    """Joint conditional flow with four 64-channel temporal blocks by default.

    Targets and results are unconstrained normalized logit residuals. White
    noise covers the full [B,T,motion_dim] trajectory; temporal mixing learns
    its joint distribution instead of decoding independent per-frame heads.
    Native frame position and observed length are supplied explicitly. There
    is no P/Q projection, geometric lift, target history, or oracle condition.

    Explicit noise permits paired interventions. Alternatively decode(seed=)
    uses a private generator and leaves the process-wide RNG untouched.
    Reproducibility is for the same model/device/runtime, batch order and
    masks; cross-platform bitwise equality is not promised.
    """
    def __init__(self, semantic_dim, global_dim=65, identity_dim=128,
                 motion_dim=9, hidden=64, depth=4):
        super().__init__()
        values = (semantic_dim, global_dim, identity_dim, motion_dim, hidden, depth)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError('all dimensions and depth must be positive integers')
        self.semantic_dim, self.global_dim, self.identity_dim = semantic_dim, global_dim, identity_dim
        self.motion_dim, self.hidden = motion_dim, hidden
        self.motion_projection = nn.Linear(motion_dim, hidden)
        self.semantic_projection = nn.Linear(semantic_dim, hidden)
        self.global_projection = nn.Linear(global_dim, hidden)
        self.identity_projection = nn.Linear(identity_dim, hidden)
        self.length_projection = nn.Linear(1, hidden)
        self.time_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(_TemporalBlock(hidden, 2 ** layer) for layer in range(depth))
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, motion_dim)
        # Nonzero initialization gives every condition a measurable first-step
        # gradient without making a random initial velocity dominate the noise.
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)

    def prepare_conditions(self, valid, semantic, global_condition, identity):
        _valid_mask(valid)
        semantic = _sequence(semantic, valid, self.semantic_dim, 'semantic')
        for name, value, width in (('global_condition', global_condition, self.global_dim),
                                   ('identity', identity, self.identity_dim)):
            if (not torch.is_tensor(value) or value.shape != (len(valid), width)
                    or not value.is_floating_point() or value.device != valid.device
                    or value.dtype != semantic.dtype or not torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite matching [B,{width}]')
        starts = valid & ~F.pad(valid[:, :-1], (1, 0), value=False)
        return {'valid': valid, 'semantic': semantic, 'global': global_condition,
                'identity': identity, 'run_ids': starts.to(torch.int64).cumsum(1)}

    @staticmethod
    def _time(time, valid, dtype):
        if (not torch.is_tensor(time) or time.shape != (len(valid),)
                or not time.is_floating_point() or time.dtype != dtype
                or time.device != valid.device or not torch.isfinite(time).all()
                or ((time < 0.) | (time > 1.)).any()):
            raise ValueError('time must be finite matching [B] in [0,1]')

    def velocity(self, noisy_motion, time, conditions):
        valid, semantic = conditions['valid'], conditions['semantic']
        x = _sequence(noisy_motion, valid, self.motion_dim, 'noisy_motion', semantic.dtype)
        self._time(time, valid, x.dtype)
        frame = torch.where(valid[..., None], self.semantic_projection(semantic), 0.)
        positions = _sinusoidal(torch.arange(x.shape[1], device=x.device, dtype=x.dtype), self.hidden)
        tokens = torch.where(valid[..., None], self.motion_projection(x) + frame + positions[None], 0.)
        length = valid.sum(1).to(x.dtype).log1p()[:, None]
        context = (self.global_projection(conditions['global']) + self.identity_projection(conditions['identity'])
                   + self.length_projection(length) + self.time_projection(_sinusoidal(time * 1000., self.hidden)))
        for block in self.blocks:
            tokens = block(tokens, frame, context, valid, conditions['run_ids'])
        return torch.where(valid[..., None], self.output(self.output_norm(tokens)), 0.)

    def flow_loss(self, target, valid, semantic, global_condition, identity, noise, time, loss_mask=None):
        """Single FM loss; optional supervision support is separate from time support.

        valid defines the acoustic/native sequence processed by the field.
        loss_mask may restrict loss entries without changing native positions,
        semantic context, flow interpolation or length conditioning.
        """
        conditions = self.prepare_conditions(valid, semantic, global_condition, identity)
        if loss_mask is None:
            loss_mask = valid
        if (not torch.is_tensor(loss_mask) or loss_mask.dtype != torch.bool
                or loss_mask.shape != valid.shape or loss_mask.device != valid.device
                or (loss_mask & ~valid).any() or not loss_mask.any()):
            raise ValueError('loss_mask must be nonempty Boolean support within valid')
        target = _sequence(target, valid, self.motion_dim, 'target', semantic.dtype)
        noise = _sequence(noise, valid, self.motion_dim, 'noise', semantic.dtype)
        self._time(time, valid, semantic.dtype)
        fraction = time[:, None, None]
        velocity = self.velocity((1. - fraction) * noise + fraction * target, time, conditions)
        error = torch.where(loss_mask[..., None], velocity - (target - noise), 0.)
        return error.square().sum() / (loss_mask.sum() * self.motion_dim)

    def decode(self, valid, semantic, global_condition, identity, noise=None, *, steps=12, seed=None):
        """Euler integration; supply exactly one of explicit noise or seed.

        Seeded noise draws only observed entries, so appending invalid padding
        does not alter the random draws assigned to any batch row. Output is
        zero outside valid. Caller chooses inference_mode/no_grad as needed.
        """
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        if (noise is None) == (seed is None):
            raise ValueError('supply exactly one of noise or seed')
        conditions = self.prepare_conditions(valid, semantic, global_condition, identity)
        if noise is None:
            if type(seed) is not int:
                raise ValueError('seed must be an integer')
            generator = torch.Generator(device=semantic.device).manual_seed(seed)
            noise = semantic.new_zeros(*valid.shape, self.motion_dim)
            for row in range(len(valid)):
                count = int(valid[row].sum())
                noise[row, valid[row]] = torch.randn(count, self.motion_dim, device=semantic.device,
                                                     dtype=semantic.dtype, generator=generator)
        x = _sequence(noise, valid, self.motion_dim, 'noise', semantic.dtype)
        for step in range(steps):
            time = x.new_full((len(x),), step / steps)
            x = torch.where(valid[..., None], x + self.velocity(x, time, conditions) / steps, 0.)
        return x


__all__ = ['SemanticUpperFlow', 'semantic_intervention']
