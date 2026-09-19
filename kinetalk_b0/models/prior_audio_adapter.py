"""A bounded audio correction around an immutable continuous motion prior.

The bound limits the adapter's velocity contribution in normalized latent
coordinates. It is not a bound on the generated trajectory, and does not
guarantee natural motion or a minimum amount of sample diversity.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .continuous_upper_motion import (
    ContinuousLatentFlow,
    _MaskedTemporalBlock,
    _clean_sequence,
    _positive_dimensions,
    _prefix_mask,
    _sinusoidal,
)


class BoundedPriorAudioAdapter(nn.Module):
    """Freeze a supplied prior and train a zero-initialized audio residual.

    ``prior`` is wrapped by reference and its parameters are frozen in place.
    Native audio frames are projected and packed in order, including an exact
    partial-tail mask. The adapter has no target input, internal RNG or dropout.
    The caller owns input normalization, native-gap splitting and sample noise.
    """

    def __init__(self, prior: ContinuousLatentFlow, hidden=96, depth=3,
                 audio_hidden=32, max_delta=.35):
        super().__init__()
        if not isinstance(prior, ContinuousLatentFlow):
            raise TypeError('prior must be a ContinuousLatentFlow')
        _positive_dimensions(hidden=hidden, depth=depth, audio_hidden=audio_hidden)
        if (isinstance(max_delta, bool) or not isinstance(max_delta, (int, float))
                or not math.isfinite(max_delta) or max_delta <= 0):
            raise ValueError('max_delta must be a finite positive scalar')
        self.prior = prior
        self.prior.requires_grad_(False)
        self.prior.eval()
        self.context_dim, self.audio_dim = prior.context_dim, prior.audio_dim
        self.latent_dim, self.block_size = prior.latent_dim, prior.block_size
        self.hidden, self.max_delta = hidden, float(max_delta)
        self.config = dict(prior=dict(prior.config), hidden=hidden, depth=depth,
                           audio_hidden=audio_hidden, max_delta=float(max_delta))
        self.motion_projection = nn.Linear(self.latent_dim, hidden)
        self.context_projection = nn.Linear(self.context_dim, hidden)
        self.audio_frame_projection = nn.Sequential(nn.Linear(self.audio_dim, audio_hidden), nn.SiLU())
        self.audio_block_projection = nn.Linear(self.block_size * (audio_hidden + 1), hidden)
        self.time_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(_MaskedTemporalBlock(hidden, 2 ** i, conditional=True)
                                    for i in range(depth))
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, self.latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def train(self, mode=True):
        super().train(mode)
        # A parent .train() must never re-enable training behavior in the prior.
        self.prior.eval()
        return self

    def adapter_parameters(self):
        """Yield only the trainable correction parameters, never prior weights."""
        return (parameter for name, parameter in self.named_parameters()
                if not name.startswith('prior.'))

    def _audio(self, valid, context, audio_blocks, dtype, audio_frame_valid):
        if (not torch.is_tensor(context) or context.shape != (len(valid), self.context_dim)
                or context.device != valid.device or context.dtype != dtype
                or not torch.isfinite(context).all()):
            raise ValueError('context must be finite matching floating [B,context_dim]')
        if (not torch.is_tensor(audio_blocks)
                or audio_blocks.shape != (*valid.shape, self.block_size, self.audio_dim)
                or audio_blocks.device != valid.device or audio_blocks.dtype != dtype):
            raise ValueError('audio_blocks must be matching floating [B,L,block_size,audio_dim]')
        if audio_frame_valid is None:
            frame_valid = valid[..., None].expand(-1, -1, self.block_size)
        else:
            frame_valid = audio_frame_valid
            if (not torch.is_tensor(frame_valid) or frame_valid.dtype != torch.bool
                    or frame_valid.shape != (*valid.shape, self.block_size)
                    or frame_valid.device != valid.device):
                raise ValueError('audio_frame_valid must be matching Boolean [B,L,block_size]')
            _prefix_mask(frame_valid.reshape(len(valid), -1), 'audio_frame_valid')
            if not torch.equal(frame_valid.any(-1), valid):
                raise ValueError('audio_frame_valid support must agree with latent valid')
        if not torch.isfinite(audio_blocks[frame_valid]).all():
            raise ValueError('observed audio_blocks must be finite')
        clean = torch.where(frame_valid[..., None], audio_blocks, 0.)
        frame = self.audio_frame_projection(clean)
        frame = torch.where(frame_valid[..., None], frame, 0.)
        packed = torch.cat((frame.flatten(-2), frame_valid.to(dtype)), -1)
        return torch.where(valid[..., None], self.audio_block_projection(packed), 0.)

    def residual(self, noisy, time, valid, context, audio_blocks, *, audio_frame_valid=None):
        """Return a masked, per-coordinate bounded latent velocity correction."""
        _prefix_mask(valid)
        noisy = _clean_sequence(noisy, valid, self.latent_dim, 'noisy')
        if (not torch.is_tensor(time) or time.shape != (len(valid),)
                or time.dtype != noisy.dtype or time.device != valid.device
                or not torch.isfinite(time).all() or ((time < 0.) | (time > 1.)).any()):
            raise ValueError('time must be finite matching [B] in [0,1]')
        audio = self._audio(valid, context, audio_blocks, noisy.dtype, audio_frame_valid)
        positions = _sinusoidal(torch.arange(valid.shape[1], device=noisy.device,
                                              dtype=noisy.dtype), self.hidden)
        x = self.motion_projection(noisy) + audio + positions[None]
        x = torch.where(valid[..., None], x, 0.)
        condition = self.context_projection(context) + self.time_projection(_sinusoidal(time * 1000., self.hidden))
        for block in self.blocks:
            x = block(x, valid, condition)
        correction = self.max_delta * torch.tanh(self.output(self.output_norm(x)))
        return torch.where(valid[..., None], correction, 0.)

    def velocity(self, noisy, time, valid, context, audio_blocks, use_audio=True, *, audio_frame_valid=None):
        if type(use_audio) is not bool:
            raise ValueError('use_audio must be Boolean')
        base = self.prior.velocity(noisy, time, valid, context, audio_blocks,
                                   use_audio=False, audio_frame_valid=audio_frame_valid)
        if not use_audio:
            return base
        return base + self.residual(noisy, time, valid, context, audio_blocks,
                                    audio_frame_valid=audio_frame_valid)

    def flow_loss(self, target, valid, context, audio_blocks, noise, time,
                  use_audio=True, *, audio_frame_valid=None):
        if type(use_audio) is not bool:
            raise ValueError('use_audio must be Boolean')
        if not use_audio:
            return self.prior.flow_loss(target, valid, context, audio_blocks, noise, time,
                                        use_audio=False, audio_frame_valid=audio_frame_valid)
        _prefix_mask(valid)
        target = _clean_sequence(target, valid, self.latent_dim, 'target')
        noise = _clean_sequence(noise, valid, self.latent_dim, 'noise', target.dtype)
        if not torch.is_tensor(time) or time.shape != (len(valid),):
            raise ValueError('time must be matching [B]')
        fraction = time[:, None, None]
        velocity = self.velocity((1. - fraction) * noise + fraction * target, time, valid,
                                 context, audio_blocks, True, audio_frame_valid=audio_frame_valid)
        error = torch.where(valid[..., None], velocity - (target - noise), 0.)
        return error.square().sum() / (valid.sum() * self.latent_dim)

    def sample(self, valid, context, audio_blocks, noise, steps=24, use_audio=True,
               *, audio_frame_valid=None):
        if type(use_audio) is not bool:
            raise ValueError('use_audio must be Boolean')
        if not use_audio:
            return self.prior.sample(valid, context, audio_blocks, noise, steps=steps,
                                     use_audio=False, audio_frame_valid=audio_frame_valid)
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        _prefix_mask(valid)
        x = _clean_sequence(noise, valid, self.latent_dim, 'noise')
        for step in range(steps):
            time = x.new_full((len(x),), step / steps)
            update = self.velocity(x, time, valid, context, audio_blocks, True,
                                   audio_frame_valid=audio_frame_valid)
            x = torch.where(valid[..., None], x + update / steps, 0.)
        return x


__all__ = ['BoundedPriorAudioAdapter']
