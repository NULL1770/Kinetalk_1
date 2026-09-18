"""Continuous upper-face compression and a joint latent rectified-flow prior.

The caller owns raw-residual normalization, frozen global/identity conditions,
full-face composition, and splitting native gaps into separate sequences. These
modules never inspect target means, apply logit/sigmoid, or choose random noise.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_dimensions(**values):
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError('dimensions and depth must be positive integers')


def _prefix_mask(valid, name='valid'):
    if (not torch.is_tensor(valid) or valid.ndim != 2 or valid.dtype != torch.bool
            or min(valid.shape) < 1 or not valid.any(1).all()):
        raise ValueError(f'{name} must be Boolean [B,T] with a nonempty prefix in every row')
    if ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError(f'{name} must be a prefix; split gaps before calling the model')


def _clean_sequence(value, valid, width, name, dtype=None):
    if (not torch.is_tensor(value) or value.shape != (*valid.shape, width)
            or not value.is_floating_point() or value.device != valid.device
            or (dtype is not None and value.dtype != dtype)):
        raise ValueError(f'{name} must be matching floating [B,T,{width}]')
    if not torch.isfinite(value[valid]).all():
        raise ValueError(f'observed {name} must be finite')
    return torch.where(valid[..., None], value, 0.)


def _frame_blocks(valid, block_size):
    tail = (-valid.shape[1]) % block_size
    return F.pad(valid, (0, tail), value=False).reshape(len(valid), -1, block_size)


def _sinusoidal(value, width):
    frequencies = torch.exp(torch.arange(0, width, 2, device=value.device, dtype=value.dtype)
                            * (-math.log(10000.) / width))
    angles = value[..., None] * frequencies
    result = value.new_zeros(*value.shape, width)
    result[..., 0::2] = angles.sin()
    result[..., 1::2] = angles[..., :width // 2].cos()
    return result


class _MaskedTemporalBlock(nn.Module):
    """Per-token normalization and masked dilated taps, without batch statistics."""
    def __init__(self, hidden, dilation, conditional=False):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(hidden)
        self.temporal = nn.Linear(3 * hidden, hidden)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden))
        self.modulation = nn.Linear(hidden, 2 * hidden) if conditional else None

    def forward(self, x, valid, context=None):
        h = self.norm(x)
        if self.modulation is not None:
            shift, scale = self.modulation(F.silu(context)).chunk(2, -1)
            h = h * (1. + scale[:, None]) + shift[:, None]
        h = torch.where(valid[..., None], F.silu(h), 0.)
        d = self.dilation
        if d >= x.shape[1]:
            left, right = torch.zeros_like(h), torch.zeros_like(h)
        else:
            left = F.pad(h[:, :-d], (0, 0, d, 0))
            right = F.pad(h[:, d:], (0, 0, 0, d))
        update = self.output(self.temporal(torch.cat((left, h, right), -1)))
        return torch.where(valid[..., None], x + update, 0.)


class ContinuousUpperAE(nn.Module):
    """Ordered native-frame packing, continuous latent sequence, native decoding.

    Five-frame blocks retain all 45 ordered coordinates before compression;
    there is no average pooling or interpolation. A tail mask is an explicit
    encoder/decoder feature. Compression can still lose peaks and must be
    evaluated separately before fitting a generative prior.
    """
    def __init__(self, motion_dim=9, block_size=5, latent_dim=16, hidden=64, depth=4):
        super().__init__()
        self.config = dict(motion_dim=motion_dim, block_size=block_size,
                           latent_dim=latent_dim, hidden=hidden, depth=depth)
        _positive_dimensions(**self.config)
        self.motion_dim, self.block_size, self.latent_dim = motion_dim, block_size, latent_dim
        self.encoder_input = nn.Linear(block_size * (motion_dim + 1), hidden)
        self.encoder_blocks = nn.ModuleList(_MaskedTemporalBlock(hidden, 2 ** i) for i in range(depth))
        self.encoder_output = nn.Linear(hidden, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + block_size, hidden)
        self.decoder_blocks = nn.ModuleList(_MaskedTemporalBlock(hidden, 2 ** i) for i in range(depth))
        self.decoder_output = nn.Linear(hidden, block_size * motion_dim)

    def encode(self, motion, valid):
        _prefix_mask(valid)
        motion = _clean_sequence(motion, valid, self.motion_dim, 'motion')
        frame_mask = _frame_blocks(valid, self.block_size)
        latent_valid = frame_mask.any(-1)
        tail = frame_mask.shape[1] * self.block_size - valid.shape[1]
        packed = F.pad(motion, (0, 0, 0, tail)).reshape(len(valid), frame_mask.shape[1], -1)
        packed = torch.cat((packed, frame_mask.to(motion.dtype)), -1)
        x = torch.where(latent_valid[..., None], self.encoder_input(packed), 0.)
        for block in self.encoder_blocks:
            x = block(x, latent_valid)
        z = torch.where(latent_valid[..., None], self.encoder_output(x), 0.)
        return z, latent_valid

    def decode(self, z, valid):
        _prefix_mask(valid)
        frame_mask = _frame_blocks(valid, self.block_size)
        latent_valid = frame_mask.any(-1)
        z = _clean_sequence(z, latent_valid, self.latent_dim, 'z')
        x = self.decoder_input(torch.cat((z, frame_mask.to(z.dtype)), -1))
        x = torch.where(latent_valid[..., None], x, 0.)
        for block in self.decoder_blocks:
            x = block(x, latent_valid)
        motion = self.decoder_output(x).reshape(len(valid), -1, self.motion_dim)[:, :valid.shape[1]]
        return torch.where(valid[..., None], motion, 0.)

    def forward(self, motion, valid):
        return self.decode(self.encode(motion, valid)[0], valid)


class ContinuousLatentFlow(nn.Module):
    """Whole-run joint flow in a continuous motion latent, with optional audio.

    Audio is projected per native frame before ordered packing. The optional
    audio_frame_valid [B,L,block_size] supplies exact partial-tail support; when
    omitted, every latent-valid token is assumed to contain a complete block.
    The runner must supply it for partial blocks. Generation uses explicit
    caller-owned noise, and has no dropout or internal random draws.
    """
    def __init__(self, context_dim, audio_dim=1540, latent_dim=16, hidden=96,
                 depth=4, block_size=5):
        super().__init__()
        self.config = dict(context_dim=context_dim, audio_dim=audio_dim, latent_dim=latent_dim,
                           hidden=hidden, depth=depth, block_size=block_size)
        _positive_dimensions(**self.config)
        self.context_dim, self.audio_dim = context_dim, audio_dim
        self.latent_dim, self.hidden, self.block_size = latent_dim, hidden, block_size
        self.motion_projection = nn.Linear(latent_dim, hidden)
        self.context_projection = nn.Linear(context_dim, hidden)
        self.audio_frame_projection = nn.Sequential(nn.Linear(audio_dim, 32), nn.SiLU())
        self.audio_block_projection = nn.Linear(block_size * 33, hidden)
        self.time_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(_MaskedTemporalBlock(hidden, 2 ** i, conditional=True) for i in range(depth))
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, latent_dim)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)

    def _conditions(self, valid, context, audio_blocks, dtype, use_audio, audio_frame_valid):
        if type(use_audio) is not bool:
            raise ValueError('use_audio must be Boolean')
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
        if use_audio:
            clean = torch.where(frame_valid[..., None], audio_blocks, 0.)
            frame = self.audio_frame_projection(clean)
            frame = torch.where(frame_valid[..., None], frame, 0.)
            packed = torch.cat((frame.flatten(-2), frame_valid.to(dtype)), -1)
            audio = torch.where(valid[..., None], self.audio_block_projection(packed), 0.)
        else:
            audio = context.new_zeros(*valid.shape, self.hidden)
        return audio, self.context_projection(context)

    def velocity(self, noisy, time, valid, context, audio_blocks, use_audio=True, *, audio_frame_valid=None):
        _prefix_mask(valid)
        noisy = _clean_sequence(noisy, valid, self.latent_dim, 'noisy')
        if (not torch.is_tensor(time) or time.shape != (len(valid),)
                or time.dtype != noisy.dtype or time.device != valid.device
                or not torch.isfinite(time).all() or ((time < 0.) | (time > 1.)).any()):
            raise ValueError('time must be finite matching [B] in [0,1]')
        audio, context = self._conditions(valid, context, audio_blocks, noisy.dtype, use_audio, audio_frame_valid)
        positions = _sinusoidal(torch.arange(valid.shape[1], device=noisy.device, dtype=noisy.dtype), self.hidden)
        x = self.motion_projection(noisy) + audio + positions[None]
        x = torch.where(valid[..., None], x, 0.)
        context = context + self.time_projection(_sinusoidal(time * 1000., self.hidden))
        for block in self.blocks:
            x = block(x, valid, context)
        return torch.where(valid[..., None], self.output(self.output_norm(x)), 0.)

    def flow_loss(self, target, valid, context, audio_blocks, noise, time, use_audio=True, *, audio_frame_valid=None):
        _prefix_mask(valid)
        target = _clean_sequence(target, valid, self.latent_dim, 'target')
        noise = _clean_sequence(noise, valid, self.latent_dim, 'noise', target.dtype)
        if not torch.is_tensor(time) or time.shape != (len(valid),):
            raise ValueError('time must be matching [B]')
        fraction = time[:, None, None]
        velocity = self.velocity((1. - fraction) * noise + fraction * target, time, valid,
                                 context, audio_blocks, use_audio, audio_frame_valid=audio_frame_valid)
        error = torch.where(valid[..., None], velocity - (target - noise), 0.)
        return error.square().sum() / (valid.sum() * self.latent_dim)

    def sample(self, valid, context, audio_blocks, noise, steps=24, use_audio=True, *, audio_frame_valid=None):
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        _prefix_mask(valid)
        x = _clean_sequence(noise, valid, self.latent_dim, 'noise')
        for step in range(steps):
            time = x.new_full((len(x),), step / steps)
            update = self.velocity(x, time, valid, context, audio_blocks, use_audio,
                                   audio_frame_valid=audio_frame_valid)
            x = torch.where(valid[..., None], x + update / steps, 0.)
        return x


__all__ = ['ContinuousUpperAE', 'ContinuousLatentFlow']
