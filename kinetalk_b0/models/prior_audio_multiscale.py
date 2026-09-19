"""Multi-scale continuous audio residual around a frozen latent motion prior.

The module keeps the source prior as the immutable baseline and gives local
acoustics two paths: a fast path that sees every 1540-D frame in each native
five-frame latent block, and a slow path that pools the four prosody channels
(log-F0, RMS, periodicity and voicing) over the next four latent blocks (20
native frames).  The slow path modulates every residual block while the fast
path enters its token state.  A zero-initialized output map makes the residual
an exact no-op at construction; the prior therefore remains the deterministic
fallback and can be evaluated with ``use_audio=False``.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .continuous_upper_motion import (
    ContinuousLatentFlow,
    _clean_sequence,
    _positive_dimensions,
    _prefix_mask,
    _sinusoidal,
)


class _TokenFiLMResidualBlock(nn.Module):
    """Masked temporal block with token-wise slow acoustic FiLM."""

    def __init__(self, hidden: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(hidden)
        self.temporal = nn.Linear(3 * hidden, hidden)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden))
        self.film = nn.Linear(hidden, 2 * hidden)

    def forward(self, x, valid, sequence_condition, slow_condition):
        h = self.norm(x)
        # sequence_condition carries global context and diffusion time; adding
        # it to the token-wise slow state retains both clocks in one FiLM map.
        shift, scale = self.film(F.silu(slow_condition + sequence_condition[:, None])).chunk(2, -1)
        h = h * (1. + scale) + shift
        h = torch.where(valid[..., None], F.silu(h), 0.)
        d = self.dilation
        if d >= x.shape[1]:
            left = right = torch.zeros_like(h)
        else:
            left = F.pad(h[:, :-d], (0, 0, d, 0))
            right = F.pad(h[:, d:], (0, 0, 0, d))
        update = self.output(self.temporal(torch.cat((left, h, right), -1)))
        return torch.where(valid[..., None], x + update, 0.)


class MultiScalePriorAudioResidual(nn.Module):
    """Bounded multi-scale audio residual on top of a frozen source prior.

    Parameters are intentionally separate from :class:`ContinuousLatentFlow`
    so a saved source-prior state can be wrapped without changing its output.
    ``audio_blocks`` has shape ``[B,L,5,1540]`` by default.  The fast branch
    consumes all channels in every five-frame block.  The slow branch pools
    channels ``1536:1540`` over four latent blocks (20 native frames), then
    applies token-wise FiLM in each residual block.  No motion/target/mask is
    accepted by generation; ``valid`` only describes the observed clock.
    """

    def __init__(self, prior: ContinuousLatentFlow, hidden: int = 96, depth: int = 4,
                 fast_hidden: int = 32, max_delta: float = .35,
                 slow_window_blocks: int = 4, prosody_slice=(1536, 1540)):
        super().__init__()
        if not isinstance(prior, ContinuousLatentFlow):
            raise TypeError('prior must be a ContinuousLatentFlow')
        _positive_dimensions(hidden=hidden, depth=depth, fast_hidden=fast_hidden,
                             slow_window_blocks=slow_window_blocks)
        if prior.audio_dim != 1540:
            raise ValueError('Multi-scale audio requires the complete 1540-D feature vector')
        if tuple(prosody_slice) != (1536, 1540):
            raise ValueError('prosody_slice must be the canonical (1536, 1540) channels')
        if isinstance(max_delta, bool) or not isinstance(max_delta, (int, float)) \
                or not math.isfinite(max_delta) or max_delta <= 0:
            raise ValueError('max_delta must be a finite positive scalar')
        self.prior = prior
        self.prior.requires_grad_(False)
        self.prior.eval()
        self.context_dim, self.audio_dim = prior.context_dim, prior.audio_dim
        self.latent_dim, self.block_size = prior.latent_dim, prior.block_size
        if self.block_size != 5:
            raise ValueError('Multi-scale audio protocol requires a five-frame latent block')
        self.hidden, self.fast_hidden = hidden, fast_hidden
        self.max_delta = float(max_delta)
        self.slow_window_blocks = int(slow_window_blocks)
        self.prosody_slice = tuple(prosody_slice)
        self.config = dict(prior=dict(prior.config), hidden=hidden, depth=depth,
                           fast_hidden=fast_hidden, max_delta=float(max_delta),
                           slow_window_blocks=int(slow_window_blocks),
                           prosody_slice=list(prosody_slice))

        self.motion_projection = nn.Linear(self.latent_dim, hidden)
        self.context_projection = nn.Linear(self.context_dim, hidden)
        self.fast_frame_projection = nn.Sequential(nn.Linear(self.audio_dim, fast_hidden), nn.SiLU())
        self.fast_block_projection = nn.Linear(self.block_size * (fast_hidden + 1), hidden)
        # Inputs already use frozen train-only per-channel normalization. Do
        # not normalize four different physical quantities against each other.
        self.slow_projection = nn.Sequential(nn.Linear(4, hidden), nn.SiLU())
        self.time_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(_TokenFiLMResidualBlock(hidden, 2 ** i) for i in range(depth))
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, self.latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def train(self, mode=True):
        super().train(mode)
        self.prior.eval().requires_grad_(False)
        return self

    def residual_parameters(self):
        return (p for name, p in self.named_parameters() if not name.startswith('prior.'))

    def _validate(self, valid, context, audio_blocks, dtype, audio_frame_valid):
        _prefix_mask(valid)
        if (not torch.is_tensor(context) or context.shape != (len(valid), self.context_dim)
                or context.device != valid.device or context.dtype != dtype
                or not torch.isfinite(context).all()):
            raise ValueError('context must be finite matching floating [B,context_dim]')
        expected = (*valid.shape, self.block_size, self.audio_dim)
        if (not torch.is_tensor(audio_blocks) or audio_blocks.shape != expected
                or audio_blocks.device != valid.device or audio_blocks.dtype != dtype):
            raise ValueError('audio_blocks must be matching floating [B,L,5,1540]')
        if audio_frame_valid is None:
            frame_valid = valid[..., None].expand(-1, -1, self.block_size)
        else:
            frame_valid = audio_frame_valid
            if (not torch.is_tensor(frame_valid) or frame_valid.dtype != torch.bool
                    or frame_valid.shape != (*valid.shape, self.block_size)
                    or frame_valid.device != valid.device):
                raise ValueError('audio_frame_valid must be matching Boolean [B,L,5]')
            _prefix_mask(frame_valid.reshape(len(valid), -1), 'audio_frame_valid')
            if not torch.equal(frame_valid.any(-1), valid):
                raise ValueError('audio_frame_valid support must agree with latent valid')
        if not torch.isfinite(audio_blocks[frame_valid]).all():
            raise ValueError('observed audio_blocks must be finite')
        clean = torch.where(frame_valid[..., None], audio_blocks, 0.)
        return clean, frame_valid

    def _fast_and_slow(self, valid, audio_blocks, frame_valid, dtype):
        frame = self.fast_frame_projection(audio_blocks)
        frame = torch.where(frame_valid[..., None], frame, 0.)
        packed = torch.cat((frame.flatten(-2), frame_valid.to(dtype)), -1)
        fast = self.fast_block_projection(packed)
        fast = torch.where(valid[..., None], fast, 0.)

        # Per-block native means, followed by a 4-block (20-frame) forward
        # window.  Prefix masks and explicit counts ensure partial tails do not
        # change the clock or introduce NaNs.
        prosody = audio_blocks[..., self.prosody_slice[0]:self.prosody_slice[1]]
        counts = frame_valid.sum(-1, keepdim=True).to(dtype)
        block = prosody.sum(-2)
        slow_sum = torch.zeros_like(block)
        slow_count = torch.zeros((*valid.shape, 1), dtype=dtype, device=valid.device)
        for offset in range(self.slow_window_blocks):
            if offset >= block.shape[1]:
                break
            values = F.pad(block[:, offset:], (0, 0, 0, offset))[:, :block.shape[1]]
            weight = F.pad(counts[:, offset:], (0, 0, 0, offset))[:, :block.shape[1]]
            slow_sum = slow_sum + values
            slow_count = slow_count + weight
        slow = slow_sum / slow_count.clamp_min(1.)
        slow = self.slow_projection(slow)
        slow = torch.where(valid[..., None], slow, 0.)
        return fast, slow

    def _field(self, noisy, time, valid, context, clean, frame_valid):
        fast, slow = self._fast_and_slow(valid, clean, frame_valid, noisy.dtype)
        positions = _sinusoidal(torch.arange(valid.shape[1], device=noisy.device,
                                              dtype=noisy.dtype), self.hidden)
        x = self.motion_projection(noisy) + fast + positions[None]
        x = torch.where(valid[..., None], x, 0.)
        condition = self.context_projection(context) + self.time_projection(
            _sinusoidal(time * 1000., self.hidden))
        for block in self.blocks:
            x = block(x, valid, condition, slow)
        return self.output(self.output_norm(x))

    def residual(self, noisy, time, valid, context, audio_blocks, *, audio_frame_valid=None):
        _prefix_mask(valid)
        noisy = _clean_sequence(noisy, valid, self.latent_dim, 'noisy')
        if (not torch.is_tensor(time) or time.shape != (len(valid),)
                or time.dtype != noisy.dtype or time.device != valid.device
                or not torch.isfinite(time).all() or ((time < 0.) | (time > 1.)).any()):
            raise ValueError('time must be finite matching [B] in [0,1]')
        clean, frame_valid = self._validate(valid, context, audio_blocks, noisy.dtype, audio_frame_valid)
        # Subtract the identical field evaluated at zero local audio. This
        # anchors the trained residual exactly, rather than approximately with
        # a null loss. Neither field reads a target or motion observation mask.
        raw = self._field(noisy, time, valid, context, clean, frame_valid)
        null = self._field(noisy, time, valid, context, torch.zeros_like(clean), frame_valid)
        correction = self.max_delta * torch.tanh(raw - null)
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


__all__ = ['MultiScalePriorAudioResidual']
