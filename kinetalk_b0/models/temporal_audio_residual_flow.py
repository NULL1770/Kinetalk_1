"""Upper-nine flow with an explicit native clock and temporal state prior.

The earlier raw residual DiT has no positional input or temporal state mixing
of its own. This variant adds those mechanisms while retaining the existing
audio/global/identity contract. These mechanisms can improve a motion prior;
they do not establish an audio timing benefit without a matched static arm.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .audio_residual_flow import AudioResidualFlow


def correlated_native_noise(noise, valid, *, rho=.9):
    """Stationary AR(1) transform of supplied iid draws, reset at mask holes.

    Valid runs stay on their native frame indices. Each run starts with its
    supplied standard normal value, avoiding a low-variance startup ramp.
    The deterministic transformation consumes no random generator state.
    """
    if (not torch.is_tensor(noise) or noise.ndim != 3 or noise.shape[-1] != 9 or min(noise.shape) < 1
            or not noise.is_floating_point() or not torch.is_tensor(valid)
            or valid.dtype != torch.bool or valid.shape != noise.shape[:2]
            or valid.device != noise.device or not valid.any(1).all()
            or not torch.isfinite(noise[valid]).all()):
        raise ValueError('Finite observed noise [B,T,9] and nonempty Boolean native valid required')
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError('noise rho must be finite in [0,1)')
    clean = torch.where(valid[..., None], noise, 0.)
    previous = torch.zeros_like(clean[:, 0])
    previous_valid = torch.zeros_like(valid[:, 0])
    innovation_scale = math.sqrt(1. - rho * rho)
    values = []
    for index in range(clean.shape[1]):
        continuing = valid[:, index] & previous_valid
        value = torch.where(continuing[:, None], rho * previous + innovation_scale * clean[:, index], clean[:, index])
        previous = torch.where(valid[:, index, None], value, 0.)
        values.append(previous)
        previous_valid = valid[:, index]
    return torch.stack(values, 1)


class TemporalAudioResidualFlow(AudioResidualFlow):
    """Bounded temporal state conditioning plus a correlated initial measure.

    ``noise`` remains a caller-provided iid draw in both ``flow_loss`` and
    ``decode``. A fixed, small sinusoidal term marks native positions in h0;
    its amplitude is fixed before fitting and is identical in both arms.
    The low-rank temporal path starts at zero and adds a bounded correction
    to the DiT state input. It does not filter or clamp the generated output.
    """

    def __init__(self, cfg=None, *, stride=16, temporal_rank=4, noise_rho=.9,
                 position_scale=.01, temporal_scale=.25):
        super().__init__(cfg, stride=stride)
        if type(temporal_rank) is not int or not 1 <= temporal_rank <= 9:
            raise ValueError('temporal_rank must be an integer in [1,9]')
        for name, value in [('noise_rho', noise_rho), ('position_scale', position_scale), ('temporal_scale', temporal_scale)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if noise_rho >= 1 or temporal_scale == 0:
            raise ValueError('noise_rho must be <1 and temporal_scale must be positive')
        self.noise_rho = float(noise_rho)
        self.position_scale = float(position_scale)
        self.temporal_scale = float(temporal_scale)
        self.temporal_down = nn.Linear(9, temporal_rank, bias=False)
        self.temporal_conv = nn.Conv1d(temporal_rank, temporal_rank, 3, padding=1, bias=False)
        self.temporal_up = nn.Linear(temporal_rank, 9, bias=False)
        nn.init.zeros_(self.temporal_up.weight)
        self.temporal_config = {'temporal_rank': temporal_rank, 'noise_rho': self.noise_rho,
                                'position_scale': self.position_scale, 'temporal_scale': self.temporal_scale}

    def _positions(self, frames, reference):
        position = torch.arange(frames, device=reference.device, dtype=reference.dtype)[:, None]
        frequencies = torch.exp(-math.log(10000.) * torch.arange(0, self.content_dim, 2,
                                    device=reference.device, dtype=reference.dtype) / self.content_dim)
        values = reference.new_zeros(frames, self.content_dim)
        values[:, 0::2] = torch.sin(position * frequencies)
        values[:, 1::2] = torch.cos(position * frequencies[:self.content_dim // 2])
        return values * self.position_scale

    def _conditions(self, valid, base_h0, identity_code, global_affect, local, state):
        content, identity, global_code, intensity, condition = super()._conditions(
            valid, base_h0, identity_code, global_affect, local, state)
        content = torch.where(valid[..., None], content + self._positions(content.shape[1], content)[None], 0.)
        return content, identity, global_code, intensity, condition

    def _temporal_state(self, x, valid):
        clean = torch.where(valid[..., None], x, 0.)
        hidden = self.temporal_down(clean).transpose(1, 2)
        # Kernel 3 sees only immediate observed neighbors. Zero states at
        # holes prevent linking observations on opposite sides of a gap.
        hidden = self.temporal_conv(hidden).transpose(1, 2)
        correction = self.temporal_scale * torch.tanh(self.temporal_up(torch.tanh(hidden)))
        return torch.where(valid[..., None], clean + correction, 0.)

    def _velocity_unrestricted(self, x, time, valid, conditions):
        return super()._velocity_unrestricted(self._temporal_state(x, valid), time, valid, conditions)

    def flow_loss(self, target, q, identity, affect, local, state, noise, time):
        start = correlated_native_noise(noise, q['valid'], rho=self.noise_rho)
        return super().flow_loss(target, q, identity, affect, local, state, start, time)

    def decode(self, q, identity, affect, local, state, noise, steps):
        start = correlated_native_noise(noise, q['valid'], rho=self.noise_rho)
        return super().decode(q, identity, affect, local, state, start, steps)


__all__ = ['TemporalAudioResidualFlow', 'correlated_native_noise']
