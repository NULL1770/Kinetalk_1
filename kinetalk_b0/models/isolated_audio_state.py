"""Independent deterministic upper-face mean and centered audio state predictors.

Statistics must be fitted on TRAIN before construction.  The anchor is an
independent enrollment reference; no query target, label, or query mean is an
inference argument.  There are no shared trainable parameters between the two
branches, and supplied global/identity conditions are treated as frozen inputs.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .label_guided_affect import TemporalBlock
from .slow_state_affect import UPPER_INDICES, lift_slow_state, masked_slow_state


def _fitted_vector(value, size, name, *, positive=False):
    if (not torch.is_tensor(value) or value.ndim != 1
            or (size is not None and value.numel() != size)
            or value.numel() == 0 or not value.is_floating_point()
            or not torch.isfinite(value).all()
            or (positive and (value <= 0).any())):
        qualifier = 'positive ' if positive else ''
        raise ValueError(f'{name} must be a finite {qualifier}TRAIN-fitted vector')
    return value.detach().clone().float()


class _StateBranch(nn.Module):
    def __init__(self, features, global_dim, identity_dim, hidden, dilations):
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.condition = nn.Linear(global_dim + identity_dim, hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, dilation) for dilation in dilations)
        self.head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, normalized, valid, conditions):
        hidden = torch.where(valid[..., None],
                             F.silu(self.input(normalized) + self.condition(conditions)[:, None]), 0.)
        for block in self.blocks:
            hidden = block(hidden, valid)
        return self.head(hidden), hidden


class IsolatedAudioState(nn.Module):
    """Predict a clip mean separately from four zero-DC slow expression states.

    ``mean`` is raw upper9 coefficients. ``state`` is the centered four-group
    state in channel-normalized units after applying TRAIN dynamic scales.
    ``upper`` is ``mean + lift(state)`` on valid frames. Invalid temporal outputs
    are zero placeholders. The model does not clamp raw coefficients.

    ``static`` retains the exact audio-predicted mean but removes all framewise
    state variation *after* encoding. ``reverse`` reverses predicted conditions
    only at observed native positions, keeping the mean and mask unchanged.
    These are condition interventions, not raw-waveform interventions.
    """

    def __init__(self, feature_mean, feature_std, channel_scales, state_dynamic_scales,
                 *, global_dim=64, identity_dim=128, hidden=128, stride=16,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        mean = _fitted_vector(feature_mean, None, 'feature_mean')
        std = _fitted_vector(feature_std, mean.numel(), 'feature_std', positive=True)
        channels = _fitted_vector(channel_scales, 52, 'channel_scales', positive=True)
        dynamics = _fitted_vector(state_dynamic_scales, 4, 'state_dynamic_scales', positive=True)
        if any(value.device != mean.device for value in (std, channels, dynamics)):
            raise ValueError('All TRAIN-fitted statistics must be on the same device')
        if any(type(value) is not int or value < 1 for value in (global_dim, identity_dim, hidden, stride)):
            raise ValueError('Dimensions and stride must be positive integers')
        if (not isinstance(dilations, (tuple, list)) or not dilations
                or any(type(value) is not int or value < 1 for value in dilations)):
            raise ValueError('dilations must be a nonempty sequence of positive integers')
        for name, value in [('feature_mean', mean), ('feature_std', std),
                            ('channel_scales', channels), ('state_dynamic_scales', dynamics)]:
            self.register_buffer(name, value)
        self.global_dim, self.identity_dim, self.hidden = global_dim, identity_dim, hidden
        self.stride, self.dilations = stride, tuple(dilations)
        self.mean_branch = nn.Sequential(
            nn.Linear(mean.numel() + global_dim + identity_dim + 9, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9))
        nn.init.zeros_(self.mean_branch[-1].weight)
        nn.init.zeros_(self.mean_branch[-1].bias)
        self.state_branch = _StateBranch(mean.numel(), global_dim, identity_dim, hidden, dilations)
        # Constructing from GPU statistics should not leave parameters on CPU.
        self.to(device=mean.device)

    def export_config(self):
        """Constructor kwargs; all TRAIN-fitted vectors live in ``state_dict``."""
        return {'global_dim': self.global_dim, 'identity_dim': self.identity_dim,
                'hidden': self.hidden, 'stride': self.stride, 'dilations': list(self.dilations)}

    def _check_inputs(self, features, valid, global_code, identity_code, anchor):
        if (not torch.is_tensor(features) or features.ndim != 3
                or min(features.shape) < 1 or features.shape[-1] != len(self.feature_mean)
                or not features.is_floating_point() or features.device != self.feature_mean.device
                or features.dtype != self.feature_mean.dtype):
            raise ValueError('features must be nonempty floating [B,T,F] matching fitted statistics')
        if (not torch.is_tensor(valid) or valid.dtype != torch.bool
                or valid.shape != features.shape[:2] or valid.device != features.device
                or not valid.any(1).all()):
            raise ValueError('valid must be Boolean [B,T] with observations in every sample')
        if not torch.isfinite(features[valid]).all():
            raise ValueError('Observed features must be finite')
        for name, value, width in [('global_code', global_code, self.global_dim),
                                   ('identity_code', identity_code, self.identity_dim),
                                   ('anchor', anchor, 52)]:
            if (not torch.is_tensor(value) or value.shape != (len(features), width)
                    or value.device != features.device or value.dtype != features.dtype):
                raise ValueError(f'{name} must match floating [B,{width}], device, and dtype')
            used = value[:, list(UPPER_INDICES)] if name == 'anchor' else value
            if not torch.isfinite(used).all():
                raise ValueError(f'Used {name} values must be finite')

    def forward(self, features, valid, global_code, identity_code, anchor, mode='audio'):
        if mode not in ('audio', 'static', 'reverse'):
            raise ValueError('mode must be audio, static, or reverse')
        self._check_inputs(features, valid, global_code, identity_code, anchor)
        frames = features.shape[1]
        # Trailing all-batch padding is removed before reduction and convolution,
        # giving bitwise invariance to appending unobserved storage positions.
        native_frames = int(torch.nonzero(valid.any(0), as_tuple=False)[-1, 0]) + 1
        features, valid = features[:, :native_frames], valid[:, :native_frames]
        clean = torch.where(valid[..., None], features, self.feature_mean)
        normalized = (clean - self.feature_mean) / self.feature_std
        counts = valid.sum(1, keepdim=True)
        pooled = normalized.sum(1) / counts
        global_code, identity_code = global_code.detach(), identity_code.detach()
        anchor_upper = anchor.detach()[:, list(UPPER_INDICES)]
        conditions = torch.cat((global_code, identity_code), -1)
        mean_input = torch.cat((pooled, conditions, anchor_upper), -1)
        mean = anchor_upper + self.channel_scales[list(UPPER_INDICES)] * self.mean_branch(mean_input)

        state_raw, local = self.state_branch(normalized, valid, conditions)
        state = masked_slow_state(state_raw, valid, stride=self.stride)['state']
        state = torch.where(valid[..., None], state - state.sum(1, keepdim=True) / counts[..., None], 0.)
        state = state * self.state_dynamic_scales
        if mode == 'static':
            state = state * 0.
            local = torch.where(valid[..., None], local.sum(1, keepdim=True) / counts[..., None], 0.)
        elif mode == 'reverse':
            state, local = state.clone(), local.clone()
            for batch_index in range(len(features)):
                indices = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
                state[batch_index, indices] = state[batch_index, indices.flip(0)].clone()
                local[batch_index, indices] = local[batch_index, indices.flip(0)].clone()
        delta = lift_slow_state(state, self.channel_scales)[..., list(UPPER_INDICES)]
        upper = torch.where(valid[..., None], mean[:, None] + delta, 0.)
        padding = frames - native_frames
        if padding:
            state, upper, local = (F.pad(value, (0, 0, 0, padding)) for value in (state, upper, local))
        return {'mean': mean, 'state': state, 'upper': upper, 'local': local}


__all__ = ['IsolatedAudioState']
