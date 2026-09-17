"""Free upper-face flow with a temporal condition path at every DiT layer.

The nine normalized residual channels are unconstrained: no signed-state lift,
spline projection, temporal centering, positive-only activation, or GT state
substitution is performed. Four predicted slow-state channels are an optional
soft condition. Direct and state-conditioned arms share all parameter shapes;
only the explicit use_state setting differs. Prediction conditions must be
provided by the caller identically during training and inference.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .dit import ResidualDiT
from .slow_state_affect import UPPER_INDICES, compose_upper_face


def _clean_sequence(value, valid, width, name):
    if (not torch.is_tensor(value) or value.shape != (*valid.shape, width)
            or not value.is_floating_point() or value.device != valid.device):
        raise ValueError(f'{name} must be floating [B,T,{width}] on the mask device')
    if not torch.isfinite(value[valid]).all():
        raise ValueError(f'Observed {name} must be finite')
    return torch.where(valid[..., None], value, 0.)


def _frame_position(frames, dim, reference):
    """Fixed native-frame coordinates, invariant to appended batch padding."""
    time = torch.arange(frames, device=reference.device, dtype=reference.dtype)[:, None]
    frequencies = torch.exp(torch.arange(0, dim, 2, device=reference.device, dtype=reference.dtype)
                            * (-math.log(10000.) / dim))
    phase = time * frequencies[None]
    position = reference.new_zeros(frames, dim)
    position[:, 0::2] = phase.sin()
    position[:, 1::2] = phase[:, :dim // 2].cos()
    return position[None]


class _LayerConditionedDiT(ResidualDiT):
    """Use existing DiT attention with an ungated same-frame condition route.

    Each block receives its own learned additive projection of the temporal
    condition before attention. This path does not depend on global AdaLN
    gates and is used identically at every training and Euler solver step.
    The optional state influences feature context only, never motion outputs
    through a fixed coefficient mapping.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dim = self.motion_input.out_features
        self.state_projection = nn.Linear(4, dim, bias=False)
        self.frame_condition = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in self.blocks)
        # Small nonzero starts allow the first backward pass to audit actual
        # conditioning gradients without beginning with a dominant new path.
        nn.init.normal_(self.state_projection.weight, std=.02)
        for layer in self.frame_condition:
            nn.init.normal_(layer.weight, std=.02)

    def velocity(self, x, time, content, global_code, intensity, identity, local, state, valid):
        frame = self.context_input(content) + self.local_emotion(local)
        if state is not None:
            frame = frame + self.state_projection(state)
        frame = torch.where(valid[..., None], frame, 0.)
        position = _frame_position(x.shape[1], frame.shape[-1], frame)
        context = torch.where(valid[..., None], frame + position, 0.)
        tokens = torch.where(valid[..., None], self.motion_input(x) + context, 0.)
        style = self.style(identity)
        global_condition = self.time(time) + self.emotion(torch.cat([global_code, intensity], -1)) + style
        for layer, block in zip(self.frame_condition, self.blocks):
            tokens = torch.where(valid[..., None], tokens + layer(F.silu(frame)), 0.)
            tokens = block(tokens, context, global_condition, style, valid)
        shift, scale = self.output_modulation(global_condition).chunk(2, -1)
        tokens = self.output_norm(tokens) * (1. + scale[:, None]) + shift[:, None]
        return torch.where(valid[..., None], self.output(tokens), 0.)


class TemporalUpperFlow(nn.Module):
    """Audio-conditioned flow over all nine upper-face residual dimensions.

    The caller defines normalized targets, e.g. (upper - independent neutral
    anchor) / train_scale. The same convention must reconstruct decode output.
    This module adds no residual_scale, clamp, state lift, or P/Q constraint.
    Use compose_upper_face to retain all other 43 baseline channels exactly.

    cfg contains the existing model dimensions. use_state=False is the direct
    matched arm, which ignores state entirely, even if the caller supplies it.
    Every arm owns identical parameters, permitting identical state_dict init.
    """

    def __init__(self, cfg=None, *, use_state=False):
        super().__init__()
        if type(use_state) is not bool:
            raise ValueError('use_state must be an explicit Boolean')
        model = {} if cfg is None else cfg.get('model', cfg)
        if float(model.get('dropout', 0.)) != 0.:
            raise ValueError('Matched temporal flow requires dropout=0 for identical train/inference condition paths')
        self.use_state = use_state
        self.content_dim = int(model.get('content_dim', 128))
        self.emotion_dim = int(model.get('emotion_dim', 64))
        self.style_dim = int(model.get('style_dim', 128))
        self.renderer = _LayerConditionedDiT(
            motion_dim=9, content_dim=self.content_dim, emotion_dim=self.emotion_dim,
            style_dim=self.style_dim, dim=int(model.get('dit_dim', 192)),
            depth=int(model.get('dit_depth', 4)), heads=int(model.get('heads', 6)),
            dropout=float(model.get('dropout', 0.)), intensity_dim=1,
            global_dropout=0., style_dropout=0.)

    def prepare_conditions(self, valid, base_h0, identity_code, global_affect, local, state=None):
        if (not torch.is_tensor(valid) or valid.ndim != 2 or valid.dtype != torch.bool
                or min(valid.shape) < 1 or not valid.any(1).all()):
            raise ValueError('Flow requires nonempty Boolean valid [B,T]')
        if not isinstance(global_affect, dict) or not {'global', 'intensity_value'} <= global_affect.keys():
            raise ValueError('global_affect requires global and intensity_value')
        content = _clean_sequence(base_h0, valid, self.content_dim, 'base_h0')
        local = _clean_sequence(local, valid, self.emotion_dim, 'local')
        state = _clean_sequence(state, valid, 4, 'state') if self.use_state else None
        global_code, intensity = global_affect['global'], global_affect['intensity_value']
        for name, value, width in (('identity_code', identity_code, self.style_dim),
                                   ('global', global_code, self.emotion_dim), ('intensity_value', intensity, 1)):
            if (not torch.is_tensor(value) or value.shape != (len(valid), width)
                    or not value.is_floating_point() or value.device != valid.device
                    or value.dtype != content.dtype or not torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite matching [B,{width}]')
        if local.dtype != content.dtype or (state is not None and state.dtype != content.dtype):
            raise ValueError('Temporal condition dtypes must match content')
        return {'content': content, 'identity': identity_code, 'global': global_code,
                'intensity': intensity, 'local': local, 'state': state, 'valid': valid}

    def velocity(self, noisy_motion, time, conditions):
        """Public diagnostic interface shared by flow_loss and decode."""
        valid = conditions['valid']
        x = _clean_sequence(noisy_motion, valid, 9, 'noisy_motion')
        if (not torch.is_tensor(time) or time.shape != (len(valid),) or time.device != valid.device
                or not time.is_floating_point() or not torch.isfinite(time).all()
                or ((time < 0.) | (time > 1.)).any()):
            raise ValueError('Flow time must be finite [B] within [0,1]')
        return self.renderer.velocity(x, time, conditions['content'], conditions['global'],
                                       conditions['intensity'], conditions['identity'],
                                       conditions['local'], conditions['state'], valid)

    def flow_loss(self, target_norm9, valid, base_h0, identity_code, global_affect,
                  local, state, noise, time):
        """Standard conditional flow matching, with no implicit GT conditioning.

        state must be the same audio-derived condition supplied at inference
        (or is ignored by the direct arm). Only target_norm9 defines the flow
        supervision; it is never converted into a condition by this method.
        """
        conditions = self.prepare_conditions(valid, base_h0, identity_code, global_affect, local, state)
        target = _clean_sequence(target_norm9, valid, 9, 'target_norm9')
        start = _clean_sequence(noise, valid, 9, 'noise')
        if not torch.is_tensor(time) or time.shape != (len(valid),):
            raise ValueError('Flow time must be [B]')
        interpolation = time[:, None, None]
        x = (1. - interpolation) * start + interpolation * target
        predicted = self.velocity(x, time, conditions)
        error = torch.where(valid[..., None], predicted - (target - start), 0.)
        return error.square().sum() / (valid.sum() * 9)

    def decode(self, valid, base_h0, identity_code, global_affect, local, state, noise, *, steps=12):
        """Euler integration with explicit seeded noise; all nine dimensions free."""
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        conditions = self.prepare_conditions(valid, base_h0, identity_code, global_affect, local, state)
        x = _clean_sequence(noise, valid, 9, 'noise')
        for index in range(steps):
            time = x.new_full((len(x),), index / steps)
            x = torch.where(valid[..., None], x + self.velocity(x, time, conditions) / steps, 0.)
        return x


__all__ = ['TemporalUpperFlow', 'UPPER_INDICES', 'compose_upper_face']
