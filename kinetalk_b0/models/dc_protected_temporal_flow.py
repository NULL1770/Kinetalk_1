"""A temporal residual flow that cannot change a clip's channel means.

The deterministic expression/state model is external.  This module receives
its fixed residual target and conditions, and models uncertainty only in the
zero-DC temporal subspace.  It removes no nonconstant spline/low-frequency
component.  Projection is in the supplied coefficient domain; a nonlinear
full-face composition does not inherit this additive mean guarantee.
"""
from __future__ import annotations

import torch

from .slow_state_affect import UPPER_INDICES
from .temporal_audio_residual_flow import (
    TemporalAudioResidualFlow,
    correlated_native_noise,
)


def project_temporal_dc(value, valid):
    """Remove the native-valid temporal mean of each of nine channels.

    This is the orthogonal projection I - 11'/N independently per clip and
    channel, restricted to observed frames.  Mask holes keep their native
    positions and remain zero; they are not compressed into a new timeline.
    Missing values may be nonfinite, but observed values must be finite.
    The projection is differentiable, idempotent up to floating point roundoff,
    and preserves every nonconstant temporal component.  Reductions and mean
    subtraction use float64 before restoring the input dtype to reduce drift.
    It does not normalize amplitudes, rescale time, or clip outputs.
    """
    if (not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != 9
            or min(value.shape) < 1 or not value.is_floating_point()):
        raise ValueError('Residual values must be nonempty floating [B,T,9]')
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != value.shape[:2] or valid.device != value.device
            or not valid.any(1).all()):
        raise ValueError('Residual valid must be nonempty Boolean [B,T] on the values device')
    if not torch.isfinite(value[valid]).all():
        raise ValueError('Observed residual values must be finite')
    mask = valid[..., None]
    clean = torch.where(mask, value, 0.).double()
    mean = clean.sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    return torch.where(mask, clean - mean, 0.).to(value.dtype)


class DCProtectedTemporalFlow(TemporalAudioResidualFlow):
    """Zero-DC upper9 residual with the temporal model's clock and AR prior.

    ``noise`` is a caller-provided IID draw, transformed to correlated noise
    exactly once before projecting.  The target, starting state, interpolation,
    velocity and Euler states all obey the same projection.  Conditions remain
    explicit inputs; callers must freeze/detach an external deterministic
    predictor if gradients must not update it.  Decode never reads motion GT.
    """

    architecture = 'dc_protected_temporal_flow_v1'

    def __init__(self, cfg=None, *, stride=16, temporal_rank=4, noise_rho=.9,
                 position_scale=.01, temporal_scale=.25):
        super().__init__(cfg, stride=stride, temporal_rank=temporal_rank,
                         noise_rho=noise_rho, position_scale=position_scale,
                         temporal_scale=temporal_scale)
        self.architecture_config = {
            'architecture': self.architecture,
            'projection': 'native_valid_clip_channel_dc',
            'stride': self.stride,
            **self.temporal_config,
        }

    def _starting_noise(self, noise, valid):
        return project_temporal_dc(
            correlated_native_noise(noise, valid, rho=self.noise_rho), valid)

    def _velocity_unrestricted(self, x, time, valid, conditions):
        # Retain the temporal state path and renderer, but constrain the actual
        # vector field.  Neither parent flow_loss nor decode is called: those
        # methods would apply AR correlation a second time.
        velocity = super()._velocity_unrestricted(x, time, valid, conditions)
        return project_temporal_dc(velocity, valid)

    def flow_loss(self, target, q, identity, affect, local, state, noise, time):
        valid = q['valid']
        target = project_temporal_dc(target, valid)
        self._check_motion(noise, valid, 'noise')
        if noise.dtype != target.dtype:
            raise ValueError('Flow noise and target must share dtype')
        if (not torch.is_tensor(time) or time.shape != (len(target),)
                or not time.is_floating_point() or time.device != target.device
                or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any()):
            raise ValueError('Flow time must be finite [B] in [0,1]')
        if 'channel_mask' in q:
            channels = q['channel_mask']
            if (not torch.is_tensor(channels) or channels.dtype != torch.bool
                    or channels.shape != (len(target), 52) or channels.device != valid.device
                    or not channels[:, list(UPPER_INDICES)].all()):
                raise ValueError('Upper flow training requires all nine observed upper channels')
        conditions = self._conditions(valid, q['h0'], identity['code'], affect, local, state)
        start = self._starting_noise(noise, valid)
        weight = time[:, None, None]
        x = project_temporal_dc((1. - weight) * start + weight * target, valid)
        velocity = self._velocity_unrestricted(x, time, valid, conditions)
        target_velocity = project_temporal_dc(target - start, valid)
        error = torch.where(valid[..., None], velocity - target_velocity, 0.)
        return error.square().sum() / (valid.sum() * 9)

    def decode(self, q, identity, affect, local, state, noise, steps):
        if type(steps) is not int or steps < 1:
            raise ValueError('Positive integer decode budget required')
        valid = q['valid']
        x = self._starting_noise(noise, valid)
        conditions = self._conditions(valid, q['h0'], identity['code'], affect, local, state)
        for index in range(steps):
            time = x.new_full((len(x),), index / steps)
            velocity = self._velocity_unrestricted(x, time, valid, conditions)
            x = project_temporal_dc(x + velocity / steps, valid)
        return x


__all__ = ['DCProtectedTemporalFlow', 'project_temporal_dc']
