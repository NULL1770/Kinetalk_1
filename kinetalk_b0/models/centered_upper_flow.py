"""Zero-mean upper-face flow with an optional train-fitted temporal prior."""
from __future__ import annotations

import torch

from .temporal_upper import TemporalUpperFlow


def center_valid(value, valid):
    if (value.ndim != 3 or valid.shape != value.shape[:2] or valid.dtype != torch.bool
            or valid.device != value.device or not valid.any(1).all()):
        raise ValueError('Require nonempty matching native observation masks')
    if not torch.isfinite(value[valid]).all():
        raise ValueError('Observed values must be finite')
    clean = torch.where(valid[..., None], value, 0.)
    mean = clean.sum(1, keepdim=True) / valid.sum(1)[:, None, None]
    return torch.where(valid[..., None], clean-mean, 0.)


def fit_dynamic_statistics(motion, valid):
    """Fit dynamic RMS and stationary adjacent correlation on fit motion only."""
    values = center_valid(motion.double(), valid)
    scales = (values.square().sum((0, 1))/valid.sum()).sqrt().clamp_min(.005)
    adjacent = (valid[:, 1:] & valid[:, :-1])[..., None]
    left, right = values[:, :-1], values[:, 1:]
    denominator = ((left.square()+right.square())*adjacent).sum((0, 1))*.5
    rho = ((left*right*adjacent).sum((0, 1))/denominator.clamp_min(1e-12)).clamp(0., .995)
    return scales.float(), rho.float()


class CenteredUpperFlow(TemporalUpperFlow):
    """Only zero-mean dynamics are generated; mean composition is external.

    The temporal prior is a stationary AR(1) Gaussian on the native frame
    clock. Its expected centered variance is normalized analytically, never
    by the realized sample amplitude. No target is used by decode or prior.
    """

    def __init__(self, cfg, rho):
        super().__init__(cfg, use_state=False)
        if rho.shape != (9,) or not torch.isfinite(rho).all() or ((rho < 0) | (rho >= 1)).any():
            raise ValueError('Require nine finite prior correlations in [0,1)')
        self.register_buffer('prior_rho', rho.detach().float().clone())

    def prior(self, white, valid):
        if white.shape != (*valid.shape, 9) or not torch.isfinite(white).all():
            raise ValueError('Finite native white noise [B,T,9] required')
        frames = white.shape[1]
        clock = torch.arange(frames, device=white.device)
        difference = clock[:, None]-clock[None]
        rho = self.prior_rho.to(white)
        causal = (difference >= 0).to(white.dtype)
        matrix = rho[:, None, None].pow(difference.clamp_min(0)[None])*causal
        innovation = (1-rho.square()).sqrt()[:, None].expand(-1, frames).clone()
        innovation[:, 0] = 1.
        matrix = matrix*innovation[:, None]
        correlated = torch.einsum('dts,bsd->btd', matrix, white)
        covariance = rho[:, None, None].pow(difference.abs()[None])
        weights = valid.to(white.dtype)/valid.sum(1, keepdim=True).clamp_min(1)
        mean_variance = torch.einsum('bt,dts,bs->bd', weights, covariance, weights)
        expected_centered_variance = (1-mean_variance).clamp_min(1e-6)
        return center_valid(correlated, valid)/expected_centered_variance.sqrt()[:, None]

    def velocity(self, noisy_motion, time, conditions):
        return center_valid(super().velocity(noisy_motion, time, conditions), conditions['valid'])

    def flow_loss(self, target_norm9, valid, base_h0, identity_code, global_affect, local, state, noise, time):
        return super().flow_loss(center_valid(target_norm9, valid), valid, base_h0,
                                 identity_code, global_affect, local, None, center_valid(noise, valid), time)

    def decode(self, valid, base_h0, identity_code, global_affect, local, state, noise, *, steps=12):
        return center_valid(super().decode(valid, base_h0, identity_code, global_affect,
                                          local, None, center_valid(noise, valid), steps=steps), valid)
