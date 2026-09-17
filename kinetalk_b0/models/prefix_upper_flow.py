"""Conditional flow with explicit clean motion tokens in the same attention.

The caller passes the full prefix+current native clock for all conditions and
an explicit known-mask. Known values are fixed observations; only unknown valid
tokens are generated. Strictly preceding prefix slicing and teacher/generated
selection belong to the caller. This class never infers known motion from GT.
"""
from __future__ import annotations

import torch
from torch import nn

from .temporal_upper import TemporalUpperFlow


class PrefixUpperFlow(TemporalUpperFlow):
    """Clamp known motion each solver step; match flow only on unknown tokens.

    target/noise/known all have shape [B,T,9]. valid and known_mask are Boolean
    [B,T], with known_mask a subset of valid and at least one unknown valid frame
    in every row. known is read only where known_mask is true; target and noise
    are read only on unknown valid frames. Invalid padding is zeroed before any
    trainable arithmetic. Decode returns known values exactly and zero padding.

    A zero-initialized two-entry embedding adds known/unknown state to the local
    condition. All motion tokens share the inherited DiT self-attention. With
    no known tokens and initial embedding weights, a copied TemporalUpperFlow
    backbone therefore has exactly its original behavior. There is no implicit
    current-target condition, history detach, centering, or coefficient clamp.

    Arbitrary known masks are supported as a numerical operation. A deployable
    autoregressive caller must only mark previous generated frames as known;
    GT-known frames are reserved for training and explicitly labeled oracle
    diagnostics. The method cannot validate temporal provenance from a tensor.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg, use_state=False)
        self.known_embedding = nn.Embedding(2, self.emotion_dim)
        nn.init.zeros_(self.known_embedding.weight)

    @staticmethod
    def _clean_selected(value, selected, reference, name):
        if (not torch.is_tensor(value) or value.shape != (*selected.shape, 9)
                or not value.is_floating_point() or value.device != reference.device
                or value.dtype != reference.dtype):
            raise ValueError(f'{name} must match condition dtype/device with shape [B,T,9]')
        if not torch.isfinite(value[selected]).all():
            raise ValueError(f'Observed {name} must be finite')
        return torch.where(selected[..., None], value, 0.)

    def prepare_prefix_conditions(self, valid, base_h0, identity_code, global_affect, local,
                                  *, known, known_mask):
        """Prepare a fresh conditions dictionary without mutating caller inputs."""
        prepared = super().prepare_conditions(valid, base_h0, identity_code, global_affect, local, None)
        if (not torch.is_tensor(known_mask) or known_mask.dtype != torch.bool
                or known_mask.shape != valid.shape or known_mask.device != valid.device
                or (known_mask & ~valid).any()):
            raise ValueError('known_mask must be matching Boolean [B,T] and a subset of valid')
        unknown = valid & ~known_mask
        if not unknown.any(1).all():
            raise ValueError('Every row requires an unknown valid frame')
        fixed = self._clean_selected(known, known_mask, prepared['local'], 'known')
        # Tensor.clone preserves autograd, unlike Python deepcopy of nonleaf
        # tensors. New dictionary + cloned tensors isolates all supplied values.
        conditions = {key: value.clone() if torch.is_tensor(value) else value for key, value in prepared.items()}
        indicator = self.known_embedding(known_mask.long()).to(conditions['local'])
        conditions['local'] = torch.where(valid[..., None], conditions['local'] + indicator, 0.)
        return conditions, fixed, unknown

    def flow_loss_prefix(self, target, valid, h0, identity, affect, local, noise, time,
                         *, known, known_mask):
        """Clean known tokens condition flow; target/noise exist only on unknowns."""
        conditions, fixed, unknown = self.prepare_prefix_conditions(
            valid, h0, identity, affect, local, known=known, known_mask=known_mask)
        target = self._clean_selected(target, unknown, conditions['local'], 'target')
        start = self._clean_selected(noise, unknown, conditions['local'], 'noise')
        if (not torch.is_tensor(time) or time.shape != (len(valid),)
                or not time.is_floating_point() or time.device != valid.device
                or not torch.isfinite(time).all() or ((time < 0.) | (time > 1.)).any()):
            raise ValueError('Flow time must be finite [B] within [0,1]')
        fraction = time[:, None, None]
        moving = (1. - fraction) * start + fraction * target
        x = torch.where(known_mask[..., None], fixed, moving)
        prediction = self.velocity(x, time, conditions)
        error = torch.where(unknown[..., None], prediction - (target - start), 0.)
        return error.square().sum() / (unknown.sum() * 9)

    def decode_prefix(self, valid, h0, identity, affect, local, noise,
                      *, known, known_mask, steps=12):
        """Return full prefix+generated chunk, retaining known coefficients exactly."""
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        conditions, fixed, unknown = self.prepare_prefix_conditions(
            valid, h0, identity, affect, local, known=known, known_mask=known_mask)
        start = self._clean_selected(noise, unknown, conditions['local'], 'noise')
        x = torch.where(known_mask[..., None], fixed, start)
        for index in range(steps):
            time = x.new_full((len(x),), index / steps)
            velocity = self.velocity(x, time, conditions)
            moving = torch.where(unknown[..., None], x + velocity / steps, 0.)
            x = torch.where(known_mask[..., None], fixed, moving)
        return x


__all__ = ['PrefixUpperFlow']
