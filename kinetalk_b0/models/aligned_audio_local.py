"""Native audio adapter for the frozen motion teacher's local coordinates.

This module predicts the teacher's low-rate controls, not an unconstrained
renderer-space local vector.  Callers keep their frozen audio-global result,
merge the three returned entries, and call ``system.project_affect`` to use
the teacher's shared local projection.  There is no global-affect head here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .label_guided_affect import TemporalBlock


class AlignedAudioLocal(nn.Module):
    """Predict rank-8, stride-4 controls from fitted native audio features.

    Defaults match the existing motion teacher; configurable dimensions allow
    the caller to match another explicitly configured teacher.  Frame logits
    are tanh-bounded *before* fixed-bin averaging and weighted centering, as in
    ``LowRateAffectEncoder``.  Centered controls therefore need not lie in
    [-1, 1].  No normalization statistics are fitted in this module.
    """

    def __init__(self, feature_mean, feature_std, *, hidden=128, rank=8, stride=4):
        super().__init__()
        if (not torch.is_tensor(feature_mean) or feature_mean.ndim != 1
                or not torch.is_tensor(feature_std) or feature_std.shape != feature_mean.shape
                or not feature_mean.is_floating_point() or not feature_std.is_floating_point()
                or feature_mean.numel() < 1 or not torch.isfinite(feature_mean).all()
                or not torch.isfinite(feature_std).all() or (feature_std <= 0).any()
                or feature_mean.device != feature_std.device):
            raise ValueError('Finite fitted feature statistics with positive std are required')
        if any(type(value) is not int or value < 1 for value in (hidden, rank, stride)):
            raise ValueError('hidden, rank and stride must be positive integers')
        self.rank, self.stride = rank, stride
        self.register_buffer('feature_mean', feature_mean.detach().float().clone())
        self.register_buffer('feature_std', feature_std.detach().float().clone())
        self.input = nn.Linear(feature_mean.numel(), hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, dilation) for dilation in (1, 2, 4, 8))
        self.control_head = nn.Linear(hidden, rank)
        self._zero_head()

    def _zero_head(self):
        nn.init.zeros_(self.control_head.weight)
        nn.init.zeros_(self.control_head.bias)

    def initialize_from_slow(self, slow):
        """Copy only a compatible SlowStateAffect acoustic trunk; zero the head.

        The fitted statistics must already match.  All compatibility checks
        precede copying, so a rejected warm start leaves this module intact.
        Global/local/state heads and classifiers are never copied or shared.
        The source module is not changed and receives no gradients through
        this independent adapter.
        """
        for name in ('feature_mean', 'feature_std'):
            source = getattr(slow, name, None)
            target = getattr(self, name)
            if (not torch.is_tensor(source) or source.shape != target.shape
                    or not torch.equal(source.detach().to(target), target)):
                raise ValueError('Warm start requires identical fitted feature statistics')
        states = {}
        for name in ('input', 'blocks'):
            source = getattr(slow, name, None)
            if not isinstance(source, nn.Module):
                raise ValueError(f'Warm start requires a compatible {name} module')
            state, expected = source.state_dict(), getattr(self, name).state_dict()
            if (state.keys() != expected.keys()
                    or any(state[key].shape != expected[key].shape for key in expected)):
                raise ValueError(f'Warm start requires compatible {name} parameters')
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError('Warm start parameters must be finite')
            states[name] = state
        # Dilation/padding are not in a state_dict; reject a same-shape trunk
        # that represents a different temporal operation.
        if len(slow.blocks) != len(self.blocks):
            raise ValueError('Warm start requires matching temporal blocks')
        for source, target in zip(slow.blocks, self.blocks):
            if (not isinstance(source, TemporalBlock)
                    or source.conv.dilation != target.conv.dilation
                    or source.conv.padding != target.conv.padding):
                raise ValueError('Warm start requires matching temporal block dilations')
        for name, state in states.items():
            getattr(self, name).load_state_dict(state, strict=True)
        self._zero_head()
        return self

    def forward(self, features, valid):
        if (not torch.is_tensor(features) or features.ndim != 3
                or not features.is_floating_point() or min(features.shape[:2]) < 1
                or features.shape[-1] != self.feature_mean.numel()
                or not torch.is_tensor(valid) or valid.shape != features.shape[:2]
                or valid.dtype != torch.bool or valid.device != features.device
                or features.device != self.feature_mean.device or not valid.any(1).all()):
            raise ValueError('Native audio features require matching nonempty Boolean [B,T] masks')
        if not torch.isfinite(features[valid]).all():
            raise ValueError('Observed native audio features must be finite')
        # Clear padding before normalization and the linear layer. Multiplying
        # NaNs by zero afterward would still poison parameter gradients.
        clean = torch.where(valid[..., None], features, self.feature_mean)
        normalized = (clean - self.feature_mean) / self.feature_std
        hidden = torch.where(valid[..., None], F.silu(self.input(normalized)), 0.)
        for block in self.blocks:
            hidden = block(hidden, valid)
        frame_controls = torch.where(valid[..., None], torch.tanh(self.control_head(hidden)), 0.)
        batch, frames, rank = frame_controls.shape
        extra = (-frames) % self.stride
        binned = F.pad(frame_controls, (0, 0, 0, extra)).reshape(batch, -1, self.stride, rank)
        weight = F.pad(valid.to(hidden.dtype), (0, extra)).reshape(batch, -1, self.stride).sum(-1)
        control_mask = weight > 0
        controls = binned.sum(2) / weight.unsqueeze(-1).clamp_min(1)
        mean = ((controls * weight.unsqueeze(-1)).sum(1, keepdim=True)
                / weight.sum(1, keepdim=True).unsqueeze(-1))
        controls = torch.where(control_mask[..., None], controls - mean, 0.)
        return {'controls': controls, 'control_mask': control_mask, 'control_weight': weight}
