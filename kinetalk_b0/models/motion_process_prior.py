"""Compact duration/delta motion priors with audio-only observation context.

The caller fits feature and motion scales on its training partition. This module
does not normalize against a query motion, find its event boundaries, or receive
motion during ``forward_context``. Event history is explicit: teacher history is
appropriate for likelihood training, while deployment must supply sampled state.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _RunConvBlock(nn.Module):
    """Three-tap depthwise convolution confined to contiguous observed runs.

    Missing neighbours replicate the current frame, including at internal gaps.
    This both prevents dilation from jumping a gap and makes constant inputs
    remain constant at run boundaries. No temporal position signal is added.
    """

    def __init__(self, hidden, dilation):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Parameter(torch.empty(3, hidden))
        nn.init.normal_(self.depthwise, std=1 / math.sqrt(3))
        self.mix = nn.Linear(hidden, hidden)

    def forward(self, value, valid, run_ids):
        normalized = self.norm(value)
        length = value.shape[1]
        positions = torch.arange(length, device=value.device)
        taps = []
        for offset in (-self.dilation, 0, self.dilation):
            source = positions + offset
            safe = source.clamp(0, length - 1)
            observed = (source >= 0) & (source < length)
            usable = (observed[None] & valid[:, safe] & valid
                      & (run_ids[:, safe] == run_ids))
            taps.append(torch.where(usable[..., None], normalized[:, safe], normalized))
        filtered = sum(tap * weight for tap, weight in zip(taps, self.depthwise))
        updated = value + self.mix(F.silu(filtered))
        return torch.where(valid[..., None], updated, torch.zeros_like(updated))


def _normal_nll(loc, log_scale, target):
    return .5 * ((target - loc) * torch.exp(-log_scale)).square() + log_scale + .5 * math.log(2 * math.pi)


def joint_event_nll(out, duration_index, delta):
    """Per-event negative log joint probability; delta is in fitted scale units."""
    logits, loc, log_scale = (out[key] for key in ('duration_logits', 'loc', 'log_scale'))
    if (logits.ndim != 2 or loc.shape != logits.shape or log_scale.shape != logits.shape
            or duration_index.shape != logits.shape[:1] or duration_index.dtype != torch.long
            or delta.shape != logits.shape[:1] or not delta.is_floating_point()
            or duration_index.device != logits.device or delta.device != logits.device
            or not torch.isfinite(logits).all() or not torch.isfinite(loc).all()
            or not torch.isfinite(log_scale).all() or not torch.isfinite(delta).all()
            or (duration_index < 0).any() or (duration_index >= logits.shape[1]).any()):
        raise ValueError('Finite event distributions, in-range duration indices and scalar deltas required')
    index = duration_index[:, None]
    categorical = -F.log_softmax(logits, dim=-1).gather(1, index).squeeze(1)
    selected_loc = loc.gather(1, index).squeeze(1)
    selected_scale = log_scale.gather(1, index).squeeze(1)
    return categorical + _normal_nll(selected_loc, selected_scale, delta)


def initial_nll(out, target):
    """Per-clip initial-state NLL, summed over independent motion groups."""
    loc, log_scale = out['loc'], out['log_scale']
    if (loc.ndim != 2 or log_scale.shape != loc.shape or target.shape != loc.shape
            or not target.is_floating_point() or target.device != loc.device
            or not torch.isfinite(loc).all() or not torch.isfinite(log_scale).all()
            or not torch.isfinite(target).all()):
        raise ValueError('Finite matching initial-state distribution and target required')
    return _normal_nll(loc, log_scale, target).sum(-1)


class MotionProcessPrior(nn.Module):
    """A small audio context encoder and persistent-state event distribution.

    ``level`` and ``previous_delta`` are scalar values in training-fitted motion
    scale units. ``previous_duration`` is in native frames (zero at a reset).
    The local encoder permits offline future *audio*, but never crosses an
    unobserved gap. ``pooled`` always encodes the observed clip's mean input via
    the same shared-weight path in both arms, independently of the local policy.
    In the global-only arm, the observed clip feature mean is replicated before
    the same encoder; every valid output frame is therefore time-constant.
    """

    def __init__(self, feature_dim=1540, hidden=96, groups=4,
                 durations=(4, 8, 12, 20, 32, 48, 64)):
        super().__init__()
        if (any(type(x) is not int or x < 1 for x in (feature_dim, hidden, groups))
                or not isinstance(durations, (list, tuple)) or not durations
                or any(type(x) is not int or x < 1 for x in durations)
                or tuple(sorted(set(durations))) != tuple(durations)):
            raise ValueError('Positive dimensions and strictly increasing integer durations required')
        self.feature_dim, self.hidden, self.groups = feature_dim, hidden, groups
        self.register_buffer('durations', torch.tensor(durations, dtype=torch.long))
        self.input = nn.Linear(feature_dim, hidden)
        self.blocks = nn.ModuleList(_RunConvBlock(hidden, dilation) for dilation in (1, 2, 4))
        self.output_norm = nn.LayerNorm(hidden)
        self.group_embedding = nn.Embedding(groups, 8)
        self.event_head = nn.Sequential(nn.Linear(2 * hidden + 8 + 3, hidden), nn.SiLU(),
                                        nn.Linear(hidden, 3 * len(durations)))
        self.initial_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                          nn.Linear(hidden, 2 * groups))

    def forward_context(self, features, valid, local_enabled=True):
        if (features.ndim != 3 or features.shape[-1] != self.feature_dim
                or not features.is_floating_point() or valid.dtype != torch.bool
                or valid.shape != features.shape[:2] or valid.device != features.device
                or features.shape[0] == 0 or features.shape[1] == 0
                or not valid.any(1).all() or not torch.isfinite(features[valid]).all()
                or type(local_enabled) is not bool):
            raise ValueError('Finite observed audio, nonempty Boolean masks and a Boolean local policy required')
        mask = valid[..., None]
        clean = torch.where(mask, features, torch.zeros_like(features))
        count = valid.sum(1, keepdim=True).to(features.dtype)[..., None]
        # Both arms receive exactly the same global audio representation. Using
        # mean(encode(local)) only in the local arm would also change its global
        # path and confound a comparison meant to isolate temporal information.
        global_input = clean.sum(1, keepdim=True) / count
        global_valid = torch.ones_like(valid[:, :1])
        global_hidden = self._encode(global_input, global_valid)
        pooled = global_hidden[:, 0]
        if not local_enabled:
            # One-value encoding followed by broadcast also avoids GEMM-tail
            # roundoff introducing tiny timing cues into the constant arm.
            return torch.where(mask, global_hidden.expand(-1, valid.shape[1], -1), 0.), pooled
        return self._encode(clean, valid), pooled

    def _encode(self, clean, valid):
        mask = valid[..., None]
        hidden = torch.where(mask, F.silu(self.input(clean)), 0.)
        # Equal cumulative invalid count means two observed frames lie in the
        # same contiguous run; checking endpoints alone would allow gap jumps.
        run_ids = (~valid).long().cumsum(1)
        for block in self.blocks:
            hidden = block(hidden, valid, run_ids)
        return torch.where(mask, self.output_norm(hidden), 0.)

    def _validate_pooled(self, pooled):
        if (pooled.ndim != 2 or pooled.shape[-1] != self.hidden
                or not pooled.is_floating_point() or not torch.isfinite(pooled).all()):
            raise ValueError('Finite pooled audio context required')

    def event_distribution(self, hidden, pooled, group, level, previous_delta, previous_duration):
        self._validate_pooled(pooled)
        if (hidden.shape != pooled.shape or hidden.device != pooled.device
                or not hidden.is_floating_point() or not torch.isfinite(hidden).all()
                or group.shape != hidden.shape[:1] or group.dtype != torch.long
                or group.device != hidden.device or (group < 0).any() or (group >= self.groups).any()):
            raise ValueError('Matching event contexts and valid group indices required')
        for value in (level, previous_delta, previous_duration):
            if (value.shape != hidden.shape[:1] or value.device != hidden.device
                    or not torch.isfinite(value).all()):
                raise ValueError('Finite scalar event history required')
        if (previous_duration < 0).any():
            raise ValueError('Previous duration cannot be negative')
        history = torch.stack((level, previous_delta,
                               previous_duration.to(hidden.dtype) / self.durations[-1]), dim=-1).to(hidden.dtype)
        encoded = torch.cat((hidden, pooled, self.group_embedding(group), history), dim=-1)
        logits, loc, log_scale = self.event_head(encoded).chunk(3, dim=-1)
        return {'duration_logits': logits, 'loc': loc, 'log_scale': log_scale.clamp(-3., 2.)}

    def initial_distribution(self, pooled):
        self._validate_pooled(pooled)
        loc, log_scale = self.initial_head(pooled).chunk(2, dim=-1)
        return {'loc': loc, 'log_scale': log_scale.clamp(-3., 2.)}

    @torch.no_grad()
    def sample_event(self, hidden, pooled, group, level, previous_delta, previous_duration, *, generator=None):
        out = self.event_distribution(hidden, pooled, group, level, previous_delta, previous_duration)
        index = torch.multinomial(out['duration_logits'].softmax(-1), 1, generator=generator)
        loc = out['loc'].gather(1, index).squeeze(1)
        log_scale = out['log_scale'].gather(1, index).squeeze(1)
        noise = torch.randn(loc.shape, dtype=loc.dtype, device=loc.device, generator=generator)
        index = index.squeeze(1)
        return {'duration_index': index, 'duration': self.durations[index],
                'delta': loc + log_scale.exp() * noise}

    @torch.no_grad()
    def sample_initial(self, pooled, *, generator=None):
        out = self.initial_distribution(pooled)
        noise = torch.randn(out['loc'].shape, dtype=pooled.dtype, device=pooled.device, generator=generator)
        return out['loc'] + out['log_scale'].exp() * noise

    joint_event_nll = staticmethod(joint_event_nll)
    initial_nll = staticmethod(initial_nll)
