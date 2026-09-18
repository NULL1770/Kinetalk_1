"""Fixed-clock audio probabilities over a separately fitted motion dictionary.

Only acoustic frames and frozen global audio features enter this controller.
Dictionary values, target distances and training-only statistics are owned by
the driver. A zero-initialized output starts all tokens at equal probability.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _statistics(mean, std, name):
    if (not torch.is_tensor(mean) or not torch.is_tensor(std)
            or mean.ndim != 1 or std.shape != mean.shape or mean.numel() == 0
            or not mean.is_floating_point() or not std.is_floating_point()
            or mean.device != std.device or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError(f'{name} statistics must be finite vectors with positive standard deviation')
    return mean.detach().float().clone(), std.detach().float().clone()


class _MaskedTemporalBlock(nn.Module):
    """Partial convolution retains missing locations on their original clock.

    Unsupported input taps contribute zero. Renormalizing the convolution by
    its observed tap count avoids treating a hole or padded tail as acoustic
    silence. The center must itself be observed to produce a frame output.
    """

    def __init__(self, hidden, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(hidden, hidden, 5, padding=2*dilation,
                              dilation=dilation, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.register_buffer('support_kernel', torch.ones(1, 1, 5))

    def forward(self, value, valid):
        clean = torch.where(valid[..., None], value, 0.)
        support = F.conv1d(valid[:, None].to(value.dtype), self.support_kernel,
                           padding=2*self.dilation, dilation=self.dilation)
        update = self.conv(clean.transpose(1, 2)) * (5/support.clamp_min(1))
        update = update.transpose(1, 2) + self.bias
        return torch.where(valid[..., None], clean + F.silu(update), 0.)


class ClockedMotionPrior(nn.Module):
    """Small temporal acoustic controller; defaults are H32, hidden64 and K128.

    Feature/global dimensions follow the supplied fit-only mean/std vectors
    (normally 1540 and 65). Buffers are detached copies and are never refitted
    from a query. Each of four bins keeps its fixed native-clock boundaries;
    gaps and tail padding are neither compressed nor stretched. Entirely
    missing acoustic rows are allowed and yield a global-only prediction.
    """

    def __init__(self, feature_mean, feature_std, global_mean, global_std,
                 *, horizon=32, hidden=64, k=128):
        super().__init__()
        if (type(horizon) is not int or horizon < 4
                or any(type(v) is not int or v < 1 for v in (hidden, k))):
            raise ValueError('Integer horizon >=4 and positive hidden/token sizes required')
        fm, fs = _statistics(feature_mean, feature_std, 'Acoustic')
        gm, gs = _statistics(global_mean, global_std, 'Global')
        self.horizon, self.hidden, self.k = horizon, hidden, k
        self.feature_dim, self.global_dim = fm.numel(), gm.numel()
        self.register_buffer('feature_mean', fm)
        self.register_buffer('feature_std', fs)
        self.register_buffer('global_mean', gm)
        self.register_buffer('global_std', gs)
        self.input = nn.Linear(self.feature_dim, hidden)
        self.blocks = nn.ModuleList(_MaskedTemporalBlock(hidden, dilation)
                                    for dilation in (1, 2))
        self.hidden_layer = nn.Linear(4*hidden + self.global_dim, hidden)
        self.output = nn.Linear(hidden, k)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def temporal_descriptor(self, features, valid):
        """Return ordered four-bin audio descriptors [B,4*hidden]."""
        if (not torch.is_tensor(features) or not features.is_floating_point()
                or features.ndim != 3 or features.shape[0] < 1
                or features.shape[1:] != (self.horizon, self.feature_dim)
                or features.device != self.feature_mean.device
                or features.dtype != self.feature_mean.dtype
                or not torch.is_tensor(valid) or valid.dtype != torch.bool
                or valid.shape != features.shape[:2] or valid.device != features.device
                or not torch.isfinite(features[valid]).all()):
            raise ValueError('Finite observed audio [B,H,F] and matching Boolean mask required')
        # Mask before subtraction: even invalid NaNs cannot poison backward.
        clean = torch.where(valid[..., None], features, self.feature_mean)
        normalized = (clean-self.feature_mean)/self.feature_std
        value = torch.where(valid[..., None], F.silu(self.input(normalized)), 0.)
        for block in self.blocks:
            value = block(value, valid)
        bins = []
        for index in range(4):
            left, right = index*self.horizon//4, (index+1)*self.horizon//4
            observed = valid[:, left:right]
            numerator = torch.where(observed[..., None], value[:, left:right], 0.).sum(1)
            bins.append(numerator/observed.sum(1, keepdim=True).clamp_min(1))
        descriptor = torch.cat(bins, -1)
        if not torch.isfinite(descriptor).all():
            raise FloatingPointError('Temporal acoustic descriptor is nonfinite')
        return descriptor

    def forward(self, features, valid, global_features):
        descriptor = self.temporal_descriptor(features, valid)
        if (not torch.is_tensor(global_features) or not global_features.is_floating_point()
                or global_features.shape != (len(descriptor), self.global_dim)
                or global_features.device != self.global_mean.device
                or global_features.dtype != self.global_mean.dtype
                or not torch.isfinite(global_features).all()):
            raise ValueError('Finite global audio [B,G] must match controller dtype and device')
        global_code = (global_features-self.global_mean)/self.global_std
        hidden = F.silu(self.hidden_layer(torch.cat((descriptor, global_code), -1)))
        logits = self.output(hidden)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Motion-token logits are nonfinite')
        return logits


def categorical_energy_score(probabilities, target_distances, pairwise_distance):
    """Exact per-example energy score of a categorical trajectory distribution.

    For p[B,K], distance-to-target d[B,K], and token distance D[K,K], return
    ``sum_k p_k d_k - 0.5 sum_kl p_k p_l D_kl``. There is no sampling correction,
    nearest-token label, clamping, batch reduction or weighting here. Gradients
    propagate to probabilities and supplied distances. Proper-score guarantees
    require the caller to construct distances from an appropriate norm (for
    example Euclidean distance on the same normalized trajectory coordinates);
    symmetry/nonnegativity checks alone cannot certify that geometric property.
    """
    arrays = (probabilities, target_distances, pairwise_distance)
    if any(not torch.is_tensor(v) or not v.is_floating_point() for v in arrays):
        raise ValueError('Probabilities and distances must be floating tensors')
    if probabilities.ndim != 2 or min(probabilities.shape) < 1:
        raise ValueError('Nonempty probabilities [B,K] required')
    batch, k = probabilities.shape
    if target_distances.shape != (batch, k) or pairwise_distance.shape != (k, k):
        raise ValueError('Expected target distances [B,K] and token distances [K,K]')
    if any(v.dtype != probabilities.dtype or v.device != probabilities.device for v in arrays):
        raise ValueError('Probability/distance dtype and device must match')
    if any(not torch.isfinite(v).all() or (v < 0).any() for v in arrays):
        raise ValueError('Probabilities and distances must be finite and nonnegative')
    if not torch.allclose(probabilities.sum(-1), probabilities.new_ones(batch), atol=1e-5, rtol=1e-5):
        raise ValueError('Each probability row must sum to one')
    if (not torch.allclose(pairwise_distance, pairwise_distance.T, atol=1e-6, rtol=1e-6)
            or not torch.allclose(pairwise_distance.diagonal(), probabilities.new_zeros(k), atol=1e-6, rtol=0)):
        raise ValueError('Token distances must be symmetric with zero diagonal')
    return (probabilities*target_distances).sum(-1) - .5*((probabilities@pairwise_distance)*probabilities).sum(-1)
