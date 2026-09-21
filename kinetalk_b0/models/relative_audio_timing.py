"""Small, identity-free timing head with clip-relative acoustic inputs.

The global expression mean is supplied by the frozen renderer outside this
module. Query motion and identity codes are not inference inputs. Statistics
and the bottleneck projection are fitted on TRAIN only.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .slow_state_affect import masked_slow_state, lift_slow_state


def masked_center(x, valid):
    clean = torch.where(valid[..., None], x, 0.)
    return torch.where(valid[..., None], clean - clean.sum(1, keepdim=True)
                       / valid.sum(1)[:, None, None].clamp_min(1), 0.)


def relative_features(features, valid, mean, std):
    """Remove clip offsets; normalize voiced pitch without unvoiced zeros."""
    x = torch.where(valid[..., None], (torch.where(valid[..., None], features, mean) - mean) / std, 0.)
    x = masked_center(x, valid)
    p = features[..., -4:]
    voiced = valid & (p[..., 3] > .5)
    f0 = torch.where(voiced, p[..., 0], 0.)
    avg = f0.sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
    pitch = torch.where(voiced, (f0 - avg) / std[-4], 0.)
    x = torch.cat((x[..., :-4], pitch[..., None], x[..., -3:]), -1)
    return torch.where(valid[..., None], x, 0.)


class RelativeAudioTiming(nn.Module):
    def __init__(self, feature_mean, feature_std, projection, channel_scales,
                 dynamic_scales, *, hidden=48, stride=16, dropout=.2):
        super().__init__()
        if feature_mean.shape != feature_std.shape or feature_mean.numel() != 1540:
            raise ValueError('Expected TRAIN-fitted 1540-dimensional audio statistics')
        if projection.ndim != 2 or projection.shape[0] != 1536:
            raise ValueError('PCA projection must describe acoustic embeddings only')
        if (feature_std <= 0).any() or (dynamic_scales <= 0).any():
            raise ValueError('Positive scales required')
        for name, value in [('feature_mean', feature_mean), ('feature_std', feature_std),
                            ('projection', projection), ('channel_scales', channel_scales),
                            ('dynamic_scales', dynamic_scales)]:
            self.register_buffer(name, value.detach().float().clone())
        self.hidden, self.stride, self.dropout = hidden, stride, dropout
        width = projection.shape[1] + 4
        self.input = nn.Linear(width * 2, hidden)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d) for d in (1, 2, 4))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in self.convs)
        self.head = nn.Linear(hidden, 4)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def export_config(self):
        return {'hidden': self.hidden, 'stride': self.stride, 'dropout': self.dropout}

    def forward(self, features, valid, mode='audio'):
        if mode not in ('audio', 'static', 'reverse') or valid.dtype != torch.bool or not valid.any(1).all():
            raise ValueError('Invalid timing inputs')
        if not torch.isfinite(features[valid]).all():
            raise ValueError('Nonfinite observed audio')
        frames = features.shape[1]
        end = int(valid.any(0).nonzero()[-1]) + 1
        features, valid = features[:, :end], valid[:, :end]
        x = relative_features(features, valid, self.feature_mean, self.feature_std)
        x = torch.cat((x[..., :1536] @ self.projection, x[..., -4:]), -1)
        diff = torch.zeros_like(x)
        diff[:, 1:] = torch.where((valid[:, 1:] & valid[:, :-1])[..., None], x[:, 1:] - x[:, :-1], 0.)
        x = torch.cat((x, diff), -1)
        if self.training:
            # Locked feature dropout cannot fabricate a framewise rhythm.
            keep = F.dropout(torch.ones(len(x), 1, x.shape[-1], device=x.device), self.dropout, True)
            x = x * keep
        h = torch.where(valid[..., None], F.silu(self.input(x)), 0.)
        for norm, conv in zip(self.norms, self.convs):
            z = torch.where(valid[..., None], F.silu(norm(h)), 0.)
            h = torch.where(valid[..., None], h + F.silu(conv(z.transpose(1, 2)).transpose(1, 2)), 0.)
        state = masked_slow_state(self.head(h), valid, stride=self.stride)['state']
        state = masked_center(state, valid) * self.dynamic_scales
        if mode == 'static':
            state = state * 0.
        elif mode == 'reverse':
            state = state.clone()
            for i in range(len(state)):
                ix = valid[i].nonzero().flatten()
                state[i, ix] = state[i, ix.flip(0)].clone()
        delta = lift_slow_state(state, self.channel_scales)
        if end < frames:
            state = F.pad(state, (0, 0, 0, frames-end)); delta = F.pad(delta, (0, 0, 0, frames-end))
        return {'state': state, 'delta': delta}
