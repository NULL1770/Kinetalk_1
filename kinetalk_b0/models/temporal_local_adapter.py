"""Small offline temporal corrections to frozen local audio features."""

import torch
from torch import nn
from torch.nn import functional as F


class _ExactResidualAdd(torch.autograd.Function):
    """Preserve a zero correction exactly, including the sign of zero."""

    @staticmethod
    def forward(ctx, baseline, residual):
        return torch.where(residual == 0, baseline, baseline + residual)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class TemporalLocalAdapter(nn.Module):
    """Mean-preserving rank-limited adapter; centering uses the whole clip.

    ``local`` is a floating tensor [batch, time, local_dim], and ``valid``
    is a boolean tensor [batch, time]. At least one frame per clip must be
    valid. Invalid payloads are excluded from all calculations and copied
    unchanged to the output. The caller owns freezing the source encoder.

    Initialization uses an isolated CPU RNG. Zero-initializing ``up`` makes
    the initial result bitwise identical to the source without preventing
    the first optimizer step from updating ``up``.
    """

    def __init__(self, local_dim=64, rank=8, *, init_seed=0):
        super().__init__()
        if (not isinstance(local_dim, int) or isinstance(local_dim, bool)
                or not isinstance(rank, int) or isinstance(rank, bool)
                or local_dim < 1 or rank < 1 or rank > local_dim):
            raise ValueError('Require integer dimensions with 1 <= rank <= local_dim')
        self.local_dim = local_dim
        self.rank = rank
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(init_seed)
            self.down = nn.Linear(local_dim, rank, bias=False, device='cpu')
            self.up = nn.Linear(rank, local_dim, bias=False, device='cpu')
            nn.init.zeros_(self.up.weight)

    @staticmethod
    def _center(value, mask):
        clean = torch.where(mask, value, torch.zeros_like(value))
        count = mask.sum(dim=1, keepdim=True).to(value.dtype)
        mean = clean.sum(dim=1, keepdim=True) / count
        return torch.where(mask, clean - mean, torch.zeros_like(clean))

    def forward(self, local, valid):
        if (local.ndim != 3 or local.shape[-1] != self.local_dim
                or not local.is_floating_point()
                or valid.dtype != torch.bool or valid.shape != local.shape[:2]
                or valid.device != local.device or local.shape[0] == 0
                or not valid.any(dim=1).all()
                or not torch.isfinite(local[valid]).all()):
            raise ValueError('Finite observed local features and a nonempty boolean frame mask required')
        mask = valid.unsqueeze(-1)
        deviation = self._center(local, mask)
        residual = self.up(F.silu(self.down(deviation)))
        residual = self._center(residual, mask)
        clean = torch.where(mask, local, torch.zeros_like(local))
        corrected = _ExactResidualAdd.apply(clean, residual)
        return torch.where(mask, corrected, local)
