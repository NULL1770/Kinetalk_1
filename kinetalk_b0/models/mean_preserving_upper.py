"""Replace upper-face dynamics while retaining the baseline temporal mean."""
from __future__ import annotations

import torch

from .slow_state_affect import UPPER_INDICES


def compose_mean_preserving_upper(baseline, dynamic_upper, valid):
    """Compose baseline mean + centered dynamics in UPPER_INDICES order.

    Inputs are floating baseline [B,T,52], matching dynamic_upper [B,T,9],
    and Boolean valid [B,T]. Every sequence needs an observed frame. The
    baseline's original nine-channel dynamics are replaced, not added to.
    All other channels and every invalid frame retain baseline values exactly.

    Only observed values must be finite. Padding is excluded before arithmetic
    so NaN padding cannot enter the mean or its gradients. No target motion,
    clipping, or implicit stop-gradient is used. The preserved mean is over
    the supplied valid window; this is not a causal streaming composition.
    """
    if (not torch.is_tensor(baseline) or baseline.ndim != 3
            or baseline.shape[-1] != 52 or min(baseline.shape[:2]) < 1
            or not baseline.is_floating_point()):
        raise ValueError('baseline must be floating nonempty [B,T,52]')
    if (not torch.is_tensor(dynamic_upper)
            or dynamic_upper.shape != (*baseline.shape[:2], len(UPPER_INDICES))
            or dynamic_upper.dtype != baseline.dtype or dynamic_upper.device != baseline.device):
        raise ValueError('dynamic_upper must match baseline dtype/device with shape [B,T,9]')
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != baseline.shape[:2] or valid.device != baseline.device
            or not valid.any(dim=1).all()):
        raise ValueError('valid must be Boolean [B,T] with an observed frame per sequence')
    if not torch.isfinite(baseline[valid]).all():
        raise ValueError('Observed baseline must be finite')
    if not torch.isfinite(dynamic_upper[valid]).all():
        raise ValueError('Observed dynamic_upper must be finite')

    indices = list(UPPER_INDICES)
    output = baseline.clone()
    accumulation_dtype = torch.float32 if baseline.dtype in (torch.float16, torch.bfloat16) else baseline.dtype
    # Select observations before reduction, including internal mask gaps.
    # Appending padding then cannot alter the reduction shape or summation order.
    for row in range(len(baseline)):
        mask = valid[row]
        base = baseline[row, mask][:, indices].to(accumulation_dtype)
        dynamic = dynamic_upper[row, mask].to(accumulation_dtype)
        upper = base.mean(dim=0) + (dynamic - dynamic.mean(dim=0))
        row_output = output[row, mask].clone()
        row_output[:, indices] = upper.to(baseline.dtype)
        output[row, mask] = row_output
    return output


__all__ = ['compose_mean_preserving_upper', 'UPPER_INDICES']
