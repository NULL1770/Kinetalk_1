"""Optional temporal smoothing of an already generated upper-face carrier.

This is deterministic output processing, not a trained motion generator or
colored-noise sampler. It reads only a frozen full-face prediction and its
valid-frame mask. Raw outputs remain available for paired evaluation; there
is no target motion, coefficient clamp, RMS restoration, or random detail.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .slow_state_affect import UPPER_INDICES


def _run_groups(valid: torch.Tensor) -> dict[int, list[tuple[int, int, int]]]:
    """Group independent contiguous runs by length for batched convolution."""
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for row, flags in enumerate(valid.detach().cpu().tolist()):
        left = None
        for position, observed in enumerate(flags + [False]):
            if observed and left is None:
                left = position
            elif not observed and left is not None:
                groups.setdefault(position - left, []).append((row, left, position))
                left = None
    return groups


def smooth_motion_carrier(
    base_full: torch.Tensor,
    valid: torch.Tensor,
    window: int = 5,
) -> dict[str, torch.Tensor]:
    """Return ``{'raw': base_full, 'smoothed': protected_prediction}``.

    The input is floating [B,T,52], with Boolean [B,T] validity. Only the
    nine brow/expression-eye channels in UPPER_INDICES are smoothed. Allowed
    windows are 1, 3, 5, 7 and 9 native frames; window 5 uses the triangular
    kernel [1,2,3,2,1]/9. Window 1 returns an exact copy of the input.

    Each continuous valid run is processed independently. At a run boundary,
    kernel weights are renormalized over available frames. The filtering
    delta is centered within each run/channel, preserving run means and
    hence each clip/channel mean without coupling opposite sides of a gap.
    Other 43 channels and invalid frames are copied bit-for-bit. No range
    clamp or amplitude/RMS renormalization is applied: smoothing can reduce
    motion amplitude, and mean restoration can produce out-of-range values.

    This is an offline operation: symmetric filtering uses future frames,
    and mean protection uses the complete valid run. ``raw`` aliases the
    input, which this function never modifies; ``smoothed`` owns a new tensor.
    Gradients are preserved for diagnostics or a separately declared use.
    """
    if (not torch.is_tensor(base_full) or base_full.ndim != 3
            or base_full.shape[-1] != 52 or min(base_full.shape[:2]) < 1
            or not base_full.is_floating_point()):
        raise ValueError("base_full must be nonempty floating [B,T,52]")
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != base_full.shape[:2]
            or valid.device != base_full.device or not valid.any(1).all()):
        raise ValueError("valid must be Boolean [B,T] with an observed frame per clip")
    if type(window) is not int or window not in (1, 3, 5, 7, 9):
        raise ValueError("window must be one of 1, 3, 5, 7, 9")
    if not torch.isfinite(base_full[valid]).all():
        raise ValueError("observed base_full values must be finite")

    output = base_full.clone()
    if window == 1:
        return {"raw": base_full, "smoothed": output}

    indices = list(UPPER_INDICES)
    upper = base_full[..., indices]
    dtype = torch.float32 if base_full.dtype in (torch.float16, torch.bfloat16) else base_full.dtype
    radius = window // 2
    clock = torch.arange(window, device=base_full.device, dtype=dtype)
    weights = (radius + 1 - (clock - radius).abs())
    weights = weights / weights.sum()
    kernel = weights.reshape(1, 1, window).expand(len(indices), 1, window).contiguous()

    for length, runs in _run_groups(valid).items():
        # Invalid payloads are never used in differentiable arithmetic. This
        # also makes poisoned NaN padding safe for model/input gradients.
        source = torch.stack([upper[row, left:right] for row, left, right in runs]).to(dtype)
        filtered = F.conv1d(source.transpose(1, 2), kernel, padding=radius,
                            groups=len(indices)).transpose(1, 2)
        support = F.conv1d(source.new_ones(1, 1, length), weights.reshape(1, 1, window),
                           padding=radius).transpose(1, 2)
        filtered = filtered / support
        delta = filtered - source
        # Center each run, not the whole clip: changing a different run must
        # not affect this run through the protection correction.
        protected = source + delta - delta.mean(dim=1, keepdim=True)
        protected = protected.to(base_full.dtype)
        for item, (row, left, right) in enumerate(runs):
            output[row, left:right, indices] = protected[item]
    return {"raw": base_full, "smoothed": output}


__all__ = ["smooth_motion_carrier"]
