"""Interpretable expression targets relative to independent neutral references.

Targets contain no emotion label and are not inference-time ground truth.
Callers must fit scales on their training split only and supply an independent
neutral anchor; this module never estimates an anchor from the query clip.
"""
from __future__ import annotations

import math

import torch

BROWS = (41, 42, 43, 44, 45)
EYES = (5, 6, 12, 13)
GROUPS = (BROWS, EYES)
SCALE_FLOOR = .02


def _deviation(motion, observed, neutral_anchor):
    if (not torch.is_tensor(motion) or motion.ndim != 3 or motion.shape[-1] != 52
            or not motion.is_floating_point() or min(motion.shape[:2]) < 1):
        raise ValueError('motion must be a nonempty floating [B,T,52] tensor')
    if (not torch.is_tensor(observed) or observed.shape != motion.shape
            or observed.dtype != torch.bool or observed.device != motion.device):
        raise ValueError('observed must be a matching Boolean [B,T,52] tensor')
    if (not torch.is_tensor(neutral_anchor) or neutral_anchor.shape != (motion.shape[0], 52)
            or not neutral_anchor.is_floating_point() or neutral_anchor.device != motion.device):
        raise ValueError('neutral_anchor must be explicit floating [B,52] on the same device')
    anchor = neutral_anchor[:, None].expand_as(motion)
    if not torch.isfinite(motion[observed]).all() or not torch.isfinite(anchor[observed]).all():
        raise ValueError('Nonfinite observed motion or neutral anchor')
    # Mask before arithmetic so missing NaN/Inf cannot contaminate values or gradients.
    difference = torch.where(observed, motion, 0) - torch.where(observed, anchor, 0)
    if not torch.isfinite(difference).all():
        raise ValueError('Nonfinite observed deviation')
    return difference


def _normalized(motion, observed, neutral_anchor, scales):
    delta = _deviation(motion, observed, neutral_anchor)
    if (not torch.is_tensor(scales) or scales.shape != (52,) or not scales.is_floating_point()
            or scales.device != motion.device or not torch.isfinite(scales).all()
            or (scales < SCALE_FLOOR).any()):
        raise ValueError('scales must be finite floating [52] with every value >= 0.02')
    return delta / scales


@torch.no_grad()
def fit_intensity_scales(train_motion, train_observed, train_neutral_anchor, *, floor=SCALE_FLOOR):
    """Fit per-channel RMS(raw motion - independent anchor) on training only.

    Missing channels receive the floor. An entirely absent brow or expression
    eye group is rejected. Neither per-query norm nor clip centering is used.
    The function accepts only supplied training tensors; the caller binds split
    provenance because a tensor alone cannot prove which split it came from.
    """
    if isinstance(floor, bool) or not math.isfinite(floor) or floor < SCALE_FLOOR:
        raise ValueError('floor must be finite and >= 0.02')
    delta = _deviation(train_motion, train_observed, train_neutral_anchor).double()
    for channels in GROUPS:
        if not train_observed[..., list(channels)].any():
            raise ValueError('Training split is missing an entire intensity group')
    count = train_observed.sum((0, 1))
    scale = (delta.square().sum((0, 1)) / count.clamp_min(1)).sqrt().clamp_min(floor)
    if not torch.isfinite(scale).all():
        raise ValueError('Nonfinite fitted scales')
    return scale.to(dtype=train_motion.dtype)


def regional_intensity(motion, observed, neutral_anchor, scales):
    """Return intensity and validity, both [B,T,1].

    Mean absolute normalized deviation is computed within each available group,
    then brows and expression eyes receive equal weight. A frame with no
    observed channel in either group is invalid and has a zero placeholder.
    Partially observed groups use only their observed channels. No label forces
    neutral clips to zero, and a persistent offset remains a nonzero intensity.
    """
    normalized = _normalized(motion, observed, neutral_anchor, scales).abs()
    means, valid = [], []
    for channels in GROUPS:
        count = observed[..., list(channels)].sum(-1, keepdim=True)
        means.append(normalized[..., list(channels)].sum(-1, keepdim=True) / count.clamp_min(1))
        valid.append(count > 0)
    valid = valid[0] & valid[1]
    return torch.where(valid, (means[0] + means[1]) * .5, 0), valid


def regional_window_activity(motion, observed, neutral_anchor, scales):
    """Auxiliary normalized ddof0 activity [B,2] and group-valid [B,2].

    Channel std uses its observed frames, with >=2 frames required. Each group
    averages available channel std. This is a readout, not a required condition.
    """
    normalized = _normalized(motion, observed, neutral_anchor, scales)
    count = observed.sum(1)
    mean = normalized.sum(1) / count.clamp_min(1)
    centered = torch.where(observed, normalized - mean[:, None], 0)
    std = torch.linalg.vector_norm(centered, dim=1) / count.clamp_min(1).sqrt()
    available = count >= 2
    values, valid = [], []
    for channels in GROUPS:
        mask = available[:, list(channels)]
        n = mask.sum(-1)
        values.append(torch.where(mask, std[:, list(channels)], 0).sum(-1) / n.clamp_min(1))
        valid.append(n > 0)
    return torch.stack(values, -1), torch.stack(valid, -1)
