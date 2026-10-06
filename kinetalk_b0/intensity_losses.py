"""Quality-aware ordinal supervision for MEAD expression intensity.

The renderer receives one clip at a time, while MEAD provides useful
same-speaker/same-sentence/emotion views at intensity 1/2/3.  This module only
contains auditable metadata grouping and differentiable pairwise losses; the
training loop decides how paired views are materialized.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch.nn import functional as F


def build_intensity_triplets(
    speaker: Sequence[object],
    sentence: Sequence[object],
    emotion: Sequence[int],
    intensity: Sequence[int],
    *,
    levels: Sequence[int] = (1, 2, 3),
    intensity_valid: Sequence[bool] | None = None,
) -> torch.Tensor:
    """Return deterministic ``[N, len(levels)]`` indices for complete groups.

    A group is keyed by speaker, sentence and emotion.  Neutral intensity 0 is
    intentionally excluded by the default levels.  Duplicate rows at one
    level are rejected rather than silently choosing an arbitrary clip.
    """
    lengths = {len(speaker), len(sentence), len(emotion), len(intensity)}
    if intensity_valid is not None:
        lengths.add(len(intensity_valid))
    if len(lengths) != 1:
        raise ValueError("intensity metadata fields must have equal length")
    levels = tuple(int(v) for v in levels)
    if len(levels) < 2 or len(set(levels)) != len(levels):
        raise ValueError("levels must contain at least two distinct values")
    valid = [True] * len(intensity) if intensity_valid is None else [bool(v) for v in intensity_valid]
    groups: dict[tuple[object, object, int], dict[int, int]] = defaultdict(dict)
    for index, (spk, sent, emo, lev, keep) in enumerate(zip(speaker, sentence, emotion, intensity, valid)):
        if not keep or int(lev) not in levels:
            continue
        key = (spk, sent, int(emo))
        level = int(lev)
        if level in groups[key]:
            raise ValueError(f"duplicate intensity {level} in group {key!r}")
        groups[key][level] = index
    rows = [tuple(values[level] for level in levels) for key, values in sorted(groups.items(), key=lambda item: repr(item[0]))
            if all(level in values for level in levels)]
    if not rows:
        return torch.empty((0, len(levels)), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def intensity_group_audit(
    speaker: Sequence[object], sentence: Sequence[object], emotion: Sequence[int],
    intensity: Sequence[int], *, levels: Sequence[int] = (1, 2, 3),
    intensity_valid: Sequence[bool] | None = None,
) -> dict[str, object]:
    """Summarize complete and incomplete same-content intensity groups."""
    triplets = build_intensity_triplets(
        speaker, sentence, emotion, intensity, levels=levels, intensity_valid=intensity_valid,
    )
    valid = [True] * len(intensity) if intensity_valid is None else [bool(v) for v in intensity_valid]
    groups: dict[tuple[object, object, int], set[int]] = defaultdict(set)
    for spk, sent, emo, lev, keep in zip(speaker, sentence, emotion, intensity, valid):
        if keep and int(lev) in set(int(v) for v in levels):
            groups[(spk, sent, int(emo))].add(int(lev))
    complete = int(len(triplets))
    return {
        "levels": [int(v) for v in levels],
        "groups": len(groups),
        "complete_groups": complete,
        "incomplete_groups": len(groups) - complete,
        "complete_rows": complete * len(tuple(levels)),
        "triplets": triplets,
    }


def masked_rms(values: torch.Tensor, valid: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Compute one uncentered RMS per clip over observed frames/channels."""
    if values.ndim != 3 or valid.shape != values.shape[:2] or channel_mask.shape != (values.shape[0], values.shape[-1]):
        raise ValueError("values/valid/channel_mask shapes are incompatible")
    if valid.dtype != torch.bool or channel_mask.dtype != torch.bool:
        raise ValueError("valid and channel_mask must be Boolean")
    observed = valid[..., None] & channel_mask[:, None, :]
    count = observed.sum((1, 2))
    clean = torch.where(observed, values, torch.zeros_like(values))
    energy = clean.square().sum((1, 2)) / count.clamp_min(1).to(values.dtype)
    result = energy.clamp_min(0).sqrt()
    return torch.where(count > 0, result, torch.zeros_like(result))


def ordinal_intensity_loss(
    predicted_energy: torch.Tensor,
    triplets: torch.Tensor,
    *,
    margin: float = 0.0,
    target_energy: torch.Tensor | None = None,
    target_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Penalize a higher-intensity view whose mouth energy is too small.

    Adjacent levels only (1<2 and 2<3) are used.  When ``target_energy`` is
    supplied, pairs whose observed target ordering disagrees are skipped; this
    prevents noisy labels from forcing a universal monotonic law.  The returned
    audit counts make the effective supervision visible in training logs.
    """
    if predicted_energy.ndim != 1 or not predicted_energy.is_floating_point():
        raise ValueError("predicted_energy must be floating [N]")
    if triplets.ndim != 2 or triplets.shape[1] < 2 or triplets.dtype not in (torch.int32, torch.int64):
        raise ValueError("triplets must be integer [groups,levels]")
    if triplets.numel() and (triplets.min() < 0 or triplets.max() >= len(predicted_energy)):
        raise ValueError("triplet index is outside predicted_energy")
    if margin < 0 or target_tolerance < 0:
        raise ValueError("margin and target_tolerance must be nonnegative")
    if target_energy is not None:
        if target_energy.shape != predicted_energy.shape or not target_energy.is_floating_point():
            raise ValueError("target_energy must match predicted_energy")
        if not torch.isfinite(target_energy).all():
            raise ValueError("target_energy must be finite")
    if triplets.numel() == 0:
        return predicted_energy.sum() * 0.0, {"candidate_pairs": 0, "used_pairs": 0}
    pairs = [(0, 1), (1, 2)] if triplets.shape[1] >= 3 else [(0, 1)]
    losses, used = [], 0
    for low, high in pairs:
        low_idx, high_idx = triplets[:, low], triplets[:, high]
        keep = torch.isfinite(predicted_energy[low_idx]) & torch.isfinite(predicted_energy[high_idx])
        if target_energy is not None:
            keep = keep & (target_energy[high_idx] >= target_energy[low_idx] - target_tolerance)
        if keep.any():
            losses.append(F.relu(float(margin) + predicted_energy[low_idx[keep]] - predicted_energy[high_idx[keep]]))
            used += int(keep.sum())
    candidate = len(pairs) * int(triplets.shape[0])
    if not losses:
        return predicted_energy.sum() * 0.0, {"candidate_pairs": candidate, "used_pairs": 0}
    return torch.cat(losses).mean(), {"candidate_pairs": candidate, "used_pairs": used}
