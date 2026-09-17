"""Training-only emotion directions and a signed, scalar expression field.

An emotion ray is the mean neutral-relative expression direction, fitted only
on training clips and with equal weight for each speaker.  The time-varying
coordinate is a signed projection onto that direction, not an unsigned motion
energy.  Audio emotion probabilities can select a ray at inference; neither
the query motion nor its emotion label is needed by ``direction_from_probs``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F


# In the existing 52-controller representation these are blink/gaze channels.
# This is target construction, not a separate region in the generation model.
NUISANCE_CHANNELS_52 = (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)


def _validate_motion(residual: torch.Tensor, valid: torch.Tensor) -> None:
    if residual.ndim != 3 or valid.shape != residual.shape[:2]:
        raise ValueError("residual [B,T,C] and valid [B,T] are required")
    if not residual.is_floating_point() or valid.dtype != torch.bool:
        raise ValueError("residual must be floating point and valid must be boolean")
    if residual.shape[1] == 0 or residual.shape[2] == 0:
        raise ValueError("motion must have at least one frame and channel")
    if residual.device != valid.device:
        raise ValueError("residual and valid must be on the same device")


@torch.no_grad()
def fit_emotion_rays(
    residual: torch.Tensor,
    valid: torch.Tensor,
    channel_mask: torch.Tensor,
    speaker_id: torch.Tensor,
    emotion_id: torch.Tensor,
    train_ids: torch.Tensor | Sequence[int],
    num_emotions: int,
    stride: int = 4,
    *,
    neutral_id: int = 0,
    exclude_nuisance: bool = True,
    missing_neutral: str = "error",
) -> dict[str, Any]:
    """Fit unit emotion directions without reading held-out clip values.

    Each speaker contributes the mean of their emotional clip means minus
    their neutral clip mean.  Speaker directions are averaged equally before
    normalization.  Channels must be observed in *every training clip*; with
    52 channels, blink/gaze can additionally be excluded.  No neutral clip from
    validation is borrowed.  ``missing_neutral='skip'`` explicitly records and
    omits speakers lacking a training neutral reference.

    ``stride`` is recorded for projection reproducibility; directions are
    computed from all valid frames, so partial bins cannot bias clip means.
    """
    _validate_motion(residual, valid)
    batch, _, channels = residual.shape
    if channel_mask.shape != (batch, channels) or channel_mask.dtype != torch.bool:
        raise ValueError("channel_mask must be boolean [B,C]")
    if speaker_id.shape != (batch,) or emotion_id.shape != (batch,):
        raise ValueError("speaker_id and emotion_id must have shape [B]")
    if any(x.device != residual.device for x in (channel_mask, speaker_id, emotion_id)):
        raise ValueError("fit inputs must be on the same device")
    if speaker_id.is_floating_point() or emotion_id.is_floating_point():
        raise ValueError("speaker_id and emotion_id must be integer tensors")
    if num_emotions < 1 or not 0 <= neutral_id < num_emotions or stride < 1:
        raise ValueError("positive num_emotions/stride and an in-range neutral_id are required")
    if missing_neutral not in ("error", "skip"):
        raise ValueError("missing_neutral must be 'error' or 'skip'")
    ids = torch.as_tensor(train_ids, device=residual.device)
    if ids.ndim != 1 or ids.numel() == 0 or ids.is_floating_point() or ids.dtype == torch.bool:
        raise ValueError("train_ids must be a nonempty vector of integer indices")
    ids = ids.long()
    if ids.min() < 0 or ids.max() >= batch or ids.unique().numel() != ids.numel():
        raise ValueError("train_ids must be unique, in-range indices")

    # Select before inspecting values: held-out labels, masks, NaNs, and motion
    # must not alter the fitted supervision or channel selection.
    x, mask = residual[ids], valid[ids]
    spk, emo = speaker_id[ids], emotion_id[ids]
    if (emo < 0).any() or (emo >= num_emotions).any():
        raise ValueError("training emotion IDs are outside num_emotions")
    if not mask.any(1).all():
        raise ValueError("every training clip must contain at least one valid frame")
    observed = channel_mask[ids].all(0)
    excluded = list(NUISANCE_CHANNELS_52) if exclude_nuisance and channels == 52 else []
    observed[excluded] = False
    if not observed.any():
        raise ValueError("no expression channels are shared by the training clips")
    x = torch.where(mask[..., None] & observed[None, None], x, 0)
    if not torch.isfinite(x).all():
        raise ValueError("observed training motion contains non-finite values")
    clip_means = x.sum(1) / mask.sum(1, keepdim=True).to(x.dtype)

    speakers = spk.unique(sorted=True)
    anchor_speakers, anchors, skipped = [], [], []
    directions: list[list[torch.Tensor]] = [[] for _ in range(num_emotions)]
    clip_counts = torch.zeros(num_emotions, device=x.device, dtype=torch.long)
    speaker_counts = torch.zeros_like(clip_counts)
    for speaker in speakers:
        own = spk == speaker
        neutral = own & (emo == neutral_id)
        if not neutral.any():
            skipped.append(int(speaker))
            continue
        anchor = clip_means[neutral].mean(0)
        anchor_speakers.append(speaker)
        anchors.append(anchor)
        for emotion in range(num_emotions):
            selected = own & (emo == emotion)
            if selected.any():
                clip_counts[emotion] += selected.sum()
                speaker_counts[emotion] += 1
                if emotion != neutral_id:
                    directions[emotion].append(clip_means[selected].mean(0) - anchor)
    if skipped and missing_neutral == "error":
        raise ValueError(f"speakers lack training neutral references: {skipped}")
    if not anchors:
        raise ValueError("no training speakers have a neutral reference")

    rays = x.new_zeros(num_emotions, channels)
    raw_norms = x.new_zeros(num_emotions)
    active = torch.zeros(num_emotions, device=x.device, dtype=torch.bool)
    eps = max(float(torch.finfo(x.dtype).eps), 1e-8)
    for emotion, values in enumerate(directions):
        if values:
            direction = torch.stack(values).mean(0)
            norm = direction.norm()
            raw_norms[emotion] = norm
            if norm > eps:
                rays[emotion] = direction / norm
                active[emotion] = True
    return {
        "target_name": "neutral_relative_emotion_ray",
        "rays": rays,
        "active": active,
        "raw_norms": raw_norms,
        "observed_channels": observed,
        "train_ids": ids.clone(),
        "counts": {"clips": clip_counts, "speakers": speaker_counts},
        "neutral_anchors": torch.stack(anchors),
        "anchor_speaker_ids": torch.stack(anchor_speakers),
        "skipped_speaker_ids": skipped,
        "neutral_id": neutral_id,
        "num_emotions": num_emotions,
        "stride": stride,
        "excluded_channels": excluded,
        "missing_neutral": missing_neutral,
    }


def project_ray_field(
    residual: torch.Tensor,
    valid: torch.Tensor,
    rays_for_clip: torch.Tensor,
    stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project centered, valid-frame motion onto one ray per clip.

    Returns scalar ``[B,K,1]``, valid-frame counts ``[B,K]``, and centered
    motion-bin means ``[B,K,C]``.  A zero ray (e.g. neutral) gives zero scalar
    supervision.  Partial final bins retain their actual frame counts.
    """
    _validate_motion(residual, valid)
    if stride < 1 or rays_for_clip.shape != (residual.shape[0], residual.shape[2]):
        raise ValueError("positive stride and rays_for_clip [B,C] are required")
    if rays_for_clip.device != residual.device or not torch.isfinite(rays_for_clip).all():
        raise ValueError("rays must be finite and on the motion device")
    clean = torch.where(valid[..., None], residual, 0)
    if not torch.isfinite(clean).all():
        raise ValueError("valid motion must be finite; clean unobserved channels before projection")
    mean = clean.sum(1, keepdim=True) / valid.sum(1)[:, None, None].clamp_min(1)
    centered = torch.where(valid[..., None], clean - mean, 0)
    extra = (-residual.shape[1]) % stride
    padded = F.pad(centered, (0, 0, 0, extra))
    padded_valid = F.pad(valid, (0, extra), value=False)
    weight = padded_valid.reshape(residual.shape[0], -1, stride).sum(-1).to(residual.dtype)
    bins = padded.reshape(residual.shape[0], -1, stride, residual.shape[-1]).sum(2)
    bins = bins / weight[..., None].clamp_min(1)
    scalar = (bins * rays_for_clip[:, None]).sum(-1, keepdim=True)
    return scalar, weight, bins


def direction_from_probs(
    prob: torch.Tensor,
    state: dict[str, Any],
    *,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Map audio emotion probabilities (or diagnostic one-hot labels) to rays.

    The weighted ray is unit-normalized; exact neutral or cancelling mixtures
    give zero.  Neutral probability never selects an emotional ray.  The
    caller should record ``neutral_mass`` and ``mixture_norm``: normalization
    intentionally represents *direction*, not calibrated emotional confidence.

    Missing nonneutral rays are masked explicitly.  By default any appreciable
    missing probability raises; ``return_info=True`` opts into masking and
    returns ``missing_mass`` for reporting instead.  No GT fallback is used.
    """
    rays = state["rays"].to(device=prob.device, dtype=prob.dtype)
    active = state["active"].to(device=prob.device)
    neutral_id = int(state["neutral_id"])
    if prob.ndim != 2 or prob.shape[1] != rays.shape[0] or not prob.is_floating_point():
        raise ValueError("prob must be floating point [B,num_emotions]")
    if not torch.isfinite(prob).all() or (prob < 0).any():
        raise ValueError("emotion probabilities must be finite and nonnegative")
    if not torch.allclose(prob.sum(-1), torch.ones_like(prob[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("emotion probabilities must sum to one")
    available = active.clone()
    available[neutral_id] = False
    missing = ~available
    missing[neutral_id] = False
    missing_mass = prob[:, missing].sum(-1)
    if not return_info and (missing_mass > 1e-6).any():
        raise ValueError("probability mass has no fitted emotion ray; use return_info=True to report masking")
    mixture = (prob * available.to(prob.dtype)[None]) @ rays
    norm = mixture.norm(dim=-1, keepdim=True)
    eps = max(float(torch.finfo(prob.dtype).eps), 1e-8)
    direction = torch.where(norm > eps, mixture / norm.clamp_min(eps), 0)
    if not return_info:
        return direction
    return direction, {
        "missing_mass": missing_mass,
        "neutral_mass": prob[:, neutral_id],
        "active_mass": prob[:, available].sum(-1),
        "mixture_norm": norm.squeeze(-1),
    }
