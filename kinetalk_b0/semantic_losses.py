"""Named-semantic supervision and one style discrimination objective.

No BS magnitude is used as an emotion/intensity label. Cross-person swaps have
no framewise motion target: only semantics, donor style and mouth timing are
constrained. All semantic targets and frozen donor embeddings are detached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=values.device, dtype=values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(values)
    # Avoid NaN * 0 for explicitly invalid pseudo-labels/padded predictions.
    selected = torch.where(weight > 0, values, torch.zeros_like(values))
    return (selected * weight).sum() / weight.sum().clamp_min(1e-8)


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long)
    valid = valid.to(device=logits.device, dtype=torch.bool) & (labels >= 0)
    if labels.shape != logits.shape[:1] or valid.shape != labels.shape:
        raise ValueError("Classification labels and validity must each have shape [batch]")
    if not valid.any():
        return logits.sum() * 0.0
    if (labels[valid] >= logits.shape[-1]).any():
        raise ValueError("A valid label is outside the classifier's configured classes")
    return F.cross_entropy(logits[valid], labels[valid])


def _time_pairs(
    times: torch.Tensor, valid: torch.Tensor, max_gap_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    if times.shape != valid.shape:
        raise ValueError("timestamps and frame validity must have matching [batch,time] axes")
    dt = times[:, 1:] - times[:, :-1]
    pair_valid = valid[:, 1:] & valid[:, :-1]
    pair_valid = pair_valid & torch.isfinite(dt) & (dt > 0) & (dt <= max_gap_s)
    # Invalid clock entries never enter a denominator, including NaN padding.
    safe_dt = torch.where(pair_valid, dt, torch.ones_like(dt))
    return safe_dt, pair_valid


def semantic_supervision(
    outputs: Mapping[str, torch.Tensor],
    branch: Mapping[str, torch.Tensor],
    va_weight: float = 1.0,
    delta_weight: float = 0.1,
    intensity_weight: float = 0.25,
    *,
    max_gap_s: float = 0.12,
    derivative_scale_s: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Category/known-level CE and confidence-weighted VA/dynamic supervision.

    Dynamic targets use true elapsed seconds. Multiplication by a fixed 0.1 s
    expresses the derivative as a VA change over 100 ms, independently of FPS;
    it is not framewise differencing or per-clip amplitude normalization.
    Invalid VA and gaps longer than max_gap_s never supervise a derivative.
    """
    prediction = outputs["va"]
    target = branch["va"].detach().to(device=prediction.device, dtype=prediction.dtype)
    valid = branch["valid"].to(device=prediction.device, dtype=torch.bool)
    va_valid = branch["va_valid"].to(device=prediction.device, dtype=torch.bool) & valid
    confidence = branch["va_confidence"].detach().to(device=prediction.device, dtype=prediction.dtype)
    if confidence.shape == valid.shape + (1,):
        confidence = confidence.squeeze(-1)
    if prediction.shape != target.shape or prediction.shape != valid.shape + (2,):
        raise ValueError("VA prediction/target must be [batch,time,2] and validity [batch,time]")
    if confidence.shape != valid.shape:
        raise ValueError("VA confidence must have shape [batch,time]")
    if not torch.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("VA confidence must be finite and lie in [0,1]")
    va_valid = va_valid & (confidence > 0)
    if not torch.isfinite(target[va_valid]).all():
        raise ValueError("Valid visual VA targets must be finite")
    clean_target = torch.where(va_valid.unsqueeze(-1), target, torch.zeros_like(target))
    clean_prediction = torch.where(va_valid.unsqueeze(-1), prediction, torch.zeros_like(prediction))
    va = _weighted_mean(
        F.smooth_l1_loss(clean_prediction, clean_target, reduction="none"),
        confidence * va_valid.to(confidence.dtype),
    )
    times = branch["times"].detach().to(device=prediction.device, dtype=torch.float64)
    dt, pair_valid = _time_pairs(times, va_valid, max_gap_s)
    if derivative_scale_s <= 0:
        raise ValueError("derivative_scale_s must be positive")
    dt = dt.to(dtype=prediction.dtype).unsqueeze(-1)
    predicted_delta = (clean_prediction[:, 1:] - clean_prediction[:, :-1]) / dt * derivative_scale_s
    target_delta = (clean_target[:, 1:] - clean_target[:, :-1]) / dt * derivative_scale_s
    pair_confidence = torch.minimum(confidence[:, 1:], confidence[:, :-1]) * pair_valid.to(confidence.dtype)
    va_delta = _weighted_mean(
        F.smooth_l1_loss(predicted_delta, target_delta, reduction="none"), pair_confidence,
    )
    example_valid = valid.any(dim=1)
    emotion = _masked_ce(outputs["emotion_logits"], branch["emotion_id"], example_valid)
    intensity = _masked_ce(
        outputs["intensity_logits"], branch["intensity_id"],
        branch["intensity_valid"].to(example_valid.device, dtype=torch.bool) & example_valid,
    )
    total = emotion + intensity_weight * intensity + va_weight * va + delta_weight * va_delta
    return {"total": total, "emotion": emotion, "intensity": intensity, "va": va, "va_delta": va_delta}


def style_contrastive(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    speaker_ids: torch.Tensor,
    temperature: float = 0.1,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """One supervised contrastive loss across all available reference views.

    positive is [B,D] or [B,K,D]. Every same-speaker view is positive, never a
    negative. valid is [B] (all views) or [B,K+1] (anchor followed by refs).
    A batch with no actual other-speaker negatives provides no discrimination
    and returns a differentiable zero, rather than a false training signal.
    """
    if positive.ndim == 2:
        positive = positive.unsqueeze(1)
    if anchor.ndim != 2 or positive.ndim != 3 or positive.shape[0] != anchor.shape[0] or positive.shape[-1] != anchor.shape[-1]:
        raise ValueError("Style anchors and positives must be [B,D] and [B,K,D]")
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    batch, references = positive.shape[:2]
    views = references + 1
    if speaker_ids.shape != (batch,):
        raise ValueError("speaker_ids must have shape [batch]")
    speaker_ids = speaker_ids.to(device=anchor.device, dtype=torch.long)
    features = torch.cat([anchor.unsqueeze(1), positive], dim=1).reshape(-1, anchor.shape[-1])
    ids = speaker_ids[:, None].expand(-1, views).reshape(-1)
    if valid is None:
        validity = torch.ones((batch, views), device=anchor.device, dtype=torch.bool)
    elif valid.shape == (batch,):
        validity = valid.to(device=anchor.device, dtype=torch.bool)[:, None].expand(-1, views)
    elif valid.shape == (batch, views):
        validity = valid.to(device=anchor.device, dtype=torch.bool)
    else:
        raise ValueError("Style validity must be [B] or [B,K+1]")
    keep = validity.reshape(-1) & (ids >= 0)
    features = features[keep]
    ids = ids[keep]
    if len(features) < 3 or ids.unique().numel() < 2:
        return torch.nan_to_num(features).sum() * 0.0
    if not torch.isfinite(features).all():
        raise ValueError("Valid style embeddings must be finite")
    features = F.normalize(features, dim=-1)
    allowed = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = ids[:, None].eq(ids[None, :]) & allowed
    negatives = ids[:, None].ne(ids[None, :]) & allowed
    eligible = positives.any(dim=1) & negatives.any(dim=1)
    if not eligible.any():
        return features.sum() * 0.0
    logits = features @ features.T / temperature
    log_probability = F.log_softmax(logits[eligible].masked_fill(~allowed[eligible], -torch.inf), dim=-1)
    positive_log_probability = log_probability.masked_fill(~positives[eligible], 0.0)
    return -(positive_log_probability.sum(dim=-1) / positives[eligible].sum(dim=-1)).mean()


def _donor_style_loss(
    generated: torch.Tensor,
    donor: torch.Tensor,
    negatives: torch.Tensor | None,
    temperature: float,
    donor_speaker_ids: torch.Tensor | None,
    negative_speaker_ids: torch.Tensor | None,
) -> torch.Tensor:
    if generated.ndim != 2 or donor.shape != generated.shape:
        raise ValueError("Generated and donor style must have matching [B,D] shapes")
    if temperature <= 0:
        raise ValueError("Style temperature must be positive")
    if negatives is None or negatives.numel() == 0:
        return generated.sum() * 0.0
    if negatives.ndim == 2:
        negatives = negatives.unsqueeze(0).expand(generated.shape[0], -1, -1)
    if negatives.ndim != 3 or negatives.shape[0] != generated.shape[0] or negatives.shape[-1] != generated.shape[-1]:
        raise ValueError("Style negatives must be [N,D] or [B,N,D]")
    allowed = torch.ones(negatives.shape[:2], device=generated.device, dtype=torch.bool)
    if (donor_speaker_ids is None) != (negative_speaker_ids is None):
        raise ValueError("Provide both donor and negative speaker ids to exclude false negatives")
    if donor_speaker_ids is not None:
        donor_ids = donor_speaker_ids.to(device=generated.device)
        negative_ids = negative_speaker_ids.to(device=generated.device)
        if donor_ids.shape != generated.shape[:1]:
            raise ValueError("Donor speaker ids must have shape [B]")
        if negative_ids.ndim == 1:
            negative_ids = negative_ids.unsqueeze(0).expand(generated.shape[0], -1)
        if negative_ids.shape != negatives.shape[:2]:
            raise ValueError("Negative speaker ids must have shape [N] or [B,N]")
        allowed = donor_ids[:, None].ne(negative_ids) & (negative_ids >= 0) & (donor_ids[:, None] >= 0)
    eligible = allowed.any(dim=1)
    if not eligible.any():
        return generated.sum() * 0.0
    generated = F.normalize(generated, dim=-1)
    donor = F.normalize(donor.detach(), dim=-1)
    negatives = F.normalize(negatives.detach(), dim=-1)
    positive_logit = (generated * donor).sum(dim=-1, keepdim=True) / temperature
    negative_logits = torch.einsum("bd,bnd->bn", generated, negatives) / temperature
    negative_logits = negative_logits.masked_fill(~allowed, -torch.inf)
    logits = torch.cat([positive_logit, negative_logits], dim=-1)[eligible]
    return F.cross_entropy(logits, torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long))


def cross_style_objective(
    generated_motion: torch.Tensor,
    base_motion: torch.Tensor,
    semantic_outputs: Mapping[str, torch.Tensor],
    target_branch: Mapping[str, torch.Tensor],
    generated_style: torch.Tensor,
    donor_style: torch.Tensor,
    style_negatives: torch.Tensor | None,
    mouth_indices: Sequence[int],
    *,
    donor_speaker_ids: torch.Tensor | None = None,
    negative_speaker_ids: torch.Tensor | None = None,
    style_weight: float = 1.0,
    semantic_weight: float = 1.0,
    mouth_weight: float = 0.1,
    temperature: float = 0.1,
    va_weight: float = 1.0,
    delta_weight: float = 0.1,
    intensity_weight: float = 0.25,
    max_gap_s: float = 0.12,
) -> dict[str, torch.Tensor]:
    """Semantic preservation, donor discrimination and articulation direction.

    base_motion is the detached content baseline (usually predicted B0).
    Mouth supervision compares *direction* of the selected articulatory
    velocity vector on moving frames; gain can change, and upper-face motion
    is wholly absent from this timing term. Supply articulatory channels, not
    smile/eye/eyebrow channels, as mouth_indices.
    """
    if generated_motion.shape != base_motion.shape or generated_motion.ndim != 3:
        raise ValueError("Generated/base motion must have matching [B,T,D] shapes")
    semantics = semantic_supervision(
        semantic_outputs, target_branch, va_weight, delta_weight, intensity_weight, max_gap_s=max_gap_s,
    )
    style = _donor_style_loss(
        generated_style, donor_style, style_negatives, temperature,
        donor_speaker_ids, negative_speaker_ids,
    )
    indices = list(dict.fromkeys(int(index) for index in mouth_indices))
    if indices and (min(indices) < 0 or max(indices) >= generated_motion.shape[-1]):
        raise ValueError("Articulatory index is outside the motion dimension")
    if not indices or generated_motion.shape[1] < 2:
        mouth = generated_motion.sum() * 0.0
    else:
        valid = target_branch["valid"].to(device=generated_motion.device, dtype=torch.bool)
        times = target_branch["times"].to(device=generated_motion.device, dtype=torch.float64)
        dt, pair_valid = _time_pairs(times, valid, max_gap_s)
        dt = dt.to(dtype=generated_motion.dtype).unsqueeze(-1)
        generated = generated_motion[..., indices]
        base = base_motion.detach()[..., indices]
        generated_velocity = (generated[:, 1:] - generated[:, :-1]) / dt
        base_velocity = (base[:, 1:] - base[:, :-1]) / dt
        pair_valid = pair_valid & (base_velocity.norm(dim=-1) > 1e-3)
        generated_velocity = torch.where(pair_valid.unsqueeze(-1), generated_velocity, torch.zeros_like(generated_velocity))
        base_velocity = torch.where(pair_valid.unsqueeze(-1), base_velocity, torch.zeros_like(base_velocity))
        direction_error = 1.0 - F.cosine_similarity(generated_velocity, base_velocity, dim=-1, eps=1e-3)
        mouth = _weighted_mean(direction_error, pair_valid)
    total = style_weight * style + semantic_weight * semantics["total"] + mouth_weight * mouth
    return {
        "total": total, "style": style, "semantic": semantics["total"], "mouth_timing": mouth,
        **{key: value for key, value in semantics.items() if key != "total"},
    }
