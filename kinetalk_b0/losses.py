from __future__ import annotations

import torch
from torch.nn import functional as F

from .utils import cosine_distance, masked_l1, masked_mse


def stage1_prototype_loss(
    prototype_logits: torch.Tensor,
    prototype_patches: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    prototype_bank,
    articulatory_indices: list[int],
) -> dict[str, torch.Tensor]:
    """VQ-style assignment and continuous residual loss on short neutral patches."""
    target_art = target[..., articulatory_indices]
    labels = prototype_bank.labels(target_art, mask)
    ce = F.cross_entropy(prototype_logits.reshape(-1, prototype_logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    target_patches = prototype_bank._patches(target_art)
    patch_mask = mask.to(prototype_patches.dtype).unsqueeze(-1).expand_as(prototype_patches[..., 0])
    raw = F.smooth_l1_loss(prototype_patches, target_patches, reduction="none", beta=0.03)
    patch_reconstruction = (raw * patch_mask.unsqueeze(-1)).sum() / patch_mask.unsqueeze(-1).expand_as(raw).sum().clamp_min(1.0)
    return {"prototype": ce, "prototype_reconstruction": patch_reconstruction, "total": ce + patch_reconstruction}


def stage1_envelope_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    articulatory_indices: list[int],
) -> torch.Tensor:
    """Match soft peak time, width, and local energy of the mouth envelope."""
    pred_env = prediction[..., articulatory_indices].abs().mean(-1)
    target_env = target[..., articulatory_indices].abs().mean(-1).detach()
    valid = mask.to(prediction.dtype)
    positions = torch.arange(prediction.shape[1], device=prediction.device, dtype=prediction.dtype).view(1, -1)

    def stats(env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.softmax(env.masked_fill(~mask, -20.0) / 0.10, dim=1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-5)
        peak = (weights * positions).sum(1)
        width = torch.sqrt((weights * (positions - peak[:, None]).square()).sum(1).clamp_min(1e-6))
        energy = (env.square() * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return peak, width, energy

    pred_peak, pred_width, pred_energy = stats(pred_env)
    tgt_peak, tgt_width, tgt_energy = stats(target_env)
    return (
        F.smooth_l1_loss(pred_peak, tgt_peak)
        + F.smooth_l1_loss(pred_width, tgt_width)
        + F.smooth_l1_loss(pred_energy, tgt_energy)
    )


def stage1_monotonic_alignment_loss(native_position: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize any backward step in the native alignment path."""
    if native_position.shape[1] < 2:
        return native_position.new_zeros(())
    valid = mask[:, 1:] & mask[:, :-1]
    backward = F.relu(-(native_position[:, 1:] - native_position[:, :-1]))
    return (backward * valid.to(backward.dtype)).sum() / valid.sum().clamp_min(1.0)


def stage1_style_contrastive_loss(style: torch.Tensor, positive_mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Pull same-speaker/different-sentence articulation styles together."""
    if style.shape[0] < 2:
        return style.new_zeros(())
    features = F.normalize(style, dim=-1)
    logits = features @ features.transpose(0, 1) / max(float(temperature), 1e-4)
    self_mask = torch.eye(style.shape[0], device=style.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))
    positives = positive_mask.to(device=style.device) & ~self_mask
    counts = positives.sum(1)
    valid = counts > 0
    if not bool(valid.any()):
        return style.new_zeros(())
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(1)
    return (-positive_log_prob / counts.clamp_min(1).to(log_prob.dtype))[valid].mean()


def stage1_style_probe_loss(logits: torch.Tensor, emotion_id: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, emotion_id)


def stage1_clean_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: torch.Tensor,
    *,
    velocity_weight: float = 0.5,
    frame_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    acceleration_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Stage1 objective on one canonical neutral target stream.

    The active contract is deliberately small: masked Huber reconstruction,
    velocity matching, and an optional acceleration term.  The canonical
    branch has one neutral target clock, so RMS, event, envelope, alignment,
    and cross-sample invariance terms do not belong in this objective.
    """
    weight = mask.unsqueeze(-1).to(prediction.dtype) * channel_weights.view(1, 1, -1).to(prediction.dtype)
    if frame_weight is not None:
        weight = weight * frame_weight.to(prediction.dtype).unsqueeze(-1)
    if sample_weight is not None:
        weight = weight * sample_weight.to(prediction.dtype).view(-1, 1, 1)
    huber_raw = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.03)
    reconstruction = (huber_raw * weight).sum() / weight.sum().clamp_min(1.0)
    valid = mask[:, 1:] & mask[:, :-1]
    if prediction.shape[1] > 1 and bool(valid.any()):
        velocity_raw = F.l1_loss(
            prediction[:, 1:] - prediction[:, :-1],
            target[:, 1:] - target[:, :-1], reduction="none",
        )
        velocity_weight_tensor = valid.unsqueeze(-1).to(prediction.dtype) * channel_weights.view(1, 1, -1).to(prediction.dtype)
        if frame_weight is not None:
            velocity_weight_tensor = velocity_weight_tensor * frame_weight[:, 1:].to(prediction.dtype).unsqueeze(-1) * frame_weight[:, :-1].to(prediction.dtype).unsqueeze(-1)
        if sample_weight is not None:
            velocity_weight_tensor = velocity_weight_tensor * sample_weight.to(prediction.dtype).view(-1, 1, 1)
        velocity = (velocity_raw * velocity_weight_tensor).sum() / velocity_weight_tensor.sum().clamp_min(1.0)
    else:
        velocity = prediction.new_zeros(())
    acceleration = prediction.new_zeros(())
    if acceleration_weight > 0.0 and prediction.shape[1] > 2:
        valid_acc = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        acc = prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]
        tgt_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
        aw = valid_acc.unsqueeze(-1).to(prediction.dtype) * channel_weights.view(1, 1, -1)
        if frame_weight is not None:
            aw = aw * frame_weight[:, 2:].to(prediction.dtype).unsqueeze(-1)
            aw = aw * frame_weight[:, 1:-1].to(prediction.dtype).unsqueeze(-1)
            aw = aw * frame_weight[:, :-2].to(prediction.dtype).unsqueeze(-1)
        if sample_weight is not None:
            aw = aw * sample_weight.to(prediction.dtype).view(-1, 1, 1)
        acceleration = (F.smooth_l1_loss(acc, tgt_acc, reduction="none", beta=0.03) * aw).sum() / aw.sum().clamp_min(1.0)
    total = reconstruction + velocity_weight * velocity + acceleration_weight * acceleration
    return {"total": total, "reconstruction": reconstruction, "velocity": velocity, "acceleration": acceleration}


def _classification_losses(
    factors: dict[str, torch.Tensor],
    emotion_id: torch.Tensor,
    intensity_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    emotion = F.cross_entropy(factors["emotion_logits"], emotion_id)
    intensity = F.cross_entropy(factors["intensity_logits"], intensity_id)
    return emotion, intensity


def stage2_loss(
    prediction: torch.Tensor,
    velocity_target: torch.Tensor,
    target_residual: torch.Tensor,
    mask: torch.Tensor,
    factors: dict[str, torch.Tensor],
    emotion_id: torch.Tensor,
    intensity_id: torch.Tensor,
    *,
    velocity_weight: float = 0.05,
    classification_weight: float = 0.5,
    intensity_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    # Flow matching predicts a velocity in scaled residual coordinates.
    flow = masked_mse(prediction, velocity_target, mask)
    velocity = masked_l1(
        prediction[:, 1:] - prediction[:, :-1],
        velocity_target[:, 1:] - velocity_target[:, :-1],
        mask[:, 1:] & mask[:, :-1],
    ) if prediction.shape[1] > 1 else prediction.new_zeros(())
    emotion_ce, intensity_ce = _classification_losses(factors, emotion_id, intensity_id)
    total = flow + velocity_weight * velocity + classification_weight * emotion_ce + intensity_weight * intensity_ce
    return {"total": total, "flow": flow, "velocity": velocity, "emotion_ce": emotion_ce, "intensity_ce": intensity_ce}


def stage2_swap_loss(
    swapped_prediction: torch.Tensor,
    target_motion: torch.Tensor,
    target_b0: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # The target is still defined with a neutral BS ground truth. It is never
    # made from a Stage-1 prediction, even for cross-emotion reconstruction.
    return masked_l1(swapped_prediction + target_b0, target_motion, mask)


def stage2_style_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    valid: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    """Cosine triplet loss with an explicit different-speaker negative.

    SupCon depends on batch composition to supply a same-speaker peer; that
    silently degenerates when a batch happens to be all-distinct speakers.
    An explicit sampled negative removes that dependency and provides a
    guaranteed push signal against the trivial style-collapse solution.
    """
    if not bool(valid.any()):
        return anchor.new_zeros(())
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negative, dim=-1)
    sim_positive = (a * p).sum(dim=-1)
    sim_negative = (a * n).sum(dim=-1)
    loss = F.relu(sim_negative - sim_positive + margin)
    return loss[valid].mean()


def stage2_style_cross_emotion_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Pull same-speaker style codes across different emotions together.

    The sampled positive is same speaker/different sentence (and prefers a
    different emotion).  A direct cosine pull removes the emotion component
    that a triplet margin can leave in both codes while still satisfying the
    margin.  This is deliberately separate from the renderer cycle, whose
    donor style must not be used as an emotion target.
    """
    if not bool(valid.any()):
        return anchor.new_zeros(())
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    distance = 1.0 - (a * p).sum(dim=-1)
    return distance[valid].mean()


def stage2_style_supcon_loss(
    style: torch.Tensor,
    same_speaker: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss on the global style code.

    Cycle-only style supervision has a trivial constant solution (map every
    input to the same latent). Pulling same-speaker style codes together and
    pushing different-speaker apart forces `z_style` to carry speaker-habit
    information a downstream renderer cannot ignore.
    """
    batch = style.shape[0]
    if batch < 2:
        return style.new_zeros(())
    features = F.normalize(style, dim=-1)
    logits = features @ features.transpose(0, 1) / max(temperature, 1e-4)
    self_mask = torch.eye(batch, device=style.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))
    positives = same_speaker & ~self_mask
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return style.new_zeros(())
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # Avoid `-inf * 0` on the self-mask position by selecting positives instead
    # of multiplying by a float mask.
    positive_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    per_sample = -positive_log_prob / positive_count.clamp_min(1).to(log_prob.dtype)
    return per_sample[valid].mean()


def style_factor_invariance_loss(style: torch.Tensor, emotion_logits: torch.Tensor, emotion_id: torch.Tensor) -> torch.Tensor:
    """Penalize linear correlation between style dimensions and emotion labels."""
    labels = F.one_hot(emotion_id, num_classes=emotion_logits.shape[-1]).to(style.dtype)
    s = F.normalize(style - style.mean(0, keepdim=True), dim=0)
    y = labels - labels.mean(0, keepdim=True)
    return (s.transpose(0, 1) @ y / max(style.shape[0], 1)).square().mean()


def style_content_invariance_loss(style: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """Remove linear correlation between global style and frame-mean content."""
    c = content.mean(dim=1) if content.ndim == 3 else content
    s = style - style.mean(0, keepdim=True)
    c = c - c.mean(0, keepdim=True)
    return (F.normalize(s, dim=0).transpose(0, 1) @ F.normalize(c, dim=0) / max(style.shape[0], 1)).square().mean()

def style_probe_loss(factors: dict[str, torch.Tensor], emotion_id: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    # The probe is connected through GradientReversal in Stage2Model.  Using
    # the detached diagnostic logits here trained neither the probe nor the
    # encoder, so the former "adversarial" loss was a no-op.  The GRL logits
    # provide normal gradients to the probe and reversed gradients to style.
    emotion = F.cross_entropy(factors["style_emotion_probe_logits"], emotion_id)
    target = content.mean(dim=1).detach()
    content_loss = F.mse_loss(factors["style_content_probe"], target)
    return emotion + content_loss


def stage2_factor_cycle_loss(
    reencoded: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cycle supervision for swaps that have no valid frame-level target.

    All expected codes are detached. The renderer must make an output that a
    fresh factor pass recognizes, instead of allowing both branches to chase a
    moving latent coordinate together.
    """
    valid = torch.ones(reencoded["global"].shape[0], device=reencoded["global"].device, dtype=torch.bool) if sample_valid is None else sample_valid
    if not bool(valid.any()):
        return reencoded["global"].new_zeros(())
    global_distance = 1.0 - F.cosine_similarity(reencoded["global"], expected["global"].detach(), dim=-1)
    style_distance = 1.0 - F.cosine_similarity(reencoded["style"], expected["style"].detach(), dim=-1)
    global_loss = global_distance[valid].mean()
    style_loss = style_distance[valid].mean()
    return global_loss + style_loss


def stage3_loss(
    audio: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    emotion_id: torch.Tensor,
    intensity_id: torch.Tensor,
    *,
    global_weight: float = 1.0,
    classification_weight: float = 1.0,
    intensity_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    global_align = cosine_distance(audio["global"], teacher["global"])
    emotion_ce = F.cross_entropy(audio["emotion_logits"], emotion_id)
    intensity_ce = F.cross_entropy(audio["intensity_logits"], intensity_id)
    total = global_weight * global_align + classification_weight * emotion_ce + intensity_weight * intensity_ce
    return {
        "total": total, "global": global_align, "emotion_ce": emotion_ce, "intensity_ce": intensity_ce,
    }


def stage4_loss(
    flow_prediction: torch.Tensor,
    flow_target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Stage-4 flow matching objective without auxiliary dynamic paths."""
    flow = masked_mse(flow_prediction, flow_target, mask)
    return {"total": flow, "flow": flow}

