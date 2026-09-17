"""Single-objective timing diagnostics; no renderer or extra motion channels.

All fitted thresholds, floors and output gains use training clips only.
Per-clip target normalization is permitted as training supervision and for
shape-only evaluation; it must never be used to rescale an inference output.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def clip_rms(values, weight):
    clean = torch.where(weight[..., None] > 0, values, 0)
    return ((clean.square() * weight[..., None]).sum(1, keepdim=True)
            / weight.sum(1).clamp_min(1)[:, None, None]).sqrt()


def normalize_target_shape(values, weight, train_ids, floor_quantile=.1):
    """Remove clip amplitude, with a training-only floor for quiet clips."""
    if not 0 <= floor_quantile <= 1 or len(train_ids) == 0:
        raise ValueError("a valid quantile and nonempty training indices are required")
    rms = clip_rms(values, weight)
    floor = torch.quantile(rms[train_ids].flatten(0, 1), floor_quantile,
                           dim=0).clamp_min(1e-4)
    normalized = values / torch.maximum(rms, floor[None, None])
    return torch.where(weight[..., None] > 0, normalized, 0), floor


def field_delta(values, weight):
    """Consecutive-bin differences, excluding missing or padded bin pairs."""
    pair_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    delta = values[:, 1:] - values[:, :-1]
    return torch.where(pair_weight[..., None] > 0, delta, 0), pair_weight


def fit_event_threshold(values, weight, train_ids, quantile=.65):
    if not 0 < quantile < 1 or len(train_ids) == 0:
        raise ValueError("a valid quantile and nonempty training indices are required")
    delta, pair_weight = field_delta(values[train_ids], weight[train_ids])
    observed = delta[pair_weight > 0].abs()
    if len(observed) == 0:
        raise ValueError("event supervision needs observed adjacent bins")
    return torch.quantile(observed, quantile, dim=0).clamp_min(1e-4)


def event_labels(values, weight, threshold):
    """0=offset, 1=steady, 2=onset, in a single scalar energy field."""
    delta, pair_weight = field_delta(values, weight)
    labels = torch.ones_like(delta, dtype=torch.long)
    labels = torch.where(delta < -threshold, 0, labels)
    labels = torch.where(delta > threshold, 2, labels)
    return labels, pair_weight


def event_logits(values, weight, threshold):
    delta, pair_weight = field_delta(values, weight)
    score = delta / threshold
    # The neutral class occupies a dead zone; this is one deterministic
    # readout from the original field, not three learned output channels.
    return torch.stack([-score - 1, torch.zeros_like(score), score - 1], -1), pair_weight


def event_loss(prediction, target, weight, threshold):
    logits, pair_weight = event_logits(prediction, weight, threshold)
    labels, _ = event_labels(target, weight, threshold)
    losses = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), reduction="none")
    losses = losses.reshape(labels.shape)
    return (losses * pair_weight[..., None]).sum() / (
        pair_weight.sum().clamp_min(1) * prediction.shape[-1])


def fit_positive_gain(prediction, target, weight, train_ids):
    """One nonnegative scalar per axis, fit on training outputs/labels only."""
    pred, truth, w = prediction[train_ids], target[train_ids], weight[train_ids]
    numerator = (pred * truth * w[..., None]).sum((0, 1))
    denominator = (pred.square() * w[..., None]).sum((0, 1)).clamp_min(1e-8)
    return (numerator / denominator).clamp_min(0)


def event_metrics(prediction, target, weight, threshold):
    predicted, pair_weight = event_labels(prediction, weight, threshold)
    labels, _ = event_labels(target, weight, threshold)
    w = pair_weight[..., None].expand_as(labels)
    confusion = prediction.new_zeros(3, 3)
    for truth in range(3):
        for pred in range(3):
            confusion[truth, pred] = w[(labels == truth) & (predicted == pred)].sum()
    precision = confusion.diagonal() / confusion.sum(0).clamp_min(1)
    recall = confusion.diagonal() / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {"cross_entropy": float(event_loss(prediction, target, weight, threshold)),
            "macro_f1": float(f1.mean()), "onset_f1": float(f1[2]),
            "offset_f1": float(f1[0]), "steady_f1": float(f1[1]),
            "balanced_accuracy": float(recall.mean()),
            "confusion_weighted": confusion.cpu().tolist()}
