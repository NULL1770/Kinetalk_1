"""Independent real-motion statistics probe; no generator components imported."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def motion_features(motion, mask):
    """Full-face amplitude and dynamics. Invalid frames never enter statistics."""
    if motion.ndim != 2 or mask.shape != motion.shape[:1]:
        raise ValueError("Expected [T,C] motion and [T] mask")
    valid = motion[mask]
    if len(valid) < 2 or not torch.isfinite(valid).all():
        raise ValueError("Probe requires at least two finite valid motion frames")
    adjacent = mask[1:] & mask[:-1]
    velocity = (motion[1:] - motion[:-1])[adjacent]
    if not len(velocity):
        raise ValueError("Probe requires adjacent valid frames")
    return torch.cat((valid.mean(0), valid.std(0, correction=0),
                      torch.quantile(valid, .1, dim=0), torch.quantile(valid, .9, dim=0),
                      velocity.abs().mean(0), velocity.std(0, correction=0)))


def center_motion(row, path, window, dim):
    with np.load(path, allow_pickle=False) as z:
        # Actual measured BS, warped onto the existing canonical evaluation
        # clock. This is NOT a Stage1 prediction or a Stage2 latent/residual.
        motion = torch.from_numpy(np.asarray(z["canonical_motion"], dtype=np.float32).copy())
        mask = torch.from_numpy(np.asarray(z["teacher_mask"], dtype=bool).copy())
    if motion.ndim != 2 or motion.shape[1] != dim or mask.shape != motion.shape[:1]:
        raise ValueError(f"Invalid real motion: {row['clip_id']}")
    start = max((len(motion) - window) // 2, 0)
    return motion[start:start + window], mask[start:start + window]


def classification_metrics(target, prediction, names):
    target, prediction = target.cpu().long(), prediction.cpu().long()
    c = len(names)
    if not target.numel() or target.shape != prediction.shape:
        raise ValueError("Empty/mismatched classification samples")
    if min(int(target.min()), int(prediction.min())) < 0 or max(int(target.max()), int(prediction.max())) >= c:
        raise ValueError("Out of range class id")
    confusion = torch.bincount(target * c + prediction, minlength=c*c).reshape(c, c)
    support, predicted = confusion.sum(1), confusion.sum(0)
    present = support > 0
    recall = confusion.diag().double() / support.clamp_min(1)
    precision = confusion.diag().double() / predicted.clamp_min(1)
    f1 = 2 * confusion.diag().double() / (support + predicted).clamp_min(1)
    return {"n": int(target.numel()), "accuracy": float(confusion.diag().sum() / target.numel()),
            "balanced_accuracy": float(recall[present].mean()),
            "macro_f1": float(f1.mean()), "macro_f1_policy": "all declared classes; absent/undefined F1=0",
            "balanced_accuracy_classes": int(present.sum()),
            "majority_baseline": float(support.max() / target.numel()), "uniform_guess": 1/c,
            "class_order": names, "confusion_orientation": "rows=true, columns=predicted", "confusion": confusion.tolist(),
            "per_class": {name: {"n": int(support[i]), "predicted_n": int(predicted[i]),
                                  "recall": float(recall[i]) if present[i] else None,
                                  "precision": float(precision[i]), "f1": float(f1[i])}
                          for i, name in enumerate(names)}}


class MotionEmotionProbe(nn.Module):
    def __init__(self, dimension, hidden, classes):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, classes))

    def fit_normalization(self, train_features):
        self.mean.copy_(train_features.mean(0))
        self.scale.copy_(train_features.std(0, correction=0).clamp_min(1e-4))

    def forward(self, features):
        return self.network((features - self.mean) / self.scale)
