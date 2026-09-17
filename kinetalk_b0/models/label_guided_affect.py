"""Explicit label guidance and frame intensity for one shared motion decoder.

The original global latent is never unit-normalized. Labels/levels are declared
inference inputs; target motion is not an argument to this module.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def probability_input(value, *, batch, classes, device, dtype, name):
    if value.shape != (batch, classes) or not value.is_floating_point():
        raise ValueError(f"{name} must be floating probabilities [B,{classes}]")
    value = value.to(device=device, dtype=dtype)
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"Invalid {name}")
    if not torch.allclose(value.sum(-1), torch.ones(batch, device=device, dtype=dtype), atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} probabilities must sum to one")
    return value


def explicit_guidance(emotion_id, level_id, *, num_emotions=8, num_levels=4):
    """Unknown (-1) level has a separate token; never silently maps to neutral."""
    if emotion_id.ndim != 1 or level_id.shape != emotion_id.shape:
        raise ValueError("Emotion and level IDs must be [B]")
    if emotion_id.dtype != torch.long or level_id.dtype != torch.long:
        raise ValueError("Guidance IDs must be torch.long")
    if ((emotion_id < 0) | (emotion_id >= num_emotions)).any():
        raise ValueError("Emotion ID outside configured label vocabulary")
    if ((level_id < -1) | (level_id >= num_levels)).any():
        raise ValueError("Level ID outside configured vocabulary")
    levels = torch.where(level_id < 0, num_levels, level_id)
    return F.one_hot(emotion_id, num_emotions).float(), F.one_hot(levels, num_levels + 1).float()


def automatic_guidance(emotion_logits, level_logits):
    """Explicitly named automatic fallback, not the label-guided main result."""
    if emotion_logits.ndim != 2 or level_logits.ndim != 2 or len(emotion_logits) != len(level_logits):
        raise ValueError("Automatic guidance logits require matching [B,C]")
    if not torch.isfinite(emotion_logits).all() or not torch.isfinite(level_logits).all():
        raise ValueError("Automatic guidance logits must be finite")
    return emotion_logits.softmax(-1), F.pad(level_logits.softmax(-1), (0, 1))


def masked_average(x, valid):
    clean = torch.where(valid[..., None], x, 0.)
    return clean.sum(1, keepdim=True) / valid.sum(1)[:, None, None].clamp_min(1)


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)

    def forward(self, x, valid):
        clean = torch.where(valid[..., None], F.silu(self.norm(x)), 0.)
        value = self.conv(clean.transpose(1, 2)).transpose(1, 2)
        return torch.where(valid[..., None], x + F.silu(value), 0.)


class LabelGuidedAffect(nn.Module):
    """Frame-rate acoustic conditioning, explicit labels, and semantic intensity.

    ``static`` is a matched trainable comparator: condition features are pooled
    after temporal encoding, so neither local condition nor intensity can gain
    time variation from convolution boundaries. B0/content remain temporal.
    """
    def __init__(self, feature_mean, feature_std, *, global_dim=64, hidden=128,
                 local_dim=64, num_emotions=8, num_levels=4, initial_intensity=1.):
        super().__init__()
        if (feature_mean.ndim != 1 or feature_mean.shape != feature_std.shape
                or not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all()
                or (feature_std <= 0).any() or initial_intensity <= 0):
            raise ValueError("Finite feature statistics and positive initial intensity required")
        self.register_buffer('feature_mean', feature_mean.detach().clone().float())
        self.register_buffer('feature_std', feature_std.detach().clone().float())
        self.num_emotions, self.num_levels = num_emotions, num_levels
        self.input = nn.Linear(len(feature_mean), hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, d) for d in (1, 2, 4, 8))
        self.emotion_embedding = nn.Linear(num_emotions, 32, bias=False)
        self.level_embedding = nn.Linear(num_levels + 1, 16, bias=False)
        self.global_input = nn.Linear(global_dim, 32)
        self.condition = nn.Linear(80, hidden)
        self.intensity_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.intensity_head.weight)
        nn.init.constant_(self.intensity_head.bias, float(torch.log(torch.expm1(torch.tensor(initial_intensity)))))
        self.intensity_film = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 64))
        self.fusion = nn.Sequential(nn.Linear(hidden + 32 + 16 + 32, hidden), nn.SiLU(), nn.Linear(hidden, local_dim))
        self.label_global_delta = nn.Linear(48, global_dim)
        # Preserve the old zero-local start without suppressing the direct
        # intensity-supervision gradient to its own output head.
        nn.init.zeros_(self.fusion[-1].weight); nn.init.zeros_(self.fusion[-1].bias)
        nn.init.zeros_(self.label_global_delta.weight); nn.init.zeros_(self.label_global_delta.bias)

    def forward(self, features, valid, global_code, emotion_prob, level_prob, *,
                temporal_mode='full', intensity_override=None):
        if (features.ndim != 3 or features.shape[-1] != len(self.feature_mean)
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or not valid.any(1).all()):
            raise ValueError("Frame features and nonempty Boolean masks must match")
        if not torch.isfinite(features[valid]).all():
            raise ValueError("Observed frame features must be finite")
        if global_code.shape != (len(features), self.global_input.in_features) or not torch.isfinite(global_code).all():
            raise ValueError("Global code shape/values differ")
        if temporal_mode not in ('full', 'static'):
            raise ValueError("Unknown temporal mode")
        b, t = features.shape[:2]
        ep = probability_input(emotion_prob, batch=b, classes=self.num_emotions,
                               device=features.device, dtype=features.dtype, name='emotion')
        lp = probability_input(level_prob, batch=b, classes=self.num_levels + 1,
                               device=features.device, dtype=features.dtype, name='level')
        e, level, g = self.emotion_embedding(ep), self.level_embedding(lp), self.global_input(global_code)
        clean = torch.where(valid[..., None], (features - self.feature_mean) / self.feature_std, 0.)
        h = F.silu(self.input(clean)) + self.condition(torch.cat((e, level, g), -1))[:, None]
        h = torch.where(valid[..., None], h, 0.)
        for block in self.blocks:
            h = block(h, valid)
        if temporal_mode == 'static':
            h = masked_average(h, valid).expand(-1, t, -1)
        intensity = torch.where(valid[..., None], F.softplus(self.intensity_head(h)), 0.)
        drive = intensity
        if intensity_override is not None:
            if (intensity_override.shape != intensity.shape
                    or not torch.isfinite(intensity_override[valid]).all()
                    or (intensity_override[valid] < 0).any()):
                raise ValueError("Intensity override must be nonnegative finite [B,T,1]")
            drive = torch.where(valid[..., None], intensity_override, 0.)
        if temporal_mode == 'static':
            drive = masked_average(drive, valid).expand(-1, t, -1)
        scale, shift = self.intensity_film(drive).chunk(2, -1)
        dynamic_emotion = e[:, None] * (1 + torch.tanh(scale)) + shift
        local = self.fusion(torch.cat((h, dynamic_emotion,
            level[:, None].expand(-1, t, -1), g[:, None].expand(-1, t, -1)), -1))
        return {'local': torch.where(valid[..., None], local, 0.),
                'global': global_code + self.label_global_delta(torch.cat((e, level), -1)),
                'predicted_intensity': intensity,
                'driving_intensity': torch.where(valid[..., None], drive, 0.)}


def apply_label_condition(original_affect, prediction):
    """Keep original fields, replacing only declared learned global/local."""
    return {**original_affect, 'global': prediction['global'], 'local': prediction['local']}
