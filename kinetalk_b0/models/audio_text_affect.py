"""Audio-query/text-token fusion with explicit frame expression intensity.

Inference receives no query emotion labels or target motion. The historical
audio-global latent remains unchanged; new local conditioning retains its DC.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .label_guided_affect import TemporalBlock, masked_average


class AudioTextAffect(nn.Module):
    def __init__(self, feature_mean, feature_std, *, text_dim, global_dim=64,
                 hidden=128, local_dim=64, heads=4, initial_intensity=1.):
        super().__init__()
        if (feature_mean.ndim != 1 or feature_mean.shape != feature_std.shape
                or not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all()
                or (feature_std <= 0).any() or initial_intensity <= 0 or hidden % heads):
            raise ValueError('Finite feature statistics and valid model dimensions required')
        self.register_buffer('feature_mean', feature_mean.detach().float().clone())
        self.register_buffer('feature_std', feature_std.detach().float().clone())
        self.audio_input = nn.Linear(len(feature_mean), hidden)
        self.global_input = nn.Linear(global_dim, hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, d) for d in (1, 2, 4, 8))
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_input = nn.Linear(text_dim, hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.intensity_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.intensity_head[-1].weight)
        nn.init.constant_(self.intensity_head[-1].bias, float(torch.log(torch.expm1(torch.tensor(initial_intensity)))))
        self.intensity_film = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, hidden * 2))
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, local_dim))
        nn.init.zeros_(self.fusion[-1].weight); nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, features, valid, global_code, tokens, token_valid, *,
                use_text=True, intensity_override=None, static_intensity=False):
        if (features.ndim != 3 or features.shape[-1] != len(self.feature_mean)
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or not valid.any(1).all()):
            raise ValueError('Audio features and nonempty Boolean mask must match')
        if not torch.isfinite(features[valid]).all():
            raise ValueError('Observed audio must be finite')
        if global_code.shape != (len(features), self.global_input.in_features) or not torch.isfinite(global_code).all():
            raise ValueError('Global feature shape/values differ')
        if (tokens.ndim != 3 or tokens.shape[0] != len(features) or tokens.shape[1] < 1
                or tokens.shape[2] != self.text_input.in_features
                or token_valid.shape != tokens.shape[:2] or token_valid.dtype != torch.bool):
            raise ValueError('Text token shapes/mask differ')
        if use_text and not torch.isfinite(tokens[token_valid]).all():
            raise ValueError('Observed text tokens must be finite')
        clean = torch.where(valid[..., None], (features - self.feature_mean) / self.feature_std, 0.)
        g = self.global_input(global_code)
        h = torch.where(valid[..., None], F.silu(self.audio_input(clean)) + g[:, None], 0.)
        for block in self.blocks:
            h = block(h, valid)
        if use_text:
            clean_text = torch.where(token_valid[..., None], tokens, 0.)
            encoded = self.text_input(self.text_norm(clean_text))
            present = token_valid.any(1)
            safe_mask = token_valid.clone()
            safe_mask[~present, 0] = True
            encoded = torch.where((token_valid & present[:, None])[..., None], encoded, 0.)
            attended = self.cross_attention(self.query_norm(h), encoded, encoded,
                                             key_padding_mask=~safe_mask, need_weights=False)[0]
            attended = torch.where((valid & present[:, None])[..., None], attended, 0.)
            h = h + attended
        intensity = torch.where(valid[..., None], F.softplus(self.intensity_head(h)), 0.)
        drive = intensity
        if intensity_override is not None:
            if (intensity_override.shape != intensity.shape
                    or not torch.isfinite(intensity_override[valid]).all()
                    or (intensity_override[valid] < 0).any()):
                raise ValueError('Oracle/control intensity must be finite nonnegative [B,T,1]')
            drive = torch.where(valid[..., None], intensity_override, 0.)
        if static_intensity:
            drive = masked_average(drive, valid).expand_as(drive)
        scale, shift = self.intensity_film(drive).chunk(2, -1)
        modulated = g[:, None] * (1 + torch.tanh(scale)) + shift
        local = self.fusion(torch.cat((h, modulated), -1))
        return {'local': torch.where(valid[..., None], local, 0.), 'global': global_code,
                'predicted_intensity': intensity,
                'driving_intensity': torch.where(valid[..., None], drive, 0.)}
