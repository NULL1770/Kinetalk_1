"""Run-local, audio-only event timing and categorical duration predictor.

No motion, target-derived boundary or teacher history is accepted at inference.
The offline TCN has no temporal positions and replicates run-edge taps, so
constant acoustic input produces a time-constant distribution even at gaps.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from kinetalk_b0.models.motion_process_prior import _RunConvBlock

DURATIONS = (5, 10, 15, 25, 40, 60, 100)


class AudioEventSchedulePredictor(nn.Module):
    """Small symmetric TCN returning onset and duration logits per audio frame."""

    def __init__(self, feature_dim: int = 1540, context_dim: int = 0, *, hidden: int = 96,
                 groups: int = 4, durations: Sequence[int] = DURATIONS,
                 dilations: Sequence[int] = (1, 2, 4, 8)):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (feature_dim, hidden, groups)):
            raise ValueError("feature_dim, hidden and groups must be positive integers")
        if type(context_dim) is not int or context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        if (not durations or any(type(v) is not int or v < 1 for v in durations)
                or tuple(sorted(set(durations))) != tuple(durations)):
            raise ValueError("durations must be increasing positive integers")
        if not dilations or any(type(v) is not int or v < 1 for v in dilations):
            raise ValueError("positive TCN dilations required")
        self.feature_dim, self.context_dim, self.hidden = feature_dim, context_dim, hidden
        self.groups, self.duration_values = groups, tuple(durations)
        self.register_buffer("durations", torch.tensor(durations, dtype=torch.long))
        self.input = nn.Linear(feature_dim, hidden)
        self.context = nn.Linear(context_dim, hidden) if context_dim else None
        self.blocks = nn.ModuleList(_RunConvBlock(hidden, d) for d in dilations)
        self.norm = nn.LayerNorm(hidden)
        self.onset_head = nn.Linear(hidden, groups)
        self.duration_head = nn.Linear(hidden, groups * len(durations))

    def forward(self, features: torch.Tensor, context: Optional[torch.Tensor] = None,
                valid: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        if (not torch.is_tensor(features) or features.ndim != 3 or not features.is_floating_point()
                or features.shape[-1] != self.feature_dim or min(features.shape[:2]) < 1):
            raise ValueError("features must be floating [B,T,feature_dim]")
        b, t, _ = features.shape
        if valid is None: valid = torch.ones(b, t, dtype=torch.bool, device=features.device)
        if (not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != (b, t)
                or valid.device != features.device or not valid.any(1).all()
                or not torch.isfinite(features[valid]).all()):
            raise ValueError("finite observed audio and nonempty Boolean valid[B,T] required")
        if self.context_dim:
            if (not torch.is_tensor(context) or context.shape != (b, self.context_dim)
                    or context.device != features.device or context.dtype != features.dtype
                    or not torch.isfinite(context).all()):
                raise ValueError("finite matching context[B,C] required")
        elif context is not None and (not torch.is_tensor(context) or context.shape != (b, 0)):
            raise ValueError("context must be absent or [B,0] for context_dim=0")
        mask = valid[..., None]
        clean = torch.where(mask, features, torch.zeros_like(features))
        hidden = F.silu(self.input(clean))
        if self.context is not None: hidden = hidden + self.context(context)[:, None]
        hidden = torch.where(mask, hidden, 0.)
        run_ids = (~valid).long().cumsum(1)
        for block in self.blocks: hidden = block(hidden, valid, run_ids)
        hidden = torch.where(mask, self.norm(hidden), 0.)
        onset = self.onset_head(hidden)
        duration = self.duration_head(hidden).reshape(b, t, self.groups, len(self.durations))
        return {"onset_logits": torch.where(mask, onset, 0.),
                "duration_logits": torch.where(mask[..., None], duration, 0.)}

    @torch.no_grad()
    def sample_schedule(self, features: torch.Tensor, context: Optional[torch.Tensor] = None,
                        valid: Optional[torch.Tensor] = None, *, generator: torch.Generator,
                        fps: float = 25., refractory_frames: int = 0) -> dict[str, torch.Tensor]:
        """Sample non-overlapping within-group events with an explicit RNG.

        Each eligible idle frame draws onset then duration. Gaps reset the
        group clock; durations are truncated at the observed run boundary.
        The condition is [activeG, phaseG, sampled_duration_secondsG].
        Sampled duration survives boundary truncation and determines phase.
        """
        if not isinstance(generator, torch.Generator): raise ValueError("explicit torch.Generator required")
        if fps <= 0 or type(refractory_frames) is not int or refractory_frames < 0:
            raise ValueError("positive fps and nonnegative refractory_frames required")
        out = self(features, context, valid)
        if valid is None: valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        b, t = valid.shape; g = self.groups
        active = features.new_zeros(b, t, g); phase = active.clone(); seconds = active.clone()
        onset = torch.zeros(b, t, g, dtype=torch.bool, device=features.device)
        sampled_duration = torch.zeros(b, t, g, dtype=torch.long, device=features.device)
        probability = out["onset_logits"].sigmoid()
        duration_probability = out["duration_logits"].softmax(-1)
        for row in range(b):
            for group in range(g):
                frame = 0
                while frame < t:
                    if not bool(valid[row, frame]): frame += 1; continue
                    if torch.rand((), device=features.device, generator=generator) >= probability[row, frame, group]:
                        frame += 1; continue
                    idx = int(torch.multinomial(duration_probability[row, frame, group], 1, generator=generator))
                    duration = int(self.durations[idx]); stop = min(t, frame + duration)
                    invalid = (~valid[row, frame:stop]).nonzero()
                    if invalid.numel(): stop = frame + int(invalid[0])
                    count = stop - frame
                    onset[row, frame, group] = True
                    sampled_duration[row, frame, group] = duration
                    active[row, frame:stop, group] = 1.
                    phase[row, frame:stop, group] = torch.arange(count, device=features.device, dtype=features.dtype) / max(duration - 1, 1)
                    seconds[row, frame:stop, group] = duration / fps
                    # Refractory state must also reset at the next gap.
                    frame = stop
                    for _ in range(refractory_frames):
                        if frame >= t or not bool(valid[row, frame]): break
                        frame += 1
        return {"condition": torch.cat((active, phase, seconds), -1), "onset": onset,
                "sampled_duration": sampled_duration, "active": active, "phase": phase,
                "duration_seconds": seconds, **out}


__all__ = ["AudioEventSchedulePredictor", "DURATIONS"]
