"""Static event prior plus a bounded local prosody residual.

The static head captures clip-level emotion/identity prevalence.  The residual
sees only a run-safe, centered 10-D prosody event condition and is zero
initialized, so its first prediction is exactly the static baseline.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from kinetalk_b0.models.motion_process_prior import _RunConvBlock


class StaticScheduleHead(nn.Module):
    def __init__(self, context_dim: int, groups: int = 4, durations: int = 7, hidden: int = 64):
        super().__init__()
        if min(context_dim, groups, durations, hidden) < 1:
            raise ValueError('positive schedule dimensions required')
        self.context_dim, self.groups, self.durations, self.hidden = context_dim, groups, durations, hidden
        self.network = nn.Sequential(nn.Linear(context_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.onset_head = nn.Linear(hidden, groups)
        self.duration_head = nn.Linear(hidden, groups * durations)

    def forward(self, context: torch.Tensor, frames: int):
        if context.ndim != 2 or context.shape[-1] != self.context_dim or frames < 1:
            raise ValueError('context[B,C] and positive frames required')
        h = self.network(context)
        onset = self.onset_head(h)[:, None].expand(-1, frames, -1)
        duration = self.duration_head(h).reshape(len(context), 1, self.groups, self.durations)
        return {'onset_logits': onset, 'duration_logits': duration.expand(-1, frames, -1, -1)}


class ProsodyResidualSchedule(nn.Module):
    def __init__(self, base: StaticScheduleHead, condition_dim: int = 10, hidden: int = 64,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        if not isinstance(base, StaticScheduleHead):
            raise TypeError('base must be StaticScheduleHead')
        self.base = base
        self.condition_dim, self.hidden = condition_dim, hidden
        self.input = nn.Linear(condition_dim, hidden)
        self.blocks = nn.ModuleList(_RunConvBlock(hidden, d) for d in dilations)
        self.norm = nn.LayerNorm(hidden)
        self.onset_head = nn.Linear(hidden, base.groups)
        self.duration_head = nn.Linear(hidden, base.groups * base.durations)
        nn.init.zeros_(self.onset_head.weight); nn.init.zeros_(self.onset_head.bias)
        nn.init.zeros_(self.duration_head.weight); nn.init.zeros_(self.duration_head.bias)
        self.freeze_base()

    def freeze_base(self):
        self.base.eval().requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        return self

    def train(self, mode: bool = True):
        # The static prior is a separately fitted control.  Reassert this on
        # every mode switch so a caller cannot accidentally update its
        # running state or enable gradients while fitting the residual.
        super().train(mode)
        self.freeze_base()
        return self

    def residual_logits(self, condition: torch.Tensor, valid: torch.Tensor):
        """Return a residual whose zero input is exactly zero.

        Subtracting the response to a zero condition matters after training:
        convolution/normalization biases otherwise let the residual alter the
        static arm even when deployment supplies no local prosody.  The
        subtraction also makes the static arm a bitwise paired control.
        """
        if (condition.ndim != 3 or condition.shape[-1] != self.condition_dim
                or valid.shape != condition.shape[:2] or valid.dtype is not torch.bool
                or valid.device != condition.device):
            raise ValueError('condition[B,T,D] and Boolean valid[B,T] required')
        mask = valid[..., None]

        def encode(value):
            h = F.silu(self.input(torch.where(mask, value, torch.zeros_like(value))))
            h = torch.where(mask, h, 0.)
            run_ids = (~valid).long().cumsum(1)
            for block in self.blocks:
                h = block(h, valid, run_ids)
            h = torch.where(mask, self.norm(h), 0.)
            onset = torch.where(mask, self.onset_head(h), 0.)
            duration = self.duration_head(h).reshape(*h.shape[:2], self.base.groups,
                                                       self.base.durations)
            duration = torch.where(mask[..., None], duration, 0.)
            return onset, duration

        onset, duration = encode(condition)
        null_onset, null_duration = encode(torch.zeros_like(condition))
        return onset.tanh()-null_onset.tanh(), duration.tanh()-null_duration.tanh()

    def forward(self, condition: torch.Tensor, context: torch.Tensor, valid: torch.Tensor):
        if (condition.ndim != 3 or condition.shape[-1] != self.condition_dim or
                valid.shape != condition.shape[:2] or valid.dtype is not torch.bool):
            raise ValueError('condition[B,T,D] and Boolean valid[B,T] required')
        if not torch.isfinite(condition[valid]).all():
            raise ValueError('finite observed prosody condition required')
        base = self.base(context, condition.shape[1])
        delta_onset, delta_duration = self.residual_logits(condition, valid)
        return {'onset_logits': base['onset_logits'] + delta_onset,
                'duration_logits': base['duration_logits'] + delta_duration}


__all__ = ['StaticScheduleHead', 'ProsodyResidualSchedule']
