"""Autoregressive audio-to-motion model with an explicit previous-action state.

The model predicts the normalized nine-channel upper-face residual directly.  It
is intentionally separate from the flow/adapter experiments: the GRU receives
the preceding generated residual at every frame, while a Gaussian head keeps
the one-to-many nature of audio-to-face mapping.  Training uses teacher forcing;
inference uses the generated sample as the next-frame history.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _check_mask(mask: torch.Tensor, batch: int, length: int, device: torch.device) -> None:
    if (not torch.is_tensor(mask) or mask.shape != (batch, length)
            or mask.dtype is not torch.bool or mask.device != device):
        raise ValueError("valid must be Boolean [B,T] on the input device")


def _check_sequence(value: torch.Tensor, shape: tuple[int, ...], name: str,
                    device: torch.device, *, finite_valid: Optional[torch.Tensor] = None) -> None:
    if (not torch.is_tensor(value) or value.shape != shape or not value.is_floating_point()
            or value.device != device):
        raise ValueError(f"{name} must be floating {shape} on the input device")
    if finite_valid is None:
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    elif finite_valid.any() and not torch.isfinite(value[finite_valid]).all():
        raise ValueError(f"observed {name} must be finite")


class HistoryConditionedMotionGRU(nn.Module):
    """Frame-causal Gaussian residual predictor.

    ``audio`` is aligned to native frames. ``context`` contains the frozen
    global-affect/identity/baseline condition. Invalid frames are treated as
    gaps: the hidden state is reset to the context state and no output is
    produced. This prevents motion from leaking across native clock gaps.
    """

    def __init__(self, audio_dim: int = 1540, context_dim: int = 202,
                 motion_dim: int = 9, hidden: int = 128, depth: int = 1):
        super().__init__()
        for name, value in (("audio_dim", audio_dim), ("context_dim", context_dim),
                            ("motion_dim", motion_dim), ("hidden", hidden), ("depth", depth)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if depth != 1:
            raise ValueError("Only depth=1 is supported; GRU depth is explicit in hidden state")
        self.config = {"audio_dim": audio_dim, "context_dim": context_dim,
                       "motion_dim": motion_dim, "hidden": hidden, "depth": depth}
        self.audio_dim, self.context_dim, self.motion_dim = audio_dim, context_dim, motion_dim
        self.hidden = hidden
        self.audio_projection = nn.Sequential(nn.Linear(audio_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.context_projection = nn.Sequential(nn.Linear(context_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.previous_projection = nn.Sequential(nn.Linear(motion_dim, hidden), nn.SiLU())
        self.gru = nn.GRUCell(2 * hidden, hidden)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 2 * motion_dim))
        # Start with a quiet residual and a conservative uncertainty. This is
        # an initialization choice, not a training-time amplitude clamp.
        nn.init.zeros_(self.output[-1].bias[:motion_dim])
        nn.init.constant_(self.output[-1].bias[motion_dim:], -1.0)

    def _validate_inputs(self, audio: torch.Tensor, context: torch.Tensor,
                         valid: torch.Tensor) -> None:
        if (not torch.is_tensor(audio) or audio.ndim != 3
                or audio.shape[-1] != self.audio_dim):
            raise ValueError(f"audio must have shape [B,T,{self.audio_dim}]")
        batch, length = audio.shape[:2]
        _check_sequence(audio, tuple(audio.shape), "audio", audio.device)
        _check_mask(valid, batch, length, audio.device)
        if (not torch.is_tensor(context) or context.shape != (batch, self.context_dim)
                or context.device != audio.device or not context.is_floating_point()
                or not torch.isfinite(context).all()):
            raise ValueError(f"context must be finite floating [B,{self.context_dim}]")

    def _rollout(self, audio: torch.Tensor, context: torch.Tensor, valid: torch.Tensor,
                 *, target: Optional[torch.Tensor] = None,
                 teacher_forcing: bool = False, generator: Optional[torch.Generator] = None,
                 temperature: float = 1.0, sample: bool = False):
        self._validate_inputs(audio, context, valid)
        batch, length = audio.shape[:2]
        if target is not None:
            _check_sequence(target, (batch, length, self.motion_dim), "target", audio.device,
                            finite_valid=valid[..., None].expand(-1, -1, self.motion_dim))
        if teacher_forcing and target is None:
            raise ValueError("teacher_forcing requires target")
        if type(temperature) not in (float, int) or not 0 < float(temperature) <= 10:
            raise ValueError("temperature must be in (0,10]")
        base = self.context_projection(context)
        hidden = base
        previous = audio.new_zeros(batch, self.motion_dim)
        means, logstds, samples = [], [], []
        for index in range(length):
            frame_valid = valid[:, index]
            # Any missing frame is a true clock gap. Resetting to base keeps
            # the next observed frame independent of an unobserved interval.
            hidden = torch.where(frame_valid[:, None], hidden, base)
            previous = torch.where(frame_valid[:, None], previous, torch.zeros_like(previous))
            inp = torch.cat((self.audio_projection(audio[:, index]),
                             self.previous_projection(previous)), -1)
            proposal = self.gru(inp, hidden)
            hidden = torch.where(frame_valid[:, None], proposal, base)
            params = self.output(hidden)
            mean, logstd = params.split(self.motion_dim, -1)
            logstd = logstd.clamp(-5.0, 2.0)
            mean = torch.where(frame_valid[:, None], mean, torch.zeros_like(mean))
            logstd = torch.where(frame_valid[:, None], logstd, torch.zeros_like(logstd))
            if sample:
                noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                                    generator=generator)
                current = mean + noise * logstd.exp() * float(temperature)
                current = torch.where(frame_valid[:, None], current, torch.zeros_like(current))
            else:
                current = mean
            means.append(mean); logstds.append(logstd); samples.append(current)
            if teacher_forcing:
                previous = torch.where(frame_valid[:, None], target[:, index], previous)
            else:
                previous = current
        stack = lambda values: torch.stack(values, 1) if values else audio.new_zeros(batch, 0, self.motion_dim)
        return stack(means), stack(logstds), stack(samples)

    def forward(self, audio: torch.Tensor, context: torch.Tensor, valid: torch.Tensor,
                target: Optional[torch.Tensor] = None, *, teacher_forcing: bool = True):
        """Return Gaussian parameters; target is used only for teacher forcing."""
        mean, logstd, _ = self._rollout(audio, context, valid, target=target,
                                         teacher_forcing=teacher_forcing, sample=False)
        return {"mean": mean, "logstd": logstd}

    def nll_loss(self, audio: torch.Tensor, context: torch.Tensor, target: torch.Tensor,
                 valid: torch.Tensor, *, teacher_forcing: bool = True) -> torch.Tensor:
        """Mean Gaussian NLL over observed residual coordinates."""
        out = self.forward(audio, context, valid, target, teacher_forcing=teacher_forcing)
        if not valid.any():
            raise ValueError("at least one valid frame is required")
        error = (target - out["mean"]) * torch.exp(-out["logstd"])
        nll = 0.5 * (error.square() + 2.0 * out["logstd"])
        mask = valid[..., None].expand_as(nll)
        return nll[mask].mean()

    @torch.no_grad()
    def sample(self, audio: torch.Tensor, context: torch.Tensor, valid: torch.Tensor,
               *, generator: Optional[torch.Generator] = None,
               temperature: float = 1.0, stochastic: bool = True) -> torch.Tensor:
        """Autoregressively generate normalized residuals frame by frame."""
        _, _, samples = self._rollout(audio, context, valid, generator=generator,
                                       temperature=temperature, sample=stochastic)
        return samples


__all__ = ["HistoryConditionedMotionGRU"]
