"""Learned upper-face execution from neutral-relative regional intensity.

The only time-varying decoder input is a two-region intensity trajectory.
Global emotion, identity and the independent neutral anchor are static. No
prior upper trajectory, query motion, generated history, clip centering or
random motion enters this module. The caller composes the returned nine
absolute coefficients with ``slow_state_affect.compose_upper_face`` to retain
the other 43 channels and invalid frames exactly.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .slow_state_affect import UPPER_INDICES


def _check_sequence(value: torch.Tensor, valid: torch.Tensor, width: int,
                    name: str) -> None:
    if (not torch.is_tensor(value) or not value.is_floating_point()
            or value.ndim != 3 or value.shape[-1] != width
            or min(value.shape[:2]) < 1):
        raise ValueError(f"{name} must be nonempty floating [B,T,{width}]")
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != value.shape[:2] or valid.device != value.device):
        raise ValueError("valid must be Boolean [B,T] on the sequence device")
    if not torch.isfinite(value[valid]).all():
        raise ValueError(f"observed {name} must be finite")


def _check_static(value: torch.Tensor, sequence: torch.Tensor, width: int,
                  name: str) -> None:
    if (not torch.is_tensor(value) or value.shape != (len(sequence), width)
            or value.dtype != sequence.dtype or value.device != sequence.device
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite [B,{width}] with sequence dtype/device")


def _map_runs(value: torch.Tensor, valid: torch.Tensor,
              transform: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Apply a [runs,channels,time] operation separately to valid runs.

    Grouping equal-length runs avoids one small convolution per sequence. All
    padding is local to a run: a gap or added batch padding cannot become a
    temporal neighbour. Assignment retains gradients to every valid input.
    """
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for row, flags in enumerate(valid.detach().cpu().tolist()):
        left = None
        for position, observed in enumerate(flags + [False]):
            if observed and left is None:
                left = position
            elif not observed and left is not None:
                groups.setdefault(position - left, []).append((row, left, position))
                left = None
    result = torch.zeros_like(value)
    for runs in groups.values():
        segment = torch.stack([value[row, left:right] for row, left, right in runs])
        transformed = transform(segment.transpose(1, 2)).transpose(1, 2)
        for index, (row, left, right) in enumerate(runs):
            result[row, left:right] = transformed[index]
    return result


def neutral_relative_intensity(upper9: torch.Tensor, anchor9: torch.Tensor,
                               scales9: torch.Tensor, valid: torch.Tensor,
                               window: int = 5) -> torch.Tensor:
    """Read brow/eye mean absolute deviation from an independent anchor.

    For each frame, normalize ``upper9 - anchor9`` by positive TRAIN-fitted
    channel scales, then average absolute values over brow5 and eye4. An odd
    moving average uses replicated endpoints within each valid run. There is
    no clip centering: expression held away from neutral has nonzero intensity.
    This is a coefficient amplitude proxy, not an emotion-label measurement.

    ``scales9`` accepts [9] or [B,9]. Invalid frames return zero and may contain
    nonfinite upper coefficients without contaminating values or gradients.
    Use the same anchor/scales/window for supervision, oracle and output loss.
    """
    _check_sequence(upper9, valid, 9, "upper9")
    _check_static(anchor9, upper9, 9, "anchor9")
    if (not torch.is_tensor(scales9) or scales9.shape not in ((9,), (len(upper9), 9))
            or scales9.dtype != upper9.dtype or scales9.device != upper9.device
            or not torch.isfinite(scales9).all() or (scales9 <= 0).any()):
        raise ValueError("scales9 must be positive finite [9] or [B,9] with upper9 dtype/device")
    if type(window) is not int or window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")
    # Sanitize before arithmetic, so NaN padding cannot enter backward paths.
    anchor = anchor9[:, None]
    clean = torch.where(valid[..., None], upper9, anchor)
    scales = scales9[None, None] if scales9.ndim == 1 else scales9[:, None]
    normalized = ((clean - anchor) / scales).abs()
    intensity = torch.stack((normalized[..., :5].mean(-1),
                             normalized[..., 5:].mean(-1)), dim=-1)
    if window == 1:
        return intensity
    radius = window // 2
    return _map_runs(intensity, valid,
                     lambda run: F.avg_pool1d(F.pad(run, (radius, radius), mode="replicate"),
                                             kernel_size=window, stride=1))


class ReferenceIntensityDecoder(nn.Module):
    """A small learned 9D decoder with a bounded neutral-relative output.

    Replicated run endpoints preserve constant inputs, including one-frame
    runs. The network therefore cannot invent boundary motion from a static
    intensity control. It can learn new motion where a frozen prior was flat,
    because it predicts nine coefficients rather than scaling an old carrier.

    Output is sigmoid(logit(clamp(anchor,.01,.99)) + learned residual). This
    anchors initialization near neutral while allowing the upper mean to change.
    The head has normal(std=.01) weights and zero bias, deliberately not a zero
    head: intensity gradients are available immediately. Other layers retain
    PyTorch initialization; callers own the random seed. No stochastic operation
    is used during forward. Invalid frames return the original supplied anchor;
    composition is responsible for copying the baseline at those frames.
    """

    def __init__(self, global_dim: int = 64, identity_dim: int = 128,
                 hidden: int = 64):
        super().__init__()
        if any(type(value) is not int or value < 1
               for value in (global_dim, identity_dim, hidden)):
            raise ValueError("global_dim, identity_dim and hidden must be positive integers")
        self.global_dim = global_dim
        self.identity_dim = identity_dim
        self.hidden = hidden
        self.input = nn.Linear(2 + global_dim + identity_dim + 9, hidden)
        self.temporal = nn.ModuleList((nn.Conv1d(hidden, hidden, 5),
                                       nn.Conv1d(hidden, hidden, 3)))
        self.norms = nn.ModuleList((nn.LayerNorm(hidden), nn.LayerNorm(hidden)))
        self.head = nn.Linear(hidden, 9)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def export_config(self) -> dict[str, int]:
        return {"global_dim": self.global_dim, "identity_dim": self.identity_dim,
                "hidden": self.hidden}

    def _temporal_run(self, hidden: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.temporal, self.norms):
            value = F.silu(norm(hidden.transpose(1, 2))).transpose(1, 2)
            radius = conv.kernel_size[0] // 2
            hidden = hidden + F.silu(conv(F.pad(value, (radius, radius), mode="replicate")))
        return hidden

    def forward(self, intensity: torch.Tensor, valid: torch.Tensor,
                global_code: torch.Tensor, identity_code: torch.Tensor,
                anchor: torch.Tensor) -> torch.Tensor:
        _check_sequence(intensity, valid, 2, "intensity")
        if (intensity[valid] < 0).any():
            raise ValueError("observed intensity must be nonnegative")
        _check_static(global_code, intensity, self.global_dim, "global_code")
        _check_static(identity_code, intensity, self.identity_dim, "identity_code")
        if not torch.is_tensor(anchor) or anchor.ndim != 2 or anchor.shape[-1] not in (9, 52):
            raise ValueError("anchor must be static [B,9] or [B,52]")
        upper_anchor = anchor[..., list(UPPER_INDICES)] if anchor.shape[-1] == 52 else anchor
        _check_static(upper_anchor, intensity, 9, "anchor")
        clean_intensity = torch.where(valid[..., None], intensity, 0.)
        context = torch.cat((global_code, identity_code, upper_anchor), dim=-1)
        features = torch.cat((clean_intensity, context[:, None].expand(-1, intensity.shape[1], -1)), -1)
        hidden = F.silu(self.input(features))
        hidden = _map_runs(hidden, valid, self._temporal_run)
        residual = self.head(hidden)
        anchor_logits = torch.logit(upper_anchor.clamp(.01, .99))[:, None]
        generated = torch.sigmoid(anchor_logits + residual)
        return torch.where(valid[..., None], generated, upper_anchor[:, None])


class ReferenceIntensityStudent(nn.Module):
    """Compact native-rate audio student for the same regional intensity.

    Inputs are frozen, TRAIN-fitted features (normally PCA24 plus prosody4).
    The temporal branch sees only audio features. A separate static branch sees
    their masked clip mean, global emotion, identity and the neutral anchor.
    Temporal convolutions never cross an invalid gap; runs intentionally share
    the clip-level static context. This is an offline, clip-conditioned model.

    ``temporal_mode='static'`` replaces temporal hidden features with their
    masked clip mean before the output head. It retains the same static context
    and produces constant valid-frame intensity. No upper activity target,
    emotion label, query motion or target-derived statistic enters forward.

    Returns intensity [B,T,2] and local [B,T,hidden], with invalid entries zero.
    Softplus gives nonnegative intensity without an arbitrary upper clamp.
    The small nonzero output head permits immediate acoustic gradients; bias
    -1 initializes predictions near softplus(-1), about .31 in supplied units.
    """

    def __init__(self, input_dim: int = 28, global_dim: int = 64,
                 identity_dim: int = 128, hidden: int = 48):
        super().__init__()
        if any(type(value) is not int or value < 1
               for value in (input_dim, global_dim, identity_dim, hidden)):
            raise ValueError("input_dim, global_dim, identity_dim and hidden must be positive integers")
        self.input_dim = input_dim
        self.global_dim = global_dim
        self.identity_dim = identity_dim
        self.hidden = hidden
        self.input = nn.Linear(input_dim, hidden)
        self.temporal = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, dilation=dilation)
                                       for dilation in (1, 2, 4))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in self.temporal)
        self.context = nn.Sequential(nn.Linear(input_dim + global_dim + identity_dim + 9, hidden),
                                     nn.SiLU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.constant_(self.head.bias, -1.)

    def export_config(self) -> dict[str, int]:
        return {"input_dim": self.input_dim, "global_dim": self.global_dim,
                "identity_dim": self.identity_dim, "hidden": self.hidden}

    def _temporal_run(self, hidden: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.temporal, self.norms):
            value = F.silu(norm(hidden.transpose(1, 2))).transpose(1, 2)
            radius = conv.dilation[0]
            hidden = hidden + F.silu(conv(F.pad(value, (radius, radius), mode="replicate")))
        return hidden

    def forward(self, features: torch.Tensor, valid: torch.Tensor,
                global_code: torch.Tensor, identity_code: torch.Tensor,
                anchor: torch.Tensor, *, temporal_mode: str = "full") -> dict[str, torch.Tensor]:
        _check_sequence(features, valid, self.input_dim, "features")
        _check_static(global_code, features, self.global_dim, "global_code")
        _check_static(identity_code, features, self.identity_dim, "identity_code")
        if temporal_mode not in ("full", "static"):
            raise ValueError("temporal_mode must be 'full' or 'static'")
        if not torch.is_tensor(anchor) or anchor.ndim != 2 or anchor.shape[-1] not in (9, 52):
            raise ValueError("anchor must be static [B,9] or [B,52]")
        upper_anchor = anchor[..., list(UPPER_INDICES)] if anchor.shape[-1] == 52 else anchor
        _check_static(upper_anchor, features, 9, "anchor")
        clean = torch.where(valid[..., None], features, 0.)
        count = valid.sum(1, keepdim=True).clamp_min(1)
        pooled = clean.sum(1) / count
        context = self.context(torch.cat((pooled, global_code, identity_code, upper_anchor), -1))
        local = _map_runs(F.silu(self.input(clean)), valid, self._temporal_run)
        if temporal_mode == "static":
            mean_local = local.sum(1) / count
            local = torch.where(valid[..., None], mean_local[:, None], 0.)
        intensity = F.softplus(self.head(F.silu(local + context[:, None])))
        intensity = torch.where(valid[..., None], intensity, 0.)
        return {"intensity": intensity, "local": local}


__all__ = ["ReferenceIntensityDecoder", "ReferenceIntensityStudent", "neutral_relative_intensity"]
