"""A small audio residual predictor around one frozen native rank8 ridge fit.

Both arms share the same field, weighted clip centering, per-bin LayerNorm,
and zero-initialized final output layer. Only temporal depthwise kernels mix
adjacent bins. A pointwise arm still receives contextual audio features and
shares the clip mean; it is not a claim of having no temporal information.
No motion, labels, identity, gate, or renderer is used at prediction time.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from kinetalk_b0.predictable_motion import weighted_clip_center


def _center(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Differentiable centering; caller has checked finite nonnegative weights."""
    valid = weight > 0
    clean = torch.where(valid[..., None], values, 0)
    mean = ((clean * weight[..., None]).sum(1, keepdim=True)
            / weight.sum(1)[:, None, None].clamp_min(1e-30))
    return torch.where(valid[..., None], clean - mean, 0)


def _native_state(state: dict[str, Any]):
    if state.get("method") != "rrr" or state.get("rank") != 8 or state.get("alpha") != 1.0:
        raise ValueError("Requires the fixed native RRR rank8/alpha1 state")
    std, basis, weights = (state[k].detach().cpu().double() for k in ("std", "basis", "weights"))
    if (std.ndim != 1 or not len(std) or basis.ndim != 2 or basis.shape[1] != 8
            or weights.shape != (len(std), 8)):
        raise ValueError("Invalid fixed ridge tensor shapes")
    if not all(torch.isfinite(v).all() for v in (std, basis, weights)) or not (std > 0).all():
        raise ValueError("Fixed ridge tensors must be finite with positive feature RMS")
    if not torch.allclose(basis.T @ basis, torch.eye(8, dtype=torch.float64), rtol=1e-6, atol=1e-7):
        raise ValueError("Requires a native Euclidean-orthonormal motion basis")
    if "coordinate_system" in state and state["coordinate_system"] != "native":
        raise ValueError("Scaled/transformed motion states cannot be used here")
    return std, basis, weights


@torch.no_grad()
def fit_control_scale(y: torch.Tensor, weight: torch.Tensor,
                      fit_ids: torch.Tensor | Sequence[int],
                      state: dict[str, Any]) -> torch.Tensor:
    """RMS of real centered fold-fit motion controls, floored at 1e-6.

Select fit IDs before reading observations. ``y`` may already contain the
state's C native channels, or full motion with ``motion_channel_indices`` in
the state. Neither validation values nor RNG state influence this calculation.
    """
    _, basis, _ = _native_state(state)
    if y.ndim != 3 or weight.shape != y.shape[:2]:
        raise ValueError("Expected y [B,K,C] and weight [B,K]")
    ids = torch.as_tensor(fit_ids, device="cpu")
    if (ids.ndim != 1 or not len(ids) or ids.dtype == torch.bool
            or ids.is_floating_point() or ids.is_complex()):
        raise ValueError("fit_ids must be a nonempty integer vector")
    ids = ids.long()
    if ids.min() < 0 or ids.max() >= len(y) or len(ids.unique()) != len(ids):
        raise ValueError("fit_ids must be unique and in range")
    selected = y.index_select(0, ids.to(y.device))
    wf = weight.index_select(0, ids.to(weight.device)).detach().cpu().double()
    if selected.shape[-1] != basis.shape[0]:
        channels = state.get("motion_channel_indices")
        if (channels is None or len(channels) != basis.shape[0]
                or len(set(channels)) != len(channels)
                or min(channels) < 0 or max(channels) >= selected.shape[-1]):
            raise ValueError("y channels differ from the fixed native basis")
        selected = selected[..., channels]
    controls = weighted_clip_center(selected, wf) @ basis
    if not wf.sum() > 0:
        raise ValueError("Control RMS requires positive fold-fit observed weight")
    scale = ((controls.square() * wf[..., None]).sum((0, 1)) / wf.sum()).sqrt().clamp_min(1e-6)
    if not torch.isfinite(scale).all():
        raise ValueError("Control RMS overflowed")
    return scale


class _MaskedResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, hidden)
        self.activation = nn.GELU()
        # Initialized after every arm-shared layer, so same-seed shared layer
        # values are identical despite different depthwise kernel shapes.
        self.depthwise = None

    def forward(self, values, valid):
        mask = valid[..., None]
        h = torch.where(mask, self.norm(values), 0)
        h = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        h = torch.where(mask, h, 0)
        h = self.output(self.activation(h))
        return torch.where(mask, values + h, 0)


class _SharedResidualField(nn.Module):
    def __init__(self, inputs: int, hidden: int, arm: str):
        super().__init__()
        self.input = nn.Linear(inputs, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([_MaskedResidualBlock(hidden) for _ in range(2)])
        self.output = nn.Linear(hidden, 8)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        kernel = 1 if arm == "pointwise" else 3
        for block, dilation in zip(self.blocks, (1, 2)):
            block.depthwise = nn.Conv1d(hidden, hidden, kernel, groups=hidden,
                                       dilation=dilation, padding=dilation * (kernel // 2),
                                       bias=False)

    def forward(self, features, valid):
        mask = valid[..., None]
        features = torch.where(mask, features, 0)
        h = self.activation(self.input_norm(self.input(features)))
        h = torch.where(mask, h, 0)
        for block in self.blocks:
            h = block(h, valid)
        return torch.where(mask, self.output(h), 0)


class FrozenRidgeRefiner(nn.Module):
    """Native prediction = [frozen ridge + scale * centered residual] @ U.T.

The frozen buffers preserve the source state's float64 values by default;
trainable layers default to float32. Normalized features are cast only at the
refiner boundary and residuals are cast back before addition. ``model.float()``
explicitly opts into a float32 frozen path. With an identical seed, all
same-shaped trainable tensors match between arms; only depthwise kernels
differ (256 additional temporal parameters for hidden=64).
    """
    def __init__(self, state: dict[str, Any], control_scale: torch.Tensor,
                 arm: str, hidden: int = 64):
        super().__init__()
        if arm not in ("pointwise", "temporal"):
            raise ValueError("arm must be pointwise or temporal")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        std, basis, weights = _native_state(state)
        scale = torch.as_tensor(control_scale).detach().cpu().double()
        if scale.shape != (8,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("control_scale must contain eight positive finite values")
        self.arm = arm
        self.register_buffer("feature_std", std.clone())
        self.register_buffer("basis", basis.clone())
        self.register_buffer("ridge_weights", weights.clone())
        self.register_buffer("control_scale", scale.clone())
        self.refiner = _SharedResidualField(len(std), hidden, arm)

    def _normalized(self, x: torch.Tensor, weight: torch.Tensor):
        if (x.ndim != 3 or x.shape[-1] != len(self.feature_std)
                or x.shape[1] < 1 or weight.shape != x.shape[:2]
                or not x.is_floating_point() or x.is_complex() or weight.is_complex()):
            raise ValueError("Expected real x [B,K,D] and weight [B,K]")
        w = weight.to(device=self.feature_std.device, dtype=self.feature_std.dtype)
        if not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError("Weights must be finite and nonnegative")
        values = x.to(device=self.feature_std.device, dtype=self.feature_std.dtype)
        values = torch.where((w > 0)[..., None], values, 0)
        if not torch.isfinite(values).all():
            raise ValueError("Observed audio values must be finite")
        return _center(values, w) / self.feature_std, w

    def controls(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normalized, w = self._normalized(x, weight)
        ridge = _center(normalized @ self.ridge_weights, w)
        features = normalized.to(dtype=self.refiner.input.weight.dtype)
        if not torch.isfinite(features).all():
            raise ValueError("Standardized audio features overflowed refiner precision")
        residual = self.refiner(features, w > 0).to(dtype=ridge.dtype)
        return ridge + self.control_scale * _center(residual, w)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.controls(x, weight) @ self.basis.T


class ConstrainedRidgeRefiner(FrozenRidgeRefiner):
    """Very small, ridge-initialized temporal correction in the native 8-D field.

    The frozen audio ridge remains the complete baseline.  A zero-initialized
    3-tap linear map acts on its eight controls, so the first forward pass is
    bitwise identical to :class:`FrozenRidgeRefiner`.  ``return_residual`` is
    used only by the isolated probe to apply one trust-region penalty; it is
    deliberately not part of the deployment interface.
    """
    def __init__(self, state: dict[str, Any], control_scale: torch.Tensor,
                 *, temporal: bool = True):
        # Do not allocate the 100k-parameter feature MLP used by the previous
        # diagnostic refiner.  Replacing it with a 8x8x3 linear filter keeps
        # the correction auditable and strongly constrained.
        nn.Module.__init__(self)
        std, basis, weights = _native_state(state)
        scale = torch.as_tensor(control_scale).detach().cpu().double()
        if scale.shape != (8,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("control_scale must contain eight positive finite values")
        self.register_buffer("feature_std", std.clone())
        self.register_buffer("basis", basis.clone())
        self.register_buffer("ridge_weights", weights.clone())
        self.register_buffer("control_scale", scale.clone())
        self.temporal = bool(temporal)
        kernel = 3 if self.temporal else 1
        self.correction = nn.Conv1d(8, 8, kernel, padding=kernel // 2,
                                    bias=False)
        nn.init.zeros_(self.correction.weight)

    def _ridge_controls(self, x: torch.Tensor, weight: torch.Tensor):
        normalized, w = self._normalized(x, weight)
        ridge = _center(normalized @ self.ridge_weights, w)
        return ridge, w

    def controls_with_residual(self, x: torch.Tensor, weight: torch.Tensor):
        ridge, w = self._ridge_controls(x, weight)
        # Convolution is performed in float32 for stable tiny updates while
        # preserving the frozen ridge buffers in their original precision.
        q = (ridge / self.control_scale).to(dtype=self.correction.weight.dtype)
        q = torch.where((w > 0)[..., None], q, 0)
        residual = self.correction(q.transpose(1, 2)).transpose(1, 2)
        residual = torch.where((w > 0)[..., None], residual, 0)
        residual = _center(residual.to(dtype=ridge.dtype), w)
        controls = ridge + self.control_scale * residual
        return controls, residual, ridge, w

    def controls(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.controls_with_residual(x, weight)[0]

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                *, return_residual: bool = False):
        controls, residual, ridge, w = self.controls_with_residual(x, weight)
        output = controls @ self.basis.T
        if return_residual:
            return output, residual, ridge, w
        return output
