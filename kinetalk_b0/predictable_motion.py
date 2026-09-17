"""Train-only weighted ridge/RRR motion subspaces on the native motion clock.

The fixed motion teacher is ``center(Y) @ basis``, never a prediction from
audio.  RRR chooses shared motion directions using training audio-motion
cross-covariance; PCA chooses directions using training motion covariance.
Both use the same ridge audio predictor and neither divides the face into
model regions.  All fitting and prediction use CPU float64 for reproducibility.
Hyperparameter/split selection belongs to the caller, outside this module.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import math
import torch
from torch.nn import functional as F


def _shape(values: torch.Tensor, weight: torch.Tensor) -> None:
    if values.ndim != 3 or weight.shape != values.shape[:2]:
        raise ValueError("values [B,K,D] and weight [B,K] are required")
    if not values.is_floating_point() or values.is_complex():
        raise ValueError("values must be real floating point")
    if values.shape[1] < 1 or values.shape[-1] < 1:
        raise ValueError("values must contain bins and feature channels")
    if weight.is_complex():
        raise ValueError("weight must be real")


def weighted_clip_center(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CPU-double centering with true bin weights; invalid padding stays zero.

    Entirely empty clips are allowed for projection/prediction and return zero.
    Nonfinite padded values are ignored, but nonfinite observed values and
    negative/nonfinite weights raise.  Fitting requires positive total weight.
    """
    _shape(values, weight)
    x = values.detach().to(device="cpu", dtype=torch.float64)
    w = weight.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("weight must be finite and nonnegative")
    valid = w > 0
    x = torch.where(valid[..., None], x, 0)
    if not torch.isfinite(x).all():
        raise ValueError("observed values must be finite")
    mean = (x * w[..., None]).sum(1, keepdim=True) / w.sum(1)[:, None, None].clamp_min(1e-30)
    return torch.where(valid[..., None], x - mean, 0)


def bin_centered_frames(frames: torch.Tensor, valid: torch.Tensor,
                        stride: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin valid original-clock frames and clip-center using partial-bin counts."""
    _shape(frames, valid)
    if valid.dtype != torch.bool or isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("boolean valid mask and positive integer stride are required")
    x = frames.detach().to(device="cpu", dtype=torch.float64)
    mask = valid.detach().to(device="cpu")
    x = torch.where(mask[..., None], x, 0)
    if not torch.isfinite(x).all():
        raise ValueError("observed frames must be finite")
    extra = (-x.shape[1]) % stride
    w = F.pad(mask.double(), (0, extra)).reshape(x.shape[0], -1, stride).sum(2)
    bins = F.pad(x, (0, 0, 0, extra)).reshape(x.shape[0], -1, stride, x.shape[-1]).sum(2)
    bins = bins / w[..., None].clamp_min(1)
    return weighted_clip_center(bins, w), w


def _train_indices(train_ids: torch.Tensor | Sequence[int], batch: int) -> torch.Tensor:
    ids = torch.as_tensor(train_ids, device="cpu")
    if ids.ndim != 1 or not len(ids) or ids.is_floating_point() or ids.is_complex() or ids.dtype == torch.bool:
        raise ValueError("train_ids must be a nonempty vector of integer indices")
    ids = ids.long()
    if ids.min() < 0 or ids.max() >= batch or len(ids.unique()) != len(ids):
        raise ValueError("train_ids must be unique and in range")
    return ids


def _canonical_columns(basis: torch.Tensor) -> torch.Tensor:
    """Choose deterministic column signs; repeated-eigenvalue spaces may rotate."""
    maxima = basis.abs().argmax(0)
    signs = basis[maxima, torch.arange(basis.shape[1])].sign()
    return basis * torch.where(signs == 0, torch.ones_like(signs), signs)


def _top_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # All matrices are theoretically symmetric; remove accumulated roundoff.
    eigenvalues, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    return _canonical_columns(vectors.flip(1)), eigenvalues.flip(0).clamp_min(0)


@torch.no_grad()
def fit_motion_path(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    train_ids: torch.Tensor | Sequence[int],
    alphas: Sequence[float],
    ranks: Sequence[int],
    *,
    methods: Sequence[str] = ("rrr", "pca", "ridge"),
    std_floor: float = 1e-8,
) -> list[dict[str, Any]]:
    """Fit a matched regularized multi-output motion prediction path.

    Each input is binned ``[B,K,D/C]`` and is re-centered per clip.  Selection
    occurs before reading values, so held-out NaNs, weights, or motion cannot
    affect fitted scales/directions.  Features use training weighted RMS;
    motion stays in native controller units.  With standardized X, define

    ``G=X'WX/sum(W)``, ``H=X'WY/sum(W)``, ``A=G+alpha I``.

    Ridge is ``B=A^-1 H``. Regularized RRR minimizes weighted mean squared
    motion error plus ``alpha*||B||_F^2`` subject to rank(B)<=r. Its output
    basis consists of the top right singular vectors of ``A^-1/2 H``, or
    equivalently the top eigenvectors of ``H' A^-1 H``. The coefficient is
    ``B_r=B U_r U_r'``. This is NOT the SVD of the unweighted ridge coefficient.

    PCA uses top eigenvectors of training ``Y'WY/sum(W)`` and fits the same
    ridge predictor to those targets. ``basis [C,r]`` is orthonormal and
    ``weights [D,r]`` predicts controls. Direct ridge has rank=C and basis=I,
    and appears once per alpha regardless of requested reduced ranks.

    The D-dimensional Gram eigendecomposition is reused across all alpha,
    method and rank choices. No validation or automatic model selection runs.
    """
    _shape(x, weight)
    _shape(y, weight)
    ids = _train_indices(train_ids, x.shape[0])
    if len(y) != len(x):
        raise ValueError("x and y must have matching batch size")
    alphas = tuple(float(a) for a in alphas)
    if not alphas or any(not math.isfinite(a) or a <= 0 for a in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be distinct finite positive numbers")
    ranks = tuple(ranks)
    if (any(isinstance(r, bool) or not isinstance(r, int) or r < 1 or r > y.shape[-1] for r in ranks)
            or len(set(ranks)) != len(ranks)):
        raise ValueError("ranks must be distinct positive integers <= motion channels")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in ("rrr", "pca", "ridge") for m in methods):
        raise ValueError("methods must be distinct choices of rrr, pca, ridge")
    if any(m != "ridge" for m in methods) and not ranks:
        raise ValueError("at least one rank is required for reduced subspaces")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be finite and positive")

    # Do not validate or move full held-out tensors before selecting train IDs.
    wt = weight.index_select(0, ids.to(weight.device)).detach().cpu().double()
    xt = weighted_clip_center(x.index_select(0, ids.to(x.device)), wt)
    yt = weighted_clip_center(y.index_select(0, ids.to(y.device)), wt)
    total_weight = wt.sum()
    if not total_weight > 0:
        raise ValueError("training data must have positive observed weight")
    raw_std = ((xt.square() * wt[..., None]).sum((0, 1)) / total_weight).sqrt()
    std = raw_std.clamp_min(std_floor)
    design = (xt / std).reshape(-1, x.shape[-1])
    target = yt.reshape(-1, y.shape[-1])
    flat_w = wt.reshape(-1, 1)
    gram = design.T @ (design * flat_w) / total_weight
    cross = design.T @ (target * flat_w) / total_weight
    if not torch.isfinite(gram).all() or not torch.isfinite(cross).all():
        raise ValueError("finite inputs overflowed covariance computation")
    eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) * .5)
    eigenvalues = eigenvalues.clamp_min(0)
    rotated_cross = eigenvectors.T @ cross
    pca_basis, pca_spectrum = (None, None)
    if "pca" in methods:
        pca_basis, pca_spectrum = _top_basis(target.T @ (target * flat_w) / total_weight)
    states = []
    channels = y.shape[-1]
    for alpha in alphas:
        ridge = eigenvectors @ (rotated_cross / (eigenvalues[:, None] + alpha))
        rrr_basis, rrr_spectrum = (None, None)
        if "rrr" in methods:
            rrr_basis, rrr_spectrum = _top_basis(cross.T @ ridge)
        for method in methods:
            for rank in ((channels,) if method == "ridge" else ranks):
                if method == "ridge":
                    basis, spectrum = torch.eye(channels, dtype=torch.float64), None
                elif method == "pca":
                    basis, spectrum = pca_basis[:, :rank].clone(), pca_spectrum.clone()
                else:
                    basis, spectrum = rrr_basis[:, :rank].clone(), rrr_spectrum.clone()
                states.append({
                    "schema": "predictable_motion_ridge_v1", "method": method,
                    "alpha": alpha, "rank": rank, "std": std.clone(),
                    "basis": basis, "weights": ridge @ basis,
                    "spectrum": spectrum, "train_ids": ids.clone(),
                    "training_weight": float(total_weight),
                    "training_bins": int((wt > 0).sum()),
                    "zero_variance_features": int((raw_std <= std_floor).sum()),
                    "input_dim": x.shape[-1], "motion_dim": channels,
                    "target_definition": "weighted_clip_center(real_motion) @ basis",
                    "preprocessing": "weighted clip centering then divide train feature RMS; no target whitening",
                    "objective": "weighted_mean_native_motion_MSE_sum_over_channels + alpha * coefficient_Frobenius_squared",
                    "dtype": "float64", "device": "cpu",
                })
    return states


def fit_predictable_motion(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
    train_ids: torch.Tensor | Sequence[int], *,
    alpha: float = 1.0, rank: int | None = None, method: str = "rrr",
    std_floor: float = 1e-8,
) -> dict[str, Any]:
    """Single-state wrapper; direct ridge ignores rank and uses all channels."""
    ranks = () if method == "ridge" else (rank,)
    return fit_motion_path(x, y, weight, train_ids, [alpha], ranks,
                           methods=(method,), std_floor=std_floor)[0]


def _state_tensors(state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    std, basis, weights = (state[k].detach().cpu().double() for k in ("std", "basis", "weights"))
    if std.ndim != 1 or basis.ndim != 2 or weights.shape != (len(std), basis.shape[1]):
        raise ValueError("invalid predictor state tensor shapes")
    if not all(torch.isfinite(v).all() for v in (std, basis, weights)) or not (std > 0).all():
        raise ValueError("predictor state must be finite with positive std")
    return std, basis, weights


def predict_controls(x: torch.Tensor, weight: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    """Audio-only prediction [B,K,r], requiring no motion, labels, or speaker IDs."""
    std, _, coefficients = _state_tensors(state)
    centered = weighted_clip_center(x, weight)
    if centered.shape[-1] != len(std):
        raise ValueError("audio feature dimension differs from fitted state")
    return weighted_clip_center((centered / std) @ coefficients, weight)


def decode_controls(controls: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    """Decode shared subspace controls to all motion channels; no regional masks."""
    _, basis, _ = _state_tensors(state)
    if controls.ndim != 3 or controls.shape[-1] != basis.shape[1]:
        raise ValueError("controls must be [B,K,rank]")
    z = controls.detach().cpu().double()
    if not torch.isfinite(z).all():
        raise ValueError("controls must be finite")
    return z @ basis.T


def predict_motion(x: torch.Tensor, weight: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    """Audio-only prediction decoded into native motion controller units."""
    return decode_controls(predict_controls(x, weight, state), state)


def motion_target(y: torch.Tensor, weight: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    """Project real centered motion onto the fixed basis, for supervision only."""
    _, basis, _ = _state_tensors(state)
    centered = weighted_clip_center(y, weight)
    if centered.shape[-1] != basis.shape[0]:
        raise ValueError("motion channel dimension differs from fitted state")
    return centered @ basis
