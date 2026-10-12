"""Fixed-rig geometry observers used by the isolated v4 candidate.

The asset is a shared mesh proxy.  It is deliberately exposed as an affine
observer rather than presented as a physiological lip-contact model.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import torch


def load_rig_geometry(path: str | Path, *, scale=None):
    """Load four fixed-rig observers and validate the provisional asset.

    Returns ``(jacobian[4,52], offset[4], scale[4], channels, metadata)``.
    ``scale`` must be fitted from TRAIN geometry targets; the default ones are
    only suitable for a shape/smoke test, not a final experiment.
    """
    source = Path(path)
    with np.load(source, allow_pickle=False) as z:
        required = {"channels", "group_names", "lip_group_weights",
                    "neutral_mm", "blendshape_deltas_mm"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(f"geometry asset missing keys: {missing}")
        channels = np.asarray(z["channels"]).astype(str)
        groups = np.asarray(z["group_names"]).astype(str)
        weights = np.asarray(z["lip_group_weights"], dtype=np.float64)
        neutral = np.asarray(z["neutral_mm"], dtype=np.float64)
        deltas = np.asarray(z["blendshape_deltas_mm"], dtype=np.float64)
        if channels.shape != (52,) or len(set(channels.tolist())) != 52:
            raise ValueError("geometry asset must contain 52 unique channels")
        if weights.ndim != 2 or weights.shape[1] != len(groups):
            raise ValueError("invalid lip group weights")
        if deltas.shape != (52, neutral.shape[0], 3):
            raise ValueError("blendshape delta shape differs from neutral mesh")

        def centroid(name):
            ix = np.flatnonzero(groups == name)
            if len(ix) != 1:
                raise ValueError(f"expected one named group: {name}")
            w = weights[:, ix[0]]
            if not np.isfinite(w).all() or w.sum() <= 0:
                raise ValueError(f"empty/nonfinite named group: {name}")
            w = w / w.sum()
            return w @ neutral, np.einsum("v,cva->ca", w, deltas)

        left, d_left = centroid("lLipCorner")
        right, d_right = centroid("rLipCorner")
        upper, d_upper = centroid("LipUpperMiddle")
        lower, d_lower = centroid("LipLowerMiddle")
        sign = np.sign(left[0] - right[0]) or 1.0
        offset = np.asarray([
            sign * (left[0] - right[0]),
            (left[2] + right[2]) / 2 - (upper[2] + lower[2]) / 2,
            upper[2] - lower[2],
            (left[1] + right[1]) / 2,
        ], dtype=np.float32)
        jacobian = np.stack([
            sign * (d_left[:, 0] - d_right[:, 0]),
            (d_left[:, 2] + d_right[:, 2]) / 2 - (d_upper[:, 2] + d_lower[:, 2]) / 2,
            d_upper[:, 2] - d_lower[:, 2],
            (d_left[:, 1] + d_right[:, 1]) / 2,
        ]).astype(np.float32)
        if not np.isfinite(jacobian).all() or not np.isfinite(offset).all():
            raise ValueError("nonfinite geometry observer")
        if np.linalg.matrix_rank(jacobian, tol=1e-7) < 4:
            raise ValueError("geometry observer is rank deficient")
        fitted_scale = np.ones(4, dtype=np.float32) if scale is None else np.asarray(scale, dtype=np.float32)
        if fitted_scale.shape != (4,) or not np.isfinite(fitted_scale).all() or (fitted_scale <= 0).any():
            raise ValueError("geometry scale must be finite positive [4]")
        metadata = {
            "schema": "kinetalk_v4_fixed_rig_geometry_v1",
            "asset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "provisional": bool(np.asarray(z["provisional"]).item()) if "provisional" in z.files else True,
            "physiological_contact_claim": False,
            "scale_is_train_fitted": scale is not None,
            "observers": ["corner_width_x_mm", "corner_lift_z_mm",
                          "middle_aperture_z_mm", "corner_depth_y_mm"],
        }
    return (torch.from_numpy(jacobian), torch.from_numpy(offset),
            torch.from_numpy(fitted_scale), channels, metadata)


__all__ = ["load_rig_geometry"]
