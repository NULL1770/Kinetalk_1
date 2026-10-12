"""Observed fixed-rig coordinates with an asset-level numerical tolerance.

The fixed .0001 mm threshold removes floating-point displacement residues, not
data-selected anatomical channels. Each observer still uses all substantive
blendshape dependencies. The unchanged raw Jacobian remains in prior assets.
"""
import torch

RIG_NUMERICAL_TOLERANCE_MM = 1e-4


def clean_geometry_jacobian(jacobian):
    if not torch.isfinite(jacobian).all():
        raise ValueError('Nonfinite geometry Jacobian')
    return torch.where(jacobian.abs() < RIG_NUMERICAL_TOLERANCE_MM,
                       torch.zeros_like(jacobian), jacobian)


def geometry_support(jacobian, channels):
    if jacobian.ndim != 2 or channels.ndim != 2 or jacobian.shape[1] != channels.shape[1]:
        raise ValueError('Expected [G,C] Jacobian and [B,C] channel support')
    dependencies = jacobian.abs().ge(RIG_NUMERICAL_TOLERANCE_MM)
    return (channels[:, None].bool() | ~dependencies[None]).all(-1)
