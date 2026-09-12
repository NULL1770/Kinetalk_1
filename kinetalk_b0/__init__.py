"""KineTalk b0 + residual-DiT training package.

All public motion tensors in this package are raw 52-D blendshape
coefficients.  Internal normalization layers are optimizer aids only; they do
not change the target coordinate system.
"""

from .data import B0ResidualDataset, collate_b0_residual

__all__ = ["B0ResidualDataset", "collate_b0_residual"]
