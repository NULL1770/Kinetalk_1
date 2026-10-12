"""Unified three-contribution KineTalk v2 architecture.

The package is intentionally isolated from ``kinetalk_b0`` so Phase53 remains
an immutable baseline.  See ``docs/architecture_upgrade_v3_20261010`` for the
training and data contracts.
"""
from .unified_model import UnifiedConfig, UnifiedExpressionModel, teacher_targets

__all__ = ["UnifiedConfig", "UnifiedExpressionModel", "teacher_targets"]
