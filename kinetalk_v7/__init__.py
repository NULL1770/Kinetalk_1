"""Proposal-aligned v7 disentanglement prototype.

The v5 checkpoint and source remain the production baseline.  This package is
an isolated, fast-to-diagnose candidate for population expression generation,
conditional speaker response, and visual-teacher/audio-student affect paths.
"""

from .disentangled_model import (
    V7Config,
    V7DisentangledModel,
    v7_loss,
)

__all__ = ["V7Config", "V7DisentangledModel", "v7_loss"]
