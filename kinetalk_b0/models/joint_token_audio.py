"""Bounded acoustic corrections to a separately fitted joint-motion prior.

The driver owns fit-only normalization and the base-prior logits. This module
does not estimate either from query targets. Zero initialization makes adding
its logits an exact no-op before training; tanh limits each learned correction
to [-1, 1] regardless of the supplied acoustic/history feature magnitude.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import torch
from torch import nn


class BoundedTokenResidual(nn.Module):
    """Two linear layers with SiLU and a bounded, zero-initialized output.

    ``input_dim`` is explicit because the driver constructs its feature vector.
    The current recipe uses65 global/intensity values, four64D acoustic bins
    and eight9D history frames, giving393 inputs. A supplied history must be
    strictly past ground truth for teacher-forced training or generated motion
    at deployment; no history interpretation or normalization occurs here.
    """

    def __init__(self, input_dim, hidden=64, k=128):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (input_dim, hidden, k)):
            raise ValueError('Positive integer input, hidden and token dimensions required')
        self.input_dim, self.hidden, self.k = input_dim, hidden, k
        self.input = nn.Linear(input_dim, hidden)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden, k)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        if (not torch.is_tensor(x) or not x.is_floating_point() or x.ndim < 1
                or x.shape[-1] != self.input_dim or x.numel() == 0
                or x.device != self.input.weight.device or x.dtype != self.input.weight.dtype
                or not torch.isfinite(x).all()):
            raise ValueError('Finite acoustic/history inputs must match model width, dtype and device')
        return torch.tanh(self.output(self.activation(self.input(x))))


def unit_audio_descriptor(local, valid, start, duration=32):
    """Flatten four fixed native-clock bins of local audio to a256D vector.

    Inputs are either both NumPy arrays or both torch tensors, with shapes
    ``local[T,64]`` and Boolean ``valid[T]``. The returned vector preserves the
    floating dtype, and for torch preserves the device and autograd path.

    Bin boundaries are ``start + floor(i * duration / 4)`` for i=0..4. Each bin
    averages only its observed native frames. Missing positions are never
    interpolated, replicated or removed from the time axis. A bin with no
    observations, including a bin wholly beyond the clip tail, returns zeros.
    Thus a short clip does not stretch its remaining audio over the32-frame
    horizon. A partially observed bin averages its valid frames without using
    any value at an internal gap. Future *audio* is allowed; motion, labels and
    teacher event boundaries are not inputs to this function.
    """
    if (isinstance(start, bool) or not isinstance(start, Integral) or start < 0
            or isinstance(duration, bool) or not isinstance(duration, Integral) or duration < 4):
        raise ValueError('Nonnegative integer start and integer duration >=4 required')
    start, duration = int(start), int(duration)
    if torch.is_tensor(local):
        if (local.ndim != 2 or local.shape[1] != 64 or len(local) < 1
                or not local.is_floating_point() or not torch.is_tensor(valid)
                or valid.dtype != torch.bool or valid.shape != local.shape[:1]
                or valid.device != local.device or not torch.isfinite(local[valid]).all()):
            raise ValueError('Finite observed local[T,64] and matching Boolean torch mask required')
        bins = []
        for index in range(4):
            left = min(start+index*duration//4, len(local))
            right = min(start+(index+1)*duration//4, len(local))
            mask = valid[left:right]
            # where occurs before reduction; invalid NaNs never enter the mean.
            clean = torch.where(mask[:, None], local[left:right], 0.)
            bins.append(clean.sum(0)/mask.sum().clamp_min(1))
        return torch.cat(bins)
    if (not isinstance(local, np.ndarray) or local.ndim != 2 or local.shape[1] != 64
            or len(local) < 1 or not np.issubdtype(local.dtype, np.floating)
            or not isinstance(valid, np.ndarray) or valid.dtype != np.bool_
            or valid.shape != local.shape[:1] or not np.isfinite(local[valid]).all()):
        raise ValueError('Finite observed local[T,64] and matching Boolean NumPy mask required')
    bins = []
    for index in range(4):
        left = min(start+index*duration//4, len(local))
        right = min(start+(index+1)*duration//4, len(local))
        mask = valid[left:right]
        clean = np.where(mask[:, None], local[left:right], 0.)
        bins.append(clean.sum(0)/max(int(mask.sum()), 1))
    return np.concatenate(bins).astype(local.dtype, copy=False)
