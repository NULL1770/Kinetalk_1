"""Event-schedule conditional flow initialized from an existing motion prior.

Conditions describe four active flags, four within-event phases and four
durations. They contain no measured coefficient amplitudes or query baseline.
The source prior is not modified; this is a separately trained conditional copy.
"""
from __future__ import annotations
import copy
import torch
from torch import nn

from .continuous_upper_motion import ContinuousLatentFlow


class EventConditionedFlow(ContinuousLatentFlow):
    def __init__(self, context_dim, latent_dim=16, hidden=96, depth=4, block_size=5):
        super().__init__(context_dim=context_dim, audio_dim=12, latent_dim=latent_dim,
                         hidden=hidden, depth=depth, block_size=block_size)

    @classmethod
    def from_prior(cls, prior):
        if not isinstance(prior, ContinuousLatentFlow):
            raise TypeError('Expected a trained ContinuousLatentFlow')
        args = {k: v for k, v in prior.config.items() if k != 'audio_dim'}
        model = cls(**args).to(next(prior.parameters()).device)
        state = copy.deepcopy(prior.state_dict())
        for key in list(state):
            if key.startswith(('audio_frame_projection.', 'audio_block_projection.')):
                del state[key]
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith(('audio_frame_projection.', 'audio_block_projection.'))
                                         for k in result.missing_keys):
            raise ValueError('Unexpected source-prior architecture mismatch')
        nn.init.zeros_(model.audio_block_projection.weight)
        nn.init.zeros_(model.audio_block_projection.bias)
        return model


def shift_schedule(schedule, valid, frames):
    """Translate events separately in acoustic runs, zero fill, never wrap."""
    from scripts.evaluate_continuous_motion_latent import contiguous_runs
    import numpy as np
    x = np.asarray(schedule)
    mask = np.asarray(valid)
    if x.shape != (len(mask), 12) or mask.dtype != bool or type(frames) is not int:
        raise ValueError('schedule[T,12], boolean valid[T], integer frames required')
    if not np.isfinite(x).all():
        raise ValueError('Finite schedule required')
    result = np.zeros_like(x)
    for left, right in contiguous_runs(mask):
        src_start, src_stop = max(left, left-frames), min(right, right-frames)
        if src_stop > src_start:
            result[src_start+frames:src_stop+frames] = x[src_start:src_stop]
    return result


__all__ = ['EventConditionedFlow', 'shift_schedule']
