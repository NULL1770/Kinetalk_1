"""Chunk flow conditioned on an explicit, strictly preceding motion window.

Teacher motion history is a training/oracle condition only. Deployable callers
must pass generated previous motion, in the same normalization as the current
target/noise. This module never extracts history from the target or a full
motion sequence. The caller owns chronological slicing and scheduled-sampling
stop-gradients; a tensor alone cannot establish that its frames precede a chunk.
"""
from __future__ import annotations

import torch
from torch import nn

from .temporal_upper import TemporalUpperFlow, _clean_sequence


class HistoryUpperFlow(TemporalUpperFlow):
    """Add a learned previous-motion code to every current local audio token.

    History [B,H,9] is chronological, right-aligned immediately before the
    current chunk, with H <= history_frames <= 8. A Boolean history_valid
    mask permits missing observations and empty history. Left padding is
    allowed. Appending right padding changes the relative native-frame ages
    and therefore describes a different history window.

    The MLP observes a frame's nine values and its relative age. A masked GRU
    preserves order; missing frames never update its state. The final state is
    projected to local condition space and added at every current valid frame.
    Empty history contributes exactly zero. The use_history=False ablation
    owns identical parameters but ignores both supplied history arguments.

    No chunk centering, clamp, output residual scaling or implicit history
    detach is applied. Current chunk lengths such as 12/16 are caller-owned.
    Audio/global context may be noncausal; only the motion-history input is
    restricted to past frames. Use the *_history methods for this condition.
    """

    def __init__(self, cfg=None, *, use_history=True, history_frames=8, history_hidden=64):
        if type(use_history) is not bool:
            raise ValueError('use_history must be an explicit Boolean')
        if type(history_frames) is not int or not 1 <= history_frames <= 8:
            raise ValueError('history_frames must be an integer in [1,8]')
        if type(history_hidden) is not int or history_hidden < 1:
            raise ValueError('history_hidden must be a positive integer')
        super().__init__(cfg, use_state=False)
        self.use_history = use_history
        self.history_frames = history_frames
        self.history_hidden = history_hidden
        # Instantiate in every arm, without branching on use_history, so seed
        # and parameter shapes/values match before training.
        self.history_input = nn.Sequential(
            nn.Linear(10, history_hidden), nn.SiLU(),
            nn.Linear(history_hidden, history_hidden), nn.SiLU())
        self.history_gru = nn.GRUCell(history_hidden, history_hidden)
        self.history_projection = nn.Linear(history_hidden, self.emotion_dim, bias=False)
        nn.init.normal_(self.history_projection.weight, std=.02)

    def _history_code(self, history, history_valid, reference):
        batch = len(reference)
        zero = reference.new_zeros(batch, self.emotion_dim)
        if not self.use_history:
            return zero
        if history is None and history_valid is None:
            return zero
        if (not torch.is_tensor(history) or history.ndim != 3
                or history.shape[0] != batch or history.shape[-1] != 9
                or history.shape[1] > self.history_frames
                or history.dtype != reference.dtype or history.device != reference.device):
            raise ValueError('history must match current dtype/device and have shape [B,H,9], H <= 8 and history_frames')
        if (not torch.is_tensor(history_valid) or history_valid.dtype != torch.bool
                or history_valid.shape != history.shape[:2] or history_valid.device != reference.device):
            raise ValueError('history_valid must be a matching Boolean [B,H] mask')
        if not torch.isfinite(history[history_valid]).all():
            raise ValueError('Observed history must be finite')
        if history.shape[1] == 0:
            return zero
        # Clear NaN padding before any trainable arithmetic, not afterwards.
        clean = torch.where(history_valid[..., None], history, 0.)
        age = torch.arange(-history.shape[1], 0, dtype=reference.dtype, device=reference.device)
        age = (age / self.history_frames)[None, :, None].expand(batch, -1, -1)
        frame_input = torch.cat((clean, age), -1)
        hidden = reference.new_zeros(batch, self.history_hidden)
        for frame in range(history.shape[1]):
            # Keep each MLP's batch shape unchanged when left padding is added.
            embedded = self.history_input(frame_input[:, frame])
            proposal = self.history_gru(embedded, hidden)
            hidden = torch.where(history_valid[:, frame, None], proposal, hidden)
        code = self.history_projection(hidden)
        return torch.where(history_valid.any(1)[:, None], code, 0.)

    def prepare_history_conditions(self, valid, base_h0, identity_code, global_affect,
                                   local, *, history=None, history_valid=None):
        """One shared conditioning path for training and all solver steps."""
        conditions = super().prepare_conditions(valid, base_h0, identity_code, global_affect, local, None)
        if self.use_history:
            code = self._history_code(history, history_valid, conditions['local'])
            conditions['local'] = torch.where(valid[..., None], conditions['local'] + code[:, None], 0.)
        return conditions

    def flow_loss_history(self, target_norm9, valid, base_h0, identity_code, global_affect,
                          local, noise, time, *, history=None, history_valid=None):
        """Conditional flow matching; target is never a history condition.

        Teacher/generated/scheduled history selection is explicit in the
        caller. In particular, this method neither slices target_norm9 nor
        silently substitutes teacher motion when history is unavailable.
        """
        conditions = self.prepare_history_conditions(valid, base_h0, identity_code, global_affect, local,
                                                     history=history, history_valid=history_valid)
        target = _clean_sequence(target_norm9, valid, 9, 'target_norm9')
        start = _clean_sequence(noise, valid, 9, 'noise')
        if (not torch.is_tensor(time) or time.shape != (len(valid),)
                or not time.is_floating_point() or time.device != valid.device
                or not torch.isfinite(time).all() or ((time < 0.) | (time > 1.)).any()):
            raise ValueError('Flow time must be finite [B] within [0,1]')
        interpolation = time[:, None, None]
        x = (1. - interpolation) * start + interpolation * target
        predicted = self.velocity(x, time, conditions)
        error = torch.where(valid[..., None], predicted - (target - start), 0.)
        return error.square().sum() / (valid.sum() * 9)

    def decode_history(self, valid, base_h0, identity_code, global_affect, local, noise,
                       *, history=None, history_valid=None, steps=12):
        """Decode a chunk from noise and explicit previous generated motion.

        Teacher history passed here is an oracle diagnostic, never a deployable
        result. This function has no target argument or stored target state.
        """
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        conditions = self.prepare_history_conditions(valid, base_h0, identity_code, global_affect, local,
                                                     history=history, history_valid=history_valid)
        x = _clean_sequence(noise, valid, 9, 'noise')
        for index in range(steps):
            time = x.new_full((len(x),), index / steps)
            x = torch.where(valid[..., None], x + self.velocity(x, time, conditions) / steps, 0.)
        return x


__all__ = ['HistoryUpperFlow']
