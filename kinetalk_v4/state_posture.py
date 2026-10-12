"""Supported sustained-state supervision for a bounded audio-posture pilot.

These coordinates describe observable residual motion, not psychological affect.
Teacher advantage is checked against TRAIN targets only; inference has no oracle
or reliability gate. Local stochastic events keep their existing likelihoods.
"""
from torch import nn
import torch


class AudioPostureAdapter(nn.Module):
    """Zero-start correction from pooled audio features; no label lookup."""
    def __init__(self, hidden, state_dim):
        super().__init__()
        self.output = nn.Linear(hidden, state_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pooled):
        return self.output(pooled)


def load_zero_posture_extension(model, source):
    """Allow only the explicitly declared zero-start checkpoint extension."""
    current = model.state_dict()
    missing = set(current) - set(source)
    expected = {'student.posture_adapter.output.weight',
                'student.posture_adapter.output.bias'}
    if set(source) - set(current) or missing not in (set(), expected):
        raise ValueError('Checkpoint mismatch exceeds the zero posture extension')
    for name in missing:
        if torch.count_nonzero(current[name]):
            raise ValueError('New posture head must start exactly at zero')
    result = model.load_state_dict(source, strict=not missing)
    if set(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError('Unexpected checkpoint keys during posture extension')
    return bool(missing)


def _moments(prediction, observed):
    safe = torch.where(observed, prediction, 0.)
    if not torch.isfinite(safe).all():
        raise ValueError('Nonfinite observed posture coordinates')
    count = observed.sum(1)
    return safe.sum(1) / count.clamp_min(1), count


def posture_supervision(student, teacher, target, valid, quality):
    """Direct posture and detached teacher guidance where teacher is better.

Losses average supported clip/coordinate pairs, giving long and short clips
equal weight. At least two observed frames are needed. Teacher guide coverage
is explicit; a worse teacher is never added as a privileged posture target.
"""
    if student.shape != target.shape or teacher.shape != target.shape:
        raise ValueError('Aligned student, teacher and target state required')
    if student.ndim != 3 or valid.shape != student.shape[:2] or quality.shape != student.shape:
        raise ValueError('Native state, frame validity and coordinate support required')
    observed = valid[..., None].bool() & quality.bool()
    s, count = _moments(student, observed)
    t, _ = _moments(teacher.detach(), observed)
    q, _ = _moments(target.detach(), observed)
    supported = count.ge(2)
    direct_error = (s - q).square()
    teacher_error = (t - q).square()
    advantage = supported & (teacher_error < direct_error.detach())
    denominator = supported.sum().clamp_min(1)
    direct = torch.where(supported, direct_error, 0.).sum() / denominator
    # Normalize by all supported axes so a tiny accepted subset does not receive
    # the same total influence as a teacher that is better on every coordinate.
    guidance = torch.where(advantage, (s - t).square(), 0.).sum() / denominator
    return dict(posture_direct=direct, posture_guidance=guidance,
                teacher_advantage_fraction=advantage.sum().to(s.dtype) / denominator,
                supported_posture_axes=supported.sum(), teacher_accepted_axes=advantage.sum())
