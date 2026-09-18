"""Small, separate motion-teacher and acoustic-only activity-probe utilities.

Teacher labels describe four groups' raw-coefficient velocity over a fixed
eight-transition core. They do not encode action direction or supply motion
to either prediction head. The caller owns fitting allowlists and normalization
statistics; every query uses the same saved activity thresholds.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FPS, HORIZON, HOP = 25, 32, 8
ACTIVITY_QUANTILE, THRESHOLD_FLOOR = .65, 1e-6
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
GROUP_NAMES = ('brow_up', 'brow_down', 'eye_squint', 'eye_wide')
PROSODY = slice(1536, 1540)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _valid_runs(valid):
    mask = _numpy(valid)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError('Expected a one-dimensional Boolean validity mask')
    boundaries = np.diff(np.r_[False, mask, False].astype(np.int8))
    return mask, list(zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)))


def window_locations(valid):
    """Full H32/hop8 windows per observed run, with one end-anchored last.

No window crosses a missing frame, stretches a short run, or compresses the
original clock. Runs shorter than 32 contribute no training/scoring windows.
    """
    _, runs = _valid_runs(valid)
    locations = []
    for left, right in runs:
        if right-left < HORIZON:
            continue
        starts = list(range(int(left), int(right)-HORIZON+1, HOP))
        if starts[-1] != right-HORIZON:
            starts.append(int(right)-HORIZON)
        locations.extend((start, HORIZON) for start in starts)
    return locations


def window_coverage(valid):
    """Count acoustic-context and target-transition coverage without motion."""
    mask, runs = _valid_runs(valid)
    locations = window_locations(mask)
    context = np.zeros(len(mask), dtype=bool)
    transitions = np.zeros(max(len(mask)-1, 0), dtype=bool)
    for start, count in locations:
        context[start:start+count] = True
        transitions[start+11:start+19] = True
    valid_count = int(mask.sum())
    transition_count = int((mask[:-1] & mask[1:]).sum())
    short = [(int(left), int(right)) for left, right in runs if right-left < HORIZON]
    return {'windows': len(locations), 'valid_frames': valid_count,
        'context_covered_frames': int(context.sum()),
        'context_fraction': float(context.sum()/valid_count) if valid_count else 0.,
        'valid_transitions': transition_count, 'target_covered_transitions': int(transitions.sum()),
        'target_transition_fraction': float(transitions.sum()/transition_count) if transition_count else 0.,
        'short_run_count': len(short), 'short_run_valid_frames': sum(right-left for left, right in short),
        'short_runs': short}


def activity_energy(upper32):
    """Return four raw-coefficient RMS speeds for a fully observed [32,9].

Three-point centered smoothing reads positions10..20 to form positions11..19;
eight differences cover the central transitions11->12 through18->19. Speed
is 25*sqrt(mean(delta**2)), averaging across those transitions and each group's
channels. No clip/window motion mean, scale or query threshold is fitted.
    """
    upper = np.asarray(_numpy(upper32), dtype=np.float64)
    if upper.shape != (HORIZON, 9) or not np.isfinite(upper).all():
        raise ValueError('A finite, fully observed raw upper-face window [32,9] is required')
    smoothed = (upper[10:19]+upper[11:20]+upper[12:21])/3.
    delta = np.diff(smoothed, axis=0)
    values = np.asarray([FPS*np.sqrt(np.square(delta[:, group]).mean()) for group in GROUPS])
    if not np.isfinite(values).all():
        raise FloatingPointError('Activity energy overflowed')
    return values


def fit_thresholds(energies, weights):
    """Fit per-group weighted65th percentiles with a fixed 1e-6 floor.

Caller supplies one positive weight per fitting window, normally the inverse
of its clip's window count. The inverse-CDF convention chooses the first
observed energy with cumulative weight >=.65; it does not interpolate labels.
Each clip therefore has equal total mass regardless of its recording length.
    """
    values = np.asarray(_numpy(energies), dtype=np.float64)
    mass = np.asarray(_numpy(weights), dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 4 or not len(values)
            or mass.shape != (len(values),) or not np.isfinite(values).all()
            or (values < 0).any() or not np.isfinite(mass).all() or (mass <= 0).any()):
        raise ValueError('Finite nonnegative energies[N,4] and positive fitting weights[N] required')
    mass = mass/mass.max()
    total = mass.sum()
    thresholds = []
    for group in range(4):
        order = np.argsort(values[:, group], kind='stable')
        index = min(int(np.searchsorted(np.cumsum(mass[order]), ACTIVITY_QUANTILE*total, side='left')), len(order)-1)
        thresholds.append(max(float(values[order[index], group]), THRESHOLD_FLOOR))
    return np.asarray(thresholds)


def activity_targets(energies, thresholds):
    """Binary targets using saved thresholds and strict greater-than ties."""
    values = np.asarray(_numpy(energies), dtype=np.float64)
    threshold = np.asarray(_numpy(thresholds), dtype=np.float64)
    if (values.ndim < 1 or values.shape[-1] != 4 or threshold.shape != (4,)
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.isfinite(threshold).all() or (threshold < THRESHOLD_FLOOR).any()):
        raise ValueError('Nonnegative energies[...,4] and saved positive thresholds[4] required')
    return values > threshold


def _prosody(features):
    if (not torch.is_tensor(features) or not features.is_floating_point()
            or features.ndim < 2 or features.shape[-1] not in (4, 1540)):
        raise ValueError('Floating acoustic features with exactly4 or1540 channels required')
    return features if features.shape[-1] == 4 else features[..., PROSODY]


def _clip_audio(features, valid):
    local = _prosody(features)
    if (local.ndim != 2 or not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != local.shape[:1] or valid.device != local.device
            or not valid.any() or not torch.isfinite(local[valid]).all()):
        raise ValueError('Finite observed clip prosody[T,4] and a nonempty matching Boolean mask required')
    return local


def prosody_mean(features, valid):
    """Original clip's valid acoustic mean; target motion is never read."""
    local = _clip_audio(features, valid)
    return torch.where(valid[:, None], local, 0.).sum(0)/valid.sum()


def intervene_prosody(features, valid, mode='real', *, donor_features=None, donor_valid=None):
    """Return a new [T,4] acoustic sequence with invalid positions zero.

Reverse and half-run circular shift operate separately in each original valid
run. Mismatch requires one continuous observed donor run, resamples it to each
recipient run, then shifts the result's whole-valid mean back to the original
recipient mean. It never mutates input features, masks or a global condition.
    """
    local = _clip_audio(features, valid)
    original_mean = prosody_mean(local, valid)
    result = torch.where(valid[:, None], local, 0.).clone()
    _, runs = _valid_runs(valid)
    if mode == 'static':
        return torch.where(valid[:, None], original_mean[None], 0.)
    if mode == 'real':
        return result
    if mode in ('reverse', 'shift'):
        for left, right in runs:
            section = local[left:right]
            result[left:right] = section.flip(0) if mode == 'reverse' else section.roll(int((right-left)//2), dims=0)
        return result
    if mode != 'mismatch':
        raise ValueError('Acoustic intervention must be real, static, reverse, shift or mismatch')
    donor = _clip_audio(donor_features, donor_valid)
    _, donor_runs = _valid_runs(donor_valid)
    if len(donor_runs) != 1 or donor.device != local.device or donor.dtype != local.dtype:
        raise ValueError('Mismatch donor requires one continuous valid run and matching dtype/device')
    source = donor[donor_valid]
    for left, right in runs:
        result[left:right] = F.interpolate(source.T[None], size=int(right-left), mode='linear', align_corners=True)[0].T
    result = torch.where(valid[:, None], result+(original_mean-prosody_mean(result, valid))[None], 0.)
    return result


def prosody_descriptor(windows, original_mean, feature_std):
    """Eight native4-frame means of centered/scaled prosody -> [B,32].

All32 frames must be observed; obtain windows from ``window_locations``. Mean
is the original recipient whole-clip acoustic mean, even under interventions.
Scale is a saved fitting-only four-channel standard deviation. No window mean
or motion-derived quantity is read, and channels0..1535 are never consumed.
    """
    local = _prosody(windows)
    if local.ndim != 3 or local.shape[0] < 1 or local.shape[1:] != (HORIZON, 4) or not torch.isfinite(local).all():
        raise ValueError('Finite fully observed acoustic windows[B,32,4|1540] required')
    if (not torch.is_tensor(original_mean) or original_mean.shape not in ((4,), (len(local), 4))
            or original_mean.device != local.device or original_mean.dtype != local.dtype
            or not torch.isfinite(original_mean).all()
            or not torch.is_tensor(feature_std) or feature_std.shape != (4,)
            or feature_std.device != local.device or feature_std.dtype != local.dtype
            or not torch.isfinite(feature_std).all() or (feature_std <= 0).any()):
        raise ValueError('Matching original acoustic mean[4|B,4] and positive saved feature_std[4] required')
    center = original_mean if original_mean.ndim == 2 else original_mean[None]
    normalized = (local-center[:, None])/feature_std
    result = normalized.reshape(len(local), 8, 4, 4).mean(2).reshape(len(local), 32)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Prosody descriptor overflowed')
    return result


def _model_input(value, width, reference, name):
    if (not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != width
            or value.dtype != reference.dtype or value.device != reference.device
            or not torch.isfinite(value).all()):
        raise ValueError(f'Finite {name}[B,{width}] matching model dtype/device required')


class StaticActivityHead(nn.Module):
    """Four activity logits from supplied normalized global65+prosody_mean4."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(69, 32), nn.SiLU(), nn.Linear(32, 4))

    def forward(self, static_context):
        _model_input(static_context, 69, self.network[0].weight, 'static context')
        return self.network(static_context)


class FrozenActivityResidual(nn.Module):
    """Learn an acoustic timing correction with an unchanged static baseline."""

    def __init__(self, base):
        super().__init__()
        if not isinstance(base, StaticActivityHead):
            raise ValueError('An independently trained StaticActivityHead base is required')
        self.base = base.eval().requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.correction = nn.Sequential(nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 4))
        self.correction.to(device=base.network[0].weight.device, dtype=base.network[0].weight.dtype)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval().requires_grad_(False)
        return self

    def residual_logits(self, descriptor):
        _model_input(descriptor, 32, self.correction[0].weight, 'prosody descriptor')
        return self.correction(descriptor).tanh()-self.correction(torch.zeros_like(descriptor)).tanh()

    def forward(self, static_context, descriptor):
        self.base.eval().requires_grad_(False)
        with torch.no_grad():
            baseline = self.base(static_context)
        delta = self.residual_logits(descriptor)
        if delta.shape != baseline.shape:
            raise ValueError('Static context and prosody descriptor batch sizes must match')
        result = baseline+delta
        if not torch.isfinite(result).all():
            raise FloatingPointError('Activity logits are nonfinite')
        return result
