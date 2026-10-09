"""Small receiving-side neutral calibration; no audio or expression input.

One fixed monotone map per mouth coefficient. Fit on neutral targets only;
this module does not modify B0 or automatically enter the expression pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
import torch
import torch.nn.functional as F

CHANNELS = tuple(range(14, 41))
JAW_CHANNEL = 17
CLOSURE_THRESHOLD = 0.05
KNOTS = np.linspace(0., 1., 9)
SCHEMA = 'neutral_monotone_amplitude_v3_fixed_endpoints'
REPLAY_SCHEMAS = (SCHEMA, 'neutral_monotone_amplitude_v2_event_preserving')


def basis(x):
    """Linear interpolation basis; identity residual extrapolation at tails."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Finite calibration inputs required')
    u = np.clip(x, 0., 1.) * (len(KNOTS)-1)
    left = np.minimum(u.astype(np.int64), len(KNOTS)-2)
    w = u-left
    eye = np.eye(len(KNOTS))
    return eye[left]*(1-w[..., None])+eye[left+1]*w[..., None]


def jaw_basis(x):
    """Basis for the open-jaw branch, whose origin is the closure threshold."""
    x = np.asarray(x, dtype=np.float64)
    z = (x-CLOSURE_THRESHOLD)/(1.-CLOSURE_THRESHOLD)
    return basis(z)


def observed(x, y, mask, times):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask, times = np.asarray(mask), np.asarray(times, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 52 or y.shape != x.shape
            or mask.shape != x.shape or mask.dtype != bool or times.shape != x.shape[:1]):
        raise ValueError('Require native [T,52] values/Boolean mask and [T] times')
    valid = mask.any(-1)
    if not np.isfinite(times[valid]).all() or np.any(np.diff(times[valid]) <= 0):
        raise ValueError('Finite increasing observed times required')
    if not np.isfinite(x[mask]).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Observed coordinates must be finite')
    # NaNs on excluded channels/frames must not enter a design matrix.
    x, y = np.where(mask, x, 0.), np.where(mask, y, 0.)
    adjacent = mask[1:] & mask[:-1] & np.isclose(
        np.diff(times), .04, rtol=1e-4, atol=1e-7)[:, None]
    return x, y, mask, adjacent


class NeutralAmplitudeFit:
    """Accumulate exact quadratic statistics, equal clip weight per channel."""
    def __init__(self):
        self.xx = np.zeros((2, len(CHANNELS), 9, 9))
        self.xy = np.zeros((2, len(CHANNELS), 9))
        self.count = np.zeros((2, len(CHANNELS)), dtype=np.int64)
        self.rows = []

    def add(self, x, y, mask, times, *, provenance):
        required = ('clip_id', 'sentence_id', 'speaker', 'role', 'target_kind', 'target_sha256')
        if not isinstance(provenance, dict) or any(not provenance.get(k) for k in required):
            raise ValueError('Explicit fit/target provenance required')
        if (provenance['role'] != 'train' or provenance['target_kind'] not in
                ('native_neutral', 'approved_neutral_pair')):
            raise ValueError('Only internal TRAIN neutral targets may enter fit')
        if provenance['clip_id'] in {r['clip_id'] for r in self.rows}:
            raise ValueError('Duplicate clip would alter fit weights')
        x, y, mask, adjacent = observed(x, y, mask, times)
        a, residual = basis(x[:, CHANNELS]), (y-x)[:, CHANNELS]
        # The jaw is fitted only on its open branch.  Its closed branch is
        # deliberately left out so a receiving calibration cannot create or
        # remove a closure event.  The open-branch map is anchored at 0.05.
        jaw = CHANNELS.index(JAW_CHANNEL)
        open_frame = mask[:, JAW_CHANNEL] & (x[:, JAW_CHANNEL] >= CLOSURE_THRESHOLD)
        a[:, jaw] = 0.
        a[open_frame, jaw] = jaw_basis(x[open_frame, JAW_CHANNEL])
        for view, (design, target, support) in enumerate((
            (a, residual, mask[:, CHANNELS]),
            (np.diff(a, axis=0), np.diff(residual, axis=0), adjacent[:, CHANNELS]))):
            if view == 0:
                support = support.copy()
                support[:, jaw] &= open_frame
            else:
                support = support.copy()
                support[:, jaw] &= open_frame[1:] & open_frame[:-1]
            n = support.sum(0)
            w = support / np.maximum(n, 1)
            self.xx[view] += np.einsum('tck,tcl,tc->ckl', design, design, w)
            self.xy[view] += np.einsum('tck,tc,tc->ck', design, target, w)
            self.count[view] += n > 0
        self.rows.append(dict(provenance))

    def solve(self):
        if len(self.rows) < 2 or np.any(self.count[0] < 2):
            raise ValueError('Every calibrated channel requires two observed fit clips')
        gram = self.xx / np.maximum(self.count[..., None, None], 1)
        cross = self.xy / np.maximum(self.count[..., None], 1)
        gram, cross = gram[0]+.5*gram[1], cross[0]+.5*cross[1]
        ridge = .001/len(KNOTS)
        difference = np.diff(np.eye(9), axis=0)
        # Optimize knot residuals d. Monotonicity applies to KNOTS+d.
        residuals, receipts = [], []
        for c in range(len(CHANNELS)):
            a, b = gram[c]+ridge*np.eye(9), cross[c]
            baseline = KNOTS
            if CHANNELS[c] == JAW_CHANNEL:
                baseline = CLOSURE_THRESHOLD + (1.-CLOSURE_THRESHOLD)*KNOTS
            fit_lower, fit_upper = -baseline.copy(), 1.-baseline.copy()
            # Fix endpoint residuals *inside* the solve, for every channel.
            # A projection after solving would change the deployed objective.
            fit_lower[[0, -1]] = fit_upper[[0, -1]] = 0.
            fit = minimize(lambda d: float(d@a@d-2*d@b), np.zeros(9),
                jac=lambda d: 2*(a@d-b), method='SLSQP',
                constraints=LinearConstraint(difference, -np.diff(baseline), np.inf),
                bounds=Bounds(fit_lower, fit_upper),
                options={'ftol': 1e-12, 'maxiter': 200})
            if not fit.success or np.diff(baseline+fit.x).min() < -1e-8:
                raise RuntimeError('Constrained fit failed: '+fit.message)
            # Remove only numerical constraint error; endpoints remain exact.
            values = np.maximum.accumulate(np.clip(baseline+fit.x, 0., 1.))
            values[0], values[-1] = baseline[0], baseline[-1]
            deployed = values-baseline
            if np.max(np.abs(deployed-fit.x)) > 1e-8:
                raise RuntimeError('Deployed map differs from constrained solution')
            residuals.append(deployed)
            receipts.append(dict(iterations=int(fit.nit), objective_change=float(fit.fun),
                deployed_objective_change=float(deployed@a@deployed-2*deployed@b),
                endpoint_residual_max_abs=float(np.max(np.abs(fit.x[[0,-1]])))))
        return dict(schema=SCHEMA, channels=list(CHANNELS), knots=KNOTS.tolist(),
            residuals=np.stack(residuals).tolist(), ridge=.001, displacement_weight=.5,
            count=self.count.tolist(), fit_rows=self.rows, solver=receipts,
            development_used_for_fit=False, test_used_for_fit=False,
            input='frozen_neutral_B0_only', endpoints='identity', tails='identity',
            jaw_channel=JAW_CHANNEL, closure_threshold=CLOSURE_THRESHOLD,
            jaw_closed_branch='identity')


def apply_calibration(x, fit):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating) or x.ndim < 2 or x.shape[-1] != 52:
        raise ValueError('Floating [...,52] B0 required')
    if (fit.get('schema') not in REPLAY_SCHEMAS or fit.get('channels') != list(CHANNELS)
            or fit.get('knots') != KNOTS.tolist()
            or fit.get('development_used_for_fit') is not False
            or fit.get('test_used_for_fit') is not False
            or fit.get('input') != 'frozen_neutral_B0_only'
            or fit.get('jaw_channel') != JAW_CHANNEL
            or fit.get('closure_threshold') != CLOSURE_THRESHOLD
            or fit.get('jaw_closed_branch') != 'identity'):
        raise ValueError('Expected bound TRAIN neutral-only candidate')
    residuals = np.asarray(fit['residuals'], dtype=np.float64)
    if (residuals.shape != (27, 9) or not np.isfinite(residuals).all()
            or np.any(residuals[:, [0, -1]] != 0)):
        raise ValueError('Invalid monotone calibration')
    jaw = CHANNELS.index(JAW_CHANNEL)
    jaw_baseline = CLOSURE_THRESHOLD + (1.-CLOSURE_THRESHOLD)*KNOTS
    base = np.broadcast_to(KNOTS, residuals.shape).copy()
    base[jaw] = jaw_baseline
    if np.min(np.diff(base+residuals, axis=-1)) < -1e-10:
        raise ValueError('Invalid monotone calibration')
    if np.min(np.diff(jaw_baseline+residuals[jaw], axis=-1)) < -1e-10:
        raise ValueError('Invalid event-preserving jaw calibration')
    out = x.copy()
    design = basis(x[..., CHANNELS])
    design[..., jaw, :] = 0.
    open_frame = x[..., JAW_CHANNEL] >= CLOSURE_THRESHOLD
    design[..., jaw, :] = np.where(
        open_frame[..., None], jaw_basis(x[..., JAW_CHANNEL]), 0.)
    out[..., CHANNELS] = x[..., CHANNELS] + np.einsum(
        '...ck,ck->...c', design, residuals)
    return out


def apply_calibration_tensor(x, fit):
    """Torch equivalent used by the frozen receiver evaluation path."""
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim < 2 or x.shape[-1] != 52:
        raise ValueError('Floating torch [...,52] B0 required')
    apply_calibration(np.zeros((1, 52), dtype=np.float64), fit)
    residuals = torch.as_tensor(fit['residuals'], dtype=x.dtype, device=x.device)
    values = x[..., CHANNELS].clamp(0., 1.)
    u = values * (len(KNOTS)-1)
    left = u.floor().long().clamp(max=len(KNOTS)-2)
    w = u - left.to(x.dtype)
    design = F.one_hot(left, len(KNOTS)).to(x.dtype) * (1.-w[..., None])
    design = design + F.one_hot(left+1, len(KNOTS)).to(x.dtype) * w[..., None]
    jaw = CHANNELS.index(JAW_CHANNEL)
    design[..., jaw, :] = 0.
    z = ((x[..., JAW_CHANNEL]-CLOSURE_THRESHOLD)/(1.-CLOSURE_THRESHOLD)).clamp(0., 1.)
    ju = z * (len(KNOTS)-1)
    jleft = ju.floor().long().clamp(max=len(KNOTS)-2)
    jw = ju-jleft.to(x.dtype)
    jdesign = F.one_hot(jleft, len(KNOTS)).to(x.dtype) * (1.-jw[..., None])
    jdesign = jdesign + F.one_hot(jleft+1, len(KNOTS)).to(x.dtype) * jw[..., None]
    design[..., jaw, :] = torch.where(
        (x[..., JAW_CHANNEL] >= CLOSURE_THRESHOLD)[..., None], jdesign, torch.zeros_like(jdesign))
    out = x.clone()
    out[..., CHANNELS] = x[..., CHANNELS] + torch.einsum(
        '...ck,ck->...c', design, residuals)
    return out


def measures(x, y, mask, times):
    x, y, mask, adjacent = observed(x, y, mask, times)
    def channel_mean(value, support):
        n = support.sum(0); active = n > 0
        return float((np.where(support, value, 0.).sum(0)[active]/n[active]).mean()) if active.any() else None
    delta = x-y
    result = dict(mouth_mse=channel_mean(delta[:, CHANNELS]**2, mask[:, CHANNELS]),
        displacement_mse=channel_mean(np.diff(delta[:, CHANNELS], axis=0)**2, adjacent[:, CHANNELS]))
    a, b = x[mask[:, 17], 17], y[mask[:, 17], 17]
    for name in ('jaw_range', 'target_jaw_range', 'jaw_range_absolute_error', 'jaw_corr', 'closure_f1'):
        result[name] = None
    if len(a) >= 2:
        ar, br = np.diff(np.quantile(a, [.1, .9])).item(), np.diff(np.quantile(b, [.1, .9])).item()
        ac, bc = a-a.mean(), b-b.mean()
        denom = np.linalg.norm(ac)*np.linalg.norm(bc)
        result.update(jaw_range=ar, target_jaw_range=br, jaw_range_absolute_error=abs(ar-br),
            jaw_corr=float(ac@bc/denom) if denom > 1e-12 else None,
            closure_f1=float(2*np.sum((a < .05)&(b < .05))/max(1, np.sum(a < .05)+np.sum(b < .05))))
    return result


def held_gate(groups):
    failures = []
    for role in ('speaker_dev', 'sentence_dev'):
        for kind in ('native_neutral', 'approved_neutral_pair'):
            for view in ('raw', 'clip'):
                key = '/'.join((role, kind, view))
                part = groups.get(key, {})
                b, c = part.get('base', {}), part.get('candidate', {})
                required = ('mouth_mse', 'displacement_mse', 'jaw_range_absolute_error', 'jaw_corr', 'closure_f1')
                if not part.get('clips') or any(v.get(k) is None or not np.isfinite(v[k]) for v in (b, c) for k in required):
                    failures.append(key+':missing'); continue
                checks = dict(position=c['mouth_mse'] <= .99*b['mouth_mse'],
                    range=c['jaw_range_absolute_error'] < b['jaw_range_absolute_error'],
                    correlation=c['jaw_corr'] >= b['jaw_corr']-.005,
                    closure=c['closure_f1'] >= b['closure_f1']-.01,
                    displacement=c['displacement_mse'] <= 1.01*b['displacement_mse'])
                failures.extend(key+':'+k for k, passed in checks.items() if not passed)
    return dict(passed=not failures, failures=failures, main_model_replaced=False)
