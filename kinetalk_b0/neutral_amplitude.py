"""Small receiving-side neutral calibration; no audio or expression input.

One fixed monotone map per mouth coefficient. Fit on neutral targets only;
this module does not modify B0 or automatically enter the expression pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

CHANNELS = tuple(range(14, 41))
KNOTS = np.linspace(0., 1., 9)
SCHEMA = 'neutral_monotone_amplitude_v1'


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
        for view, (design, target, support) in enumerate((
            (a, residual, mask[:, CHANNELS]),
            (np.diff(a, axis=0), np.diff(residual, axis=0), adjacent[:, CHANNELS]))):
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
        constraint = LinearConstraint(difference, -np.diff(KNOTS), np.inf)
        lower, upper = -KNOTS.copy(), 1-KNOTS
        lower[[0, -1]] = 0.; upper[[0, -1]] = 0.
        residuals, receipts = [], []
        for c in range(len(CHANNELS)):
            a, b = gram[c]+ridge*np.eye(9), cross[c]
            fit = minimize(lambda d: float(d@a@d-2*d@b), np.zeros(9),
                jac=lambda d: 2*(a@d-b), method='SLSQP',
                constraints=constraint, bounds=Bounds(lower, upper),
                options={'ftol': 1e-12, 'maxiter': 200})
            if not fit.success or np.diff(KNOTS+fit.x).min() < -1e-8:
                raise RuntimeError('Constrained fit failed: '+fit.message)
            # Remove only numerical constraint error; endpoints remain exact.
            values = np.maximum.accumulate(np.clip(KNOTS+fit.x, 0., 1.))
            values[0], values[-1] = 0., 1.
            residuals.append(values-KNOTS)
            receipts.append(dict(iterations=int(fit.nit), objective_change=float(fit.fun)))
        return dict(schema=SCHEMA, channels=list(CHANNELS), knots=KNOTS.tolist(),
            residuals=np.stack(residuals).tolist(), ridge=.001, displacement_weight=.5,
            count=self.count.tolist(), fit_rows=self.rows, solver=receipts,
            development_used_for_fit=False, test_used_for_fit=False,
            input='frozen_neutral_B0_only', endpoints='identity', tails='identity')


def apply_calibration(x, fit):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating) or x.ndim < 2 or x.shape[-1] != 52:
        raise ValueError('Floating [...,52] B0 required')
    if (fit.get('schema') != SCHEMA or fit.get('channels') != list(CHANNELS)
            or fit.get('knots') != KNOTS.tolist()
            or fit.get('development_used_for_fit') is not False
            or fit.get('test_used_for_fit') is not False
            or fit.get('input') != 'frozen_neutral_B0_only'):
        raise ValueError('Expected bound TRAIN neutral-only candidate')
    residuals = np.asarray(fit['residuals'], dtype=np.float64)
    if (residuals.shape != (27, 9) or not np.isfinite(residuals).all()
            or np.any(residuals[:, [0, -1]] != 0)
            or np.min(np.diff(KNOTS+residuals, axis=-1)) < -1e-10):
        raise ValueError('Invalid monotone calibration')
    out = x.copy()
    out[..., CHANNELS] = x[..., CHANNELS] + np.einsum(
        '...ck,ck->...c', basis(x[..., CHANNELS]), residuals)
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
