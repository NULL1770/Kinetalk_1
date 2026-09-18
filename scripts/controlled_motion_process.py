"""Reference-controlled critically damped Gaussian motion, not audio timing.

The nine logit coefficients follow a stationary Matérn3/2 process. For each
channel, d[x,v]/dt = [v,-lambda**2*x-2*lambda*v] plus joint velocity noise.
At dt=.04 its exact transition is
 A=exp(-lambda*dt)*[[1+lambda*dt,dt],[-lambda**2*dt,1-lambda*dt]].
P=diag(S,lambda**2*S), S=D C D; Q=P-A P A.T supplies exact innovations.

Only fit recordings determine C, channel profiles and reference-style limits.
An independent reference becomes four group RMS values and one time constant;
its mean, directions and frame sequence never enter the generator. This is a
controlled stochastic baseline, not evidence of semantic action discovery or
predictable audio timing. Sigmoid boundedness does not certify naturalness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np
from scipy.special import expit


SCHEMA = 'controlled_critical_matern_motion_v1'
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
DT = 1/25
EPS = 1e-4
TAU_GRID = np.geomspace(.08, 2., 64)
LAGS = (1, 4, 8, 16)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _finite(value, shape, name):
    out = np.asarray(_array(value), dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'{name} must be finite with shape{shape}')
    return out


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _summary(clip):
    cid = clip['clip_id']; raw = np.asarray(_array(clip['upper']), dtype=np.float64)
    valid = _array(clip['valid'])
    if (not isinstance(cid, str) or not cid or raw.ndim != 2 or raw.shape[1] != 9
            or valid.dtype != np.bool_ or valid.shape != raw.shape[:1]
            or valid.sum() < 2 or not np.isfinite(raw[valid]).all()):
        raise ValueError('Unique clip ID, observed upper[T,9] and Boolean valid[T] required')
    logit = np.zeros_like(raw)
    clipped = np.clip(raw[valid], EPS, 1-EPS)
    logit[valid] = np.log(clipped)-np.log1p(-clipped)
    centered = np.zeros_like(logit); centered[valid] = logit[valid]-logit[valid].mean(0)
    rms = np.sqrt(np.square(centered[valid]).mean(0))
    group_rms = np.asarray([np.sqrt(np.square(rms[list(group)]).mean()) for group in GROUPS])
    normalized = centered[valid]/np.maximum(rms, 1e-6)
    covariance = normalized.T@normalized/len(normalized)
    lag_values, counts = [], []
    for lag in LAGS:
        numerator = denominator = 0.; count = 0
        for left, right in _runs(valid):
            if right-left <= lag:
                continue
            first, second = centered[left:right-lag], centered[left+lag:right]
            numerator += float((first*second).sum())
            denominator += float((np.square(first).sum()+np.square(second).sum())/2)
            count += len(first)
        lag_values.append(numerator/max(denominator, 1e-15) if count else np.nan)
        counts.append(count)
    supported = np.asarray(counts) > 0
    if not supported.any():
        raise ValueError('Reference has no within-run lag observations')
    observed = np.asarray(lag_values)[supported]
    times = np.asarray(LAGS)[supported]*DT
    predicted = (1+times[None]/TAU_GRID[:, None])*np.exp(-times[None]/TAU_GRID[:, None])
    error = np.square(predicted-observed).mean(1)
    tau = float(TAU_GRID[int(error.argmin())])
    return {'clip_id': cid, 'channel_rms': rms, 'group_rms': group_rms, 'tau': tau,
        'correlation': covariance, 'lag_correlation': [None if not n else float(v) for v, n in zip(lag_values, counts)],
        'lag_pair_counts': counts, 'tau_fit_mse': float(error.min()),
        'observed_frames': int(valid.sum()), 'logit_clipped_fraction': float(np.mean(clipped != raw[valid]))}


def fit_prior(clips):
    """Fit clip-equal correlation, channel profiles and q10/q90 style limits.

The caller supplies the exact fitting allowlist. Each clip contributes once,
independent of its duration. Summaries retain no motion arrays. Lag products
use a single observed-clip logit mean and never cross missing runs.
    """
    summaries = [_summary(clip) for clip in clips]
    ids = [row['clip_id'] for row in summaries]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('At least two uniquely named fitting recordings required')
    correlation = np.mean([row['correlation'] for row in summaries], axis=0)
    correlation = .9*correlation+.1*np.eye(9)
    values, vectors = np.linalg.eigh((correlation+correlation.T)/2)
    correlation = (vectors*np.maximum(values, 1e-6))@vectors.T
    std = np.sqrt(np.diag(correlation)); correlation /= std[:, None]*std[None]
    channel = np.median([row['channel_rms'] for row in summaries], axis=0)
    groups = np.median([row['group_rms'] for row in summaries], axis=0)
    profile = np.ones(9)
    for index, group in enumerate(GROUPS):
        ids_group = list(group)
        raw_profile = channel[ids_group]/max(groups[index], 1e-6)
        rms = float(np.sqrt(np.square(raw_profile).mean()))
        profile[ids_group] = raw_profile/rms if rms > 1e-6 else 1.
    styles = np.stack([np.r_[row['group_rms'], row['tau']] for row in summaries])
    lower, upper = np.quantile(styles, [.1, .9], axis=0)
    lower[:4] = np.maximum(lower[:4], 1e-6); upper[:4] = np.maximum(upper[:4], lower[:4])
    serial = [{key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in row.items() if key != 'correlation'} for row in summaries]
    return {'schema': SCHEMA, 'fit_clip_ids': ids, 'correlation': correlation, 'channel_profile': profile,
        'style_lower': lower, 'style_upper': upper, 'style_median': np.median(styles, axis=0),
        'clip_summaries': serial, 'logit_eps': EPS, 'dt': DT, 'groups': [list(g) for g in GROUPS],
        'tau_grid': TAU_GRID.tolist(), 'lag_frames': list(LAGS), 'correlation_shrinkage': .1,
        'fit_weighting': 'equal recording; no motion arrays retained'}


def _check_fit(fitted):
    if fitted.get('schema') != SCHEMA or fitted.get('dt') != DT:
        raise ValueError('Compatible saved fitting statistics required')
    correlation = _finite(fitted['correlation'], (9, 9), 'correlation')
    profile = _finite(fitted['channel_profile'], (9,), 'channel profile')
    lower = _finite(fitted['style_lower'], (5,), 'style lower bound')
    upper = _finite(fitted['style_upper'], (5,), 'style upper bound')
    if ((profile < 0).any() or (lower < 0).any() or (upper < lower).any() or lower[4] <= 0
            or not np.allclose(correlation, correlation.T, atol=1e-10)
            or np.linalg.eigvalsh(correlation).min() < -1e-9):
        raise ValueError('Invalid covariance/profile/style fitting statistics')
    return correlation, profile, lower, upper


def encode_reference(reference, fitted):
    """Compress one independent fit reference to bounded style5, no sequence.

Caller enforces same-speaker, different-sentence and reference/query separation.
The core additionally requires reference membership in the saved fit allowlist.
    """
    _, _, lower, upper = _check_fit(fitted)
    if reference['clip_id'] not in fitted['fit_clip_ids']:
        raise ValueError('Reference must belong to the fitting source allowlist')
    summary = _summary(reference)
    raw = np.r_[summary['group_rms'], summary['tau']]
    style = np.clip(raw, lower, upper)
    return {'style': style, 'raw_style': raw, 'source_clip_id': summary['clip_id'],
        'clipped_fraction': float(np.mean(style != raw)),
        'logit_clipped_fraction': summary['logit_clipped_fraction'],
        'retained_reference_arrays': False, 'reference_mean_used': False}


def _gain(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError('Explicit activity gain must be finite in[0,2]')
    return float(value)


def _style(value, fitted):
    style = _finite(value, (5,), 'style')
    _, _, lower, upper = _check_fit(fitted)
    return np.clip(style, lower, upper)


def transition(style, fitted, activity_gain=1.):
    """Exact18D transition/innovation factors and stationary covariance.

Q numerical eigenvalues below -1e-9*max(1,||P||2) fail. Tiny numerical negative
values are clipped to zero; no learned or query-derived noise rescaling occurs.
    """
    correlation, profile, _, _ = _check_fit(fitted)
    style = _finite(style, (5,), 'style'); gain = _gain(activity_gain)
    if (style[:4] < 0).any() or style[4] <= 0:
        raise ValueError('Nonnegative RMS and positive time constant required')
    channel = np.zeros(9)
    for index, group in enumerate(GROUPS):
        channel[list(group)] = style[index]*profile[list(group)]*gain
    covariance = correlation*channel[:, None]*channel[None]
    lam = 1/style[4]; decay = np.exp(-lam*DT)
    a2 = decay*np.array([[1+lam*DT, DT], [-lam*lam*DT, 1-lam*DT]])
    identity = np.eye(9)
    a = np.block([[a2[0, 0]*identity, a2[0, 1]*identity], [a2[1, 0]*identity, a2[1, 1]*identity]])
    zero = np.zeros((9, 9)); stationary = np.block([[covariance, zero], [zero, lam*lam*covariance]])
    q = stationary-a@stationary@a.T; q = (q+q.T)/2
    values, vectors = np.linalg.eigh(q)
    if values.min() < -1e-9*max(1., float(np.linalg.norm(stationary, 2))):
        raise FloatingPointError('Innovation covariance is materially non-PSD')
    factor = vectors@np.diag(np.sqrt(np.maximum(values, 0.)))
    return a, factor, stationary


def _stationary_factor(covariance):
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    return vectors@np.diag(np.sqrt(np.maximum(values, 0.)))


@dataclass
class MotionState:
    level: np.ndarray
    fitted: dict
    style: np.ndarray
    gain: float
    rng: np.random.Generator
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    frame: int = 0
    mode: str = 'run'
    target_style: np.ndarray | None = None
    target_gain: float | None = None
    blend_start_style: np.ndarray | None = None
    blend_start_gain: float = 1.
    blend_age: int = 8
    control_coefficients: np.ndarray | None = None
    control_age: int = 0
    control_frames: int = 0
    control_endpoint: np.ndarray | None = None
    transition_cache: tuple | None = field(default=None, repr=False)


def initialize(level9, style5, fitted, seed, key, *, stationary=True, activity_gain=1.):
    """Start from independently predicted logit level and reference style only.

PCG64 is keyed by SHA256(seed/key);18 normals initialize stationary state and
exactly18 are consumed each subsequent frame, including HOLD/RELEASE frames.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not isinstance(key, str) or not key:
        raise ValueError('Integer seed and nonempty string key required')
    level = _finite(level9, (9,), 'independent logit level').copy()
    style = _style(style5, fitted); gain = _gain(activity_gain)
    digest = hashlib.sha256(f'controlled_matern:{int(seed)}:{key}'.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], 'little')))
    a, factor, covariance = transition(style, fitted, gain)
    initial_noise = rng.standard_normal(18)
    initial = _stationary_factor(covariance)@initial_noise if stationary else np.zeros(18)
    state = MotionState(level, fitted, style.copy(), gain, rng, initial[:9], initial[9:], np.zeros(9))
    state.target_style = style.copy(); state.target_gain = gain
    state.transition_cache = (a, factor, covariance)
    return state


def _quintic(start, velocity, acceleration, endpoint, duration):
    # Coefficients are powers of normalized time u=t/duration. Initial and
    # final position, velocity and acceleration constraints define one quintic.
    c0 = start.copy(); c1 = velocity*duration; c2 = acceleration*duration*duration/2
    rhs = np.stack((endpoint-c0-c1-c2, -c1-2*c2, -2*c2))
    final = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]), rhs)
    return np.vstack((c0, c1, c2, final))


def _set_mode(state, mode):
    if mode not in ('run', 'hold', 'release'):
        raise ValueError('Control mode must be run, hold or release')
    if mode == state.mode:
        return
    state.mode = mode
    state.control_age = 0
    if mode in ('hold', 'release'):
        frames = 8 if mode == 'hold' else 16
        endpoint = state.x+state.v*(frames*DT)/2 if mode == 'hold' else np.zeros(9)
        state.control_coefficients = _quintic(state.x, state.v, state.a, endpoint, frames*DT)
        state.control_endpoint = endpoint
        state.control_frames = frames
    else:
        state.control_coefficients = None; state.control_endpoint = None; state.control_frames = 0


def sample(state, frames, mode='run', style=None, activity_gain=None):
    """Mutate persistent state and return bounded raw coefficients[frames,9].

Repeated calls are exactly equivalent to one call with the same mode/control
schedule. Style/gain requests interpolate parameters over8 frames; actual x/v
are never rescaled. HOLD eases to x+v*T/2 in8 frames; RELEASE eases to x=0 in16.
    """
    if not isinstance(state, MotionState) or isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or frames < 0:
        raise ValueError('MotionState and nonnegative integer frame count required')
    new_style = state.target_style if style is None else _style(style, state.fitted)
    new_gain = state.target_gain if activity_gain is None else _gain(activity_gain)
    if not np.array_equal(new_style, state.target_style) or new_gain != state.target_gain:
        state.blend_start_style = state.style.copy(); state.blend_start_gain = state.gain
        state.target_style = new_style.copy(); state.target_gain = new_gain; state.blend_age = 0
    _set_mode(state, mode)
    output = np.empty((frames, 9), dtype=np.float64)
    for frame in range(frames):
        noise = state.rng.standard_normal(18)
        if state.blend_age < 8:
            state.blend_age += 1; weight = state.blend_age/8
            state.style = (1-weight)*state.blend_start_style+weight*state.target_style
            state.gain = (1-weight)*state.blend_start_gain+weight*state.target_gain
            state.transition_cache = transition(state.style, state.fitted, state.gain)
        if state.mode == 'run':
            a, factor, _ = state.transition_cache
            previous_velocity = state.v.copy()
            next_state = a@np.r_[state.x, state.v]+factor@noise
            state.x, state.v = next_state[:9], next_state[9:]
            state.a = (state.v-previous_velocity)/DT
        elif state.control_age < state.control_frames:
            state.control_age += 1
            u = state.control_age/state.control_frames; duration = state.control_frames*DT
            coeff = state.control_coefficients
            state.x = np.power(u, np.arange(6))@coeff
            state.v = (np.arange(1, 6)*np.power(u, np.arange(5)))@coeff[1:]/duration
            state.a = (np.arange(2, 6)*np.arange(1, 5)*np.power(u, np.arange(4)))@coeff[2:]/duration**2
            if state.control_age == state.control_frames:
                state.x = state.control_endpoint.copy(); state.v = np.zeros(9); state.a = np.zeros(9)
        else:
            state.v = np.zeros(9); state.a = np.zeros(9)
        if not all(np.isfinite(value).all() for value in (state.x, state.v, state.a)):
            raise FloatingPointError('Controlled state became nonfinite')
        output[frame] = expit(state.level+state.x)
        state.frame += 1
    return output


def state_summary(state):
    return {'frame': state.frame, 'mode': state.mode, 'style': state.style.tolist(),
        'activity_gain': state.gain, 'x': state.x.tolist(), 'v': state.v.tolist(), 'a': state.a.tolist(),
        'control_age': state.control_age, 'control_frames': state.control_frames,
        'style_blend_age': state.blend_age, 'noise_draws_per_frame': 18,
        'reference_frames_retained': False, 'query_motion_used': False}
