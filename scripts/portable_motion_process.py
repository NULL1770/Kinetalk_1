"""Principal symmetric PSD factors for reproducible controlled-prior coupling.

The frozen v2 sampler uses V sqrt(eigenvalues) as its noise factor. Eigenvector
column signs/order can differ across LAPACK implementations, changing seeded
paths despite identical laws. This module uses the unique principal symmetric
square root V sqrt(eigenvalues) V.T for both P and Q. It preserves the Gaussian
transition law while fixing that arbitrary basis dependence, including singular
PSD matrices (gain0). It does not improve fitted quality or reproduce v2 paths.

Cross-platform equality remains numerical, not guaranteed bitwise: different
BLAS arithmetic can still cause tiny floating-point differences. The original
v2 implementation and artifacts remain untouched.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import expit

from scripts import controlled_motion_process as legacy


FACTOR_SCHEMA = 'controlled_matern_principal_symmetric_psd_v1'
MotionState = legacy.MotionState
DT = legacy.DT
GROUPS = legacy.GROUPS
fit_prior = legacy.fit_prior
encode_reference = legacy.encode_reference


def principal_psd_sqrt(covariance, *, tolerance_scale=None):
    """Unique symmetric nonnegative square root, accepting rank deficiency.

Only numerical negative eigenvalues within1e-9*max(1,tolerance_scale) are
clipped. Material indefiniteness fails. Sorting the eigenpairs by eigenvalue
also keeps permutations from changing summation order when values are distinct.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError('Finite nonempty square covariance required')
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError('Symmetric covariance required')
    matrix = (matrix+matrix.T)/2
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values, kind='stable'); values, vectors = values[order], vectors[:, order]
    scale = float(np.max(np.abs(values))) if tolerance_scale is None else float(tolerance_scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('Finite nonnegative tolerance scale required')
    if values.min() < -1e-9*max(1., scale):
        raise FloatingPointError('Covariance is materially non-PSD')
    result = (vectors*np.sqrt(np.maximum(values, 0.)))@vectors.T
    return (result+result.T)/2


def transition(style, fitted, activity_gain=1.):
    """Same v2 A and P; replace only the nonunique innovation factor."""
    a, _, stationary = legacy.transition(style, fitted, activity_gain)
    covariance = stationary-a@stationary@a.T
    factor = principal_psd_sqrt((covariance+covariance.T)/2,
                                tolerance_scale=float(np.linalg.norm(stationary, 2)))
    return a, factor, stationary


def initialize(level9, style5, fitted, seed, key, *, stationary=True, activity_gain=1.):
    """Same PCG64 clock and style limits; symmetric factor for initial P."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not isinstance(key, str) or not key:
        raise ValueError('Integer seed and nonempty string key required')
    level = legacy._finite(level9, (9,), 'independent logit level').copy()
    style = legacy._style(style5, fitted); gain = legacy._gain(activity_gain)
    digest = hashlib.sha256(f'controlled_matern:{int(seed)}:{key}'.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], 'little')))
    a, factor, covariance = transition(style, fitted, gain)
    initial_noise = rng.standard_normal(18)
    initial = principal_psd_sqrt(covariance)@initial_noise if stationary else np.zeros(18)
    state = MotionState(level, fitted, style.copy(), gain, rng, initial[:9], initial[9:], np.zeros(9))
    state.target_style = style.copy(); state.target_gain = gain
    state.transition_cache = (a, factor, covariance)
    return state


def sample(state, frames, mode='run', style=None, activity_gain=None):
    """V2 control/state equations, with principal Q on every style blend step."""
    if not isinstance(state, MotionState) or isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or frames < 0:
        raise ValueError('MotionState and nonnegative integer frame count required')
    new_style = state.target_style if style is None else legacy._style(style, state.fitted)
    new_gain = state.target_gain if activity_gain is None else legacy._gain(activity_gain)
    if not np.array_equal(new_style, state.target_style) or new_gain != state.target_gain:
        state.blend_start_style = state.style.copy(); state.blend_start_gain = state.gain
        state.target_style = new_style.copy(); state.target_gain = new_gain; state.blend_age = 0
    legacy._set_mode(state, mode)
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
    return {**legacy.state_summary(state), 'factor_schema': FACTOR_SCHEMA}
