"""Noise-factor representation invariance, PSD support and seeded persistence."""
import numpy as np
import pytest
from scipy.special import expit

from scripts import controlled_motion_process as legacy
from scripts import portable_motion_process as portable


def fixture():
    time = np.arange(160)*.04
    clips = [{'clip_id': f'r{i}', 'upper': expit(.2*(i+1)*np.sin(time[:, None]*(i+1)+np.arange(9)[None]/3)),
              'valid': np.ones(160, bool)} for i in range(3)]
    fitted = legacy.fit_prior(clips)
    return fitted, fitted['style_median']


def test_principal_root_supports_psd_zero_and_rejects_indefinite():
    rng = np.random.default_rng(77); matrix = rng.normal(size=(18, 4)); covariance = matrix@matrix.T
    root = portable.principal_psd_sqrt(covariance)
    np.testing.assert_allclose(root, root.T, rtol=0, atol=0)
    np.testing.assert_allclose(root@root.T, covariance, rtol=1e-12, atol=1e-12)
    assert np.linalg.eigvalsh(root).min() > -1e-12
    np.testing.assert_array_equal(portable.principal_psd_sqrt(np.zeros((18, 18))), np.zeros((18, 18)))
    with pytest.raises(FloatingPointError, match='non-PSD'):
        portable.principal_psd_sqrt(np.diag([1., -.1]))


def test_eigenvector_sign_and_order_changes_leave_seeded_paths_unchanged(monkeypatch):
    fitted, style = fixture(); level = np.linspace(-1, .5, 9)
    def sequence():
        state = portable.initialize(level, style, fitted, 42, 'portable')
        return np.concatenate([portable.sample(state, 70), portable.sample(state, 20, 'hold'),
            portable.sample(state, 23, 'release'), portable.sample(state, 41, 'run',
                style=fitted['style_upper'], activity_gain=1.5)])
    expected = sequence()
    original = np.linalg.eigh
    def shuffled(matrix):
        values, vectors = original(matrix)
        permutation = np.arange(len(values))[::-1]
        signs = np.where(np.arange(len(values)) % 2, -1., 1.)
        return values[permutation], vectors[:, permutation]*signs
    monkeypatch.setattr(np.linalg, 'eigh', shuffled)
    actual = sequence()
    # Sorting and the symmetric product remove the exact sign/permutation
    # degrees of freedom; other LAPACK rounding may still need tolerance.
    np.testing.assert_array_equal(actual, expected)


def test_rotating_repeated_eigenspace_preserves_principal_factor(monkeypatch):
    matrix = np.diag([.2, .2, 1., 3.]); expected = portable.principal_psd_sqrt(matrix)
    original = np.linalg.eigh
    def rotated(value):
        eigenvalues, vectors = original(value)
        angle = .713
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        vectors[:, :2] = vectors[:, :2]@rotation
        return eigenvalues, vectors
    monkeypatch.setattr(np.linalg, 'eigh', rotated)
    np.testing.assert_allclose(portable.principal_psd_sqrt(matrix), expected, rtol=1e-14, atol=1e-14)


def test_transition_law_unchanged_and_chunk_invariance():
    fitted, style = fixture()
    a, l, p = portable.transition(style, fitted)
    old_a, old_l, old_p = legacy.transition(style, fitted)
    np.testing.assert_array_equal(a, old_a); np.testing.assert_array_equal(p, old_p)
    np.testing.assert_allclose(l@l.T, old_l@old_l.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a@p@a.T+l@l.T, p, rtol=1e-12, atol=1e-12)
    first = portable.initialize(np.zeros(9), style, fitted, 42, 'chunk')
    second = portable.initialize(np.zeros(9), style, fitted, 42, 'chunk')
    expected = portable.sample(first, 100, style=fitted['style_upper'])
    actual = np.concatenate([portable.sample(second, 5, style=fitted['style_upper']), portable.sample(second, 95)])
    np.testing.assert_array_equal(actual, expected)
    zero = portable.initialize(np.zeros(9), style, fitted, 42, 'zero', activity_gain=0.)
    np.testing.assert_array_equal(portable.sample(zero, 50), np.full((50, 9), .5))
