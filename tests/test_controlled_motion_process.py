"""Matérn covariance, fitting isolation and persistent-control contracts."""
import copy

import numpy as np
import pytest
from scipy.special import expit

from scripts import controlled_motion_process as m


def fitted_fixture():
    clips = []
    t = np.arange(200)*m.DT
    for index in range(5):
        waves = np.sin(t[:, None]*(1+.2*index)+np.arange(9)[None]*.2)*(.1+.05*index)
        clips.append({'clip_id': f'fit{index}', 'upper': expit(waves-.6), 'valid': np.ones(200, dtype=bool)})
    return m.fit_prior(clips), clips


def test_fit_clip_balanced_summaries_and_reference_no_arrays():
    fitted, clips = fitted_fixture()
    reference = m.encode_reference(clips[0], fitted)
    assert reference['style'].shape == (5,) and not reference['retained_reference_arrays']
    assert reference['source_clip_id'] == 'fit0' and not reference['reference_mean_used']
    assert fitted['fit_clip_ids'] == [clip['clip_id'] for clip in clips]
    assert np.allclose(np.diag(fitted['correlation']), 1)
    for group in m.GROUPS:
        assert np.isclose(np.mean(fitted['channel_profile'][list(group)]**2), 1)
    with pytest.raises(ValueError, match='allowlist'):
        m.encode_reference({**clips[0], 'clip_id': 'query'}, fitted)
    bad = copy.deepcopy(clips); bad[1]['clip_id'] = bad[0]['clip_id']
    with pytest.raises(ValueError, match='uniquely'):
        m.fit_prior(bad)


def test_no_gap_crossing_and_constant_shift_does_not_change_reference_style():
    fitted, clips = fitted_fixture()
    reference = copy.deepcopy(clips[0]); reference['valid'][60:80] = False; reference['upper'][60:80] = np.nan
    summary = m._summary(reference)
    expected = [sum(max(right-left-lag, 0) for left, right in [(0, 60), (80, 200)]) for lag in m.LAGS]
    assert summary['lag_pair_counts'] == expected
    raw = reference['upper']; logits = np.log(raw)-np.log1p(-raw)
    shifted = {**reference, 'upper': expit(logits+1)}
    a, b = m.encode_reference(reference, fitted), m.encode_reference(shifted, fitted)
    np.testing.assert_allclose(a['raw_style'], b['raw_style'], rtol=1e-12, atol=1e-12)


def test_exact_transition_stationary_covariance_and_stable_critical_roots():
    fitted, _ = fitted_fixture(); style = fitted['style_median']
    a, l, p = m.transition(style, fitted)
    np.testing.assert_allclose(a@p@a.T+l@l.T, p, rtol=1e-11, atol=1e-12)
    assert np.abs(np.linalg.eigvals(a)).max() < 1
    np.testing.assert_allclose(np.linalg.eigvals(a).real, np.exp(-m.DT/style[-1]), atol=1e-7)
    assert np.linalg.eigvalsh(l@l.T).min() > -1e-12


def test_exact_chunk_invariance_including_style_blending_and_state():
    fitted, _ = fitted_fixture(); style = fitted['style_median']; level = np.linspace(-2., 1., 9)
    first = m.initialize(level, style, fitted, 17, 'clip')
    second = m.initialize(level, style, fitted, 17, 'clip')
    a = m.sample(first, 50)
    b = np.concatenate([m.sample(second, n) for n in (1, 7, 11, 31)])
    np.testing.assert_array_equal(a, b)
    a = m.sample(first, 35, style=fitted['style_upper'], activity_gain=1.5)
    b = np.concatenate([m.sample(second, 3, style=fitted['style_upper'], activity_gain=1.5),
                        m.sample(second, 32, style=fitted['style_upper'], activity_gain=1.5)])
    np.testing.assert_array_equal(a, b)
    assert m.state_summary(first) == m.state_summary(second)


def test_hold_release_preserve_endpoints_and_resume_from_actual_state():
    fitted, _ = fitted_fixture(); level = np.linspace(-1.5, .5, 9)
    state = m.initialize(level, fitted['style_median'], fitted, 3, 'clip')
    m.sample(state, 25); endpoint = state.x+state.v*(8*m.DT)/2
    held = m.sample(state, 20, 'hold')
    np.testing.assert_array_equal(state.x, endpoint)
    np.testing.assert_array_equal(state.v, np.zeros(9)); np.testing.assert_array_equal(state.a, np.zeros(9))
    np.testing.assert_array_equal(held[7:], np.broadcast_to(held[7], held[7:].shape))
    released = m.sample(state, 30, 'release')
    np.testing.assert_array_equal(released[15:], np.broadcast_to(expit(level), released[15:].shape))
    np.testing.assert_array_equal(state.x, np.zeros(9))
    resumed = m.sample(state, 1, 'run'); assert not np.array_equal(resumed[0], expit(level))
    assert state.frame == 76


def test_hold_release_chunk_invariance_and_framewise_rng_alignment():
    fitted, _ = fitted_fixture(); style = fitted['style_median']
    a = m.initialize(np.zeros(9), style, fitted, 7, 'same')
    b = m.initialize(np.zeros(9), style, fitted, 7, 'same')
    m.sample(a, 10); m.sample(b, 10)
    np.testing.assert_array_equal(m.sample(a, 20, 'hold'), np.concatenate([m.sample(b, 3, 'hold'),m.sample(b, 17, 'hold')]))
    np.testing.assert_array_equal(m.sample(a, 19, 'release'), np.concatenate([m.sample(b, 9, 'release'),m.sample(b, 10, 'release')]))
    c = m.initialize(np.zeros(9), style, fitted, 7, 'same'); m.sample(c, 49)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state == c.rng.bit_generator.state


def test_long_run_stationary_covariance_matches_exact_target():
    fitted, _ = fitted_fixture(); style = fitted['style_median'].copy(); style[-1] = .2
    fitted['style_lower'][-1] = .08; fitted['style_upper'][-1] = 2
    state = m.initialize(np.zeros(9), style, fitted, 200, 'long')
    raw = m.sample(state, 35000)
    x = np.log(raw)-np.log1p(-raw)
    target = m.transition(style, fitted)[2][:9, :9]
    empirical = np.cov(x.T, bias=True)
    np.testing.assert_allclose(np.diag(empirical), np.diag(target), rtol=.09, atol=1e-4)
    assert np.linalg.norm(empirical-target)/np.linalg.norm(target) < .09
    assert np.linalg.norm(x.mean(0)) < .08


def test_gain_order_same_seed_and_zero_gain_stationary_output():
    fitted, _ = fitted_fixture(); style = fitted['style_median']
    gains = [0., .5, 1., 1.5]; energies = []
    for gain in gains:
        values = []
        for seed in (1, 2, 3):
            state = m.initialize(np.full(9, -.5), style, fitted, seed, 'gain', activity_gain=gain)
            raw = m.sample(state, 1000)
            values.append(np.mean((np.log(raw)-np.log1p(-raw)+.5)**2))
            assert np.isfinite(raw).all() and ((raw >= 0)&(raw <= 1)).all()
            if gain == 0:
                np.testing.assert_array_equal(raw, np.full((1000, 9), expit(-.5)))
        energies.append(np.mean(values))
    assert all(a < b for a, b in zip(energies, energies[1:]))


def test_style_swap_never_directly_rescales_physical_state():
    fitted, _ = fitted_fixture(); state = m.initialize(np.zeros(9), fitted['style_median'], fitted, 18, 'swap')
    m.sample(state, 10); x, v = state.x.copy(), state.v.copy()
    m.sample(state, 0, style=fitted['style_upper'], activity_gain=.5)
    np.testing.assert_array_equal(state.x, x); np.testing.assert_array_equal(state.v, v)
    m.sample(state, 8)
    np.testing.assert_array_equal(state.style, fitted['style_upper']); assert state.gain == .5


@pytest.mark.parametrize('gain', [True, -.1, float('nan'), 3.])
def test_bad_gain_rejected(gain):
    fitted, _ = fitted_fixture()
    with pytest.raises(ValueError, match='gain'):
        m.initialize(np.zeros(9), fitted['style_median'], fitted, 1, 'x', activity_gain=gain)
