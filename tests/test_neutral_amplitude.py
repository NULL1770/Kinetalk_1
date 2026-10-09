import copy
import numpy as np
import pytest
from kinetalk_b0.neutral_amplitude import (
    KNOTS, CHANNELS, SCHEMA, NeutralAmplitudeFit, apply_calibration, basis, held_gate, measures)


def sample(n=80, offset=0.):
    x = np.tile(np.linspace(.01, .8, n)[:, None], (1, 52))
    x = np.clip(x+offset, 0., 1.)
    y = x.copy(); y[:, CHANNELS] = x[:, CHANNELS]+.2*x[:, CHANNELS]*(1-x[:, CHANNELS])
    return x, y, np.ones_like(x, dtype=bool), np.arange(n)*.04


def provenance(i):
    return dict(clip_id=str(i), sentence_id='s'+str(i), speaker='1', role='train',
        target_kind='native_neutral', target_sha256='synthetic')


def fit_sample():
    f = NeutralAmplitudeFit()
    for i in range(2):
        f.add(*sample(offset=i*.03), provenance=provenance(i))
    return f.solve()


def test_improves_independent_monotone_target():
    state = fit_sample(); x, y, mask, t = sample(n=99, offset=.01)
    out = apply_calibration(x, state)
    assert np.mean((out-y)**2) < .01*np.mean((x-y)**2)
    assert state['count'][0] == [2]*27
    assert all(v['objective_change'] <= 1e-10 for v in state['solver'])


def test_no_time_mixing_unmodified_channels_and_tails():
    fit = fit_sample(); x = np.tile(np.linspace(-.2, 1.2, 80)[:, None], (1, 52))
    p = apply_calibration(x, fit)
    np.testing.assert_allclose(apply_calibration(x[::-1], fit), p[::-1], rtol=0, atol=0)
    np.testing.assert_array_equal(p[:, :14], x[:, :14])
    np.testing.assert_array_equal(p[:, 41:], x[:, 41:])
    np.testing.assert_array_equal(p[(x[:, 0] < 0)|(x[:, 0] > 1)], x[(x[:, 0] < 0)|(x[:, 0] > 1)])
    assert np.diff(p, axis=0).min() >= -1e-12
    for endpoint in (0., 1.):
        value = np.full((2, 52), endpoint)
        np.testing.assert_array_equal(apply_calibration(value, fit), value)


def test_masked_nan_and_observed_nan():
    x, y, m, t = sample(); m[20:30, 17] = False; m[10] = False
    a, b = NeutralAmplitudeFit(), NeutralAmplitudeFit()
    a.add(x, y, m, t, provenance=provenance(0))
    x[~m] = np.nan; y[~m] = np.nan
    b.add(x, y, m, t, provenance=provenance(0))
    np.testing.assert_array_equal(a.xx, b.xx); np.testing.assert_array_equal(a.xy, b.xy)
    x[0, 17] = np.nan
    with pytest.raises(ValueError, match='finite'):
        b.add(x, y, m, t, provenance=provenance(1))


def test_no_difference_across_time_gap():
    x, y, m, t = sample(); t[40:] += 1.
    f = NeutralAmplitudeFit(); f.add(x, y, m, t, provenance=provenance(0))
    a = basis(x[:, CHANNELS]); d = np.diff(a, axis=0)
    keep = np.ones(79, bool); keep[39] = False
    expected = np.einsum('tck,tcl->ckl', d[keep], d[keep])/keep.sum()
    np.testing.assert_allclose(f.xx[1], expected, rtol=1e-12, atol=1e-12)
    # Measurement must obey the same gap, including a large target jump.
    y[40:] += 5.
    result = measures(x, y, m, t)
    err = np.diff((x-y)[:, CHANNELS], axis=0)
    np.testing.assert_allclose(result['displacement_mse'], np.mean(err[keep]**2))


def test_clip_weighting_not_duration_weighting():
    x, y, m, t = sample()
    a, b = NeutralAmplitudeFit(), NeutralAmplitudeFit()
    a.add(x, y, m, t, provenance=provenance(0))
    # Repeat the performance across a gap so all within-clip averages match.
    b.add(np.tile(x, (2, 1)), np.tile(y, (2, 1)), np.tile(m, (2, 1)),
        np.r_[t, t+10], provenance=provenance(0))
    np.testing.assert_allclose(a.xx, b.xx, atol=1e-14)
    np.testing.assert_allclose(a.xy, b.xy, atol=1e-14)


@pytest.mark.parametrize('change', [dict(role='speaker_dev'), dict(target_kind='emotional_gt'), dict(target_sha256='')])
def test_target_contract(change):
    p = provenance(0); p.update(change)
    with pytest.raises(ValueError):
        NeutralAmplitudeFit().add(*sample(), provenance=p)


def test_duplicate_clip_and_absent_channel():
    f = NeutralAmplitudeFit(); f.add(*sample(), provenance=provenance(0))
    with pytest.raises(ValueError, match='Duplicate'):
        f.add(*sample(), provenance=provenance(0))
    x, y, m, t = sample(); m[:, 17] = False
    f.add(x, y, m, t, provenance=provenance(1))
    with pytest.raises(ValueError, match='two observed'):
        f.solve()


def test_identity_state_roundtrip_and_reject_invalid():
    state = dict(schema=SCHEMA, channels=list(CHANNELS), knots=KNOTS.tolist(),
        residuals=np.zeros((27, 9)).tolist(), development_used_for_fit=False,
        test_used_for_fit=False, input='frozen_neutral_B0_only')
    x = sample()[0]
    np.testing.assert_array_equal(apply_calibration(x, state), x)
    for change in (dict(test_used_for_fit=True), dict(input='audio'),
            dict(residuals=np.ones((27, 9)).tolist())):
        with pytest.raises(ValueError):
            apply_calibration(x, state|change)


def test_gate_requires_all_held_subsets_and_both_views():
    groups = {}
    for role in ('speaker_dev', 'sentence_dev'):
        for kind in ('native_neutral', 'approved_neutral_pair'):
            for view in ('raw', 'clip'):
                b = dict(mouth_mse=1., displacement_mse=1., jaw_range_absolute_error=.1, jaw_corr=.7, closure_f1=.8)
                c = b|dict(mouth_mse=.9, jaw_range_absolute_error=.08)
                groups['/'.join((role, kind, view))] = dict(clips=3, base=b, candidate=c)
    assert held_gate(groups)['passed']
    for field, value in [('jaw_corr', .69), ('closure_f1', .78), ('displacement_mse', 1.02),
            ('mouth_mse', .995), ('jaw_range_absolute_error', .11), ('jaw_corr', None)]:
        bad = copy.deepcopy(groups); bad['speaker_dev/native_neutral/raw']['candidate'][field] = value
        assert not held_gate(bad)['passed']
    groups.pop('sentence_dev/approved_neutral_pair/clip')
    assert not held_gate(groups)['passed']
