import copy

import numpy as np
import pytest

from scripts.joint_motion_dictionary import (
    EPS, clipping_statistics, decode_unit, extract_units, fit_dictionary,
    logit, oracle_nearest, oracle_reconstruct, prior_logits,
    resample_trajectory, sigmoid, to_residual,
)


def clip(clip_id, frames=96, offset=0.):
    t = np.arange(frames)[:, None]
    upper = .4+offset+.15*np.sin(t*.11+np.arange(9)[None]*.13)
    return {'clip_id': clip_id, 'upper': upper, 'valid': np.ones(frames, bool),
        'anchor_upper': np.linspace(.25, .4, 9), 'global': np.arange(4)*.2+offset}


@pytest.fixture(scope='module')
def fitted():
    clips = [clip('a'), clip('b', offset=.1), clip('unused', offset=.2)]
    units = extract_units(clips, [0, 1])
    return clips, units, fit_dictionary(units, k=4)


def test_extract_fit_allowlist_no_gaps_or_short_tails():
    clips = [clip('fit'), clip('heldout')]
    clips[0]['valid'][29:36] = False
    clips[0]['upper'][29:36] = np.nan
    units = extract_units(clips, [0], durations=(16, 32, 48))
    assert units and {u['clip_id'] for u in units} == {'fit'}
    for u in units:
        assert clips[0]['valid'][u['start']-8:u['start']+u['duration']].all()
        assert len(u['future']) == u['duration']
        assert u['duration'] in (16, 32, 48)
        np.testing.assert_array_equal(u['raw_future'], clips[0]['upper'][u['start']:u['start']+u['duration']])
    assert extract_units([clip('short', frames=23)], [0]) == []


def test_extraction_stable_under_order_and_padding_and_no_mutation():
    clips = [clip('a'), clip('b')]
    original = copy.deepcopy(clips)
    x = extract_units(clips, [0, 1]); y = extract_units(clips, [1, 0])
    descriptor = lambda units: sorted((u['clip_id'], u['start'], u['duration']) for u in units)
    assert descriptor(x) == descriptor(y)
    padded = copy.deepcopy(clips[0]); padded['upper'] = np.r_[padded['upper'], np.full((20, 9), np.nan)]
    padded['valid'] = np.r_[padded['valid'], np.zeros(20, bool)]
    assert descriptor(extract_units([padded], [0])) == descriptor(extract_units(clips, [0]))
    np.testing.assert_array_equal(clips[0]['upper'], original[0]['upper'])
    with pytest.raises(ValueError, match='duplicate'):
        extract_units(clips, [0, 0])


def test_residual_roundtrip_and_clipping_report_do_not_hide_outliers():
    raw = np.tile(np.linspace(0, 1, 9), (5, 1))
    raw[1, 2] = -.3; raw[2, 7] = 1.7
    anchor = np.full(9, .2); anchor[0] = 0
    residual = to_residual(raw, anchor)
    np.testing.assert_allclose(sigmoid(residual+logit(anchor)), np.clip(raw, EPS, 1-EPS), atol=1e-15)
    report = clipping_statistics(raw, np.ones(5, bool), anchor)
    assert report['out_of_domain_values'] == 2
    assert report['clipped_values'] == 12
    assert report['anchor_clipped_values'] == 1
    assert report['raw_range'] == [-.3, 1.7]
    assert report['absolute_clipping_error_sum'] > 1
    assert sum(report['clipped_dynamic_energy']) < sum(report['dynamic_energy'])


def test_fitting_preserves_real_complete_medoid_and_training_lineage(fitted):
    clips, units, dictionary = fitted
    assert dictionary['source_clip_ids'] == ['a', 'b']
    assert len(set(dictionary['medoid_unit_indices'])) == 4
    for u, ix in zip(dictionary['medoids'], dictionary['medoid_unit_indices']):
        np.testing.assert_array_equal(u['future'], units[ix]['future'])
        assert len(u['future']) == u['duration']
        assert u['clip_id'] in ('a', 'b')
    assert dictionary['descriptors'].shape == (4, 360)
    assert dictionary['prefix_descriptors'].shape == (4, 8, 9)
    assert dictionary['future_descriptors'].shape == (4, 32, 9)
    assert dictionary['input_clipping']['observed_frames'] == len(clips[0]['upper'])+len(clips[1]['upper'])
    assert dictionary['input_clipping']['unique_source_count'] == 2
    # Inputs omitted from the fitting allowlist cannot alter fitted statistics.
    changed = copy.deepcopy(clips); changed[2]['upper'][:] = 1000; changed[2]['global'][:] = 1e9
    other = fit_dictionary(extract_units(changed, [0, 1]), k=4)
    np.testing.assert_array_equal(other['scales'], dictionary['scales'])
    assert other['medoid_unit_indices'] == dictionary['medoid_unit_indices']
    with pytest.raises(ValueError, match='no larger'):
        fit_dictionary(units[:2], k=3)


def test_stats_clip_balanced_not_domination_by_repeated_windows():
    source = extract_units([clip('a'), clip('b', offset=.1)], [0, 1])
    a = [u for u in source if u['clip_id'] == 'a']
    b = [u for u in source if u['clip_id'] == 'b']
    original = fit_dictionary(a+b, k=2)
    duplicate = fit_dictionary(a*3+b, k=2)
    np.testing.assert_allclose(original['scales'], duplicate['scales'], atol=1e-12)
    np.testing.assert_allclose(original['global_mean'], duplicate['global_mean'], atol=1e-12)
    np.testing.assert_allclose(original['global_scales'], duplicate['global_scales'], atol=1e-12)


def test_prior_has_no_query_target_and_global_history_ablation(fitted):
    _, _, d = fitted
    np.testing.assert_array_equal(prior_logits(d, None, None, use_global=False), np.zeros(4))
    prototype = d['medoids'][2]
    p = prior_logits(d, prototype['global'], prototype['prefix'])
    assert p[2] == pytest.approx(0.) and (p <= 1e-12).all()
    plain = prior_logits(d, None, prototype['prefix'], use_global=False)
    assert plain[2] == pytest.approx(0.)
    explicit = -((d['prefix_descriptors']-prototype['prefix']/d['scales'])**2).mean((1, 2))
    np.testing.assert_allclose(plain, explicit)
    with pytest.raises(ValueError, match='shape'):
        prior_logits(d, prototype['global'], prototype['prefix'][:7])


def test_decoder_maps_target_anchor_and_seam_correction_dies_out(fitted):
    _, _, d = fitted
    proto = d['medoids'][0]
    native = decode_unit(d, 0, proto['anchor_upper'])
    np.testing.assert_allclose(native, np.clip(proto['raw_future'], EPS, 1-EPS), atol=1e-15)
    target_anchor = np.linspace(.1, .7, 9); previous = np.linspace(.04, .93, 9)
    moved = decode_unit(d, 0, target_anchor)
    joined = decode_unit(d, 0, target_anchor, previous, blend_frames=8)
    np.testing.assert_allclose(joined[0], previous, atol=1e-15)
    np.testing.assert_array_equal(joined[7:], moved[7:])
    assert np.isfinite(joined).all() and (joined > 0).all() and (joined < 1).all()
    np.testing.assert_array_equal(decode_unit(d, 0, target_anchor, previous, blend_frames=0), moved)
    with pytest.raises(ValueError, match='at least two'):
        decode_unit(d, 0, target_anchor, previous, blend_frames=1)


def test_long_same_token_rollout_cannot_accumulate_delta_drift(fitted):
    _, _, d = fitted
    anchor = np.linspace(.15, .4, 9)
    baseline = decode_unit(d, 0, anchor)
    endpoint = np.full(9, .9999)
    for _ in range(100):
        output = decode_unit(d, 0, anchor, endpoint)
        np.testing.assert_array_equal(output[7:], baseline[7:])
        endpoint = output[-1]
    np.testing.assert_array_equal(endpoint, baseline[-1])


def test_oracle_explicitly_uses_future_and_same_prefix_count(fitted):
    _, _, d = fitted; u = d['medoids'][1]
    result = oracle_nearest(d, u['prefix'], u['future'])
    assert result['index'] == 1 and result['uses_query_future']
    assert result['prefix_frames'] == 8 and result['future_descriptor_frames'] == 32
    assert result['distances'][1] == pytest.approx(0.)
    reconstructed = oracle_reconstruct(d, u['prefix'], u['future'], u['anchor_upper'])
    np.testing.assert_allclose(reconstructed['raw'], np.clip(u['raw_future'], EPS, 1-EPS), atol=1e-15)
    assert reconstructed['uses_query_length']
    # A waveform-only oracle ablation is separately explicit.
    result = oracle_nearest(d, np.zeros((8, 9)), u['future'], prefix_weight=0.)
    assert result['index'] == 1


def test_time_resampling_endpoint_preservation_and_validation():
    x = np.arange(45.).reshape(5, 9)
    y = resample_trajectory(x, 32)
    np.testing.assert_array_equal(y[0], x[0]); np.testing.assert_array_equal(y[-1], x[-1])
    np.testing.assert_array_equal(resample_trajectory(x[:1], 4), np.repeat(x[:1], 4, axis=0))
    with pytest.raises(ValueError):
        resample_trajectory(np.zeros((0, 9)), 3)
    with pytest.raises(ValueError):
        logit(np.array([np.nan]))
