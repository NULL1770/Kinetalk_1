"""Reference independence and saved-level sampling contracts, no real-data claim."""
import copy
import inspect
import json

import numpy as np
import pytest
from scipy.special import expit

from scripts import run_controlled_prior as driver


def clip(cid, speaker, sentence, emotion):
    return {'clip_id': cid, 'speaker': speaker, 'sentence': sentence, 'emotion': emotion}


def test_references_same_person_different_query_sentence_with_audio_affect_priority():
    query = {**clip('query', 1, 'query_sentence', 6), 'predicted_emotion': 5}
    clips = [clip('query', 1, 'other_sentence', 5),  # SameID forbidden even if metadata inconsistent.
             clip('a_same_sentence', 1, 'query_sentence', 5),
             clip('a_other_person', 2, 'r0', 5),
             clip('b_reference', 1, 'r1', 5), clip('c_reference', 1, 'r2', 5),
             clip('a_nonpredicted', 1, 'r3', 6), clip('a_outside_fit', 1, 'r4', 5)]
    fit_ids = list(range(6))
    pair, matched = driver.choose_references(query, clips, fit_ids)
    assert pair == (3, 4) and matched
    for i in pair:
        assert i in fit_ids and clips[i]['speaker'] == query['speaker']
        assert clips[i]['clip_id'] != query['clip_id'] and clips[i]['sentence'] != query['sentence']
    assert clips[pair[0]]['sentence'] != clips[pair[1]]['sentence']
    # Query GT emotion is irrelevant; the provided audio prediction is used.
    query['emotion'] = 0
    assert driver.choose_references(query, clips, fit_ids) == (pair, True)


def test_reference_affect_fallback_requires_two_distinct_sentences():
    query = {**clip('query', 1, 'query_sentence', 6), 'predicted_emotion': 5}
    clips = [clip('a_first', 1, 'shared', 5), clip('b_same_reference_sentence', 1, 'shared', 5),
             clip('c_fallback', 1, 'other', 0)]
    pair, matched = driver.choose_references(query, clips, [0, 1, 2])
    assert pair == (0, 2) and not matched
    with pytest.raises(ValueError, match='another sentence|independent'):
        driver.choose_references(query, clips, [0, 1])


class NoQueryMotion(dict):
    def __getitem__(self, key):
        if key in ('upper', 'state', 'target', 'motion', 'energy'):
            raise AssertionError('Query teacher must not be read for control generation')
        return super().__getitem__(key)


def fit_fixture():
    clock = np.arange(128)*.04
    refs = [{'clip_id': f'reference{i}', 'upper': expit(.15*(i+1)*np.sin(clock[:, None]*(i+1)+np.arange(9)[None])),
             'valid': np.ones(128, bool)} for i in range(3)]
    fitted = driver.motion.fit_prior(refs)
    style = driver.motion.encode_reference(refs[0], fitted)['style']
    return fitted, style


def test_rollout_signature_motion_free_and_independent_level_matches_source_equation():
    assert list(inspect.signature(driver.rollout).parameters) == ['level', 'style', 'fitted', 'valid', 'seed', 'key', 'gain']
    fitted, style = fit_fixture()
    query = NoQueryMotion(global_=np.array([.1, .2]), anchor_upper=np.full(9, .3), upper=np.full((50, 9), np.nan))
    # Source equilibrium consumes only frozen global audio and an independent
    # neutral anchor. No query mean or query target enters this equation.
    eq = {'mean': np.zeros(11), 'std': np.ones(11), 'coef': np.zeros((12, 9))}
    eq['coef'][0] = np.linspace(-.05, .05, 9)
    anchor = driver.coordinates.logit(query['anchor_upper'])
    level = driver.c.equilibrium(query['global_'], anchor, eq)
    np.testing.assert_array_equal(level, anchor+eq['coef'][0])
    valid = np.ones(50, bool); valid[20:25] = False
    reference = driver.rollout(level, style, fitted, valid, 42, 'query', 0.)
    np.testing.assert_array_equal(reference[valid], np.broadcast_to(expit(level), reference[valid].shape))
    np.testing.assert_array_equal(reference[~valid], np.zeros((5, 9)))
    query['upper'] = np.random.default_rng(1).random((50, 9))
    changed = driver.rollout(level, style, fitted, valid, 42, 'query', 0.)
    np.testing.assert_array_equal(reference, changed)
    dynamic = driver.rollout(level, style, fitted, valid, 42, 'query')
    np.testing.assert_array_equal(dynamic, driver.rollout(level, style, fitted, valid, 42, 'query'))
    assert not np.array_equal(dynamic[valid], reference[valid])


def test_rollout_gaps_reset_independent_run_without_cross_gap_differences():
    fitted, style = fit_fixture(); valid = np.ones(90, bool); valid[32:40] = False
    level = np.zeros(9)
    actual = driver.rollout(level, style, fitted, valid, 3, 'gapped')
    for left, right in [(0, 32), (40, 90)]:
        state = driver.motion.initialize(level, style, fitted, 3, f'gapped:run:{left}', stationary=True)
        expected = driver.motion.sample(state, right-left)
        np.testing.assert_array_equal(actual[left:right], expected)
    np.testing.assert_array_equal(actual[32:40], np.zeros((8, 9)))


def test_stationary_statistics_finite_zero_energy_and_json_safe(tmp_path):
    raw = np.full((64, 9), .5)
    stats = driver.trajectory_stats(raw)
    assert stats['finite'] and stats['in_domain']
    np.testing.assert_array_equal(stats['group_rms'], np.zeros(4))
    assert stats['velocity_p95'] == stats['acceleration_p95'] == stats['above_3hz_energy_fraction'] == 0
    assert stats['activity_fraction_speed_norm_gt_005'] == 0 and stats['activity_run_seconds'] == []
    assert stats['autocorrelation_lags_1_4_8_16'] == [None]*4
    driver.save(tmp_path/'stats.json', stats)
    assert json.loads((tmp_path/'stats.json').read_text()) == stats


def test_long_controls_fixed_seed_hold_release_and_reference_swap(monkeypatch):
    fitted, style = fit_fixture(); style2 = fitted['style_upper']
    monkeypatch.setattr(driver, 'SEEDS', (42,))
    monkeypatch.setattr(driver, 'GAINS', (0., 1.))
    level = np.linspace(-1., .2, 9)
    records, arrays = driver.long_controls(level, [style, style2], fitted, 'metadata-first')
    assert len(records) == 3 and set(arrays) == {'gain_0.0', 'gain_1.0', 'controls'}
    np.testing.assert_array_equal(arrays['gain_0.0'], np.broadcast_to(expit(level).astype(np.float32), (1500, 9)))
    control = records[-1]
    assert control['control_times_frames'] == [128, 192, 256, 384]
    assert control['hold_speed_max'] == 0 and control['release_equilibrium_max_error'] == 0
    assert control['finite'] and control['in_domain']
    assert len(records[0]['quarter_group_rms']) == 4 and arrays['controls'].shape == (512, 9)
    json.dumps(records, allow_nan=False)


def test_metadata_picks_ignore_motion_quality():
    clips = []
    for speaker in ('M003', 'M004', 'M005', 'W009', 'W010', 'W011'):
        for emotion in (0, 1, 5, 6):
            for index in range(3):
                clips.append({**clip(f'mead_{speaker}_{emotion}_{index}', speaker, f's{index}', emotion),
                              'upper': np.full((32, 9), np.nan)})
    result = driver.picks_by_metadata(clips, list(range(len(clips))))
    assert len(result) == 32
    assert {clips[i]['speaker'] for i in result} == {'M003', 'M004', 'W009', 'W010'}
    assert all(clips[i]['clip_id'].endswith(('_0', '_1')) for i in result)


def test_fit_reference_native_windows_do_not_cross_missing_frames():
    raw = np.tile(np.arange(80, dtype=float)[:, None]/100, (1, 9))
    valid = np.ones(80, bool); valid[32:48] = False
    raw[32:48] = np.nan
    result = driver.fit_dynamics_reference([{'clip_id': 'gapped', 'upper': raw, 'valid': valid}])
    assert result['rows'][0]['windows'] == 2
    assert result['rows'][0]['velocity_p95'] == pytest.approx(.25)
    assert result['rows'][0]['acceleration_p95'] < 1e-12
    json.dumps(result, allow_nan=False)
