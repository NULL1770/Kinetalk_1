"""Protocol tests for the end-to-end empirical joint-prior driver."""
import inspect

import numpy as np
import pytest

from scripts import train_joint_motion_prior as p


def _clips(sentence_count=54, copies=3):
    return [{'sentence': f'sentence-{s:02d}', 'clip_id': f'clip-{s:02d}-{j}',
             'speaker': j, 'emotion': j % 4} for s in range(sentence_count) for j in range(copies)]


def test_inner_split_has_fixed_disjoint_sentence_cells_and_expected_counts():
    clips = _clips()
    original = {'fit': list(range(len(clips))), 'speaker': [0], 'joint': [1]}
    split = p.split_inner(clips, original)
    assert len(split['fit']) == 34 * 3
    assert len(split['calibration']) == 10 * 3
    assert len(split['confirmation']) == 10 * 3
    cells = [set(split[key]) for key in ('fit', 'calibration', 'confirmation')]
    assert not (cells[0] & cells[1] or cells[0] & cells[2] or cells[1] & cells[2])
    assert set().union(*cells) == set(original['fit'])
    for key in ('speaker', 'joint'):
        assert split['old_' + key] == original[key]


def test_inner_split_is_metadata_hash_bound_and_rejects_too_few_sentences():
    clips = _clips()
    reversed_clips = list(reversed(clips))
    # Compare membership by immutable clip id, independent of input order.
    a = p.split_inner(clips, {'fit': list(range(len(clips)))})
    b = p.split_inner(reversed_clips, {'fit': list(range(len(clips)))})
    for key in ('fit', 'calibration', 'confirmation'):
        assert {clips[i]['clip_id'] for i in a[key]} == {reversed_clips[i]['clip_id'] for i in b[key]}
    short = _clips(sentence_count=23)
    with pytest.raises(ValueError, match='24'):
        p.split_inner(short, {'fit': list(range(len(short)))})


def test_rollout_api_does_not_accept_query_motion_or_teacher_boundaries():
    signature = inspect.signature(p.rollout)
    assert not {'target', 'motion', 'query', 'teacher', 'boundaries', 'future'} & set(signature.parameters)
    source = inspect.getsource(p.rollout)
    assert 'c[\'upper\']' not in source
    assert 'oracle_nearest' not in source


def test_driver_payload_selection_excludes_metadata_only_old_cells():
    source = inspect.getsource(p.main)
    assert "for key in ('fit','calibration','confirmation') for i in split[key]" in source
    assert "for ids in split.values()" not in source


def test_local_audio_training_is_strictly_blocked_after_motion_gate_failure():
    assert p.local_training_allowed(True)
    assert not p.local_training_allowed(False)
    # The driver condition must use the gate helper rather than smoke as an
    # override; smoke validates the no-local branch when motion is infeasible.
    source = inspect.getsource(p.main)
    assert 'if local_training_allowed(best is not None):' in source
    assert 'best is not None or args.smoke' not in source


def test_confirmation_arms_share_one_frozen_dictionary_and_fixed_setting():
    source = inspect.getsource(p.main)
    # Every confirmation arm is evaluated through the same dictionary/scales;
    # only mode/network/declared local scale may vary.
    assert "evaluate(clips,split['confirmation'],dictionary,scales" in source
    assert "for label,mode,sett,net,weight in" in source
    assert source.count("dictionary,scales,sett") >= 1
    assert "dictionary=d.fit_dictionary" in source
    assert "fit_dictionary(units" in source


def test_terminal_status_is_complete_and_declares_no_integration_or_test_read():
    status = p.final_status(smoke=False, seconds=12.5, local_trained=False,
                            local_epochs=0, motion_calibration_passed=False,
                            motion_confirmation_passed=False)
    assert status == {'schema': p.SCHEMA, 'status': 'complete', 'smoke': False,
        'seconds': 12.5, 'local_trained': False, 'local_epochs': 0,
        'motion_calibration_passed': False, 'motion_confirmation_passed': False,
        'generator_integrated': False, 'test_loaded': False, 'default_replaced': False}


def _report(temporal=True):
    support = {'sum_squares': 1., 'count': 1}
    return {'summary': {'rms_ratio': [1., 1., 1., 1.],
        'variogram': {'aggregate': 1. if temporal else None},
        'covariance_distance': {'velocity': 1. if temporal else None},
        'speed': {'seam': support if temporal else None, 'within': support if temporal else None}}}


def test_quality_gate_returns_failed_result_for_empty_temporal_support():
    result = p.quality_gate(_report(False), _report(True))
    assert result['passed'] is False
    assert result['reason'] == 'missing_temporal_support_or_nonfinite_metrics'


def test_quality_gate_requires_same_motion_support_baseline_and_finite_values():
    good = p.quality_gate(_report(True), _report(True))
    assert good['passed'] is True
    bad = _report(True); bad['summary']['rms_ratio'][0] = np.nan
    result = p.quality_gate(bad, _report(True))
    assert result['passed'] is False
    assert result['reason'] == 'missing_temporal_support_or_nonfinite_metrics'


def test_bootstrap_gain_is_baseline_minus_real_and_pair_order_is_checked():
    real = {'rows': [{'clip_id': 'a', 'sentence': 's1', 'joint_fair_es': {'centered': 1.}},
                     {'clip_id': 'b', 'sentence': 's2', 'joint_fair_es': {'centered': 2.}}]}
    baseline = {'rows': [{'clip_id': 'a', 'sentence': 's1', 'joint_fair_es': {'centered': 3.}},
                         {'clip_id': 'b', 'sentence': 's2', 'joint_fair_es': {'centered': 4.}}]}
    gain = p.bootstrap_gain(real, baseline)
    assert gain['gain'] == pytest.approx(2.)
    with pytest.raises(ValueError, match='Paired IDs'):
        p.bootstrap_gain(real, {'rows': list(reversed(baseline['rows']))})


def test_paired_subset_recomputes_summary_on_explicit_common_support():
    rows = [{'clip_id': f'c{i}', 'sentence': f's{i}', 'joint_fair_es': {'centered': 0.}
             } for i in range(3)]
    # Use a real score row so summarize receives every required metric field.
    from scripts.joint_motion_metrics import score_clip
    target = np.zeros((2, 9)); raw = np.zeros((2, 9));
    row = score_clip(np.stack([raw, raw]), target, np.ones(2, bool), np.ones(9))
    rows = [{**rows[i], **row} for i in range(3)]
    result = {'summary': {'stale': True}, 'rows': rows, 'mode': 'global'}
    subset = p.paired_subset(result, {'c0', 'c2'})
    assert [r['clip_id'] for r in subset['rows']] == ['c0', 'c2']
    assert subset['summary']['clips'] == 2
