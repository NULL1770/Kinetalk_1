import json

import numpy as np
import pytest
import torch

from scripts import summarize_prior_audio_adapter as s
from scripts.audit_prior_audio_adapter import audit
from scripts.joint_motion_metrics import score_clip, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def archive(path, role, split='holdout'):
    path.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 12, 35); target = .4 + .09 * np.sin(t)[:, None] * np.ones((1, 9))
    samples = target[None] + np.array([-.01, -.003, .003, .01])[:, None, None] * np.cos(t)[None, :, None]
    if role == 'prior':
        samples += .06 * np.cos(.7 * t)[None, :, None]
    elif role != 'audio':
        samples += .035 * np.cos(.7 * t)[None, :, None]
    curves = {}; scores = []; checks = []
    for index in range(4):
        cid = ('inner' if split == 'inner_validation' else 'outer') + str(index)
        meta = {'clip_id': cid, 'sentence': 'sentence' + str(index // 2), 'split': split, 'speaker': index % 2, 'emotion': 0}
        mask = np.ones(35, bool)
        metric = score_clip(samples, target, mask, np.full(9, .2)); metric.update(meta); scores.append(metric)
        curves[cid] = {'metadata': meta, 'samples': samples.copy(), 'target': target.copy(),
                       'native_valid': mask.copy(), 'score_mask': mask.copy(), 'generated_mask': mask.copy(),
                       'seeds': [42, 123, 2026, 77], 'run_records': [{'noise_seed': index}],
                       'raw_coefficient_prediction': True}
        checks.append({'clip_id': cid, 'nonfinite_values': 0, 'other43_exact': True,
                       'native_invalid_exact_baseline': True, 'prediction_clipped': False})
    result = {'schema': s.EVALUATION_SCHEMA, 'mode': 'free_generation', 'clips': 4, 'scored_clips': 4,
              'per_clip_scores': scores, 'summary': summarize(scores), 'numerics': checks,
              'numerical_gate': {'passed': True}, 'use_audio': role != 'prior', 'steps': 24,
              'intervention': 'real' if role in ('audio', 'prior') else 'static' if role in ('static', 'matched_static') else role}
    write(path / 'result.json', result)
    torch.save({'schema': s.EVALUATION_SCHEMA, 'clips': curves, 'result': result}, path / 'curves.pt')
    return path


def checkpoints(root, protocol):
    state = {'linear.weight': torch.tensor([1., 2.]), 'linear.bias': torch.tensor([.3])}
    digest = s._value_sha(state)
    binding = {'protocol_sha256': s._value_sha(protocol), 'ae_sha256': 'AE', 'stats_sha256': 'stats'}
    torch.save({'stage': 'prior', 'step': 10, 'budget': 10, 'state': state, 'binding': binding}, root / 'prior_final.pt')
    for step in protocol['milestones']:
        point = root / f'point_{step}'; point.mkdir()
        for arm in ('audio', 'matched_static'):
            torch.save({'stage': arm, 'step': step, 'budget': 4,
                        'state': {**{'prior.' + k: v for k, v in state.items()}, 'output.bias': torch.tensor([.001])},
                        'binding': {**binding, 'prior_sha256': digest}, 'config': {'max_delta': .35},
                        'initial_state_sha256': 'same', 'order_sha256': f'order{step}'}, point / (arm + '.pt'))
        write(point / 'pairing.json', {'initial_state_sha256': True, 'order_sha256': True, 'step': True})


def complete_mock(root):
    protocol = {'schema': 'bounded_audio_experiment_v1', 'milestones': [2, 4], 'budgets': [5, 10, 4],
                'outer_ids': ['outer' + str(i) for i in range(4)], 'default_replaced': False}
    write(root / 'protocol.json', protocol)
    write(root / 'status.json', {'state': 'complete', 'default_replaced': False})
    checkpoints(root, protocol)
    prior = archive(root / 'prior_generation/inner_validation', 'prior', 'inner_validation')
    records = []
    for step in (2, 4):
        point = root / f'point_{step}'
        audio = archive(point / 'audio_generation/inner_validation', 'audio', 'inner_validation')
        static = archive(point / 'audio_static/inner_validation', 'static', 'inner_validation')
        matched = archive(point / 'matched_static_generation/inner_validation', 'matched_static', 'inner_validation')
        value = audit(prior, audio, static, point / 'acceptance.json', matched_static_dir=matched)
        records.append({'step': step, 'accepted': value['accepted'], 'audit': str(point / 'acceptance.json')})
    passing = [row for row in records if row['accepted']]
    write(root / 'decision.json', {'records': records, 'chosen': passing[0] if passing else None,
                                  'selection_data': 'inner_validation only', 'default_replaced': False})
    mapping = {'prior': 'prior_generation', 'audio': 'audio_generation', 'matched_static': 'matched_static_generation',
               'static': 'audio_static', 'reverse': 'audio_reverse', 'mismatch': 'audio_mismatch'}
    for role, folder in mapping.items():
        archive(root / 'outer' / folder / 'holdout', role)
    return protocol


def test_full_summary_keeps_inner_and_outer_separate_without_model_promotion(tmp_path):
    complete_mock(tmp_path / 'run')
    value = s.summarize_run(tmp_path / 'run', tmp_path / 'summary')
    assert value['accepted_inner'] and value['selected_step'] == 2
    assert value['outer']['evaluated_step'] == 2 and not value['outer']['used_for_selection']
    assert value['checkpoint_checks']['all_saved_priors_exact']
    assert len(value['inner']) == len(value['checkpoint_checks']['points']) == 2
    assert not value['default_replaced'] and not value['naturalness_certified']
    delta = value['outer']['comparisons']['audio_minus_prior']['centered']
    assert delta['mean_delta'] < 0 and delta['ci95'][1] < 0
    assert (tmp_path / 'summary/summary.json').exists()
    markdown = (tmp_path / 'summary/summary.md').read_text(encoding='utf8')
    assert 'Inner validation' in markdown and 'Outer diagnostic' in markdown


@pytest.mark.parametrize('field', ['frozen', 'pairing'])
def test_checkpoint_mutation_is_detected(tmp_path, field):
    protocol = complete_mock(tmp_path / 'run')
    path = tmp_path / 'run/point_2/audio.pt'; checkpoint = s.load(path)
    if field == 'frozen':
        checkpoint['state']['prior.linear.bias'][0] += .1
    else:
        checkpoint['order_sha256'] = 'different'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='Frozen prior|pairing differs'):
        s.verify_checkpoints(tmp_path / 'run', protocol)


def test_outer_target_tampering_or_wrong_split_is_rejected(tmp_path):
    prior = archive(tmp_path / 'prior', 'prior'); audio = archive(tmp_path / 'audio', 'audio')
    p = s.load_outer_evaluation(prior, 'prior'); a = s.load_outer_evaluation(audio, 'audio')
    a['rows']['outer0']['target'] = a['rows']['outer0']['target'].copy()
    a['rows']['outer0']['target'][0, 0] += .1
    with pytest.raises(ValueError, match='target differs'):
        s.paired_comparison(p, a)
    inner = archive(tmp_path / 'inner', 'audio', 'inner_validation')
    with pytest.raises(ValueError, match='holdout'):
        s.load_outer_evaluation(inner, 'audio')


def test_recovery_pointer_supports_a_downloaded_copy(tmp_path):
    destination = archive(tmp_path / 'holdout_recovery1', 'prior')
    write(tmp_path / 'holdout_artifact.json', {'directory': '/not-present-remote/holdout_recovery1'})
    assert s.evaluation_directory(tmp_path / 'holdout') == destination


def test_changed_decision_or_saved_acceptance_is_rejected(tmp_path):
    root = tmp_path / 'run'; complete_mock(root)
    path = root / 'decision.json'; decision = s.read(path)
    decision['chosen'] = decision['records'][1]; write(path, decision)
    with pytest.raises(ValueError, match='earliest'):
        s.summarize_run(root)
    decision['chosen'] = decision['records'][0]; write(path, decision)
    path = root / 'point_2/acceptance.json'; acceptance = s.read(path)
    acceptance['accepted'] = not acceptance['accepted']; write(path, acceptance)
    with pytest.raises(ValueError, match='independent recomputation'):
        s.summarize_run(root)


def test_mismatch_subset_reports_only_paired_support(tmp_path):
    p = s.load_outer_evaluation(archive(tmp_path / 'p', 'prior'), 'prior')
    a = s.load_outer_evaluation(archive(tmp_path / 'a', 'audio'), 'audio')
    a['rows'].pop('outer0'); a['rows'].pop('outer1')
    result = s.paired_comparison(p, a, allow_subset=True)
    assert result['clip_count'] == 2 and result['centered']['ci95'] is None
    with pytest.raises(ValueError, match='membership'):
        s.paired_comparison(p, a)


def test_empty_mismatch_and_unbound_adapter_stats_are_explicit(tmp_path):
    path = archive(tmp_path / 'mismatch', 'mismatch')
    result = s.read(path / 'result.json')
    result.update(clips=0, scored_clips=0, per_clip_scores=[], summary=None, numerics=[],
                  numerical_gate={'passed': False})
    write(path / 'result.json', result)
    torch.save({'schema': s.EVALUATION_SCHEMA, 'clips': {}, 'result': result}, path / 'curves.pt')
    empty = s.load_outer_evaluation(path, 'mismatch')
    prior = s.load_outer_evaluation(archive(tmp_path / 'prior', 'prior'), 'prior')
    assert s.paired_comparison(prior, empty, allow_subset=True)['supported'] is False
    root = tmp_path / 'run'; protocol = complete_mock(root)
    checkpoint_path = root / 'point_2/audio.pt'; checkpoint = s.load(checkpoint_path)
    checkpoint['binding']['stats_sha256'] = 'changed'; torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match='stat/protocol binding'):
        s.verify_checkpoints(root, protocol)
