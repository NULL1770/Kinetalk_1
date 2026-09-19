import json

import numpy as np
import pytest
import torch

from scripts import audit_prior_audio_adapter as a
from scripts.joint_motion_metrics import score_clip, summarize


def _clips(kind, *, split='inner_validation'):
    rows = {}
    t = np.linspace(0, 4 * np.pi, 65)
    scale = np.linspace(.7, 1.2, 9)[None]
    target = .4 + .11 * np.sin(t)[:, None] * scale
    noise = np.array([-.012, -.004, .004, .012])[:, None, None] * np.cos(1.4 * t)[None, :, None] * scale
    for index in range(6):
        cid = f'clip_{index}'
        y = target + .003 * index
        samples = y[None] + noise
        if kind == 'prior':
            samples += .06 * np.cos(.7 * t)[None, :, None] * scale
        elif kind in ('static', 'matched_static'):
            samples += .035 * np.cos(.7 * t)[None, :, None] * scale
        elif kind == 'collapse':
            samples = np.repeat(y[None], 4, axis=0)
        elif kind == 'constant':
            samples = np.full((4, len(t), 9), .4)
        mask = np.ones(len(t), dtype=bool); mask[22:25] = False
        rows[cid] = {'metadata': {'clip_id': cid, 'split': split, 'sentence': f's{index // 2}',
                                 'speaker': index % 2, 'emotion': 0},
                     'samples': samples.copy(), 'target': y.copy(),
                     'native_valid': mask.copy(), 'score_mask': mask.copy(), 'generated_mask': mask.copy(),
                     'seeds': [42, 123, 2026, 77], 'raw_coefficient_prediction': True,
                     'run_records': [{'seed': s, 'noise_seed': s + index, 'run_index': 0} for s in (42, 123, 2026, 77)]}
    return rows


def _save(path, rows, role, *, scales=None, passed=True):
    path.mkdir(parents=True, exist_ok=True)
    scales = np.full(9, .2) if scales is None else scales
    metrics = []
    checks = []
    for cid, row in rows.items():
        metric = score_clip(row['samples'], row['target'], row['score_mask'], scales)
        metric.update(row['metadata']); metrics.append(metric)
        checks.append({'clip_id': cid, 'nonfinite_values': 0, 'other43_exact': True,
                       'native_invalid_exact_baseline': True, 'prediction_clipped': False})
    result = {'schema': a.EVALUATION_SCHEMA, 'mode': 'free_generation', 'clips': len(rows),
              'scored_clips': len(rows), 'per_clip_scores': metrics,
              'summary': summarize(metrics) if metrics else None,
              'numerical_gate': {'passed': passed}, 'numerics': checks,
              'steps': 24, 'use_audio': role != 'prior',
              'intervention': 'static' if role in ('static', 'matched_static') else 'real'}
    (path / 'result.json').write_text(json.dumps(result), encoding='utf8')
    torch.save({'schema': a.EVALUATION_SCHEMA, 'clips': rows, 'result': result}, path / 'curves.pt')
    return path


def _setup(tmp_path, audio_kind='audio'):
    return {role: _save(tmp_path / role, _clips(audio_kind if role == 'audio' else role), role)
            for role in ('prior', 'audio', 'static', 'matched_static')}


def _audit(paths, **kwargs):
    return a.audit(paths['prior'], paths['audio'], paths['static'], **kwargs)


def test_success_case_recomputes_scores_and_reports_auxiliary_matched_control(tmp_path):
    paths = _setup(tmp_path)
    result = _audit(paths, matched_static_dir=paths['matched_static'], output=tmp_path / 'audit.json')
    assert result['accepted'] and not result['failures']
    assert result['naturalness_certified'] is False and result['split'] == 'inner_validation'
    assert result['metrics']['spread_ratio_to_prior'] == pytest.approx(1.)
    interval = result['metrics']['comparisons']['audio_minus_prior']['centered']
    assert interval['ci95'][1] < 0 and interval['resamples'] == 4096
    assert interval['sentence_count'] == 3 and interval['clip_count'] == 6
    assert result['gate_contract']['matched_static_mean_improvement_required']
    assert result['metrics']['arms']['audio']['speed_ratio_to_reference'] == pytest.approx(1, abs=.02)
    assert len(result['per_clip']) == 6
    assert json.loads((tmp_path / 'audit.json').read_text())['accepted']


@pytest.mark.parametrize('kind', ['constant', 'collapse'])
def test_constant_or_seed_collapsed_output_cannot_pass(tmp_path, kind):
    result = _audit(_setup(tmp_path, kind))
    assert not result['accepted']
    assert any('diversity' in message for message in result['failures'])
    if kind == 'constant':
        assert any('amplitude' in message for message in result['failures'])
        assert any('speed' in message for message in result['failures'])


@pytest.mark.parametrize('field', ['target', 'score_mask', 'native_valid', 'seeds', 'run_records'])
def test_rejects_mismatched_targets_masks_and_seed_pairing(tmp_path, field):
    paths = _setup(tmp_path)
    rows = _clips('audio')
    if field == 'target':
        rows['clip_0'][field][0, 0] += .1
    elif field == 'score_mask':
        rows['clip_0'][field][0] = False
    elif field == 'native_valid':
        for key in ('native_valid', 'score_mask', 'generated_mask'):
            rows['clip_0'][key][0] = False
    elif field == 'seeds':
        for row in rows.values():
            row['seeds'].reverse()
    else:
        rows['clip_0'][field][0]['noise_seed'] += 1
    _save(paths['audio'], rows, 'audio')
    with pytest.raises(ValueError, match='differ'):
        _audit(paths)


@pytest.mark.parametrize('split', ['holdout', 'train', 'test', 'internal_holdout'])
def test_external_or_training_split_is_never_accepted(tmp_path, split):
    paths = _setup(tmp_path)
    _save(paths['audio'], _clips('audio', split=split), 'audio')
    with pytest.raises(ValueError, match='inner_validation'):
        _audit(paths)


def test_rejects_incomplete_membership_and_different_fit_scales(tmp_path):
    paths = _setup(tmp_path)
    rows = _clips('audio'); rows.pop('clip_0')
    _save(paths['audio'], rows, 'audio')
    with pytest.raises(ValueError, match='membership'):
        _audit(paths)
    _save(paths['audio'], _clips('audio'), 'audio', scales=np.ones(9))
    with pytest.raises(ValueError, match='scales differ'):
        _audit(paths)


def test_requires_four_unique_draws_and_multiple_sentences(tmp_path):
    paths = _setup(tmp_path)
    rows = _clips('audio')
    for row in rows.values():
        row['samples'] = row['samples'][:3]; row['seeds'] = row['seeds'][:3]
    _save(paths['audio'], rows, 'audio')
    with pytest.raises(ValueError, match='four or more'):
        _audit(paths)
    rows = _clips('audio')
    for row in rows.values():
        row['metadata']['sentence'] = 'same'
    _save(paths['audio'], rows, 'audio')
    with pytest.raises(ValueError, match='two internal validation sentences'):
        _audit(paths)


def test_cannot_pass_by_forging_saved_score_or_numerical_success(tmp_path):
    paths = _setup(tmp_path)
    path = paths['audio'] / 'result.json'
    value = json.loads(path.read_text()); value['per_clip_scores'][0]['joint_fair_es']['centered'] = 0.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='saved centered score'):
        _audit(paths)
    _save(paths['audio'], _clips('audio'), 'audio', passed=False)
    result = _audit(paths)
    assert not result['accepted'] and any('numerical' in message for message in result['failures'])


def test_static_or_matched_static_equality_blocks_acceptance(tmp_path):
    paths = _setup(tmp_path)
    _save(paths['matched_static'], _clips('audio'), 'matched_static')
    assert _audit(paths)['accepted']
    result = _audit(paths, matched_static_dir=paths['matched_static'])
    assert not result['accepted'] and any('matched_static' in message for message in result['failures'])
    _save(paths['static'], _clips('audio'), 'static')
    result = _audit(paths)
    assert not result['accepted'] and any('static mean' in message for message in result['failures'])


def test_sentence_bootstrap_is_paired_deterministic_and_not_clip_bootstrap():
    value = a.sentence_cluster_bootstrap([-1., -1., -1., .5], ['a', 'a', 'a', 'b'])
    assert value == a.sentence_cluster_bootstrap([-1., -1., -1., .5], ['a', 'a', 'a', 'b'])
    assert value['mean_delta'] == -.625
    assert value['ci95'] == [-1., .5]
    assert value['sentence_mean_deltas'] == {'a': -1., 'b': .5}


def test_negative_mean_without_sentence_cluster_confidence_is_rejected(tmp_path):
    paths = _setup(tmp_path)
    rows = _clips('audio'); t = np.linspace(0, 4 * np.pi, 65)
    for cid in ('clip_4', 'clip_5'):
        rows[cid]['samples'] += .085 * np.cos(.7 * t)[None, :, None] * np.linspace(.7, 1.2, 9)[None, None]
    _save(paths['audio'], rows, 'audio')
    result = _audit(paths)
    comparison = result['metrics']['comparisons']['audio_minus_prior']['centered']
    assert comparison['mean_delta'] < 0 and comparison['ci95'][1] > 0
    assert not result['accepted'] and any('CI upper' in failure for failure in result['failures'])


def test_spread_centers_each_observed_run_and_never_links_missing_gap():
    samples = np.zeros((4, 8, 9))
    mask = np.array([True, True, True, False, False, True, True, True])
    for k in range(4):
        samples[k, :3] = k * 100
        samples[k, 5:] = -k * 100
    samples[:, ~mask] = np.nan
    assert a.half_pair_spread(samples, mask, np.ones(9)) == 0.


def test_real_evaluator_archives_are_accepted_as_inputs_without_extra_fields(tmp_path):
    from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE, ContinuousLatentFlow
    from scripts.evaluate_continuous_motion_latent import UPPER, evaluate_generation

    torch.manual_seed(302)
    ae = ContinuousUpperAE(hidden=16, depth=1)
    flow = ContinuousLatentFlow(context_dim=3, hidden=16, depth=1)
    stats = {'residual_scale': torch.full((9,), .1), 'latent_mean': torch.zeros(16),
             'latent_scale': torch.ones(16), 'audio_mean': torch.zeros(1540),
             'audio_scale': torch.ones(1540), 'context_mean': torch.zeros(3),
             'context_scale': torch.ones(3)}
    clips = []
    for index in range(2):
        baseline = torch.full((13, 52), .4)
        target = baseline + torch.linspace(-.1, .1, 13)[:, None]
        clips.append({'clip_id': f'native_{index}', 'split': 'inner_validation',
                      'sentence': f's{index}', 'speaker': 0, 'emotion': 0,
                      'valid': torch.ones(13, dtype=torch.bool),
                      'motion_mask': torch.ones(13, 9, dtype=torch.bool),
                      'features': torch.randn(13, 1540), 'context': torch.zeros(3),
                      'b9': torch.full((9,), .4), 'baseline52': baseline,
                      'target52': target, 'motion9': target[:, UPPER]})
    for role in ('prior', 'audio', 'static'):
        evaluate_generation(flow, ae, clips, stats, tmp_path / role,
                            use_audio=role != 'prior',
                            intervention='static' if role == 'static' else 'real', steps=2)
    result = a.audit(tmp_path / 'prior', tmp_path / 'audio', tmp_path / 'static')
    assert result['clip_count'] == 2 and result['sentence_count'] == 2
    assert not result['naturalness_certified']
