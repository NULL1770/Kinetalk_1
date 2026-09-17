"""Cross-identity analysis must retain failures and real paired baselines."""
import copy
from pathlib import Path

import pytest
import torch

from scripts.audit_cross_identity_projection import audit_cross_curves, groups_of_clips, lowrate_head_audit, validate_curve_sidecar
from scripts.train_predictable_renderer import basic_metrics


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_grouping_keeps_every_identity_and_emotion_and_excludes_left_out_person():
    query = {'emotion_id': torch.tensor([0, 1, 0, 1, 0, 1]), 'speaker_id': torch.tensor([4, 4, 5, 5, 6, 6])}
    groups = groups_of_clips(query, {'a': 4, 'b': 5, 'c': 6}, ['neutral', 'angry'])
    assert groups['pooled/nonneutral'] == [1, 3, 5]
    assert groups['leave_out/a/nonneutral'] == [3, 5]
    assert groups['speaker/b/neutral'] == [2]
    assert groups['emotion/angry'] == [1, 3, 5]
    with pytest.raises(ValueError, match='identities'):
        groups_of_clips(query, {'a': 4, 'b': 5}, ['neutral', 'angry'])


def test_original_curve_sidecar_cannot_be_relabelled_as_selected_checkpoint(tmp_path):
    schema = 'formal_predictable_projection_curves_provenance_v1'
    path = tmp_path / 'original_curves.provenance.json'
    record = {'schema': schema, 'path': str(path), 'sha256': 'sidecar'}
    sidecar = {'schema': schema, 'curve_sha256': 'curve', 'recipe_sha256': 'recipe',
               'selected_checkpoint_sha256': 'original-checkpoint', 'cache_sha256': 'cross'}
    args = dict(path=path, actual_sha='sidecar', curve_sha='curve', recipe_sha='recipe', cache_sha='cross')
    validate_curve_sidecar(record, sidecar, checkpoint_sha='original-checkpoint', **args)
    with pytest.raises(ValueError, match='provenance'):
        validate_curve_sidecar(record, sidecar, checkpoint_sha='selected-adapter', **args)


def test_brow_failure_is_preserved_despite_positive_upper_and_real_original_pairing():
    torch.manual_seed(14)
    target = torch.randn(12, 8, 52) * .1
    target[..., 41:46] *= .2  # weak brows cannot be hidden by stronger eyes
    speakers = torch.tensor([4] * 4 + [5] * 4 + [6] * 4)
    labels = torch.tensor([0, 1, 1, 1] * 3)
    split = {'q': {'motion': target, 'valid': torch.ones(12, 8, dtype=torch.bool),
        'channel_mask': torch.ones(12, 52, dtype=torch.bool), 'emotion_id': labels, 'speaker_id': speakers,
        'sentence_id': ['n', 'a', 'b', 'c'] * 3, 'times': torch.arange(8).double()[None].expand(12, -1) / 25},
        'base': {'b0': torch.zeros_like(target)}, 'identity': {'baseline': torch.zeros(12, 52)}}
    full = target * .8; full[..., 41:46] = 0
    zero = target * .5
    modes = {'full': full, 'zero': zero, 'reverse': full.flip(1), 'oracle': target}
    original = target * .25
    seeds = (42, 123, 2026)
    selected_curves = {'motion': {str(seed): modes for seed in seeds}}
    original_curves = {'motion': {str(seed): {'full': original} for seed in seeds}}
    summary = {'before': {}, 'after': {}}
    for seed in seeds:
        summary['before'][str(seed)] = {'full': {**basic_metrics(original, split), 'frozen_teacher_emotion_accuracy': .9}}
        summary['after'][str(seed)] = {mode: {**basic_metrics(value, split), 'frozen_teacher_emotion_accuracy': .9} for mode, value in modes.items()}
    payload = {'split': split, 'provenance': {'speaker_to_id': {'a': 4, 'b': 5, 'c': 6}}}
    report = audit_cross_curves(selected_curves, original_curves, payload, summary, ['neutral', 'angry'], samples=100)
    assert report['transfer_diagnostics']['upper_expression']['pooled_delta_r2'] > 0
    assert report['transfer_diagnostics']['brows']['pooled_delta_r2'] < 0
    assert report['transfer_diagnostics']['brows']['positive_identity_count'] == 0
    assert not report['same_engineering_checks']['brows_tolerated']
    assert not report['cross_identity_engineering_pass']
    assert 'original' in report['paired_full_vs']['pooled/nonneutral']
    assert report['paired_full_vs']['pooled/nonneutral']['original']['centered_residual']['upper_expression']['r2_improvement_ci95'] is not None


def test_lowrate_projection_oracle_is_real_motion_and_separate_from_audio():
    torch.manual_seed(3)
    basis = torch.zeros(52, 8); basis[torch.arange(5, 13), torch.arange(8)] = 1
    y = torch.randn(6, 3, 52)
    y[..., :5] = 0; y[..., 13:] = 0
    q = {'emotion_id': torch.tensor([0, 1] * 3), 'speaker_id': torch.tensor([4, 4, 5, 5, 6, 6]),
         'sentence_id': ['a', 'b'] * 3}
    features = torch.zeros(6, 3, 2)
    payload = {'split': {'q': q, 'affect': {'emotion_logits': torch.tensor([[2., 0.], [0., 2.]] * 3)}},
        'provenance': {'speaker_to_id': {'a': 4, 'b': 5, 'c': 6}},
        'bundle': {'motion_bins': y, 'weight': torch.full((6, 3), 4.),
                   'features': {'content': features, 'middle': features, 'prosody': features}}}
    head = {'basis': basis, 'channels': torch.arange(52), 'feature_std': torch.ones(6),
            'target_scale': torch.ones(8), 'linear.weight': torch.zeros(8, 6)}
    report = lowrate_head_audit(payload, head, ['neutral', 'angry'], samples=100)
    score = report['scores']['pooled/nonneutral']
    assert score['motion_projection_oracle']['upper_expression']['r2_against_zero'] == pytest.approx(1)
    assert score['audio_prediction']['upper_expression']['r2_against_zero'] == pytest.approx(0)
    assert 'not numerically comparable' in report['scope']
