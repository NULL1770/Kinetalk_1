"""Formal audit contracts: saved curves stay distinct from original points."""
import copy

import pytest
import torch

from scripts.audit_formal_projection import (
    MODES, SEEDS, assert_metric_agreement, audit_curves, validate_curve_metadata, validate_curve_binding,
)
from scripts.train_predictable_renderer import basic_metrics


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(3)
    target = torch.randn(6, 8, 52) * .1
    valid = torch.ones(6, 8, dtype=torch.bool)
    ref = {'q': {'motion': target, 'valid': valid,
        'channel_mask': torch.ones(6, 52, dtype=torch.bool),
        'emotion_id': torch.tensor([0, 0, 1, 1, 1, 1]),
        'times': torch.arange(8).double()[None].expand(6, -1) / 25,
        'sentence_id': ['a', 'b', 'c', 'd', 'c', 'd']},
        'base': {'b0': torch.zeros_like(target)}, 'identity': {'baseline': torch.zeros(6, 52)}}
    modes = {'full': target * .7, 'zero': target * .5, 'reverse': target.flip(1) * .7, 'oracle': target * .9}
    curves = {'noise_seeds': list(SEEDS), 'decode_steps': 12,
              'motion': {str(seed): {mode: value.clone() for mode, value in modes.items()} for seed in SEEDS}}
    summary = {'after': {str(seed): {mode: {**basic_metrics(value, ref), 'frozen_teacher_emotion_accuracy': .9}
        for mode, value in modes.items()} for seed in SEEDS}}
    original = {str(seed): {'full': {**basic_metrics(target * .4, ref), 'frozen_teacher_emotion_accuracy': .9}} for seed in SEEDS}
    return ref, curves, summary, original


def test_formal_curve_metadata_rejects_seed_mode_or_clock_changes():
    ref, curves, _, _ = fixture()
    recipe = {'noise_seeds': list(SEEDS), 'args': {'decode_steps': 12}}
    validate_curve_metadata(curves, recipe, ref['q']['motion'].shape)
    changed = copy.deepcopy(curves); changed['decode_steps'] = 4
    with pytest.raises(ValueError, match='twelve'):
        validate_curve_metadata(changed, recipe, ref['q']['motion'].shape)
    changed = copy.deepcopy(curves); del changed['motion']['42']['oracle']
    with pytest.raises(ValueError, match='intervention'):
        validate_curve_metadata(changed, recipe, ref['q']['motion'].shape)


def test_formal_audit_keeps_original_as_point_reference_only_and_rechecks_metrics():
    ref, curves, summary, original = fixture()
    report = audit_curves(curves, ref, summary, original, samples=100)
    assert set(report['scores']) == set(MODES)
    assert 'original' not in report['paired_dynamic']
    assert 'original_saved_point_reference' in report
    assert report['checks']['upper_useful']
    assert report['checks']['neutral_mouth_one_sided_90_within_3pct']
    assert report['teacher_accuracy_delta'] == 0
    assert report['paired_dynamic']['zero']['nonneutral']['upper_expression']['r2_improvement'] > 0
    # A valid-shaped but mismatched curve must fail against the saved report.
    changed = copy.deepcopy(curves); changed['motion']['42']['full'] += .01
    with pytest.raises(ValueError, match='Curve/summary'):
        audit_curves(changed, ref, summary, original, samples=100)


def test_formal_audit_does_not_reconstruct_original_uncertainty():
    ref, curves, summary, original = fixture()
    for seed in SEEDS:
        original[str(seed)]['full']['all']['mouth']['raw_motion']['pooled_centered_correlation'] = 1.
    report = audit_curves(curves, ref, summary, original, samples=100)
    assert 'no paired original uncertainty' in report['original_reference_note']
    assert report['mouth_correlation_delta']['original_saved_point']['all'] == pytest.approx(0, abs=1e-6)


def test_formal_curve_binding_rejects_missing_or_cross_checkpoint_provenance(tmp_path):
    sidecar = tmp_path / 'selected_development_curves.provenance.json'
    schema = 'formal_predictable_projection_curves_provenance_v1'
    summary = {'selected_checkpoint_sha256': 'selected', 'selected_development_curves_provenance':
               {'schema': schema, 'path': str(sidecar), 'sha256': 'sidecar'}}
    binding = {'schema': schema, 'curve_sha256': 'curves', 'recipe_sha256': 'recipe',
               'selected_checkpoint_sha256': 'selected', 'cache_sha256': 'cache'}
    kwargs = dict(sidecar_path=sidecar, sidecar_sha256='sidecar', curve_sha256='curves',
                  recipe_sha256='recipe', cache_sha256='cache')
    validate_curve_binding(summary, binding, **kwargs)
    changed = copy.deepcopy(binding); changed['selected_checkpoint_sha256'] = 'another-epoch'
    with pytest.raises(ValueError, match='sidecar differs'):
        validate_curve_binding(summary, changed, **kwargs)
    changed = copy.deepcopy(summary); changed['selected_development_curves_provenance'] = None
    with pytest.raises(ValueError, match='required'):
        validate_curve_binding(changed, binding, **kwargs)
