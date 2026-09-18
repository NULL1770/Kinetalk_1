"""Independent contract checks for the temporal semantic student.

Synthetic acoustic/teacher inputs avoid any experiment dataset and deliberately
exercise short tail windows, teacher missingness, native gaps and wrong folds.
"""
import copy

import numpy as np
import pytest
import torch

from scripts import fit_temporal_semantic_student as temporal
from scripts import fit_visual_semantic_student as ridge


def make_clip(index=0, frames=12, split='train'):
    rng = np.random.default_rng(100 + index)
    features = rng.normal(size=(frames, 1540)).astype(np.float32)
    va = np.tanh(features[:, 768:770]).astype(np.float32)
    logits = features[:, 770:778].astype(np.float64)
    posterior = np.exp(logits - logits.max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)
    return {'clip_id': f'clip_{index}', 'sentence': f'sentence_{index}', 'split': split,
            'features': features, 'valid': np.ones(frames, bool),
            'semantic_valid': np.ones(frames, bool), 'va': va,
            'posterior': posterior.astype(np.float32)}


def baseline_for(clips):
    preprocessing = {'mean': np.zeros(772), 'scale': np.ones(772),
                     'components': np.eye(772, 2)}
    coefficient = np.zeros((6, 10)); coefficient[0, 0] = .1
    coefficient[1, 1] = .1; coefficient[0, 4] = .25
    head = {'coefficient': coefficient, 'intercept': np.linspace(-.2, .2, 10)}
    return {'schema': ridge.SCHEMA, 'preprocessing': preprocessing,
            'heads': {'actual': copy.deepcopy(head), 'static': copy.deepcopy(head)},
            'fit_clip_ids': [c['clip_id'] for c in clips],
            'fit_sentences': sorted({c['sentence'] for c in clips})}


@pytest.fixture(scope='module')
def trained():
    old_threads = torch.get_num_threads(); torch.set_num_threads(2)
    clips = [make_clip(0), make_clip(1, frames=17)]
    baseline = baseline_for(clips)
    model = temporal.fit_model(clips, baseline, epochs=1, batch_size=2)
    yield clips, baseline, model
    torch.set_num_threads(old_threads)


class AcousticOnly(dict):
    def __getitem__(self, key):
        if key not in ('features', 'valid'):
            raise AssertionError(f'deployment accessed forbidden key {key}')
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in ('features', 'valid'):
            raise AssertionError(f'deployment accessed forbidden key {key}')
        return super().get(key, default)


def test_deployment_reads_only_acoustics_and_preserves_exact_old_static(trained):
    _, baseline, model = trained
    clip = make_clip(10, frames=19)
    clip['valid'][7:10] = False
    clip['features'][7:10] = np.nan
    guarded = AcousticOnly(clip)
    prediction = temporal.predict_audio(model, guarded)
    old = ridge.predict_audio(baseline, guarded)
    for field in ('va', 'posterior'):
        assert torch.equal(prediction[field]['static'], old[field]['static'])
        for mode in ridge.MODES:
            assert torch.isfinite(prediction[field][mode]).all()
            assert torch.count_nonzero(prediction[field][mode][7:10]) == 0
    np.testing.assert_allclose(prediction['posterior']['actual'][clip['valid']].sum(-1), 1., atol=1e-6)


def test_acoustic_centering_uses_native_frame_counts_not_equal_windows():
    clip = make_clip(frames=7)
    clip['features'][:, 768:770] = np.array([0., 2.])
    clip['features'][5:, 768:770] = np.array([7., -5.])
    item = temporal.audio_input(AcousticOnly(clip), baseline_for([clip])['preprocessing'])
    np.testing.assert_array_equal(item['weights'], [5., 2.])
    np.testing.assert_allclose(item['features'][:, :2], [[0., 2.], [7., -5.]])
    # The native-frame mean is [2, 0], not the equally weighted window mean.
    np.testing.assert_allclose(item['features'][:, 2:], [[-2., 2.], [5., -5.]])
    np.testing.assert_allclose(np.average(item['features'][:, 2:], axis=0, weights=item['weights']), 0.)


def test_constant_audio_is_native_mean_with_no_centered_input_signal():
    clip = make_clip(frames=7)
    clip['features'][:, 768:770] = [0., 2.]
    clip['features'][5:, 768:770] = [7., -5.]
    item = temporal.audio_input(AcousticOnly(clip), baseline_for([clip])['preprocessing'], constant_audio=True)
    np.testing.assert_allclose(item['features'][:, :2], [[2., 0.], [2., 0.]])
    np.testing.assert_allclose(item['features'][:, 2:], 0., atol=1e-7)
    with pytest.raises(ValueError, match='intervention'):
        temporal.audio_input(clip, baseline_for([clip])['preprocessing'], constant_audio=True, reverse=True)


def test_teacher_centering_uses_observed_counts_and_missing_windows_are_unsupervised():
    clip = make_clip(frames=12)
    clip['semantic_valid'][:] = False
    clip['semantic_valid'][[0, 5, 6]] = True
    clip['va'][:] = np.nan; clip['posterior'][:] = np.nan
    clip['va'][0] = [.2, .4]; clip['va'][[5, 6]] = [.8, -.2]
    clip['posterior'][[0, 5, 6]] = .125
    windows = ridge.audio_windows(clip)
    target = temporal.training_target(clip, windows)
    np.testing.assert_array_equal(target['counts'], [1., 2., 0.])
    np.testing.assert_array_equal(target['known'], [True, True, False])
    np.testing.assert_allclose(target['teacher_mean'][:2], [.6, 0.], atol=1e-7)
    np.testing.assert_allclose(target['target'][:, :2], [[-.4, .4], [.2, -.2], [0., 0.]], atol=1e-7)
    np.testing.assert_allclose(np.average(target['target'], axis=0, weights=target['counts']), 0., atol=1e-7)
    assert np.isfinite(target['target']).all()


def test_teacher_absence_is_rejected_in_training_only():
    clip = make_clip(); clip['semantic_valid'][:] = False
    with pytest.raises(ValueError, match='observed semantic window'):
        temporal.training_target(clip, ridge.audio_windows(clip))
    # Acoustic generation support must remain present even without a teacher.
    assert len(temporal.audio_input(AcousticOnly(clip), baseline_for([clip])['preprocessing'])['weights']) == 3


def test_reverse_preserves_old_window_intervention_and_never_crosses_a_gap():
    clip = make_clip(frames=19)
    clip['valid'][7:10] = False; clip['features'][7:10] = np.nan
    preprocessing = baseline_for([clip])['preprocessing']
    actual = temporal.audio_input(clip, preprocessing)
    reverse = temporal.audio_input(clip, preprocessing, reverse=True)
    np.testing.assert_array_equal(actual['weights'], [5., 2., 5., 4.])
    np.testing.assert_array_equal(actual['run_ids'], [0, 0, 1, 1])
    np.testing.assert_allclose(reverse['features'][:, :2], actual['features'][[1, 0, 3, 2], :2])
    np.testing.assert_array_equal(reverse['weights'], actual['weights'])
    assert reverse['windows']['spans'] == actual['windows']['spans']


def test_local_convolution_cannot_read_another_native_run_and_centering_is_weighted():
    torch.manual_seed(7)
    net = temporal.TemporalResidualTCN(4, hidden=8)
    x = torch.randn(1, 6, 4)
    valid = torch.tensor([[True, True, True, True, False, False]])
    run_ids = torch.tensor([[0, 0, 1, 1, -1, -1]])
    weights = torch.tensor([[5., 2., 5., 4., 0., 0.]])
    x[:, 4:] = float('nan')
    before = net.local_field(x, valid, run_ids)
    changed = x.clone(); changed[:, 2:4] += 50.
    after = net.local_field(changed, valid, run_ids)
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    residual = net(x, valid, run_ids, weights)
    torch.testing.assert_close((residual * weights[..., None]).sum(1), torch.zeros(1, 10), atol=1e-6, rtol=0)
    assert torch.count_nonzero(residual[:, 4:]) == 0
    # Clip centering deliberately introduces a common acoustic level shift;
    # only the uncentered local field has the strict cross-run independence.


def test_loss_is_equal_clips_and_observed_frames_within_each_clip():
    prediction = torch.zeros(2, 3, 10)
    target = torch.tensor([[1., 3., 0.], [2., 0., 0.]])[..., None].expand(2, 3, 10)
    counts = torch.tensor([[1., 2., 0.], [9., 0., 0.]])
    value = temporal.normalized_residual_loss(prediction, target, counts, torch.ones(10))
    assert float(value) == pytest.approx(((1. + 18.) / 3. + 4.) / 2.)


def test_wrong_oof_baseline_is_rejected_before_training():
    clips = [make_clip(i) for i in range(3)] + [make_clip(9, split='holdout')]
    folds = {f'sentence_{i}': i for i in range(3)}
    rows = [{'clip_id': c['clip_id'], 'sentence': c['sentence'], 'split': c['split'],
             'fold': folds.get(c['sentence']), 'audio_valid': c['valid'].copy(),
             'prediction_source': 'sentence_oof' if c['split'] == 'train' else 'all_train'} for c in clips]
    baseline = {'all_train': baseline_for(clips[:3]),
                'oof': {fold: baseline_for([c for c in clips[:3] if folds[c['sentence']] != fold]) for fold in range(3)}}
    temporal.validate_baseline({'clips': clips}, {'clips': rows, 'fold_by_sentence': folds}, baseline)
    baseline['oof'][0] = copy.deepcopy(baseline['all_train'])
    with pytest.raises(ValueError, match='membership'):
        temporal.validate_baseline({'clips': clips}, {'clips': rows, 'fold_by_sentence': folds}, baseline)


def test_same_seed_fit_repeats_and_baseline_parameters_stay_frozen(trained):
    clips, baseline, first = trained
    frozen = copy.deepcopy(baseline)
    second = temporal.fit_model(clips, baseline, epochs=1, batch_size=2)
    assert first['order_sha256'] == second['order_sha256']
    for key in first['state']:
        torch.testing.assert_close(first['state'][key], second['state'][key], rtol=0, atol=0)
    for key in ('mean', 'scale', 'components'):
        np.testing.assert_array_equal(baseline['preprocessing'][key], frozen['preprocessing'][key])
    for key in ('coefficient', 'intercept'):
        np.testing.assert_array_equal(baseline['heads']['static'][key], frozen['heads']['static'][key])
