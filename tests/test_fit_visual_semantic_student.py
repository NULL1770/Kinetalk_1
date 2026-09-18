import copy

import numpy as np
import pytest
import torch

from scripts import fit_visual_semantic_student as s


def clip(index, split='train', sentence=None):
    generator = torch.Generator().manual_seed(index+500)
    features = torch.randn(15, 1540, generator=generator)
    valid = torch.ones(15, dtype=torch.bool)
    values = features[:, 768:770].tanh()
    posterior = torch.softmax(features[:, 770:778], -1)
    return {'clip_id': f'clip_{index}', 'sentence': sentence or f'sentence_{index}', 'split': split,
            'features': features, 'valid': valid, 'semantic_valid': valid.clone(),
            'va': values, 'posterior': posterior}


@pytest.fixture(scope='module')
def fitted():
    clips = [clip(i) for i in range(6)]+[clip(7, 'holdout')]
    return clips, s.fit_dataset({'clips': clips})


def test_three_fold_oof_has_sentence_disjoint_preprocessing_and_training(fitted):
    clips, (predictions, models, scores) = fitted
    for original, predicted in zip(clips, predictions['clips']):
        if original['split'] == 'train':
            model = models['oof'][predicted['fold']]
            assert original['sentence'] not in model['fit_sentences']
            assert original['clip_id'] not in model['fit_clip_ids']
            assert predicted['prediction_source'] == 'sentence_oof'
        else:
            assert predicted['prediction_source'] == 'all_train'
        for name, width in [('va', 2), ('posterior', 8)]:
            for mode in s.MODES:
                value = predicted[name][mode]
                assert value.shape == (15, width) and torch.isfinite(value).all()
        assert torch.all(predicted['va']['actual'].abs() <= 1)
        torch.testing.assert_close(predicted['posterior']['actual'].sum(-1), torch.ones(15))
    assert scores['train']['va']['actual']['clip_count'] == 6
    assert models['all_train']['heads']['actual']['alpha'] == 100.


def test_predict_audio_requires_no_targets_and_is_immune_to_target_changes(fitted):
    clips, (_, models, _) = fitted
    target = copy.deepcopy(clips[-1])
    a = s.predict_audio(models['all_train'], target)
    target['va'][:] = float('nan'); target['posterior'][:] = float('nan')
    target['semantic_valid'][:] = False
    b = s.predict_audio(models['all_train'], target)
    c = s.predict_audio(models['all_train'], {key: target[key] for key in ('features', 'valid')})
    for name in ('va', 'posterior'):
        for mode in s.MODES:
            assert torch.equal(a[name][mode], b[name][mode]) and torch.equal(a[name][mode], c[name][mode])


def test_window_context_and_reverse_do_not_cross_missing_run():
    c = clip(0); c['valid'][5:8] = False; c['features'][5:8] = float('nan')
    w = s.audio_windows(c)
    assert w['spans'] == [(0, 5), (8, 13), (13, 15)]
    preprocessing = {'mean': np.zeros(772), 'scale': np.ones(772), 'components': np.eye(772, 2)}
    actual = s._design(w, preprocessing, 'actual')
    reverse = s._design(w, preprocessing, 'reverse')
    np.testing.assert_allclose(actual[0, :2], actual[0, 2:4])
    np.testing.assert_allclose(actual[0, 4:], actual[0, 2:4])
    np.testing.assert_allclose(actual[1, :2], actual[1, 2:4])
    np.testing.assert_allclose(reverse[0], actual[0])
    np.testing.assert_allclose(reverse[1, 2:4], actual[2, 2:4])


def test_static_is_deployable_audio_mean_not_target_mean(fitted):
    clips, (_, models, _) = fitted
    predicted = s.predict_audio(models['all_train'], clips[-1])
    for name in ('va', 'posterior'):
        static = predicted[name]['static']
        torch.testing.assert_close(static, static[:1].expand_as(static), rtol=0, atol=0)


def test_pca_and_scaling_are_fitted_only_on_declared_training_clips(fitted):
    clips, (_, models, _) = fitted
    for model in [models['all_train'], *models['oof'].values()]:
        fitted_clips = [c for c in clips if c['clip_id'] in model['fit_clip_ids']]
        windows = np.concatenate([s.audio_windows(c)['values'] for c in fitted_clips])
        np.testing.assert_allclose(model['preprocessing']['mean'], windows.mean(0), rtol=0, atol=0)
        np.testing.assert_allclose(model['preprocessing']['scale'], windows.std(0).clip(1e-6), rtol=0, atol=0)
        assert clips[-1]['clip_id'] not in model['fit_clip_ids']


def test_invalid_padding_is_zero_and_never_a_softmax_uniform_fallback(fitted):
    clips, (_, models, _) = fitted
    c = copy.deepcopy(clips[-1]); c['valid'][-4:] = False; c['features'][-4:] = float('nan')
    prediction = s.predict_audio(models['all_train'], c)
    for name in ('va', 'posterior'):
        for mode in s.MODES:
            assert torch.count_nonzero(prediction[name][mode][-4:]) == 0


def test_holdout_sentence_overlap_rejected_before_fitting():
    clips = [clip(i) for i in range(3)]+[clip(4, 'holdout', sentence='sentence_1')]
    with pytest.raises(ValueError, match='sentence-disjoint'):
        s.fit_dataset({'clips': clips})


def test_bad_semantic_probabilities_are_not_silently_normalized():
    c = clip(0); c['posterior'][0, 0] = -1
    with pytest.raises(ValueError, match='normalized posterior'):
        s._targets(c, s.audio_windows(c))


def test_invalid_holdout_teacher_mask_is_rejected_before_model_fit():
    clips = [clip(i) for i in range(3)]+[clip(4, 'holdout')]
    clips[-1]['valid'][0] = False
    with pytest.raises(ValueError, match='normalized posterior'):
        s.fit_dataset({'clips': clips})


def test_static_target_prediction_cannot_claim_dynamic_success():
    c = clip(1)
    truth = c['va']; static = truth.mean(0, keepdim=True).expand_as(truth)
    prediction = {'va': {mode: static for mode in s.MODES},
                  'posterior': {mode: c['posterior'].mean(0, keepdim=True).expand_as(c['posterior']) for mode in s.MODES}}
    score = s._scores([prediction], [c])
    assert score['train']['va']['actual']['clip_equal_centered'] > 0
