"""Independent synthetic checks for the six-arm visual-semantic pilot."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts import build_visual_semantic_dataset as builder
from scripts import train_visual_semantic_pilot as train
from kinetalk_b0.models.semantic_upper_flow import SemanticUpperFlow


def sample(index, split='train'):
    generator = torch.Generator().manual_seed(802+index)
    frames = 30
    valid = torch.ones(frames, dtype=torch.bool); valid[-2:] = False
    semantic_valid = valid.clone(); semantic_valid[-3] = False
    return {'clip_id': f'clip_{index}', 'sentence': f'sentence_{index}', 'split': split,
            'speaker': index % 2, 'speaker_name': 'speaker_'+str(index % 2), 'emotion': index % 2,
            'times': torch.arange(frames, dtype=torch.float64)/25,
            'features': torch.randn(frames, 1540, generator=generator),
            'valid': valid, 'semantic_valid': semantic_valid,
            'motion': .1+.8*torch.rand(frames, 52, generator=generator),
            'baseline52': .1+.8*torch.rand(frames, 52, generator=generator),
            'va': torch.rand(frames, 2, generator=generator)*2-1,
            'posterior': torch.softmax(torch.randn(frames, 8, generator=generator), -1),
            'affect_global': torch.randn(4, generator=generator),
            'identity_code': torch.randn(3, generator=generator)}


def student(clip):
    generator = torch.Generator().manual_seed(310+int(clip['clip_id'].split('_')[-1]))
    frames = len(clip['valid'])
    va = torch.rand(frames, 2, generator=generator)*2-1
    posterior = torch.softmax(torch.randn(frames, 8, generator=generator), -1)
    return {**{key: clip[key] for key in ('clip_id', 'sentence', 'split')},
            'prediction_source': 'sentence_oof' if clip['split'] == 'train' else 'all_train',
            'audio_valid': clip['valid'].clone(), 'semantic_valid': clip['semantic_valid'].clone(),
            'va': {'actual': va, 'static': va.mean(0, keepdim=True).expand_as(va).clone(), 'reverse': va.flip(0)},
            'posterior': {'actual': posterior, 'static': posterior.mean(0, keepdim=True).expand_as(posterior).clone(),
                          'reverse': posterior.flip(0)}}


def inputs():
    clips = [sample(0), sample(1), sample(2, 'holdout')]
    return clips, [student(c) for c in clips]


def test_all_fit_statistics_ignore_holdout_motion_semantics_and_identity():
    clips, _ = inputs(); before = train.stats(clips)
    altered = copy.deepcopy(clips)
    for name in ('motion', 'va', 'posterior', 'affect_global', 'identity_code'):
        altered[-1][name][:] = float('nan')
    after = train.stats(altered)
    assert before.keys() == after.keys()
    for key in before: torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_six_arms_share_motion_support_global_identity_and_eight_dim_interface():
    clips, predictions = inputs(); scales = train.stats(clips)
    batches = {arm: train.batch(clips, predictions, [0, 1], arm, scales, 'cpu') for arm in train.ARMS}
    source = batches['va_oracle']
    for arm, values in batches.items():
        assert values[2].shape == (2, 30, 8)
        for index in (0, 1, 3, 4, 5):
            torch.testing.assert_close(source[index], values[index], rtol=0, atol=0)
        assert torch.count_nonzero(values[2][~values[1]]) == 0
        if arm.startswith('va_'): assert torch.count_nonzero(values[2][..., 2:]) == 0
    assert not torch.equal(batches['va_audio'][2], batches['va_static'][2])
    assert not torch.equal(batches['posterior_audio'][2], batches['posterior_static'][2])


def test_audio_conditions_never_read_visual_values_or_motion_targets():
    clips, predictions = inputs(); scales = train.stats(clips)
    altered = copy.deepcopy(clips)
    for c in altered:
        c['motion'][:] = float('nan'); c['va'][:] = float('nan'); c['posterior'][:] = float('nan')
    for arm in ('va_audio', 'va_static', 'posterior_audio', 'posterior_static'):
        torch.testing.assert_close(train.semantic(clips[0], predictions[0], arm, scales),
                                   train.semantic(altered[0], predictions[0], arm, scales), rtol=0, atol=0)


def test_same_seed_evaluation_repeats_native_samples_and_preserves_unsupported_base():
    clips, predictions = inputs(); scales = train.stats(clips)
    torch.manual_seed(924)
    model = SemanticUpperFlow(8, global_dim=4, identity_dim=3, hidden=16, depth=2)
    _, a = train.evaluate(model, clips, predictions, [2], 'va_audio', scales, 'cpu', steps=2)
    _, b = train.evaluate(model, clips, predictions, [2], 'va_audio', scales, 'cpu', steps=2)
    c = clips[2]; mask = c['valid'].numpy()
    assert np.array_equal(a[c['clip_id']], b[c['clip_id']])
    samples = a[c['clip_id']]
    assert samples.shape == (4, 30, 9)
    assert ((samples[:, mask] > 0) & (samples[:, mask] < 1)).all()
    np.testing.assert_array_equal(samples[:, ~mask], np.broadcast_to(c['baseline52'][~torch.from_numpy(mask)][:, train.UPPER].numpy(), samples[:, ~mask].shape))
    # Compose the documented output, then independently check untouched values.
    baseline = c['baseline52'].numpy()
    full = np.broadcast_to(baseline, (4, *baseline.shape)).copy()
    full[:, :, train.UPPER] = samples
    protected = [k for k in range(52) if k not in train.UPPER]
    np.testing.assert_array_equal(full[:, :, protected], np.broadcast_to(baseline[:, protected], full[:, :, protected].shape))
    np.testing.assert_array_equal(full[:, ~mask], np.broadcast_to(baseline[~mask], full[:, ~mask].shape))


def test_audio_generation_is_independent_of_visual_missingness():
    clips, predictions = inputs(); scales = train.stats(clips)
    changed = copy.deepcopy(clips)
    changed[2]['semantic_valid'][5:15] = False
    changed[2]['va'][5:15] = float('nan'); changed[2]['posterior'][5:15] = float('nan')
    for arm in ('va_audio', 'va_static', 'posterior_audio', 'posterior_static'):
        a = train.batch(clips, predictions, [2], arm, scales, 'cpu')
        b = train.batch(changed, predictions, [2], arm, scales, 'cpu')
        for index in range(5):torch.testing.assert_close(a[index], b[index], rtol=0, atol=0)
        assert not torch.equal(a[5], b[5])
    torch.manual_seed(924)
    model = SemanticUpperFlow(8, global_dim=4, identity_dim=3, hidden=16, depth=2)
    _, a = train.evaluate(model, clips, predictions, [2], 'va_audio', scales, 'cpu', steps=2)
    _, b = train.evaluate(model, changed, predictions, [2], 'va_audio', scales, 'cpu', steps=2)
    np.testing.assert_array_equal(a['clip_2'], b['clip_2'])
    missing = clips[2]['valid'] & ~clips[2]['semantic_valid']
    assert not np.array_equal(a['clip_2'][0, missing], clips[2]['baseline52'][missing][:, train.UPPER].numpy())


def test_oracle_matches_student_window_expansion_with_explicit_oracle_fallback():
    clips, predictions = inputs(); scales = train.stats(clips)
    c = clips[0]; c['semantic_valid'][5:10] = False
    for kind, width in [('va', 2), ('posterior', 8)]:
        result = train.semantic(c, predictions[0], kind+'_oracle', scales)[:, :width]
        result = result*scales[kind+'_std']+scales[kind+'_mean']
        common = c['valid'] & c['semantic_valid']
        torch.testing.assert_close(result[0:5], c[kind][:5].mean(0).expand(5, -1))
        torch.testing.assert_close(result[5:10], c[kind][common].mean(0).expand(5, -1))


def test_visual_interpolation_does_not_extrapolate_or_cross_missing_frames():
    sample_times = np.array([0., .2, .4, .6])
    values = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    times = np.arange(20)/25
    native_valid = np.ones(20, bool); native_valid[2] = False
    out, mask = builder.align_visual(sample_times, np.array([True, True, False, True]), values, times, native_valid)
    assert np.flatnonzero(mask).tolist() == [0, 5, 15]
    np.testing.assert_array_equal(out[~mask], 0.)
    np.testing.assert_allclose(out[5], [1., 1.])


@pytest.mark.parametrize('corruption', ['metadata', 'source', 'support', 'sentence'])
def test_existing_membership_and_oof_contract_rejections(corruption):
    clips, predictions = inputs()
    if corruption == 'metadata': predictions[0]['clip_id'] = 'wrong'
    if corruption == 'source': predictions[0]['prediction_source'] = 'all_train'
    if corruption == 'support': predictions[0]['semantic_valid'] = torch.ones(30, dtype=torch.bool)
    if corruption == 'sentence': clips[-1]['sentence'] = clips[0]['sentence']; predictions[-1]['sentence'] = clips[0]['sentence']
    with pytest.raises(ValueError): train.validate_inputs({'clips': clips}, {'clips': predictions})


@pytest.mark.parametrize('corruption', ['audio_mask', 'nonfinite', 'va_range', 'posterior_negative', 'posterior_sum'])
def test_student_condition_masks_and_ranges_are_verified(corruption):
    clips, predictions = inputs()
    if corruption == 'audio_mask': predictions[0]['audio_valid'][0] = False
    if corruption == 'nonfinite': predictions[0]['va']['actual'][0, 0] = float('nan')
    if corruption == 'va_range': predictions[0]['va']['actual'][0, 0] = 2.
    if corruption == 'posterior_negative': predictions[0]['posterior']['actual'][0, 0] = -.1
    if corruption == 'posterior_sum': predictions[0]['posterior']['actual'][0] *= .5
    with pytest.raises(ValueError):train.validate_inputs({'clips': clips}, {'clips': predictions})


def test_cli_six_arm_single_epoch_smoke(tmp_path):
    clips, predictions = inputs()
    dataset = tmp_path/'dataset.pt'; torch.save({'clips': clips}, dataset)
    # The trainer now binds both prediction records and the fitted OOF model
    # membership. Keep this fixture intentionally minimal but provenance-valid.
    fold_by_sentence = {c['sentence']: i % 3 for i, c in enumerate(x for x in clips if x['split']=='train')}
    for row in predictions:
        row['fold'] = fold_by_sentence.get(row['sentence'])
    student_payload = {'clips': predictions, 'fold_by_sentence': fold_by_sentence}
    student_path = tmp_path/'predictions.pt'; torch.save(student_payload, student_path)
    model_payload = {
        'all_train': {'fit_clip_ids': [c['clip_id'] for c in clips if c['split']=='train'],
                      'fit_sentences': sorted({c['sentence'] for c in clips if c['split']=='train'})},
        'oof': {fold: {'fit_clip_ids': [c['clip_id'] for c in clips if c['split']=='train' and fold_by_sentence[c['sentence']] != fold],
                       'fit_sentences': sorted({c['sentence'] for c in clips if c['split']=='train' and fold_by_sentence[c['sentence']] != fold})}
                for fold in range(3)}}
    model_path = tmp_path/'models.pt'; torch.save(model_payload, model_path)
    (tmp_path/'provenance.json').write_text(json.dumps({'dataset_sha256': train.sha(dataset),
        'outputs': {'predictions.pt': train.sha(student_path), 'models.pt': train.sha(model_path)}}), encoding='utf8')
    output = tmp_path/'run'
    environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    result = subprocess.run([sys.executable, '-m', 'scripts.train_visual_semantic_pilot',
        '--dataset', str(dataset), '--student', str(student_path), '--output', str(output),
        '--epochs', '1', '--batch-size', '2', '--device', 'cpu'],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    status = json.loads((output/'status.json').read_text(encoding='utf8'))
    assert status['status'] == 'complete' and status['arms'] == list(train.ARMS)
    reports = json.loads((output/'reports.json').read_text(encoding='utf8'))
    assert set(reports) == set(train.ARMS)
    for arm in train.ARMS:
        losses = json.loads((output/(arm+'_losses.json')).read_text(encoding='utf8'))
        assert len(losses) == 1 and losses[0]['steps'] == 1
    packed = torch.load(output/'predictions.pt', map_location='cpu', weights_only=False)
    assert set(packed['clips']['clip_2']['samples']) == set(train.ARMS)
    assert packed['clips']['clip_2']['target52'].shape == (30, 52)
    checkpoints = [torch.load(output/(arm+'_last.pt'), map_location='cpu', weights_only=False) for arm in train.ARMS]
    assert len({c['order_sha256'] for c in checkpoints}) == 1
    assert all(torch.equal(checkpoints[0]['noise_rng'], c['noise_rng']) for c in checkpoints)
