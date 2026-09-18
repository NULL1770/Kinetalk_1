"""Deployment/fold contracts and a small real fitting smoke for temporal students."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts import fit_temporal_semantic_student as s
from scripts import fit_visual_semantic_student as ridge


def clip(index, split='train'):
    generator = torch.Generator().manual_seed(index+1601)
    frames = 13
    features = torch.randn(frames, 1540, generator=generator)
    valid = torch.ones(frames, dtype=torch.bool)
    return {'clip_id': 'clip_'+str(index), 'sentence': 'sentence_'+str(index), 'split': split,
            'features': features, 'valid': valid, 'semantic_valid': valid.clone(),
            'va': torch.tanh(features[:, 768:770]), 'posterior': torch.softmax(features[:, 770:778], -1)}


@pytest.fixture(scope='module')
def baseline():
    torch.set_num_threads(2)
    data = {'clips': [clip(i) for i in range(6)]+[clip(7, 'holdout')]}
    predictions, models, _ = ridge.fit_dataset(data)
    return data, predictions, models


@pytest.fixture(scope='module')
def fitted(baseline):
    data, predictions, models = baseline
    return s.fit_dataset(data, predictions, models, epochs=1, device='cpu', batch_size=4)


def test_training_scale_is_fit_only_and_missing_labels_do_not_become_zero_targets(baseline):
    data, _, models = baseline
    c = copy.deepcopy(data['clips'][0]); c['semantic_valid'][5:10] = False
    c['va'][5:10] = float('nan'); c['posterior'][5:10] = float('nan')
    record = s.audio_input(c, models['all_train']['preprocessing'])
    target = s.training_target(c, record['windows'])
    assert target['counts'].tolist() == [5, 0, 3]
    assert target['known'].tolist() == [True, False, True]
    np.testing.assert_allclose((target['target']*target['counts'][:, None]).sum(0), 0, atol=2e-6)
    changed = copy.deepcopy(c); changed['va'][5:10] = 500.; changed['posterior'][5:10] = 700.
    after = s.training_target(changed, record['windows'])
    np.testing.assert_array_equal(after['target'], target['target'])
    scale = s._fit_scale([target])
    np.testing.assert_allclose(scale, np.sqrt((target['target'].astype(float)**2*target['counts'][:, None]).sum(0)/8).clip(.05), atol=1e-7)


def test_audio_input_is_raw_pca_plus_centered_change_and_reverse_preserves_clock(baseline):
    data, _, models = baseline
    c = data['clips'][0]; prep = models['all_train']['preprocessing']
    a = s.audio_input(c, prep); b = s.audio_input(c, prep, reverse=True)
    dimension = prep['components'].shape[1]
    np.testing.assert_allclose(np.average(a['features'][:, dimension:], axis=0, weights=a['weights']), 0, atol=2e-6)
    np.testing.assert_allclose(a['features'][:, :dimension][::-1], b['features'][:, :dimension], atol=1e-6)
    assert a['windows']['spans'] == b['windows']['spans'] == [(0, 5), (5, 10), (10, 13)]
    np.testing.assert_array_equal(a['weights'], b['weights'])


def test_tcn_has_condition_gradient_and_frame_weighted_zero_raw_mean():
    torch.manual_seed(930)
    net = s.TemporalResidualTCN(6)
    x = torch.randn(2, 11, 6, requires_grad=True)
    valid = torch.ones(2, 11, dtype=torch.bool); valid[0, -2:] = False
    runs = torch.zeros_like(valid, dtype=torch.int64)
    weights = valid.float()*5; weights[1, -1] = 2
    result = net(x, valid, runs, weights)
    torch.testing.assert_close((result*weights[..., None]).sum(1), torch.zeros(2, 10), atol=2e-6, rtol=0)
    target = torch.randn_like(result); counts = weights.clone(); counts[:, 3] = 0
    loss = s.normalized_residual_loss(result, target, counts, torch.full((10,), .3))
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(x.grad[~valid]) == 0
    assert net.input.weight.grad.abs().sum() > 0


def test_temporal_convolution_never_crosses_run_gaps():
    torch.manual_seed(931)
    net = s.TemporalResidualTCN(6)
    x = torch.randn(1, 12, 6); valid = torch.ones(1, 12, dtype=torch.bool)
    runs = torch.tensor([[0]*5+[1]*7])
    original = net.local_field(x, valid, runs)
    modified = x.clone(); modified[:, :5] += 50
    after = net.local_field(modified, valid, runs)
    # Global zero-mean subtraction may couple run means, but never the local
    # temporal convolution. The assertion intentionally measures local_field.
    torch.testing.assert_close(original[:, 5:], after[:, 5:], rtol=0, atol=0)


def test_paired_small_constant_input_is_exact_zero_even_with_tail_weights_and_padding():
    torch.manual_seed(941)
    net = s.TemporalResidualTCN(12, hidden=16, paired_small=True)
    valid = torch.ones(2, 11, dtype=torch.bool); valid[0, -3:] = False
    raw = torch.randn(2, 1, 6).expand(-1, 11, -1).clone()
    # Small floating residuals in precomputed centering are recomputed from
    # the raw half, so cannot inject a false constant-audio response.
    x = torch.cat((raw, torch.full_like(raw, 1e-7)), -1)
    x[~valid] = float('nan')
    weights = valid.float()*5; weights[0, 7] = 2; weights[1, -1] = 3
    runs = torch.zeros_like(valid, dtype=torch.int64)
    result = net(x, valid, runs, weights)
    assert torch.count_nonzero(result) == 0


def test_paired_small_real_condition_and_both_twin_branches_receive_gradients():
    torch.manual_seed(942)
    net = s.TemporalResidualTCN(12, hidden=16, paired_small=True)
    x = torch.randn(2, 11, 12, requires_grad=True)
    valid = torch.ones(2, 11, dtype=torch.bool); weights = valid.float()*5
    runs = torch.zeros_like(valid, dtype=torch.int64)
    intermediates = []
    hook = net.output.register_forward_hook(lambda module, inputs, output: (output.retain_grad(), intermediates.append(output)) and None)
    result = net(x, valid, runs, weights)
    hook.remove()
    assert result.abs().sum() > 0
    loss = (result-torch.randn_like(result)).square().mean(); loss.backward()
    assert len(intermediates) == 2 and all(v.grad is not None and v.grad.abs().sum() > 0 for v in intermediates)
    assert x.grad[..., :6].abs().sum() > 0 and net.input.weight.grad.abs().sum() > 0


def test_paired_small_padding_invariance_and_local_field_gap_isolation():
    torch.manual_seed(943)
    net = s.TemporalResidualTCN(8, hidden=16, paired_small=True)
    x = torch.randn(1, 11, 8); valid = torch.ones(1, 11, dtype=torch.bool)
    runs = torch.tensor([[0]*5+[1]*6]); weights = valid.float()*5; weights[0, -1] = 2
    a = net(x, valid, runs, weights)
    b = net(torch.nn.functional.pad(x, (0, 0, 0, 4), value=float('nan')),
            torch.nn.functional.pad(valid, (0, 4), value=False),
            torch.nn.functional.pad(runs, (0, 4), value=-1),
            torch.nn.functional.pad(weights, (0, 4), value=0))
    torch.testing.assert_close(a, b[:, :11], rtol=2e-6, atol=2e-6)
    original = net.local_field(x, valid, runs)
    changed = x.clone(); changed[:, :5] += 50
    torch.testing.assert_close(original[:, 5:], net.local_field(changed, valid, runs)[:, 5:], rtol=0, atol=0)


def test_paired_small_fit_saves_config_and_constant_output_equals_old_static(baseline):
    data, _, old_models = baseline
    model = s.fit_model(data['clips'][:6], old_models['all_train'], epochs=1, paired_small=True)
    assert model['config']['paired_small'] and model['config']['hidden'] == 16
    predictions = s.predict_audio(model, data['clips'][-1])
    old = ridge.predict_audio(old_models['all_train'], data['clips'][-1])
    for name in ('va', 'posterior'):
        assert torch.equal(predictions[name]['constant_audio'], old[name]['static'])
        assert torch.equal(predictions[name]['static'], old[name]['static'])


def test_prediction_is_features_only_static_exact_and_bounded(baseline, fitted):
    data, _, old_models = baseline
    _, models, _ = fitted
    c = data['clips'][-1]
    before = s.predict_audio(models['all_train'], c)
    audio = {key: c[key] for key in ('features', 'valid')}
    audio['va'] = None; audio['semantic_valid'] = None; audio['posterior'] = None
    after = s.predict_audio(models['all_train'], audio)
    old = ridge.predict_audio(old_models['all_train'], c)
    for name in ('va', 'posterior'):
        assert torch.equal(before[name]['static'], old[name]['static'])
        for mode in (*ridge.MODES, 'constant_audio'):
            assert torch.equal(before[name][mode], after[name][mode])
            assert torch.isfinite(after[name][mode]).all()
    assert before['va']['actual'].abs().max() <= 1
    torch.testing.assert_close(before['posterior']['actual'].sum(-1), torch.ones(13))


def test_oof_models_reuse_exact_fold_baselines_and_holdout_never_fitted(baseline, fitted):
    data, _, old_models = baseline
    predictions, models, report = fitted
    for row in predictions['clips']:
        if row['split'] == 'train':
            model = models['oof'][row['fold']]
            assert row['sentence'] not in model['fit_sentences']
            assert row['clip_id'] not in model['fit_clip_ids']
            old = old_models['oof'][row['fold']]
        else:
            model = models['all_train']; old = old_models['all_train']
            assert row['clip_id'] not in model['fit_clip_ids']
        for key in ('mean', 'scale', 'components'):
            np.testing.assert_array_equal(model['baseline']['preprocessing'][key], old['preprocessing'][key])
        assert model['epochs'] == 1
    assert report['holdout']['va']['actual']['clip_count'] == 1
    assert report['constant_audio_diagnostic']['holdout']['va']['clip_count'] == 1
    assert report['output_mean_diagnostics']['nonlinear_mean_preservation_claim'] is False


def test_constant_audio_is_same_network_flat_input_not_static_ridge(baseline, fitted):
    data, _, old_models = baseline
    audio = s.audio_input(data['clips'][0], old_models['all_train']['preprocessing'], constant_audio=True)
    dimension = audio['features'].shape[1]//2
    np.testing.assert_allclose(audio['features'][:, :dimension], np.broadcast_to(audio['features'][0, :dimension], audio['features'][:, :dimension].shape))
    np.testing.assert_allclose(audio['features'][:, dimension:], 0, atol=1e-6)
    predictions = fitted[0]['clips']
    report = s.output_mean_diagnostics(predictions)
    first = predictions[0]; valid = first['audio_valid']
    direct = first['va']['actual'][valid].double().mean(0)-first['va']['static'][valid].double().mean(0)
    np.testing.assert_allclose(report['clips'][0]['groups']['va']['actual']['mean_shift_vector'], direct.numpy(), atol=1e-10)


def test_bad_oof_baseline_members_rejected_before_training(baseline):
    data, predictions, models = copy.deepcopy(baseline)
    models['oof'][0] = models['all_train']
    with pytest.raises(ValueError, match='membership'):
        s.fit_dataset(data, predictions, models, epochs=1)


def test_one_epoch_cli_saves_existing_trainer_compatible_contract(tmp_path, baseline):
    data, old_predictions, old_models = baseline
    dataset = tmp_path/'dataset.pt'; torch.save(data, dataset)
    root = tmp_path/'baseline'; root.mkdir()
    old_path = root/'predictions.pt'; models_path = root/'models.pt'
    torch.save(old_predictions, old_path); torch.save(old_models, models_path)
    (root/'provenance.json').write_text(json.dumps({'schema': ridge.SCHEMA, 'dataset_sha256': ridge._sha(dataset),
        'outputs': {'predictions.pt': ridge._sha(old_path), 'models.pt': ridge._sha(models_path)}}), encoding='utf8')
    output = tmp_path/'new'
    result = subprocess.run([sys.executable, '-m', 'scripts.fit_temporal_semantic_student',
        '--dataset', str(dataset), '--baseline-student', str(old_path), '--output', str(output),
        '--epochs', '1', '--device', 'cpu', '--batch-size', '4'],
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    provenance = json.loads((output/'provenance.json').read_text(encoding='utf8'))
    assert provenance['dataset_sha256'] == ridge._sha(dataset)
    assert provenance['epochs'] == 1 and not provenance['holdout_used_for_selection']
    for name, digest in provenance['outputs'].items():assert ridge._sha(output/name) == digest
    loaded = torch.load(output/'predictions.pt', weights_only=False)
    assert loaded['fold_by_sentence'] == old_predictions['fold_by_sentence']
    # A tampered model cannot be silently used even when predictions still match.
    models_path.write_bytes(models_path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        s.load_baseline(dataset, old_path)
