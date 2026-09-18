"""Prosody correction isolation, clock, intervention and frozen-source checks."""
import copy
import json

import numpy as np
import pytest
import torch

from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior
from scripts import train_prosody_clocked_residual as p


def model_fixture(k=4):
    torch.manual_seed(151)
    stats = (torch.zeros(1540), torch.ones(1540), torch.zeros(2), torch.ones(2))
    base = ClockedMotionPrior(*stats, hidden=4, k=k)
    with torch.no_grad():
        base.output.weight.normal_(std=.1)
        base.feature_std[p.PROSODY] = torch.tensor([1., 2., 3., 4.])
    return p.ProsodyStaticResidualPrior(base)


def clip_fixture():
    clips = []
    clock = np.arange(48, dtype=np.float64)
    for i in range(2):
        features = torch.zeros(48, 1540)
        features[:, :1536] = .1+i
        features[:, p.PROSODY] = torch.tensor(np.stack((np.sin(clock/5+i), np.cos(clock/9),
                                                       clock/48+i, np.sin(clock/3)), -1), dtype=torch.float32)
        clips.append({'clip_id': f'clip{i}', 'sentence': f'sentence{i}', 'speaker': 0, 'emotion': 1,
            'features': features, 'valid': torch.ones(48, dtype=torch.bool),
            'upper': .4+.04*np.sin(clock[:, None]/(7+i)+np.arange(9)[None]/4),
            'global': np.array([.2*i, .3], dtype=np.float32), 'anchor_upper': np.full(9, .4)})
    transformed, _ = p.r.coordinate_clips(clips, [0, 1])
    windows = p.c.shapes.extract_windows(transformed, [0, 1])
    dictionary = {'shapes': np.stack([w['future'] for w in windows]), 'scales': np.full(9, .15)}
    fitted = p.c.fit_equilibrium(transformed, [0, 1])
    return clips, windows, dictionary, fitted


def test_zero_initialization_and_exact_static_null_after_update():
    model = model_fixture()
    features = torch.randn(3, 32, 1540)
    static = features.mean(1, keepdim=True).expand_as(features)
    valid, global_ = torch.ones(3, 32, dtype=torch.bool), torch.randn(3, 2)
    assert torch.equal(model.residual_logits(features, valid, global_, static), torch.zeros(3, 4))
    with torch.no_grad():
        model.correction[-1].weight.normal_()
        model.correction[-1].bias.normal_()
    assert torch.equal(model.residual_logits(static, valid, global_, static), torch.zeros(3, 4))
    logits = model.residual_logits(features, valid, global_, static)
    assert (logits.abs() <= 2).all() and logits.abs().max() > 0


def test_only_four_prosody_channels_enter_correction_no_global_bypass():
    model = model_fixture()
    with torch.no_grad():
        model.correction[-1].weight.normal_()
    features = torch.randn(2, 32, 1540)
    static = features.mean(1, keepdim=True).expand_as(features).clone()
    valid = torch.ones(2, 32, dtype=torch.bool)
    reference = model.residual_logits(features, valid, torch.ones(2, 2), static)
    features[:, :, :1536] = float('nan')
    static[:, :, :1536] = float('nan')
    changed = model.residual_logits(features, valid, torch.full((2, 2), float('nan')), static)
    torch.testing.assert_close(reference, changed, rtol=0, atol=0)


def test_descriptor_fixed_bins_mask_gaps_without_compressing_or_refitting():
    model = model_fixture()
    features = torch.zeros(1, 32, 1540)
    static = torch.zeros_like(features)
    valid = torch.ones(1, 32, dtype=torch.bool)
    features[0, 4:8, p.PROSODY] = torch.tensor([1., 2., 3., 4.])*2
    features[0, 12:16, p.PROSODY] = torch.tensor([1., 2., 3., 4.])*5
    valid[0, 5:7] = False; valid[0, 12:16] = False
    features[~valid] = float('nan'); static[~valid] = float('nan')
    descriptor = model.temporal_descriptor(features, valid, static).reshape(1, 8, 4)
    expected = torch.zeros_like(descriptor); expected[:, 1] = 2
    torch.testing.assert_close(descriptor, expected, rtol=0, atol=0)
    features.requires_grad_(True)
    with torch.no_grad():
        model.correction[-1].weight.normal_()
    model.residual_logits(features, valid, None, static).sum().backward()
    assert torch.isfinite(features.grad).all() and features.grad[~valid].abs().sum() == 0


def test_whole_original_clip_mean_is_used_without_window_recentring():
    model = model_fixture()
    features = torch.zeros(48, 1540)
    features[:, p.PROSODY] = torch.arange(48)[:, None].float()
    valid = torch.ones(48, dtype=torch.bool)
    static = p.c.static_acoustics(features, valid)
    x, mask, _ = p.c.acoustic_windows(features, valid)
    sx, _, _ = p.c.acoustic_windows(static, valid)
    descriptor = model.temporal_descriptor(x, mask, sx).reshape(-1, 8, 4)
    expected = (torch.arange(8).float()*4+1.5-23.5)[:, None]/model.base.feature_std[p.PROSODY]
    torch.testing.assert_close(descriptor[0], expected)
    assert descriptor[0].mean() != 0


def test_training_has_no_base_grad_and_preserves_source_state(tmp_path):
    clips, windows, dictionary, _ = clip_fixture()
    model = model_fixture(len(dictionary['shapes']))
    before = copy.deepcopy(model.base.state_dict())
    data = p.c.build_training(clips, windows, dictionary)
    history = p.train_model(model, data, dictionary, tmp_path, 'cpu', epochs=2)
    for key, value in model.base.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(param.grad is None and not param.requires_grad for param in model.base.parameters())
    assert model.correction[-1].weight.abs().sum() > 0
    matching = json.loads((tmp_path/'matching.json').read_text())
    assert matching['base_unchanged_exact'] and matching['optimizer_base_parameters'] == 0
    assert matching['trainable_parameters'] == 528+17*len(dictionary['shapes'])
    assert len(history) == 2


def test_scale_zero_exact_for_all_interventions_and_static_control_null():
    clips, _, dictionary, fitted = clip_fixture()
    model = model_fixture(len(dictionary['shapes'])).eval()
    with torch.no_grad():
        model.correction[-1].weight.normal_()
    reference = None
    for intervention, scale in [('real', 0.), ('reverse', 0.), ('mismatch', 0.), ('static', 1.)]:
        _, curves = p.r.evaluate(clips, [0, 1], dictionary, fitted, np.full(9, .04), .1,
                                 model, 'cpu', intervention=intervention, residual_scale=scale)
        if reference is None:
            reference = curves
        for cid, result in curves.items():
            for key in ('samples', 'probabilities', 'tokens', 'logit_level'):
                np.testing.assert_array_equal(result[key], reference[cid][key])


def test_teacher_target_has_no_effect_on_audio_distribution_or_samples():
    clips, _, dictionary, fitted = clip_fixture()
    model = model_fixture(len(dictionary['shapes'])).eval()
    with torch.no_grad():
        model.correction[-1].weight.normal_()
    _, reference = p.r.evaluate(clips, [0, 1], dictionary, fitted, np.full(9, .04), .1, model, 'cpu')
    altered = copy.deepcopy(clips)
    for clip in altered:
        clip['upper'] = 1.-clip['upper']
    _, changed = p.r.evaluate(altered, [0, 1], dictionary, fitted, np.full(9, .04), .1, model, 'cpu')
    for cid in reference:
        for key in ('samples', 'probabilities', 'tokens', 'logit_level'):
            np.testing.assert_array_equal(reference[cid][key], changed[cid][key])


def test_mismatch_preserves_recipient_static_and_global(monkeypatch):
    clips, _, dictionary, fitted = clip_fixture()
    captured = []
    original = p.r.probability
    def capture(model, features, static, valid, global_vec, device, scale):
        captured.append((features.clone(), static.clone(), np.array(global_vec)))
        return original(model, features, static, valid, global_vec, device, scale)
    monkeypatch.setattr(p.r, 'probability', capture)
    model = model_fixture(len(dictionary['shapes'])).eval()
    report, _ = p.r.evaluate(clips, [0, 1], dictionary, fitted, np.full(9, .04), .1,
                             model, 'cpu', intervention='mismatch')
    assert report['donor_mapping'] == {'clip0': 'clip1', 'clip1': 'clip0'}
    for i, (features, static, global_) in enumerate(captured):
        torch.testing.assert_close(features, clips[1-i]['features'])
        torch.testing.assert_close(static, p.c.static_acoustics(clips[i]['features'], clips[i]['valid']))
        np.testing.assert_array_equal(global_, clips[i]['global'])


@pytest.mark.parametrize('scale', [True, float('nan'), -.1, 1.01, '1'])
def test_invalid_scale_rejected(scale):
    model = model_fixture()
    x, mask, global_ = torch.zeros(1, 32, 1540), torch.ones(1, 32, dtype=torch.bool), torch.zeros(1, 2)
    with pytest.raises(ValueError, match='Residual scale'):
        model(x, mask, global_, x, scale=scale)


def source_fixture(path):
    model = model_fixture(128)
    checkpoint = {'schema': p.r.SCHEMA, 'state': model.base.state_dict(),
                  'epochs': 30, 'order_sha256': 'fixed-source-order'}
    torch.save(checkpoint, path/'static_final.pt')
    torch.save({**checkpoint, 'state': {'base.'+key: value for key, value in checkpoint['state'].items()}},
               path/'residual_final.pt')
    torch.save({'coordinate_system': 'logit raw minus observed training clip logit mean',
        'horizon': 32, 'hop': 16, 'shapes': np.zeros((128, 32, 9)), 'scales': np.ones(9)}, path/'dictionary.pt')
    torch.save({'fit_clip_ids': ['fit-only']}, path/'equilibrium.pt')
    (path/'scales.json').write_text(json.dumps([.1]*9))
    (path/'training_data.json').write_text(json.dumps({'low_activity_train_quantile20': .2}))
    (path/'protocol.json').write_text(json.dumps({'schema': p.r.SCHEMA, 'code_sha256': {}}))
    (path/'status.json').write_text(json.dumps({'schema': p.r.SCHEMA, 'status': 'complete',
                                                'smoke': False, 'test_loaded': False}))
    return model


def test_source_reuse_is_exact_and_rejects_modified_base(tmp_path, monkeypatch):
    original = source_fixture(tmp_path)
    def fail_refit(*args, **kwargs):
        raise AssertionError('Source statistics or dictionary must never be refitted')
    monkeypatch.setattr(p.c.shapes, 'fit_shape_dictionary', fail_refit)
    monkeypatch.setattr(p.c, 'fit_equilibrium', fail_refit)
    monkeypatch.setattr(p.c.old, 'fit_feature_stats', fail_refit)
    model, _, _, scales, low_cut, _, binding = p.load_source(tmp_path, 'cpu')
    for key, value in original.base.state_dict().items():
        torch.testing.assert_close(value, model.base.state_dict()[key], rtol=0, atol=0)
    assert binding['base_unchanged_exact'] and not binding['normalization_refitted']
    assert set(binding['files_sha256']) == set(p.SOURCE_FILES)
    np.testing.assert_array_equal(scales, [.1]*9)
    assert low_cut == .2 and sum(param.numel() for param in model.parameters() if param.requires_grad) == 2704
    bad = torch.load(tmp_path/'residual_final.pt', weights_only=False)
    bad['state']['base.output.bias'] += 1
    torch.save(bad, tmp_path/'residual_final.pt')
    with pytest.raises(ValueError, match='changed its static base'):
        p.load_source(tmp_path, 'cpu')
