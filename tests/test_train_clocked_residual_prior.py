"""Bounded-coordinate residual driver isolation and invariance checks."""
import copy
import json

import numpy as np
import torch

from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior, TemporalResidualPrior
from scripts import train_clocked_residual_prior as driver


def fixture():
    clips = []
    clock = np.arange(48, dtype=np.float64)
    for i in range(2):
        features = torch.tensor(np.stack((np.sin(clock/5+i), np.cos(clock/9), clock/48+i), -1), dtype=torch.float32)
        clips.append({'clip_id': f'clip{i}', 'sentence': f'sentence{i}',
                      'speaker': 0, 'emotion': 1, 'features': features,
                      'valid': torch.ones(48, dtype=torch.bool),
                      'upper': .4 + .04*np.sin(clock[:, None]/(7+i)+np.arange(9)[None]/4),
                      'global': np.array([.2*i, .3], dtype=np.float32),
                      'anchor_upper': np.full(9, .4)})
    transformed, _ = driver.coordinate_clips(clips, [0, 1])
    windows = driver.c.shapes.extract_windows(transformed, [0, 1])
    dictionary = {'shapes': np.stack([w['future'] for w in windows]), 'scales': np.full(9, .15)}
    stats = (torch.zeros(3), torch.ones(3), torch.zeros(2), torch.ones(2))
    fitted = driver.c.fit_equilibrium(transformed, [0, 1])
    return clips, windows, dictionary, stats, fitted


def test_coordinate_transform_does_not_mutate_input_or_read_holdout():
    clips, _, _, _, _ = fixture()
    clips[0]['valid'][4] = False
    clips[0]['upper'][4] = float('nan')
    clips[0]['upper'][0, 0] = 0.
    clips[1]['upper'][:] = float('nan')
    before = copy.deepcopy(clips)
    transformed, counts = driver.coordinate_clips(clips, [0])
    assert transformed[1] is clips[1]
    assert transformed[0] is not clips[0]
    for old, new in zip(before, clips):
        np.testing.assert_array_equal(old['upper'], new['upper'])
        np.testing.assert_array_equal(old['anchor_upper'], new['anchor_upper'])
    assert transformed[0]['upper'][4].tolist() == [0.]*9
    assert np.isfinite(transformed[0]['upper']).all()
    assert counts['clip0']['clipped_values'] == 1
    assert set(counts) == {'clip0'}
    valid = clips[0]['valid'].numpy()
    np.testing.assert_allclose(transformed[0]['upper'][valid], driver.original_dictionary.logit(clips[0]['upper'][valid]))


def test_mismatch_keeps_original_static_audio_global_and_anchor(monkeypatch):
    clips, _, dictionary, _, fitted = fixture()
    captured = []
    def capture(model, features, static, valid, global_vec, device, scale):
        captured.append((features.clone(), static.clone(), np.array(global_vec)))
        _, _, locations = driver.c.acoustic_windows(features, valid)
        return np.full((len(locations), len(dictionary['shapes'])), 1/len(dictionary['shapes'])), locations
    monkeypatch.setattr(driver, 'probability', capture)
    report, curves = driver.evaluate(clips, [0, 1], dictionary, fitted,
                                     np.full(9, .04), .1, None, 'cpu', intervention='mismatch')
    assert report['donor_mapping'] == {'clip0': 'clip1', 'clip1': 'clip0'}
    for i, (features, static, global_) in enumerate(captured):
        torch.testing.assert_close(features, clips[1-i]['features'])
        torch.testing.assert_close(static, driver.c.static_acoustics(clips[i]['features'], clips[i]['valid']))
        np.testing.assert_array_equal(global_, clips[i]['global'])
        expected = driver.c.equilibrium(clips[i]['global'], driver.original_dictionary.logit(clips[i]['anchor_upper']), fitted)
        np.testing.assert_array_equal(curves[clips[i]['clip_id']]['logit_level'], expected)


def test_zero_scale_is_exact_across_audio_interventions_and_sigmoid_is_bounded():
    torch.manual_seed(132)
    clips, _, dictionary, stats, fitted = fixture()
    base = ClockedMotionPrior(*stats, hidden=4, k=len(dictionary['shapes']))
    model = TemporalResidualPrior(base).eval()
    with torch.no_grad():
        model.base.output.weight.normal_()
        model.correction.output.weight.normal_()
    reference = None
    for intervention in ('real', 'static', 'reverse', 'mismatch'):
        _, curves = driver.evaluate(clips, [0, 1], dictionary, fitted, np.full(9, .04), .1,
                                     model, 'cpu', intervention=intervention, residual_scale=0.)
        if reference is None:
            reference = curves
        for cid, item in curves.items():
            assert np.isfinite(item['samples']).all()
            assert (item['samples'] >= 0).all() and (item['samples'] <= 1).all()
            for key in ('samples', 'probabilities', 'tokens', 'logit_level'):
                np.testing.assert_array_equal(item[key], reference[cid][key])


def test_query_target_does_not_affect_probability_samples_or_predicted_level():
    clips, _, dictionary, stats, fitted = fixture()
    model = TemporalResidualPrior(ClockedMotionPrior(*stats, hidden=4, k=len(dictionary['shapes']))).eval()
    _, reference = driver.evaluate(clips, [0, 1], dictionary, fitted, np.full(9, .04), .1, model, 'cpu')
    altered = copy.deepcopy(clips)
    for clip in altered:
        clip['upper'] = 1.-clip['upper']
    _, changed = driver.evaluate(altered, [0, 1], dictionary, fitted, np.full(9, .04), .1, model, 'cpu')
    for cid in reference:
        for key in ('samples', 'probabilities', 'tokens', 'logit_level'):
            np.testing.assert_array_equal(reference[cid][key], changed[cid][key])


def test_training_freezes_static_stage_exactly_and_saves_matching_order(tmp_path):
    clips, windows, dictionary, stats, _ = fixture()
    data = driver.c.build_training(clips, windows, dictionary)
    model, histories = driver.train_models(data, stats, dictionary, tmp_path, 'cpu', epochs=2)
    base_saved = torch.load(tmp_path/'static_final.pt', weights_only=False)
    residual_saved = torch.load(tmp_path/'residual_final.pt', weights_only=False)
    assert base_saved['order_sha256'] == residual_saved['order_sha256']
    for key, value in base_saved['state'].items():
        torch.testing.assert_close(value, model.base.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(value, residual_saved['state']['base.'+key], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    assert model.correction.output.weight.abs().sum() > 0
    assert len(histories['static']) == len(histories['residual']) == 2
    assert json.loads((tmp_path/'matching.json').read_text())['base_unchanged_exact']


def test_empty_mismatch_support_is_safe():
    clips, _, dictionary, stats, fitted = fixture()
    clips[1]['sentence'] = clips[0]['sentence']
    model = TemporalResidualPrior(ClockedMotionPrior(*stats, hidden=4, k=len(dictionary['shapes']))).eval()
    report, curves = driver.evaluate(clips, [0, 1], dictionary, fitted, np.full(9, .04), .1,
                                     model, 'cpu', intervention='mismatch')
    assert report['rows'] == [] and report['summary'] is None and curves == {}
