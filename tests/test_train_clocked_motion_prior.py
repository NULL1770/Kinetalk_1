"""Driver-level clock, paired-training, and inference target isolation checks."""
import copy

import numpy as np
import torch

from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior
from scripts import clocked_motion_dictionary as shapes
from scripts import train_clocked_motion_prior as driver


def fixture():
    t = np.arange(64, dtype=np.float64)
    clips = []
    for i in range(2):
        upper = .4 + .035*np.sin(t[:, None]/(6+i)+np.arange(9)[None]/3)
        features = torch.full((64, 3), .25 + .125*i)
        clips.append({'clip_id': f'clip{i}', 'sentence': f'sentence{i}',
                      'speaker': 1, 'emotion': 2, 'features': features,
                      'valid': torch.ones(64, dtype=torch.bool), 'upper': upper,
                      'global': np.array([.2*i, .3], dtype=np.float32),
                      'anchor_upper': np.full(9, .4)})
    windows = shapes.extract_windows(clips, [0, 1])
    dictionary = {'shapes': np.stack([w['future'] for w in windows[:3]]),
                  'scales': np.full(9, .04)}
    stats = (torch.zeros(3), torch.ones(3), torch.zeros(2), torch.ones(2))
    return clips, windows, dictionary, stats


def test_window_scheduler_preserves_gaps_tail_and_original_frame_indices():
    x = torch.arange(70.)[:, None].repeat(1, 3)
    valid = torch.ones(70, dtype=torch.bool)
    valid[20:27] = False
    x[~valid] = float('nan')
    values, masks, locations = driver.acoustic_windows(x, valid)
    assert locations == [(0, 20), (27, 32), (38, 32)]
    for row, (start, count) in enumerate(locations):
        torch.testing.assert_close(values[row, :count], x[start:start+count])
        assert masks[row, :count].all()
        assert not masks[row, count:].any()
        assert values[row, count:].count_nonzero() == 0
    assert torch.isfinite(values).all()


def test_static_audio_mean_ignores_invalid_nan_without_motion_inputs():
    features = torch.tensor([[1., 2.], [float('nan'), float('nan')], [5., 6.]])
    actual = driver.static_acoustics(features, torch.tensor([True, False, True]))
    torch.testing.assert_close(actual, torch.tensor([[3., 4.], [0., 0.], [3., 4.]]))


def test_equilibrium_fit_does_not_read_unlisted_query_motion():
    clips, _, _, _ = fixture()
    holdout = copy.deepcopy(clips[0])
    holdout['upper'][:] = float('nan')
    holdout['features'][:] = float('nan')
    holdout['clip_id'] = 'forbidden'
    fitted = driver.fit_equilibrium(clips + [holdout], [0, 1])
    assert fitted['fit_clip_ids'] == ['clip0', 'clip1']
    assert np.isfinite(driver.equilibrium(clips[0]['global'], clips[0]['anchor_upper'], fitted)).all()


def test_identical_static_temporal_inputs_train_identical_paired_models(tmp_path):
    clips, windows, dictionary, stats = fixture()
    data = driver.build_training(clips, windows, dictionary)
    # All frames within each fixture clip are constant, so both arms must match.
    torch.testing.assert_close(data['temporal'], data['static'], rtol=0, atol=0)
    models, losses = driver.train_pair(data, stats, dictionary, tmp_path, 'cpu', epochs=2)
    assert losses['static'] == losses['temporal']
    for name, state in models['static'].state_dict().items():
        torch.testing.assert_close(state, models['temporal'].state_dict()[name], rtol=0, atol=0)
    saved = [torch.load(tmp_path/(arm+'_final.pt'), weights_only=False) for arm in ('static', 'temporal')]
    assert saved[0]['order_sha256'] == saved[1]['order_sha256']
    assert models['temporal'].output.weight.abs().sum() > 0


def test_evaluation_samples_do_not_change_when_only_query_target_changes():
    clips, _, dictionary, stats = fixture()
    fitted = driver.fit_equilibrium(clips, [0, 1])
    model = ClockedMotionPrior(*stats, hidden=4, k=3).eval()
    with torch.no_grad():
        model.output.weight.normal_()
    report, curves = driver.evaluate(clips, [0, 1], dictionary, fitted,
                                     np.full(9, .04), .01, model, 'cpu')
    altered = copy.deepcopy(clips)
    for c in altered:
        c['upper'] = .65 - c['upper']*.2
    changed_report, changed_curves = driver.evaluate(altered, [0, 1], dictionary, fitted,
                                                     np.full(9, .04), .01, model, 'cpu')
    for cid in curves:
        np.testing.assert_array_equal(curves[cid]['samples'], changed_curves[cid]['samples'])
        np.testing.assert_array_equal(curves[cid]['probabilities'], changed_curves[cid]['probabilities'])
        np.testing.assert_array_equal(curves[cid]['tokens'], changed_curves[cid]['tokens'])
    assert report['summary']['joint_fair_es'] != changed_report['summary']['joint_fair_es']


def test_shape_sampling_is_seed_stable_and_separates_missing_runs():
    _, _, dictionary, _ = fixture()
    valid = np.ones(69, dtype=bool)
    valid[18:25] = False
    x = torch.zeros(69, 3)
    _, _, locations = driver.acoustic_windows(x, torch.from_numpy(valid))
    probabilities = np.full((len(locations), 3), 1/3)
    level = np.full(9, .4)
    first, tokens = driver.sample_shapes(probabilities, locations, dictionary, level, valid, 'sample')
    repeated, repeated_tokens = driver.sample_shapes(probabilities, locations, dictionary, level, valid, 'sample')
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(tokens, repeated_tokens)
    assert np.isfinite(first).all()
    assert not first[:, ~valid].any()
    # Each run starts with its own selected shape rather than carrying preceding motion.
    j = next(i for i, (start, _) in enumerate(locations) if start == 25)
    np.testing.assert_allclose(first[:, 25], dictionary['shapes'][tokens[:, j], 0] + level)


def test_no_mismatch_donor_is_reported_and_cannot_pass_gate():
    clips, _, dictionary, stats = fixture()
    clips[1]['sentence'] = clips[0]['sentence']
    fitted = driver.fit_equilibrium(clips, [0, 1])
    model = ClockedMotionPrior(*stats, hidden=4, k=3).eval()
    report, curves = driver.evaluate(clips, [0, 1], dictionary, fitted,
                                     np.full(9, .04), .01, model, 'cpu', intervention='mismatch')
    assert report['rows'] == [] and report['summary'] is None and curves == {}
    outcome = driver.assessment({k: report for k in ('temporal', 'static_trained', 'static', 'reverse', 'mismatch')})
    assert not outcome['timing_passed'] and not outcome['quality_passed']
    assert outcome['reason'] == 'missing_intervention_support'


def test_assessment_uses_current_full_clip_metric_schema():
    clips, _, dictionary, stats = fixture()
    fitted = driver.fit_equilibrium(clips, [0, 1])
    model = ClockedMotionPrior(*stats, hidden=4, k=3).eval()
    report, _ = driver.evaluate(clips, [0, 1], dictionary, fitted,
                                np.full(9, .04), .01, model, 'cpu')
    outcome = driver.assessment({k: report for k in ('temporal', 'static_trained', 'static', 'reverse', 'mismatch')})
    assert outcome['support'] == 2 and outcome['support_sentences'] == 2
    assert outcome['support_fraction'] == 1.
    assert not outcome['timing_passed']
    assert outcome['gains_baseline_minus_real']['static_trained']['gain'] == 0.
    assert set(outcome['quality_checks']) == {'group_rms', 'speed', 'acceleration', 'raw_domain', 'clamp_retention'}
