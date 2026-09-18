"""Fitting-scope, acoustic isolation and score/gate tests for the activity probe."""
import copy
import json

import numpy as np
import torch

from scripts import train_activity_condition as driver


def fixture():
    rng = np.random.default_rng(71)
    clips = []
    for i, length in enumerate((40, 57, 48)):
        clock = np.arange(length)
        features = torch.zeros(length, 1540)
        features[:, 1536:] = torch.tensor(rng.normal(size=(length, 4))+i, dtype=torch.float32)
        clips.append({'clip_id': f'clip{i}', 'sentence': f'sentence{i}', 'speaker': 0, 'emotion': 1,
            'features': features, 'valid': torch.ones(length, dtype=torch.bool),
            'global': np.linspace(.01, .1, 65, dtype=np.float32)+i*.02,
            'upper': .4+.06*np.sin(clock[:, None]/(5+i)+np.arange(9)[None]/3)})
    targets = driver.prepare_targets(clips, [0, 1, 2])
    stats = driver.fit_statistics(clips, [0, 1], targets)
    return clips, targets, stats


def test_targets_align_native_windows_and_smoothed_and_raw_energy():
    clips, _, _ = fixture()
    clips[0]['upper'][:] = .3
    clips[0]['upper'][14, 2] = .9
    clips[1]['valid'][24:30] = False  # Remaining runs both shorter than32.
    targets = driver.prepare_targets(clips, [0, 1])
    assert set(targets) == {0} and targets[0]['starts'] == [0, 8]
    for row, start in enumerate(targets[0]['starts']):
        raw = clips[0]['upper'][start:start+32]
        smoothed = (raw[10:19]+raw[11:20]+raw[12:21])/3
        smooth_delta = np.diff(smoothed, axis=0)*25
        raw_delta = np.diff(raw[11:20], axis=0)*25
        smooth_expected = [np.sqrt(np.square(smooth_delta[:, group]).mean()) for group in driver.core.GROUPS]
        raw_expected = [np.sqrt(np.square(raw_delta[:, group]).mean()) for group in driver.core.GROUPS]
        np.testing.assert_allclose(targets[0]['energy'][row], smooth_expected, atol=1e-15)
        np.testing.assert_allclose(targets[0]['raw_energy'][row], raw_expected, atol=1e-15)
    assert targets[0]['raw_energy'][0, 0] > targets[0]['energy'][0, 0]


def test_statistics_and_training_strictly_read_fitting_ids_and_balance_clips():
    clips, targets, reference = fixture()
    expected = driver.training_data(clips, [0, 1], targets, reference)
    altered = copy.deepcopy(clips); altered_targets = copy.deepcopy(targets)
    for key in ('features', 'global', 'upper'):
        if torch.is_tensor(altered[2][key]):
            altered[2][key][:] = float('nan')
        else:
            altered[2][key][:] = np.nan
    altered_targets[2]['energy'][:] = np.nan
    stats = driver.fit_statistics(altered, [0, 1], altered_targets)
    assert stats['fit_clip_ids'] == ['clip0', 'clip1']
    for key in ('static_mean', 'static_std', 'prosody_std'):
        torch.testing.assert_close(stats[key], reference[key], rtol=0, atol=0)
    np.testing.assert_array_equal(stats['thresholds'], reference['thresholds'])
    data = driver.training_data(altered, [0, 1], altered_targets, stats)
    for key in data:
        torch.testing.assert_close(data[key], expected[key], rtol=0, atol=0)
    n0, n1 = len(targets[0]['starts']), len(targets[1]['starts'])
    assert n0 != n1 and len(data['weight']) == n0+n1
    torch.testing.assert_close(data['weight'][:n0].sum(), data['weight'][n0:].sum())
    torch.testing.assert_close(data['weight'].mean(), torch.tensor(1.))
    # One independent whole-clip context per clip, rather than window weighting.
    contexts = torch.stack([torch.cat((torch.as_tensor(clips[i]['global']),
        driver.core.prosody_mean(clips[i]['features'], clips[i]['valid']))) for i in (0, 1)])
    torch.testing.assert_close(stats['static_mean'], contexts.double().mean(0).float(), rtol=0, atol=0)


class AcousticOnly(dict):
    def __getitem__(self, key):
        if key not in ('features', 'valid', 'global'):
            raise AssertionError('Prediction attempted to read non-acoustic key: '+str(key))
        return super().__getitem__(key)


def test_inputs_never_read_motion_and_mismatch_preserves_original_static_context():
    clips, targets, stats = fixture()
    item, donor = AcousticOnly(clips[0]), AcousticOnly(clips[1])
    reference_context, reference_descriptor = driver.inputs(item, targets[0]['starts'], stats)
    for mode in ('real', 'static', 'reverse', 'shift', 'mismatch'):
        context, descriptor = driver.inputs(item, targets[0]['starts'], stats, mode, donor if mode == 'mismatch' else None)
        torch.testing.assert_close(context, reference_context, rtol=0, atol=0)
        if mode == 'static':
            assert torch.equal(descriptor, torch.zeros_like(descriptor))
        if mode == 'mismatch':
            assert not torch.equal(descriptor, reference_descriptor)
    modified = copy.deepcopy(clips[0]); modified['upper'][:] = np.nan
    context, descriptor = driver.inputs(modified, targets[0]['starts'], stats)
    torch.testing.assert_close(context, reference_context, rtol=0, atol=0)
    torch.testing.assert_close(descriptor, reference_descriptor, rtol=0, atol=0)


def test_donor_selection_is_same_cell_speaker_emotion_and_different_sentence():
    clips, targets, _ = fixture()
    # clip1 is the only eligible in-cell donor for clip0. Candidate outside the
    # passed cell has a lexically smaller ID and must never be considered.
    clips[2]['clip_id'] = 'aaa_outside'
    for reason in ('speaker', 'emotion', 'sentence', 'gap'):
        bad = copy.deepcopy(clips[1]); bad['clip_id'] = 'a_bad_'+reason
        if reason == 'speaker':
            bad['speaker'] = 5
        elif reason == 'emotion':
            bad['emotion'] = 7
        elif reason == 'sentence':
            bad['sentence'] = clips[0]['sentence']
        else:
            bad['valid'][10] = False
        clips.append(bad); targets[len(clips)-1] = copy.deepcopy(targets[1])
    ids = [0, 1, 3, 4, 5, 6]
    donors = driver.donor_map(clips, ids, targets)
    assert donors[0] == 1
    for i, j in donors.items():
        assert i in ids and j in ids and j != 2
        assert clips[i]['speaker'] == clips[j]['speaker'] and clips[i]['emotion'] == clips[j]['emotion']
        assert clips[i]['sentence'] != clips[j]['sentence']
        assert len(driver.c.old.runs(clips[j]['valid'].numpy())) == 1


def test_small_training_pair_preserves_saved_base_and_matching_orders(tmp_path):
    clips, targets, stats = fixture()
    data = driver.training_data(clips, [0, 1], targets, stats)
    model, matching = driver.train_pair(data, 20260918, 2, tmp_path, 'cpu')
    static = torch.load(tmp_path/'20260918_static.pt', weights_only=False)
    temporal = torch.load(tmp_path/'20260918_temporal.pt', weights_only=False)
    assert static['order_sha256'] == temporal['order_sha256'] == matching['order_sha256']
    assert matching['base_unchanged_exact'] and matching['trainable_temporal_parameters'] == 596
    for key, value in static['state'].items():
        torch.testing.assert_close(value, model.base.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(value, temporal['state']['base.'+key], rtol=0, atol=0)
    assert all(param.grad is None and not param.requires_grad for param in model.base.parameters())
    assert model.correction[-1].weight.abs().sum() > 0


def score_row(clip_id, sentence, target, probabilities):
    return {'clip_id': clip_id, 'sentence': sentence, 'speaker': 0, 'emotion': 1,
            'target': target, 'probabilities': probabilities}


def test_score_rows_ensembles_probabilities_before_brier_and_balances_clips():
    # First clip ensemble=.5 gives Brier.25; averaging individual-seed scores
    # would give.5 and is intentionally not the estimator used here.
    rows = [score_row('short', 's0', np.zeros((1, 4)),
        {'real': np.stack((np.zeros((1, 4)), np.ones((1, 4))))}),
        score_row('long', 's1', np.zeros((9, 4)), {'real': np.ones((2, 9, 4))})]
    result = driver.score_rows(rows, 'real')
    assert result['brier'] == .625  # Equal clips: (.25+1)/2, not equal windows.
    np.testing.assert_array_equal(result['group_brier'], [.625]*4)
    assert driver.score_rows(rows, 'real', 0)['brier'] == .5
    assert driver.score_rows(rows, 'real', 1)['brier'] == 1.


def test_assessment_rejects_better_reverse_even_when_real_beats_static():
    rows = []
    y = np.array([[0., 1., 0., 1.], [1., 0., 1., 0.]])
    for i in range(8):
        probabilities = {}
        for arm in driver.ARMS:
            p = (.2+.6*y) if arm == 'real' else ((.5+0*y) if arm in ('static', 'static_trained') else (.4+.2*y))
            probabilities[arm] = np.stack([p, p, p])
        rows.append(score_row(f'clip{i}', f'sentence{i}', y, probabilities))
    passed = driver.assessment(rows)
    assert passed['passed'] and all(passed['checks'].values())
    assert json.loads(json.dumps(passed, allow_nan=False))['passed'] is True
    for row in rows:
        row['probabilities']['reverse'] = np.stack([.1+.8*y]*3)
    failed = driver.assessment(rows)
    assert not failed['passed'] and not failed['checks']['interventions']
    assert failed['checks']['static_gain'] and failed['gains']['reverse']['ci95'][1] < 0
    assert json.loads(json.dumps(failed, allow_nan=False))['passed'] is False
