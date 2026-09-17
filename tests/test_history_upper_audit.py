"""Synthetic audit contracts; no experiment or sealed dataset files are read."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from scripts import audit_history_upper as audit
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_history_upper import teacher_probability


UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def curves(count=405, frames=96):
    time = torch.arange(frames, dtype=torch.float64) / 25
    target = (.3 + .1 * torch.sin(torch.arange(frames).float() / 5))[None, :, None].expand(count, -1, 52).clone()
    prediction = target + .02
    return {'schema': audit.SCHEMA, 'clip_id': ['synthetic_' + str(i) for i in range(count)],
            'target': target, 'b0': torch.zeros_like(target), 'valid': torch.ones(count, frames, dtype=torch.bool),
            'channel_mask': torch.ones(count, 52, dtype=torch.bool), 'times': time[None].expand(count, -1).clone(),
            'emotion_id': torch.arange(count).remainder(2), 'speaker_id': torch.arange(count).remainder(3),
            'noise_seeds': list(audit.SEEDS), 'decode_steps': 12,
            'predictions': {key: prediction.clone() for key in
                            ({'42/' + mode for mode in audit.MODES} | {f'{seed}/full' for seed in audit.SEEDS})}}


def load_fixture(monkeypatch, *, arm='scheduled_history'):
    recipe = {'schema': audit.SCHEMA, 'arm': arm, 'smoke': False, 'epochs': 12,
              'chunk': 16, 'history': 8, 'test_loaded': False, 'default_replaced': False,
              'fixed_final_epoch': True, 'frozen': {'system': 'system', 'audio': 'audio'},
              'scales': [.1] * 9, 'decode_steps': 12,
              'teacher_probability': [teacher_probability(i, 12) for i in range(12)]}
    digest = canonical_hash(recipe)
    c = curves()
    final = {'arm': arm, 'schema': audit.SCHEMA, 'completed_epochs': 12, 'recipe_sha256': digest,
             'scales': torch.full((9,), .1), 'total_steps': 1740}
    evaluation = {'schema': audit.SCHEMA, 'clips': 405, 'nonupper_exact': True,
                  'test_loaded': False, 'default_replaced': False,
                  'modes': {'42/' + mode: {'oracle_target_history': mode == 'oracle_history'} for mode in audit.MODES}}
    records = {'provenance.json': {'recipe': recipe, 'recipe_sha256': digest},
               'complete.json': {'recipe_sha256': digest, 'completed_epochs': 12,
                                 'frozen': recipe['frozen'].copy(), 'final_sha256': 'final', 'curves_sha256': 'curves'},
               'evaluation.json': evaluation}
    for i in range(1, 13):
        records[f'epoch{i:03d}.json'] = {'arm': arm, 'epoch': i, 'teacher_probability': recipe['teacher_probability'][i-1],
                                      'no_history_arm_ignores_all_history': arm == 'no_history',
                                      'loss': 1., 'total_steps': i*145, 'teacher_selected_count': 0,
                                      'history_available_count': 100}
    monkeypatch.setattr(audit, 'read', lambda path: records[Path(path).name])
    monkeypatch.setattr(audit, 'load_pt', lambda path: final if Path(path).name == 'final.pt' else c)
    monkeypatch.setattr(audit, 'sha', lambda path: Path(path).stem)
    return recipe, records, final, c, evaluation


def test_load_history_accepts_complete_bound_fixed_final_curves(monkeypatch):
    recipe, records, final, c, _ = load_fixture(monkeypatch)
    result = audit.load_history(Path('synthetic'), 'scheduled_history')
    assert result['curves'] is c and result['recipe'] is recipe
    assert result['bindings'] == {'final': 'final', 'curves': 'curves'}


@pytest.mark.parametrize('case', ['file_hash', 'recipe_hash', 'scale', 'native_clock', 'duplicate_id',
                                  'prediction_shape', 'missing_mode', 'seed_order', 'observed_nan',
                                  'oracle_flag', 'step_budget', 'nonzero_final_teacher', 'missing_upper', 'wrong_length', 'teacher_count'])
def test_load_history_rejects_broken_data_binding_or_scoring_contract(monkeypatch, case):
    recipe, records, final, c, evaluation = load_fixture(monkeypatch)
    if case == 'file_hash': records['complete.json']['curves_sha256'] = 'wrong'
    elif case == 'recipe_hash': records['provenance.json']['recipe_sha256'] = 'wrong'
    elif case == 'scale': final['scales'][0] = .2
    elif case == 'native_clock': c['times'][0, 1] += .001
    elif case == 'duplicate_id': c['clip_id'][1] = c['clip_id'][0]
    elif case == 'prediction_shape': c['predictions']['42/full'] = c['predictions']['42/full'][..., :51]
    elif case == 'missing_mode': del c['predictions']['42/oracle_history']
    elif case == 'seed_order': c['noise_seeds'] = [123, 42, 2026]
    elif case == 'observed_nan': c['predictions']['42/full'][0, 0, 41] = float('nan')
    elif case == 'oracle_flag': evaluation['modes']['42/oracle_history']['oracle_target_history'] = False
    elif case == 'step_budget': records['epoch012.json']['total_steps'] -= 1
    elif case == 'missing_upper': c['channel_mask'][0, 41] = False
    elif case == 'wrong_length':
        for key in ('target', 'b0', 'valid', 'times'): c[key] = c[key][:, :80]
        c['predictions'] = {key: value[:, :80] for key, value in c['predictions'].items()}
    elif case == 'teacher_count': records['epoch012.json']['teacher_selected_count'] = 1
    elif case == 'nonzero_final_teacher':
        recipe['teacher_probability'][-1] = .1
        records['epoch012.json']['teacher_probability'] = .1
        digest = canonical_hash(recipe)
        records['provenance.json']['recipe_sha256'] = records['complete.json']['recipe_sha256'] = final['recipe_sha256'] = digest
    with pytest.raises(ValueError):
        audit.load_history(Path('synthetic'), 'scheduled_history')


def test_chunk_profile_preserves_static_error_and_masked_nan_does_not_enter_scores():
    target = torch.zeros(2, 32, 52)
    pred = torch.zeros_like(target)
    valid = torch.ones(2, 32, dtype=torch.bool)
    channels = torch.ones(2, 52, dtype=torch.bool)
    pred[:, :16, UPPER] = 1.; pred[:, 16:, UPPER] = 3.
    valid[1, :16] = False
    pred[1, :16] = float('nan'); target[1, :16] = float('nan')
    channels[:, 41] = False
    pred[..., 41] = target[..., 41] = float('nan')
    result = audit.chunk_profile(pred, target, valid, channels)
    brows = result['brows'] if isinstance(result['brows'], list) else result['brows']['all_available']
    assert brows[0]['observed_values'] == 16*4
    assert brows[0]['raw_mse'] == brows[0]['mean_error_rms'] == 1.
    assert brows[0]['outside_fraction'] == 0.
    assert brows[1]['observed_values'] == 2*16*4
    assert brows[1]['raw_mse'] == 9.
    assert brows[1]['mean_error_rms'] == 3.
    assert brows[1]['outside_fraction'] == 1.
    assert brows[1]['mean_change_error_rms'] == brows[1]['prediction_mean_change_rms'] == 2.
    assert brows[1]['target_mean_change_rms'] == 0.


def test_chunk_profile_target_motion_is_not_misreported_as_error_drift():
    target = torch.zeros(1, 32, 52); target[:, 16:] = .6
    pred = target + .1
    result = audit.chunk_profile(pred, target, torch.ones(1, 32, dtype=torch.bool), torch.ones(1, 52, dtype=torch.bool))
    for value in result.values():
        rows = value if isinstance(value, list) else value['all_available']
        assert rows[1]['prediction_mean_change_rms'] == pytest.approx(.6, abs=1e-7)
        assert rows[1]['target_mean_change_rms'] == pytest.approx(.6, abs=1e-7)
        assert rows[1]['mean_change_error_rms'] == pytest.approx(0., abs=1e-7)


def main_fixture(monkeypatch, tmp_path):
    reference = curves(count=9)
    base = copy.deepcopy(reference)
    base['predictions'] = {f'{seed}/base': reference['predictions'][f'{seed}/full'].clone() for seed in audit.SEEDS}
    controls = {name: copy.deepcopy(reference) for name in ('state_aligned', 'state_white')}
    for c in controls.values():
        c['predictions'] = {f'{seed}/full': c['predictions'][f'{seed}/full'] for seed in audit.SEEDS}
    runs = {}
    for arm in audit.ARMS:
        c = copy.deepcopy(reference)
        if arm == 'scheduled_history':
            c['predictions']['42/empty'][:, 16:, UPPER] += .03
            c['predictions']['42/reverse_history'][:, 16:, UPPER] += .04
            c['predictions']['42/oracle_history'][:, 16:, UPPER] -= .01
        frozen = {'system': audit.state_hash({'weight': torch.tensor([1.])}), 'audio': audit.state_hash({'weight': torch.tensor([2.])})}
        recipe = {'arm': arm, 'initial': {'upper': 'same', 'local': 'same'}, 'baseline_curves_sha256': 'base', 'epochs': 12,
            'source_sha256': {'fake_source.py': 'fake_source'}, 'frozen': frozen,
            'source_bindings': {key: {'path': str(tmp_path/(key+'.pt')), 'sha256': key} for key in ('teacher', 'audio')}}
        runs[arm] = {'recipe': recipe, 'curves': c, 'bindings': {'final': arm + '_final', 'curves': arm + '_curves'},
                     'evaluation': {'modes': {key: {'generated_emotion_accuracy_nonindependent': .8} for key in c['predictions']}},
                     'epochs': [{'batch_noise_time_decision_sha256': str(i)} for i in range(12)]}
    matched = {'initial_equal': True, 'random_streams_equal': True,
               'initial_sha256': {arm: runs[arm]['recipe']['initial'] for arm in audit.ARMS},
               'epoch_draw_sha256': {arm: [str(i) for i in range(12)] for arm in audit.ARMS}}
    picks = [{'clip_id': reference['clip_id'][i], 'speaker': 'person_' + str(i//3)} for i in range(9)]
    previous = {'nine_plot_clips': picks, 'jobs': [{'speaker': 'person_' + str(i), 'input': f'video_npz/person_{i}.npz'} for i in range(3)]}
    def read(path):
        path = Path(path)
        if path.name == 'status.json': return {'status': 'complete', 'epochs_per_arm': 12, 'smoke': False, 'seconds': 123.}
        if path.name == 'matched_audit.json': return matched
        if path.parent.name == 'baseline': return {'curves_sha256': 'base'}
        if path.parent.name == 'state': return {name: name for name in controls}
        if path.parent.name == 'previous': return previous
        raise AssertionError(str(path))
    monkeypatch.setattr(audit, 'read', read)
    monkeypatch.setattr(audit, 'load_history', lambda folder, arm: runs[arm])
    def load_pt(path):
        path = Path(path)
        if path.name == 'teacher.pt': return {'system': {'weight': torch.tensor([1.])}}
        if path.name == 'audio.pt': return {'audio': {'weight': torch.tensor([2.])}}
        return base if path.parent.name == 'baseline' else controls[path.stem.removesuffix('_curves')]
    monkeypatch.setattr(audit, 'load_pt', load_pt)
    monkeypatch.setattr(audit, 'sha', lambda path: 'base' if Path(path).parent.name == 'baseline' else Path(path).stem.removesuffix('_curves'))
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['audit_history_upper.py', '--history-run', str(tmp_path/'history'),
        '--state-run', str(tmp_path/'state'), '--baseline-run', str(tmp_path/'baseline'),
        '--previous-visual', str(tmp_path/'previous'), '--output', str(output)])
    return runs, base, controls, output


@pytest.mark.parametrize('case,match', [('first_chunk', 'empty first chunk'), ('no_history', 'unexpectedly uses history'),
                                       ('nonupper', 'Nonupper'), ('metadata', 'native metadata')])
def test_main_rejects_history_leakage_nonupper_changes_and_metadata_mismatch(monkeypatch, tmp_path, case, match):
    runs, _, _, _ = main_fixture(monkeypatch, tmp_path)
    if case == 'first_chunk': runs['scheduled_history']['curves']['predictions']['42/oracle_history'][0, 0, 41] += .1
    elif case == 'no_history': runs['no_history']['curves']['predictions']['42/empty'][0, 20, 41] += .1
    elif case == 'nonupper': runs['scheduled_history']['curves']['predictions']['123/full'][0, 20, 17] += .1
    elif case == 'metadata': runs['scheduled_history']['curves']['target'][0, 0, 41] += .1
    with pytest.raises(ValueError, match=match):
        audit.main()


def test_main_keeps_oracle_out_of_deployment_and_exports_unambiguous_tensor_shapes(monkeypatch, tmp_path):
    runs, _, _, output = main_fixture(monkeypatch, tmp_path)
    audit.main()
    report = json.loads((output/'audit.json').read_text())
    assert set(report['deployment']) == {'no_history', 'scheduled_history', 'state_aligned', 'state_white'}
    for arm in audit.ARMS:
        assert '42/oracle_history' not in report['deployable_interventions_seed42'][arm]
        assert 'single_seed_interventions' not in report['deployment'][arm]
        assert 'brows' in report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY'][arm]
    assert report['test_loaded'] is False and report['default_replaced'] is False
    assert report['first_chunk_history_interventions_equal'] and report['no_history_interventions_equal']
    with np.load(output/'visual/nine_clip_curves.npz', allow_pickle=False) as values:
        assert values['motions'].shape == (6, 9, 96, 52)
        assert values['times'].shape == values['valid'].shape == (9, 96)
        assert values['channel_mask'].shape == (9, 52)
        assert str(values['mode_names'][-1]) == 'Empty history (ablation)'
        assert not any('ORACLE' in str(name) for name in values['mode_names'])
        np.testing.assert_array_equal(values['motions'][-1], runs['scheduled_history']['curves']['predictions']['42/empty'].numpy())
    with np.load(output/'visual/video_npz/person_0.npz', allow_pickle=False) as values:
        assert values['motions'].shape == (6, 96, 52)
        assert values['valid'].shape == values['times'].shape == (96,)
        assert values['channel_mask'].shape == (52,)
        assert int(values['noise_seed']) == 42
    visual = json.loads((output/'visual/provenance.json').read_text())
    assert visual['oracle_in_main_visual'] is False


def test_common_profiles_prevent_changing_clip_composition():
    target = torch.zeros(2, 32, 52); pred = torch.zeros_like(target)
    pred[0] = .2; pred[1] = .8
    valid = torch.ones(2, 32, dtype=torch.bool); valid[1, 16:] = False
    channels = torch.ones(2, 52, dtype=torch.bool)
    result = audit.common_clip_profiles(pred, target, valid, channels)
    assert result['clip_indices'] == [0] and result['clip_count'] == 1
    assert result['profiles']['brows'][0]['raw_mse'] == pytest.approx(.04)
    assert result['profiles']['brows'][1]['raw_mse'] == pytest.approx(.04)


def test_main_rejects_invalid_frame_upper_changed(monkeypatch, tmp_path):
    runs, base, controls, _ = main_fixture(monkeypatch, tmp_path)
    for c in [base, *controls.values(), *[run['curves'] for run in runs.values()]]:
        c['valid'][0, 5] = False
    for value in runs['scheduled_history']['curves']['predictions'].values():
        value[0, 5, 41] += .1
    with pytest.raises(ValueError, match='Invalid-frame baseline'):
        audit.main()
