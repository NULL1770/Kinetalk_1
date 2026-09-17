import json

import numpy as np
import pytest
import torch

from scripts.export_full_staged_examples import ARKIT_NAMES, RUN_SCHEMA, export, sha


def setup(tmp_path):
    run, previous = tmp_path / 'run', tmp_path / 'previous'
    (run / 'dynamics').mkdir(parents=True); previous.mkdir()
    ids = [f'person{i//3}_clip{i}' for i in range(9)]
    picks = [{'index': i, 'clip_id': cid, 'speaker': f'person{i//3}', 'emotion': 'angry'} for i, cid in enumerate(ids)]
    target = torch.arange(9*8*52).reshape(9, 8, 52).float() / 4000
    valid = torch.ones(9, 8, dtype=torch.bool); valid[:, -1] = False
    times = torch.arange(8, dtype=torch.float64)[None].repeat(9, 1) * .04
    channels = torch.ones(9, 52, dtype=torch.bool)
    predictions = {f'42/{mode}': target.clone() for mode in ('base', 'full', 'static_state', 'oracle_state')}
    predictions['42/full'][..., 43] += .123
    curves = {'schema': RUN_SCHEMA, 'stage': 'dynamics', 'clip_id': ids, 'target': target, 'valid': valid,
              'times': times, 'channel_mask': channels, 'b0': target * .5, 'predictions': predictions}
    save_curves(run, curves)
    (previous / 'provenance.json').write_text(json.dumps({'selection_uses_metadata_only': True,
        'outcome_based_selection': False, 'nine_plot_clips': picks, 'videos': []}), encoding='utf8')
    np.savez(previous / 'nine_clip_curves.npz', channels=np.asarray(ARKIT_NAMES), clip_id=np.asarray(ids),
             mode_names=np.asarray(['GT']), motions=target.numpy()[None], times=times.numpy(),
             valid=valid.numpy(), channel_mask=channels.numpy())
    return run, previous, curves


def save_curves(run, curves):
    path = run / 'dynamics/curves.pt'; torch.save(curves, path)
    (run / 'dynamics/complete.json').write_text(json.dumps({'stage': 'dynamics', 'curves_sha256': sha(path), 'completed_epochs': 12}))


def test_saved_curves_export_is_exact_reordered_by_lock_and_not_rendered(tmp_path):
    run, previous, curves = setup(tmp_path)
    permutation = torch.tensor([8, 2, 3, 1, 5, 6, 4, 0, 7])
    for key in ('target', 'valid', 'times', 'channel_mask', 'b0'): curves[key] = curves[key][permutation]
    curves['clip_id'] = [curves['clip_id'][i] for i in permutation]
    curves['predictions'] = {k: v[permutation] for k, v in curves['predictions'].items()}
    save_curves(run, curves)
    report = export(run, previous, tmp_path / 'out', make_plots=False)
    assert report['actual_rendering_performed'] is False and report['model_loaded'] is False
    assert len(report['videos']) == 3
    with np.load(tmp_path / 'out/nine_clip_curves.npz', allow_pickle=False) as values:
        assert values['motions'].shape == (6, 9, 8, 52)
        assert values['clip_id'][0] == 'person0_clip0'
        i = curves['clip_id'].index('person0_clip0')
        np.testing.assert_array_equal(values['motions'][3, 0], curves['predictions']['42/full'][i].numpy())
        assert values['motions'].max() > 1  # Export did not silently display-clamp raw coefficients.


@pytest.mark.parametrize('change', ['missing_clip', 'time_shift', 'mouth_drift'])
def test_export_rejects_wrong_selection_clock_or_unprotected_mouth(tmp_path, change):
    run, previous, curves = setup(tmp_path)
    if change == 'missing_clip': curves['clip_id'][0] = 'different_clip'
    elif change == 'time_shift': curves['times'] += .04
    else: curves['predictions']['42/full'][0, 0, 17] += .2
    save_curves(run, curves)
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError): export(run, previous, output, make_plots=False)
    assert not output.exists()


def bind_audio_and_checkpoints(run, curves, *, difference=1e-5, changed_weight=False):
    import copy
    (run / 'audio').mkdir()
    audio = copy.deepcopy(curves)
    audio['stage'] = 'audio'
    audio['predictions'] = {'42/full': curves['predictions']['42/base'] + difference}
    torch.save(audio, run / 'audio/curves.pt')
    for stage in ('audio', 'dynamics'):
        checkpoint = {'schema': RUN_SCHEMA, 'stage': stage, 'completed_epochs': 12, 'inference_only': True,
                      'recipe_sha256': 'test_recipe', 'config': {'model': {'dim': 2}},
                      'system': {'weight': torch.tensor([.4, .9])}, 'audio': {'weight': torch.tensor([.1, .2])}}
        if changed_weight and stage == 'dynamics': checkpoint['audio']['weight'][0] += .01
        torch.save(checkpoint, run / stage / 'final.pt')
        (run / stage / 'complete.json').write_text(json.dumps({'stage': stage, 'completed_epochs': 12,
            'curves_sha256': sha(run / stage / 'curves.pt'), 'final_sha256': sha(run / stage / 'final.pt')}))


def test_roundoff_requires_bitwise_frozen_weights_and_is_reported(tmp_path):
    run, previous, curves = setup(tmp_path)
    bind_audio_and_checkpoints(run, curves)
    result = export(run, previous, tmp_path / 'out', make_plots=False)
    audit = result['source_curves']['audio']['frozen_audit']
    assert audit['frozen_modules_equal_exactly'] is True
    assert audit['cross_stage_output_equal_exactly'] is False
    assert audit['max_absolute_difference'] > 0
    assert (tmp_path / 'out/frozen_cross_stage_audit.json').is_file()


@pytest.mark.parametrize('difference,changed_weight', [(0.01, False), (1e-5, True)])
def test_frozen_audit_rejects_large_output_or_any_weight_drift(tmp_path, difference, changed_weight):
    run, previous, curves = setup(tmp_path)
    bind_audio_and_checkpoints(run, curves, difference=difference, changed_weight=changed_weight)
    with pytest.raises(ValueError, match='numerical limits|tensor changed'):
        export(run, previous, tmp_path / 'rejected', make_plots=False)
    assert not (tmp_path / 'rejected').exists()
