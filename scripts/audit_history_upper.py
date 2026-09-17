"""Read-only audit and fixed historical export of the paired history experiment.

No model inference, fitting, epoch selection, or new data roles are performed.
Only the three generated-history seeds are deployment scores. Oracle remains a
separate diagnostic even though the training report stores all interventions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import (
    GROUPS, NOT_UPPER, SEEDS, compare, load_pt, metadata_equal, read, sha,
    summarize, validate_curves,
)
from scripts.export_full_staged_examples import ARKIT_NAMES
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import state_hash
from scripts.train_history_upper import SCHEMA, CHUNK, HISTORY, CC, boundary_report

ARMS = ('no_history', 'scheduled_history')
MODES = ('full', 'empty', 'reverse_history', 'static', 'reverse', 'oracle_history')


def chunk_profile(pred, target, valid, channel_mask, chunk=CHUNK):
    """Uncentered errors and between-chunk mean changes, using observed data.

    These are descriptive rollout diagnostics. A changing target trajectory can
    itself cause a changing error profile; late error alone is not proof of drift.
    """
    observed = valid[..., None] & channel_mask[:, None]
    result = {}
    for name, channels in GROUPS.items():
        p, t, mask = pred[..., list(channels)].double(), target[..., list(channels)].double(), observed[..., list(channels)]
        rows, previous = [], None
        for start in range(0, pred.shape[1], chunk):
            stop = min(start + chunk, pred.shape[1]); m = mask[:, start:stop]
            pp, tt = p[:, start:stop], t[:, start:stop]
            count = m.sum(1); usable = count > 0
            pm = torch.where(m, pp, 0.).sum(1) / count.clamp_min(1)
            tm = torch.where(m, tt, 0.).sum(1) / count.clamp_min(1)
            row = {'start_frame': start, 'stop_frame_exclusive': stop, 'observed_values': int(m.sum())}
            if m.any():
                row.update(raw_mse=float((pp-tt)[m].square().mean()),
                    signed_error_mean=float((pp-tt)[m].mean()),
                    mean_error_rms=float((pm-tm)[usable].square().mean().sqrt()),
                    outside_fraction=float(((pp[m] < 0) | (pp[m] > 1)).double().mean()))
            else:
                row.update(raw_mse=None, signed_error_mean=None, mean_error_rms=None, outside_fraction=None)
            if previous is not None:
                last_p, last_t, last_valid = previous; shared = last_valid & usable
                row['mean_change_error_rms'] = float(((pm-last_p)-(tm-last_t))[shared].square().mean().sqrt()) if shared.any() else None
                row['prediction_mean_change_rms'] = float((pm-last_p)[shared].square().mean().sqrt()) if shared.any() else None
                row['target_mean_change_rms'] = float((tm-last_t)[shared].square().mean().sqrt()) if shared.any() else None
            previous = (pm, tm, usable); rows.append(row)
        result[name] = rows
    return result


def common_clip_profiles(pred, target, valid, channel_mask, chunk=CHUNK):
    # One declared coverage rule used for every chunk/mode/arm. Requiring half
    # each chunk avoids treating clips with only one late frame as comparable.
    eligible = torch.ones(len(valid), dtype=torch.bool)
    for start in range(0, valid.shape[1], chunk):
        length = min(chunk, valid.shape[1]-start)
        eligible &= valid[:, start:start+length].sum(1) >= (length+1)//2
    return {'coverage_rule': 'At least half the native frames observed in every chunk; same clips for all modes',
        'clip_count': int(eligible.sum()), 'clip_indices': eligible.nonzero(as_tuple=True)[0].tolist(),
        'profiles': chunk_profile(pred[eligible], target[eligible], valid[eligible], channel_mask[eligible], chunk)}


def load_history(folder, arm):
    provenance, complete, evaluation = (read(folder/name) for name in ('provenance.json', 'complete.json', 'evaluation.json'))
    recipe = provenance['recipe']; digest = canonical_hash(recipe)
    if (recipe['schema'] != SCHEMA or recipe['arm'] != arm or recipe['smoke']
        or recipe['epochs'] != 12 or recipe['chunk'] != CHUNK or recipe['history'] != HISTORY
        or recipe['test_loaded'] or recipe['default_replaced'] or not recipe['fixed_final_epoch']
        or provenance['recipe_sha256'] != digest or complete['recipe_sha256'] != digest
        or complete['completed_epochs'] != 12 or complete['frozen'] != recipe['frozen']):
        raise ValueError('History protocol or completed recipe differs: '+arm)
    bindings = {name: sha(folder/(name+'.pt')) for name in ('final', 'curves')}
    if any(complete[name+'_sha256'] != value for name, value in bindings.items()):
        raise ValueError('History file binding differs: '+arm)
    final, curves = load_pt(folder/'final.pt'), load_pt(folder/'curves.pt')
    if (final['arm'] != arm or final['schema'] != SCHEMA or curves['schema'] != SCHEMA
        or final['completed_epochs'] != 12 or final['recipe_sha256'] != digest
        or not torch.equal(final['scales'], torch.tensor(recipe['scales'], dtype=final['scales'].dtype))
        or curves['decode_steps'] != recipe['decode_steps']):
        raise ValueError('History final checkpoint or curve schema differs')
    validate_curves(curves)
    if (len(curves['clip_id']) != 405 or curves['target'].shape[1] != 96
        or not curves['channel_mask'][:, CC].all()):
        raise ValueError('Expected locked 405 development clips, 96 frames and nine observed upper channels')
    expected = {'42/'+mode for mode in MODES} | {f'{seed}/full' for seed in SEEDS}
    if set(curves['predictions']) != expected:
        raise ValueError('Unexpected prediction keys')
    epochs = [read(folder/f'epoch{i:03d}.json') for i in range(1, 13)]
    for i, row in enumerate(epochs, 1):
        if (row['arm'] != arm or row['epoch'] != i or row['teacher_probability'] != recipe['teacher_probability'][i-1]
            or row['no_history_arm_ignores_all_history'] != (arm == 'no_history')
            or not np.isfinite(row['loss']) or row['total_steps'] != i*145
            or (row['teacher_probability'] == 0 and row['teacher_selected_count'] != 0)):
            raise ValueError('Invalid epoch record')
    if final['total_steps'] != epochs[-1]['total_steps'] or any(recipe['teacher_probability'][7:]):
        raise ValueError('Budget or generated-history final schedule differs')
    if (evaluation['schema'] != SCHEMA or evaluation['clips'] != 405 or not evaluation['nonupper_exact']
        or evaluation['test_loaded'] or evaluation['default_replaced']):
        raise ValueError('Evaluation contract differs')
    for mode in MODES:
        if evaluation['modes']['42/'+mode]['oracle_target_history'] != (mode == 'oracle_history'):
            raise ValueError('Oracle classification differs')
    return {'recipe': recipe, 'curves': curves, 'evaluation': evaluation, 'epochs': epochs, 'bindings': bindings}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('history-run', 'state-run', 'baseline-run', 'previous-visual', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    if a.output.exists(): raise FileExistsError('Fresh audit output required')
    status, matched = read(a.history_run/'status.json'), read(a.history_run/'matched_audit.json')
    if status['status'] != 'complete' or status['epochs_per_arm'] != 12 or status['smoke']:
        raise ValueError('History training is not complete')
    runs = {arm: load_history(a.history_run/arm, arm) for arm in ARMS}
    recipes = [dict(runs[arm]['recipe']) for arm in ARMS]
    for recipe in recipes: recipe.pop('arm')
    if recipes[0] != recipes[1] or not matched['initial_equal'] or not matched['random_streams_equal']:
        raise ValueError('Paired recipes or initial/random audit differs')
    code = Path(__file__).resolve().parents[1]
    for name, digest in recipes[0]['source_sha256'].items():
        if sha(code/name) != digest: raise ValueError('Training source has changed: '+name)
    source_hashes = {}
    for name, key, frozen_name in [('teacher', 'system', 'system'), ('audio', 'audio', 'audio')]:
        binding = recipes[0]['source_bindings'][name]; path = Path(binding['path'])
        if sha(path) != binding['sha256']: raise ValueError('Source checkpoint binding changed: '+name)
        source = load_pt(path)
        source_hashes[frozen_name] = state_hash(source[key])
    if source_hashes != recipes[0]['frozen']:
        raise ValueError('Frozen subsystem hashes differ from source checkpoints')
    for arm in ARMS:
        if (matched['initial_sha256'][arm] != runs[arm]['recipe']['initial'] or
            matched['epoch_draw_sha256'][arm] != [row['batch_noise_time_decision_sha256'] for row in runs[arm]['epochs']]):
            raise ValueError('Initial or random-stream records differ')
    if matched['epoch_draw_sha256'][ARMS[0]] != matched['epoch_draw_sha256'][ARMS[1]]:
        raise ValueError('Epoch random streams differ')
    base_path = a.baseline_run/'curves.pt'; base_hash = sha(base_path)
    if base_hash != read(a.baseline_run/'complete.json')['curves_sha256']:
        raise ValueError('Base curve binding differs')
    base = load_pt(base_path); controls = {}; state_complete = read(a.state_run/'complete.json')
    for name in ('state_aligned', 'state_white'):
        path = a.state_run/(name+'_curves.pt')
        if sha(path) != state_complete[name]: raise ValueError('State comparison binding differs')
        controls[name] = load_pt(path)
    reference = runs['no_history']['curves']
    for c in [base, *controls.values(), runs['scheduled_history']['curves']]: metadata_equal(reference, c)
    if any(run['recipe']['baseline_curves_sha256'] != base_hash for run in runs.values()):
        raise ValueError('History baseline source differs')
    summaries = {}; diagnostics = {}; oracle = {}; profiles = {}; common_profiles = {}; boundaries = {}; emotion = {}
    for arm, run in runs.items():
        c = run['curves']; full = c['predictions']['42/full']
        for key, pred in c['predictions'].items():
            seed, mode = key.split('/')
            if not torch.equal(pred[..., list(NOT_UPPER)], base['predictions'][seed+'/base'][..., list(NOT_UPPER)]):
                raise ValueError('Nonupper protection failed: '+arm+'/'+key)
            if not torch.equal(pred[~c['valid']], base['predictions'][seed+'/base'][~c['valid']]):
                raise ValueError('Invalid-frame baseline protection failed: '+arm+'/'+key)
            if mode in ('empty', 'reverse_history', 'oracle_history'):
                if not torch.equal(pred[:, :CHUNK], full[:, :CHUNK]):
                    raise ValueError('History intervention affected the empty first chunk')
                if arm == 'no_history' and not torch.equal(pred, full):
                    raise ValueError('No-history arm unexpectedly uses history')
        score = summarize(c, c['emotion_id'])
        interventions = score.pop('single_seed_interventions')
        oracle[arm] = interventions.pop('42/oracle_history')
        diagnostics[arm] = interventions; summaries[arm] = score
        profiles[arm] = {key: chunk_profile(pred, c['target'], c['valid'], c['channel_mask']) for key, pred in c['predictions'].items()}
        common_profiles[arm] = {key: common_clip_profiles(pred, c['target'], c['valid'], c['channel_mask']) for key, pred in c['predictions'].items()}
        boundaries[arm] = {key: boundary_report(pred, c['target'], c['valid']) for key, pred in c['predictions'].items()}
        emotion[arm] = {key: value['generated_emotion_accuracy_nonindependent'] for key, value in run['evaluation']['modes'].items()}
    for name, c in controls.items():
        score = summarize(c, c['emotion_id']); score.pop('single_seed_interventions', None); summaries[name] = score
    report = {'schema': 'history_result_audit_v1', 'training_seconds': status['seconds'], 'clips': 405,
        'scope': 'Repeated internal development; not sealed test; fixed final epoch and three fixed seeds',
        'paired_audit': matched, 'bindings': {arm: run['bindings'] for arm, run in runs.items()},
        'training_sources_and_frozen_checkpoint_hashes_verified': True,
        'control_bindings': {'base': base_hash, **{key: state_complete[key] for key in controls}},
        'deployment': summaries, 'history_minus_no_history': compare(summaries['scheduled_history'], summaries['no_history']),
        'history_minus_previous_aligned': compare(summaries['scheduled_history'], summaries['state_aligned']),
        'deployable_interventions_seed42': diagnostics, 'ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY': oracle,
        'per_chunk_profiles_all_modes': profiles, 'boundary_displacement_all_modes': boundaries,
        'common_clip_profiles_all_modes': common_profiles,
        'motion_teacher_emotion_accuracy_NONINDEPENDENT_all_modes': emotion,
        'nonupper_exact': True, 'invalid_frames_equal_baseline': True, 'first_chunk_history_interventions_equal': True,
        'no_history_interventions_equal': True, 'test_loaded': False, 'default_replaced': False,
        'limitations': ['Only the two new arms isolate history; comparisons to old controls also change local trunk, mean handling and chunking.',
            'Initial hashes were recorded at runtime; this trainer did not save initial parameter snapshots.',
            'Profile error by chunk is descriptive and does not alone identify recursive drift.',
            'Audio has offline full-window context; this is not causal audio generation.',
            'The nine modeled channels omit blink and gaze. Forty-three copied channels preserve prior output, not proof of perceptual quality.',
            'Tracked reference coefficients are pseudo-labels, not independently verified facial motion truth.']}
    previous = read(a.previous_visual/'provenance.json'); picks = previous['nine_plot_clips']
    if len(picks) != 9 or len({row['clip_id'] for row in picks}) != 9: raise ValueError('Historical selection differs')
    ids = [reference['clip_id'].index(row['clip_id']) for row in picks]
    names = ['Tracked reference', 'Previous audio mean + aligned', 'Previous audio mean + flow',
             'No history', 'Generated history', 'Empty history (ablation)']
    values = [reference['target'], controls['state_aligned']['predictions']['42/full'], controls['state_white']['predictions']['42/full'],
        reference['predictions']['42/full'], runs['scheduled_history']['curves']['predictions']['42/full'],
        runs['scheduled_history']['curves']['predictions']['42/empty']]
    motions = torch.stack([value[ids] for value in values]).numpy()
    data = {key: reference[key][ids].numpy() for key in ('times', 'valid', 'channel_mask')}
    data.update(channels=np.asarray(ARKIT_NAMES), clip_id=np.asarray([row['clip_id'] for row in picks]))
    a.output.mkdir(parents=True); visual = a.output/'visual'; (visual/'video_npz').mkdir(parents=True)
    save_json(a.output/'audit.json', report)
    np.savez_compressed(visual/'nine_clip_curves.npz', **data, motions=motions, mode_names=np.asarray(names))
    for job in previous['jobs']:
        index = next(i for i, row in enumerate(picks) if row['speaker'] == job['speaker'])
        np.savez_compressed(visual/'video_npz'/f"{job['speaker']}.npz", motions=motions[:, index], mode_names=np.asarray(names),
            channels=data['channels'], noise_seed=np.asarray(42), **{key: data[key][index] for key in ('times', 'valid', 'channel_mask', 'clip_id')})
    save_json(visual/'provenance.json', {'nine_plot_clips': picks, 'jobs': previous['jobs'], 'mode_names': names,
        'bindings': report['bindings'], 'control_bindings': report['control_bindings'], 'noise_seed': 42,
        'selection_unchanged': True, 'raw_clamped': False, 'gain_or_lag_fitted': False,
        'oracle_in_main_visual': False, 'test_loaded': False})
    manifest = {str(path.relative_to(a.output)): sha(path) for path in a.output.rglob('*') if path.is_file()}
    save_json(a.output/'complete.json', {'status': 'complete', 'files': manifest})
    print('HISTORY_RESULT_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__': main()
