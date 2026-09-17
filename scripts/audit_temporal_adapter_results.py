"""Read saved three-arm endpoints, recompute scores, and export fixed review clips.

No sampling, fitting, teacher inference, or large reconstructed files are written.
The 608-clip set is held out only from the new updates, not inherited supervision.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_prefix_upper as p
from scripts.audit_audio_prefix_results import compare_tree
from scripts.audit_temporal_repair import read, sha, load_pt, metadata_equal, summarize
from scripts.evaluate_audio_prefix_adaptation import _composition_invariants
from scripts.evaluate_prefix_formal import _same_bits
from scripts.evaluate_temporal_adapter_transfer import _score_predictions, compose_dc_upper
from scripts.train_audio_prefix_adaptation import compose_dc
from scripts.train_temporal_adapter_transfer import ARMS, SCHEMA, TRAIN_SEED, SPLIT_SEED, check_gradients
from kinetalk_b0.models.temporal_local_adapter import TemporalLocalAdapter


MODES = ('Old audio mean + flow', 'Previous adapted local + audio mean',
         'New frozen local + audio mean', 'New rank8 adapter + audio mean', 'New full local + audio mean')
FULL_KEYS = {f'{seed}/full' for seed in p.r.SEEDS}
LOCAL_KEYS = {'42/local_static', '42/local_reverse'}
DEPLOY_KEYS = FULL_KEYS | LOCAL_KEYS | {'42/empty', '42/reverse_history', '42/static', '42/reverse'}
ORACLE_KEYS = {'42/oracle_history', '42/oracle_reverse_history'}
COMPLETE_FILES = {'final.pt', 'curves.pt', 'dc_curves.pt', 'transfer_curves.pt', 'fit_curves.pt',
                  'evaluation.json', 'dc_evaluation.json', 'transfer_evaluation.json',
                  'fit_evaluation.json', 'feature_drift.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reconstruct_compact(compact, bases, basepath, basehash, *, dc=False):
    """Independent reconstruction also rejects ignored, corrupt invalid payloads."""
    require(compact['storage_schema'] == 'upper9_with_exact_bound_base_v1'
            and compact['upper_indices'] == list(p.CC)
            and compact['baseline_sha256'] == basehash
            and Path(compact['baseline_path']) == Path(basepath)
            and compact['nonupper_invalid_exact_before_storage'] is True,
            'Compact storage binding differs')
    metadata_equal(bases, compact)
    valid = compact['valid']
    require(valid.dtype == torch.bool and valid.shape == (405, 96) and bool(valid.any(1).all()),
            'Expected complete native 405-clip development archive')
    require(compact['times'].shape == valid.shape and bool(torch.isfinite(compact['times']).all())
            and torch.allclose(compact['times'][:, 1:] - compact['times'][:, :-1],
                               torch.full_like(compact['times'][:, 1:], .04), atol=1e-7, rtol=1e-5)
            and compact['channel_mask'].dtype == torch.bool and compact['channel_mask'].shape == (405, 52)
            and bool(compact['channel_mask'][:, p.CC].all()), 'Development native clock/masks differ')
    expected = DEPLOY_KEYS if dc else DEPLOY_KEYS | ORACLE_KEYS
    require(set(compact['upper_predictions9']) == expected, 'Compact prediction modes differ')
    require(compact['noise_seeds'] == list(p.r.SEEDS) and compact['decode_steps'] == 12,
            'Compact seed/solver contract differs')
    predictions = {}
    for key, upper in compact['upper_predictions9'].items():
        baseline = bases['predictions'][key.split('/')[0] + '/base']
        require(upper.dtype == torch.float32 and upper.shape == (*valid.shape, 9)
                and bool(torch.isfinite(upper[valid]).all()), 'Invalid upper payload: ' + key)
        require(_same_bits(upper[~valid], baseline[..., p.CC][~valid]),
                'Stored invalid upper payload differs from bound base: ' + key)
        prediction = baseline.clone()
        prediction[..., p.CC] = torch.where(valid[..., None], upper, baseline[..., p.CC])
        require(_same_bits(prediction[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)])
                and _same_bits(prediction[~valid], baseline[~valid]), 'Reconstruction changed protected values')
        require(_same_bits(prediction[..., p.CC], upper), 'Compact roundtrip changed upper payload')
        predictions[key] = prediction
    return {**compact, 'predictions': predictions}


def validate_split(selection, historical):
    require(selection['schema'] == SCHEMA and selection['seed'] == SPLIT_SEED
            and selection['uses_motion_for_selection'] is False and selection['test_loaded'] is False,
            'Sentence selection policy differs')
    require(selection['split'] == historical['split'], 'Historical sentence split differs')
    fit, hold = selection['split']['fit'], selection['split']['internal_sentence_holdout']
    fit_ix, hold_ix = selection['fit_indices'], selection['holdout_indices']
    require(len(fit['clips']) == len(fit_ix) == 1707 and len(hold['clips']) == len(hold_ix) == 608,
            'Expected exact 1707/608 partition')
    require(len(set(fit['clips'] + hold['clips'])) == 2315
            and not set(fit['sentences']) & set(hold['sentences'])
            and fit_ix == sorted(set(fit_ix)) and hold_ix == sorted(set(hold_ix))
            and sorted(fit_ix + hold_ix) == list(range(2315)), 'Partition overlap, index, or membership differs')


def transfer_metadata_equal(reference, candidate):
    for key in ('clip_id', 'sentence_id'):
        require(reference[key] == candidate[key], 'Transfer metadata differs: ' + key)
    for key in ('target_upper9', 'valid', 'times', 'channel_mask_upper', 'emotion_id', 'speaker_id', 'static_upper'):
        require(_same_bits(reference[key], candidate[key]), 'Transfer tensor metadata differs: ' + key)


def score_transfer(curves, report, selection, where):
    hold = selection['split']['internal_sentence_holdout']
    valid, target = curves['valid'], curves['target_upper9']
    require(curves['schema'] == SCHEMA and curves['clip_id'] == hold['clips']
            and sorted(set(curves['sentence_id'])) == hold['sentences']
            and valid.shape == (608, 96) and valid.dtype == torch.bool and bool(valid.any(1).all())
            and target.shape == (608, 96, 9) and curves['channel_mask_upper'].shape == (608, 9)
            and bool(curves['channel_mask_upper'].all()), 'Transfer metadata scope differs: ' + where)
    require(curves['upper_channel_indices'] == list(p.CC) and curves['noise_seeds'] == list(p.r.SEEDS)
            and curves['decode_steps'] == 12 and curves['GT_was_input'] is False
            and curves['fullface_deployment_evaluation'] is False and curves['dc_saved'] is False,
            'Transfer inference/archive declaration differs: ' + where)
    require(report['clips'] == 608 and report['sentence_count'] == len(hold['sentences'])
            and report['per_clip_order'] == curves['clip_id'] and report['GT_was_input'] is False
            and report['oracle_modes'] == [] and report['teacher_readout_scored'] is False
            and report['nonupper_scored'] is False and report['test_loaded'] is False,
            'Transfer report scope differs: ' + where)
    require(curves['times'].shape == valid.shape and bool(torch.isfinite(curves['times']).all())
            and torch.allclose(curves['times'][:, 1:] - curves['times'][:, :-1],
                               torch.full_like(curves['times'][:, 1:], .04), atol=1e-7, rtol=1e-5),
            'Transfer native clock differs')
    predictions = curves['upper_predictions9']
    require(set(predictions) == FULL_KEYS | LOCAL_KEYS and bool(torch.isfinite(target).all())
            and not bool(target[~valid].count_nonzero()), 'Transfer modes/target placeholders differ')
    for key, value in predictions.items():
        require(value.dtype == torch.float32 and value.shape == target.shape
                and bool(torch.isfinite(value).all()) and not bool(value[~valid].count_nonzero()),
                'Invalid transfer upper9 payload: ' + key)
    score_args = (target, valid, curves['channel_mask_upper'], curves['emotion_id'])
    raw_score = _score_predictions(predictions, *score_args)
    compare_tree(raw_score, report['compositions']['raw'], where + '/raw')
    del raw_score
    dc = {key: compose_dc_upper(value, curves['static_upper'], valid) for key, value in predictions.items()}
    invariants = {key: _composition_invariants(value, dc[key], curves['static_upper'], valid)
                  for key, value in predictions.items()}
    compare_tree(invariants, report['dc_invariants'], where + '/dc_invariants')
    dc_score = _score_predictions(dc, *score_args)
    compare_tree(dc_score, report['compositions']['dc'], where + '/dc')
    return {'raw_and_dc_score_predictions_recomputed': True, 'clips': 608,
            'modes': sorted(predictions), 'all_per_clip_scores_recomputed': True,
            'teacher_readout_recomputed': False, 'fullface_protection_claimed': False}


def upper_mean(curves, seed=42):
    valid = curves['valid']
    value = curves['predictions'][f'{seed}/full'][..., p.CC]
    return torch.where(valid[..., None], value, 0.).sum(1) / valid.sum(1)[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-name', default='adapter_transfer12_v2')
    args = parser.parse_args()
    require(Path(args.run_name).name == args.run_name, 'Run name must be one directory name')
    root, run = args.root, args.root / args.run_name
    code = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    status, process = read(run / 'status.json'), read(root / (args.run_name + '_process.json'))
    require(status['status'] == process['status'] == 'complete' and process['exit_code'] == 0
            and status['epochs_per_arm'] == 12 and status['smoke'] is False, 'Run incomplete')
    sourcepath = root / 'context12/chunk_teacher/final.pt'
    source, source_done = load_pt(sourcepath), read(sourcepath.with_name('complete.json'))
    source_record = read(sourcepath.with_name('provenance.json'))
    source_hash, source_recipe_hash = sha(sourcepath), p.canonical_hash(source_record['recipe'])
    require(source_hash == source_done['final_sha256']
            and source['recipe_sha256'] == source_done['recipe_sha256'] == source_record['recipe_sha256'] == source_recipe_hash
            and source['arm'] == 'chunk_teacher' and source['completed_epochs'] == 12
            and source['frozen'] == source_done['frozen'] == source_record['recipe']['frozen'], 'Source binding differs')
    initialpath = run / 'initial.pt'
    initial, initial_hash = load_pt(initialpath), sha(initialpath)
    require(set(initial) == {'upper', 'local', 'adapter'}, 'Initial snapshot keys differ')
    shared = {key: p.state_hash(value) for key, value in initial.items()}
    require(shared['upper'] == p.state_hash(source['upper']) and shared['local'] == source['frozen']['local']
            and shared['adapter'] == p.state_hash(TemporalLocalAdapter(64, 8, init_seed=TRAIN_SEED).state_dict()),
            'Initial upper/local/zero-adapter differs')
    selectionpath = run / 'sentence_split.json'
    historicalpath = root / 'repair12/scale_diagnosis/report.json'
    selection, historical = read(selectionpath), read(historicalpath)
    validate_split(selection, historical)
    fitpath = run / 'fit_selection.json'
    fit_selection = read(fitpath)
    fit_clips = [row['clip_id'] for row in fit_selection['clips']]
    require(len(fit_clips) == len(set(fit_clips)) == fit_selection['count'] == 128
            and fit_selection['selection_uses_motion'] is False
            and set(fit_clips) <= set(selection['split']['fit']['clips']), 'Fit diagnostic membership differs')
    for row in fit_selection['clips']:
        require(selection['split']['fit']['clips'][row['original_fit_index']] == row['clip_id'],
                'Fit diagnostic index is not in the adaptation subset')
    priorpath = root / 'audio_prefix12/local_temporal_transfer_audit/report.json'
    priorhash = sha(priorpath)
    require(priorhash == read(priorpath.with_name('complete.json'))['report_sha256'], 'Prerequisite audit binding differs')
    basepath = root / 'repair12/centered_prior/white/curves.pt'
    basehash, bases = sha(basepath), load_pt(basepath)
    oldpath = root / 'repair12/centered_prior/state_mean/state_white_curves.pt'
    oldhash = sha(oldpath)
    require(oldhash == read(oldpath.parent / 'complete.json')['state_white'], 'Old visual reference binding differs')
    old = load_pt(oldpath)
    metadata_equal(bases, old)
    old_mean = upper_mean(old)
    for seed in p.r.SEEDS:
        require(float((upper_mean(old, seed) - old_mean).abs().max()) <= 2e-6, 'Old static mean varies with seed')
    visualpath = root / 'context12/fixed_visual_selection.json'
    visual = read(visualpath)
    picks = visual['nine_plot_clips']
    require(len(picks) == 9 and len({row['clip_id'] for row in picks}) == 9
            and visual['noise_seed'] == 42 and visual['oracle_in_main_visual'] is False
            and visual['control_bindings']['state_white'] == oldhash, 'Historical visual selection binding differs')
    indices = torch.tensor([bases['clip_id'].index(row['clip_id']) for row in picks])
    require(all(int(index) == row['index'] for index, row in zip(indices, picks)), 'Fixed visual indices differ')
    visual_values = [old['predictions']['42/full'][indices].numpy().copy()]
    del old
    previouspath = root / 'audio_prefix12/adapt_local/dc_curves.pt'
    previoushash = sha(previouspath)
    require(previoushash == read(previouspath.with_name('complete.json'))['dc_curves_sha256'],
            'Previous adapted visual reference binding differs')
    previous = load_pt(previouspath)
    metadata_equal(bases, previous)
    require(not any('oracle' in key.lower() for key in previous['predictions']), 'Previous DC contains oracle')
    visual_values.append(previous['predictions']['42/full'][indices].numpy().copy())
    del previous
    step0 = load_pt(run / 'step0_transfer.pt')
    step0_audit = score_transfer(step0, read(run / 'step0_transfer.json'), selection, 'step0_transfer')
    step0_reference = {key: value for key, value in step0.items() if key != 'upper_predictions9'}
    del step0
    replay = read(run / 'step0_replay.json')
    require(replay['adapter_step0_bit_exact'] is True and replay['clips'] == 32 and replay['seed'] == 42
            and replay['max_abs'] <= 2e-6 and replay['source_curves_sha256'] == source_done['curves_sha256'],
            'Step0 replay record differs')
    arms, matched, bindings = {}, {}, {'old_white': oldhash, 'previous_adapt_local_dc': previoushash}
    common_static = None
    for arm in ARMS:
        print('AUDIT_ARM_START ' + arm, flush=True)
        folder = run / arm
        record, complete = read(folder / 'provenance.json'), read(folder / 'complete.json')
        recipe = record['recipe']
        digest = p.canonical_hash(recipe)
        require(digest == record['recipe_sha256'] == complete['recipe_sha256'] and complete['status'] == 'complete',
                'Recipe/complete binding differs: ' + arm)
        require(set(complete['files']) == COMPLETE_FILES, 'Complete file coverage differs')
        for name, expected in complete['files'].items():
            require(sha(folder / name) == expected, 'Artifact hash differs: ' + arm + '/' + name)
        for name, expected in recipe['code_sha256'].items():
            require(sha(code / name) == expected, 'Bound code changed: ' + name)
        require(recipe['source_sha256'] == source_hash and recipe['source_recipe_sha256'] == source_recipe_hash
                and recipe['initial'] == shared and recipe['initial_file_sha256'] == initial_hash
                and recipe['baseline_curves_sha256'] == basehash and recipe['sentence_split_sha256'] == sha(selectionpath)
                and recipe['historical_split_sha256'] == sha(historicalpath)
                and recipe['prior_diagnostic_sha256'] == priorhash and recipe['fit_selection_sha256'] == sha(fitpath),
                'Recipe input lineage differs: ' + arm)
        require(recipe['schema'] == SCHEMA and recipe['arm'] == arm and recipe['epochs'] == recipe['requested_epochs'] == 12
                and recipe['batch_size'] == 16 and recipe['seed'] == TRAIN_SEED and recipe['solver_steps'] == 12
                and recipe['fit_update_clips'] == 1707 and recipe['transfer_clips'] == 608
                and recipe['rank'] == 8 and recipe['adapter_parameters'] == 1024
                and recipe['smoke'] is False and recipe['test_loaded'] is False and recipe['fixed_final_epoch'] is True
                and recipe['upper_trainable'] is True and recipe['local_trainable'] is (arm == 'full_local')
                and recipe['adapter_trainable'] is (arm == 'rank8_adapter'), 'Fixed budget/policy differs: ' + arm)
        require(recipe['data_provenance']['fit_clips'] == 2315 and recipe['data_provenance']['development_clips'] == 405
                and recipe['data_provenance']['test_loaded'] is False, 'Source data scope differs')
        final = load_pt(folder / 'final.pt')
        require(final['schema'] == SCHEMA and final['arm'] == arm and final['recipe_sha256'] == digest
                and final['completed_epochs'] == complete['completed_epochs'] == 12
                and final['total_steps'] == complete['total_steps'] == 1284
                and final['frozen'] == complete['frozen'] == recipe['frozen'] == source['frozen'], 'Endpoint contract differs')
        require(torch.equal(final['scales'], source['scales'])
                and final['scales'].tolist() == recipe['scales'], 'Frozen scales differ')
        protected = lambda state: {key: value for key, value in state.items()
                                   if not key.startswith(('input.', 'blocks.', 'local_head.'))}
        require(p.state_hash(protected(final['local'])) == p.state_hash(protected(initial['local'])),
                'Protected local heads/normalization changed')
        changed = {key: p.state_hash(final[key]) != shared[key] for key in shared}
        require(changed == {'upper': True, 'local': arm == 'full_local', 'adapter': arm == 'rank8_adapter'},
                'Observed parameter update policy differs')
        require(set(final['adapter']) == {'down.weight', 'up.weight'}
                and final['adapter']['down.weight'].shape == (8, 64)
                and final['adapter']['up.weight'].shape == (64, 8), 'Adapter rank/parameter contract differs')
        check_gradients(arm, complete['gradient_audit'])
        rows = [read(folder / f'epoch{epoch:03d}.json') for epoch in range(1, 13)]
        for epoch, row in enumerate(rows, 1):
            require(row['arm'] == arm and row['epoch'] == epoch and row['total_steps'] == epoch * 107
                    and row['first_two_step_gradients'] == complete['gradient_audit']
                    and row['initial_features']['same_forward_bit_exact'] is True, 'Epoch contract differs')
        matched[arm] = {'initial': shared, 'draws': [row['draw_sha256'] for row in rows], 'updates': 1284}
        raw_compact, dc_compact = load_pt(folder / 'curves.pt'), load_pt(folder / 'dc_curves.pt')
        metadata_equal(raw_compact, dc_compact)
        require(_same_bits(raw_compact['static_upper'], dc_compact['static_upper']), 'Raw/DC static differs')
        if common_static is None:
            common_static = raw_compact['static_upper'].clone()
        require(_same_bits(common_static, raw_compact['static_upper'])
                and float((raw_compact['static_upper'] - old_mean).abs().max()) <= 2e-6,
                'Arms do not share the bound audio-only static mean')
        composition_results = {}
        for label, compact in (('raw', raw_compact), ('dc', dc_compact)):
            require(compact['adaptation_arm'] == arm and compact['recipe_sha256'] == digest, 'Curve recipe differs')
            full = reconstruct_compact(compact, bases, basepath, basehash, dc=label == 'dc')
            if label == 'raw':
                for key in {'42/empty', '42/reverse_history'} | ORACLE_KEYS:
                    require(_same_bits(full['predictions']['42/full'][:, :16], full['predictions'][key][:, :16]),
                            'History intervention changed history-free first chunk')
            else:
                for key, pred in full['predictions'].items():
                    baseline = bases['predictions'][key.split('/')[0] + '/base']
                    raw_upper = raw_compact['upper_predictions9'][key]
                    expected = compose_dc(baseline, raw_upper, compact['static_upper'], compact['valid'])
                    require(torch.allclose(expected, pred, atol=2e-6, rtol=1e-6), 'DC recomposition differs')
                    _composition_invariants(raw_upper, pred[..., p.CC], compact['static_upper'], compact['valid'])
                visual_values.append(full['predictions']['42/full'][indices].numpy().copy())
            report = read(folder / ('evaluation.json' if label == 'raw' else 'dc_evaluation.json'))
            deploy = {**full, 'predictions': {key: value for key, value in full['predictions'].items() if key in DEPLOY_KEYS}}
            recalculated = summarize(deploy, compact['emotion_id'])
            interventions = recalculated.pop('single_seed_interventions')
            compare_tree(recalculated, report['distribution'], arm + '/' + label + '/distribution')
            compare_tree(interventions, report['deployable_interventions_seed42'], arm + '/' + label + '/interventions')
            composition_results[label] = {'distribution_and_all_deployable_interventions_recomputed': True,
                                          'protected_43_and_invalid_exact_after_reconstruction': True,
                                          'compact_upper_roundtrip_exact': True}
            del full, deploy, recalculated, interventions
            gc.collect()
        del raw_compact, dc_compact, compact
        transfer = load_pt(folder / 'transfer_curves.pt')
        transfer_metadata_equal(step0_reference, transfer)
        transfer_audit = score_transfer(transfer, read(folder / 'transfer_evaluation.json'), selection, arm + '/transfer')
        del transfer
        fit = load_pt(folder / 'fit_curves.pt')
        require(fit['clip_id'] == fit_clips and fit['nonupper_placeholders'] is True, 'Fit archive membership differs')
        require(all(not bool(value[..., list(p.r.NOT_UPPER)].count_nonzero()) for value in fit['predictions'].values()),
                'Fit nonupper placeholders differ')
        for label in ('speaker_id', 'emotion_id'):
            require(fit[label].tolist() == [row[label] for row in fit_selection['clips']], 'Fit metadata differs')
        drift = read(folder / 'feature_drift.json')
        require([row['clip_id'] for row in drift['development']] == bases['clip_id']
                and [row['clip_id'] for row in drift['transfer']] == selection['split']['internal_sentence_holdout']['clips']
                and [row['clip_id'] for row in drift['fit']] == fit_clips, 'Feature drift membership differs')
        adapter_mean_error = max(row['same_forward_mean_correction_max_abs'] for group in drift.values() for row in group)
        if arm == 'rank8_adapter':
            require(adapter_mean_error <= 2e-6, 'Recorded adapter mean-preservation check differs')
        arms[arm] = {'file_recipe_code_and_source_hashes_verified': True, 'parameter_changes': changed,
                     'source_scales_and_frozen_record_hashes_verified': True, 'completed_epochs': 12, 'updates': 1284,
                     'development': composition_results, 'transfer': transfer_audit,
                     'fit_selection_and_zero_nonupper_verified': True, 'fit_metric_recomputation_performed': False,
                     'recorded_feature_mean_correction_max_abs': adapter_mean_error,
                     'teacher_emotion_readout_recomputed': False}
        bindings[arm + '/raw'] = complete['files']['curves.pt']
        bindings[arm + '/dc'] = complete['files']['dc_curves.pt']
        del final, fit, drift
        gc.collect()
        print('AUDIT_ARM_PASSED ' + arm, flush=True)
    recorded = read(run / 'matched_audit.json')
    require(recorded['equal'] is True and recorded['arms'] == matched
            and all(value == matched[ARMS[0]] for value in matched.values()), 'Matched initialization/RNG/budget differs')
    subsetpath = run / 'new_nine_clips.npz'
    np.savez_compressed(subsetpath, motions=np.stack(visual_values), target=bases['target'][indices].numpy(),
                        mode_names=np.asarray(MODES), clip_id=np.asarray([row['clip_id'] for row in picks]),
                        **{key: bases[key][indices].numpy() for key in ('times', 'valid', 'channel_mask')})
    manifest = {'output_sha256': sha(subsetpath), 'selection_sha256': sha(visualpath), 'source_curves': bindings,
                'baseline_sha256': basehash, 'noise_seed': 42, 'oracle_included': False, 'raw_clamped': False,
                'test_loaded': False, 'mode_names': list(MODES),
                'scope': 'Historical fixed nine 405-dev clips; all five modes are full52 deployable offline compositions.'}
    p.save_json(run / 'new_nine_clips_manifest.json', manifest)
    p.save_json(run / 'integrity_audit.json', {
        'status': 'passed', 'schema': 'temporal_adapter_endpoint_integrity_v1', 'run': args.run_name,
        'audit_source_sha256': sha(__file__), 'source_sha256': source_hash, 'initial_sha256': initial_hash,
        'sentence_split_sha256': sha(selectionpath), 'fixed_visual_selection_sha256': sha(visualpath),
        'arms': arms, 'step0_transfer': step0_audit,
        'step0_file_hashes_current_audit': {name: sha(run / name) for name in ('step0_transfer.pt', 'step0_transfer.json', 'step0_replay.json')},
        'matched_initialization_rng_and_update_records': True, 'exact_historical_1707_608_membership_verified': True,
        'full52_reconstructed_in_memory_only': True, 'frozen_system_audio_source_tensors_reloaded': False,
        'teacher_emotion_readout_recomputed': False, 'test_loaded': False, 'training_artifacts_modified': False,
        'limitations': [
            'The 608 sentences are held out only from new updates; shared inherited sources saw the full 2315 fit pool.',
            'Frozen system/audio hashes and gradient/feature-drift records are crosschecked; source tensors and feature forwards are not recomputed.',
            'Compact storage cannot independently recover discarded nonupper values; reconstruction is exact to the bound base and the pre-storage check is recorded by training.',
            'Development distributions/interventions and all608 raw/DC scores are recomputed; teacher emotion, fit numerical metrics, and raw oracle scores are not recomputed.',
            'Step0 archives have current audit hashes rather than an original completion-file binding.',
            'Integrity does not certify naturalness, identity, lip synchronization, or sealed-test generalization.',
        ],
    })
    print('TEMPORAL_ADAPTER_INTEGRITY_PASSED', flush=True)


if __name__ == '__main__':
    main()
