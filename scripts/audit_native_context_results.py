"""Read-only endpoint integrity/metric audit and historical nine-clip export.

Only saved train/internal-development artifacts are read. No fitting, model
inference, checkpoint selection, sealed-test access, or training-file mutation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_prefix_upper as p
from scripts.audit_audio_prefix_results import compare_tree
from scripts.audit_temporal_repair import read, sha, load_pt
from scripts.audit_temporal_adapter_results import validate_split
from scripts.evaluate_audio_prefix_adaptation import _composition_invariants
from scripts.evaluate_prefix_formal import _same_bits
from scripts.evaluate_native_context import SCHEMA as EVAL_SCHEMA, actual_decoder_seams
from scripts.evaluate_temporal_adapter_transfer import _score_predictions, compose_dc_upper
from scripts.native_context_runtime import paired_noise
from scripts.train_native_context import (ARMS, SCHEMA, CODE_FILES, subset_query,
    completed_files, verify_resume)

KEYS = {f'{seed}/full' for seed in p.r.SEEDS} | {'42/local_static', '42/local_reverse'}
MODES = ('center96/raw', 'full_native/raw', 'center96/dc', 'full_native/dc')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def equal_metadata(left, right):
    for key in ('clip_id', 'sentence_id'):
        require(left[key] == right[key], 'Common metadata differs: '+key)
    for key in ('target_upper9', 'valid', 'times', 'channel_mask_upper',
                'emotion_id', 'speaker_id', 'native_lengths', 'center_starts'):
        require(_same_bits(left[key], right[key]), 'Common tensor metadata differs: '+key)


def validate_curves(curves, report, query, entries, *, mode, expected_n):
    valid, target = curves['valid'], curves['target_upper9']
    require(curves['schema'] == report['schema'] == EVAL_SCHEMA
            and curves['mode'] == report['mode'] == mode
            and curves['clip_id'] == query['clip_id']
            and curves['sentence_id'] == query['sentence_id']
            and valid.dtype == torch.bool and valid.shape == (expected_n, 96)
            and bool(valid.any(1).all()), 'Archive schema/membership/common clock differs')
    for key in ('valid', 'times', 'emotion_id', 'speaker_id'):
        require(_same_bits(curves[key], query[key]), 'Source query metadata differs: '+key)
    expected = torch.where(valid[..., None], query['motion'][..., p.CC].float(), 0.)
    require(_same_bits(target, expected) and target.shape == (expected_n, 96, 9)
            and _same_bits(curves['channel_mask_upper'], query['channel_mask'][:, p.CC])
            and bool(curves['channel_mask_upper'].all()), 'Source target/channel mask differs')
    require(curves['times'].shape == valid.shape and bool(torch.isfinite(curves['times']).all())
            and torch.allclose(curves['times'][:, 1:]-curves['times'][:, :-1],
                torch.full_like(curves['times'][:, 1:], .04), atol=1e-7, rtol=1e-5), 'Native25Hz clock differs')
    lengths = torch.tensor([entries[cid]['native_frames'] for cid in curves['clip_id']])
    starts = torch.tensor([entries[cid]['center_start'] for cid in curves['clip_id']])
    require(_same_bits(curves['native_lengths'], lengths) and _same_bits(curves['center_starts'], starts),
            'Native lengths/crop offsets differ from bound extraction')
    frames = ((int(torch.maximum(lengths, starts+96).max())+15)//16)*16
    require(curves['population_noise_draw_frames'] == report['population_noise_draw_frames'] == frames,
            'Population noise clock differs')
    require(set(curves['upper_predictions9']) == KEYS and curves['upper_channel_indices'] == list(p.CC)
            and curves['noise_seeds'] == report['noise_seeds'] == list(p.r.SEEDS)
            and curves['decode_steps'] == report['decode_steps'] == 12
            and curves['GT_was_input'] is report['GT_was_input'] is False
            and curves['oracle_modes'] == report['oracle_modes'] == []
            and curves['dc_saved'] is False and report['test_loaded'] is False
            and report['teacher_readout_scored'] is report['nonupper_scored'] is False
            and curves['fullface_deployment_evaluation'] is report['fullface_deployment_evaluation'] is False,
            'Inference/report scope differs')
    require(report['clips'] == expected_n and report['sentence_count'] == len(set(curves['sentence_id']))
            and report['per_clip_order'] == curves['clip_id'] and report['primary_composition'] == 'raw'
            and report['secondary_composition'] == 'dc', 'Report population/composition scope differs')
    for key, value in curves['upper_predictions9'].items():
        require(value.dtype == torch.float32 and value.shape == target.shape
                and bool(torch.isfinite(value).all()) and not bool(value[~valid].count_nonzero()),
                'Corrupt upper payload, including invalid placeholders: '+key)
    require(curves['static_used'].dtype == torch.float32 and curves['static_used'].shape == (expected_n, 9)
            and bool(torch.isfinite(curves['static_used']).all())
            and _same_bits(curves['static_used'], curves['static_upper']), 'Static mean metadata differs')


def rescore(curves, report, where):
    predictions, valid, static = curves['upper_predictions9'], curves['valid'], curves['static_used']
    dc = {key: compose_dc_upper(value, static, valid) for key, value in predictions.items()}
    compare_tree({key: _composition_invariants(value, dc[key], static, valid)
                  for key, value in predictions.items()}, report['dc_invariants'], where+'/dc_invariants')
    summaries = {}
    for kind, values in (('raw', predictions), ('dc', dc)):
        score = _score_predictions(values, curves['target_upper9'], valid,
                                   curves['channel_mask_upper'], curves['emotion_id'])
        for key, value in values.items():
            score['modes'][key]['actual_decoder_seams'] = actual_decoder_seams(
                value, curves['target_upper9'], valid, curves['center_starts'], mode=curves['mode'])
        compare_tree(score, report['compositions'][kind], where+'/'+kind)
        summaries[kind] = {'distribution': score['distribution'],
            'deployable_interventions_seed42': score['deployable_interventions_seed42'],
            'temporal_seed42': score['modes']['42/full']['temporal'],
            'actual_decoder_seams_seed42': score['modes']['42/full']['actual_decoder_seams']}
    return summaries, dc


def restore_compact(curves, bases, baseline_path, baseline_hash, dc=None):
    """Restore all52 from the exact bound base; never discard corrupt payloads."""
    valid = curves['valid']
    require(curves.get('baseline_sha256') == baseline_hash
            and Path(curves.get('baseline_path', '')) == Path(baseline_path)
            and curves['nonupper_protection_checked'] is True
            and curves['clip_id'] == bases['clip_id'], 'Compact baseline binding differs')
    for key in ('valid', 'times', 'emotion_id', 'speaker_id'):
        require(_same_bits(curves[key], bases[key]), 'Baseline metadata differs: '+key)
    require(_same_bits(curves['target_upper9'], torch.where(valid[..., None], bases['target'][..., p.CC], 0.))
            and _same_bits(curves['channel_mask_upper'], bases['channel_mask'][:, p.CC]),
            'Baseline targets/masks differ')
    restored, checks = {}, {}
    for kind, values in (('raw', curves['upper_predictions9']), ('dc', dc or {})):
        for key, value in values.items():
            require(value.dtype == torch.float32 and value.shape == (*valid.shape, 9)
                    and bool(torch.isfinite(value).all()) and not bool(value[~valid].count_nonzero()),
                    'Corrupt upper payload, including invalid placeholders: '+key)
            base = bases['predictions'][key.split('/')[0]+'/base']
            require(base.dtype == torch.float32 and base.shape == (*valid.shape, 52), 'Invalid baseline tensor')
            output = base.clone()
            output[..., p.CC] = torch.where(valid[..., None], value, base[..., p.CC])
            require(_same_bits(output[..., list(p.r.NOT_UPPER)], base[..., list(p.r.NOT_UPPER)])
                    and _same_bits(output[~valid], base[~valid])
                    and _same_bits(output[..., p.CC][valid], value[valid]), 'Protected reconstruction differs')
            checks[key+'/'+kind] = {'nonupper43_bit_exact': True, 'invalid_baseline_bit_exact': True}
            if key == '42/full':
                restored[kind] = output
    return restored, checks


def replay_draws(lengths, starts, epochs=30):
    """Independent replay uses only clip lengths/offsets and dedicated CPU RNG."""
    generator = torch.Generator().manual_seed(97)
    result = []
    for _ in range(epochs):
        digest = hashlib.sha256()
        for ids in torch.randperm(len(lengths), generator=generator).split(16):
            _, times, draws = paired_noise(lengths[ids], starts[ids], generator, 'full')
            for value in (ids, draws['full_noise'], times):
                digest.update(value.numpy().tobytes())
        result.append(digest.hexdigest())
    return result


def command_path(command, option):
    require(command.count(option) == 1, 'Missing/repeated process argument: '+option)
    return Path(command[command.index(option)+1])


def check_hash(path, expected, where):
    require(path.is_file() and sha(path) == expected, 'File binding differs: '+where)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-name', default='native_context30')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(Path(args.run_name).name == args.run_name, 'Run name must be one directory name')
    root, run = args.root, args.root/args.run_name
    output = args.output or run/'result_audit'
    require(not output.exists(), 'Use a fresh audit output directory')
    torch.set_num_threads(4)
    process, status = read(run.with_name(run.name+'_process.json')), read(run/'status.json')
    require(process['status'] == status['status'] == 'complete' and process['exit_code'] == 0
            and status['smoke'] is False and status['epochs_per_arm'] == 30
            and status['test_loaded'] is status['default_replaced'] is False, 'Formal run incomplete')
    common_record = read(run/'provenance.json'); common = common_record['recipe']
    common_hash = p.canonical_hash(common)
    require(common_hash == common_record['recipe_sha256'] and common['schema'] == SCHEMA
            and common['epochs'] == common['requested_epochs'] == 30 and common['batch_size'] == 16
            and common['seed'] == 97 and common['decode_steps'] == 12
            and common['smoke'] is common['test_loaded'] is common['default_replaced'] is False,
            'Common recipe/budget differs')
    code_root = Path(__file__).resolve().parents[1]
    require(set(common['code_sha256']) == set(CODE_FILES), 'Training code binding inventory differs')
    for name, digest in common['code_sha256'].items(): check_hash(code_root/name, digest, name)
    inputs = common['data_provenance']; paths = {k:Path(v) for k,v in inputs['source_paths'].items()}
    command = process['command']; source_run = command_path(command, '--source-run')
    paths.update(source_provenance=source_run/'provenance.json', source_adapter=source_run/'final_epoch008.pt',
                 **{key:command_path(command, '--'+key) for key in ('audio','targets','enrollment')})
    require(set(paths) == set(inputs['input_sha256']) and inputs['test_loaded'] is False, 'Input scope differs')
    for key, path in paths.items(): check_hash(path, inputs['input_sha256'][key], key)
    for row in inputs['reference_bindings']: check_hash(Path(row['artifact']), row['artifact_sha256'], row['clip_id'])
    sourcepath = root/'context12/chunk_teacher/final.pt'
    source, source_done = load_pt(sourcepath), read(sourcepath.with_name('complete.json'))
    source_record = read(sourcepath.with_name('provenance.json'))
    require(sha(sourcepath) == source_done['final_sha256'] == common['source_sha256']
            and source['recipe_sha256'] == source_done['recipe_sha256'] == source_record['recipe_sha256']
                == common['source_recipe_sha256'] == p.canonical_hash(source_record['recipe'])
            and source['arm'] == 'chunk_teacher' and source['completed_epochs'] == 12
            and source['frozen'] == source_done['frozen'] == common['frozen'], 'Context source binding differs')
    historypath = root/'history12/no_history/final.pt'; history = load_pt(historypath)
    hdone = read(historypath.with_name('complete.json')); hrecord = read(historypath.with_name('provenance.json'))
    require(sha(historypath) == hdone['final_sha256'] and history['recipe_sha256'] == hrecord['recipe_sha256']
            == p.canonical_hash(hrecord['recipe']) == source_record['recipe']['source_recipe_sha256']
            and history['arm'] == 'no_history' and history['completed_epochs'] == 12, 'Local source lineage differs')
    scales = source['scales']
    require(_same_bits(scales, history['scales']) and scales.tolist() == common['scales'], 'Frozen residual scales differ')
    initial = {'upper':p.state_hash(source['upper']), 'local':p.state_hash(history['local'])}
    require(initial['local'] == common['frozen']['local'], 'Initial local weight hash differs')
    protected = lambda state: {k:v for k,v in state.items() if not k.startswith(('input.','blocks.','local_head.'))}
    protected_hash = p.state_hash(protected(history['local']))
    for role, key, field in (('teacher','system','system'), ('audio','audio','audio')):
        path = root/'run12'/role/'final.pt'; done = read(path.with_name('complete.json'))
        check_hash(path, done['final_sha256'], 'frozen '+key)
        payload = load_pt(path)
        require(p.state_hash(payload[field]) == common['frozen'][key], 'Original frozen tensors changed: '+key)
        del payload
    basepath = root/'repair12/centered_prior/white/curves.pt'; basehash = sha(basepath)
    require(basehash == common['baseline_sha256'] == hrecord['recipe']['baseline_curves_sha256'], 'Output base hash differs')
    bases = load_pt(basepath)
    selection = read(run/'sentence_split.json')
    validate_split(selection, read(root/'repair12/scale_diagnosis/report.json'))
    require(selection == common['split'], 'Common sentence partition differs')
    cache = load_pt(paths['cache'])
    require(set(cache['splits']) == {'train','validation'}, 'Only locked internal populations permitted')
    queries = {'fit':subset_query(cache['splits']['train']['q'], selection['fit_indices']),
               'hold':subset_query(cache['splits']['train']['q'], selection['holdout_indices']),
               'dev':cache['splits']['validation']['q']}
    require(common['active_members'] == {key:q['clip_id'] for key,q in queries.items()}, 'Active membership differs')
    delta_dir = command_path(command, '--delta-dir'); delta = read(delta_dir/'manifest.json')
    ddone = read(delta_dir/'complete.json')
    require(sha(delta_dir/'manifest.json') == common['delta_manifest_sha256'] == ddone['manifest_sha256']
            and delta['status'] == ddone['status'] == 'complete' and delta['config']['smoke'] is False
            and ddone['selected_count'] == ddone['completed_count'] == len(delta['clips']) == 2720
            and delta['selected_roles'] == {role:cache['splits'][role]['q']['clip_id'] for role in ('train','validation')},
            'Fullnative delta completion/membership differs')
    for cid, entry in delta['clips'].items():
        path = (delta_dir/entry['file']).resolve()
        require(path.is_relative_to(delta_dir.resolve()) and entry['status'] == 'complete', 'Delta artifact path/scope differs')
        check_hash(path, entry['sha256'], 'delta '+cid)
    native_manifest = command_path(command, '--native-manifest')
    check_hash(native_manifest, common['native_manifest_sha256'], 'native manifest')
    require(delta['source']['manifest_sha256'] == common['native_manifest_sha256']
            and delta['source']['audio_sha256'] == inputs['input_sha256']['audio'], 'Delta source differs')
    diagdir = command_path(command, '--initial-diagnostic'); diagdone = read(diagdir/'complete.json')
    require(diagdone['status'] == 'complete' and diagdone['smoke'] is False and diagdone['no_training'] is True
            and diagdone['files']['fit128.json'] == common['initial_diagnostic_sha256'], 'Initial diagnostic prerequisite differs')
    check_hash(diagdir/'fit128.json', common['initial_diagnostic_sha256'], 'initial diagnostic')
    coverage = read(run/'coverage.json')
    for name, q in queries.items():
        expected = {'clips':len(q['clip_id']), 'center_valid':int(q['valid'].sum()),
            'full_valid':int(q['valid'].sum())+sum(delta['clips'][cid]['extra_valid_frames'] for cid in q['clip_id']),
            'native_max_length':max(delta['clips'][cid]['native_frames'] for cid in q['clip_id'])}
        require(coverage[name] == expected, 'Coverage record differs: '+name)
    fit_lengths = torch.tensor([delta['clips'][cid]['native_frames'] for cid in queries['fit']['clip_id']])
    fit_starts = torch.tensor([delta['clips'][cid]['center_start'] for cid in queries['fit']['clip_id']])
    expected_draws = replay_draws(fit_lengths, fit_starts)
    print('SOURCES_AND_REPLAYED_DRAWS_PASSED', flush=True)
    arms, matched, common_curves, visuals, source_curves = {}, {}, {}, {}, {}
    for arm in ARMS:
        out = run/arm; record = read(out/'provenance.json'); recipe = record['recipe']; digest = p.canonical_hash(recipe)
        require(recipe == {'common_sha256':common_hash,'arm':arm,'initial':initial}
                and record['recipe_sha256'] == digest, 'Arm recipe/initial differs')
        complete = read(out/'complete.json')
        require(complete['status'] == 'complete' and complete['schema'] == SCHEMA
                and complete['completed_epochs'] == 30 and complete['total_steps'] == 3210
                and complete['recipe_sha256'] == digest and complete['frozen'] == common['frozen']
                and complete['test_loaded'] is False and set(complete['files']) == completed_files(30), 'Arm completion differs')
        for name, value in complete['files'].items(): check_hash(out/name, value, arm+'/'+name)
        last, final = load_pt(out/'last.pt'), load_pt(out/'final.pt')
        verify_resume(last,digest,arm,30,frozen=common['frozen'],scales=scales,
                      protected_local_sha256=protected_hash,expected_updates=107)
        require(last['completed_epochs'] == 30 and last['draws'] == expected_draws, 'Actual noise/order/time draw replay differs')
        for key in ('schema','arm','completed_epochs','total_steps','recipe_sha256','frozen'):
            require(final[key] == last[key], 'Final/last metadata differs: '+key)
        require(_same_bits(final['scales'], scales)
                and all(p.state_hash(final[k]) == p.state_hash(last[k]) for k in ('upper','local'))
                and p.state_hash(protected(final['local'])) == protected_hash, 'Final state/protected tensors differ')
        require(all(p.state_hash(final[k]) != initial[k] for k in ('upper','local')), 'Expected trained subsystem did not change')
        for epoch, row in enumerate(last['epochs'], 1):
            require(row == read(out/f'epoch{epoch:03d}.json')
                    and row['supervised_frames'] == coverage['fit']['center_valid' if arm=='center96' else 'full_valid']
                    and np.isfinite(row['loss']) and row['seconds'] > 0
                    and set(row['first_batch_local_gradients']) == {'input','blocks','local_head'}
                    and all(np.isfinite(v) and v>0 for v in row['first_batch_local_gradients'].values()), 'Epoch trace differs')
        matched[arm] = read(out/'matched.json')
        require(matched[arm] == {'initial':initial,'draws':expected_draws,'updates':3210}, 'Matched training record differs')
        mode = 'center' if arm == 'center96' else 'full'; populations = {}
        for name, curvefile, reportfile in (('step0','step0_holdout_upper9.pt','step0_holdout.json'),
                ('hold','hold_upper9.pt','hold_evaluation.json'),('dev','dev_upper9.pt','dev_evaluation.json')):
            curves, report = load_pt(out/curvefile), read(out/reportfile)
            role = 'hold' if name=='step0' else name; n = 405 if role=='dev' else 608
            validate_curves(curves, report, queries[role], delta['clips'], mode=mode, expected_n=n)
            if role in common_curves: equal_metadata(common_curves[role],curves)
            else: common_curves[role] = {k:curves[k] for k in ('clip_id','sentence_id','target_upper9','valid','times',
                'channel_mask_upper','emotion_id','speaker_id','native_lengths','center_starts')}
            if name != 'step0':
                require(curves['recipe_sha256'] == report['recipe_sha256'] == digest
                        and curves['arm'] == report['arm'] == arm and report['population'] == name, 'Endpoint recipe differs')
            scores, dc = rescore(curves, report, arm+'/'+name)
            populations[name] = scores
            if role == 'dev':
                restored, checks = restore_compact(curves,bases,basepath,basehash,dc)
                compare_tree(checks, report['protection_checks'], arm+'/protection')
                require(report['nonupper_protection_checked'] is True, 'Development protection declaration differs')
                for kind in ('raw','dc'): visuals[arm+'/'+kind] = restored[kind]
                source_curves[arm] = complete['files'][curvefile]
            else:
                require(curves['nonupper_protection_checked'] is report['nonupper_protection_checked'] is False
                        and report['protection_checks'] == {}, 'Holdout must not claim full52 protection')
            print('METRICS_RECOMPUTED '+arm+'/'+name, flush=True)
            del curves, report, dc
        arms[arm] = {'completed_epochs':30,'updates':3210,'file_recipe_code_source_hashes_verified':True,
            'protected_local_state_exact':True,'final_equals_last':True,'draws_independently_replayed':True,
            'all_saved_raw_dc_per_clip_distribution_and_seam_scores_recomputed':True,'scores':populations}
        del last, final
        gc.collect()
    require(read(run/'matched_audit.json') == {'equal':True,'arms':matched}, 'Final paired-arm audit differs')
    visualpath = root/'context12/fixed_visual_selection.json'; visual = read(visualpath); picks = visual['nine_plot_clips']
    require(len(picks) == 9 and len({row['clip_id'] for row in picks}) == 9
            and visual['noise_seed'] == 42 and visual['oracle_in_main_visual'] is False, 'Historical visual selection differs')
    indices = torch.tensor([bases['clip_id'].index(row['clip_id']) for row in picks])
    require(all(int(index) == row['index'] for index,row in zip(indices,picks)), 'Historical visual index changed')
    output.mkdir(parents=True)
    npz = output/'nine_clips.npz'
    np.savez_compressed(npz,predictions=np.stack([visuals[name][indices].numpy() for name in MODES]),
        target=bases['target'][indices].numpy(),modes=np.asarray(MODES),clip_id=np.asarray([row['clip_id'] for row in picks]),
        upper_indices=np.asarray(p.CC),noise_seed=np.asarray(42),
        **{key:bases[key][indices].numpy() for key in ('times','valid','channel_mask')})
    limitations = ['608 is held out only from these updates; inherited sources saw all2315 fitting clips.',
        'Only common historical center96 is scored; full-native prefixes/suffixes are not certified.',
        'Full context changes coverage/conditioning/static origin and compute; it is not a pure context-length ablation.',
        'Compact archives cannot recover discarded nonupper outputs; exact reconstruction uses the bound old base.',
        'Frozen source tensors and saved state/gradient declarations are checked; no fresh neural forward is run.',
        'Integrity and upper9 metrics do not certify identity, emotion, lipsync, naturalness or sealed-test generalization.']
    manifest = {'schema':'native_context_nine_review_v1','export_sha256':sha(npz),'source_curves':source_curves,
        'selection':visual,'selection_sha256':sha(visualpath),'baseline_sha256':basehash,'modes':list(MODES),
        'noise_seed':42,'oracle_included':False,'raw_clamped':False,'test_loaded':False,'limitations':limitations}
    p.save_json(output/'manifest.json',manifest)
    report = {'schema':'native_context_results_integrity_v1','status':'passed','audit_source_sha256':sha(__file__),
        'common_recipe_sha256':common_hash,'process':process,'coverage':coverage,'arms':arms,
        'source_system_audio_tensors_reloaded_and_hash_checked':True,'train_draws_replayed':True,
        'all2720_delta_file_hashes_checked':True,'common_608_405_targets_masks_timestamps_checked':True,
        'development43_invalid_exact_reconstruction_checked':True,'test_loaded':False,
        'training_artifacts_modified':False,'limitations':limitations}
    p.save_json(output/'report.json',report)
    p.save_json(output/'complete.json',{'status':'passed','files':{name:sha(output/name)
        for name in ('report.json','manifest.json','nine_clips.npz')}})
    print('NATIVE_CONTEXT_RESULTS_AUDIT_PASSED '+str(output),flush=True)


if __name__ == '__main__': main()
