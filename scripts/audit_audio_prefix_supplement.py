"""Audit saved prefix contracts and boundary/fit metrics without model inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_prefix_upper as p
from scripts.audit_audio_prefix_results import compare_tree
from scripts.audit_temporal_repair import load_pt, metadata_equal, read, sha
from scripts.evaluate_context_mechanism import ORACLE_MODES, supplied_history_endpoints
from scripts.evaluate_prefix_formal import actual_prefix_continuation, chunk_diagnostics
from scripts.train_audio_prefix_adaptation import SCHEMA


ARMS = ('frozen_local', 'adapt_local')
MEAN_TOLERANCE = 2e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_upper_mean(curves, seed):
    valid = curves['valid']
    require(bool(valid.any(1).all()), 'Cannot compare means of empty clips')
    upper = curves['predictions'][f'{seed}/full'][..., p.CC]
    return torch.where(valid[..., None], upper, 0.).sum(1) / valid.sum(1)[:, None]


def audit_boundary_metrics(curves, report, *, raw):
    records = dict(report['modes'])
    if raw:
        records.update(report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes'])
    require(set(records) == set(curves['predictions']), 'Boundary report mode coverage differs')
    continuation_modes = []
    for key, prediction in curves['predictions'].items():
        record = records[key]
        prefix = ('raw/' if raw else 'dc/') + key
        compare_tree(p.h.boundary_report(prediction, curves['target'], curves['valid']),
                     record['boundaries'], prefix + '/boundaries')
        compare_tree(chunk_diagnostics(prediction, curves['target'], curves['valid'], curves['channel_mask']),
                     record['chunk_diagnostics'], prefix + '/chunk_diagnostics')
        # DC translates the whole output after rollout; it has no actual supplied-prefix score.
        if not raw:
            require('actual_prefix_continuation' not in record,
                    'DC must not report an actual supplied-prefix continuation')
        elif 'actual_prefix_continuation' in record:
            mode = key.split('/', 1)[1]
            scored = mode != 'empty'
            saved = record['actual_prefix_continuation']
            require(saved['scored'] is scored, 'Actual-prefix score policy differs')
            expected = None
            if scored:
                supplied = supplied_history_endpoints(prediction, curves['target'], curves['valid'], mode)
                expected = actual_prefix_continuation(prediction, curves['target'], curves['valid'],
                                                      curves['channel_mask'], supplied)
            compare_tree(expected, saved['metrics'], prefix + '/actual_prefix_continuation')
            diagnostic = actual_prefix_continuation(prediction, curves['target'], curves['valid'],
                                                   curves['channel_mask'], curves['target'])
            compare_tree(diagnostic, record['same_gt_endpoint_output_diagnostic']['metrics'],
                         prefix + '/same_gt_endpoint_output_diagnostic')
            continuation_modes.append(key)
    return {'boundary_and_chunk_profile_modes': sorted(records),
            'actual_prefix_and_common_GT_endpoint_modes': sorted(continuation_modes),
            'teacher_emotion_readout_recomputed': False}


def audit_fit_metrics(curves, report, selection):
    require(curves['clip_id'] == [row['clip_id'] for row in selection['clips']], 'Fit clip order differs')
    require(len(curves['clip_id']) == 128 and curves['nonupper_placeholders'] is True,
            'Fixed fit diagnostic contract differs')
    for label in ('speaker_id', 'emotion_id'):
        require(curves[label].tolist() == [row[label] for row in selection['clips']],
                'Fit selection labels differ: ' + label)
    require(report['role'] == 'fixed_fit_reconstruction_diagnostic' and report['clips'] == 128
            and report['seed'] == 42 and report['arm'] == 'chunk_teacher'
            and report['test_loaded'] is False and report['nonupper_scored'] is False,
            'Fit report scope differs')
    expected_modes = {'full', 'empty', 'reverse_history', 'static', 'reverse', *ORACLE_MODES}
    require(set(report['metrics']) == expected_modes
            and set(curves['predictions']) == {'42/' + mode for mode in expected_modes},
            'Fit metric mode coverage differs')
    for mode in sorted(expected_modes):
        prediction = curves['predictions']['42/' + mode]
        require(not bool(torch.count_nonzero(prediction[..., list(p.r.NOT_UPPER)])),
                'Fit nonupper placeholders are nonzero')
        expected = {}
        for region in ('brows', 'eyes_expression'):
            channels = list(p.GROUPS[region])
            observed = curves['valid'][..., None] & curves['channel_mask'][:, None, channels]
            expected[region] = p.paired_metrics(prediction[..., channels], curves['target'][..., channels], observed)
        expected['stitched_16frame_grid'] = p.h.boundary_report(prediction, curves['target'], curves['valid'])
        expected['grid_is_decoder_boundary'] = True
        expected['chunk_diagnostics'] = chunk_diagnostics(prediction, curves['target'], curves['valid'], curves['channel_mask'])
        expected['actual_supplied_prefix'] = None
        if mode != 'empty':
            supplied = supplied_history_endpoints(prediction, curves['target'], curves['valid'], mode)
            expected['actual_supplied_prefix'] = actual_prefix_continuation(
                prediction, curves['target'], curves['valid'], curves['channel_mask'], supplied)
        expected['GT_was_input'] = mode in ORACLE_MODES
        expected['same_GT_endpoint_DIAGNOSTIC'] = actual_prefix_continuation(
            prediction, curves['target'], curves['valid'], curves['channel_mask'], curves['target'])
        compare_tree(expected, report['metrics'][mode], 'fit/' + mode)
    adjacent = curves['valid'][:, 1:] & curves['valid'][:, :-1]
    seam = torch.arange(1, curves['valid'].shape[1]).remainder(p.CHUNK) == 0
    require(report['adjacent_prefix_boundary_pairs'] == int((adjacent & seam[None]).sum()),
            'Fit adjacent prefix pair count differs')
    return {'all_reported_fit_metric_modes_recomputed': sorted(expected_modes),
            'regions_scored': ['brows', 'eyes_expression'], 'nonupper_scored': False,
            'oracle_modes_remain_diagnostic_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, run = args.root, args.root / 'audio_prefix12'
    source_path = root / 'context12/chunk_teacher/final.pt'
    source, source_complete = load_pt(source_path), read(source_path.with_name('complete.json'))
    source_record = read(source_path.with_name('provenance.json'))
    source_digest = p.canonical_hash(source_record['recipe'])
    require(sha(source_path) == source_complete['final_sha256']
            and source['recipe_sha256'] == source_complete['recipe_sha256'] == source_record['recipe_sha256'] == source_digest,
            'Source checkpoint/recipe binding differs')
    require(source['arm'] == 'chunk_teacher' and source['completed_epochs'] == 12,
            'Unexpected source arm or epoch')
    require(source['frozen'] == source_complete['frozen'] == source_record['recipe']['frozen'],
            'Source frozen records differ')
    initial_path = run / 'initial.pt'
    initial = load_pt(initial_path)
    shared = {key: p.state_hash(value) for key, value in initial.items()}
    require(shared['upper'] == p.state_hash(source['upper']) and shared['local'] == source['frozen']['local'],
            'Initial snapshot does not match source upper/frozen local')
    selection_path = run / 'fit_selection.json'
    selection, selection_sha = read(selection_path), sha(selection_path)
    require(selection_sha == sha(root / 'context12/fit_selection.json'), 'Historical fit selection differs')
    require(selection['count'] == 128 and selection['selection_uses_motion'] is False,
            'Unexpected fit selection policy')
    old_path = root / 'repair12/centered_prior/state_mean/state_white_curves.pt'
    require(sha(old_path) == read(old_path.parent / 'complete.json')['state_white'], 'Old audio-mean curves hash differs')
    old = load_pt(old_path)
    old_means = {str(seed): valid_upper_mean(old, seed) for seed in p.r.SEEDS}
    matched, arms, first_static, first_fit = {}, {}, None, None
    for arm in ARMS:
        folder = run / arm
        record, complete = read(folder / 'provenance.json'), read(folder / 'complete.json')
        recipe = record['recipe']
        digest = p.canonical_hash(recipe)
        require(digest == record['recipe_sha256'] == complete['recipe_sha256'], 'Recipe hash differs: ' + arm)
        require(recipe['source_sha256'] == sha(source_path) and recipe['source_recipe_sha256'] == source_digest,
                'Source binding differs: ' + arm)
        require(recipe['initial'] == shared and recipe['initial_file_sha256'] == sha(initial_path)
                and recipe['fit_selection_sha256'] == selection_sha, 'Initial/fit binding differs: ' + arm)
        for stem, key in (('final', 'final_sha256'), ('curves', 'curves_sha256'),
                          ('dc_curves', 'dc_curves_sha256'), ('fit_curves', 'fit_curves_sha256')):
            require(sha(folder / (stem + '.pt')) == complete[key], 'Artifact hash differs: ' + arm + '/' + stem)
        final = load_pt(folder / 'final.pt')
        require(final['schema'] == recipe['schema'] == SCHEMA and final['arm'] == recipe['arm'] == arm
                and final['completed_epochs'] == complete['completed_epochs'] == recipe['epochs'] == 12
                and final['total_steps'] == complete['total_steps'] == 1740
                and final['recipe_sha256'] == digest, 'Endpoint contract differs: ' + arm)
        require(final['frozen'] == complete['frozen'] == recipe['frozen'] == source['frozen'],
                'Source/endpoint frozen records differ: ' + arm)
        require(torch.equal(final['scales'], source['scales'])
                and torch.equal(final['scales'], torch.tensor(recipe['scales'], dtype=final['scales'].dtype)),
                'Fixed normalization scales differ: ' + arm)
        require(recipe['test_loaded'] is False and recipe['local_trainable'] is (arm == 'adapt_local'),
                'Recipe deployment/training policy differs: ' + arm)
        raw, dc, fit = (load_pt(folder / name) for name in ('curves.pt', 'dc_curves.pt', 'fit_curves.pt'))
        metadata_equal(old, raw)
        metadata_equal(raw, dc)
        require(torch.equal(raw['static_upper'], dc['static_upper']), 'Raw/DC static upper differs: ' + arm)
        if first_static is None:
            first_static, first_fit = raw['static_upper'], fit
        else:
            require(torch.equal(first_static, raw['static_upper']), 'Two arms use different static upper')
            metadata_equal(first_fit, fit)
        mean_errors = {}
        for seed, mean in old_means.items():
            require(mean.shape == raw['static_upper'].shape, 'Old/new mean shapes differ')
            error = float((mean - raw['static_upper']).abs().max())
            require(error <= MEAN_TOLERANCE, 'Static upper is not bound old audio mean: ' + arm + '/' + seed)
            mean_errors[seed] = error
        rows = [read(folder / f'epoch{epoch:03d}.json') for epoch in range(1, 13)]
        require(all(row['arm'] == arm and row['epoch'] == epoch and row['total_steps'] == epoch * 145
                    for epoch, row in enumerate(rows, 1)), 'Epoch index/update records differ: ' + arm)
        matched[arm] = {'initial': shared, 'draws': [row['draw_sha256'] for row in rows], 'updates': 1740}
        raw_report, dc_report = read(folder / 'evaluation.json'), read(folder / 'dc_evaluation.json')
        fit_report = read(folder / 'fit_evaluation.json')
        arms[arm] = {
            'source_endpoint_frozen_scales_contracts_verified': True,
            'static_mean_max_abs_error_vs_old_by_seed': mean_errors,
            'fit_selection_sha256': selection_sha,
            'raw_boundary_recomputation': audit_boundary_metrics(raw, raw_report, raw=True),
            'dc_boundary_recomputation': audit_boundary_metrics(dc, dc_report, raw=False),
            'fit_recomputation': audit_fit_metrics(fit, fit_report, selection),
            'report_sha256': {name: sha(folder / name) for name in ('evaluation.json', 'dc_evaluation.json', 'fit_evaluation.json')},
        }
    recorded = read(run / 'matched_audit.json')
    require(recorded['equal'] is True and recorded['arms'] == matched and matched[ARMS[0]] == matched[ARMS[1]],
            'Matched manifest does not match actual initial/epoch/update records')
    output = {
        'status': 'passed', 'schema': 'audio_prefix_supplemental_integrity_v1',
        'source_sha256': sha(source_path), 'audit_source_sha256': sha(__file__),
        'old_audio_mean_curves_sha256': sha(old_path), 'arms': arms,
        'matched_manifest_crosschecked_against_epoch_records': True,
        'static_upper_exact_across_arms': True, 'static_mean_absolute_tolerance': MEAN_TOLERANCE,
        'teacher_emotion_readout_recomputed': False, 'system_audio_source_tensors_reloaded': False,
        'deployment_distributions_recomputed_in_this_supplement': False, 'model_inference_performed': False,
        'test_loaded': False, 'training_artifacts_modified': False,
        'limitations': [
            'Frozen system/audio hashes are crosschecked across records; their source tensors are not reloaded.',
            'The old deployment audio mean is bound by its existing file manifest, not regenerated from raw audio.',
            'No teacher emotion accuracy or full deployment distribution is recomputed in this supplement.',
            'Numerical consistency on reused internal development and fixed fit diagnostics does not establish perceptual quality or generalization.',
        ],
    }
    p.save_json(run / 'supplemental_integrity_audit.json', output)
    print('AUDIO_PREFIX_SUPPLEMENTAL_INTEGRITY_PASSED', flush=True)


if __name__ == '__main__':
    main()
