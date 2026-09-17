"""Read-only local audit of the fixed-final paired prefix experiment.

Audits the downloaded curves and records, not remote checkpoints or training
execution. Missing checkpoints are explicit; optional final files are hashed
without loading tensors. Frozen source tensors are never opened. No fitting,
inference, or test-set access.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import NOT_UPPER, SEEDS, compare, load_pt, read, sha, summarize, validate_curves
from scripts.evaluate_prefix_formal import _same_bits, actual_prefix_continuation
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_history_upper import CHUNK, HISTORY, CC, teacher_probability
from scripts.train_predictable_renderer import state_hash


ARMS = ('no_prefix', 'scheduled_prefix')
MODES = ('full', 'empty', 'reverse_history', 'static', 'reverse', 'oracle_history')
SCHEMA = 'prefix_formal_v1'
CURVE_SCHEMA = 'explicit_prefix_upper_v1'
DEVELOPMENT_CLIPS = 405
FIT_CLIPS = 2315
EPOCHS = 12
CODE_FILES = (
    'scripts/train_prefix_formal.py', 'scripts/train_prefix_upper.py',
    'scripts/evaluate_prefix_formal.py', 'scripts/audit_history_upper.py',
    'scripts/audit_temporal_repair.py', 'scripts/train_history_upper.py',
    'kinetalk_b0/models/prefix_upper_flow.py', 'kinetalk_b0/models/temporal_upper.py',
    'docs/PREFIX_FORMAL_PROTOCOL_20260917.md')
DISCARDED = {
    'history_input.0.weight', 'history_input.0.bias', 'history_input.2.weight', 'history_input.2.bias',
    'history_gru.weight_ih', 'history_gru.weight_hh', 'history_gru.bias_ih',
    'history_gru.bias_hh', 'history_projection.weight'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def tree_close(actual, expected, path='metric'):
    """Compare recomputed numeric records while rejecting extra/missing fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), path+' keys differ')
        for key in expected:
            tree_close(actual[key], expected[key], path+'/'+str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path+' list differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            tree_close(left, right, path+'/'+str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9), path+' numeric value differs')
    else:
        require(type(actual) is type(expected) and actual == expected, path+' value differs')


def same_metadata(reference, candidate):
    require(reference['clip_id'] == candidate['clip_id'], 'Cross-run clip order differs')
    for key in ('target', 'valid', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id'):
        require(key in candidate and _same_bits(reference[key], candidate[key]), 'Cross-run metadata differs: '+key)


def validate_epoch_records(epochs, recipe, arm):
    require(len(epochs) == EPOCHS, 'Expected all 12 epoch records')
    counter_names = ('teacher_draw_count', 'valid_history_count', 'teacher_eligible_count', 'teacher_used_count')
    for index, row in enumerate(epochs, 1):
        require(row['arm'] == arm and row['epoch'] == index
                and row['total_steps'] == index*math.ceil(FIT_CLIPS/16)
                and row['teacher_probability'] == recipe['teacher_probabilities'][index-1]
                and row['no_prefix_ignores_history'] is (arm == 'no_prefix')
                and math.isfinite(row['loss']) and row['loss'] >= 0
                and math.isfinite(row['seconds']) and row['seconds'] > 0
                and is_digest(row['draw_sha256']), 'Invalid epoch record: '+arm+'/'+str(index))
        counts = [row[key] for key in counter_names]
        require(all(type(value) is int and 0 <= value <= FIT_CLIPS*6 for value in counts), 'Invalid teacher counters')
        draw, history, eligible, used = counts
        require(eligible <= min(draw, history) and used == (eligible if arm == 'scheduled_prefix' else 0),
                'Teacher eligibility or usage differs')
        if row['teacher_probability'] == 0:
            require(draw == eligible == used == 0, 'Final generated-only epochs used teacher history')


def load_arm(folder, arm):
    provenance, complete, evaluation = (read(folder/name) for name in ('provenance.json', 'complete.json', 'evaluation.json'))
    recipe = provenance['recipe']; digest = canonical_hash(recipe)
    require(provenance['recipe_sha256'] == complete['recipe_sha256'] == digest, 'Recipe digest differs: '+arm)
    require(recipe['schema'] == SCHEMA and recipe['arm'] == arm and recipe['epochs'] == EPOCHS
            and recipe['requested_epochs'] == EPOCHS and recipe['batch_size'] == 16 and recipe['seed'] == 79
            and recipe['smoke'] is False and recipe['fixed_final_epoch'] is True
            and recipe['test_loaded'] is False and recipe['default_replaced'] is False
            and recipe['local_frozen'] is True and recipe['chunk'] == CHUNK and recipe['history'] == HISTORY
            and recipe['decode_steps'] == 12, 'Formal protocol differs: '+arm)
    require(recipe['teacher_probabilities'] == [teacher_probability(e, EPOCHS) for e in range(EPOCHS)]
            and not any(recipe['teacher_probabilities'][7:]), 'Teacher schedule differs')
    require(set(recipe['explicit_discarded_source_keys']) == DISCARDED, 'Discarded warmstart keys differ')
    require(set(recipe['frozen']) == {'system', 'audio', 'local'} and all(is_digest(v) for v in recipe['frozen'].values())
            and complete['frozen'] == recipe['frozen'], 'Frozen runtime records differ')
    require(complete['completed_epochs'] == EPOCHS and is_digest(complete['final_sha256']), 'Incomplete final epoch')
    for key in ('initial', 'initial_file_sha256', 'warmstart_sha256', 'warmstart_recipe_sha256', 'baseline_curves_sha256'):
        require(is_digest(recipe[key]), 'Invalid recorded digest: '+key)
    scales = torch.tensor(recipe['scales'])
    require(scales.shape == (9,) and torch.isfinite(scales).all() and (scales > 0).all(), 'Invalid fixed scales')
    data = recipe['data_provenance']
    require(data['fit_clips'] == FIT_CLIPS and data['development_clips'] == DEVELOPMENT_CLIPS
            and data['test_loaded'] is False and data['outer_development_loaded'] is False
            and data['default_replaced'] is False, 'Data roles differ')
    require(data['input_sha256'] and all(is_digest(v) for v in data['input_sha256'].values()), 'Invalid data input bindings')
    pilot = recipe['pilot_binding']
    require(pilot['gate']['passed'] is True and pilot['original_warmstart_sha256'] == recipe['warmstart_sha256']
            and len(pilot['selected_fit_ids']) == len(set(pilot['selected_fit_ids'])) == 8,
            'Pilot gate or original warmstart binding differs')
    require(set(recipe['code_sha256']) == set(CODE_FILES), 'Training code binding inventory differs')
    root = Path(__file__).resolve().parents[1]
    for name, value in recipe['code_sha256'].items():
        require(is_digest(value) and sha(root/name) == value, 'Local training source differs: '+name)

    curves_hash = sha(folder/'curves.pt')
    require(curves_hash == complete['curves_sha256'], 'Curve file hash differs: '+arm)
    curves = load_pt(folder/'curves.pt')
    require(curves['schema'] == CURVE_SCHEMA and curves['formal_schema'] == SCHEMA
            and curves['arm'] == arm and curves['recipe_sha256'] == digest
            and curves['decode_steps'] == 12 and curves['prefix_enabled'] is (arm == 'scheduled_prefix'),
            'Curve schema or binding differs')
    validate_curves(curves)
    require(curves['target'].shape == (DEVELOPMENT_CLIPS, 96, 52) and curves['channel_mask'][:, CC].all(),
            'Expected locked development count, clock and nine observed channels')
    expected = {'42/'+mode for mode in MODES} | {str(seed)+'/full' for seed in SEEDS}
    require(set(curves['predictions']) == expected and curves['oracle_prediction_keys'] == ['42/oracle_history'],
            'Unexpected curve modes or oracle classification')
    epochs = [read(folder/f'epoch{i:03d}.json') for i in range(1, EPOCHS+1)]
    validate_epoch_records(epochs, recipe, arm)
    initial_path = folder/'initial.pt'
    initial_verified = initial_path.is_file()
    if initial_verified:
        require(sha(initial_path) == recipe['initial_file_sha256'], 'Initial checkpoint file hash differs')
        initial = load_pt(initial_path)['upper']
        require(state_hash(initial) == recipe['initial'] and not initial['known_embedding.weight'].any(),
                'Initial tensor hash or new embedding differs')
    final_path = folder/'final.pt'
    final_verified = final_path.is_file()
    if final_verified:
        require(sha(final_path) == complete['final_sha256'], 'Final checkpoint file hash differs')
    bindings = {'curves_sha256': curves_hash, 'provenance_sha256': sha(folder/'provenance.json'),
                'evaluation_sha256': sha(folder/'evaluation.json'), 'complete_sha256': sha(folder/'complete.json'),
                'initial_tensor_and_file_locally_verified': initial_verified,
                'initial_recorded_state_sha256': recipe['initial'],
                'final_recorded_sha256': complete['final_sha256'],
                'final_checkpoint_file_hash_verified_at_audit_location': final_verified,
                'final_checkpoint_tensors_loaded': False,
                'frozen_source_tensors_locally_verified': False}
    return {'recipe': recipe, 'curves': curves, 'evaluation': evaluation, 'epochs': epochs, 'bindings': bindings}


def validate_output_contract(curves, evaluation, arm, baseline):
    prefix = arm == 'scheduled_prefix'
    expected = set(curves['predictions']); oracle_key = '42/oracle_history'
    require(evaluation['schema'] == SCHEMA and evaluation['arm'] == arm
            and evaluation['recipe_sha256'] == curves['recipe_sha256'] and evaluation['smoke'] is False
            and evaluation['clips'] == DEVELOPMENT_CLIPS and evaluation['noise_seeds'] == list(SEEDS)
            and evaluation['decode_steps'] == 12 and evaluation['evaluation_role'] == 'internal_development'
            and evaluation['prefix_enabled'] is prefix and evaluation['test_loaded'] is False
            and evaluation['default_replaced'] is False, 'Evaluation schema or role differs')
    for flag in ('nonupper_exact', 'invalid_baseline_exact', 'first_chunk_history_interventions_equal'):
        require(evaluation[flag] is True, 'Evaluation contract flag differs: '+flag)
    require(evaluation['no_prefix_history_interventions_equal'] is (None if prefix else True),
            'No-prefix history equality flag differs')
    oracle = evaluation['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']
    require(set(evaluation['modes']) == expected-{oracle_key} and set(oracle['modes']) == {oracle_key},
            'Oracle must remain separate from deployment modes')
    controls = {'42/'+mode for mode in ('empty', 'reverse_history', 'static', 'reverse')}
    require(set(evaluation['deployable_interventions_seed42']) == controls
            and 'single_seed_interventions' not in evaluation['distribution'], 'Deployment control modes differ')
    full = curves['predictions']['42/full']
    for key, pred in curves['predictions'].items():
        seed, mode = key.split('/')
        base = baseline['predictions'][seed+'/base']
        require(_same_bits(pred[..., list(NOT_UPPER)], base[..., list(NOT_UPPER)]), 'Nonupper protection failed: '+arm+'/'+key)
        require(_same_bits(pred[~curves['valid']], base[~curves['valid']]), 'Invalid baseline protection failed: '+arm+'/'+key)
        if mode in ('empty', 'reverse_history', 'oracle_history'):
            require(_same_bits(pred[:, :CHUNK], full[:, :CHUNK]), 'History affected first chunk: '+arm+'/'+key)
            if not prefix:
                require(_same_bits(pred, full), 'No-prefix output depends on history: '+key)
        record = oracle['modes'][key] if mode == 'oracle_history' else evaluation['modes'][key]
        require(record['oracle_target_history'] is (mode == 'oracle_history') and record['prefix_enabled'] is prefix,
                'Mode prefix/GT classification differs')
        accuracy = record['generated_emotion_accuracy_nonindependent']
        require(math.isfinite(accuracy) and 0 <= accuracy <= 1, 'Invalid nonindependent emotion readout')
        scored = prefix and mode in ('full', 'oracle_history')
        actual = record['actual_prefix_continuation']
        source = ('tracked_reference_GT' if mode == 'oracle_history' else 'generated_past') if scored else None
        require(actual['scored'] is scored and actual['source'] == source, 'Actual-prefix metric source differs')
        expected_actual = actual_prefix_continuation(pred, curves['target'], curves['valid'], curves['channel_mask'],
                    curves['target'] if mode == 'oracle_history' else pred) if scored else None
        tree_close(actual['metrics'], expected_actual, key+'/actual_prefix')
        diagnostic = record['same_gt_endpoint_output_diagnostic']
        require(diagnostic['GT_was_supplied_as_history'] is (prefix and mode == 'oracle_history'),
                'Output diagnostic mislabeled as GT input')
        tree_close(diagnostic['metrics'], actual_prefix_continuation(pred, curves['target'], curves['valid'],
                    curves['channel_mask'], curves['target']), key+'/same_gt_endpoint')


def audit(run, baseline):
    run, baseline = Path(run), Path(baseline)
    status, matched = read(run/'status.json'), read(run/'matched_audit.json')
    require(status['status'] == 'complete' and status['epochs_per_arm'] == EPOCHS and status['smoke'] is False,
            'Formal training is not complete')
    require(math.isfinite(status['seconds']) and status['seconds'] > 0, 'Invalid completed run duration')
    runs = {arm: load_arm(run/arm, arm) for arm in ARMS}
    paired = []
    for arm in ARMS:
        recipe = dict(runs[arm]['recipe'])
        recipe.pop('arm'); recipe.pop('initial_file_sha256')
        paired.append(recipe)
    require(paired[0] == paired[1] and matched['equal'] is True and set(matched['arms']) == set(ARMS),
            'Paired recipes differ')
    for arm, item in runs.items():
        require(matched['arms'][arm] == {'initial': item['recipe']['initial'],
                'draws': [row['draw_sha256'] for row in item['epochs']]}, 'Initial or draw records differ: '+arm)
    require(matched['arms'][ARMS[0]] == matched['arms'][ARMS[1]], 'Paired random streams differ')
    for left, right in zip(runs[ARMS[0]]['epochs'], runs[ARMS[1]]['epochs']):
        require(all(left[key] == right[key] for key in ('teacher_draw_count', 'valid_history_count', 'teacher_eligible_count')),
                'Paired teacher draw or eligibility counters differ')

    base_hash = sha(baseline)
    require(all(item['recipe']['baseline_curves_sha256'] == base_hash for item in runs.values()), 'Baseline hash differs')
    base = load_pt(baseline)
    baseline_complete = baseline.with_name('complete.json')
    if baseline_complete.is_file():
        require(read(baseline_complete)['curves_sha256'] == base_hash, 'Baseline completion hash differs')
    reference = runs[ARMS[0]]['curves']
    for candidate in (base, runs[ARMS[1]]['curves']):
        same_metadata(reference, candidate)
    for seed in SEEDS:
        value = base['predictions'].get(str(seed)+'/base')
        require(torch.is_tensor(value) and value.shape == reference['target'].shape and value.dtype == torch.float32,
                'Missing fixed-seed baseline')
    deployment, interventions, oracles = {}, {}, {}
    for arm, item in runs.items():
        validate_output_contract(item['curves'], item['evaluation'], arm, base)
        summary = summarize(item['curves'], item['curves']['emotion_id'])
        controls = summary.pop('single_seed_interventions')
        oracle = controls.pop('42/oracle_history')
        tree_close(item['evaluation']['distribution'], summary, arm+'/deployment_distribution')
        tree_close(item['evaluation']['deployable_interventions_seed42'], controls, arm+'/deployment_controls')
        tree_close(item['evaluation']['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['paired_metrics_vs_full'], oracle, arm+'/oracle_scores')
        deployment[arm], interventions[arm], oracles[arm] = summary, controls, oracle
    return {'schema': 'prefix_formal_local_audit_v1', 'status': 'passed',
            'scope': 'Completed records and downloaded internal-development curves; passing integrity checks is not a model-quality gate.',
            'clips': DEVELOPMENT_CLIPS, 'training_seconds': status['seconds'], 'epochs_per_arm': EPOCHS,
            'bindings': {arm: item['bindings'] for arm, item in runs.items()}, 'baseline_curves_sha256': base_hash,
            'checks': {'recipes_match_except_arm_and_initial_serialized_file': True,
                       'all_12_recorded_random_streams_match': True, 'recorded_initial_state_hashes_match': True,
                       'frozen_runtime_hash_records_match': True, 'training_source_file_hashes_verified_locally': True,
                       'exact_405_native_metadata_and_order': True, 'all_modes_43_nonupper_exact': True,
                       'all_modes_invalid_frames_exact': True, 'first_chunk_history_interventions_equal': True,
                       'no_prefix_history_interventions_equal': True, 'oracle_separated_from_deployment': True,
                       'reported_distributions_and_actual_prefix_metrics_recomputed': True},
            'deployment': deployment, 'prefix_minus_no_prefix': compare(deployment[ARMS[1]], deployment[ARMS[0]]),
            'deployable_interventions_seed42': interventions, 'ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY': oracles,
            'test_loaded': False, 'default_replaced': False,
            'limitations': ['Final checkpoint files are hashed only when present at the audit location; missing files require separate remote verification. Final tensors are never loaded.',
                           'Frozen source tensors were not opened; only recipe and completion hash records were matched.',
                           'Recorded random and initial hashes are not an independent replay of training; initial snapshots are verified only when present.',
                           'Repeated internal development is not sealed-test evaluation or evidence of unseen-identity generalization.',
                           'The no-prefix control removes previous acoustic tokens as well as motion tokens.',
                           'Forty-three copied channels preserve prior coefficients; they do not certify perceptual lip sync or identity.',
                           'Emotion classification uses a nonindependent motion teacher. Upper nine channels exclude blink and gaze.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'baseline', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh audit output required')
    torch.set_num_threads(4)
    report = audit(args.run, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print('PREFIX_FORMAL_LOCAL_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
