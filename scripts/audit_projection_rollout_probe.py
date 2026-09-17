"""Two-arm fixed-epoch audit: closed-loop motion supervision vs flow matching.

Both arms use the same constant teacher probability .5, internal19/3 people,
frozen backbone/head and512-parameter projection. Epoch18 is primary and
epoch2 auxiliary. No best/arm selection or external development reads occur.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_audio_activity_gate import SEEDS, write_json
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_predictable_motion_predictions import paired_summary, scalar_summary
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_teacher_schedule_probe import (
    EPOCHS, GROUPS, KINDS, MODES, audit_populations, load_arm,
    statistics_for_curves, validate_curves, validate_training_inputs,
)
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import basic_metrics, sha, state_hash

ARMS = ('rollout', 'flow_constant_teacher')
ROLLOUT_SCHEMA = 'projection_rollout_probe_v1'
ROLLOUT_ARM = 'constant_teacher_rollout'
ROLLOUT_LOSS = 'observed_12step_rollout_motion_mse_divided_by_residual_scale_squared_only'


def validate_matched_rng(rollout_records, flow_records):
    for epoch in range(1, 19):
        left, right = rollout_records[epoch], flow_records[epoch]
        for key in ('minibatch_sha256', 'noise_time_sha256', 'teacher_choice_draw_sha256'):
            value = left.get(key)
            if not isinstance(value, str) or len(value) != 64 or value != right.get(key):
                raise ValueError(f'Rollout/flow RNG mismatch at epoch{epoch}: {key}')
        for key in ('step', 'samples_seen', 'teacher_fraction'):
            if left.get(key) != right.get(key):
                raise ValueError(f'Rollout/flow schedule or coverage mismatch: epoch{epoch}/{key}')


def validate_recipe_pair(rollout, flow):
    if rollout.get('schema') != ROLLOUT_SCHEMA or rollout.get('loss') != ROLLOUT_LOSS or rollout['args'].get('arm') != ROLLOUT_ARM:
        raise ValueError('Unexpected rollout objective or arm')
    if flow.get('schema') != 'projection_schedule_ablation_v1' or flow.get('loss') != 'observed_flow_mse_only' or flow['args'].get('arm') != 'constant_teacher':
        raise ValueError('Control must be the actual constant-teacher flow arm')
    if rollout.get('teacher_probability') != .5 or rollout.get('decode_steps') != 12 or rollout.get('flow_time_draws') != 'consumed and hashed identically to constant_teacher, unused in rollout loss':
        raise ValueError('Rollout teacher/noise/decode contract differs')
    extra_sources = set(rollout['source_sha256']) - set(flow['source_sha256'])
    if len(extra_sources) != 1 or not next(iter(extra_sources)).replace('\\', '/').endswith('/scripts/train_projection_rollout_probe.py'):
        raise ValueError('Only the rollout entry source may be added')
    if set(flow['source_sha256']) - set(rollout['source_sha256']) or any(rollout['source_sha256'][path] != digest for path, digest in flow['source_sha256'].items()):
        raise ValueError('Shared model/data/optimizer source changed between loss arms')
    left, right = copy.deepcopy(rollout), copy.deepcopy(flow)
    for recipe in (left, right):
        for key in ('schema', 'loss', 'source_sha256'):
            recipe.pop(key)
        for key in ('arm', 'output'):
            recipe['args'].pop(key)
    for key in ('teacher_probability', 'decode_steps', 'flow_time_draws'):
        left.pop(key)
    if left != right:
        raise ValueError('Loss arms differ beyond explicitly authorized objective/entry source')


def load_rollout(run, *, schema=ROLLOUT_SCHEMA, arm=ROLLOUT_ARM, final_epoch=18):
    summary_path, provenance_path = run / 'summary.json', run / 'provenance.json'
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    recipe = provenance['recipe']; digest = canonical_hash(recipe)
    if summary.get('schema') != schema or recipe.get('schema') != schema or summary.get('arm') != arm or recipe['args'].get('arm') != arm:
        raise ValueError('Wrong rollout schema/arm')
    if provenance.get('recipe_sha256') != digest or summary.get('recipe_sha256') != digest:
        raise ValueError('Rollout recipe hash mismatch')
    if summary.get('completed_epochs') != final_epoch or recipe['args'].get('epochs') != final_epoch or recipe['args'].get('seed') != 46:
        raise ValueError('Require declared fixed-epoch seed46 rollout')
    for value in (recipe, summary):
        for flag in ('outer280_loaded', 'new_identity439_loaded', 'test_loaded'):
            if value.get(flag) is not False:
                raise ValueError('Rollout read forbidden development/test')
    if summary.get('checkpoint_selection_performed') is not False or not summary.get('frozen_unchanged') or not summary.get('head_unchanged') or summary.get('default_replaced'):
        raise ValueError('Rollout selection/frozen/default contract failed')
    records = {epoch: json.loads((run / f'epoch{epoch:03d}.json').read_text(encoding='utf8')) for epoch in range(1, final_epoch + 1)}
    curves, checkpoints, reports, hashes = {}, {}, {}, {'summary': sha(summary_path), 'provenance': sha(provenance_path)}
    for epoch, ck_name, curve_name, binding_key in (
        (2, 'epoch002_auxiliary.pt', 'epoch002_diagnostic_curves.pt', 'epoch002_auxiliary_curve_provenance'),
        (final_epoch, f'final_epoch{final_epoch:03d}.pt', f'final_epoch{final_epoch:03d}_curves.pt', 'curve_provenance')):
        ck_path, curve_path = run / ck_name, run / curve_name
        checkpoint = torch.load(ck_path, map_location='cpu', weights_only=False)
        if checkpoint.get('schema') != schema or checkpoint.get('recipe') != recipe or checkpoint.get('recipe_sha256') != digest or checkpoint.get('completed_epochs') != epoch or checkpoint.get('selection') != 'none':
            raise ValueError('Rollout fixed-epoch checkpoint contract differs')
        if state_hash(checkpoint['head']) != checkpoint['head_sha256'] or checkpoint['local_projection']['weight'].shape != (64, 8):
            raise ValueError('Rollout head or512parameter projection differs')
        for key in ('minibatch_sha256', 'noise_time_sha256', 'teacher_choice_draw_sha256'):
            if checkpoint[key] != records[epoch][key] or (epoch == final_epoch and checkpoint[key] != summary[key]):
                raise ValueError('Rollout checkpoint/epoch/summary RNG mismatch')
        record = summary[binding_key]
        sidecar_path = curve_path.with_name(curve_path.stem + '.provenance.json')
        if Path(record['path']).resolve() != sidecar_path.resolve() or record['sha256'] != sha(sidecar_path):
            raise ValueError('Rollout curve sidecar record mismatch')
        sidecar = json.loads(sidecar_path.read_text(encoding='utf8'))
        expected = {'schema': 'projection_schedule_curves_provenance_v1', 'curve_sha256': sha(curve_path),
                    'checkpoint_sha256': sha(ck_path), 'recipe_sha256': digest, 'cache_sha256': recipe['input_sha256']['cache']}
        if any(sidecar.get(key) != value for key, value in expected.items()):
            raise ValueError('Rollout curves not bound to fixed checkpoint/cache')
        curves[epoch] = torch.load(curve_path, map_location='cpu', weights_only=False)
        checkpoints[epoch] = checkpoint
        reports[epoch] = summary['final'] if epoch == final_epoch else json.loads((run / 'development_epoch002.json').read_text(encoding='utf8'))['noise_reports']
        hashes[str(epoch)] = {'checkpoint': sha(ck_path), 'curves': sha(curve_path), 'sidecar': sha(sidecar_path)}
    return {'recipe': recipe, 'summary': summary, 'records': records, 'checkpoints': checkpoints,
            'curves': curves, 'reports': reports, 'hashes': hashes}


def summarize_teacher(reports):
    means, per_noise = {}, {}
    for arm in ARMS:
        per_noise[arm], means[arm] = {}, {}
        for mode in MODES:
            values = [float(reports[arm][str(seed)][mode]['frozen_teacher_emotion_accuracy']) for seed in SEEDS]
            if not all(np.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError('Invalid frozen teacher accuracy')
            per_noise[arm][mode] = values
            means[arm][mode] = float(np.mean(values))
    return {'mean_accuracy': means, 'per_noise_accuracy': per_noise,
        'full_minus_zero': {arm: means[arm]['full'] - means[arm]['zero'] for arm in ARMS},
        'rollout_full_minus_flow_full': means['rollout']['full'] - means['flow_constant_teacher']['full'],
        'note': 'Same frozen training motion teacher, not an independent emotion/perceptual evaluator; no added model-selection threshold.'}


def audit_two_arm_epoch(curves, reference, *, samples=5000, arms=ARMS):
    if set(curves) != set(arms):
        raise ValueError('Exactly the two explicitly declared arms are required')
    for seed in SEEDS:
        if not torch.equal(curves[arms[0]]['motion'][str(seed)]['zero'], curves[arms[1]]['motion'][str(seed)]['zero']):
            raise ValueError('Frozen zero-local generation differs between loss arms')
    q = reference['q']; populations = audit_populations(q)
    if 'neutral' not in populations or 'nonneutral' not in populations:
        raise ValueError('Require neutral and nonneutral evaluation')
    stats, velocity, raw_per_noise = {}, {}, {}
    for arm in arms:
        stats[arm], velocity[arm], raw_per_noise[arm] = statistics_for_curves(curves[arm], reference)
    scores = {arm: {mode: {kind: {group: {pop: scalar_summary(stats[arm][(mode, kind, group)][ids])
        for pop, ids in populations.items()} for group in GROUPS} for kind in KINDS} for mode in MODES} for arm in arms}
    full_vs = {arm: {base: {pop: {group: paired_summary(stats[arm][('full', 'centered_residual', group)],
        stats[arm][(base, 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for pop, ids in populations.items()} for base in ('zero', 'reverse', 'oracle')} for arm in arms}
    oracle_vs_zero = {arm: {pop: {group: paired_summary(stats[arm][('oracle', 'centered_residual', group)],
        stats[arm][('zero', 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for pop, ids in populations.items()} for arm in arms}
    between = {mode: {pop: {kind: {group: paired_summary(stats[arms[0]][(mode, kind, group)],
        stats[arms[1]][(mode, kind, group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for kind in KINDS} for pop, ids in populations.items()} for mode in ('full', 'oracle')}
    neutral = {arm: {group: relative_error_audit(stats[arm][('full', 'raw_motion', group)][:, [0, 3]],
        stats[arm][('zero', 'raw_motion', group)][:, [0, 3]], q['sentence_id'], populations['neutral'], samples=samples)
        for group in GROUPS} for arm in arms}
    velocity_relative = {arm: {pop: {group: relative_error_audit(velocity[arm][('full', group)],
        velocity[arm][('zero', group)], q['sentence_id'], ids, samples=samples) for group in GROUPS}
        for pop, ids in populations.items()} for arm in arms}
    corr = {arm: {pop: scores[arm]['full']['raw_motion']['mouth'][pop]['pooled_centered_correlation'] -
        scores[arm]['zero']['raw_motion']['mouth'][pop]['pooled_centered_correlation'] for pop in populations} for arm in arms}
    checks = {}
    for arm in arms:
        upper = full_vs[arm]['zero']['nonneutral']['upper_expression']
        brows = full_vs[arm]['zero']['nonneutral']['brows']
        checks[arm] = {
            'upper_positive_vs_zero': upper['r2_improvement'] > 0,
            'upper_ci95_lower_positive': upper['r2_improvement_ci95'] is not None and upper['r2_improvement_ci95'][0] > 0,
            'upper_positive_vs_reverse': full_vs[arm]['reverse']['nonneutral']['upper_expression']['r2_improvement'] > 0,
            'brows_positive_and_ci_lower_at_least_minus005': brows['r2_improvement'] > 0 and brows['r2_improvement_ci95'] is not None and brows['r2_improvement_ci95'][0] >= -.005,
            'neutral_mouth_one_sided90_within_3pct': neutral[arm]['mouth']['one_sided_90_upper'] is not None and neutral[arm]['mouth']['one_sided_90_upper'] <= .03,
            'neutral_upper_one_sided90_within_3pct': neutral[arm]['upper_expression']['one_sided_90_upper'] is not None and neutral[arm]['upper_expression']['one_sided_90_upper'] <= .03,
            'all_mouth_corr_within_01': corr[arm]['all'] >= -.01,
            'velocity_mean_within_5pct': all(velocity_relative[arm][pop][group]['relative_mse_increase'] is not None and
                velocity_relative[arm][pop][group]['relative_mse_increase'] <= .05 for pop, group in
                (('neutral', 'mouth'), ('neutral', 'upper_expression'), ('all', 'mouth'), ('nonneutral', 'upper_expression'))),
        }
    per_noise = {arm: {seed: {mode: {kind: {group: {pop: scalar_summary(row[ids]) for pop, ids in populations.items()
        if pop in ('all', 'neutral', 'nonneutral')} for group, row in rows.items()} for kind, rows in kinds.items()}
        for mode, kinds in modes.items()} for seed, modes in seeds.items()} for arm, seeds in raw_per_noise.items()}
    return {'scores': scores, 'full_vs_same_arm': full_vs, 'oracle_vs_same_arm_zero': oracle_vs_zero,
        arms[0] + '_minus_' + arms[1]: between, 'neutral_raw_relative_vs_zero': neutral,
        'velocity_relative_vs_zero': velocity_relative, 'mouth_correlation_full_minus_zero': corr,
        'same_arm_checks': checks, 'same_arm_checks_all_pass': {arm: all(values.values()) for arm, values in checks.items()},
        'per_noise': per_noise, 'population_counts': {pop: {'clips': len(ids),
            'sentences': len({q['sentence_id'][i] for i in ids}), 'identities': len({int(q['speaker_id'][i]) for i in ids})}
            for pop, ids in populations.items()},
        'acceptance_note': 'Improvement over the comparison control alone is insufficient. Full must also improve over its frozen zero/reversed condition with brow and neutral/mouth protection. No checkpoint or arm is selected.',
        'mouth_correlation_note': 'Correlation pools mean sufficient covariance/energies over noises; it is a point diagnostic without correlation CI. SSE/R2 and relative-MSE comparisons use paired sentence bootstrap.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--constant-teacher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        raise ValueError('Require fresh output and positive bootstrap count')
    torch.set_num_threads(4)
    rollout = load_rollout(args.rollout)
    flow = load_arm(args.constant_teacher, 'constant_teacher')
    validate_recipe_pair(rollout['recipe'], flow['recipe'])
    validate_matched_rng(rollout['records'], flow['records'])
    # Reuse source-lineage validation on the real single flow control. This
    # does not fabricate another schedule arm or duplicate outcomes.
    reference, lock, scope, frozen = validate_training_inputs({'constant_teacher': flow})
    for epoch in EPOCHS:
        checkpoint = rollout['checkpoints'][epoch]
        if checkpoint['head_sha256'] != frozen['fixed_head'] or checkpoint['frozen_state_sha256'] != frozen['frozen_backbone']:
            raise ValueError('Rollout head/backbone differs from refitted training-only control')
        validate_curves(rollout['curves'][epoch], reference)
        for seed in SEEDS:
            for mode in MODES:
                assert_metric_agreement(basic_metrics(rollout['curves'][epoch]['motion'][str(seed)][mode], reference),
                                        rollout['reports'][epoch][str(seed)][mode])
    arms = {'rollout': rollout, 'flow_constant_teacher': flow}
    epochs = {}
    for epoch in EPOCHS:
        epochs[str(epoch)] = audit_two_arm_epoch({arm: value['curves'][epoch] for arm, value in arms.items()}, reference, samples=args.samples)
        epochs[str(epoch)]['frozen_teacher_emotion_readout'] = summarize_teacher({arm: value['reports'][epoch] for arm, value in arms.items()})
    report = {'schema': 'projection_rollout_vs_flow_fixed_epoch_audit_v1', 'script_sha256': sha(__file__),
        'arm_hashes': {arm: value['hashes'] for arm, value in arms.items()}, 'frozen_hashes': frozen,
        'split_lock': lock, 'data_scope': scope, 'primary_epoch': 18, 'auxiliary_epoch': 2, 'epochs': epochs,
        'same_rng_all18epochs': True, 'same_data_parameters_teacher_schedule': True,
        'single_factor_change': 'Flow velocity MSE replaced by12-step final observed motion MSE/residual_scale²; constant teacher=.5 retained',
        'bootstrap_samples': args.samples, 'bootstrap_seed': 45, 'noise_seeds': SEEDS,
        'training_losses': {arm: {str(epoch): {key: value for key, value in data['records'][epoch].items()
            if key in ('mean_rollout_loss', 'mean_flow_loss', 'rollout_mse_by_condition_source', 'flow_mse_by_condition_source', 'teacher_fraction')}
            for epoch in range(1, 19)} for arm, data in arms.items()},
        'loss_comparison_note': 'Flow and rollout training losses measure different objectives and are not numerically comparable.',
        'scope': 'Internal19fit/3heldout dynamic-adapter identity diagnosis only; existing B0/global and previous formal runs may have seen them. No novel-sentence or whole-system unseen-generalization claim.',
        'checkpoint_or_arm_selected': False, 'outer280_loaded': False, 'new_identity439_loaded': False,
        'test_loaded': False, 'default_replaced': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    main_epoch = epochs['18']
    print(json.dumps({'epoch': 18, 'rollout_same_arm_checks': main_epoch['same_arm_checks']['rollout'],
        'rollout_minus_flow': {group: {'delta_r2': row['r2_improvement'], 'ci95': row['r2_improvement_ci95']}
            for group, row in main_epoch['rollout_minus_flow_constant_teacher']['full']['nonneutral']['centered_residual'].items()}}), flush=True)


if __name__ == '__main__':
    main()
