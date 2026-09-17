"""Fixed-epoch three-arm teacher-schedule diagnostic on train-internal people.

Epoch18 is the declared main comparison; epoch2 is diagnostic only. No best
checkpoint or arm is selected. Reports preserve negative emotion-region and
neutral/mouth outcomes. Frozen pretrained modules may already have seen the
internal held-out people, so this tests dynamic-adapter transfer only.
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
from scripts.audit_audio_activity_gate import SEEDS, velocity_clip_statistics, write_json
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_projection_delivery import mean_noise_statistics
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_projection_schedule_ablation import SCHEMA, read_allowlist, validate_internal_split
from scripts.prepare_teacher_schedule_probe import choose_identity_split, read_rows, tensor_group_hashes
from scripts.train_predictable_renderer import PredictableAudioHead, basic_metrics, center, sha, state_hash

ARMS = ('constant_teacher', 'audio_only', 'decay_teacher')
EPOCHS = (2, 18)
MODES = ('full', 'zero', 'reverse', 'oracle')
GROUPS = {'upper_expression': [5, 6, 12, 13, 41, 42, 43, 44, 45], 'brows': list(range(41, 46)),
          'eyes_expression': [5, 6, 12, 13], 'mouth': list(range(14, 41)), 'jaw17': [17]}
KINDS = ('raw_motion', 'centered_residual')


def audit_populations(q):
    labels, speakers = q['emotion_id'].tolist(), q['speaker_id'].tolist()
    result = {'all': list(range(len(labels))), 'neutral': [i for i, e in enumerate(labels) if e == 0],
              'nonneutral': [i for i, e in enumerate(labels) if e != 0]}
    for speaker in sorted(set(speakers)):
        for population in ('neutral', 'nonneutral'):
            result[f'speaker_{speaker}/{population}'] = [i for i, (s, e) in enumerate(zip(speakers, labels))
                if s == speaker and ((e == 0) if population == 'neutral' else (e != 0))]
    for emotion in sorted(set(labels)):
        result[f'emotion_{emotion}'] = [i for i, e in enumerate(labels) if e == emotion]
    return {key: ids for key, ids in result.items() if ids}


def validate_curves(curves, reference):
    if curves.get('noise_seeds') != list(SEEDS) or curves.get('decode_steps') != 12:
        raise ValueError('Require fixed three noises and twelve decoding steps')
    if set(curves.get('motion', {})) != set(map(str, SEEDS)):
        raise ValueError('Curve noise keys differ')
    for modes in curves['motion'].values():
        if set(modes) != set(MODES):
            raise ValueError('Require full/zero/reverse/oracle curves')
        if any(prediction.shape != reference['q']['motion'].shape for prediction in modes.values()):
            raise ValueError('Curve/reference shapes differ')


def statistics_for_curves(curves, reference):
    validate_curves(curves, reference)
    q = reference['q']; target = q['motion'].float(); weight = q['valid'].float()
    common = q['channel_mask'].all(0)
    if not torch.equal(q['channel_mask'], common[None].expand_as(q['channel_mask'])):
        raise ValueError('Require common observed channels')
    if any(c >= len(common) or not common[c] for values in GROUPS.values() for c in values):
        raise ValueError('Required expression channels missing')
    baseline = reference['base']['b0'].float() + reference['identity']['baseline'].float()[:, None]
    targets = {'raw_motion': target, 'centered_residual': center(target - baseline, weight)}
    fixed_targets = {(kind, group): clip_statistics(torch.zeros_like(target), yy, weight, channels)
                     for kind, yy in targets.items() for group, channels in GROUPS.items()}
    collected, velocity_rows, per_noise = {}, {}, {}
    for seed in SEEDS:
        per_noise[str(seed)] = {}
        for mode, pred in curves['motion'][str(seed)].items():
            fields = {'raw_motion': pred, 'centered_residual': center(pred - baseline, weight)}
            per_noise[str(seed)][mode] = {}
            for kind, value in fields.items():
                per_noise[str(seed)][mode][kind] = {}
                for group, channels in GROUPS.items():
                    row = clip_statistics(value, targets[kind], weight, channels)
                    collected.setdefault((mode, kind, group), []).append(row)
                    per_noise[str(seed)][mode][kind][group] = row
            for group, channels in GROUPS.items():
                velocity_rows.setdefault((mode, group), []).append(velocity_clip_statistics(pred, reference, channels))
    stats = {key: mean_noise_statistics(rows, fixed_targets[(key[1], key[2])]) for key, rows in collected.items()}
    velocities = {}
    for key, rows in velocity_rows.items():
        if any(not np.array_equal(row[:, 1], rows[0][:, 1]) for row in rows):
            raise ValueError('Velocity observation counts differ between noises')
        row = np.stack(rows).mean(0); row[:, 1] = rows[0][:, 1]
        velocities[key] = row
    return stats, velocities, per_noise


def audit_epoch(arm_curves, reference, *, samples=5000):
    """Compare all three fixed checkpoints without ranking/selecting an arm."""
    if set(arm_curves) != set(ARMS):
        raise ValueError('Require exactly the declared three teacher schedules')
    # All arms retain identical frozen zero-local behavior/noise. A difference
    # indicates model/input pairing drift, invalidating the causal comparison.
    for seed in SEEDS:
        baseline = arm_curves[ARMS[0]]['motion'][str(seed)]['zero']
        if any(not torch.equal(values['motion'][str(seed)]['zero'], baseline) for values in arm_curves.values()):
            raise ValueError('Zero-local curves differ across schedule arms')
    q = reference['q']; populations = audit_populations(q)
    if 'neutral' not in populations or 'nonneutral' not in populations:
        raise ValueError('Schedule audit needs both neutral and nonneutral observations')
    stats, velocity, per_noise = {}, {}, {}
    for arm, curves in arm_curves.items():
        stats[arm], velocity[arm], per_noise[arm] = statistics_for_curves(curves, reference)
    scores = {arm: {mode: {kind: {group: {pop: scalar_summary(stats[arm][(mode, kind, group)][ids])
        for pop, ids in populations.items()} for group in GROUPS} for kind in KINDS} for mode in MODES} for arm in ARMS}
    own = {arm: {base: {pop: {group: paired_summary(stats[arm][('full', 'centered_residual', group)],
        stats[arm][(base, 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for pop, ids in populations.items()} for base in ('zero', 'reverse', 'oracle')} for arm in ARMS}
    oracle_gain = {arm: {pop: {group: paired_summary(stats[arm][('oracle', 'centered_residual', group)],
        stats[arm][('zero', 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for pop, ids in populations.items()} for arm in ARMS}
    contrasts = [('constant_teacher', 'decay_teacher'), ('audio_only', 'decay_teacher'), ('constant_teacher', 'audio_only')]
    between = {left + '__vs__' + right: {mode: {pop: {group: paired_summary(stats[left][(mode, 'centered_residual', group)],
        stats[right][(mode, 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for pop, ids in populations.items()} for mode in ('full', 'oracle')}
        for left, right in contrasts}
    neutral = {arm: {group: relative_error_audit(stats[arm][('full', 'raw_motion', group)][:, [0, 3]],
        stats[arm][('zero', 'raw_motion', group)][:, [0, 3]], q['sentence_id'], populations['neutral'], samples=samples)
        for group in GROUPS} for arm in ARMS}
    velocity_relative = {arm: {pop: {group: relative_error_audit(velocity[arm][('full', group)], velocity[arm][('zero', group)],
        q['sentence_id'], ids, samples=samples) for group in GROUPS} for pop, ids in populations.items()} for arm in ARMS}
    corr = {arm: {pop: scores[arm]['full']['raw_motion']['mouth'][pop]['pooled_centered_correlation'] -
        scores[arm]['zero']['raw_motion']['mouth'][pop]['pooled_centered_correlation'] for pop in populations} for arm in ARMS}
    protection = {arm: {
        'neutral_mouth_one_sided90_within_3pct': neutral[arm]['mouth']['one_sided_90_upper'] is not None and neutral[arm]['mouth']['one_sided_90_upper'] <= .03,
        'neutral_upper_one_sided90_within_3pct': neutral[arm]['upper_expression']['one_sided_90_upper'] is not None and neutral[arm]['upper_expression']['one_sided_90_upper'] <= .03,
        'all_mouth_corr_within_01': corr[arm]['all'] >= -.01,
        'velocity_mean_within_5pct': all(velocity_relative[arm][pop][group]['relative_mse_increase'] is not None and
            velocity_relative[arm][pop][group]['relative_mse_increase'] <= .05 for pop, group in
            (('neutral', 'mouth'), ('neutral', 'upper_expression'), ('all', 'mouth'), ('nonneutral', 'upper_expression')))} for arm in ARMS}
    per_noise_summary = {arm: {seed: {mode: {kind: {group: {pop: scalar_summary(row[ids]) for pop, ids in populations.items()
        if pop in ('all', 'neutral', 'nonneutral')} for group, row in rows.items()} for kind, rows in kinds.items()}
        for mode, kinds in modes.items()} for seed, modes in seeds.items()} for arm, seeds in per_noise.items()}
    return {'scores': scores, 'full_vs_same_arm': own, 'oracle_vs_same_arm_zero': oracle_gain,
        'schedule_contrasts': between, 'neutral_raw_relative_vs_zero': neutral,
        'velocity_relative_vs_zero': velocity_relative, 'mouth_correlation_full_minus_zero': corr,
        'same_arm_protection_checks': protection,
        'per_noise': per_noise_summary, 'population_counts': {pop: {'clips': len(ids),
            'sentences': len({q['sentence_id'][i] for i in ids}), 'identities': len({int(q['speaker_id'][i]) for i in ids})}
            for pop, ids in populations.items()}}


def same_experiment_recipe(recipe):
    common = copy.deepcopy(recipe)
    for key in ('arm', 'output'):
        common['args'].pop(key, None)
    return common


def teacher_readout_summary(arm_reports):
    """Saved frozen-teacher class readout only; never a selection criterion."""
    means, per_noise = {}, {}
    for arm in ARMS:
        means[arm], per_noise[arm] = {}, {}
        for mode in MODES:
            values = [float(arm_reports[arm][str(seed)][mode]['frozen_teacher_emotion_accuracy']) for seed in SEEDS]
            if not all(np.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError('Invalid frozen teacher class accuracy')
            means[arm][mode] = float(np.mean(values))
            per_noise[arm][mode] = {str(seed): value for seed, value in zip(SEEDS, values)}
    return {'mean_accuracy': means, 'per_noise_accuracy': per_noise,
        'full_minus_zero': {arm: means[arm]['full'] - means[arm]['zero'] for arm in ARMS},
        'full_minus_decay_teacher_full': {arm: means[arm]['full'] - means['decay_teacher']['full'] for arm in ARMS},
        'same_mode_minus_decay_teacher': {arm: {mode: means[arm][mode] - means['decay_teacher'][mode] for mode in MODES} for arm in ARMS},
        'aggregation': 'Arithmetic mean of accuracy over three noises on exactly the same clips; noise repetitions are not independent samples',
        'scope': 'Saved readout of the frozen training motion teacher. Not independent global-emotion/perceptual evidence, no accuracy confidence interval, and no added selection or acceptance criterion.'}


def validate_rng_evidence(records):
    """Every arm must consume identical random draws, despite different use."""
    if set(records) != set(ARMS):
        raise ValueError('Missing teacher-schedule RNG evidence')
    keys = ('minibatch_sha256', 'noise_time_sha256', 'teacher_choice_draw_sha256')
    for epoch in range(1, 19):
        for key in keys:
            values = [records[arm][epoch].get(key) for arm in ARMS]
            if any(not isinstance(value, str) or len(value) != 64 for value in values) or len(set(values)) != 1:
                raise ValueError(f'Mismatched random-draw evidence at epoch{epoch}: {key}')
        if len({records[arm][epoch]['step'] for arm in ARMS}) != 1 or len({records[arm][epoch]['samples_seen'] for arm in ARMS}) != 1:
            raise ValueError('Unequal training steps or epoch sample counts')


def load_arm(run, expected_arm):
    summary_path, provenance_path = run / 'summary.json', run / 'provenance.json'
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    recipe = provenance['recipe']; digest = canonical_hash(recipe)
    if summary.get('schema') != SCHEMA or recipe.get('schema') != SCHEMA or summary.get('arm') != expected_arm or recipe['args'].get('arm') != expected_arm:
        raise ValueError('Wrong teacher-schedule schema/arm')
    if provenance.get('recipe_sha256') != digest or summary.get('recipe_sha256') != digest:
        raise ValueError('Schedule recipe hash mismatch')
    if summary.get('completed_epochs') != 18 or recipe['args'].get('epochs') != 18 or recipe['args'].get('seed') != 46:
        raise ValueError('Require fixed18epoch seed46 experiment')
    for value in (recipe, summary):
        for flag in ('outer280_loaded', 'new_identity439_loaded', 'test_loaded'):
            if value.get(flag) is not False:
                raise ValueError('Teacher-schedule experiment read forbidden development/test')
    if summary.get('checkpoint_selection_performed') is not False or not summary.get('frozen_unchanged') or not summary.get('head_unchanged') or summary.get('default_replaced'):
        raise ValueError('Selection/frozen/default contract failed')
    if recipe.get('trainable') != ['local_projection.weight'] or not recipe.get('head_basis_scale_frozen') or recipe.get('loss') != 'observed_flow_mse_only':
        raise ValueError('Unexpected trainable modules or objectives')
    records = {epoch: json.loads((run / f'epoch{epoch:03d}.json').read_text(encoding='utf8')) for epoch in range(1, 19)}
    curves, checkpoints, reports, hashes = {}, {}, {}, {'summary': sha(summary_path), 'provenance': sha(provenance_path)}
    for epoch, ck_name, curve_name, binding_key in (
        (2, 'epoch002_auxiliary.pt', 'epoch002_diagnostic_curves.pt', 'epoch002_auxiliary_curve_provenance'),
        (18, 'final_epoch018.pt', 'final_epoch018_curves.pt', 'curve_provenance')):
        ck_path, curve_path = run / ck_name, run / curve_name
        checkpoint = torch.load(ck_path, map_location='cpu', weights_only=False)
        if checkpoint.get('schema') != SCHEMA or checkpoint.get('recipe') != recipe or checkpoint.get('recipe_sha256') != digest or checkpoint.get('completed_epochs') != epoch or checkpoint.get('selection') != 'none':
            raise ValueError('Fixed epoch checkpoint contract differs')
        if state_hash(checkpoint['head']) != checkpoint['head_sha256'] or checkpoint['local_projection']['weight'].shape != (64, 8):
            raise ValueError('Fixed head or512-parameter projection differs')
        for key in ('minibatch_sha256', 'noise_time_sha256', 'teacher_choice_draw_sha256'):
            if checkpoint[key] != records[epoch][key] or (epoch == 18 and checkpoint[key] != summary[key]):
                raise ValueError('Checkpoint/epoch/summary RNG evidence mismatch')
        record = summary[binding_key]
        sidecar_path = curve_path.with_name(curve_path.stem + '.provenance.json')
        if Path(record['path']).resolve() != sidecar_path.resolve() or record['sha256'] != sha(sidecar_path):
            raise ValueError('Curve sidecar record mismatch')
        sidecar = json.loads(sidecar_path.read_text(encoding='utf8'))
        expected = {'schema': 'projection_schedule_curves_provenance_v1', 'curve_sha256': sha(curve_path),
                    'checkpoint_sha256': sha(ck_path), 'recipe_sha256': digest, 'cache_sha256': recipe['input_sha256']['cache']}
        if any(sidecar.get(key) != value for key, value in expected.items()):
            raise ValueError('Schedule curves not bound to fixed checkpoint/cache')
        curves[epoch] = torch.load(curve_path, map_location='cpu', weights_only=False)
        checkpoints[epoch] = checkpoint
        reports[epoch] = summary['final'] if epoch == 18 else json.loads((run / 'development_epoch002.json').read_text(encoding='utf8'))['noise_reports']
        hashes[str(epoch)] = {'checkpoint': sha(ck_path), 'curves': sha(curve_path), 'sidecar': sha(sidecar_path)}
    return {'recipe': recipe, 'summary': summary, 'records': records, 'checkpoints': checkpoints,
            'curves': curves, 'reports': reports, 'hashes': hashes}


def validate_training_inputs(arms):
    recipe = arms[ARMS[0]]['recipe']
    if any(same_experiment_recipe(arm['recipe']) != same_experiment_recipe(recipe) for arm in arms.values()):
        raise ValueError('Three-arm recipes differ beyond schedule and output')
    paths = {key: Path(value) for key, value in recipe['args'].items() if key in recipe['input_sha256']}
    for key, path in paths.items():
        if sha(path) != recipe['input_sha256'][key]:
            raise ValueError(f'Changed schedule input {key}')
    cache = torch.load(paths['cache'], map_location='cpu', weights_only=False, mmap=True)
    bundle = torch.load(paths['bundle'], map_location='cpu', weights_only=False, mmap=True)
    weights = torch.load(paths['weights'], map_location='cpu', weights_only=False)
    lock = json.loads(paths['split_lock'].read_text(encoding='utf8'))
    fit_ids, val_ids = read_allowlist(paths['fit_ids']), read_allowlist(paths['validation_ids'])
    scope = validate_internal_split(cache, bundle, lock, fit_ids, val_ids)
    source_paths = lock.get('source_paths', {})
    if set(source_paths) != {'bundle', 'cache', 'selection', 'train', 'enrollment'}:
        raise ValueError('Internal split needs explicit source paths for metadata-lineage verification')
    for key in ('selection', 'train', 'enrollment'):
        if sha(source_paths[key]) != lock['source_sha256'][key]:
            raise ValueError('Formal-training source metadata hash changed')
    formal_selection = json.loads(Path(source_paths['selection']).read_text(encoding='utf8'))
    rows, references = read_rows(source_paths['train']), read_rows(source_paths['enrollment'])
    expected_fit, expected_hold, expected_split = choose_identity_split(rows, references, formal_selection,
        seed=lock['identity_selection']['seed'])
    if expected_split != lock['identity_selection'] or [rows[int(i)]['clip_id'] for i in expected_fit] != fit_ids or [rows[int(i)]['clip_id'] for i in expected_hold] != val_ids:
        raise ValueError('Internal identity split differs from metadata-only formal-training partition')
    for key in ('train', 'enrollment'):
        if formal_selection['manifest_sha256'][key] != lock['source_sha256'][key]:
            raise ValueError('Formal lock and source manifest hash differ')
    if bundle['provenance']['input_sha256'] != lock['source_sha256']:
        raise ValueError('Prepared tensor source lineage differs from split lock')
    if bundle['provenance']['source_roles_accessed'] != ['source_bundle.bundles.internal', 'source_cache.splits.train']:
        raise ValueError('Preparation read nontraining source roles')
    for flag in ('original_formal_validation_target_values_accessed', 'new_identity439_loaded', 'test_manifests_loaded', 'test_targets_loaded'):
        if bundle['provenance'].get(flag) is not False:
            raise ValueError('Prepared probe accessed forbidden targets')
    for role in ('train', 'validation'):
        if tensor_group_hashes(cache['splits'][role]) != bundle['provenance']['copied_tensor_group_sha256'][role]:
            raise ValueError('Cached frozen tensor slice differs from audited preparation')
    if len(scope['fit_speakers']) != 19 or len(scope['heldout_speakers']) != 3:
        raise ValueError('Require exactly19fit/3heldout identities')
    if len(fit_ids) + len(val_ids) != 2720:
        raise ValueError('Internal probe must cover the fixed2720formal training queries')
    if cache['provenance']['bundle_sha256'] != recipe['input_sha256']['bundle'] or weights['provenance']['bundle_sha256'] != recipe['input_sha256']['bundle']:
        raise ValueError('Internal cache/basis provenance mismatch')
    state = weights['states']['rrr_rank8']
    if state['rank'] != 8 or not torch.equal(state['train_ids'], torch.arange(len(fit_ids))) or not torch.equal(bundle['train_ids'], state['train_ids']):
        raise ValueError('Rank8 must be fit only on all remapped19identity training rows')
    train = bundle['bundles']['internal']
    inner_selection = weights['selection']
    if inner_selection.get('outer_heldout_used') is not False or len(inner_selection.get('folds', [])) != 3:
        raise ValueError('Require three inner sentence folds without heldout identity selection')
    seen_validation = []
    for fold in inner_selection['folds']:
        fit, val = set(fold['fit_indices']), set(fold['validation_indices'])
        if fit & val or fit | val != set(range(len(fit_ids))):
            raise ValueError('Inner fold indices escape fit-only data or overlap')
        if set(fold['fit_sentences']) & set(fold['validation_sentences']):
            raise ValueError('Inner alpha selection sentence overlap')
        if set(fold['fit_sentences']) != {str(train['sentence_id'][i]) for i in fit} or set(fold['validation_sentences']) != {str(train['sentence_id'][i]) for i in val}:
            raise ValueError('Inner fold sentence metadata differs from fit-only rows')
        seen_validation.extend(fold['validation_indices'])
    if sorted(seen_validation) != list(range(len(fit_ids))) or state['alpha'] != inner_selection['selected_per_method_rank']['rrr_rank8']['alpha']:
        raise ValueError('Inner fold coverage or selected alpha differs')
    expected_head = PredictableAudioHead(state, train['motion_bins'], train['weight'])
    expected_head_hash = state_hash(expected_head.state_dict())
    original = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    frozen_hash = state_hash({key: value for key, value in original['model'].items() if not key.startswith('local_projection.')})
    for arm in arms.values():
        for epoch, checkpoint in arm['checkpoints'].items():
            if checkpoint['head_sha256'] != expected_head_hash or checkpoint['frozen_state_sha256'] != frozen_hash:
                raise ValueError('Checkpoint frozen backbone or fit-only head differs')
        for epoch in EPOCHS:
            validate_curves(arm['curves'][epoch], cache['splits']['validation'])
            for seed in SEEDS:
                for mode in MODES:
                    actual = basic_metrics(arm['curves'][epoch]['motion'][str(seed)][mode], cache['splits']['validation'])
                    assert_metric_agreement(actual, arm['reports'][epoch][str(seed)][mode])
    return cache['splits']['validation'], lock, scope, {'frozen_backbone': frozen_hash, 'fixed_head': expected_head_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument('--' + arm.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        raise ValueError('Require fresh report and positive bootstrap count')
    torch.set_num_threads(4)
    arms = {arm: load_arm(getattr(args, arm), arm) for arm in ARMS}
    validate_rng_evidence({arm: value['records'] for arm, value in arms.items()})
    reference, lock, scope, frozen = validate_training_inputs(arms)
    results = {str(epoch): audit_epoch({arm: value['curves'][epoch] for arm, value in arms.items()}, reference,
                                     samples=args.samples) for epoch in EPOCHS}
    for epoch in EPOCHS:
        results[str(epoch)]['frozen_teacher_emotion_readout'] = teacher_readout_summary({arm: value['reports'][epoch] for arm, value in arms.items()})
        results[str(epoch)]['mouth_correlation_note'] = (
            'Pooled within-clip covariance and energies are first averaged as sufficient statistics across the three noises, '
            'then correlation is computed. Predictions and per-noise correlations are not averaged. '
            'Reported mouth-correlation differences are point diagnostics without confidence intervals; '
            'paired sentence bootstrap uncertainty applies to native SSE/R2 and relative-MSE comparisons, not correlation.')
    report = {'schema': 'teacher_schedule_fixed_epoch_audit_v1', 'source_sha256': sha(__file__),
        'arm_hashes': {arm: value['hashes'] for arm, value in arms.items()}, 'frozen_hashes': frozen,
        'split_lock': lock, 'data_scope': scope, 'same_rng_all_18_epochs': True, 'primary_epoch': 18,
        'auxiliary_epoch': 2, 'checkpoint_or_arm_selected': False, 'bootstrap_samples': args.samples,
        'bootstrap_seed': 45, 'noise_seeds': SEEDS, 'epochs': results,
        'training_condition_fractions': {arm: {str(epoch): {'teacher_fraction': value['records'][epoch]['teacher_fraction'],
            'flow_mse_by_condition_source': value['records'][epoch].get('flow_mse_by_condition_source')}
            for epoch in range(1, 19)} for arm, value in arms.items()},
        'scope': 'Internal19fit/3heldout dynamic-adapter identities, derived only from formal2720training. Existing B0/global may have seen all people. Shared sentences permitted; not novel-sentence or whole-system unseen identity evidence.',
        'interpretation': 'An oracle gain alone shows teacher-condition use, not audio transfer. Audio-only full must improve over same-arm zero/reverse and the paired schedule baseline, with neutral/mouth protection. Epoch2 is diagnostic and cannot replace the fixed epoch18 main result.',
        'outer280_loaded': False, 'new_identity439_loaded': False, 'test_loaded': False, 'default_replaced': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for name, value in results['18']['schedule_contrasts'].items():
        print(json.dumps({'comparison': name, 'epoch': 18, 'nonneutral_full': {
            group: {'delta_r2': row['r2_improvement'], 'ci95': row['r2_improvement_ci95']}
            for group, row in value['full']['nonneutral'].items()}}), flush=True)


if __name__ == '__main__':
    main()
