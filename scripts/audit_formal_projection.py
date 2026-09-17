"""Descriptive final audit of a formally selected shared-projection checkpoint.

The 280 development clips were already used to choose the checkpoint. Paired
sentence intervals here are descriptive rechecks, not independent test CIs.
Original-checkpoint curves are not fabricated or regenerated: their saved
point metrics are reported separately without paired uncertainty.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_audio_activity_gate import SEEDS, velocity_clip_statistics, write_json
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.audit_projection_readiness import relative_error_audit
from scripts.train_formal_predictable_projection import SCHEMA, canonical_hash
from scripts.train_predictable_renderer import PredictableAudioHead, basic_metrics, center, sha, state_hash

MODES = ('full', 'zero', 'reverse', 'oracle')
KINDS = ('raw_motion', 'centered_residual')
GROUPS = {'upper_expression': [5, 6, 12, 13, 41, 42, 43, 44, 45], 'brows': list(range(41, 46)),
          'eyes_expression': [5, 6, 12, 13], 'mouth': list(range(14, 41)), 'jaw17': [17]}


def validate_curve_metadata(curves, recipe, target_shape):
    if curves.get('noise_seeds') != list(SEEDS) or recipe.get('noise_seeds') != list(SEEDS):
        raise ValueError('Require the fixed three generation noise seeds')
    if curves.get('decode_steps') != 12 or recipe['args'].get('decode_steps') != 12:
        raise ValueError('Formal audit requires twelve fixed Euler decoding steps')
    if set(curves.get('motion', {})) != set(map(str, SEEDS)):
        raise ValueError('Saved curve noise keys differ')
    for rows in curves['motion'].values():
        if set(rows) != set(MODES):
            raise ValueError('Saved curve intervention keys differ')
        for prediction in rows.values():
            if tuple(prediction.shape) != tuple(target_shape):
                raise ValueError('Saved curve and cache target shapes differ')


def validate_curve_binding(summary, binding, *, sidecar_path, sidecar_sha256,
                           curve_sha256, recipe_sha256, cache_sha256):
    record = summary.get('selected_development_curves_provenance') or {}
    schema = 'formal_predictable_projection_curves_provenance_v1'
    if record.get('schema') != schema:
        raise ValueError('Explicit final curve/checkpoint provenance is required')
    if record.get('sha256') != sidecar_sha256 or Path(record['path']).resolve() != Path(sidecar_path).resolve():
        raise ValueError('Summary curve provenance reference differs')
    expected = {'schema': schema, 'curve_sha256': curve_sha256, 'recipe_sha256': recipe_sha256,
                'selected_checkpoint_sha256': summary['selected_checkpoint_sha256'], 'cache_sha256': cache_sha256}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError('Curve provenance sidecar differs')


def assert_metric_agreement(actual, recorded):
    """Tie saved curves to summary by recomputing every native metric field."""
    for population in ('all', 'neutral', 'nonneutral'):
        for group in GROUPS:
            for kind in KINDS:
                for key in ('native_mse', 'r2_against_zero', 'pooled_centered_correlation'):
                    left, right = actual[population][group][kind][key], recorded[population][group][kind][key]
                    if left is None or right is None:
                        if left != right:
                            raise ValueError('Undefined metric differs between curves and summary')
                    elif not np.isclose(left, right, rtol=2e-7, atol=2e-9):
                        raise ValueError(f'Curve/summary metric mismatch: {population}/{group}/{kind}/{key}')
            if not np.isclose(actual[population][group]['velocity_mse_per_second'],
                              recorded[population][group]['velocity_mse_per_second'], rtol=2e-7, atol=2e-9):
                raise ValueError('Curve/summary velocity mismatch')


def final_checks(paired, neutral_raw, velocity, correlation_delta, teacher_delta):
    upper = paired['zero']['nonneutral']['upper_expression']
    reverse = paired['reverse']['nonneutral']['upper_expression']
    brows = paired['zero']['nonneutral']['brows']
    return {
        'upper_useful': upper['r2_improvement'] >= .005 and upper['r2_improvement_ci95'] is not None and upper['r2_improvement_ci95'][0] > 0,
        'upper_beats_reverse': reverse['r2_improvement'] > 0,
        'brows_tolerated': brows['r2_improvement'] > 0 and brows['r2_improvement_ci95'] is not None and brows['r2_improvement_ci95'][0] >= -.005,
        'neutral_mouth_one_sided_90_within_3pct': neutral_raw['mouth']['one_sided_90_upper'] is not None and neutral_raw['mouth']['one_sided_90_upper'] <= .03,
        'neutral_upper_one_sided_90_within_3pct': neutral_raw['upper_expression']['one_sided_90_upper'] is not None and neutral_raw['upper_expression']['one_sided_90_upper'] <= .03,
        'all_mouth_corr_within_01_vs_zero_and_original': min(row['all'] for row in correlation_delta.values()) >= -.01,
        'velocity_mean_within_5pct': all(velocity[pop][group]['relative_mse_increase'] is not None and
            velocity[pop][group]['relative_mse_increase'] <= .05 for pop, group in
            (('neutral', 'upper_expression'), ('neutral', 'mouth'), ('all', 'mouth'), ('nonneutral', 'upper_expression'))),
        'teacher_readout_within_1pp': teacher_delta >= -.01,
        'frozen_all_confirmed': True,
    }


def audit_curves(curves, reference, summary, original, *, samples=5000):
    q = reference['q']; target = q['motion'].float(); weight = q['valid'].float()
    common = q['channel_mask'].all(0)
    if not torch.equal(q['channel_mask'], common[None].expand_as(q['channel_mask'])):
        raise ValueError('Require common audited observed channels')
    if any(c >= len(common) or not common[c] for channels in GROUPS.values() for c in channels):
        raise ValueError('Required facial expression channels are not observed')
    groups = GROUPS
    baseline = reference['base']['b0'].float() + reference['identity']['baseline'].float()[:, None]
    targets = {'raw_motion': target, 'centered_residual': center(target - baseline, weight)}
    populations = {'all': list(range(len(target))), 'neutral': (q['emotion_id'] == 0).nonzero(as_tuple=True)[0].tolist(),
                   'nonneutral': (q['emotion_id'] != 0).nonzero(as_tuple=True)[0].tolist()}
    if not all(populations.values()):
        raise ValueError('Require both neutral and nonneutral development observations')
    stats, velocities, static_shifts, per_seed = {}, {}, {}, {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for mode, prediction in curves['motion'][str(seed)].items():
            metrics = basic_metrics(prediction, reference)
            assert_metric_agreement(metrics, summary['after'][str(seed)][mode])
            transformed = {'raw_motion': prediction, 'centered_residual': center(prediction - baseline, weight)}
            per_seed[str(seed)][mode] = {}
            for kind, pred in transformed.items():
                per_seed[str(seed)][mode][kind] = {}
                for group, channels in groups.items():
                    row = clip_statistics(pred, targets[kind], weight, channels)
                    key = mode, kind, group
                    stats[key] = stats.get(key, np.zeros_like(row)) + row / len(SEEDS)
                    per_seed[str(seed)][mode][kind][group] = {pop: scalar_summary(row[ids]) for pop, ids in populations.items()}
            delta = (torch.where(q['valid'][..., None], prediction - target, 0) * weight[..., None]).sum(1) / weight.sum(1)[:, None].clamp_min(1)
            for group, channels in groups.items():
                row = velocity_clip_statistics(prediction, reference, channels)
                key = mode, group
                velocities[key] = velocities.get(key, np.zeros_like(row)) + row / len(SEEDS)
                values = delta[:, channels].double().square().mean(-1).numpy()
                static_shifts[key] = static_shifts.get(key, np.zeros_like(values)) + values / len(SEEDS)
    scores = {mode: {kind: {group: {pop: scalar_summary(stats[(mode, kind, group)][ids]) for pop, ids in populations.items()}
        for group in groups} for kind in KINDS} for mode in MODES}
    paired = {base: {pop: {group: paired_summary(stats[('full', 'centered_residual', group)],
        stats[(base, 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in groups} for pop, ids in populations.items()} for base in ('zero', 'reverse', 'oracle')}
    neutral_raw = {group: relative_error_audit(stats[('full', 'raw_motion', group)][:, [0, 3]],
        stats[('zero', 'raw_motion', group)][:, [0, 3]], q['sentence_id'], populations['neutral'], samples=samples)
        for group in groups}
    velocity = {pop: {group: relative_error_audit(velocities[('full', group)], velocities[('zero', group)],
        q['sentence_id'], ids, samples=samples) for group in groups} for pop, ids in populations.items()}
    # Original curves are absent. Use mean per-seed correlations on BOTH sides
    # for this point comparison, rather than mixing two aggregation formulas.
    original_points, correlation_delta = {}, {'zero': {}, 'original_saved_point': {}}
    for population in populations:
        original_points[population] = {}
        for group in groups:
            original_points[population][group] = {key: float(np.mean([original[str(seed)]['full'][population][group]['raw_motion'][key]
                for seed in SEEDS])) for key in ('native_mse', 'pooled_centered_correlation')}
        correlation_delta['zero'][population] = scores['full']['raw_motion']['mouth'][population]['pooled_centered_correlation'] - scores['zero']['raw_motion']['mouth'][population]['pooled_centered_correlation']
        correlation_delta['original_saved_point'][population] = float(np.mean([
            summary['after'][str(seed)]['full'][population]['mouth']['raw_motion']['pooled_centered_correlation'] -
            original[str(seed)]['full'][population]['mouth']['raw_motion']['pooled_centered_correlation'] for seed in SEEDS]))
    teacher = {'full': [summary['after'][str(seed)]['full']['frozen_teacher_emotion_accuracy'] for seed in SEEDS],
               'original': [original[str(seed)]['full']['frozen_teacher_emotion_accuracy'] for seed in SEEDS]}
    teacher_delta = float(np.mean(teacher['full']) - np.mean(teacher['original']))
    checks = final_checks(paired, neutral_raw, velocity, correlation_delta, teacher_delta)
    return {'scores': scores, 'paired_dynamic': paired, 'neutral_raw_relative_vs_zero': neutral_raw,
        'velocity_relative_vs_zero': velocity, 'mouth_correlation_delta': correlation_delta,
        'original_saved_point_reference': original_points,
        'original_reference_note': 'Saved three-noise mean point metrics only; no original curves, no paired original uncertainty; correlation comparison uses per-seed differences on both sides',
        'teacher_accuracy': teacher, 'teacher_accuracy_delta': teacher_delta,
        'teacher_note': 'Saved frozen training-teacher readout, not independent emotion-quality evidence',
        'static_mean_shift_rms': {mode: {group: {pop: float(np.sqrt(static_shifts[(mode, group)][ids].mean()))
            for pop, ids in populations.items()} for group in groups} for mode in MODES},
        'per_seed': per_seed, 'checks': checks, 'single_seed_engineering_pass': all(checks.values()),
        'brows_significantly_positive': paired['zero']['nonneutral']['brows']['r2_improvement_ci95'] is not None and
            paired['zero']['nonneutral']['brows']['r2_improvement_ci95'][0] > 0}


def validate_artifacts(run):
    summary_path, provenance_path = run / 'summary.json', run / 'provenance.json'
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    recipe = provenance['recipe']; digest = canonical_hash(recipe)
    if summary.get('schema') != 'formal_predictable_projection_result_v1' or recipe.get('schema') != SCHEMA:
        raise ValueError('Wrong formal artifact schema')
    if provenance.get('recipe_sha256') != digest or summary.get('recipe_sha256') != digest:
        raise ValueError('Formal recipe hash mismatch')
    if not summary.get('frozen_unchanged') or not summary.get('head_unchanged') or summary.get('new_test_loaded') or summary.get('default_replaced'):
        raise ValueError('Formal frozen/test/default contract failed')
    paths = {key: Path(recipe['args'][key]) for key in ('cache', 'bundle', 'weights', 'checkpoint', 'config')}
    for key, path in paths.items():
        if sha(path) != recipe['input_sha256'][key]:
            raise ValueError(f'Formal input {key} hash mismatch')
    selected_path = Path(summary['selected_checkpoint'])
    if sha(selected_path) != summary['selected_checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    selected = torch.load(selected_path, map_location='cpu', weights_only=False)
    if selected.get('schema') != SCHEMA or selected.get('recipe') != recipe or selected.get('recipe_sha256') != digest:
        raise ValueError('Selected checkpoint recipe mismatch')
    if selected.get('frozen_state_sha256') != provenance['frozen_state_sha256'] or selected.get('head_sha256') != provenance['head_sha256']:
        raise ValueError('Selected checkpoint frozen/head provenance mismatch')
    if state_hash(selected['head']) != selected['head_sha256']:
        raise ValueError('Selected head tensor hash mismatch')
    projection = selected['local_projection'].get('weight')
    if projection is None or projection.shape != (64, 8) or not torch.isfinite(projection).all():
        raise ValueError('Invalid 512-parameter projection')
    if summary.get('has_eligible_checkpoint') and selected['completed_epochs'] != summary['best_epoch']:
        raise ValueError('Selected checkpoint epoch differs from best epoch')
    cache = torch.load(paths['cache'], map_location='cpu', weights_only=False)
    if cache.get('schema') != 'predictable_renderer_cache_v1':
        raise ValueError('Wrong formal renderer cache schema')
    for key in ('bundle', 'checkpoint', 'config'):
        if cache['provenance'][key + '_sha256'] != recipe['input_sha256'][key]:
            raise ValueError('Cache internal source provenance mismatch')
    cfg = yaml.safe_load(paths['config'].read_text(encoding='utf8'))
    if cfg['data']['emotion_classes'][0] != 'neutral':
        raise ValueError('Neutral index contract differs')
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    frozen = {key: value for key, value in checkpoint['model'].items() if not key.startswith('local_projection.')}
    if state_hash(frozen) != provenance['frozen_state_sha256']:
        raise ValueError('Original checkpoint frozen tensors differ')
    del checkpoint, frozen
    bundle = torch.load(paths['bundle'], map_location='cpu', weights_only=False)
    weights = torch.load(paths['weights'], map_location='cpu', weights_only=False)
    if weights['provenance']['bundle_sha256'] != recipe['input_sha256']['bundle']:
        raise ValueError('Fixed basis bundle differs')
    state = weights['states'][recipe['args']['mode'] + '_rank8']
    if state['rank'] != 8 or not torch.equal(state['train_ids'], bundle['train_ids']):
        raise ValueError('Fixed basis rank/training IDs differ')
    train = bundle['bundles']['internal']
    expected_head = PredictableAudioHead(state, train['motion_bins'], train['weight'])
    if state_hash(expected_head.state_dict()) != selected['head_sha256']:
        raise ValueError('Saved head differs from fixed train-only fit')
    reference = cache['splits']['validation']
    dev = bundle['bundles']['external_dev']
    for key in ('clip_id', 'sentence_id'):
        if reference['q'][key] != dev[key]:
            raise ValueError('Formal cache and bundle development IDs differ')
    del cache, bundle, weights, train, expected_head
    curves_path = run / 'selected_development_curves.pt'
    curves = torch.load(curves_path, map_location='cpu', weights_only=False)
    validate_curve_metadata(curves, recipe, reference['q']['motion'].shape)
    binding_path = run / 'selected_development_curves.provenance.json'
    if not binding_path.exists():
        raise ValueError('Explicit final curve/checkpoint provenance is required')
    binding = json.loads(binding_path.read_text(encoding='utf8'))
    validate_curve_binding(summary, binding, sidecar_path=binding_path, sidecar_sha256=sha(binding_path),
        curve_sha256=sha(curves_path), recipe_sha256=digest, cache_sha256=recipe['input_sha256']['cache'])
    binding_status = 'summary SHA256 -> curve sidecar -> selected checkpoint, recipe, cache and curves; saved metrics also recomputed'
    original_path = run / 'original_development.json'
    original = json.loads(original_path.read_text(encoding='utf8'))
    if set(original) != set(map(str, SEEDS)):
        raise ValueError('Original point-report noise seeds differ')
    original_accuracy = float(np.mean([original[str(seed)]['full']['frozen_teacher_emotion_accuracy'] for seed in SEEDS]))
    if not np.isclose(original_accuracy, selected['original_class_accuracy'], rtol=0, atol=1e-12):
        raise ValueError('Original class readout differs from the value stored in selected checkpoint')
    hashes = {'summary': sha(summary_path), 'provenance': sha(provenance_path), 'selected_checkpoint': sha(selected_path),
              'curves': sha(curves_path), 'original_point_report': sha(original_path), **recipe['input_sha256']}
    return summary, recipe, reference, curves, original, hashes, binding_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.samples < 1 or args.output.exists():
        raise ValueError('Require positive bootstrap samples and a fresh output path')
    torch.set_num_threads(4)
    summary, recipe, reference, curves, original, hashes, binding = validate_artifacts(args.run)
    result = audit_curves(curves, reference, summary, original, samples=args.samples)
    result['checks']['selected_checkpoint_was_eligible'] = bool(summary['has_eligible_checkpoint'] and summary['final_selection']['eligible'])
    result['single_seed_engineering_pass'] = all(result['checks'].values())
    report = {'schema': 'formal_projection_descriptive_audit_v1', 'run': str(args.run), 'hashes': hashes,
        'script_sha256': sha(__file__), 'training_seed': recipe['args']['seed'], 'method': recipe['args']['mode'],
        'selected_epoch': summary['best_epoch'], 'completed_epochs': summary['completed_epochs'],
        'has_eligible_checkpoint': summary['has_eligible_checkpoint'], 'curve_checkpoint_binding': binding,
        'bootstrap_samples': args.samples, 'bootstrap_seed': 45, 'noise_seeds': SEEDS,
        'noise_aggregation': 'Average per-clip sufficient error statistics across noise, then resample whole sentence clusters',
        'scope': 'This exact development set selected the formal checkpoint; intervals are descriptive post-selection checks, not independent validation or test confidence intervals',
        'test_loaded': False, 'default_replaced': False, 'independent_test_evidence': False,
        'single_seed_does_not_authorize_default_replacement': True, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'training_seed': report['training_seed'], 'checks': report['checks'],
        'single_seed_engineering_pass': report['single_seed_engineering_pass'], 'selected_epoch': report['selected_epoch']}), flush=True)


if __name__ == '__main__':
    main()
