"""Read-only evaluation of the epoch10 checkpoint after SSH/service interruption.

Never resumes training or calls the incomplete run a completed18epoch probe.
The user shortened future diagnostic budgets to8epochs; saved epoch8/10 reports
are compared at equal budgets, with full curve audit only for saved epoch10.
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
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.audit_audio_activity_gate import SEEDS, write_json
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_predictable_motion_predictions import paired_summary, scalar_summary
from scripts.audit_projection_centered_probe import validate_centered_pair
from scripts.audit_projection_mean_dynamic_tradeoff import (
    exact_error_components, average_error_components, component_summary, paired_component_change,
)
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_projection_rollout_probe import validate_recipe_pair
from scripts.audit_teacher_schedule_probe import (
    GROUPS, KINDS, MODES, audit_populations, load_arm, statistics_for_curves, validate_training_inputs,
)
from scripts.train_formal_predictable_projection import canonical_hash, evaluate_compact, frozen_hash, save_checkpoint
from scripts.train_predictable_renderer import PredictableAudioHead, observed, sha, state_hash
from scripts.train_projection_centered_rollout_probe import SCHEMA
from scripts.train_projection_schedule_ablation import curve_binding, diagnostics


def audit_single(curves, reference, *, samples=5000):
    q = reference['q']; populations = audit_populations(q)
    stats, velocity, _ = statistics_for_curves(curves, reference)
    scores = {mode: {kind: {group: {pop: scalar_summary(stats[(mode, kind, group)][ids])
        for pop, ids in populations.items()} for group in GROUPS} for kind in KINDS} for mode in MODES}
    contrasts = {base: {pop: {group: paired_summary(stats[('full', 'centered_residual', group)],
        stats[(base, 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for pop, ids in populations.items()} for base in ('zero', 'reverse', 'oracle')}
    oracle = {pop: {group: paired_summary(stats[('oracle', 'centered_residual', group)],
        stats[('zero', 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for pop, ids in populations.items()}
    raw_relative = {pop: {group: relative_error_audit(stats[('full', 'raw_motion', group)][:, [0, 3]],
        stats[('zero', 'raw_motion', group)][:, [0, 3]], q['sentence_id'], ids, samples=samples)
        for group in GROUPS} for pop, ids in populations.items()}
    velocity_relative = {pop: {group: relative_error_audit(velocity[('full', group)], velocity[('zero', group)],
        q['sentence_id'], ids, samples=samples) for group in GROUPS} for pop, ids in populations.items()}
    mouth_corr = {pop: scores['full']['raw_motion']['mouth'][pop]['pooled_centered_correlation'] -
        scores['zero']['raw_motion']['mouth'][pop]['pooled_centered_correlation'] for pop in populations}
    components = {}
    for mode in MODES:
        for group, channels in GROUPS.items():
            components[(mode, group)] = average_error_components([exact_error_components(
                curves['motion'][str(seed)][mode], q['motion'], observed(q), channels,
                reference['base']['b0'], reference['identity']['baseline']) for seed in SEEDS])
    decomposition = {mode: {group: {pop: component_summary(components[(mode, group)][ids])
        for pop, ids in populations.items()} for group in GROUPS} for mode in MODES}
    component_change = {group: {pop: paired_component_change(components[('full', group)], components[('zero', group)],
        q['sentence_id'], ids, samples=samples) for pop, ids in populations.items()} for group in GROUPS}
    upper, brows = contrasts['zero']['nonneutral']['upper_expression'], contrasts['zero']['nonneutral']['brows']
    checks = {'upper_positive_vs_zero': upper['r2_improvement'] > 0,
        'upper_ci95_lower_positive': upper['r2_improvement_ci95'][0] > 0,
        'upper_positive_vs_reverse': contrasts['reverse']['nonneutral']['upper_expression']['r2_improvement'] > 0,
        'brows_positive_and_ci_lower_at_least_minus005': brows['r2_improvement'] > 0 and brows['r2_improvement_ci95'][0] >= -.005,
        'neutral_mouth_one_sided90_within_3pct': raw_relative['neutral']['mouth']['one_sided_90_upper'] <= .03,
        'neutral_upper_one_sided90_within_3pct': raw_relative['neutral']['upper_expression']['one_sided_90_upper'] <= .03,
        'all_mouth_corr_within_01': mouth_corr['all'] >= -.01,
        'velocity_mean_within_5pct': all(velocity_relative[pop][group]['relative_mse_increase'] <= .05
            for pop, group in (('neutral', 'mouth'), ('neutral', 'upper_expression'), ('all', 'mouth'), ('nonneutral', 'upper_expression')))}
    return {'scores': scores, 'full_vs_same_arm': contrasts, 'oracle_vs_zero': oracle,
        'raw_relative_vs_zero': raw_relative, 'velocity_relative_vs_zero': velocity_relative,
        'mouth_correlation_full_minus_zero': mouth_corr, 'decomposition': decomposition,
        'decomposition_full_minus_zero': component_change, 'existing_engineering_checks': checks,
        'checks_scope': 'Dynamic/neutral/mouth checks only; nonneutral raw/mean and global readout must be inspected separately. Not publication or deployment certification.',
        'population_counts': {pop: {'clips': len(ids), 'sentences': len({q['sentence_id'][i] for i in ids}),
            'identities': len({int(q['speaker_id'][i]) for i in ids})} for pop, ids in populations.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('centered', 'raw', 'flow-control', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Require fresh output directory')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint_path = args.centered / 'last.pt'
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    recipe = json.loads((args.centered / 'provenance.json').read_text())['recipe']
    if ck.get('schema') != SCHEMA or ck.get('completed_epochs') != 10 or ck.get('step') != 1450 or ck.get('recipe') != recipe or ck.get('recipe_sha256') != canonical_hash(recipe):
        raise ValueError('Require the actual complete epoch10 checkpoint from the interrupted run')
    raw_recipe = json.loads((args.raw / 'provenance.json').read_text())['recipe']
    validate_centered_pair(recipe, raw_recipe)
    flow = load_arm(args.flow_control, 'constant_teacher')
    validate_recipe_pair(raw_recipe, flow['recipe'])
    for epoch in range(1, 11):
        left = json.loads((args.centered / f'epoch{epoch:03d}.json').read_text())
        right = json.loads((args.raw / f'epoch{epoch:03d}.json').read_text())
        for key in ('minibatch_sha256', 'noise_time_sha256', 'teacher_choice_draw_sha256', 'step', 'samples_seen', 'teacher_fraction'):
            if left[key] != right[key] or right[key] != flow['records'][epoch][key]:
                raise ValueError(f'Unequal matched training evidence: epoch{epoch}/{key}')
        if epoch == 10:
            for key in ('minibatch_sha256', 'noise_time_sha256', 'teacher_choice_draw_sha256'):
                if ck[key] != left[key]: raise ValueError('last checkpoint does not match epoch10 record')
    reference, lock, scope, fixed = validate_training_inputs({'constant_teacher': flow})
    if ck['head_sha256'] != fixed['fixed_head'] or state_hash(ck['head']) != fixed['fixed_head'] or ck['frozen_state_sha256'] != fixed['frozen_backbone']:
        raise ValueError('Changed frozen model/head')
    for path, digest in recipe['source_sha256'].items():
        if sha(path) != digest: raise ValueError('Training source changed: ' + path)
    cfg = yaml.safe_load(Path(recipe['args']['config']).read_text())
    original = torch.load(recipe['args']['checkpoint'], map_location='cpu', weights_only=False)
    system = NeutralAffectSystem(cfg).cuda().eval()
    system.load_state_dict(original['model'], strict=True); system.local_projection.load_state_dict(ck['local_projection'], strict=True)
    if frozen_hash(system) != fixed['frozen_backbone']: raise ValueError('Restoration changed frozen system')
    bundle = torch.load(recipe['args']['bundle'], map_location='cpu', weights_only=False, mmap=True)
    fitted = torch.load(recipe['args']['weights'], map_location='cpu', weights_only=False)
    tr = bundle['bundles']['internal']
    head = PredictableAudioHead(fitted['states']['rrr_rank8'], tr['motion_bins'], tr['weight']).cuda().eval()
    head.load_state_dict(ck['head'], strict=True)
    args.output.mkdir(parents=True)
    archived_checkpoint = args.output / 'epoch010_interrupted.pt'
    save_checkpoint(archived_checkpoint, ck)
    curves_path = args.output / 'epoch010_curves.pt'
    reports = evaluate_compact(system, head, reference, bundle['bundles']['external_dev'], device='cuda',
        modes=MODES, curves_path=curves_path)
    saved = json.loads((args.centered / 'development_epoch010.json').read_text())['noise_reports']
    curves = torch.load(curves_path, map_location='cpu', weights_only=False)
    for seed in SEEDS:
        for mode in ('full', 'zero'):
            assert_metric_agreement(reports[str(seed)][mode], saved[str(seed)][mode])
        if not torch.equal(curves['motion'][str(seed)]['zero'], flow['curves'][18]['motion'][str(seed)]['zero']):
            raise ValueError('Frozen zero-local generation differs')
    binding = curve_binding(curves_path, archived_checkpoint, recipe, recipe['input_sha256']['cache'])
    equal_budget = {str(epoch): {arm: json.loads((run / f'development_epoch{epoch:03d}.json').read_text())['diagnostics']
        for arm, run in (('centered_rollout', args.centered), ('raw_rollout', args.raw))} for epoch in (2, 4, 6, 8, 10)}
    audit = audit_single(curves, reference)
    readout = {mode: float(np.mean([reports[str(seed)][mode]['frozen_teacher_emotion_accuracy'] for seed in SEEDS])) for mode in MODES}
    result = {'schema': 'interrupted_centered_epoch010_audit_v1', 'training_completed_epochs': 10,
        'training_optimizer_steps': 1450, 'original_planned_epochs': 18, 'original_primary_completed': False,
        'stop_reason': 'Connection/service interruption, then user explicitly requested shorter8epoch future diagnostic budget; no resumed training',
        'checkpoint_chosen_by_best_metric': False, 'checkpoint_reason': 'Last complete saved checkpoint, not best',
        'current_source_sha256': sha(__file__), 'last_checkpoint_sha256': sha(checkpoint_path),
        'recipe_sha256': canonical_hash(recipe), 'curve_provenance': binding, 'frozen_hashes': fixed,
        'split_lock': lock, 'data_scope': scope, 'same_rng_first10epochs': True,
        'new_optimizer_steps': 0, 'actual_epoch010': audit, 'noise_reports': reports,
        'diagnostics': diagnostics(reports), 'equal_budget_saved_diagnostics': equal_budget,
        'equal_budget_note': 'Saved per-noise evaluation summaries, not full saved curves/checkpoints at epoch8/raw10. Do not claim paired-curve CI or equal-epoch raw10 reverse/oracle comparison.',
        'frozen_teacher_readout': {'mean_accuracy': readout, 'note': 'Frozen training motion teacher, not independent emotion/perceptual evaluation'},
        'scope': 'Internal405development, 3heldout dynamic-adapter identities; shared sentences, inherited frozen backbone exposure possible',
        'outer280_loaded': False, 'new_identity439_loaded': False, 'test_loaded': False, 'default_replaced': False}
    write_json(args.output / 'report.json', result)
    print(json.dumps({'complete': True, 'epoch': 10, 'checks': audit['existing_engineering_checks'],
        'delta_r2': {g: row['r2_improvement'] for g, row in audit['full_vs_same_arm']['zero']['nonneutral'].items()}}), flush=True)


if __name__ == '__main__':
    main()
