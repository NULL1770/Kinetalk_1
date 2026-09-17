"""Read-only cross-identity development audit of a preselected formal adapter.

These shared-script native-val identities were not used to fit the dynamic
adapter. Three identities cannot establish broad identity generalization;
sentence-cluster intervals are conditional on these recorded identities.
No parameters, gates, ranks or checkpoints are selected by this script.
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
from scripts.audit_formal_projection import GROUPS, KINDS, assert_metric_agreement, final_checks, validate_curve_metadata
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.audit_projection_readiness import relative_error_audit
from scripts.evaluate_cross_identity_projection import RestoredFixedAudioHead, validate_cross_payload, validate_selected_run
from scripts.train_predictable_renderer import audio_features, basic_metrics, center, sha, state_hash

MODES = ('full', 'zero', 'reverse', 'oracle', 'original')


def groups_of_clips(query, speaker_to_id, emotion_classes):
    labels, speakers = query['emotion_id'].tolist(), query['speaker_id'].tolist()
    if set(speakers) != set(speaker_to_id.values()):
        raise ValueError('Query identities differ from locked preparation mapping')
    result = {'pooled/all': list(range(len(labels))),
              'pooled/neutral': [i for i, e in enumerate(labels) if e == 0],
              'pooled/nonneutral': [i for i, e in enumerate(labels) if e != 0]}
    for name, speaker in speaker_to_id.items():
        for population, predicate in [('all', lambda e: True), ('neutral', lambda e: e == 0), ('nonneutral', lambda e: e != 0)]:
            result[f'speaker/{name}/{population}'] = [i for i, (s, e) in enumerate(zip(speakers, labels)) if s == speaker and predicate(e)]
        result[f'leave_out/{name}/nonneutral'] = [i for i, (s, e) in enumerate(zip(speakers, labels)) if s != speaker and e != 0]
    for emotion in sorted(set(labels)):
        result['emotion/' + emotion_classes[emotion]] = [i for i, e in enumerate(labels) if e == emotion]
    return {key: ids for key, ids in result.items() if ids}


def validate_curve_sidecar(record, sidecar, *, path, actual_sha, curve_sha, recipe_sha, checkpoint_sha, cache_sha):
    schema = 'formal_predictable_projection_curves_provenance_v1'
    if record.get('schema') != schema or record.get('sha256') != actual_sha or Path(record['path']).resolve() != path.resolve():
        raise ValueError('Cross-identity summary curve sidecar reference differs')
    expected = {'schema': schema, 'curve_sha256': curve_sha, 'recipe_sha256': recipe_sha,
                'selected_checkpoint_sha256': checkpoint_sha, 'cache_sha256': cache_sha}
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise ValueError('Cross-identity curve provenance differs')


@torch.no_grad()
def lowrate_head_audit(payload, saved_head, classes, *, samples=5000):
    """Native-bin diagnostic, explicitly separate from full-frame generation."""
    head = RestoredFixedAudioHead(saved_head).cpu().eval()
    bundle, split = payload['bundle'], payload['split']
    q = split['q']; weight = bundle['weight'].float()
    target = center(bundle['motion_bins'].float(), weight)
    oracle_z = head.teacher(target, weight)
    audio_z = head(audio_features(bundle), weight)
    gate = 1 - split['affect']['emotion_logits'].detach().float().softmax(-1)[:, classes.index('neutral')]
    def decode(z):
        result = torch.zeros_like(target)
        result[..., head.channels] = (z * head.target_scale) @ head.basis.T
        return result
    values = {'motion_projection_oracle': decode(oracle_z),
              'motion_projection_oracle_gated': decode(oracle_z * gate[:, None, None]), 'audio_prediction': decode(audio_z),
              'audio_prediction_gated': decode(audio_z * gate[:, None, None]), 'zero': torch.zeros_like(target)}
    populations = groups_of_clips(q, payload['provenance']['speaker_to_id'], classes)
    rows = {(mode, group): clip_statistics(pred, target, weight, channels)
            for mode, pred in values.items() for group, channels in GROUPS.items()}
    scores = {pop: {mode: {group: scalar_summary(rows[(mode, group)][ids]) for group in GROUPS}
                   for mode in values} for pop, ids in populations.items()}
    comparisons = {pop: {mode: {group: paired_summary(rows[(mode, group)], rows[('zero', group)],
        q['sentence_id'], ids, samples=samples, seed=45) for group in GROUPS}
        for mode in ('motion_projection_oracle', 'motion_projection_oracle_gated', 'audio_prediction', 'audio_prediction_gated')}
        for pop, ids in populations.items()}
    return {'scores': scores, 'paired_vs_zero': comparisons,
        'scope': 'Real centered low-rate motion projected through the frozen train-only rank8 basis, and frozen audio controls decoded through that same basis. Native bin resolution only; these R2 values are not numerically comparable to original-frame renderer R2.',
        'no_fit': True, 'native_time_resolution': 'four-frame weighted bins',
        'interpretation': 'A high gated motion-projection oracle but weak generated motion-oracle suggests the frozen renderer/identity-conditioned interface also limits transfer; it is not solely an audio prediction failure. The ungated projection also isolates losses caused by the audio-derived activity gate.'}


def audit_cross_curves(selected_curves, original_curves, payload, summary, classes, *, samples=5000):
    split, provenance = payload['split'], payload['provenance']
    q = split['q']; target = q['motion'].float(); w = q['valid'].float()
    common = q['channel_mask'].all(0)
    if not torch.equal(q['channel_mask'], common[None].expand_as(q['channel_mask'])):
        raise ValueError('Require common observed channels')
    if any(not common[c] for channels in GROUPS.values() for c in channels):
        raise ValueError('Required expression channels missing')
    populations = groups_of_clips(q, provenance['speaker_to_id'], classes)
    baseline = split['base']['b0'].float() + split['identity']['baseline'].float()[:, None]
    targets = {'raw_motion': target, 'centered_residual': center(target - baseline, w)}
    stats, velocity, per_seed = {}, {}, {}
    for seed in SEEDS:
        curves = {**selected_curves['motion'][str(seed)], 'original': original_curves['motion'][str(seed)]['full']}
        per_seed[str(seed)] = {}
        for mode, prediction in curves.items():
            metrics = basic_metrics(prediction, split)
            recorded = summary['before'][str(seed)]['full'] if mode == 'original' else summary['after'][str(seed)][mode]
            assert_metric_agreement(metrics, recorded)
            transformed = {'raw_motion': prediction, 'centered_residual': center(prediction - baseline, w)}
            per_seed[str(seed)][mode] = {}
            for kind, pred in transformed.items():
                per_seed[str(seed)][mode][kind] = {}
                for group, channels in GROUPS.items():
                    row = clip_statistics(pred, targets[kind], w, channels)
                    key = mode, kind, group
                    stats[key] = stats.get(key, np.zeros_like(row)) + row / len(SEEDS)
                    per_seed[str(seed)][mode][kind][group] = {pop: scalar_summary(row[ids]) for pop, ids in populations.items() if pop.startswith('pooled/')}
            for group, channels in GROUPS.items():
                row = velocity_clip_statistics(prediction, split, channels)
                key = mode, group
                velocity[key] = velocity.get(key, np.zeros_like(row)) + row / len(SEEDS)
    scores = {population: {mode: {kind: {group: scalar_summary(stats[(mode, kind, group)][ids]) for group in GROUPS}
        for kind in KINDS} for mode in MODES} for population, ids in populations.items()}
    comparisons = {}
    for population, ids in populations.items():
        comparisons[population] = {}
        for base in ('zero', 'reverse', 'original', 'oracle'):
            comparisons[population][base] = {kind: {group: paired_summary(stats[('full', kind, group)],
                stats[(base, kind, group)], q['sentence_id'], ids, samples=samples, seed=45)
                for group in GROUPS} for kind in KINDS}
    relative_errors = {population: {base: {group: relative_error_audit(
        stats[('full', 'raw_motion', group)][:, [0, 3]], stats[(base, 'raw_motion', group)][:, [0, 3]],
        q['sentence_id'], ids, samples=samples) for group in GROUPS} for base in ('zero', 'original')}
        for population, ids in populations.items()}
    velocity_ratios = {population: {base: {group: relative_error_audit(velocity[('full', group)], velocity[(base, group)],
        q['sentence_id'], ids, samples=samples) for group in GROUPS} for base in ('zero', 'original')}
        for population, ids in populations.items()}
    correlation_delta = {base: {pop: scores['pooled/' + pop]['full']['raw_motion']['mouth']['pooled_centered_correlation'] -
        scores['pooled/' + pop][base]['raw_motion']['mouth']['pooled_centered_correlation'] for pop in ('all', 'neutral', 'nonneutral')}
        for base in ('zero', 'original')}
    teacher = {mode: [summary['before'][str(seed)]['full']['frozen_teacher_emotion_accuracy'] if mode == 'original' else
                     summary['after'][str(seed)][mode]['frozen_teacher_emotion_accuracy'] for seed in SEEDS] for mode in MODES}
    teacher_delta = float(np.mean(teacher['full']) - np.mean(teacher['original']))
    shared_paired = {base: {pop: comparisons['pooled/' + pop][base]['centered_residual'] for pop in ('all', 'neutral', 'nonneutral')}
                     for base in ('zero', 'reverse')}
    checks = final_checks(shared_paired, relative_errors['pooled/neutral']['zero'],
        {pop: velocity_ratios['pooled/' + pop]['zero'] for pop in ('all', 'neutral', 'nonneutral')}, correlation_delta, teacher_delta)
    transfer = {}
    for group in ('upper_expression', 'brows', 'eyes_expression'):
        pooled = comparisons['pooled/nonneutral']['zero']['centered_residual'][group]
        per_identity = {name: comparisons[f'speaker/{name}/nonneutral']['zero']['centered_residual'][group]
            for name in provenance['speaker_to_id'] if f'speaker/{name}/nonneutral' in comparisons}
        leave_out = {name: comparisons[f'leave_out/{name}/nonneutral']['zero']['centered_residual'][group]
            for name in provenance['speaker_to_id'] if f'leave_out/{name}/nonneutral' in comparisons}
        transfer[group] = {'pooled_delta_r2': pooled['r2_improvement'], 'pooled_ci95': pooled['r2_improvement_ci95'],
            'identity_deltas': {name: row['r2_improvement'] for name, row in per_identity.items()},
            'identity_error_reduction_sse': {name: row['baseline']['sse'] - row['candidate']['sse'] for name, row in per_identity.items()},
            'leave_one_identity_out_delta_r2': {name: row['r2_improvement'] for name, row in leave_out.items()},
            'positive_identity_count': sum(row['r2_improvement'] > 0 for row in per_identity.values()),
            'all_leave_one_identity_out_positive': all(row['r2_improvement'] > 0 for row in leave_out.values()),
            'positive_pooled_ci': pooled['r2_improvement_ci95'] is not None and pooled['r2_improvement_ci95'][0] > 0}
    oracle_vs_zero = {population: {group: paired_summary(stats[('oracle', 'centered_residual', group)],
        stats[('zero', 'centered_residual', group)], q['sentence_id'], ids, samples=samples, seed=45)
        for group in GROUPS} for population, ids in populations.items()}
    return {'population_counts': {key: {'clips': len(ids), 'sentences': len({q['sentence_id'][i] for i in ids}),
                'identities': len({int(q['speaker_id'][i]) for i in ids})} for key, ids in populations.items()},
        'scores': scores, 'paired_full_vs': comparisons, 'raw_relative_full_vs': relative_errors,
        'velocity_relative_full_vs': velocity_ratios, 'mouth_correlation_delta': correlation_delta,
        'teacher_accuracy': teacher, 'teacher_accuracy_delta': teacher_delta, 'transfer_diagnostics': transfer,
        'generated_motion_oracle_vs_zero': oracle_vs_zero,
        'same_engineering_checks': checks, 'cross_identity_engineering_pass': all(checks.values()), 'per_seed_pooled': per_seed,
        'identity_sampling_note': 'Only these three identities are observed. Bootstrap clusters by sentence, conditional on these identities; no population-of-identities confidence claim.',
        'teacher_note': 'Saved frozen training-motion-teacher class readout is a diagnostic, not an independent emotion evaluator'}


def load_verified(args):
    evaluation = args.evaluation
    summary_path, provenance_path = evaluation / 'summary.json', evaluation / 'provenance.json'
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    if summary.get('schema') != 'cross_identity_projection_evaluation_v1' or summary.get('provenance') != provenance:
        raise ValueError('Cross-identity evaluation provenance differs')
    if not summary.get('frozen_unchanged') or not summary.get('head_unchanged') or summary.get('default_replaced'):
        raise ValueError('Cross-identity frozen/default contract failed')
    for key in ('new_test_loaded', 'training_targets_read', 'rank_or_scale_or_gate_refitted'):
        if provenance.get(key) is not False:
            raise ValueError('Cross-identity test/refit isolation failed')
    formal_provenance_path, formal_summary_path = args.formal_run / 'provenance.json', args.formal_run / 'summary.json'
    if sha(formal_provenance_path) != provenance['formal_provenance_sha256'] or sha(formal_summary_path) != provenance['formal_summary_sha256']:
        raise ValueError('Formal run provenance/selection changed')
    formal_provenance = json.loads(formal_provenance_path.read_text(encoding='utf8'))
    formal_summary = json.loads(formal_summary_path.read_text(encoding='utf8'))
    selected_path = args.selected_checkpoint or Path(formal_summary['selected_checkpoint'])
    selected = torch.load(selected_path, map_location='cpu', weights_only=False)
    selected_hash = sha(selected_path)
    recipe = validate_selected_run(formal_provenance, formal_summary, selected, selected_hash)
    if selected_hash != provenance['selected_checkpoint_sha256'] or selected['recipe_sha256'] != provenance['recipe_sha256']:
        raise ValueError('Cross-identity candidate differs from selected formal checkpoint')
    if state_hash(selected['local_projection']) != summary['local_projection_sha256']:
        raise ValueError('Cross-identity projection differs')
    if selected['head_sha256'] != provenance['head_sha256'] or selected['frozen_state_sha256'] != provenance['frozen_state_sha256']:
        raise ValueError('Cross-identity frozen/head hashes differ')
    cross_hash = sha(args.cross_data)
    if cross_hash != provenance['cross_data_sha256']:
        raise ValueError('Cross-identity payload hash mismatch')
    payload = torch.load(args.cross_data, map_location='cpu', weights_only=False, mmap=True)
    config_path = args.config or Path(recipe['args']['config'])
    checkpoint_path = args.checkpoint or Path(recipe['args']['checkpoint'])
    bundle_path = args.formal_bundle or Path(recipe['args']['bundle'])
    if sha(bundle_path) != provenance['formal_bundle_sha256'] or sha(bundle_path) != recipe['input_sha256']['bundle']:
        raise ValueError('Formal bundle hash mismatch')
    formal_bundle = torch.load(bundle_path, map_location='cpu', weights_only=False, mmap=True)
    ck_hash, cfg_hash = sha(checkpoint_path), sha(config_path)
    if ck_hash != provenance['checkpoint_sha256'] or cfg_hash != provenance['config_sha256']:
        raise ValueError('Frozen checkpoint/config hash mismatch')
    prepared = validate_cross_payload(payload, formal_bundle, recipe, ck_hash, cfg_hash)
    if prepared != provenance['cross_preparation_provenance']:
        raise ValueError('Nested cross preparation provenance differs')
    cfg = yaml.safe_load(config_path.read_text(encoding='utf8'))
    if cfg['data']['emotion_classes'][0] != 'neutral':
        raise ValueError('Neutral index contract differs')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if state_hash(checkpoint['model']) != prepared['frozen_before'] or prepared['frozen_after'] != prepared['frozen_before']:
        raise ValueError('Prepared frozen full model differs from original checkpoint')
    if state_hash({k: v for k, v in checkpoint['model'].items() if not k.startswith('local_projection.')}) != selected['frozen_state_sha256']:
        raise ValueError('Frozen backbone tensor hash differs')
    del formal_bundle, checkpoint
    curves = {}
    for role, expected_checkpoint in (('selected', selected_hash), ('original', ck_hash)):
        path = evaluation / f'{role}_curves.pt'
        sidecar_path = evaluation / f'{role}_curves.provenance.json'
        sidecar = json.loads(sidecar_path.read_text(encoding='utf8'))
        validate_curve_sidecar(summary['curves'][role], sidecar, path=sidecar_path, actual_sha=sha(sidecar_path),
            curve_sha=sha(path), recipe_sha=provenance['recipe_sha256'], checkpoint_sha=expected_checkpoint, cache_sha=cross_hash)
        curves[role] = torch.load(path, map_location='cpu', weights_only=False)
    validate_curve_metadata(curves['selected'], recipe, payload['split']['q']['motion'].shape)
    original = curves['original']
    if original.get('noise_seeds') != list(SEEDS) or original.get('decode_steps') != 12 or set(original['motion']) != set(map(str, SEEDS)):
        raise ValueError('Original curve noise or decode metadata mismatch')
    if any(set(row) != {'full'} or row['full'].shape != payload['split']['q']['motion'].shape for row in original['motion'].values()):
        raise ValueError('Original curve mode/shape mismatch')
    hashes = {'summary': sha(summary_path), 'provenance': sha(provenance_path), 'cross_data': cross_hash,
        'selected_checkpoint': selected_hash, 'formal_summary': sha(formal_summary_path), 'formal_provenance': sha(formal_provenance_path),
        'selected_curves': sha(evaluation / 'selected_curves.pt'), 'original_curves': sha(evaluation / 'original_curves.pt')}
    return summary, payload, curves, cfg['data']['emotion_classes'], recipe, hashes, selected['head']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'cross-data', 'formal-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('selected-checkpoint', 'formal-bundle', 'checkpoint', 'config'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.samples < 1 or args.output.exists():
        raise ValueError('Require positive bootstrap count and fresh output')
    torch.set_num_threads(4)
    summary, payload, curves, classes, recipe, hashes, saved_head = load_verified(args)
    result = audit_cross_curves(curves['selected'], curves['original'], payload, summary, classes, samples=args.samples)
    result['lowrate_projection_diagnostic'] = lowrate_head_audit(payload, saved_head, classes, samples=args.samples)
    report = {'schema': 'cross_identity_projection_paired_audit_v1', 'hashes': hashes, 'script_sha256': sha(__file__),
        'training_seed': recipe['args']['seed'], 'selected_epoch': summary['provenance']['selected_epoch'],
        'bootstrap_samples': args.samples, 'bootstrap_seed': 45, 'noise_seeds': SEEDS,
        'scope': 'Preselected adapter evaluated on native-val cross-identity development with shared scripts; not independent novel-sentence testing or broad unseen-identity proof. B0/pretrained exposure remains unknown.',
        'selection': 'No refitting, checkpoint/rank/gate/temperature tuning; each identity and emotion is reported including failures',
        'test_loaded': False, 'default_replaced': False, 'noise_aggregation': 'Average per-clip sufficient errors over noise, then sentence-cluster resampling', **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'transfer': result['transfer_diagnostics'], 'checks': result['same_engineering_checks'],
        'cross_identity_engineering_pass': result['cross_identity_engineering_pass']}), flush=True)


if __name__ == '__main__':
    main()
