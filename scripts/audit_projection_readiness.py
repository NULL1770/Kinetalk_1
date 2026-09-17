"""Fixed engineering readiness checks for one projection-only training seed.

Read-only native-generation audit. Uses paired sentence-cluster bootstrap
after averaging per-clip errors over fixed noise seeds, never predictions.
An individual pass still requires the second training seed and data/resource
checks specified in FORMAL_TRAINING_READINESS.md before a formal run.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_audio_activity_gate import SEEDS, velocity_clip_statistics, write_json
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.train_predictable_renderer import center, sha


def relative_error_audit(candidate, baseline, sentences, indices, *, samples=5000, seed=45):
    """Bootstrap the paired ratio of summed SSE, not mean per-clip ratios."""
    ids = np.asarray(indices, dtype=np.int64)
    if not len(ids):
        return None
    if not np.array_equal(candidate[ids, -1], baseline[ids, -1]):
        raise ValueError('Error ratios require identical observation counts')
    groups = {}
    for i in ids:
        row = groups.setdefault(str(sentences[i]), np.zeros(2))
        row += [candidate[i, 0], baseline[i, 0]]
    total = np.stack(list(groups.values()))
    denominator = total[:, 1].sum()
    point = float(total[:, 0].sum() / denominator - 1) if denominator > 0 else None
    upper, interval = None, None
    if len(groups) > 1 and samples > 0:
        selected = np.random.default_rng(seed).integers(len(total), size=(samples, len(total)))
        draws = total[selected].sum(1)
        ratios = draws[draws[:, 1] > 0, 0] / draws[draws[:, 1] > 0, 1] - 1
        if len(ratios):
            upper = float(np.quantile(ratios, .9))
            interval = np.quantile(ratios, [.025, .975]).tolist()
    return {'clips': len(ids), 'sentences': len(groups), 'relative_mse_increase': point,
            'one_sided_90_upper': upper, 'ci95': interval, 'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh readiness report')
    torch.set_num_threads(4)
    summary_path = args.run / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    if summary.get('adaptation') != 'projection_only' or not all(summary.get(key) for key in
        ('frozen_unchanged', 'renderer_unchanged', 'head_unchanged')) or summary.get('new_test_loaded'):
        raise ValueError('Require completed strictly frozen projection-only development run')
    if summary['frozen_after'] != summary['provenance']['frozen_before']:
        raise ValueError('Frozen state hash mismatch')
    reference_path = args.run / 'validation_reference.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    q = reference['q']; target = q['motion'].float(); weight = q['valid'].float()
    common = q['channel_mask'].all(0)
    if not torch.equal(q['channel_mask'], common[None].expand_as(q['channel_mask'])):
        raise ValueError('Require common audited channel layout')
    groups = {'upper_expression': [5, 6, 12, 13, 41, 42, 43, 44, 45], 'brows': list(range(41, 46)),
              'eyes_expression': [5, 6, 12, 13], 'mouth': list(range(14, 41)), 'jaw17': [17]}
    groups = {group: [c for c in channels if common[c]] for group, channels in groups.items()}
    baseline = reference['base']['b0'].float() + reference['identity']['baseline'].float()[:, None]
    targets = {'raw_motion': target, 'centered_residual': center(target - baseline, weight)}
    populations = {'all': list(range(len(target))), 'neutral': (q['emotion_id'] == 0).nonzero(as_tuple=True)[0].tolist(),
                   'nonneutral': (q['emotion_id'] != 0).nonzero(as_tuple=True)[0].tolist()}
    if not all(populations.values()):
        raise ValueError('Readiness requires both neutral and nonneutral evaluation coverage')
    stats, velocity, per_noise, static_shifts, curve_hashes = {}, {}, {}, {}, {}
    for seed in SEEDS:
        final_path = args.run / f'final_seed{seed}_curves.pt'
        original_path = args.run / f'original_seed{seed}_curves.pt'
        final = torch.load(final_path, map_location='cpu', weights_only=False)
        original = torch.load(original_path, map_location='cpu', weights_only=False)
        for value in (final, original):
            if value['noise_seed'] != seed or value['decode_steps'] != 12:
                raise ValueError('Require fixed matched noise and decode settings')
        if set(final['motion']) != {'full', 'zero', 'reverse', 'oracle'}:
            raise ValueError('Require all final interventions')
        curves = {**final['motion'], 'original': original['motion']['full']}
        curve_hashes[str(seed)] = {'final': sha(final_path), 'original': sha(original_path)}
        per_noise[str(seed)] = {}
        for mode, prediction in curves.items():
            if prediction.shape != target.shape:
                raise ValueError('Curve/reference shapes differ')
            transformed = {'raw_motion': prediction, 'centered_residual': center(prediction - baseline, weight)}
            per_noise[str(seed)][mode] = {}
            for kind, pred in transformed.items():
                per_noise[str(seed)][mode][kind] = {}
                for group, channels in groups.items():
                    row = clip_statistics(pred, targets[kind], weight, channels)
                    key = mode, kind, group
                    stats[key] = stats.get(key, np.zeros_like(row)) + row / len(SEEDS)
                    per_noise[str(seed)][mode][kind][group] = {pop: scalar_summary(row[ids]) for pop, ids in populations.items()}
            for group, channels in groups.items():
                row = velocity_clip_statistics(prediction, reference, channels)
                key = mode, group
                velocity[key] = velocity.get(key, np.zeros_like(row)) + row / len(SEEDS)
            for group, channels in groups.items():
                delta = ((prediction - target) * weight[..., None]).sum(1) / weight.sum(1)[:, None].clamp_min(1)
                values = delta[:, channels].double().square().mean(-1).numpy()
                static_shifts[(mode, group)] = static_shifts.get((mode, group), np.zeros_like(values)) + values / len(SEEDS)
    modes = ('full', 'zero', 'reverse', 'oracle', 'original')
    scores = {mode: {kind: {group: {pop: scalar_summary(stats[(mode, kind, group)][ids])
        for pop, ids in populations.items()} for group in groups} for kind in targets} for mode in modes}
    paired = {base: {pop: {group: paired_summary(stats[('full', 'centered_residual', group)],
        stats[(base, 'centered_residual', group)], q['sentence_id'], ids, samples=args.samples, seed=45)
        for group in groups} for pop, ids in populations.items()} for base in ('zero', 'reverse', 'original')}
    neutral_raw = {base: {group: relative_error_audit(stats[('full', 'raw_motion', group)][:, [0, 3]],
        stats[(base, 'raw_motion', group)][:, [0, 3]], q['sentence_id'], populations['neutral'], samples=args.samples)
        for group in groups} for base in ('zero', 'original')}
    velocity_ratios = {base: {pop: {group: relative_error_audit(velocity[('full', group)], velocity[(base, group)],
        q['sentence_id'], ids, samples=args.samples) for group in groups} for pop, ids in populations.items()}
        for base in ('zero', 'original')}
    corr_drop = {base: {pop: scores['full']['raw_motion']['mouth'][pop]['pooled_centered_correlation'] -
        scores[base]['raw_motion']['mouth'][pop]['pooled_centered_correlation'] for pop in populations}
        for base in ('zero', 'original')}
    teacher = {'full': [summary['after'][str(seed)]['full']['frozen_teacher_emotion_accuracy'] for seed in SEEDS],
               'original': [summary['before'][str(seed)]['full']['frozen_teacher_emotion_accuracy'] for seed in SEEDS]}
    teacher_drop = float(np.mean(teacher['full']) - np.mean(teacher['original']))
    upper = paired['zero']['nonneutral']['upper_expression']
    reverse = paired['reverse']['nonneutral']['upper_expression']
    brows = paired['zero']['nonneutral']['brows']
    checks = {
        'upper_useful': upper['r2_improvement'] >= .005 and upper['r2_improvement_ci95'] is not None and upper['r2_improvement_ci95'][0] > 0,
        'upper_beats_reverse': reverse['r2_improvement'] > 0,
        'brows_tolerated': brows['r2_improvement'] > 0 and brows['r2_improvement_ci95'] is not None and brows['r2_improvement_ci95'][0] >= -.005,
        'neutral_mouth_one_sided_90_within_3pct': neutral_raw['zero']['mouth']['one_sided_90_upper'] is not None and neutral_raw['zero']['mouth']['one_sided_90_upper'] <= .03,
        'neutral_upper_one_sided_90_within_3pct': neutral_raw['zero']['upper_expression']['one_sided_90_upper'] is not None and neutral_raw['zero']['upper_expression']['one_sided_90_upper'] <= .03,
        'all_mouth_corr_within_01_vs_zero_and_original': min(corr_drop[base]['all'] for base in corr_drop) >= -.01,
        # Fixed protocol says no >5% systematic velocity increase. Report the
        # full interval, and use the paired aggregate point as this check.
        'velocity_mean_within_5pct': all(velocity_ratios['zero'][pop][group]['relative_mse_increase'] <= .05
            for pop, group in (('neutral', 'upper_expression'), ('neutral', 'mouth'), ('all', 'mouth'), ('nonneutral', 'upper_expression'))),
        'teacher_readout_within_1pp': teacher_drop >= -.01,
        'frozen_all_confirmed': True,
    }
    report = {'schema': 'projection_readiness_single_seed_v1', 'run': str(args.run),
        'training_seed': summary['provenance']['args']['seed'], 'audio_gate': summary['provenance'].get('audio_activity_gate', 'disabled'),
        'source_summary_sha256': sha(summary_path), 'reference_sha256': sha(reference_path), 'curve_sha256': curve_hashes,
        'script_sha256': sha(__file__), 'bootstrap_samples': args.samples, 'bootstrap_seed': 45,
        'noise_seeds': SEEDS, 'noise_aggregation': 'mean per-clip sufficient error statistics, followed by sentence bootstrap',
        'scores': scores, 'paired_dynamic': paired, 'neutral_raw_relative': neutral_raw, 'velocity_relative': velocity_ratios,
        'mouth_correlation_delta': corr_drop, 'teacher_accuracy': teacher, 'teacher_accuracy_delta': teacher_drop,
        'static_mean_shift_rms': {mode: {group: {pop: float(np.sqrt(static_shifts[(mode, group)][ids].mean()))
            for pop, ids in populations.items()} for group in groups} for mode in modes},
        'per_noise': per_noise, 'checks': checks, 'single_seed_engineering_pass': all(checks.values()),
        'brows_significantly_positive': brows['r2_improvement_ci95'][0] > 0,
        'not_formal_authorization': 'Requires two training seeds, PCA/context review, locked data roles, resources and raw supervision inspection; this report alone never launches training',
        'teacher_note': 'Same frozen training motion teacher, not independent perceptual evaluator'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'seed': report['training_seed'], 'checks': checks, 'pass': report['single_seed_engineering_pass'],
        'upper': [upper['r2_improvement'], upper['r2_improvement_ci95']],
        'neutral_mouth': neutral_raw['zero']['mouth'], 'brows': [brows['r2_improvement'], brows['r2_improvement_ci95']]}), flush=True)


if __name__ == '__main__':
    main()
