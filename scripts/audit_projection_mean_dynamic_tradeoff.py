"""Exact observed-frame mean/dynamic error decomposition of saved projections.

For every clip and channel, raw SSE equals centered error SSE plus observation
count times squared temporal mean error. Region/population reports sum these
sufficient errors before dividing by counts. No refit, new decoding or tuning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_audio_activity_gate import SEEDS, write_json
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_projection_rollout_probe import load_rollout, validate_matched_rng, validate_recipe_pair
from scripts.audit_teacher_schedule_probe import (
    ARMS, EPOCHS, GROUPS, MODES, audit_populations, load_arm, validate_curves,
    validate_rng_evidence, validate_training_inputs,
)
from scripts.train_predictable_renderer import basic_metrics, observed, sha

COLUMNS = ('raw_sse', 'centered_error_sse', 'mean_error_sse', 'observed_values')


def exact_error_components(prediction, target, mask, channels, base, identity):
    """Return clip sufficient statistics with arbitrary frame/channel masks."""
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError('Prediction/target/boolean observed mask shapes must agree')
    if base.shape != target.shape or identity.shape != (target.shape[0], target.shape[2]):
        raise ValueError('Cached B0 or identity baseline shape differs')
    p, y, b, identity = [value.detach().cpu().double() for value in (prediction, target, base, identity)]
    mask = mask.cpu()
    error = torch.where(mask, p - y, 0)
    residual_error = torch.where(mask, (p - b - identity[:, None]) - (y - b - identity[:, None]), 0)
    if not torch.isfinite(error).all() or not torch.isfinite(residual_error).all():
        raise ValueError('Nonfinite observed motion error')
    # Subtraction uses double to avoid importing avoidable float32 cancellation
    # noise. The baseline is fixed and must cancel even when B0 varies in time.
    torch.testing.assert_close(error, residual_error, rtol=2e-12, atol=2e-12)
    error, mask = error[..., channels], mask[..., channels]
    counts = mask.sum(1).double()
    mean = error.sum(1) / counts.clamp_min(1)
    centered = torch.where(mask, error - mean[:, None], 0)
    raw_sse = error.square().sum((1, 2))
    dynamic_sse = centered.square().sum((1, 2))
    mean_sse = (counts * mean.square()).sum(1)
    torch.testing.assert_close(raw_sse, dynamic_sse + mean_sse, rtol=2e-12, atol=2e-12)
    return torch.stack((raw_sse, dynamic_sse, mean_sse, counts.sum(1)), -1).numpy()


def average_error_components(rows):
    if not rows or any(row.shape != rows[0].shape or not np.array_equal(row[:, 3], rows[0][:, 3]) for row in rows):
        raise ValueError('Noise conditions have different observed counts')
    mean = np.stack(rows).mean(0)
    mean[:, 3] = rows[0][:, 3]
    if not np.allclose(mean[:, 0], mean[:, 1] + mean[:, 2], rtol=2e-12, atol=2e-12):
        raise ValueError('Mean noise decomposition identity failed')
    return mean


def component_summary(rows):
    total = rows.sum(0)
    if total[3] <= 0:
        return None
    values = total[:3] / total[3]
    return {'raw_mse': float(values[0]), 'centered_error_mse': float(values[1]), 'mean_error_mse': float(values[2]),
            'observed_values': float(total[3]), 'decomposition_abs_error': float(abs(values[0] - values[1] - values[2]))}


def paired_component_change(candidate, baseline, sentence_ids, indices, *, samples=5000):
    ids = np.asarray(indices, dtype=np.int64)
    if not np.array_equal(candidate[ids, 3], baseline[ids, 3]):
        raise ValueError('Decomposition comparison requires identical observations')
    count = candidate[ids, 3].sum()
    changes = candidate[ids, :3] - baseline[ids, :3]
    grouped = {}
    for i, original_id in enumerate(ids):
        row = grouped.setdefault(str(sentence_ids[original_id]), np.zeros(4))
        row += np.r_[changes[i], candidate[original_id, 3]]
    delta = changes.sum(0) / count
    ci = None
    if len(grouped) > 1 and samples > 0:
        groups = np.stack(list(grouped.values()))
        draws = np.random.default_rng(45).integers(len(groups), size=(samples, len(groups)))
        totals = groups[draws].sum(1)
        valid = totals[:, 3] > 0
        ci = np.quantile(totals[valid, :3] / totals[valid, 3, None], [.025, .975], axis=0)
    names = ('raw_mse', 'centered_error_mse', 'mean_error_mse')
    return {'candidate': component_summary(candidate[ids]), 'baseline': component_summary(baseline[ids]),
        'delta_candidate_minus_baseline': {name: float(value) for name, value in zip(names, delta)},
        'delta_ci95': {name: ci[:, i].tolist() if ci is not None else None for i, name in enumerate(names)},
        'raw_delta_sum_error': float(abs(delta[0] - delta[1] - delta[2])),
        'mean_improves_while_dynamic_worsens': bool(delta[2] < 0 and delta[1] > 0),
        'raw_improvement_hides_dynamic_worsening': bool(delta[0] < 0 and delta[1] > 0),
        'clips': len(ids), 'sentences': len(grouped), 'sign': 'Negative delta means improvement; all deltas share the same observed-value denominator'}


def analyze_saved_curves(arms, reference, *, samples=5000):
    q = reference['q']; populations = audit_populations(q)
    mask = observed(q)
    stats, per_noise = {}, {}
    for arm, values in arms.items():
        per_noise[arm] = {}
        for epoch in EPOCHS:
            curves = values['curves'][epoch]
            validate_curves(curves, reference)
            per_noise[arm][str(epoch)] = {}
            for mode in MODES:
                rows = {group: [] for group in GROUPS}
                for seed in SEEDS:
                    prediction = curves['motion'][str(seed)][mode]
                    for group, channels in GROUPS.items():
                        row = exact_error_components(prediction, q['motion'], mask, channels,
                            reference['base']['b0'], reference['identity']['baseline'])
                        rows[group].append(row)
                        per_noise[arm][str(epoch)].setdefault(str(seed), {}).setdefault(mode, {})[group] = {
                            pop: component_summary(row[ids]) for pop, ids in populations.items() if pop in ('all', 'neutral', 'nonneutral')}
                for group, records in rows.items():
                    stats[(arm, epoch, mode, group)] = average_error_components(records)
    summaries = {arm: {str(epoch): {mode: {group: {pop: component_summary(stats[(arm, epoch, mode, group)][ids])
        for pop, ids in populations.items()} for group in GROUPS} for mode in MODES} for epoch in EPOCHS} for arm in arms}
    full_zero = {arm: {str(epoch): {group: {pop: paired_component_change(stats[(arm, epoch, 'full', group)],
        stats[(arm, epoch, 'zero', group)], q['sentence_id'], ids, samples=samples)
        for pop, ids in populations.items()} for group in GROUPS} for epoch in EPOCHS} for arm in arms}
    epoch_change = {arm: {mode: {group: {pop: paired_component_change(stats[(arm, 18, mode, group)],
        stats[(arm, 2, mode, group)], q['sentence_id'], ids, samples=samples)
        for pop, ids in populations.items()} for group in GROUPS} for mode in MODES} for arm in arms}
    return {'scores': summaries, 'full_minus_zero': full_zero, 'epoch18_minus_epoch2': epoch_change,
        'per_noise_pooled': per_noise, 'population_counts': {pop: {'clips': len(ids),
            'sentences': len({q['sentence_id'][i] for i in ids}), 'people': len({int(q['speaker_id'][i]) for i in ids})}
            for pop, ids in populations.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument('--' + arm.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--rollout', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        raise ValueError('Require fresh output and positive bootstrap count')
    torch.set_num_threads(4)
    arms = {arm: load_arm(getattr(args, arm), arm) for arm in ARMS}
    validate_rng_evidence({arm: value['records'] for arm, value in arms.items()})
    reference, lock, scope, frozen = validate_training_inputs(arms)
    if len(reference['q']['clip_id']) != 405:
        raise ValueError('This diagnosis requires the locked internal405heldout set')
    if args.rollout is not None:
        rollout = load_rollout(args.rollout)
        validate_recipe_pair(rollout['recipe'], arms['constant_teacher']['recipe'])
        validate_matched_rng(rollout['records'], arms['constant_teacher']['records'])
        for epoch in EPOCHS:
            ck = rollout['checkpoints'][epoch]
            if ck['head_sha256'] != frozen['fixed_head'] or ck['frozen_state_sha256'] != frozen['frozen_backbone']:
                raise ValueError('Optional rollout changed fixed head/backbone')
            for seed in SEEDS:
                for mode in MODES:
                    assert_metric_agreement(basic_metrics(rollout['curves'][epoch]['motion'][str(seed)][mode], reference),
                        rollout['reports'][epoch][str(seed)][mode])
        arms['rollout'] = rollout
    results = analyze_saved_curves(arms, reference, samples=args.samples)
    report = {'schema': 'projection_mean_dynamic_tradeoff_v1', 'source_sha256': sha(__file__),
        'arm_hashes': {arm: value['hashes'] for arm, value in arms.items()}, 'frozen_hashes': frozen,
        'split_lock': lock, 'data_scope': scope, 'epochs': list(EPOCHS), 'noise_seeds': SEEDS,
        'bootstrap_samples': args.samples, 'bootstrap_seed': 45, 'frame_clock': 'Original observed frame clock; no resampling, U projection, or fitted normalization',
        'identity_and_B0_cancel_verified': True, 'per_clip_channel_decomposition_verified': True,
        'equation': 'SSE(pred-target) = SSE(error-mean_t(error)) + sum_clip_channel(n_observed * mean_t(error)^2)',
        'aggregation': 'Means use each clip/channel actual observed count. Sum region sufficient errors and divide by total observed values. Average noise errors, never prediction curves.',
        'interpretation': 'Negative raw-error change with positive centered-error change means mean correction outweighs temporal degradation. This diagnoses where error changed; it does not establish the optimizer mechanism or prove identity is its unique cause. A final-motion objective can also trade these terms because it contains both.',
        'scope': 'Training-internal405development recordings of3people, inherited frozen backbone/global exposure possible. Read-only saved outputs, no new fitting, training, generation, checkpoint selection, or outer/test reads.',
        'outer280_loaded': False, 'new_identity439_loaded': False, 'test_loaded': False, 'default_replaced': False, **results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for arm in arms:
        print(json.dumps({'arm': arm, 'epoch': 18, 'nonneutral_upper_full_minus_zero':
            results['full_minus_zero'][arm]['18']['upper_expression']['nonneutral']}), flush=True)


if __name__ == '__main__':
    main()
