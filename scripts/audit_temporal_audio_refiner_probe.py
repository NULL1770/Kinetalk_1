"""Verify saved eight-epoch audio OOF curves and prespecified protection checks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.predictable_motion import weighted_clip_center
from kinetalk_b0.temporal_motion_refiner import FrozenRidgeRefiner, fit_control_scale
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.probe_scaled_motion_basis import write_json
from scripts.probe_predictable_motion import intervene_input
from scripts.train_predictable_renderer import audio_features, sha, state_hash
from scripts.train_temporal_audio_refiner_probe import load_sources, predict_batches, buffer_hash


def relative_error_protection(candidate, baseline, sentences, ids, samples=5000):
    """One-sided90 upper bound for relative pooled native MSE increase."""
    a, b = candidate[ids], baseline[ids]
    if not np.array_equal(a[:, 1], b[:, 1]) or not np.array_equal(a[:, 3], b[:, 3]):
        raise ValueError('Unpaired protection targets/counts')
    clusters = {}
    for i, index in enumerate(ids):
        clusters.setdefault(str(sentences[index]), np.zeros(2))[:] += [a[i, 0], b[i, 0]]
    totals = np.stack(list(clusters.values()))
    picked = np.random.default_rng(45).integers(len(totals), size=(samples, len(totals)))
    sums = totals[picked].sum(1)
    valid = sums[:, 1] > 0
    ratio = sums[valid, 0] / sums[valid, 1] - 1
    summary_a, summary_b = scalar_summary(a), scalar_summary(b)
    corr = summary_a['pooled_centered_correlation'] - summary_b['pooled_centered_correlation']
    return {'relative_native_mse_increase': float(a[:, 0].sum() / b[:, 0].sum() - 1),
        'relative_native_mse_increase_upper90': float(np.quantile(ratio, .9)),
        'correlation_delta': corr, 'mse_protection_pass': bool(np.quantile(ratio, .9) <= .01),
        'mouth_correlation_protection_pass': bool(corr >= -.005),
        'bootstrap_valid_samples': int(valid.sum()), 'samples': samples}


def derivative_statistics(prediction, target, weight, channels):
    """Native bin differences, not a claim of physical velocity at partial bins."""
    adjacent = (weight[:, 1:] > 0) & (weight[:, :-1] > 0)
    w = torch.where(adjacent, torch.minimum(weight[:, 1:], weight[:, :-1]), 0)
    return clip_statistics(prediction[:, 1:] - prediction[:, :-1],
                           target[:, 1:] - target[:, :-1], w, channels)


def make_report(curves, *, samples=5000):
    target, w = curves['target'], curves['weight']
    channels = curves['provenance']['motion_channel_indices']
    groups = {name: [channels.index(c) for c in cc] for name, cc in curves['groups'].items()}
    stats, diffs = {}, {}
    for arm, modes in curves['predictions'].items():
        stats[arm], diffs[arm] = {}, {}
        for mode, pred in modes.items():
            torch.testing.assert_close(pred, weighted_clip_center(pred, w), rtol=1e-8, atol=1e-10)
            stats[arm][mode] = {}
            diffs[arm][mode] = {}
            for group, cc in groups.items():
                stats[arm][mode][group] = clip_statistics(pred, target, w, cc)
                np.testing.assert_array_equal(stats[arm][mode][group], curves['statistics'][arm][mode][group])
                diffs[arm][mode][group] = derivative_statistics(pred, target, w, cc)
    emotions = curves['emotion_id'].tolist()
    populations = {'all': list(range(len(target))), 'nonneutral': [i for i, e in enumerate(emotions) if e != 0],
                   'neutral': [i for i, e in enumerate(emotions) if e == 0]}
    comparisons = []
    for arm in ('ridge', 'pointwise', 'temporal'):
        comparisons += [(f'{arm}__vs__zero', (arm, 'full'), ('ridge', 'zero')),
                        (f'{arm}__vs__reverse', (arm, 'full'), (arm, 'reverse'))]
        if arm != 'ridge':
            comparisons.append((f'{arm}__vs__ridge', (arm, 'full'), ('ridge', 'full')))
    comparisons += [('temporal__vs__pointwise', ('temporal', 'full'), ('pointwise', 'full')),
                    ('oracle__vs__zero', ('ridge', 'metric_projection_oracle'), ('ridge', 'zero'))]
    results, protections = {}, {}
    sentences, speakers = curves['sentence_id'], curves['speaker_id']
    for name, left, right in comparisons:
        a, b = stats[left[0]][left[1]], stats[right[0]][right[1]]
        def compare(ids, source_a=a, source_b=b, draws=samples):
            return {g: paired_summary(source_a[g], source_b[g], sentences, ids, samples=draws, seed=45) for g in groups}
        results[name] = {'candidate': left, 'baseline': right,
            'populations': {p: compare(ids) for p, ids in populations.items()},
            'by_speaker_nonneutral': {s: compare([i for i in populations['nonneutral'] if speakers[i] == s])
                                      for s in sorted(set(speakers))},
            'bin_difference_comparison': {p: compare(ids, diffs[left[0]][left[1]], diffs[right[0]][right[1]])
                                           for p, ids in populations.items()}}
    for arm in ('pointwise', 'temporal'):
        protections[arm] = {p: {g: relative_error_protection(stats[arm]['full'][g],
            stats['ridge']['full'][g], sentences, ids, samples) for g in ('mouth', 'eyes_expression')}
            for p, ids in populations.items() if p != 'all'}
    temporal = results['temporal__vs__ridge']
    eyebrow = temporal['populations']['nonneutral']['brows']
    positive_people = sum(r['brows']['r2_improvement'] > 0 for r in temporal['by_speaker_nonneutral'].values())
    gates = {'brow_improvement_ci_positive': eyebrow['r2_improvement_ci95'][0] > 0,
        'brow_correlation_improved': eyebrow['candidate']['pooled_centered_correlation'] > eyebrow['baseline']['pooled_centered_correlation'],
        'brow_beats_zero_and_reverse': all(results['temporal__vs__' + m]['populations']['nonneutral']['brows']['r2_improvement_ci95'][0] > 0 for m in ('zero', 'reverse')),
        'brow_at_least_ten_people_improve': positive_people >= 10,
        'eyes_mouth_mse_protection': all(v['mse_protection_pass'] for groups_ in protections['temporal'].values() for v in groups_.values()),
        'mouth_correlation_protection': all(groups_['mouth']['mouth_correlation_protection_pass'] for groups_ in protections['temporal'].values())}
    return {'comparisons': results, 'protection': protections, 'generation_entry_checks': gates,
        'generation_entry_pass': all(gates.values()), 'positive_brow_people': positive_people,
        'temporal_brow_advantage_vs_pointwise_ci_positive': results['temporal__vs__pointwise']['populations']['nonneutral']['brows']['r2_improvement_ci95'][0] > 0,
        'derivative_note': 'Difference of adjacent valid bins weighted by minimum bin count; not physical-time velocity for partial bins',
        'scope': 'Conditional sentence bootstrap on saved existing-training OOF predictions; no refit uncertainty, no unseen-identity claim; exploratory per-person intervals.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = args.study / 'paired_audit.json'
    if out.exists():
        raise FileExistsError('Fresh audit required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    hashes = json.loads((args.study / 'output_hashes.json').read_text())
    for name, digest in hashes.items():
        if sha(args.study / name) != digest:
            raise ValueError('Output artifact changed: ' + name)
    curves = torch.load(args.study / 'oof_predictions.pt', map_location='cpu', weights_only=False, mmap=True)
    recipe = curves['provenance']
    for name, digest in recipe['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Training source changed: ' + name)
    source_args = recipe['args']
    bundle, channels, speakers, folds, saved, previous, input_hashes, binding = load_sources(
        Path(source_args['bundle']), Path(source_args['weights']), Path(source_args['basis_study']))
    if input_hashes != recipe['input_sha256'] or binding != recipe['prior_oof_output_sha256']:
        raise ValueError('Source binding differs')
    for key in ('clip_id', 'sentence_id', 'speaker_id'):
        if curves[key] != previous[key]:
            raise ValueError('OOF metadata differs: ' + key)
    for key in ('target', 'weight', 'emotion_id', 'source_speaker_id', 'oof_fold'):
        if not torch.equal(curves[key], previous[key]):
            raise ValueError('OOF target/clock differs: ' + key)
    for mode in curves['predictions']['ridge']:
        if not torch.equal(curves['predictions']['ridge'][mode], previous['predictions']['native'][mode]):
            raise ValueError('Ridge reference changed')
    x, y, w = audio_features(bundle), bundle['motion_bins'][..., channels], bundle['weight']
    checkpoint_reproduction = {}
    for fold, (fit, val) in enumerate(folds):
        state = saved['states'][f'fold{fold}_native']['transformed_model']
        scale = fit_control_scale(y, w, fit, state)
        expected_hash = None
        for arm in ('pointwise', 'temporal'):
            ck = torch.load(args.study / f'fold{fold}_{arm}/last.pt', map_location='cpu', weights_only=False)
            if ck['provenance'] != recipe or ck['epoch'] != 8 or ck['step'] != 8 * ((len(fit) + 31) // 32):
                raise ValueError('Incomplete/wrong fixed-budget checkpoint')
            if not torch.equal(ck['fit_ids'], fit):
                raise ValueError('Checkpoint fit differs')
            if expected_hash is not None and ck['batch_order_sha256'] != expected_hash:
                raise ValueError('Arm training batch mismatch')
            expected_hash = ck['batch_order_sha256']
            order_gen = torch.Generator().manual_seed(46 + fold)
            batch_digest = hashlib.sha256()
            for epoch in range(8):
                order = torch.randperm(len(fit), generator=order_gen)
                batch_digest.update(fit[order].numpy().tobytes())
                if ck['history'][epoch]['batch_order_sha256'] != batch_digest.hexdigest():
                    raise ValueError('Epoch batch hash mismatch')
            model = FrozenRidgeRefiner(state, scale, arm)
            before = buffer_hash(model)
            model.load_state_dict(ck['model'], strict=True)
            if buffer_hash(model) != before or before != ck['frozen_buffer_sha256']:
                raise ValueError('Frozen model changed')
            model.to(args.device)
            errors = {}
            for mode, features in [('full', x[val]), ('reverse', intervene_input(x[val], w[val], 'reverse'))]:
                prediction = predict_batches(model, features, w[val], args.device)
                reference = curves['predictions'][arm][mode][val]
                torch.testing.assert_close(prediction, reference, rtol=1e-7, atol=1e-9)
                errors[mode] = float((prediction - reference).abs().max())
            checkpoint_reproduction[f'fold{fold}_{arm}'] = errors
    report = {'schema': 'temporal_audio_refiner_audit_v1', 'input_output_sha256': hashes,
        'script_sha256': sha(__file__), 'provenance': recipe,
        'checkpoint_reproduction_max_abs_error': checkpoint_reproduction,
        **make_report(curves)}
    write_json(out, report)
    print(json.dumps({'generation_entry_checks': report['generation_entry_checks'],
        'nonneutral_temporal_vs_ridge': {g: report['comparisons']['temporal__vs__ridge']['populations']['nonneutral'][g]['r2_improvement']
                                       for g in ('brows', 'eyes_expression', 'mouth')},
        'positive_brow_people': report['positive_brow_people']}), flush=True)


if __name__ == '__main__':
    main()
