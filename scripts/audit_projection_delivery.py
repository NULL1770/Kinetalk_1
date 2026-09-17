"""Compare audio/teacher targets and delivered generated motion on ONE clock.

Generated residuals are binned/centered on the original four-frame clock,
without projecting them through U. Thus generator-added directions remain
visible. All methods are evaluated against the same real native motion bins.
This is read-only diagnosis of the selected seed46 model, not a new method.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.predictable_motion import bin_centered_frames
from scripts.audit_audio_activity_gate import SEEDS, write_json
from scripts.audit_cross_identity_projection import groups_of_clips, load_verified
from scripts.audit_formal_projection import validate_artifacts
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.evaluate_cross_identity_projection import RestoredFixedAudioHead
from scripts.train_predictable_renderer import audio_features, center, observed, sha

GROUPS = {'upper_expression': [5, 6, 12, 13, 41, 42, 43, 44, 45],
          'brows': list(range(41, 46)), 'eyes_expression': [5, 6, 12, 13]}


def mean_noise_statistics(rows, reference):
    """Average stochastic errors while keeping shared targets bit-identical.

    Target energy/count/centered energy do not depend on prediction or noise.
    Validate them before aggregation, then preserve their original values so
    deterministic and stochastic conditions remain exactly paired.
    """
    if not rows:
        raise ValueError('Require noise statistics')
    target_columns = [1, 3, 6]
    for row in rows:
        if row.shape != reference.shape or not np.array_equal(row[:, target_columns], reference[:, target_columns]):
            raise ValueError('Noise conditions have different target statistics or weights')
    averaged = np.stack(rows).mean(0)
    averaged[:, target_columns] = reference[:, target_columns]
    return averaged


def residual_bins(prediction, split, bundle, stride=4):
    q = split['q']
    if prediction.shape != q['motion'].shape:
        raise ValueError('Generated/reference shape mismatch')
    residual = torch.where(observed(q), prediction - split['base']['b0'] -
                           split['identity']['baseline'][:, None], 0)
    bins, weights = bin_centered_frames(residual, q['valid'], stride)
    if bins.shape != bundle['motion_bins'].shape or not torch.equal(weights, bundle['weight'].double()):
        raise ValueError('Generated bin clock/valid counts differ from native target bundle')
    return bins


@torch.no_grad()
def delivery_diagnostic(split, bundle, curves, saved_head, classes, speaker_to_id, *, samples=5000):
    head = RestoredFixedAudioHead(saved_head).cpu().eval()
    q, w = split['q'], bundle['weight'].float()
    rebuilt_target = residual_bins(q['motion'], split, bundle)
    # Nuisance channels were cleared by bundle preparation; the audited
    # expression channels must reproduce the original real-motion targets.
    channels = sorted({c for group in GROUPS.values() for c in group})
    target = bundle['motion_bins'].double()
    torch.testing.assert_close(rebuilt_target[..., channels], target[..., channels], rtol=2e-5, atol=2e-6)
    target = center(target, w.double())
    gate = 1 - split['affect']['emotion_logits'].detach().float().softmax(-1)[:, classes.index('neutral')]
    oracle_z = head.teacher(bundle['motion_bins'].float(), w)
    audio_z = head(audio_features(bundle), w)
    def decode(z):
        out = torch.zeros_like(target)
        out[..., head.channels] = ((z * head.target_scale) @ head.basis.T).double()
        return out
    deterministic = {'audio_gated_native': decode(audio_z * gate[:, None, None]),
                     'motion_projection_oracle_gated': decode(oracle_z * gate[:, None, None]),
                     'zero_field': torch.zeros_like(target)}
    stats, per_noise, noise_rows = {}, {}, {}
    for mode, value in deterministic.items():
        for group, indices in GROUPS.items():
            stats[(mode, group)] = clip_statistics(value, target, w, indices)
    for seed in SEEDS:
        per_noise[str(seed)] = {}
        for mode in ('full', 'zero', 'oracle'):
            # Deliberately no @U U.T on generated bins: that would hide
            # generator drift outside the predictable motion subspace.
            prediction = residual_bins(curves['motion'][str(seed)][mode], split, bundle)
            name = 'generated_' + mode
            per_noise[str(seed)][name] = {}
            for group, indices in GROUPS.items():
                row = clip_statistics(prediction, target, w, indices)
                key = name, group
                noise_rows.setdefault(key, []).append(row)
                per_noise[str(seed)][name][group] = scalar_summary(row)
    for (mode, group), rows in noise_rows.items():
        stats[(mode, group)] = mean_noise_statistics(rows, stats[('zero_field', group)])
    populations = groups_of_clips(q, speaker_to_id, classes)
    populations = {key: ids for key, ids in populations.items() if key.startswith(('pooled/', 'speaker/'))}
    modes = tuple(deterministic) + ('generated_full', 'generated_zero', 'generated_oracle')
    scores = {pop: {mode: {group: scalar_summary(stats[(mode, group)][ids]) for group in GROUPS}
                   for mode in modes} for pop, ids in populations.items()}
    specs = [('generated_full', 'generated_zero'), ('generated_oracle', 'generated_zero'),
             ('generated_oracle', 'motion_projection_oracle_gated'), ('generated_full', 'audio_gated_native'),
             ('motion_projection_oracle_gated', 'zero_field'), ('audio_gated_native', 'zero_field')]
    comparisons = {pop: {candidate + '__vs__' + base: {group: paired_summary(stats[(candidate, group)],
        stats[(base, group)], q['sentence_id'], ids, samples=samples, seed=45) for group in GROUPS}
        for candidate, base in specs} for pop, ids in populations.items()}
    return {'scores': scores, 'paired': comparisons, 'per_noise_pooled_all': per_noise,
        'population_counts': {pop: {'clips': len(ids), 'sentences': len({q['sentence_id'][i] for i in ids})}
                              for pop, ids in populations.items()},
        'clock': {'stride_frames': 4, 'bins': int(target.shape[1]), 'target_channels': int(target.shape[2]),
                  'weight': 'actual valid-frame counts per bin'},
        'generated_projection': 'No U projection applied to generated residuals; bin/center only',
        'target': 'Same real center(motion - cached B0 - identity baseline) native bins for all rows',
        'noise_aggregation': 'Average generated per-clip squared errors over three noises; deterministic direct rows entered once'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('formal-run', 'cross-evaluation', 'cross-data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        raise ValueError('Require a fresh output and positive bootstrap count')
    torch.set_num_threads(4)
    cross_args = argparse.Namespace(evaluation=args.cross_evaluation, cross_data=args.cross_data,
        formal_run=args.formal_run, selected_checkpoint=None, formal_bundle=None, checkpoint=None, config=None)
    cross_summary, payload, cross_curves, classes, cross_recipe, cross_hashes, saved_head = load_verified(cross_args)
    formal_summary, recipe, reference, formal_curves, _, formal_hashes, binding = validate_artifacts(args.formal_run)
    if recipe != cross_recipe or int(recipe['args']['seed']) != 46:
        raise ValueError('This fixed delivery diagnosis requires the identical selected seed46 recipe')
    if formal_hashes['selected_checkpoint'] != cross_hashes['selected_checkpoint']:
        raise ValueError('Old and new identity evaluation used different selected checkpoints')
    bundle = torch.load(recipe['args']['bundle'], map_location='cpu', weights_only=False, mmap=True)['bundles']['external_dev']
    if reference['q']['clip_id'] != bundle['clip_id']:
        raise ValueError('Original development cache/bundle order differs')
    # Formal split retains actual speaker strings; fall back only to the
    # recorded numeric IDs when a cache format omitted readable names.
    query = reference['q']
    names = query.get('speaker')
    if names is None:
        old_mapping = {str(int(sid)): int(sid) for sid in query['speaker_id'].unique()}
    else:
        old_mapping = {str(name): int(sid) for name, sid in zip(names, query['speaker_id'])}
    old = delivery_diagnostic(reference, bundle, formal_curves, saved_head, classes, old_mapping, samples=args.samples)
    new = delivery_diagnostic(payload['split'], payload['bundle'], cross_curves['selected'], saved_head,
        classes, payload['provenance']['speaker_to_id'], samples=args.samples)
    report = {'schema': 'same_clock_projection_delivery_audit_v1', 'training_seed': 46,
        'selected_epoch': formal_summary['best_epoch'], 'formal_hashes': formal_hashes, 'cross_hashes': cross_hashes,
        'script_sha256': sha(__file__), 'curve_binding': binding, 'samples': args.samples, 'bootstrap_seed': 45,
        'old_development': old, 'new_identity_development': new,
        'interpretation': 'Audio prediction and delivered generator dynamics now share the same target, channels, bin clock and weighted zero baseline. A strong gated motion-projection oracle with weaker generated oracle identifies loss through the frozen generator/shared conditioning path; it does not prove identity is the sole cause. Generator noise, conditioning distribution and data quality may also contribute.',
        'scope': 'Old development selected the checkpoint; new-identity development has only three identities and shared scripts. Both are diagnostics, not independent broad generalization evidence. No fitting, tuning, new generation, test reads or default replacement.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for name, data in (('old_development', old), ('new_identity_development', new)):
        print(json.dumps({'population': name, 'nonneutral_r2': {mode: {group: value['r2_against_zero']
            for group, value in groups.items()} for mode, groups in data['scores']['pooled/nonneutral'].items()}}), flush=True)


if __name__ == '__main__':
    main()
