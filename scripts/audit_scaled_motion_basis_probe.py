"""Recompute raw curves, fold coverage, coordinates and OOF paired scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary
from kinetalk_b0.predictable_motion import weighted_clip_center


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    p = args.study
    if (p / 'independent_raw_audit.json').exists():
        raise FileExistsError('A fresh independent audit output is required')
    torch.set_num_threads(4)
    binding = json.loads((p / 'output_hashes.json').read_text())
    for name, digest in binding.items():
        assert sha(p / name) == digest, name
    report = json.loads((p / 'paired_audit.json').read_text())
    for name, digest in report['source_sha256'].items():
        assert sha(name) == digest, name
    curves = torch.load(p / 'oof_predictions.pt', weights_only=False, map_location='cpu')
    states = torch.load(p / 'fold_states.pt', weights_only=False, map_location='cpu')['states']
    channels = report['motion_channel_indices']
    groups = {name: [channels.index(c) for c in cc] for name, cc in report['groups'].items()}
    w, target = curves['weight'], curves['target']
    torch.testing.assert_close(weighted_clip_center(target, w), target, rtol=1e-12, atol=1e-12)
    n = len(target)
    assert n == 2315 and len(set(curves['clip_id'])) == n
    for arm in ('native', 'scaled'):
        seen = torch.zeros(n, dtype=torch.long)
        for fold in range(3):
            state = states[f'fold{fold}_{arm}']
            tr, val = state['fit_ids'], state['validation_ids']
            assert sorted(tr.tolist() + val.tolist()) == list(range(n))
            assert not ({curves['sentence_id'][i] for i in tr.tolist()} & {curves['sentence_id'][i] for i in val.tolist()})
            assert torch.all(curves['oof_fold'][val] == fold)
            seen[val] += 1
            model, scale = state['transformed_model'], state['metric']['sqrt_metric']
            assert model['alpha'] == 1.0 and model['rank'] == 8
            torch.testing.assert_close(state['inverse_native_basis'], model['basis'] / scale[:, None])
            torch.testing.assert_close(state['teacher_analysis_basis'], model['basis'] * scale[:, None])
            oracle = (target[val] @ state['teacher_analysis_basis']) @ state['inverse_native_basis'].T
            torch.testing.assert_close(oracle, curves['predictions'][arm]['metric_projection_oracle'][val], rtol=1e-9, atol=1e-11)
        assert torch.all(seen == 1)
        for mode, prediction in curves['predictions'][arm].items():
            torch.testing.assert_close(weighted_clip_center(prediction, w), prediction, rtol=1e-9, atol=1e-11)
            for group, ids in groups.items():
                recomputed = clip_statistics(prediction, target, w, ids)
                np.testing.assert_array_equal(recomputed, curves['statistics'][arm][mode][group])
    result = {'source_output_hashes': binding, 'raw_statistics_recomputed': True,
              'fold_coverage_and_sentence_isolation': True, 'oracle_coordinate_roundtrip': True,
              'comparisons': {}}
    for name, value in report['comparisons'].items():
        left, right = value['candidate'], value['baseline']
        result['comparisons'][name] = {}
        for pop in ('all', 'nonneutral', 'neutral'):
            idx = [i for i, emotion in enumerate(curves['emotion_id'].tolist())
                   if pop == 'all' or (emotion != 0) == (pop == 'nonneutral')]
            result['comparisons'][name][pop] = {}
            for group in groups:
                s = paired_summary(curves['statistics'][left[0]][left[1]][group],
                                   curves['statistics'][right[0]][right[1]][group],
                                   curves['sentence_id'], idx, samples=report['bootstrap_samples'],
                                   seed=report['bootstrap_seed'])
                assert s == value['populations'][pop][group]
                result['comparisons'][name][pop][group] = {
                    'candidate_r2': s['candidate']['r2_against_zero'],
                    'baseline_r2': s['baseline']['r2_against_zero'],
                    'delta_r2': s['r2_improvement'], 'ci95': s['r2_improvement_ci95'],
                    'rms_ratio': s['candidate']['prediction_rms_amplitude_ratio'],
                    'correlation': s['candidate']['pooled_centered_correlation']}
    result['by_speaker_nonneutral'] = {}
    for speaker in sorted(set(curves['speaker_id'])):
        result['by_speaker_nonneutral'][speaker] = {}
        for group in ('brows', 'eyes_expression', 'mouth'):
            v = report['comparisons']['scaled__vs__native_full']['by_speaker'][speaker]['nonneutral'][group]
            result['by_speaker_nonneutral'][speaker][group] = {
                'native_r2': v['baseline']['r2_against_zero'],
                'scaled_r2': v['candidate']['r2_against_zero'],
                'delta_r2': v['r2_improvement'], 'ci95': v['r2_improvement_ci95']}
    (p / 'independent_raw_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({name: result['comparisons'][name]['nonneutral'] for name in
        ('native_full__vs__zero', 'scaled_full__vs__zero', 'scaled__vs__native_full',
         'native_metric_projection_oracle__vs__zero', 'scaled_metric_projection_oracle__vs__zero')}, indent=2))


if __name__ == '__main__':
    main()
