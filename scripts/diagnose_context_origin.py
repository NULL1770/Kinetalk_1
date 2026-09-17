"""Score a fixed audio-mean composition without changing temporal dynamics.

Uses the old state_white deployment means only after source metadata and file
bindings match. GT-mean substitution is explicitly an evaluation oracle.
"""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import read, sha, load_pt, metadata_equal, summarize, GROUPS
from scripts.train_formal_predictable_projection import save_json, save_checkpoint
from kinetalk_b0.models.mean_preserving_upper import compose_mean_preserving_upper
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-run', type=Path, required=True)
    parser.add_argument('--state-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    torch.set_num_threads(4)
    current = args.context_run/'chunk_teacher/curves.pt'
    old = args.state_run/'state_white_curves.pt'
    if sha(current) != read(current.with_name('complete.json'))['curves_sha256']:
        raise ValueError('Current curves hash mismatch')
    if sha(old) != read(args.state_run/'complete.json')['state_white']:
        raise ValueError('State curves hash mismatch')
    a, b = load_pt(current), load_pt(old)
    metadata_equal(a, b)
    cc = list(UPPER_INDICES)
    valid = a['valid']
    old_values = b['predictions']['42/full'][..., cc]
    static = torch.where(valid[..., None], old_values, 0.).sum(1)/valid.sum(1)[:, None]
    max_old_mean_difference = 0.
    for seed in (123, 2026):
        values = b['predictions'][f'{seed}/full'][..., cc]
        mean = torch.where(valid[..., None], values, 0.).sum(1)/valid.sum(1)[:, None]
        max_old_mean_difference = max(max_old_mean_difference, float((mean-static).abs().max()))
    if max_old_mean_difference > 2e-6:
        raise ValueError('Old state means are stochastic or inconsistent')
    truth_mean = torch.where(valid[..., None], a['target'][..., cc], 0.).sum(1)/valid.sum(1)[:, None]
    outputs = {'raw': {}, 'audio_mean': {}, 'ORACLE_GT_mean': {}}
    for seed in (42, 123, 2026):
        key = f'{seed}/full'
        pred = a['predictions'][key]
        outputs['raw'][key] = pred
        for name, mean in (('audio_mean', static), ('ORACLE_GT_mean', truth_mean)):
            base = pred.clone()
            base[..., cc] = mean[:, None]
            # Invalid frames retain the original complete baseline, not means.
            base[~valid] = pred[~valid]
            outputs[name][key] = compose_mean_preserving_upper(base, pred[..., cc], valid)
    report = {'scope': '405 repeatedly-used development clips; fixed checkpoint, no fitting',
              'source_sha256': {'context': sha(current), 'state_white': sha(old), 'script': sha(__file__)},
              'state_mean_max_across_seed_difference': max_old_mean_difference,
              'audio_mean_source': 'Mean of hash-bound old deployable state_white, independent anchor plus frozen audio state',
              'test_loaded': False, 'new_training': False, 'results': {}}
    for name, values in outputs.items():
        result = summarize({**a, 'predictions': values}, a['emotion_id'])
        report['results'][name] = result
    invariants = {}
    for name in ('brows', 'eyes_expression'):
        x = report['results']['raw']['populations']['all']['groups'][name]['mean_over_three_seeds']
        y = report['results']['audio_mean']['populations']['all']['groups'][name]['mean_over_three_seeds']
        invariant_keys = ('centered_mse', 'centered_correlation', 'rms_ratio', 'frame_displacement_mse')
        invariants[name] = {key: abs(x[key]-y[key]) for key in invariant_keys}
        if any(value > 2e-6 for value in invariants[name].values()):
            raise RuntimeError('Composition changed dynamics')
    report['temporal_invariant_abs_differences'] = invariants
    report['limitations'] = ['Offline whole-window translation; not a causal initialization repair.',
        'Dynamics, ordering, speed and seams unchanged; this cannot establish audio timing success.',
        'Oracle GT means are scoring-only, never a deployment path or generated-history condition.']
    args.output.mkdir(parents=True)
    save_json(args.output/'report.json', report)
    save_checkpoint(args.output/'audio_mean_curves.pt', {**a, 'predictions': outputs['audio_mean'],
        'oracle_prediction_keys': [], 'oracle_has_target_input': False, 'composition': 'offline_audio_mean'})
    save_json(args.output/'complete.json', {'curves_sha256': sha(args.output/'audio_mean_curves.pt'), 'report_sha256': sha(args.output/'report.json')})
    for name in GROUPS:
        print(name, {mode: report['results'][mode]['populations']['all']['groups'][name]['mean_over_three_seeds']['raw_mse'] for mode in outputs}, flush=True)


if __name__ == '__main__':
    main()
