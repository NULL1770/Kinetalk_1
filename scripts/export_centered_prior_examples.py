"""Export the historical nine clips from both centered-prior final runs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import read, sha, metadata_equal, summarize, tensor_state_equal
from scripts.export_full_staged_examples import ARKIT_NAMES
from scripts.train_formal_predictable_projection import save_json, canonical_hash


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'old-run', 'previous-visual', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Fresh export required')
    curves, reports, bindings, finals, recipes = {}, {}, {}, {}, {}
    for arm in ('white', 'ar1'):
        path = a.root/arm
        complete = read(path/'complete.json')
        recipes[arm] = read(path/'provenance.json')['recipe']
        if complete['recipe_sha256'] != canonical_hash(recipes[arm]) or complete['completed_epochs'] != 12:
            raise ValueError('Incomplete final training or recipe mismatch')
        if recipes[arm]['smoke'] or recipes[arm]['test_loaded']:
            raise ValueError('Require full internal-development runs')
        for name in ('curves', 'final'):
            if sha(path/(name+'.pt')) != complete[name+'_sha256']:
                raise ValueError('Source hash mismatch')
        curves[arm] = torch.load(path/'curves.pt', map_location='cpu', weights_only=False, mmap=True)
        finals[arm] = torch.load(path/'final.pt', map_location='cpu', weights_only=False, mmap=True)
        reports[arm] = read(path/'evaluation.json')
        bindings[arm] = complete
    if recipes['white']['initial'] != recipes['ar1']['initial'] or recipes['white']['frozen'] != recipes['ar1']['frozen']:
        raise ValueError('Paired initialization or frozen sources differ')
    for epoch in range(1, 13):
        x, y = [read(a.root/arm/f'epoch{epoch:03d}.json') for arm in ('white', 'ar1')]
        if x['batch_raw_noise_time_sha256'] != y['batch_raw_noise_time_sha256'] or x['total_steps'] != y['total_steps']:
            raise ValueError('Paired draw streams differ')
    white, ar1 = curves['white'], curves['ar1']
    metadata_equal(white, ar1)
    if not white['channel_mask'][:, [41, 42, 43, 44, 45, 5, 6, 12, 13]].all():
        raise ValueError('Temporal coherence summaries require all nine observed dev channels')
    for seed in (42, 123, 2026):
        for mode in ('base', 'aligned_centered'):
            if not torch.equal(white['predictions'][f'{seed}/{mode}'], ar1['predictions'][f'{seed}/{mode}']):
                raise ValueError('Frozen paired controls differ')
    old = torch.load(a.old_run/'dynamics/curves.pt', map_location='cpu', weights_only=False, mmap=True)
    if sha(a.old_run/'dynamics/curves.pt') != read(a.old_run/'dynamics/complete.json')['curves_sha256']:
        raise ValueError('Old comparison hash differs')
    metadata_equal(white, old, compare_b0=False)
    previous = read(a.previous_visual/'provenance.json')
    picks = previous['nine_plot_clips']; ids = [white['clip_id'].index(row['clip_id']) for row in picks]
    modes = ['GT', 'original local baseline', 'aligned with fixed mean', 'centered white flow', 'centered AR1 flow', 'previous run12']
    values = [white['target'], white['predictions']['42/base'], white['predictions']['42/aligned_centered'],
              white['predictions']['42/full'], ar1['predictions']['42/full'], old['predictions']['42/full']]
    motions = torch.stack([value[ids] for value in values]).numpy()
    data = {key: white[key][ids].numpy() for key in ('times', 'valid', 'channel_mask')}
    data.update(channels=np.asarray(ARKIT_NAMES), clip_id=np.asarray([row['clip_id'] for row in picks]))
    a.output.mkdir(parents=True); (a.output/'video_npz').mkdir()
    np.savez_compressed(a.output/'nine_clip_curves.npz', **data, motions=motions, mode_names=np.asarray(modes))
    for job in previous['jobs']:
        i = next(i for i, row in enumerate(picks) if row['speaker'] == job['speaker'])
        np.savez_compressed(a.output/'video_npz'/f"{job['speaker']}.npz", motions=motions[:, i], mode_names=np.asarray(modes),
            channels=data['channels'], **{key: data[key][i] for key in ('times', 'valid', 'channel_mask', 'clip_id')})
    save_json(a.output/'provenance.json', {'source_bindings': bindings, 'nine_plot_clips': picks, 'jobs': previous['jobs'],
        'mode_names': modes, 'noise_seed': 42, 'selection_unchanged': True, 'raw_clamped': False, 'gain_or_lag_fitted': False})
    comparison = {arm: reports[arm]['distribution'] for arm in reports}
    for mode in ('base', 'aligned_centered'):
        control = {**white, 'predictions': {f'{seed}/full': white['predictions'][f'{seed}/{mode}'] for seed in (42, 123, 2026)}}
        comparison[mode] = summarize(control, white['emotion_id'])
    save_json(a.output/'comparison.json', comparison)
    print('CENTERED_EXPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
