"""Historical fixed-nine comparison of static/dynamic separated candidates."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import sha, read, metadata_equal
from scripts.export_full_staged_examples import ARKIT_NAMES
from scripts.train_formal_predictable_projection import save_json


def main():
    p = argparse.ArgumentParser()
    for name in ('state-run', 'repair-root', 'old-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Fresh export required')
    complete = read(a.state_run/'complete.json')
    loaded = {}
    bindings = {}
    for key in ('state_aligned', 'state_white'):
        path = a.state_run/(key+'_curves.pt')
        if sha(path) != complete[key]:
            raise ValueError('State curve binding differs')
        loaded[key] = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        bindings[key] = sha(path)
    for key, folder in [('old', a.old_run/'dynamics'), ('candidate', a.repair_root/'candidate'), ('base', a.repair_root/'centered_prior/white')]:
        path = folder/'curves.pt'
        if sha(path) != read(folder/'complete.json')['curves_sha256']:
            raise ValueError('Control curve binding differs')
        loaded[key] = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        bindings[key] = sha(path)
    reference = loaded['state_aligned']
    for key, value in loaded.items():
        metadata_equal(reference, value, compare_b0=key != 'old')
    previous = read(a.repair_root/'visual/provenance.json')
    picks = previous['nine_plot_clips']; ids = [reference['clip_id'].index(row['clip_id']) for row in picks]
    modes = ['GT', 'previous run12', 'previous regional candidate', 'audio mean + aligned dynamics', 'audio mean + centered flow', 'original local baseline']
    values = [reference['target'], loaded['old']['predictions']['42/full'], loaded['candidate']['predictions']['42/full'],
              reference['predictions']['42/full'], loaded['state_white']['predictions']['42/full'], loaded['base']['predictions']['42/base']]
    motions = torch.stack([value[ids] for value in values]).numpy()
    data = {key: reference[key][ids].numpy() for key in ('times', 'valid', 'channel_mask')}
    data.update(channels=np.asarray(ARKIT_NAMES), clip_id=np.asarray([row['clip_id'] for row in picks]))
    a.output.mkdir(parents=True); (a.output/'video_npz').mkdir()
    np.savez_compressed(a.output/'nine_clip_curves.npz', **data, motions=motions, mode_names=np.asarray(modes))
    for job in previous['jobs']:
        i = next(i for i, row in enumerate(picks) if row['speaker'] == job['speaker'])
        np.savez_compressed(a.output/'video_npz'/f"{job['speaker']}.npz", motions=motions[:, i], mode_names=np.asarray(modes), channels=data['channels'],
            **{key: data[key][i] for key in ('times', 'valid', 'channel_mask', 'clip_id')})
    save_json(a.output/'provenance.json', {'source_sha256': bindings, 'nine_plot_clips': picks, 'jobs': previous['jobs'],
        'mode_names': modes, 'seed': 42, 'selection_unchanged': True, 'development_selected_architecture': True,
        'amplitude_or_lag_fitted': False, 'raw_clamped': False})
    print('STATE_MEAN_EXPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
