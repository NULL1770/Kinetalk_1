"""Add frozen-teacher diagnostics to a completed gate audit without decoding.

Reads only its saved curves and original checkpoint/cache. Preserves a backup
of the original summary before adding diagnostics; never tunes conditions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.audit_audio_activity_gate import readout_saved_prediction, velocity_clip_statistics, write_json, SEEDS
from scripts.summarize_predictable_renderer import verify_reference
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_predictable_renderer import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'checkpoint', 'cache', 'curves-dir', 'audit-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Require positive batch size')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    summary_path = args.audit_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    provenance = summary['provenance']
    if provenance.get('schema') != 'audio_activity_gate_audit_v1' or not provenance.get('frozen_unchanged'):
        raise ValueError('Require completed frozen gate audit')
    for key in ('config', 'checkpoint', 'cache'):
        if sha(getattr(args, key)) != provenance['hashes'][key]:
            raise ValueError(f'Original audit {key} mismatch')
    config = yaml.safe_load(args.config.read_text(encoding='utf8'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if cache.get('schema') != 'predictable_renderer_cache_v1':
        raise ValueError('Wrong cache schema')
    reference = torch.load(args.curves_dir / 'validation_reference.pt', map_location='cpu', weights_only=False)
    validation = cache['splits']['validation']
    verify_reference(reference, validation)
    if len(validation['q']['emotion_id']) != summary['clips']:
        raise ValueError('Audit clip count mismatch')
    system = NeutralAffectSystem(config).to(args.device).eval()
    system.load_state_dict(checkpoint['model'], strict=True)
    freeze_module(system)
    before = state_hash(system.state_dict())
    readout, curve_hashes, velocity = {}, {}, {}
    for seed in SEEDS:
        path = args.curves_dir / f'gate_seed{seed}_curves.pt'
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value.get('noise_seed') != seed or value.get('decode_steps') != 12:
            raise ValueError('Saved generation settings differ')
        if set(value['motion']) != set(summary['scores']):
            raise ValueError('Saved intervention names differ')
        curve_hashes[str(seed)] = sha(path)
        readout[str(seed)] = {mode: readout_saved_prediction(system, validation, prediction,
            device=args.device, batch_size=args.batch_size) for mode, prediction in value['motion'].items()}
        for mode, prediction in value['motion'].items():
            for group, channels in summary['groups'].items():
                row = velocity_clip_statistics(prediction, validation, channels)
                key = mode + '__' + group
                velocity[key] = velocity.get(key, np.zeros_like(row)) + row / len(SEEDS)
        print(json.dumps({'seed': seed, 'accuracy': {mode: row['accuracy'] for mode, row in readout[str(seed)].items()}}), flush=True)
    if state_hash(system.state_dict()) != before or any(p.grad is not None for p in system.parameters()):
        raise RuntimeError('Frozen readout mutated model or accumulated gradients')
    neutral = config['data']['emotion_classes'].index('neutral')
    labels = validation['q']['emotion_id']
    populations = {'all': torch.arange(len(labels)), 'neutral': (labels == neutral).nonzero(as_tuple=True)[0],
                   'nonneutral': (labels != neutral).nonzero(as_tuple=True)[0]}
    velocity_summary = {}
    for key, values in velocity.items():
        velocity_summary[key] = {}
        for population, indices in populations.items():
            total = values[indices.numpy()].sum(0)
            velocity_summary[key][population] = {'mse_per_second': float(total[0] / total[1]) if total[1] else None,
                                                'sse': float(total[0]), 'observed_velocity_values': float(total[1])}
    supplement = {'schema': 'audio_activity_gate_teacher_readout_v1', 'checkpoint_sha256': sha(args.checkpoint),
        'original_summary_sha256': sha(summary_path), 'curve_sha256': curve_hashes, 'reference_sha256': sha(args.curves_dir / 'validation_reference.pt'),
        'source_sha256': sha(__file__), 'frozen_unchanged': True, 'new_decoding': False, 'new_test_loaded': False,
        'frozen_teacher_readout': readout,
        'frozen_teacher_accuracy': {seed: {mode: row['accuracy'] for mode, row in values.items()} for seed, values in readout.items()},
        'velocity': velocity_summary,
        'note': 'Same frozen training motion teacher; class readout is a diagnostic, not an independent perceptual score'}
    supplement_path = args.audit_dir / 'teacher_readout.json'
    if supplement_path.exists():
        raise FileExistsError('Teacher supplement already exists; refuse to overwrite')
    backup = args.audit_dir / 'summary_before_teacher_readout.json'
    if backup.exists():
        raise FileExistsError('Original summary backup already exists; inspect before rerunning')
    shutil.copyfile(summary_path, backup)
    np.savez_compressed(args.audit_dir / 'velocity_clip_statistics.npz', **velocity)
    write_json(supplement_path, supplement)
    summary.update(frozen_teacher_readout=readout, frozen_teacher_accuracy=supplement['frozen_teacher_accuracy'],
                   teacher_readout_supplement='teacher_readout.json', velocity=velocity_summary)
    write_json(summary_path, summary)
    print('COMPLETE: added frozen teacher readout from saved curves; no generation/training', flush=True)


if __name__ == '__main__':
    main()
