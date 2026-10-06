"""Replay frozen auxiliary probes on previously saved validation features.

No fitting, generator updates, metric replacement, or sealed-test access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh replay output required')
    torch.set_num_threads(2)
    src = json.loads((args.diagnostic / 'report.json').read_text(encoding='utf8'))
    if src['test_loaded'] or src['training_performed']:
        raise ValueError('Frozen validation diagnostic required')
    manifest = src['data_manifest_sha256']
    probes = []
    metadata = []
    for path in args.probe:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if (ck['kind'] != 'stable_train_motion_statistics_probe_v1'
                or ck['train_manifest_sha256'] != manifest or ck['test_used_for_selection']
                or ck['generator_outputs_used_for_fitting'] or ck['classes'] != src['classes']):
            raise ValueError('Probe does not match independent stability protocol')
        model = MotionEmotionProbe(ck['feature_dim'], ck['hidden'], len(ck['classes'])).eval()
        model.load_state_dict(ck['model'], strict=True)
        probes.append((model, ck['feature_mask']))
        metadata.append({'path': str(path), 'sha256': sha(path), 'feature_count': ck['feature_dim'],
                         'training_on_generated_motion': False, 'kind': ck['kind']})
    results = {}
    with np.load(args.diagnostic / 'per_clip_features_metrics.npz', allow_pickle=False) as z:
        labels = torch.from_numpy(z['labels']).long()
        if len(labels) != src['clips']:
            raise ValueError('Diagnostic feature count mismatch')
        for key in z.files:
            if not key.endswith('__features'):
                continue
            full = torch.from_numpy(z[key])
            if full.shape[0] != len(labels) or not torch.isfinite(full).all():
                raise ValueError('Invalid feature archive')
            scores = []
            for model, fm in probes:
                if full.shape[1] != len(fm):
                    raise ValueError('Feature width mismatch')
                scores.append(classification_metrics(labels, model(full[:, fm]).argmax(-1), src['classes']))
            results[key.removesuffix('__features')] = scores
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'schema': 'auxiliary_stable_probe_feature_replay_v1', 'test_loaded': False,
        'training_performed': False, 'clips': src['clips'],
        'source_diagnostic': str(args.diagnostic), 'source_report_sha256': sha(args.diagnostic / 'report.json'),
        'source_features_sha256': sha(args.diagnostic / 'per_clip_features_metrics.npz'),
        'generator_checkpoint_sha256': src['checkpoint_sha256'], 'data_manifest_sha256': manifest,
        'probes': metadata, 'results': results, 'source_sha256': sha(Path(__file__)),
        'policy': 'Auxiliary stability readout only; original F1 remains unchanged.'
    }, indent=2), encoding='utf8')
    print(json.dumps({k: [round(s['macro_f1'], 4) for s in scores] for k, scores in results.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
