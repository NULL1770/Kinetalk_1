"""Read-only train/validation probe audit using perturbed real validation motion.

No fitting, generator update, or sealed-test access. Perturbations diagnose the
assessor's sensitivity and cannot be used as generated-motion quality scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features, classification_metrics
from scripts.packed_trainval_cache import load_packed
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, sha


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh audit output required')
    torch.set_num_threads(2)
    data = load_packed(args.data, materialize=False)
    if set(data['splits']) != {'train', 'validation'} or data['provenance']['test_loaded']:
        raise ValueError('Only train/validation data allowed')
    manifest = data['provenance']['manifest_sha256']
    probes, metadata, feature_masks, full_scales = [], [], [], []
    support = None
    names = None
    for path in args.probe:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if (ck['train_manifest_sha256'] != manifest or ck.get('test_used_for_selection')
                or ck.get('generator_outputs_used_for_fitting')):
            raise ValueError('Probe must be fitted on real train motion only')
        if support is not None and (not torch.equal(support, ck['channel_support']) or names != ck['classes']):
            raise ValueError('Probe channel support/class order differ')
        support, names = ck['channel_support'].bool(), ck['classes']
        p = MotionEmotionProbe(ck['feature_dim'], ck['hidden'], len(names)).eval()
        p.load_state_dict(ck['model'], strict=True)
        probes.append(p)
        full_dim = 6 * int(support.sum())
        fm = ck.get('feature_mask', torch.ones(full_dim, dtype=torch.bool))
        if fm.shape != (full_dim,) or fm.dtype != torch.bool or int(fm.sum()) != ck['feature_dim']:
            raise ValueError('Invalid feature mask')
        feature_masks.append(fm)
        full_scales.append(ck.get('train_full_feature_scale', p.scale))
        metadata.append({'path': str(path), 'sha256': sha(path), 'kind': ck.get('kind'),
                         'feature_count': int(fm.sum())})
    c = int(support.sum())
    # This selection uses only train-fitted normalization, never generated or
    # validation values. It is an audit intervention, not a new metric mask.
    near_constant = (full_scales[0].reshape(6, c) <= 1.0001e-4).all(0)
    if any(not torch.equal((s.reshape(6, c) <= 1.0001e-4).all(0), near_constant) for s in full_scales):
        raise ValueError('Probes must share train-derived constant channels')
    q = data['splits']['validation']
    if not q['channel_mask'][:, support].all():
        raise ValueError('Missing validation observations on probe support')
    ids = torch.arange(len(q['_lengths']))
    if args.limit_per_class:
        selected = []
        for cls in range(len(names)):
            eligible = (q['emotion_id'] == cls).nonzero().flatten()
            at = torch.linspace(0, len(eligible) - 1, min(args.limit_per_class, len(eligible))).long()
            selected.extend(eligible[at].tolist())
        ids = torch.tensor(sorted(selected))
    rows = {}
    labels = q['emotion_id'][ids]
    generator = torch.Generator().manual_seed(args.seed)
    for n, i in enumerate(ids.tolist()):
        b = q.batch(torch.tensor([i]), keys=('motion', 'valid'))
        motion, valid = b['motion'][0][:, support], b['valid'][0]
        noise = torch.randn(motion.shape, generator=generator)
        for region, selected in [('all', torch.ones(c, dtype=torch.bool)), ('constant', near_constant)]:
            for sigma in args.sigmas:
                x = motion + sigma * noise * selected[None]
                for variant, z in [('raw', x), ('clip', x.clamp(0., 1.))]:
                    key = f'{region}_sigma{sigma:g}_{variant}'
                    rows.setdefault(key, []).append(motion_features(z, valid))
        if n % 200 == 0:
            print(json.dumps({'event': 'audit_clips', 'clips': n + 1, 'total': len(ids)}), flush=True)
    results = {}
    for key, fs in rows.items():
        features = torch.stack(fs)
        results[key] = [classification_metrics(labels, p(features[:, fm]).argmax(-1), names)
                        for p, fm in zip(probes, feature_masks)]
    channel_indices = support.nonzero().flatten()[near_constant].tolist()
    report = {'schema': 'real_motion_probe_robustness_v1', 'clips': len(ids),
              'test_loaded': False, 'training_performed': False,
              'data_manifest_sha256': manifest, 'probes': metadata,
              'constant_selection': 'all six train-fitted feature scales <= 1.0001e-4',
              'constant_channels': [ARKIT_NAMES[i] for i in channel_indices],
              'seed': args.seed, 'sigmas': args.sigmas, 'results': results,
              'source_sha256': sha(__file__),
              'policy': 'Perturbed GT diagnostic, not generated scores or replacement metrics.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({'event': 'complete', 'output': str(args.output)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit-per-class', type=int, default=0)
    p.add_argument('--seed', type=int, default=2718)
    p.add_argument('--sigmas', type=float, nargs='+', default=[0., .0001, .001, .005, .01])
    args = p.parse_args()
    if args.limit_per_class < 0 or any(not np.isfinite(s) or s < 0 for s in args.sigmas):
        raise ValueError('Invalid perturbation configuration')
    run(args)


if __name__ == '__main__':
    main()
