"""Replay bound frozen intensity heads on saved codes; no fitting or generation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_semantic_code_subspace import probabilities


def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh intensity report required')
    z = np.load(a.source / 'per_clip_conditions.npz', allow_pickle=False)
    source = json.loads((a.source / 'report.json').read_text())
    assert not source['test_loaded']
    heads = {}
    for name in ('audio', 'teacher'):
        path = a.heads / (name + '_heads.npz')
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['checkpoint_sha256'] == source['checkpoint_sha256']
        assert meta['data_manifest_sha256'] == source['data_manifest_sha256']
        assert meta['npz_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        heads[name] = dict(np.load(path, allow_pickle=False))
    report = {'schema': 'saved_intensity_head_audit_v1', 'test_loaded': False,
              'training_performed': False, 'generation_performed': False,
              'checkpoint_sha256': source['checkpoint_sha256'],
              'data_manifest_sha256': source['data_manifest_sha256'], 'results': {}}
    for role in ('train', 'validation'):
        target = z[role + '__intensity']
        rows = report['results'][role] = {}
        for name, head in heads.items():
            code = z[role + '__' + name + '_global'].astype(np.float64)
            p = probabilities(code, head['intensity_weight'], head['intensity_bias'])
            replayed = p.argmax(1)
            predicted = z[role + '__' + name + '_intensity']
            mismatch = replayed != predicted
            logits = code @ head['intensity_weight'].T + head['intensity_bias']
            sorted_logits = np.sort(logits, axis=1)
            value = p @ np.arange(4)
            cm = np.bincount(target * 4 + predicted, minlength=16).reshape(4, 4)
            f1 = 2 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1)
            rows[name] = {'accuracy': float(np.mean(predicted == target)),
                'class_metrics_policy': 'Saved original GPU argmax; expected value uses float64 frozen-head replay',
                'float64_argmax_exact_parity': bool(not mismatch.any()),
                'float64_argmax_mismatches': int(mismatch.sum()),
                'mismatch_top2_logit_gaps': (sorted_logits[mismatch, -1]-sorted_logits[mismatch, -2]).tolist(),
                'macro_f1': float(f1.mean()), 'confusion': cm.tolist(),
                'expected_value_mse': float(np.mean((value-target)**2)),
                'by_real_level': {str(level): {'clips': int((target==level).sum()),
                    'mean_argmax': float(predicted[target==level].mean()),
                    'mean_expected_value': float(value[target==level].mean()),
                    'expected_value_mae': float(np.abs(value[target==level]-level).mean())}
                    for level in range(4)}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--heads', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
