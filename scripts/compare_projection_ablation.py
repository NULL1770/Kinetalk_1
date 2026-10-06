"""Compare matched validation continuations, with original and auxiliary probes."""
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


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def macro_f1(y, pred, classes):
    cm = np.bincount(y * classes + pred, minlength=classes * classes).reshape(classes, classes)
    return float(np.mean(2 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1)))


def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh comparison required')
    torch.set_num_threads(2)
    branch = a.experiment_name
    if branch == 'control' or not branch.replace('_', '').isalnum():
        raise ValueError('Expected a distinct alphanumeric experiment name')
    roots = {'control': a.control, branch: a.experiment}
    reports = {k: json.loads((v / 'report.json').read_text()) for k, v in roots.items()}
    arrays = {k: dict(np.load(v / 'per_clip_features_metrics.npz', allow_pickle=False)) for k, v in roots.items()}
    manifests = {r['data_manifest_sha256'] for r in reports.values()}
    if len(manifests) != 1 or any(r['test_loaded'] or r['training_performed'] for r in reports.values()):
        raise ValueError('Expected matching frozen validation diagnostics')
    for k in ('clip_id', 'labels'):
        np.testing.assert_array_equal(arrays['control'][k], arrays[branch][k])
    if reports['control']['metric_columns'] != reports[branch]['metric_columns']:
        raise ValueError('Metric columns differ')
    names = reports['control']['classes']; labels = arrays['control']['labels']; n = len(labels)
    models, meta = [], []
    for path in a.probe:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if (ck['train_manifest_sha256'] not in manifests or ck['classes'] != names
                or ck.get('test_used_for_selection') or ck.get('generator_outputs_used_for_fitting')):
            raise ValueError('Independent real-train probe binding mismatch')
        model = MotionEmotionProbe(ck['feature_dim'], ck['hidden'], len(names)).eval()
        model.load_state_dict(ck['model'], strict=True)
        fm = ck.get('feature_mask', torch.ones(6 * int(ck['channel_support'].sum()), dtype=torch.bool))
        models.append((model, fm))
        meta.append({'path': str(path), 'sha256': sha(path), 'kind': ck['kind'], 'fitted_on_generated': False})
    rows, predictions = {}, {}
    with torch.no_grad():
        for branch, z in arrays.items():
            for variant in ('raw', 'clip_all'):
                key = f'audio_euler12__{variant}'
                features = torch.from_numpy(z[key + '__features'])
                ps = [model(features[:, fm]).argmax(-1).numpy() for model, fm in models]
                predictions[(branch, variant)] = ps
                rows[f'{branch}_{variant}'] = {
                    'metrics': reports[branch]['results']['audio_euler12']['variants'][variant]['metrics'],
                    'probes': [classification_metrics(torch.from_numpy(labels), torch.from_numpy(p), names) for p in ps],
                }
    # Equal final clipping isolates training effect from projection-at-inference.
    rng = np.random.default_rng(2718)
    metric_delta = arrays[branch]['audio_euler12__clip_all__metrics'] - arrays['control']['audio_euler12__clip_all__metrics']
    bs_metrics, bs_f1 = [], []
    for _ in range(2000):
        ix = rng.integers(0, n, n)
        bs_metrics.append(metric_delta[ix].mean(0))
        bs_f1.append([macro_f1(labels[ix], pb[ix], len(names)) - macro_f1(labels[ix], pc[ix], len(names))
                      for pc, pb in zip(predictions[('control', 'clip_all')], predictions[(branch, 'clip_all')])])
    intervals = {
        'metric_delta': dict(zip(reports['control']['metric_columns'],
                                np.quantile(bs_metrics, [.025, .975], axis=0).T.tolist())),
        'probe_f1_delta': np.quantile(bs_f1, [.025, .975], axis=0).T.tolist(),
        'policy': '2000 paired clip bootstraps seed2718; validation uncertainty, not a test claim',
    }
    result = {'schema': 'matched_continuation_comparison_v2', 'experiment_name': branch,
              'delta_direction': f'{branch} minus control', 'clips': n, 'test_loaded': False,
              'data_manifest_sha256': next(iter(manifests)), 'probes': meta, 'rows': rows,
              'matched_clip_bootstrap_95ci': intervals,
              'diagnostic_report_sha256': {k: sha(v / 'report.json') for k, v in roots.items()},
              'checkpoint_sha256': {k: r['checkpoint_sha256'] for k, r in reports.items()},
              'policy': 'Original probe and independent auxiliary readouts reported separately; no probe training here.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: {'metrics': v['metrics'], 'F1': [s['macro_f1'] for s in v['probes']]} for k, v in rows.items()}, indent=2))
    print(json.dumps(intervals, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--experiment', '--bounded', dest='experiment', type=Path, required=True)
    p.add_argument('--experiment-name', default='bounded')
    p.add_argument('--probe', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
