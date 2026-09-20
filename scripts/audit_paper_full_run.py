"""Independently recompute final scores from hash-bound saved validation curves.

Reads completed run outputs only. Does not load datasets, fit statistics,
generate new samples, select a checkpoint, or consume sealed test targets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

UPPER = (41, 42, 43, 44, 45, 5, 6, 12, 13)
LIPS = (15, 16, 27, 28, 19, 20, 31, 32, 33, 34, 39, 40)
FDD_REGION = tuple(range(14)) + (49, 50)
NONUPPER = tuple(i for i in range(52) if i not in UPPER)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def independent_scores(x, y, valid, channels, scales):
    """Float64 formulas written independently of the production scorers."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    mask = valid[:, None] & channels[None, :]
    result = {}
    for name, cc in [('arkit_mbe', tuple(range(52))), ('arkit_lbe', LIPS)]:
        m = mask[:, cc]
        keep = m.any(1)
        diff = np.where(m[None], x[:, :, cc] - y[None, :, cc], 0.)
        result[name] = float(np.sqrt((diff[:, keep] ** 2).sum(-1)).mean())
    for suffix, cc in [('', FDD_REGION), ('supp_upper9_', UPPER)]:
        keep = mask[:, cc].all(1)
        truth = (y[keep][:, cc] ** 2).sum(-1).std()
        pred = (x[:, keep][:, :, cc] ** 2).sum(-1).std(axis=1)
        result['arkit_fdd_signed' if not suffix else suffix + 'fdd_signed'] = float((truth - pred).mean())
        result['arkit_fdd_absolute' if not suffix else suffix + 'fdd_absolute'] = float(np.abs(truth - pred).mean())
    observed = mask[:, UPPER].all(1)
    xx, yy = x[:, :, UPPER] / scales, y[:, UPPER] / scales
    edges = np.diff(np.r_[False, observed, False].astype(np.int8))
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    for centered in (False, True):
        a, b = xx.copy(), yy.copy()
        if centered:
            for left, right in spans:
                a[:, left:right] -= a[:, left:right].mean(1, keepdims=True)
                b[left:right] -= b[left:right].mean(0, keepdims=True)
        a, b = a[:, observed].reshape(len(a), -1), b[observed].reshape(-1)
        distance = np.sqrt(((a - b) ** 2).mean(1)).mean()
        # Ordered off-diagonal pairs; unbiased finite-ensemble correction.
        spread = sum(np.sqrt(((a[i] - a[j]) ** 2).mean())
                     for i in range(len(a)) for j in range(len(a)) if i != j)
        score = distance - spread / (2 * len(a) * (len(a) - 1))
        result['centered_es' if centered else 'raw_es'] = float(score)
    return result


def audit(root, artifacts=None):
    stage4 = root / 'audio/audio'
    complete4 = read(stage4 / 'complete.json')
    require(sha(stage4 / 'final.pt') == complete4['final_sha256'], 'Stage4 hash mismatch')
    reference = torch.load(stage4 / 'final.pt', map_location='cpu', weights_only=False, mmap=True)
    arms, previous = {}, None
    for arm in ('audio', 'static'):
        folder = root / arm / 'dynamics'
        complete = read(folder / 'complete.json')
        bindings = {}
        curve_path = artifacts / arm / 'dynamics/curves.pt' if artifacts else folder / 'curves.pt'
        for filename, key in [('final.pt', 'final_sha256'), ('curves.pt', 'curves_sha256')]:
            bindings[filename] = sha(curve_path if filename == 'curves.pt' else folder / filename)
            require(bindings[filename] == complete[key], f'{arm} {filename} hash mismatch')
        ck = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False, mmap=True)
        require(ck['data_manifest_sha256'] == reference['data_manifest_sha256'], 'Data protocol changed')
        require(ck['completed_epochs'] == 12 and ck['test_loaded'] is False, 'Endpoint/test protocol differs')
        for module in ('system', 'audio'):
            require(set(ck[module]) == set(reference[module]), 'Frozen module keys differ')
            require(all(torch.equal(v, ck[module][k]) for k, v in reference[module].items()), 'Frozen module changed: ' + module)
        curves = torch.load(curve_path, map_location='cpu', weights_only=False, mmap=True)
        ids = curves['clip_id']
        require(len(ids) == len(set(ids)) == 446, 'Expected exactly 446 distinct validation clips')
        require(curves['noise_seeds'] == [42, 123, 2026], 'Noise protocol differs')
        if previous is not None:
            require(ids == previous['clip_id'], 'Arms differ in membership/order')
            for key in ('target', 'valid', 'channel_mask', 'times', 'b0'):
                require(torch.equal(curves[key], previous[key]), 'Arms differ in ' + key)
        require(torch.allclose(curves['times'][:, 1:] - curves['times'][:, :-1],
                               torch.full_like(curves['times'][:, 1:], .04), atol=1e-7, rtol=1e-5), 'Native clock differs')
        report = read(folder / 'benchmark_summary.json')
        scales = ck['scales'][list(UPPER)].numpy().astype(np.float64)
        results = {}
        for mode in ('base', 'full', 'static_state', 'oracle_state', 'reverse_audio'):
            values = []
            for seed in curves['noise_seeds']:
                pred = curves['predictions'][f'{seed}/{mode}']
                base = curves['predictions'][f'{seed}/base']
                require(torch.equal(pred[..., list(NONUPPER)], base[..., list(NONUPPER)]), 'Non-upper protection changed')
            for i in range(len(ids)):
                x = np.stack([curves['predictions'][f'{s}/{mode}'][i].numpy() for s in curves['noise_seeds']])
                values.append(independent_scores(x, curves['target'][i].numpy(), curves['valid'][i].numpy(),
                                                  curves['channel_mask'][i].numpy(), scales))
            means = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
            expected = {k: v['value'] for k, v in report[mode]['coefficient'].items()}
            expected.update(raw_es=report[mode]['temporal']['joint_fair_es']['raw'],
                            centered_es=report[mode]['temporal']['joint_fair_es']['centered'])
            maximum = max(abs(v - expected[k]) for k, v in means.items())
            require(maximum < 1e-9, f'Independent scores differ: {arm}/{mode}: {maximum}')
            results[mode] = {'scores': means, 'max_report_difference': maximum}
        arms[arm] = {'bindings': bindings, 'modes': results, 'frozen_stage4_modules_exact': True,
                     'nonupper43_exact_within_arm': True, 'clips': len(ids), 'valid_frames': int(curves['valid'].sum())}
        previous = curves
    return {'passed': True, 'arms': arms, 'test_targets_loaded': False,
            'scope': 'Saved validation outputs only; numerical audit is not a perceptual or scientific success claim.',
            'implementation_sha256': sha(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--artifacts', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run, a.artifacts)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'passed': result['passed'], 'report': str(a.output)}))


if __name__ == '__main__':
    main()
