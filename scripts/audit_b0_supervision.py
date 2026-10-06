"""Read-only B0 lineage, safe-pair coverage and native-clock jaw diagnostics.

Fixed lag profiles are diagnostic only. No lag/gain is selected or applied to
inference, no probe is fitted, and no optimizer or sealed-test input is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_matched_motion_curves import REGIONS, RC, region_values, means
from scripts.diagnose_flow_sampling import sha
from scripts.packed_trainval_cache import load_packed
from scripts.train_full_staged import load_safe_native_targets, canonical_hash, subset


def corr(x, y):
    x, y = x.double(), y.double()
    x, y = x - x.mean(), y - y.mean()
    d = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / d) if d > 1e-12 else float('nan')


def lag_profile(pred, target, valid, times, radius=5):
    """p[t] versus target[t+lag], with the same contiguous anchors for all lags."""
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    anchors = []
    for t in range(radius, len(valid) - radius - 1):
        window = slice(t - radius, t + radius + 2)
        dt = times[window][1:] - times[window][:-1]
        if valid[window].all() and ((dt > 0) & (dt <= .075)).all():
            anchors.append(t)
    if len(anchors) < 8:
        return np.full((2, radius * 2 + 1), np.nan), len(anchors)
    ix = torch.tensor(anchors, dtype=torch.long)
    out = []
    for lag in range(-radius, radius + 1):
        j = ix + lag
        out.append([corr(pred[ix], target[j]),
                    corr(pred[ix + 1] - pred[ix], target[j + 1] - target[j])])
    return np.asarray(out).T, len(anchors)


def lineage(root, checkpoint_root):
    index = {}
    for p in checkpoint_root.glob('*/**/complete.json'):
        try:
            c = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if c.get('final_sha256'):
            index[c['final_sha256']] = p
    chain, visited = [], set()
    stage = root / 'audio'
    while True:
        c = json.loads((stage / 'complete.json').read_text())
        s = c['final_sha256']
        if s in visited:
            raise ValueError('Cyclic checkpoint lineage')
        visited.add(s)
        r = json.loads((stage.parent / 'provenance.json').read_text())['recipe']
        assert not r['test_loaded']
        chain.append({'run': str(stage.parent), 'stage': stage.name,
                      'checkpoint_sha256': s, 'stages': r['stages'],
                      'articulation_scope': r.get('articulation_scope'),
                      'parent_sha256': r.get('stage_checkpoint_sha256'),
                      'safe_dtw_manifest_sha256': r.get('safe_dtw_manifest_sha256')})
        parent = r.get('stage_checkpoint_sha256')
        if not parent:
            break
        if parent not in index:
            chain.append({'unresolved_parent_sha256': parent})
            break
        stage = index[parent].parent
    return chain


def grouped_region(rows, groups):
    out = {'all': means(rows, RC)}
    for name, labels in groups.items():
        out[name] = {str(k): {'clips': int((labels == k).sum()),
                            **means(rows[labels == k], RC)} for k in np.unique(labels)}
    return out


@torch.no_grad()
def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh diagnostic required')
    torch.set_num_threads(2)
    provenance = json.loads((a.run_root / 'provenance.json').read_text())
    recipe = provenance['recipe']
    assert not recipe['test_loaded']
    complete = json.loads((a.run_root / 'audio/complete.json').read_text())
    curves = a.run_root / 'audio/curves.pt'
    assert sha(curves) == complete['curves_sha256']
    assert sha(a.run_root / 'audio/final.pt') == complete['final_sha256']
    data = load_packed(a.data, materialize=False, with_refs=False)
    assert set(data['splits']) == {'train', 'validation'}
    assert not data['provenance']['test_loaded']
    assert data['provenance']['manifest_sha256'] == recipe['data_provenance']['manifest_sha256']
    manifest = a.safe_root / 'teacher_manifest_gated.jsonl'
    assert sha(manifest) == recipe['safe_dtw_manifest_sha256']
    q = data['splits']['train']
    ids = [str(c) for c in q['clip_id']]
    safe = load_safe_native_targets(a.safe_root, ids)
    neutral = q['emotion_id'].eq(0).nonzero(as_tuple=True)[0].tolist()
    for i in neutral:
        safe.pop(ids[i], None)
    assert len(safe) == recipe['articulation_scope']['safe_target_clip_count']
    assert canonical_hash({cid: v[4] for cid, v in sorted(safe.items())}) == recipe['safe_dtw_artifacts_sha256']
    lookup = {cid: i for i, cid in enumerate(ids)}
    manifest_rows = {str(r['source_clip_id']): r for r in
                     (json.loads(s) for s in manifest.read_text().splitlines() if s.strip())}
    coverage = {'neutral_clips': len(neutral), 'paired_clips': len(safe),
                'paired_full_gate_clips': 0, 'paired_partial_gate_clips': 0,
                'paired_zero_mouth_frames_clips': 0, 'paired_native_valid_frames': 0,
                'paired_retained_mouth_frames': 0, 'neutral_native_valid_frames': 0}
    pair_rows = []
    for cid, (teacher, mask, channel, times, _) in safe.items():
        i = lookup[cid]
        b = subset(q, torch.tensor([i]), 'cpu', keys=('valid', 'channel_mask', 'times'))
        v = b['valid'][0, :len(mask)] & mask
        torch.testing.assert_close(b['times'][0, :len(times)], times, rtol=0, atol=1e-7)
        mouth = v & channel[:, 14:41].all(-1) & b['channel_mask'][0, 14:41].all()
        gate = bool(manifest_rows[cid].get('mouth_event_gate', False))
        coverage['paired_full_gate_clips' if gate else 'paired_partial_gate_clips'] += 1
        coverage['paired_zero_mouth_frames_clips'] += int(not mouth.any())
        coverage['paired_native_valid_frames'] += int(v.sum())
        coverage['paired_retained_mouth_frames'] += int(mouth.sum())
        pair_rows.append([i, int(v.sum()), int(mouth.sum()), int(gate)])
    for ix in torch.tensor(neutral).split(128):
        b = subset(q, ix, 'cpu', keys=('valid',))
        coverage['neutral_native_valid_frames'] += int(b['valid'].sum())
    coverage['effective_clips'] = len(neutral) + len(safe)
    coverage['paired_retained_frame_fraction'] = (coverage['paired_retained_mouth_frames'] /
                                                 coverage['paired_native_valid_frames'])
    z = torch.load(curves, map_location='cpu', weights_only=False)
    val = data['splits']['validation']
    val_lookup = {str(c): i for i, c in enumerate(val['clip_id'])}
    assert len(z['clip_id']) == len(val_lookup)
    ix = [val_lookup[str(c)] for c in z['clip_id']]
    groups = {k: val[v][ix].numpy() for k, v in
              [('emotion', 'emotion_id'), ('speaker', 'speaker_id'), ('intensity', 'intensity_id')]}
    arrays = {'clip_id': np.asarray(z['clip_id']), **groups, 'pair_coverage': np.asarray(pair_rows)}
    report = {'schema': 'b0_supervision_native_audit_v1', 'test_loaded': False,
              'training_performed': False, 'model_changed': False,
              'checkpoint_sha256': complete['final_sha256'], 'curves_sha256': sha(curves),
              'data_manifest_sha256': data['provenance']['manifest_sha256'],
              'audit_script_sha256': sha(__file__),
              'loader_script_sha256': sha(Path(__file__).with_name('train_full_staged.py')),
              'coverage_scope': 'reconstructed native mask intersection; artifact hashes match stored recipe',
              'coverage': coverage, 'lineage': lineage(a.run_root, a.checkpoint_root),
              'classes': data['config']['data']['emotion_classes'], 'region_columns': RC,
              'regions': {}, 'lag_frames': list(range(-5, 6)),
              'lag_definition': 'prediction[t] vs GT[t+lag]; common contiguous native anchors, no correction applied',
              'lag_results': {}}
    branches = {'b0': z['b0'], **z['predictions']}
    for branch, pred in branches.items():
        p = pred.clamp(0, 1)
        rows, lag_rows, counts = [], [], []
        for i in range(len(ix)):
            t, v, ch, tm = z['target'][i], z['valid'][i], z['channel_mask'][i], z['times'][i]
            rows.append([region_values(p[i], t, v, ch, r) for r in REGIONS.values()])
            lags, n = lag_profile(p[i, :, 17], t[:, 17], v, tm)
            lag_rows.append(lags); counts.append(n)
        rows, lag_rows = np.asarray(rows), np.asarray(lag_rows)
        arrays[branch + '__regions'], arrays[branch + '__lags'] = rows, lag_rows
        report['regions'][branch] = {name: grouped_region(rows[:, j], groups)
                                      for j, name in enumerate(REGIONS)}
        def aggregate(selected):
            arows = lag_rows[selected]
            return {'clips': int(selected.sum()), 'scored_clips': int(np.isfinite(arows[:, 0, 5]).sum()),
                    'motion_correlation': means(arows[:, 0], [str(k) for k in range(-5, 6)]),
                    'displacement_correlation': means(arows[:, 1], [str(k) for k in range(-5, 6)])}
        report['lag_results'][branch] = {'all': aggregate(np.ones(len(ix), dtype=bool)),
            'emotion': {str(k): aggregate(groups['emotion'] == k) for k in np.unique(groups['emotion'])},
            'neutral_by_speaker': {str(k): aggregate((groups['emotion'] == 0) & (groups['speaker'] == k))
                                  for k in np.unique(groups['speaker'])}}
        arrays['lag_anchor_counts'] = np.asarray(counts)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'per_clip_audit.npz', **arrays)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'complete': str(a.output), 'coverage': coverage}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--safe-root', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
