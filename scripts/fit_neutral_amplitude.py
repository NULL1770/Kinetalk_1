"""Fit one frozen-B0 neutral amplitude candidate; never update the student."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_amplitude import NeutralAmplitudeFit, apply_calibration, measures, held_gate
from scripts.train_expression_response import configure, load_runtime, cache_base, development_fold, state_digest, sha, write
from scripts.train_full_staged import load_safe_native_targets, safe_target_batch


def aggregate(rows):
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
        if any(r[k] is not None for r in rows) else None for k in rows[0]}


@torch.no_grad()
def run(a):
    configure(47)
    out = Path(a.output)
    out.mkdir(exist_ok=False)
    device = torch.device(a.device)
    binding = json.loads(Path(a.binding).read_text())
    source = Path(__file__).resolve().parents[1]
    for name, digest in binding['source_files'].items():
        assert sha(source/name) == digest, name
    manifest = Path(binding['safe_root'])/'teacher_manifest_gated.jsonl'
    assert sha(manifest) == binding['safe_manifest_sha256']
    data, base = load_runtime(binding, device)
    digest = state_digest(base.stage1)
    q = data['splits']['train']
    folds = development_fold(q, data['fit_sids'])
    neutral = data['config']['data']['emotion_classes'].index('neutral')
    approved = {r['source_clip_id'] for r in map(json.loads, manifest.read_text().splitlines())
        if r.get('teacher_eligible') is True}
    selected = {role: [i for i in folds[role] if int(q['emotion_id'][i]) == neutral or q['clip_id'][i] in approved]
        for role in ('train', 'speaker_dev', 'sentence_dev')}
    if a.smoke:
        selected = {role: sorted([i for i in ids if int(q['emotion_id'][i]) == neutral][:4]+
            [i for i in ids if int(q['emotion_id'][i]) != neutral][:4]) for role, ids in selected.items()}
    elif sum(map(len, selected.values())) != 3298:
        raise ValueError('Expected original 715 native neutral + 2583 approved pairs')
    all_ids = sorted(i for ids in selected.values() for i in ids)
    paired_ids = [i for i in all_ids if int(q['emotion_id'][i]) != neutral]
    targets = load_safe_native_targets(binding['safe_root'], [q['clip_id'][i] for i in paired_ids])
    assert set(targets) == {q['clip_id'][i] for i in paired_ids}
    canonical = sorted({j for i in all_ids for j in range(i//32*32, min((i//32+1)*32, len(q['clip_id'])))})
    write(out/'state.json', dict(status='cache_frozen_b0', selected=len(all_ids), test_loaded=False))
    cache_base(data, base, device, {'train': canonical, 'validation': []})
    rows = []
    # Keep just selected coordinates in RAM. No repeated B0 forward or disk cache.
    for role, ids in selected.items():
        for sub in torch.tensor(ids, dtype=torch.long).split(32):
            b = q.batch(sub, 'cpu', keys=('motion', 'b0', 'valid', 'channel_mask', 'times'))
            for j, i in enumerate(sub.tolist()):
                n = int(q['_lengths'][i])
                x, times = b['b0'][j, :n].numpy().copy(), b['times'][j, :n].numpy().copy()
                mask = (b['valid'][j, :n, None]&b['channel_mask'][j, None]&base.motion_support.cpu()[None]).numpy().copy()
                if int(q['emotion_id'][i]) == neutral:
                    y = b['motion'][j, :n].numpy().copy()
                    kind = 'native_neutral'
                else:
                    nt, nv, nc = safe_target_batch(targets, [q['clip_id'][i]], n, 'cpu', native_times=b['times'][j:j+1, :n])
                    y = nt[0].numpy().copy()
                    mask &= (nv[0, :, None]&nc[0]).numpy()
                    kind = 'approved_neutral_pair'
                mask[:, 51] = False
                h = hashlib.sha256()
                for arr in (np.where(mask, y, 0.), mask, times):
                    h.update(np.ascontiguousarray(arr).tobytes())
                provenance = dict(clip_id=q['clip_id'][i], sentence_id=str(q['sentence_id'][i]),
                    speaker=str(int(q['speaker_id'][i])), role=role, target_kind=kind,
                    target_sha256=h.hexdigest(), emotion=int(q['emotion_id'][i]))
                rows.append(dict(x=x, y=y, mask=mask, times=times, provenance=provenance))
    fitter = NeutralAmplitudeFit()
    for row in rows:
        if row['provenance']['role'] == 'train':
            fitter.add(row['x'], row['y'], row['mask'], row['times'], provenance=row['provenance'])
    fit = fitter.solve()
    fit.update(neutral_checkpoint=binding['neutral_checkpoint'], safe_manifest_sha256=binding['safe_manifest_sha256'],
        implementation_git=binding['implementation_git'])
    write(out/'calibration.json', fit)
    write(out/'state.json', dict(status='score_held_candidate', fit_clips=len(fitter.rows), test_loaded=False))
    scores, groups = [], {}
    for row in rows:
        x, y, mask, times = (row[k] for k in ('x', 'y', 'mask', 'times'))
        pred = apply_calibration(x, fit)
        for view in ('raw', 'clip'):
            values = (x, pred) if view == 'raw' else (x.clip(0., 1.), pred.clip(0., 1.))
            # Clip only predictions, preserving the original target convention.
            value = dict(provenance=row['provenance'], view=view,
                base=measures(values[0], y, mask, times), candidate=measures(values[1], y, mask, times))
            scores.append(value)
            p = row['provenance']; key = '/'.join((p['role'], p['target_kind'], view))
            groups.setdefault(key, []).append(value)
            groups.setdefault(key+'/emotion_'+str(p['emotion']), []).append(value)
    summary = {key: dict(clips=len(part), **{mode: aggregate([v[mode] for v in part])
        for mode in ('base', 'candidate')}) for key, part in groups.items()}
    counts = {role: {kind: sum(r['provenance']['role'] == role and r['provenance']['target_kind'] == kind for r in rows)
        for kind in ('native_neutral', 'approved_neutral_pair')} for role in selected}
    assert state_digest(base.stage1) == digest
    report = dict(schema='neutral_amplitude_result_v2', calibration_schema=fit['schema'], smoke=a.smoke, counts=counts,
        summary=summary, gate=held_gate(summary), frozen_neutral_exact=True, neutral_state_sha256=digest,
        fit_sha256=sha(out/'calibration.json'), validation_queries_used=False, test_loaded=False,
        main_model_replaced=False, expression_student_loaded=False, neutral_amplitude_fit_performed=True,
        caveats=['Internal holds test only the added calibration; B0 saw original TRAIN',
            'No final generation claim before receiver adaptation and full development evaluation'])
    write(out/'per_clip.json', scores)
    write(out/'report.json', report)
    write(out/'state.json', dict(status='complete', candidate_passed=report['gate']['passed'],
        report_sha256=sha(out/'report.json'), test_loaded=False, main_model_replaced=False))
    print(json.dumps(dict(counts=counts, gate=report['gate'], output=str(out))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
