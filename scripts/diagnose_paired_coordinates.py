"""Frozen TRAIN paired-coordinate decomposition on original native masks.

Energy terms are not orthogonal: retain the cross term rather than claiming
each squared norm is a percentage of the total. No model fitting or dev query.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import configure, load_runtime, cache_base, development_fold, state_digest, sha, write
from scripts.train_full_staged import load_safe_native_targets, safe_target_batch
from scripts.audit_reference_style_swap import REGIONS


def coordinate_terms(motion, teacher, base, observed, times):
    """Return per-clip/channel energies under exactly the same observation set."""
    x, n, c = [np.asarray(a, np.float64) for a in (motion, teacher, base)]
    mask = np.asarray(observed, bool)
    times = np.asarray(times, np.float64)
    if not (x.shape == n.shape == c.shape == mask.shape) or x.ndim != 2:
        raise ValueError('Native coordinate/mask shape mismatch')
    if times.shape != (len(x),) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid native time axis')
    if not all(np.isfinite(a[mask]).all() for a in (x,n,c)):
        raise ValueError('Invalid observed native values')
    e, b = np.where(mask, x-n, 0.), np.where(mask, n-c, 0.)
    total = np.where(mask, x-c, 0.)
    np.testing.assert_allclose(e+b, total, atol=1e-12, rtol=1e-12)
    count = mask.sum(0)
    mean = lambda a: np.divide(np.where(mask,a,0.).sum(0), count, out=np.zeros(x.shape[1]), where=count>0)
    means = [mean(a) for a in (total,e,b)]
    adjacent = mask[1:] & mask[:-1] & np.isclose(np.diff(times),.04,rtol=1e-4,atol=1e-7)[:,None]
    def energy(a, v):
        cnt=v.sum(0)
        return np.divide(np.where(v,a,0.).sum(0),cnt,out=np.zeros(x.shape[1]),where=cnt>0)
    result = {'observed':count,'adjacent':adjacent.sum(0)}
    for name, arrays, valid in (
        ('position',(total,e,b),mask),
        ('centered',tuple(a-m[None] for a,m in zip((total,e,b),means)),mask),
        ('displacement',tuple(np.diff(a,axis=0) for a in (total,e,b)),adjacent)):
        r, a, z=arrays
        terms=[energy(v*v,valid) for v in (r,a,z)]+[2*energy(a*z,valid)]
        np.testing.assert_allclose(terms[0],terms[1]+terms[2]+terms[3],atol=1e-12,rtol=1e-10)
        result[name]=np.stack(terms)
    result['mean']=np.stack(means)
    return result


def aggregate(rows):
    if not rows:return {'clips':0}
    result={'clips':len(rows),'regions':{}}
    for region,indices in REGIONS.items():
        dst={}
        for kind in ('position','centered','displacement'):
            values=[]
            for row in rows:
                support=row['terms']['adjacent' if kind=='displacement' else 'observed'][indices]>0
                if support.any():values.append(row['terms'][kind][:,indices][:,support].mean(1))
            if values:
                av=np.mean(values,0)
                dst[kind]=dict(zip(('total','expression_difference','neutral_base_error','cross'),av.tolist()),clips=len(values))
        result['regions'][region]=dst
    return result


@torch.no_grad()
def run(a):
    configure(47);device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('Fresh diagnostic output required')
    out.mkdir(parents=True)
    binding=json.loads(Path(a.binding).read_text())
    for n,h in binding['source_files'].items():assert sha(Path(__file__).resolve().parents[1]/n)==h,n
    manifest=Path(a.safe_root)/'teacher_manifest_gated.jsonl'
    assert sha(manifest)==binding['safe_manifest_sha256']
    data,base=load_runtime(binding,device);neutral=state_digest(base.stage1)
    q=data['splits']['train'];fold=development_fold(q,data['fit_sids'])
    manifest_rows={r['source_clip_id']:r for r in map(json.loads,manifest.read_text().splitlines()) if r.get('teacher_eligible') is True}
    neutral_class=data['config']['data']['emotion_classes'].index('neutral')
    selected={role:[i for i in fold[role] if q['clip_id'][i] in manifest_rows and int(q['emotion_id'][i])!=neutral_class]
              for role in ('train','speaker_dev','sentence_dev')}
    if a.smoke:selected={k:v[:8] for k,v in selected.items()}
    ids=sorted(i for values in selected.values() for i in values)
    if not ids:raise ValueError('No approved emotional TRAIN pairs')
    targets=load_safe_native_targets(a.safe_root,[q['clip_id'][i] for i in ids])
    assert set(targets)=={q['clip_id'][i] for i in ids}
    # Cache canonical batches containing selected rows; preserve inference context.
    canonical=sorted(set(j for i in ids for j in range(i//32*32,min((i//32+1)*32,len(q['clip_id'])))))
    cache_base(data,base,device,{'train':canonical,'validation':[]})
    rows=[] # Physical coefficient energies, no fitted scale.
    for role,chosen in selected.items():
        for sub in torch.tensor(chosen,dtype=torch.long).split(32):
            b=q.batch(sub,device)
            teacher,valid,channel=safe_target_batch(targets,b['clip_id'],b['valid'].shape[1],device,native_times=b['times'])
            observed=valid[...,None]&channel&b['valid'][...,None]&b['channel_mask'][:,None]&base.motion_support
            observed[:,:,51]=False
            for j,i in enumerate(sub.tolist()):
                n=int(q['_lengths'][i]);cid=q['clip_id'][i]
                t=coordinate_terms(b['motion'][j,:n].cpu().numpy(),teacher[j,:n].cpu().numpy(),b['b0'][j,:n].cpu().numpy(),observed[j,:n].cpu().numpy(),b['times'][j,:n].cpu().numpy())
                rows.append(dict(index=i,clip_id=cid,role=role,speaker=int(b['speaker_id'][j]),emotion=int(b['emotion_id'][j]),terms=t))
        write(out/'state.json',dict(status='running',role=role,clips=len(rows),test_loaded=False))
        print(json.dumps(dict(role=role,clips=len(chosen))),flush=True)
    assert state_digest(base.stage1)==neutral
    assert sha(manifest)==binding['safe_manifest_sha256']
    summary={role:aggregate([r for r in rows if r['role']==role]) for role in selected}
    by_emotion={str(e):aggregate([r for r in rows if r['role']=='train' and r['emotion']==e]) for e in sorted({r['emotion'] for r in rows})}
    arrays={key:np.stack([r['terms'][key] for r in rows]) for key in rows[0]['terms']}
    np.savez_compressed(out/'terms.npz',**arrays,clip_ids=np.asarray([r['clip_id'] for r in rows]),roles=np.asarray([r['role'] for r in rows]))
    artifacts={cid:targets[cid][4] for cid in sorted(targets)}
    report=dict(schema='phase50_paired_native_coordinates_v1',summary=summary,by_emotion=by_emotion,
        test_loaded=False,validation_queries_used=False,training_performed=False,frozen_neutral_exact=True,
        manifest_sha256=sha(manifest),binding_sha256=sha(a.binding),artifacts=artifacts,
        neutral_checkpoint=binding['neutral_checkpoint'],data_manifest_sha256=binding['data_manifest_sha256'],
        source_sha256=sha(__file__),selected_counts={k:len(v) for k,v in selected.items()},
        mouth_mask='native_teacher_mask AND event_local_mask AND source_observation AND supported_channel',
        upper_mask='native_teacher_mask AND source_observation AND supported_channel',
        terms=['GT-B0','GT-aligned_neutral','aligned_neutral-B0','2*expression*base_error'],
        weighting='Clip-equal mean over supported channels and observed frames, physical coefficients squared.',
        limits=['Aligned neutral is an imperfect counterfactual; both expression and alignment error can enter GT-neutral.',
                'Components are nonorthogonal; squared energies are not causal percentages.',
                'This frozen descriptive diagnostic neither proves predictability nor fits a new model.'])
    write(out/'report.json',report);write(out/'state.json',dict(status='complete',report_sha256=sha(out/'report.json'),test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--safe-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
