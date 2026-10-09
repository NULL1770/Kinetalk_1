"""Frozen safe-pair target sensitivity audit; no new target or model fitting."""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import configure, load_runtime, development_fold, state_digest, sha, write
from scripts.train_full_staged import load_safe_native_targets
from scripts.audit_reference_style_swap import REGIONS

WINDOWS = (1, 5, 11)  # Fixed 40/200/440ms sample supports, no sweep/selection.


def links(times):
    return np.isclose(np.diff(times), .04, rtol=1e-4, atol=1e-7)


def check(x, mask, times):
    if x.ndim != 2 or x.shape != mask.shape or mask.dtype != bool:
        raise ValueError('Native channel mask shape/type mismatch')
    if times.shape != (len(x),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid native times')
    if not np.isfinite(x[mask]).all():
        raise ValueError('Nonfinite observed target')


def window_moments(x, mask, times, width):
    """Require every frame/channel and native interval; never pad or bridge."""
    check(x, mask, times)
    if width < 1 or width % 2 != 1:
        raise ValueError('Positive odd window required')
    mean, std = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
    valid = np.zeros_like(mask)
    if width > len(x):
        return mean, std, valid
    z = np.where(mask, x, 0.).astype(np.float64)
    acc = lambda v: np.concatenate((np.zeros((1,v.shape[1]),dtype=v.dtype), np.cumsum(v,axis=0)),axis=0)
    sums, squares, counts = acc(z), acc(z*z), acc(mask.astype(int))
    native = np.r_[0, np.cumsum(~links(times))]
    start = np.arange(len(x)-width+1); end = start+width; center = start+width//2
    ok = (counts[end]-counts[start] == width) & ((native[end-1]-native[start]) == 0)[:,None]
    mu = (sums[end]-sums[start])/width
    variance = np.maximum((squares[end]-squares[start])/width-mu*mu,0.)
    mean[center] = np.where(ok,mu,0.)
    std[center] = np.where(ok,np.sqrt(variance),0.)
    valid[center] = ok
    return mean, std, valid


def centered(x, mask):
    count = mask.sum(0)
    mu = np.divide(np.where(mask,x,0.).sum(0),count,out=np.zeros(x.shape[1]),where=count>0)
    return np.where(mask,x-mu[None],0.)


def mean_square(x, mask):
    count = mask.sum(0); active = count > 0
    if not active.any():
        return None
    per = np.divide(np.where(mask,x*x,0.).sum(0),count,out=np.zeros(x.shape[1]),where=active)
    return float(per[active].mean())


def ratio(numerator, denominator):
    return numerator/denominator if numerator is not None and denominator is not None and denominator>1e-12 else None


def rank_corr(x, y):
    """Spearman correlation with average tie ranks; null for constant input."""
    x,y = np.asarray(x,float),np.asarray(y,float)
    if x.shape != y.shape or len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    def ranks(a):
        _,inverse,count = np.unique(a,return_inverse=True,return_counts=True)
        ranks = np.cumsum(count)-count+(count-1)/2.
        return ranks[inverse]
    rx,ry = ranks(x),ranks(y)
    if np.std(rx)==0 or np.std(ry)==0:
        return None
    return float(np.corrcoef(rx,ry)[0,1])


def target_metrics(query, teacher, mask, times):
    """All comparisons use identical observed support, never zero fill as GT."""
    check(query,mask,times);check(teacher,mask,times)
    query,teacher = query.astype(np.float64),teacher.astype(np.float64)
    diff = np.where(mask,query-teacher,0.)
    adjacency = mask[1:] & mask[:-1] & links(times)[:,None]
    result = dict(native_observations=int(mask.sum()),native_frames=int(mask.any(1).sum()),
                  native_fraction=float(mask.mean()),centered_energy=mean_square(centered(diff,mask),mask),
                  displacement_energy=mean_square(np.diff(diff,axis=0),adjacency),
                  adjacent_observations=int(adjacency.sum()))
    result['roughness_ratio'] = ratio(result['displacement_energy'],result['centered_energy'])
    # A perturbation stress test, NOT an empirical alignment error estimate.
    common = np.zeros_like(mask)
    if len(mask)>2:
        common[1:-1] = mask[:-2]&mask[1:-1]&mask[2:]&(links(times)[:-1]&links(times)[1:])[:,None]
    variants = [diff]
    for offset in (-1,1):
        shifted = np.zeros_like(teacher)
        shifted[1:-1] = teacher[1+offset:len(teacher)-1+offset]
        variants.append(np.where(common,query-shifted,0.))
    for width in WINDOWS:
        smooth, envelope, supported = window_moments(diff,mask,times,width)
        signal = mean_square(centered(diff,supported),supported)
        smooth_signal = mean_square(centered(smooth,supported),supported)
        residual = mean_square(diff-smooth,supported)
        products = [window_moments(v,common,times,width) for v in variants]
        shared = products[0][2]
        assert all(np.array_equal(v[2],shared) for v in products)
        e0=products[0][0];sig=mean_square(centered(e0,shared),shared)
        adjacent=shared[1:]&shared[:-1]&links(times)[:,None]
        sig_d=mean_square(np.diff(e0,axis=0),adjacent)
        errors=[mean_square(v[0]-e0,shared) for v in products[1:]]
        errors_d=[mean_square(np.diff(v[0]-e0,axis=0),adjacent) for v in products[1:]]
        env_errors=[mean_square(v[1]-products[0][1],shared) for v in products[1:]]
        av=lambda v:float(np.mean(v)) if all(x is not None for x in v) else None
        result['w'+str(width)] = dict(
            observations=int(supported.sum()),coverage_of_native=ratio(float(supported.sum()),float(mask.sum())),
            centered_energy_same_support=signal,smoothed_centered_energy=smooth_signal,
            centered_energy_ratio=ratio(smooth_signal,signal),highpass_energy=residual,
            # Finite-window residual and smooth are not orthogonal bands.
            shift_observations=int(shared.sum()),shift_coverage_of_native=ratio(float(shared.sum()),float(mask.sum())),
            shift_signal_energy=sig,shift_displacement_signal_energy=sig_d,
            shift_error=av(errors),shift_displacement_error=av(errors_d),
            shift_error_ratio=ratio(av(errors),sig),shift_displacement_error_ratio=ratio(av(errors_d),sig_d),
            local_std_energy=mean_square(envelope,supported),
            shift_local_std_error=av(env_errors),
            shift_local_std_error_ratio=ratio(av(env_errors),mean_square(products[0][1],shared)))
    return result


def flatten(value,prefix=''):
    out={}
    for k,v in value.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flatten(v,key+'/'))
        else:out[key]=v
    return out


def summarize(records):
    if not records:return {'n':0}
    flat=[flatten(r) for r in records]
    stats={}
    for key in flat[0]:
        values=[r[key] for r in flat if r[key] is not None]
        stats[key]=dict(n=len(values),mean=float(np.mean(values)) if values else None,
                       median=float(np.median(values)) if values else None)
    return dict(n=len(records),stats=stats)


def run(a):
    configure(47);out=Path(a.output)
    if out.exists():raise FileExistsError('Fresh audit output required')
    out.mkdir(parents=True)
    binding=json.loads(Path(a.binding).read_text())
    source=Path(__file__).resolve().parents[1]
    for name,digest in binding['source_files'].items():assert sha(source/name)==digest,name
    manifest=Path(a.safe_root)/'teacher_manifest_gated.jsonl'
    assert sha(manifest)==binding['safe_manifest_sha256']
    data,base=load_runtime(binding,torch.device('cpu'));before=state_digest(base.stage1)
    q=data['splits']['train'];fold=development_fold(q,data['fit_sids'])
    rows_manifest={r['source_clip_id']:r for r in map(json.loads,manifest.read_text().splitlines()) if r.get('teacher_eligible') is True}
    neutral=data['config']['data']['emotion_classes'].index('neutral')
    selected={role:[i for i in fold[role] if q['clip_id'][i] in rows_manifest and int(q['emotion_id'][i])!=neutral]
              for role in ('train','speaker_dev','sentence_dev')}
    if a.smoke:selected={k:v[:8] for k,v in selected.items()}
    else:assert {k:len(v) for k,v in selected.items()}==dict(train=2024,speaker_dev=281,sentence_dev=278)
    ids=[i for values in selected.values() for i in values]
    targets=load_safe_native_targets(a.safe_root,[q['clip_id'][i] for i in ids])
    expected=json.loads(Path(a.prior_artifacts).read_text())
    assert all(targets[cid][4]==expected[cid] for cid in targets)
    rows=[];missing_b=0
    for role,chosen in selected.items():
        for sub in torch.tensor(chosen,dtype=torch.long).split(32):
            b=q.batch(sub,'cpu',keys=('motion','valid','times','channel_mask','speaker_id','emotion_id','clip_id'))
            for j,i in enumerate(sub.tolist()):
                cid=q['clip_id'][i];teacher,tvalid,tchannel,ttimes,artifact=targets[cid]
                n=int(q['_lengths'][i]);times=b['times'][j,:n].numpy()
                np.testing.assert_allclose(times,ttimes.numpy(),atol=1e-7,rtol=0)
                mask=(tvalid.numpy()&b['valid'][j,:n].numpy())[:,None]&tchannel.numpy()&b['channel_mask'][j].numpy()[None]&base.motion_support.numpy()[None]
                mask[:,51]=False
                row=rows_manifest[cid];alt=Path(row['audit_artifact_b']) if row.get('audit_artifact_b') else None
                missing_b+=int(alt is None or not alt.is_file())
                motion=b['motion'][j,:n].numpy();t=teacher.numpy()
                records={region:target_metrics(motion[:,ix],t[:,ix],mask[:,ix],times) for region,ix in REGIONS.items()}
                rows.append(dict(clip_id=cid,index=i,role=role,speaker=int(b['speaker_id'][j]),emotion=int(b['emotion_id'][j]),
                    path_p95_error_ms=row.get('path_p95_error_ms'),mouth_event_agreement=row.get('mouth_event_agreement'),
                    mouth_event_gate=row.get('mouth_event_gate'),metrics=records))
            write(out/'state.json',dict(status='running',role=role,clips=len(rows),test_loaded=False))
        print(json.dumps(dict(role=role,clips=len(chosen))),flush=True)
    by_role={};correlations={}
    for role in selected:
        subset=[r for r in rows if r['role']==role]
        cohorts={'all':subset}
        cohorts.update({'emotion_'+str(e):[r for r in subset if r['emotion']==e] for e in sorted({r['emotion'] for r in subset})})
        # Fixed disjoint categories, not derived from results; retain both gates.
        cohorts.update({'event_gate_'+str(v):[r for r in subset if bool(r['mouth_event_gate'])==v] for v in (False,True)})
        for key,cuts in [('path_p95_error_ms',(40.,60.)),('mouth_event_agreement',(.5,.75))]:
            for low,high,name in [(-np.inf,cuts[0],'low'),(cuts[0],cuts[1],'mid'),(cuts[1],np.inf,'high')]:
                cohorts[key+'_'+name]=[r for r in subset if r[key] is not None and low<=round(r[key],6)<high]
        by_role[role]={name:{reg:summarize([r['metrics'][reg] for r in group]) for reg in REGIONS} for name,group in cohorts.items()}
        correlations[role]={}
        for reg in REGIONS:
            flat=[dict(flatten(r['metrics'][reg]),path_p95_error_ms=r['path_p95_error_ms'],
                       mouth_event_agreement=r['mouth_event_agreement']) for r in subset]
            correlations[role][reg]={}
            for metric in ('roughness_ratio','w1/shift_error_ratio','w5/shift_error_ratio','w11/shift_error_ratio'):
                correlations[role][reg][metric]={}
                for quality in ('path_p95_error_ms','mouth_event_agreement'):
                    pairs=[(r[metric],r[quality]) for r in flat if r[metric] is not None and r[quality] is not None]
                    correlations[role][reg][metric][quality]=dict(n=len(pairs),
                        spearman=rank_corr([x for x,y in pairs],[y for x,y in pairs]))
    assert sha(manifest)==binding['safe_manifest_sha256'] and state_digest(base.stage1)==before
    with gzip.open(out/'per_clip.json.gz','wt',encoding='utf8') as stream:
        json.dump(rows,stream,allow_nan=False,separators=(',',':'))
    write(out/'pair_artifacts.json',{cid:targets[cid][4] for cid in sorted(targets)})
    report=dict(schema='phase56_safe_pair_sensitivity_v2',selected_counts={k:len(v) for k,v in selected.items()},
        windows=WINDOWS,summary=by_role,correlations=correlations,missing_second_alignment_artifacts=missing_b,
        original_pair_hashes_exact=True,frozen_neutral_exact=True,neural_training_performed=False,
        query_audio_read=False,validation_queries_used=False,test_loaded=False,
        safe_manifest_sha256=sha(manifest),binding_sha256=sha(a.binding),source_sha256=sha(__file__),
        limits=['Fixed plus/minus one-frame sensitivity is not measured alignment error or target correctness.',
                'Approved pairs were selected using path agreement; associations are selected and not causal.',
                'Window comparisons have different coverage; all energy ratios use the same support within a comparison.',
                'Residual and moving average are not orthogonal frequency components; no pure emotion ground truth claim.',
                'No student/predictability model fitted, no target/channels changed or proposed smoothing selected.'])
    write(out/'report.json',report)
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<32*2**20,size
    write(out/'state.json',dict(status='complete',report_sha256=sha(out/'report.json'),bytes=size,test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('binding','safe-root','output','prior-artifacts'):p.add_argument('--'+name,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:run(args)
    except Exception as exc:
        if not isinstance(exc,FileExistsError) and Path(args.output).is_dir():
            write(Path(args.output)/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
