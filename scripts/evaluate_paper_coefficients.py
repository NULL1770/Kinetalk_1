"""Descriptive ARKit52 paper tables from fixed completed evaluation archives.

These are coefficient-space development metrics, not official FLAME/MetaHuman
benchmarks. Missing reference values are never treated as ground truth.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch
from scripts.joint_motion_metrics import score_clip, summarize

UPPER = [41,42,43,44,45,5,6,12,13]
LIP = list(range(18,41))
MOUTH = list(range(14,41))
ARMS = {'base':None, 'prior':'prior_generation', 'audio':'audio_generation',
        'matched_static':'matched_static_generation', 'static':'audio_static',
        'reverse':'audio_reverse', 'mismatch':'audio_mismatch'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def coefficient_metrics(samples, target, mask, times):
    """Clip-equal metrics; average sample scores rather than sample trajectories."""
    x=np.asarray(samples,dtype=np.float64); y=np.asarray(target,dtype=np.float64)
    mask=np.asarray(mask); times=np.asarray(times,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=y.shape or y.ndim!=2 or y.shape[1]!=52 or len(x)<1:
        raise ValueError('Expected samples[S,T,52], target[T,52]')
    if mask.dtype!=bool or mask.shape!=y.shape or times.shape!=(len(y),):
        raise ValueError('Boolean channel mask and matching clock required')
    dt=np.diff(times)
    if not np.isfinite(times).all() or (dt<=0).any():raise ValueError('Strictly increasing finite clock required')
    if not np.isfinite(y[mask]).all() or not np.isfinite(x[:,mask]).all():raise ValueError('Observed values must be finite')
    result={}
    for name,channels in [('lip',LIP),('mouth',MOUTH),('upper',UPPER),('all52',list(range(52)))]:
        support=mask[:,channels]; error=np.where(support[None],x[:,:,channels]-y[None,:,channels],0.)
        if support.any():
            result[name+'_mae']=float(np.abs(error).sum()/len(x)/support.sum())
            result[name+'_rmse']=float(np.sqrt(np.square(error).sum((1,2))/support.sum()).mean())
        else:result[name+'_mae']=result[name+'_rmse']=None
        if name in ('mouth','upper'):
            pair=support[1:]&support[:-1]
            e=np.diff(x[:,:,channels],axis=1)-np.diff(y[:,channels],axis=0)[None]
            velocity=np.where(pair[None],e/dt[None,:,None],0.)
            result[name+'_velocity_mae_per_second']=float(np.abs(velocity).sum()/len(x)/pair.sum()) if pair.any() else None
    gaps=[]; signed=[]
    for c in UPPER:
        valid=mask[:,c]
        if valid.sum()<2:continue
        gap=y[valid,c].std(ddof=0)-x[:,valid,c].std(axis=1,ddof=0)
        gaps.extend(np.abs(gap).tolist());signed.extend(gap.tolist())
    result['upper_std_absolute_gap']=float(np.mean(gaps)) if gaps else None
    result['upper_std_signed_gt_minus_pred']=float(np.mean(signed)) if signed else None
    joint=mask[:,UPPER].all(1)
    if joint.any():
        intensity=np.abs(x[:,joint][:,:,UPPER]).mean(-1)-np.abs(y[joint][:,UPPER]).mean(-1)[None]
        result['upper_intensity_mae']=float(np.abs(intensity).mean())
        spread=[np.sqrt(np.square(x[a,joint][:,UPPER]-x[b,joint][:,UPPER]).mean())
                for a in range(len(x)) for b in range(a+1,len(x))]
        result['upper_pairwise_rms_diversity']=float(np.mean(spread)) if spread else 0.
    else:result['upper_intensity_mae']=result['upper_pairwise_rms_diversity']=None
    result['raw_upper_oob_fraction']=float(((x[:,:,UPPER]<0)|(x[:,:,UPPER]>1))[:,mask[:,UPPER]].mean()) if mask[:,UPPER].any() else None
    return result

def cluster_interval(values, sentences, seed=20260919, draws=4096):
    good=[(v,s) for v,s in zip(values,sentences) if v is not None and np.isfinite(v)]
    if not good:return {'mean':None,'ci95':None,'n':0,'sentences':0}
    values=np.array([v for v,s in good]);sentences=np.array([s for v,s in good]);groups=sorted(set(sentences))
    result={'mean':float(values.mean()),'ci95':None,'n':len(values),'sentences':len(groups)}
    if len(groups)>1:
        sums=np.array([values[sentences==g].sum() for g in groups]);counts=np.array([(sentences==g).sum() for g in groups])
        pick=np.random.default_rng(seed).integers(0,len(groups),(draws,len(groups)))
        result['ci95']=np.quantile(sums[pick].sum(1)/counts[pick].sum(1),[.025,.975]).tolist()
    return result

def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(root, output, reference):
    raw=torch.load(reference,weights_only=False,map_location="cpu")
    refs=raw["clips"]
    output.mkdir(parents=True,exist_ok=True)
    anchor=root/'outer/audio_generation/holdout'
    main=json.loads((anchor/'result.json').read_text(encoding='utf8'))
    metadata={r['clip_id']:r for r in main['per_clip_scores']}
    ids=sorted(metadata);rows=[];sources={};distribution=[];distribution_per_clip=[]
    if set(ids)!=set(refs):raise ValueError('Raw reference membership differs')
    for arm,stage in ARMS.items():
        directory=anchor if stage is None else root/'outer'/stage/'holdout'
        result=json.loads((directory/'result.json').read_text(encoding='utf8'))
        if set(result['selected_clip_ids'])-set(ids):raise ValueError('Unexpected selected member')
        if {r['clip_id'] for r in result['per_clip_scores']}!=set(ids):raise ValueError('Arm membership differs')
        sources[arm]={'result_sha256':sha(directory/'result.json'),'npz':{}}
        joint=[]
        for cid in ids:
            path=directory/'npz'/f'{cid}.npz'; sources[arm]['npz'][cid]=sha(path)
            with np.load(path,allow_pickle=False) as f, np.load(anchor/'npz'/f'{cid}.npz',allow_pickle=False) as a:
                for key in ('times','native_valid','reference_display_baseline_filled','channels'):
                    np.testing.assert_array_equal(f[key],a[key])
                for i in (0,1):np.testing.assert_array_equal(f['motions'][i],a['motions'][i])
                samples=f['motions'][1:2] if stage is None else f['motions'][2:]
                if stage and len(samples)!=4:raise ValueError('Expected all four fixed draws')
                r=refs[cid];target=np.asarray(r['target52']);native=np.asarray(r['valid'],dtype=bool);channel=np.asarray(r['channel_mask'],dtype=bool)
                np.testing.assert_array_equal(f['native_valid'],native)
                np.testing.assert_allclose(f['times'],np.asarray(r['times']),atol=0,rtol=0)
                np.testing.assert_array_equal(f['motions'][1],np.asarray(r['baseline52']))
                support=native[:,None]&channel&np.isfinite(target)
                available=support&~f['reference_display_baseline_filled']
                np.testing.assert_allclose(f['motions'][0][available],target[available],atol=0,rtol=0)
                metrics=coefficient_metrics(samples,target,support,f['times'])
                samples9=samples[:,:,UPPER]
                if len(samples9)==1:samples9=np.repeat(samples9,2,axis=0)
                j=score_clip(samples9,target[:,UPPER],native&channel[:,UPPER].all(1),main['summary']['scales']);joint.append(j)
                distribution_per_clip.append(dict(arm=arm,clip_id=cid,sentence=metadata[cid]['sentence'],centered_fair_es=j['joint_fair_es']['centered'],raw_fair_es=j['joint_fair_es']['raw']))
                m=metadata[cid]
                rows.append(dict(arm=arm,clip_id=cid,sentence=m['sentence'],speaker=str(m['speaker']),
                                 emotion=cid.split('_')[2],samples=len(samples),frames=len(f['times']),
                                 observed_channel_values=int(support.sum()),**metrics))
        s=summarize(joint)
        if stage is not None:
            for k in ('raw','centered'):np.testing.assert_allclose(s['joint_fair_es'][k],result['summary']['joint_fair_es'][k],atol=1e-12,rtol=1e-10)
        distribution.append({'arm':arm,'centered_fair_es':s['joint_fair_es']['centered'],
            'raw_fair_es':s['joint_fair_es']['raw'],'variogram':s['variogram']['aggregate'],
            'speed_ratio':s['speed']['all']['rms']/s['speed']['reference_all']['rms'],
            **{f'{g}_rms_ratio':v for g,v in zip(('up','down','squint','wide'),s['rms_ratio'])}})
    metrics=list(coefficient_metrics(np.zeros((1,2,52)),np.zeros((2,52)),np.ones((2,52),bool),np.array([0,.04])))
    summaries={};flat=[];groups=[];paired=[]
    for arm in ARMS:
        sub=[r for r in rows if r['arm']==arm]; summaries[arm]={}
        for metric in metrics:
            value=cluster_interval([r[metric] for r in sub],[r['sentence'] for r in sub]);summaries[arm][metric]=value
            flat.append({'arm':arm,'metric':metric,**{k:v for k,v in value.items() if k!='ci95'},
                         'ci95_low':value['ci95'][0] if value['ci95'] else None,'ci95_high':value['ci95'][1] if value['ci95'] else None})
        for key in ('emotion','speaker'):
            for group in sorted({r[key] for r in sub}):
                selected=[r for r in sub if r[key]==group]
                groups.append({'arm':arm,'group_type':key,'group':group,'clips':len(selected),
                               **{m:float(np.mean([r[m] for r in selected if r[m] is not None])) for m in metrics}})
    by_arm={arm:{r['clip_id']:r for r in rows if r['arm']==arm} for arm in ARMS}
    for candidate,control in [('prior','base'),('audio','prior'),('audio','matched_static'),('audio','static'),('audio','reverse'),('audio','mismatch')]:
        for metric in metrics:
            values=[by_arm[candidate][cid][metric]-by_arm[control][cid][metric]
                    if by_arm[candidate][cid][metric] is not None and by_arm[control][cid][metric] is not None else None for cid in ids]
            v=cluster_interval(values,[metadata[cid]['sentence'] for cid in ids])
            paired.append({'candidate':candidate,'control':control,'metric':metric,'mean_delta':v['mean'],
                           'ci95_low':v['ci95'][0] if v['ci95'] else None,'ci95_high':v['ci95'][1] if v['ci95'] else None})
    result={'schema':'arkit_paper_development_metrics_v2_raw_reference','reference_sha256':sha(reference),'dataset_sha256':raw['dataset_sha256'],'scope':'historically exposed development diagnostics; not final test',
        'clips':len(ids),'sentence_count':len({m['sentence'] for m in metadata.values()}),'summaries':summaries,
        'protocol':'docs/PAPER_EVALUATION_PROTOCOL_20260919.md','official_benchmark_reproduced':False,
        'aggregation':'equal clip weight; average per-draw metrics; 4096-resample sentence-cluster CI',
        'sources':sources,'source_code_sha256':sha(__file__),'distribution_metrics':distribution}
    (output/'coefficient_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    write_csv(output/'distribution_per_clip.csv',distribution_per_clip);write_csv(output/'per_clip_metrics.csv',rows);write_csv(output/'coefficient_summary.csv',flat)
    write_csv(output/'subgroup_metrics.csv',groups);write_csv(output/'paired_ablations.csv',paired);write_csv(output/'distribution_summary.csv',distribution)
    columns=['lip_mae','mouth_mae','upper_mae','upper_std_absolute_gap','upper_intensity_mae','upper_velocity_mae_per_second','upper_pairwise_rms_diversity']
    lines=['# Current model: development coefficient results','',result['scope'], '',
           '| Arm | '+' | '.join(columns)+' |','|---|'+'---:|'*len(columns)]
    latex=['\\begin{tabular}{l'+'r'*len(columns)+'}','\\toprule','Arm & '+' & '.join(c.replace('_','\\_') for c in columns)+' \\\\','\\midrule']
    for arm in ARMS:
        vals=[f'{summaries[arm][c]["mean"]:.6f}' for c in columns]
        lines.append('| '+arm+' | '+' | '.join(vals)+' |');latex.append(arm.replace('_','\\_')+' & '+' & '.join(vals)+' \\\\')
    latex+=['\\bottomrule','\\end{tabular}'];(output/'table_coefficients.tex').write_text('\n'.join(latex),encoding='utf8')
    lines+=['','All units are native coefficient units (velocity per second). Intensity is mean absolute upper9 coefficient. Not official MLE/MEE/EIE/FRD/FDD/LVE. Lip preservation is not independent lip-sync validation. Signed std gap is diagnostic, not lower-is-better. Full95% intervals and paired differences are in CSV.']
    (output/'coefficient_results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'output':str(output.resolve()),'clips':len(ids),'arms':len(ARMS)}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    a=parser.parse_args();run(a.root,a.output,a.reference)
