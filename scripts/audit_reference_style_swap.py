"""Frozen reference stability, target-direction and timing audit; never trains.

Same source audio/B0/g/u, all three development enrollment identities and own
A/B. Cross-person GT is compared by native clip statistics, never by retiming
or framewise scores. A visible difference alone is not identity correctness.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.static_expression import StaticExpressionCorrection,receiver_features
from scripts.train_expression_response import (configure,load_runtime,cache_base,reference_batch,
    audit_data,development_fold,state_digest,sha,write)
from scripts.refine_expression_prior import restore_model
from scripts.evaluate_expression_response import finite,aggregate
from scripts.audit_matched_motion_curves import longest_valid_span
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

REGIONS={'all51':list(range(51)),'mouth':list(range(14,41)),'brows':list(range(41,46)),'eyes':list(range(14))}
STATS=('mean','centered_rms','q90_q10','displacement_rms')

def matched_pairs(sentence,emotion,intensity,speaker):
    """Exact label/text-key match, reject ambiguous same-person repetitions."""
    groups=defaultdict(list)
    for i,(a,b,c) in enumerate(zip(sentence,emotion,intensity)):
        groups[(str(a),int(b),int(c))].append(i)
    result=[]
    for ids in groups.values():
        people=[int(speaker[i]) for i in ids]
        if len(set(people))!=len(people):raise ValueError('Ambiguous matched-person group')
        result.extend((i,j) for i in ids for j in ids if i!=j)
    return result,groups

def adjacent(valid,times):
    return valid[1:]&valid[:-1]&np.isclose(times[1:]-times[:-1],.04,rtol=1e-4,atol=1e-7)

def summary(x,valid,channels,times):
    x=np.asarray(x,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    channels=np.asarray(channels,dtype=bool);times=np.asarray(times,dtype=np.float64)
    if x.shape!=(len(valid),52) or times.shape!=valid.shape or channels.shape!=(52,) or not valid.any():
        raise ValueError('Invalid native statistics inputs')
    if not np.isfinite(x[valid][:,channels]).all():raise ValueError('Nonfinite observed motion')
    out=np.zeros((4,52),dtype=np.float64);observed=x[valid][:,channels]
    out[0,channels]=observed.mean(0)
    out[1,channels]=np.sqrt(np.mean((observed-observed.mean(0))**2,axis=0))
    out[2,channels]=np.quantile(observed,.9,axis=0)-np.quantile(observed,.1,axis=0)
    a=adjacent(valid,times)
    if a.any():out[3,channels]=np.sqrt(np.mean((x[1:]-x[:-1])[a][:,channels]**2,axis=0))
    return out

def correlation(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    den=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return float(np.sum(a*b)/den) if den>1e-12 else None

def change(a,b,valid,channels,times):
    """Native changes relative to another prediction, not phoneme accuracy."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool);channels=np.asarray(channels,dtype=bool)
    adj=adjacent(valid,times);result={}
    for name,ix in REGIONS.items():
        ix=[j for j in ix if channels[j]]
        if not ix:continue
        x,y=a[valid][:,ix],b[valid][:,ix]
        xc,yc=x-x.mean(0),y-y.mean(0)
        result[name+'/mae']=float(np.mean(np.abs(x-y)))
        result[name+'/mean_delta_rms']=float(np.sqrt(np.mean((x.mean(0)-y.mean(0))**2)))
        result[name+'/centered_delta_rms']=float(np.sqrt(np.mean((xc-yc)**2)))
        result[name+'/centered_correlation']=correlation(xc,yc)
        result[name+'/velocity_correlation']=correlation((a[1:]-a[:-1])[adj][:,ix],(b[1:]-b[:-1])[adj][:,ix]) if adj.any() else None
    result['jaw_closure_disagreement']=float(np.mean((a[valid,17]<.05)!=(b[valid,17]<.05)))
    lags=[]
    for lag in range(-5,6):
        left=np.arange(max(0,-lag),min(len(valid),len(valid)-lag))
        right=left+lag;m=valid[left]&valid[right]
        m &= np.isclose(np.asarray(times)[right]-np.asarray(times)[left],lag*.04,rtol=1e-4,atol=1e-7)
        if m.sum()<2:continue
        x,y=a[left[m],17],b[right[m],17]
        c=correlation(x-x.mean(),y-y.mean())
        if c is not None:lags.append((c,lag))
    # Tie prefers zero, then the smallest absolute lag; descriptive only.
    result['jaw_best_lag_frames']=min(lags,key=lambda v:(-v[0],abs(v[1]),v[1]))[1] if lags else None
    return result

def target_direction(own,swapped,source_gt,donor_gt,source_anchor,donor_anchor,channels,scales):
    """Native statistics of separate performances; no frame correspondences."""
    result={}
    for region,ix in REGIONS.items():
        ix=[j for j in ix if channels[j]]
        if not ix:continue
        s=np.asarray(scales)[ix]
        for k,name in enumerate(STATS):
            before=np.mean(((own[k,ix]-donor_gt[k,ix])/s)**2)
            after=np.mean(((swapped[k,ix]-donor_gt[k,ix])/s)**2)
            result[f'{region}/{name}_before']=float(before)
            result[f'{region}/{name}_after']=float(after)
            result[f'{region}/{name}_improvement']=float(before-after)
            result[f'{region}/{name}_improved']=float(after<before)
        real=(donor_gt[0,ix]-source_gt[0,ix])/s
        predicted=(swapped[0,ix]-own[0,ix])/s
        result[region+'/mean_delta_cosine']=correlation(real,predicted)
        gt_expr=(donor_gt[0,ix]-donor_anchor[ix])/s
        before=(own[0,ix]-source_anchor[ix])/s
        after=(swapped[0,ix]-donor_anchor[ix])/s
        result[region+'/neutral_relative_mean_before']=float(np.mean((before-gt_expr)**2))
        result[region+'/neutral_relative_mean_after']=float(np.mean((after-gt_expr)**2))
    return result

def export(path,motions,valid,times,channels,clip_id,mode_names,reference_ids):
    start,stop=longest_valid_span(valid);sl=slice(start,stop)
    if not np.isclose(np.diff(times[sl]),.04,rtol=1e-5,atol=1e-7).all():
        raise ValueError('Non-native display clock')
    np.savez_compressed(path,channels=np.asarray(ARKIT_NAMES),motions=np.asarray(motions)[:,sl],
        valid=np.asarray(valid)[sl],times=np.asarray(times)[sl],channel_mask=channels,
        mode_names=np.asarray(mode_names),clip_id=np.asarray(clip_id),
        source_start_frame=start,source_stop_frame=stop,noise_seed=-1,
        reference_clip_ids=np.asarray(reference_ids),query_gt_used_for_deployment=False)
    return dict(clip_id=clip_id,frames=stop-start,native_start_frame=start,native_stop_frame=stop,sha256=sha(path))

def refs_for(data,b,device,sid):
    other=dict(b,speaker_id=torch.full_like(b['speaker_id'],int(sid)))
    return reference_batch(data,other,device)

def compute(model,correction,b,p,refs):
    s=model.encode_style(refs)['code']
    parent=model.decode(b['b0'],p,s,b['valid'])
    latent=correction(parent,receiver_features(model,p,s,refs,'latent'),b['valid'],model.scales)
    return {'parent':parent,'latent':latent},s

@torch.no_grad()
def run(a):
    configure(47);device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('Existing diagnostic; never rerun')
    out.mkdir(parents=True);(out/'render_inputs').mkdir()
    binding=json.loads(Path(a.binding).read_text(encoding='utf-8-sig'))
    for name,digest in binding['source_files'].items():
        assert sha(Path(__file__).resolve().parents[1]/name)==digest,name
    model,_=restore_model(binding,device);model.eval().requires_grad_(False)
    ck=torch.load(a.correction,map_location=device,weights_only=False)
    assert ck['mode']=='latent' and ck['parent_checkpoint_sha256']==binding['parent_checkpoint']['sha256']
    assert ck['data_manifest_sha256']==binding['data_manifest_sha256'] and not ck['test_loaded']
    correction=StaticExpressionCorrection('latent',**ck['state']).to(device).eval().requires_grad_(False)
    data,base=load_runtime(binding,device);q=data['splits']['validation']
    audit=audit_data(data,development_fold(data['splits']['train'],data['fit_sids']))
    people=sorted(data['dev_sids']);assert len(people)==3
    query_ids=set(q['clip_id'])|set(data['splits']['train']['clip_id'])
    refs_meta={}
    for sid in people:
        r=data['refs'][sid]
        assert len(r['clip_id'])==2 and not query_ids.intersection(r['clip_id'])
        assert all('_neutral_' in cid for cid in r['clip_id'])
        refs_meta[sid]={'speaker':r['speaker'][0],'clip_ids':r['clip_id'],'sentences':r['sentence_id']}
    pairs,groups=matched_pairs(q['sentence_id'],q['emotion_id'],q['intensity_id'],q['speaker_id'])
    ids=list(range(len(q['valid'])))
    if a.smoke:
        ids=[next(i for i in ids if int(q['speaker_id'][i])==sid and int(q['emotion_id'][i])==e)
             for sid in people for e in range(8)]
    # Always cache validation B0 in canonical full order, even for the
    # 24-clip smoke. The saved parent curves were produced from this exact
    # order; subset/reordered caching can change reduction widths and create
    # needless false failures in the bit-exact gate.
    cache_base(data,base,device,{'train':[],'validation':list(range(len(q['valid'])))})
    before=(state_digest(model),state_digest(base.stage1),state_digest(correction))
    saved={}
    for method,path in [('parent',a.parent_curves),('latent',a.latent_curves)]:
        assert sha(path)==binding['style_audit']['baseline_curves'][method]['sha256']
        z=torch.load(path,map_location='cpu',weights_only=False)
        assert z['clip_id']==q['clip_id'] and not z['test_loaded'];saved[method]=z['predictions']
    n=len(q['valid']);methods=('parent','latent');policies=('raw','clip_all')
    pred_stats=np.zeros((2,2,n,3,4,52));gt_stats=np.zeros((n,4,52))
    changes={};maximum={'parent':0.,'latent':0.,'base':0.};display={};style_rows=[]
    display_lookup={cid:name for name,cid in binding['display_clips'].items()}
    processed=0
    for sub in torch.tensor(ids).split(16):
        b=q.batch(sub,device);refs=reference_batch(data,b,device)
        p=model.audio_prior(b['audio_features'],b['valid']);p_before={k:x.clone() for k,x in p.items()}
        normal,own_style=compute(model,correction,b,p,refs)
        own_views={}
        for label,part in [('A',{k:x[:,:1] for k,x in refs.items()}),('B',{k:x[:,1:2] for k,x in refs.items()})]:
            own_views[label],_=compute(model,correction,b,p,part)
        swapped={}
        for sid in people:
            swapped[sid],_=compute(model,correction,b,p,refs_for(data,b,device,sid))
        assert all(torch.equal(p[k],x) for k,x in p_before.items())
        for j,i in enumerate(sub.tolist()):
            length=int(q['_lengths'][i]);v=b['valid'][j,:length].cpu().numpy()
            ch=b['channel_mask'][j].cpu().numpy();tm=b['times'][j,:length].cpu().numpy()
            gt=b['motion'][j,:length].cpu().numpy();sid=int(q['speaker_id'][i]);sindex=people.index(sid)
            gt_stats[i]=summary(gt,v,ch,tm)
            torch.testing.assert_close(b['b0'][j,:length].cpu(),saved['parent'][i]['neutral_b0'],rtol=0,atol=0)
            maximum['base']=max(maximum['base'],float((b['b0'][j,:length].cpu()-saved['parent'][i]['neutral_b0']).abs().max()))
            for m,method in enumerate(methods):
                actual=normal[method][j,:length].cpu();old=saved[method][i]['prior_mean']
                maximum[method]=max(maximum[method],float((actual-old).abs().max()))
                torch.testing.assert_close(actual,old,rtol=1e-5,atol=2e-5)
                torch.testing.assert_close(swapped[sid][method][j,:length].cpu(),actual,rtol=1e-5,atol=2e-5)
                for pol,policy in enumerate(policies):
                    value=lambda x:np.asarray(x.clamp(0,1).cpu() if policy=='clip_all' else x.cpu())
                    own=value(normal[method][j,:length])
                    pa=value(own_views['A'][method][j,:length]);pb=value(own_views['B'][method][j,:length])
                    key=method+'/'+policy
                    changes.setdefault(key+'/own_A_B',[]).append(dict(index=i,speaker=sid,emotion=int(q['emotion_id'][i]),**change(pa,pb,v,ch,tm)))
                    for target in people:
                        y=own if target==sid else value(swapped[target][method][j,:length])
                        pred_stats[m,pol,i,people.index(target)]=summary(y,v,ch,tm)
                        if target!=sid:
                            changes.setdefault(key+'/cross_AB',[]).append(dict(index=i,speaker=sid,target=target,emotion=int(q['emotion_id'][i]),**change(own,y,v,ch,tm)))
                if q['clip_id'][i] in display_lookup:
                    e=display_lookup[q['clip_id'][i]];ordered=[sid]+[x for x in people if x!=sid]
                    rows=[gt,normal[method][j,:length].cpu().numpy()]+[swapped[x][method][j,:length].cpu().numpy() for x in ordered[1:]]
                    rows += [own_views[x][method][j,:length].cpu().numpy() for x in ('A','B')]
                    labels=['GT']+[refs_meta[x]['speaker'].replace('mead_','')+' AB' for x in ordered]+['Own A','Own B']
                    reference_ids=['none']+['|'.join(refs_meta[x]['clip_ids']) for x in ordered]+[refs_meta[sid]['clip_ids'][0],refs_meta[sid]['clip_ids'][1]]
                    path=out/'render_inputs'/f'{method}_{e}.npz'
                    display[method+'/'+e]=export(path,rows,v,tm,ch,q['clip_id'][i],labels,reference_ids)
        processed+=len(sub)
        write(out/'state.json',dict(status='auditing',clips_processed=processed,total=len(ids),test_loaded=False))
    anchors={};all_codes={}
    for sid in people:
        ref=data['refs'][sid];means=[]
        codes={}
        for slot in ('AB','A','B'):
            fake={'speaker_id':torch.tensor([sid],device=device)};r=refs_for(data,fake,device,sid)
            if slot!='AB':r={k:v[:,0:1] if slot=='A' else v[:,1:2] for k,v in r.items()}
            codes[slot]=model.encode_style(r)['code'].cpu().numpy()[0]
        for j in range(2):
            means.append(summary(ref['motion'][j].numpy(),ref['valid'][j].numpy(),ref['channel_mask'][j].numpy(),ref['times'][j].numpy())[0])
        anchors[sid]=np.mean(means,axis=0)
        all_codes[sid]=codes
        style_rows.append(dict(speaker_id=sid,**refs_meta[sid],A_B_cosine=correlation(codes['A'],codes['B']),
            A_B_rms=float(np.sqrt(np.mean((codes['A']-codes['B'])**2))),AB_norm=float(np.linalg.norm(codes['AB']))))
    cross_codes=[dict(source=s,target=t,AB_cosine=correlation(all_codes[s]['AB'],all_codes[t]['AB']),
        AB_rms=float(np.sqrt(np.mean((all_codes[s]['AB']-all_codes[t]['AB'])**2)))) for s in people for t in people if s<t]
    chosen=set(ids);pairs=[(i,j) for i,j in pairs if i in chosen and j in chosen]
    target_rows={};scales=model.scales.cpu().numpy()
    for m,method in enumerate(methods):
        for pol,policy in enumerate(policies):
            rows=[]
            for i,j in pairs:
                source=int(q['speaker_id'][i]);target=int(q['speaker_id'][j])
                ch=(q['channel_mask'][i]&q['channel_mask'][j]).numpy();ch[51]=False
                vals=target_direction(pred_stats[m,pol,i,people.index(source)],pred_stats[m,pol,i,people.index(target)],
                    gt_stats[i],gt_stats[j],anchors[source],anchors[target],ch,scales)
                rows.append(dict(source_index=i,target_index=j,source_speaker=source,target_speaker=target,
                    emotion=int(q['emotion_id'][i]),**vals))
            target_rows[method+'/'+policy]=rows
    def summarize(records,keys):
        values=[{k:v for k,v in r.items() if k not in keys} for r in records]
        return aggregate(values) if values else {}
    names=data['config']['data']['emotion_classes']
    report=dict(schema='phase48_frozen_style_audit_v1',test_loaded=False,training_performed=False,
        default_replaced=False,smoke=a.smoke,clips=len(ids),parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'],
        correction_sha256=sha(a.correction),source_sha256=sha(__file__),data_manifest_sha256=binding['data_manifest_sha256'],
        frozen_state_exact=True,baseline_replay_max_abs=maximum,reference_metadata=refs_meta,style_code=style_rows,
        cross_speaker_style_code=cross_codes,
        baseline_curves=binding['style_audit']['baseline_curves'],
        data_audit=audit,matched_groups=sum(len(x)>1 for x in groups.values()),
        three_person_groups=sum(len(x)==3 for x in groups.values()),matched_directed_pairs=len(pairs),
        unmatched_clips=sum(len(x)==1 for x in groups.values()),render_inputs=display,
        response={k:dict(n=len(rows),metrics=summarize(rows,('index','speaker','target','emotion')),
            by_emotion={name:summarize([r for r in rows if r['emotion']==e],('index','speaker','target','emotion')) for e,name in enumerate(names)}) for k,rows in changes.items()},
        target_direction={k:dict(n=len(rows),metrics=summarize(rows,('source_index','target_index','source_speaker','target_speaker','emotion')),
            by_speaker_direction={f'{s}->{t}':dict(n=len([r for r in rows if r['source_speaker']==s and r['target_speaker']==t]),
                metrics=summarize([r for r in rows if r['source_speaker']==s and r['target_speaker']==t],('source_index','target_index','source_speaker','target_speaker','emotion'))) for s in people for t in people if s!=t},
            by_emotion={name:summarize([r for r in rows if r['emotion']==e],('source_index','target_index','source_speaker','target_speaker','emotion')) for e,name in enumerate(names)}) for k,rows in target_rows.items()},
        limits=['Shared rig tests behavioral style, not facial shape identity.',
            'Cross-person native clip-statistics include audio and performance differences; exact frame/phoneme matching is not claimed.',
            'Motion correlation/lag are not independent content recognition.',
            'Matched targets and development evidence are evaluation only, not deployment inputs or supervised training.',
            'Three development identities and this trained model do not prove general identity transfer.'])
    assert before==(state_digest(model),state_digest(base.stage1),state_digest(correction))
    np.savez_compressed(out/'native_statistics.npz',predictions=pred_stats[:,:,ids],gt=gt_stats[ids],indices=np.asarray(ids),
        methods=np.asarray(methods),policies=np.asarray(policies),stats=np.asarray(STATS),target_speaker_ids=np.asarray(people))
    # Preserve every pair in a compact lossless artifact on the nearly full host.
    with gzip.open(out/'per_pair.json.gz','wt',encoding='utf-8') as f:
        json.dump(finite(target_rows),f,allow_nan=False,separators=(',',':'))
    write(out/'report.json',finite(report))
    manifest={p.relative_to(out).as_posix():dict(size=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name!='state.json'}
    write(out/'manifest.json',dict(files=manifest,test_loaded=False))
    write(out/'state.json',dict(status='complete',clips=len(ids),report_sha256=sha(out/'report.json'),test_loaded=False))
    print(json.dumps(dict(complete=True,clips=len(ids),matched_pairs=len(pairs),maximum=maximum,render_inputs=len(display))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--correction',required=True)
    p.add_argument('--parent-curves',required=True);p.add_argument('--latent-curves',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:run(a)
    except Exception as e:
        out=Path(a.output)
        if out.exists():write(out/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
