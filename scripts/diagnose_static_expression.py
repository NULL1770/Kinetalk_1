"""TRAIN-only mean-response fit with fixed speaker/sentence holdouts.

The parent generator, B0, classifiers and all their parameters remain frozen.
Two prespecified ridge maps share the target: latent-only and an independent
neutral-reference skip. No external development/test data are used for fitting
or model selection in this diagnostic.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.static_expression import receiver_features, fit_static_correction
from kinetalk_b0.models.expression_response import masked_pool
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features, classification_metrics
from scripts.train_expression_response import (
    configure, load_runtime, cache_base, development_fold, audit_data, reference_batch,
    state_digest, sha, write)
from scripts.refine_expression_prior import restore_model
from scripts.evaluate_expression_response import metrics, aggregate, finite


@torch.no_grad()
def run(a):
    configure(47)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'started.json').exists():
        raise FileExistsError('Use a fresh diagnostic directory')
    write(out/'started.json',dict(time=time.time(),test_loaded=False))
    binding=json.loads(Path(a.binding).read_text(encoding='utf-8-sig'))
    for n,h in binding['source_files'].items():
        assert sha(Path(__file__).resolve().parents[1]/n)==h,n
    device=torch.device(a.device)
    data,base=load_runtime(binding,device)
    fold=development_fold(data['splits']['train'],data['fit_sids'])
    audit=audit_data(data,fold)
    model,parent=restore_model(binding,device)
    model.eval().requires_grad_(False)
    before,neutral=state_digest(model),state_digest(base.stage1)
    assert neutral==parent['neutral_digest']
    roles={k:list(fold[k]) for k in ('train','speaker_dev','sentence_dev')}
    if a.smoke:
        roles={k:v[:16] for k,v in roles.items()}
    all_ids=sum(roles.values(),[])
    cache_base(data,base,device,{'train':all_ids,'validation':[]})
    q=data['splits']['train']
    records={}
    for role,ids in roles.items():
        chunks={k:[] for k in ('latent','reference','target','support','emotion','speaker')}
        for sub in torch.tensor(ids).split(16):
            b=q.batch(sub,device);refs=reference_batch(data,b,device)
            prior=model.audio_prior(b['audio_features'],b['valid'])
            style=model.encode_style(refs)['code']
            y=model.decode(b['b0'],prior,style,b['valid'])
            obs=b['valid'][...,None] & b['channel_mask'][:,None]
            residual=torch.where(obs,b['motion']-y,0.)/model.scales
            target=residual.sum(1)/obs.sum(1).clamp_min(1)
            for mode in ('latent','reference'):
                chunks[mode].append(receiver_features(model,prior,style,refs,mode).cpu())
            chunks['target'].append(target.cpu());chunks['support'].append(obs.any(1).cpu())
            chunks['emotion'].append(b['emotion_id'].cpu());chunks['speaker'].append(b['speaker_id'].cpu())
        records[role]={k:torch.cat(v) for k,v in chunks.items()}
        print(json.dumps(dict(event='extracted',role=role,clips=len(ids))),flush=True)
    corrections={mode:fit_static_correction(records['train'][mode],records['train']['target'],
        records['train']['support'],mode) for mode in ('latent','reference')}
    for mode,c in corrections.items():
        torch.save(dict(schema='static_expression_ridge_v1',mode=mode,state=c.state_dict(),
            parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'],
            data_manifest_sha256=binding['data_manifest_sha256'],strength=.001,
            fit_clips=len(roles['train']),test_loaded=False),out/(mode+'.pt'))
        corrections[mode]=c.to(device)
    # Real data smoke: the public deployment path must ignore content features.
    b=q.batch(torch.tensor(roles['speaker_dev'][:2]),device)
    refs=reference_batch(data,b,device)
    audio=b['audio_features'].clone();audio[...,:768]=float('nan')
    for c in corrections.values():
        expected=c.predict(model,b['audio_features'],b['b0'],b['valid'],refs)
        observed=c.predict(model,audio,b['b0'],b['valid'],refs)
        torch.testing.assert_close(expected,observed,rtol=0,atol=0)
    write(out/'fold.json',roles)
    torch.save(records,out/'fit_records.pt')
    target_errors={}
    for role,z in records.items():
        target_errors[role]={}
        for mode in ('parent','latent','reference'):
            err=z['target'] if mode=='parent' else z['target']-corrections[mode].offset(z[mode].to(device)).cpu()
            # Clip-equal, then observed-channel average.
            score=(torch.where(z['support'],err.square(),0.).sum(1)/z['support'].sum(1)).mean()
            target_errors[role][mode]=float(score)
    write(out/'target_errors.json',target_errors)
    print(json.dumps(dict(event='fit',mean_response_errors=target_errors)),flush=True)
    probes=[];names=data['config']['data']['emotion_classes']
    for spec in binding['probes']:
        assert sha(spec['path'])==spec['sha256']
        ck=torch.load(spec['path'],map_location='cpu',weights_only=False)
        assert not ck.get('generator_outputs_used_for_fitting') and not ck.get('test_used_for_selection')
        assert ck['train_manifest_sha256']==binding['data_manifest_sha256'] and ck['classes']==names
        net=MotionEmotionProbe(ck['feature_dim'],ck['hidden'],len(names)).eval()
        net.load_state_dict(ck['model'])
        probes.append((net,ck.get('feature_mask',torch.ones(ck['feature_dim'],dtype=torch.bool))))
    assert sha(binding['rig'])==binding['rig_sha256']
    with np.load(binding['rig'],allow_pickle=False) as z:rig={k:z[k].copy() for k in z.files}
    reports={};all_rows={};max_displacement_error=0.
    for role in ('speaker_dev','sentence_dev'):
        rows={k:[] for k in ('parent/raw','parent/clip_all','latent/raw','latent/clip_all','reference/raw','reference/clip_all')}
        features={k:[] for k in rows};support=torch.tensor([True]*51+[False])
        for sub in torch.tensor(roles[role]).split(16):
            b=q.batch(sub,device);refs=reference_batch(data,b,device)
            p=model.audio_prior(b['audio_features'],b['valid']);s=model.encode_style(refs)['code']
            y=model.decode(b['b0'],p,s,b['valid']);preds={'parent':y}
            for mode,c in corrections.items():
                preds[mode]=c(y,receiver_features(model,p,s,refs,mode),b['valid'],model.scales)
                adjacent=b['valid'][:,1:] & b['valid'][:,:-1]
                err=((preds[mode][:,1:]-preds[mode][:,:-1])-(y[:,1:]-y[:,:-1]))[adjacent]
                max_displacement_error=max(max_displacement_error,float(err.abs().max()))
            for j,i in enumerate(sub.tolist()):
                for mode,pred in preds.items():
                    for policy in ('raw','clip_all'):
                        v=pred[j] if policy=='raw' else pred[j].clamp(0,1);k=mode+'/'+policy
                        rows[k].append(metrics(v,b,j,rig))
                        features[k].append(motion_features(v.cpu()[:,support],b['valid'][j].cpu()))
            write(out/'state.json',dict(status='internal_evaluation',role=role,
                clips=len(rows['parent/raw']),total=len(roles[role]),test_loaded=False))
        reports[role]={}
        for k,r in rows.items():
            f=torch.stack(features[k]);labels=records[role]['emotion']
            reports[role][k]=dict(metrics=aggregate(r),probes=[classification_metrics(labels,net(f[:,mask]).argmax(-1),names) for net,mask in probes],
                per_speaker={str(sid):aggregate([rr for rr,s in zip(r,records[role]['speaker']) if int(s)==sid]) for sid in records[role]['speaker'].unique().tolist()})
        all_rows[role]=rows
        print(json.dumps(dict(event='internal_evaluation_complete',role=role)),flush=True)
    assert state_digest(model)==before and state_digest(base.stage1)==neutral
    assert max_displacement_error<2e-6
    report=dict(schema='phase47_train_internal_static_response_v1',parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'],
        binding_sha256=sha(a.binding),audit=audit,smoke=a.smoke,fit_clips=len(roles['train']),
        features='frozen g32+style64; plus neutral52 and soft audio-emotion8 x neutral52',
        target='clip-equal TRAIN normalized mean reconstruction error; no query GT deployment',
        analytic_fit_passes=1,ridge_strength=.001,mean_response_errors=target_errors,results=reports,
        frozen_state_exact=True,neutral_exact=True,max_raw_displacement_change=max_displacement_error,
        external_validation_used=False,hubert_nan_isolated=True,test_loaded=False,default_replaced=False)
    write(out/'per_clip.json',finite(all_rows));write(out/'report.json',finite(report))
    write(out/'state.json',dict(status='complete',test_loaded=False))
    manifest={str(p.relative_to(out)):dict(sha256=sha(p),size=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(out/'manifest.json',dict(files=manifest))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        write(Path(a.output)/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
