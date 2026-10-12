"""Resumable MEAD/HDTF joint-model experiment, separate from Phase53.

Initial runs are implementation pilots, not paper results. Every checkpoint
records whether calibrated phonetic constraints, text, HDTF and TRAIN-only
guidance calibration are actually enabled. Evaluation always includes a full
teacher-free event rollout and separately reports the observed-event oracle.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import load_runtime,development_fold,configure,state_digest,write,save
from scripts.train_full_staged import base_forward
from kinetalk_b0.models.expression_response import motion_objective
from kinetalk_v3.unified_model_v3 import (UnifiedV3Config,UnifiedExpressionV3Model,teacher_targets,
    v3_supervision_losses,relation_style_loss,anchor_matrix)


def select_balanced(q,ids,limit,seed):
    if not limit or len(ids)<=limit:return list(ids)
    rng=random.Random(seed);groups={}
    for i in ids:groups.setdefault((int(q['speaker_id'][i]),int(q['emotion_id'][i])),[]).append(i)
    for values in groups.values():rng.shuffle(values)
    chosen=[]
    while len(chosen)<limit:
        for key in sorted(groups):
            if groups[key]:chosen.append(groups[key].pop())
            if len(chosen)>=limit:break
    return chosen


def reference_ids(q,query_ids,pool,rng):
    refs=[]
    for i in query_ids:
        candidates=[j for j in pool[int(q['speaker_id'][i])] if q['clip_id'][j]!=q['clip_id'][i]
            and q['sentence_id'][j]!=q['sentence_id'][i]]
        cross=[j for j in candidates if q['emotion_id'][j]!=q['emotion_id'][i]]
        if not candidates:raise ValueError('No independent cross-sentence reference for '+q['clip_id'][i])
        refs.append(rng.choice(cross or candidates))
    return refs


@torch.no_grad()
def build_base_cache(q,base,ids,device):
    cached={}
    for start in range(0,len(ids),16):
        use=ids[start:start+16];b=q.batch(torch.tensor(use),device,keys=('content','valid'))
        result=base_forward(base,b['content'],b['valid'])
        for j,i in enumerate(use):
            n=int(q['_lengths'][i]);cached[i]={k:result[k][j,:n].cpu() for k in ('b0','h0')}
    return cached


def attach_base(b,ids,cache):
    for key in ('b0','h0'):
        chunks=[cache[int(i)][key] for i in ids];width=b['valid'].shape[1]
        b[key]=torch.stack([F.pad(x,(0,0,0,width-len(x))) for x in chunks]).to(b['valid'].device)
    return b


def make_refs(q,ids,device):
    b=q.batch(torch.tensor(ids),device,keys=('motion','valid','channel_mask','clip_id','sentence_id'))
    return b


def pair_batches(q,ids,batch_size,rng):
    groups={}
    for i in ids:groups.setdefault(int(q['speaker_id'][i]),[]).append(i)
    units=[]
    for values in groups.values():
        rng.shuffle(values);units.extend(values[j:j+2] for j in range(0,len(values),2))
    rng.shuffle(units);order=[i for unit in units for i in unit]
    return [order[i:i+batch_size] for i in range(0,len(order),batch_size)]


@torch.no_grad()
def calibrate_guidance(model,q,ids,cache,device,seed):
    """Small TRAIN-only grouped ridge probe of observed teacher coordinates.

    Both speaker and sentence group holds are evaluated. This finite predictor
    calibrates guidance; it is not an information-theoretic acoustic ceiling.
    Random projection is fixed and shared across folds; standardization and
    every regression fit occur on each fold's training data only.
    """
    rng=torch.Generator().manual_seed(seed)
    projection=torch.randn(772,48,generator=rng)/np.sqrt(772)
    features=[];target=[];speaker=[];sentence=[];emotions=[]
    for start in range(0,len(ids),16):
        use=ids[start:start+16];b=attach_base(q.batch(torch.tensor(use),device),use,cache)
        teacher=model.teacher(b['motion'],b['b0'],b['valid'],b['channel_mask'],model.scales)['state'].cpu()
        a=((b['audio_features'][...,768:]-model.feature_mean)/model.feature_std).cpu()@projection
        for j,i in enumerate(use):
            n=int(q['_lengths'][i]);mask=b['valid'][j,:n].cpu()
            ix=torch.arange(0,n,5);ix=ix[mask[ix]]
            if not len(ix):continue
            features.append(a[j,ix]);target.append(teacher[j,ix]);speaker.extend([int(q['speaker_id'][i])]*len(ix))
            sentence.extend([str(q['sentence_id'][i])]*len(ix));emotions.extend([int(q['emotion_id'][i])]*len(ix))
    x=torch.cat(features).double();y=torch.cat(target).double();emotion=torch.tensor(emotions)
    reports=[];skills=[]
    for name,labels in [('speaker',speaker),('sentence',sentence)]:
        labels=sorted(set(labels));assignment={v:k%3 for k,v in enumerate(labels)}
        rows=speaker if name=='speaker' else sentence;fold=torch.tensor([assignment[v] for v in rows])
        sse=torch.zeros(10,dtype=torch.double);baseline=torch.zeros_like(sse);count=0
        for k in range(3):
            fit=fold.ne(k);held=fold.eq(k)
            if fit.sum()<60 or not held.any():continue
            mean=x[fit].mean(0);std=x[fit].std(0).clamp_min(.001)
            xf=torch.cat(((x[fit]-mean)/std,torch.ones(int(fit.sum()),1)),1)
            xh=torch.cat(((x[held]-mean)/std,torch.ones(int(held.sum()),1)),1)
            penalty=torch.eye(xf.shape[1],dtype=torch.double)*20;penalty[-1,-1]=.01
            coef=torch.linalg.solve(xf.T@xf+penalty,xf.T@y[fit])
            pred=xh@coef;constant=y[fit].mean(0).expand(int(held.sum()),-1).clone()
            for e in emotion.unique():
                use=fit&emotion.eq(e);take=emotion[held].eq(e)
                if use.any():constant[take]=y[use].mean(0)
            sse+=(pred-y[held]).square().sum(0);baseline+=(constant-y[held]).square().sum(0);count+=int(held.sum())
        skill=1-sse/baseline.clamp_min(1e-9)
        reports.append({'group':name,'held_rows':count,'skill':skill.tolist()});skills.append(skill)
    conservative=torch.stack(skills).amin(0).clamp(0,1)
    weights=.1+.9*conservative
    model.guidance_weights.copy_(weights.float().to(device))
    return {'schema':'train_grouped_ridge_guidance_v1','fit_clip_ids':[q['clip_id'][i] for i in ids],
        'reports':reports,'weights':weights.tolist(),'floor':.1,'heldout_targets_loaded':False,
        'limitation':'48D random-projected linear finite-model probe; not conditional entropy or final temporal validation'}


@torch.no_grad()
def evaluate(model,q,ids,cache,data,base,device,seed,refs_from_query,pool):
    model.eval();rng=random.Random(seed);totals={};count=0
    for start in range(0,len(ids),8):
        use=ids[start:start+8];b=attach_base(q.batch(torch.tensor(use),device),use,cache)
        if refs_from_query:
            refs=make_refs(q,reference_ids(q,use,pool,rng),device)
        else:
            values=[data['refs'][int(q['speaker_id'][i])] for i in use]
            refs={k:torch.stack([r[k][0] for r in values]).to(device) for k in ('motion','valid','channel_mask')}
            if any(q['clip_id'][i]==r['clip_id'][0] for i,r in zip(use,values)):raise ValueError('Query/reference overlap')
        targets=teacher_targets(b['motion'],b['b0'],b['valid'],b['channel_mask'],b['times'])
        observed=model.training_forward(b,refs,targets)
        rollout=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,times=b['times'],seed=seed+start)
        for name,out in [('observed',observed),('rollout',rollout)]:
            _,parts=motion_objective(out['motion'],b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)
            for key,value in parts.items():totals[name+'_'+key]=totals.get(name+'_'+key,0.)+float(value)*len(use)
        known=b['emotion_id'].ge(0);correct=rollout['student']['emotion_logits'].argmax(-1).eq(b['emotion_id'])&known
        totals['emotion_correct']=totals.get('emotion_correct',0)+int(correct.sum());totals['emotion_known']=totals.get('emotion_known',0)+int(known.sum())
        for name,events in [('teacher',targets['events']),('rollout',rollout['schedule']['events'])]:
            active=events.reshape(len(use),-1,4,3)[...,0].bool()
            totals[name+'_active']=totals.get(name+'_active',0)+int((active&b['valid'][...,None]).sum())
        totals['valid_group_frames']=totals.get('valid_group_frames',0)+int(b['valid'].sum())*4
        count+=len(use)
    result={k:v/count for k,v in totals.items() if k.endswith('_position') or k.endswith('_velocity')}
    result.update(clips=count,emotion_accuracy=totals['emotion_correct']/max(totals['emotion_known'],1),
        teacher_event_occupancy=totals['teacher_active']/max(totals['valid_group_frames'],1),
        rollout_event_occupancy=totals['rollout_active']/max(totals['valid_group_frames'],1))
    model.train();return result


def train(a):
    configure(a.seed);device=torch.device(a.device);root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    binding=json.loads(Path(a.binding).read_text());data,base=load_runtime(binding,device)
    base_hash=state_digest(base.stage1);q=data['splits']['train'];fold=development_fold(q,data['fit_sids'])
    fit=select_balanced(q,fold['train'],a.limit,a.seed);held=select_balanced(q,fold['speaker_dev']+fold['sentence_dev'],a.eval_limit,a.seed)
    pool={}
    for i in fold['train']:pool.setdefault(int(q['speaker_id'][i]),[]).append(i)
    # Cache selected training/evaluation plus every chosen reference once.
    rng=random.Random(a.seed);reference_by_query={i:reference_ids(q,[i],pool,rng)[0] for i in fit}
    required=sorted(set(fit+held+list(reference_by_query.values())))
    print(json.dumps({'event':'cache_start','clips':len(required)}),flush=True)
    cache=build_base_cache(q,base,required,device)
    content_dim=next(iter(cache.values()))['h0'].shape[-1]
    cfg=UnifiedV3Config(content_dim=content_dim,hidden=a.hidden,decoder_hidden=a.decoder_hidden)
    stats=data['feature_stats'];model=UnifiedExpressionV3Model(cfg,stats['mean'][768:],stats['std'][768:],data['target_scales']).to(device)
    protocol={'schema':'unified_v3_training_pilot_v1','seed':a.seed,'epochs':a.epochs,'warmup_epochs':a.warmup_epochs,
        'fit_clips':len(fit),'internal_dev_clips':len(held),'data_manifest_sha256':binding['data_manifest_sha256'],
        'neutral_checkpoint':binding['neutral_checkpoint'],'phase53_untouched':True,'test_loaded':False,
        'phonetic_constraints_enabled':False,'text_enabled':False,'hdtf_enabled':False,
        'status':'implementation_pilot','missing':['ASR/phonetic phase calibration','HDTF audio feature ingestion','multi-seed matched-budget ablations'],
        'calibration':'TRAIN-only grouped finite linear probe after teacher warmup',
        'event_training':'observed event NLL + reconstruction; sampled event rollout has no full-face GT L2',
        'rollout_consistency_every':a.rollout_every,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(root/'protocol.json',protocol);write(root/'fold.json',{'fit_ids':fit,'held_ids':held,'reference_by_query':reference_by_query})
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    start_epoch=0;step=0;history=[];rng=random.Random(a.seed)
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False)
        if ck['protocol']!=protocol:raise ValueError('Resume protocol differs')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);start_epoch=ck['epoch'];step=ck['step'];history=ck['history']
        torch.set_rng_state(ck['torch_rng']);rng.setstate(ck['python_rng'])
        if 'cuda_rng' in ck and device.type=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
    teacher_frozen=False;initial=None
    for epoch in range(start_epoch,a.epochs):
        warm=epoch<a.warmup_epochs
        if not warm and not teacher_frozen:
            model.teacher.eval().requires_grad_(False);teacher_frozen=True
            if not (root/'guidance_calibration.json').exists():
                calibration=calibrate_guidance(model,q,fit,cache,device,a.seed)
                write(root/'guidance_calibration.json',calibration)
        model.train()
        if teacher_frozen:model.teacher.eval()
        sums={};batches=pair_batches(q,fit,a.batch_size,rng);begin=time.monotonic()
        for index,use in enumerate(batches):
            b=attach_base(q.batch(torch.tensor(use),device),use,cache)
            # Each forward receives exactly one reference. The independent
            # same-speaker/cross-emotion pair is for relation supervision.
            ref_ids=[reference_by_query[i] for i in use];refs=make_refs(q,ref_ids,device)
            if any(q['sentence_id'][i]==q['sentence_id'][j] for i,j in zip(use,ref_ids)):raise ValueError('Reference sentence overlaps')
            targets=teacher_targets(b['motion'],b['b0'],b['valid'],b['channel_mask'],b['times'])
            optimizer.zero_grad(set_to_none=True)
            out=model.training_forward(b,refs,targets,teacher_state=warm)
            motion,parts=motion_objective(out['motion'],b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)
            supervision=v3_supervision_losses(model,out,targets,b,teacher_warmup=warm)
            query_codes=model.style(b['motion'],b['valid'],b['channel_mask'],model.scales)
            relation=relation_style_loss(torch.cat((out['style'],query_codes)),b['speaker_id'].repeat(2))
            consistency=F.smooth_l1_loss(out['style'],query_codes)
            loss=motion+supervision['total']+.05*relation+.05*consistency
            rollout_consistency=loss*0
            if not warm and a.rollout_every and step%a.rollout_every==0:
                # Sampled events are alternatives, not another framewise GT.
                # Only the ongoing expression state is expected to agree with
                # the requested student state. Regions containing stochastic
                # events get no sampled trajectory reconstruction objective.
                generated=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,
                    times=b['times'],seed=a.seed+step)
                generated_state=(generated['motion']-b['b0'])@anchor_matrix().to(device)
                mask=targets['quality'].bool()
                error=F.smooth_l1_loss(generated_state,generated['student']['state'].detach(),reduction='none')
                rollout_consistency=torch.where(mask,error,0.).sum()/mask.sum().clamp_min(1)
                loss=loss+.1*rollout_consistency
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if not warm and not model.student.state.weight.grad.abs().sum()>0:raise RuntimeError('Student reconstruction/state gradient missing')
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();step+=1
            values={'loss':float(loss),'position':float(parts['position']),'velocity':float(parts['velocity']),
                'state':float(supervision['state_direct']),'hazard':float(supervision['hazard']),
                'relation':float(relation),'rollout_consistency':float(rollout_consistency),'grad_norm':float(grad)}
            for key,value in values.items():sums[key]=sums.get(key,0.)+value
            if initial is None:initial=values['loss']
            if index%20==0:
                state={'status':'training','epoch':epoch+1,'step':step,'batch':index+1,'batches':len(batches),'teacher_warmup':warm,**values}
                write(root/'state.json',state);print(json.dumps(state),flush=True)
        # Internal held-sentence/speaker uses independent enrollment refs; not
        # source query GT as conditioning. These repeated dev sets are not test.
        evaluation=evaluate(model,q,held,cache,data,base,device,a.seed,False,pool)
        if state_digest(base.stage1)!=base_hash:raise RuntimeError('Frozen neutral B0 changed')
        entry={'epoch':epoch+1,'step':step,'seconds':time.monotonic()-begin,'train':{k:v/len(batches) for k,v in sums.items()},'internal_dev':evaluation}
        history.append(entry);write(root/'history.json',history)
        ck={'config':asdict(cfg),'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'step':step,
            'history':history,'protocol':protocol,'torch_rng':torch.get_rng_state(),'python_rng':rng.getstate(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        save(root/'resume.pt',ck)
        if epoch==a.epochs-1:save(root/'final.pt',{k:v for k,v in ck.items() if k!='optimizer'})
        write(root/'state.json',{'status':'epoch_complete',**entry});print(json.dumps({'event':'epoch_complete',**entry}),flush=True)
    write(root/'complete.json',{'status':'complete','epochs':a.epochs,'updates':step,'initial_loss':initial,
        'final_loss':history[-1]['train']['loss'],'neutral_frozen_sha256':base_hash,'protocol':protocol,
        'final_internal_dev':history[-1]['internal_dev'],'improvement_over_phase53_verified':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--limit',type=int,default=512);p.add_argument('--eval-limit',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--hidden',type=int,default=128);p.add_argument('--decoder-hidden',type=int,default=192)
    p.add_argument('--rollout-every',type=int,default=4)
    p.add_argument('--resume');train(p.parse_args())

