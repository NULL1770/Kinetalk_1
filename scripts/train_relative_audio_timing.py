"""TRAIN-identity selection, full-data refit, and protected-mean timing audit.

The development partition is never used for epoch/representation selection.
Static removes the new temporal prediction; frozen global expression and
all non-upper coefficients stay identical across paired interventions.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.relative_audio_timing import RelativeAudioTiming, relative_features, masked_center
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from scripts.train_isolated_audio_state import (load_context, trim, state_metrics, paired_ci, old_prediction)
from scripts.train_full_staged import subset, obs, batch_identity, NOT_UPPER
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, canonical_hash
from scripts.train_predictable_renderer import state_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.paper_generation_report import report_generation
from scripts.compact_native_curves import compact_curves
from scripts.mouth_protection import protection_report
from scripts.run_calibrated_temporal_queue import interval

SEEDS = (42, 123, 2026)


def set_status(out, **record):
    record.update(test_loaded=False, default_replaced=False)
    save_json(out/'status.json', record)
    print(json.dumps(record), flush=True)


def fit_statistics(q, ids, *, rank=24):
    """Audio-only TRAIN stats/PCA; no held identity or development frames."""
    f = q['audio_features']; n = f.shape[-1]
    s = torch.zeros(n, dtype=torch.float64); ss = s.clone(); count = 0
    for ix in ids.split(32):
        x = f[ix][q['valid'][ix]].double()
        s += x.sum(0); ss += x.square().sum(0); count += len(x)
    mean = (s/count).float(); std = (ss/count-(s/count).square()).clamp_min(1e-6).sqrt().float()
    samples = []
    for ix in ids.split(32):
        x = relative_features(f[ix], q['valid'][ix], mean, std)[..., :1536]
        samples.append(x[:, ::8][q['valid'][ix, ::8]])
    x = torch.cat(samples).cuda()
    torch.manual_seed(20260921)
    _, _, v = torch.pca_lowrank(x, q=rank, center=False, niter=3)
    del x
    y = q['target_centered_state'][ids]; valid = q['valid'][ids]
    ds = (y[valid].square().mean(0)).sqrt().clamp_min(.02)
    return mean, std, v.cpu(), ds


def new_model(stats, scales):
    mean, std, projection, ds = stats
    return RelativeAudioTiming(mean, std, projection, scales, ds).cuda()


@torch.no_grad()
def predict(model, q, ids, mode='audio'):
    rows = []
    for ix in ids.split(32):
        b = subset(q, ix, 'cuda')
        rows.append(model(b['audio_features'], b['valid'], mode)['state'].cpu())
    return torch.cat(rows)


def train_epoch(model, q, ids, opt, rng):
    model.train(); losses = []
    for ix in ids[torch.randperm(len(ids), generator=rng)].split(32):
        b = trim(q, ix, 'cuda')
        pred = model(b['audio_features'], b['valid'])['state']
        err = ((pred-b['target_centered_state'])/model.dynamic_scales).square().mean(-1)
        # Equal clip weight prevents long recordings dominating the objective.
        loss = (torch.where(b['valid'], err, 0.).sum(1)/b['valid'].sum(1)).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite timing loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); losses.append(float(loss.detach()))
    model.eval(); return float(np.mean(losses))


def metrics(states, target, valid, sentences):
    # Full prediction comparison is clip-weighted for paired CIs.
    per = {k: ((v-target).square().sum((1,2))/(valid.sum(1)*4)).numpy() for k,v in states.items()}
    return {'states': {k: state_metrics(v,target,valid) for k,v in states.items()},
            'state_pairs': {k: paired_ci(per['full']-per[k], sentences) for k in ('static','reverse')}}


@torch.no_grad()
def final_evaluation(model, data, system, audio, identities, out):
    q = data['splits']['validation']; valid = q['valid']; cc = list(UPPER_INDICES)
    ids = torch.arange(len(valid))
    states = {name: predict(model,q,ids,mode) for name,mode in [('full','audio'),('static','static'),('reverse','reverse')]}
    report = metrics(states,q['target_centered_state'],valid,q['sentence_id'])
    report.update(clips=len(valid), test_loaded=False, scope='complete development; no selection',
                  intervention_scope='new timing only; frozen Stage4 clip mean retained')
    predictions={}; mouth={}; emotions={}; mean_drift={}; nonupper={}
    for seed in SEEDS:
        noise=torch.randn(q['motion'].shape,generator=torch.Generator().manual_seed(seed))
        base=[]
        for ix in ids.split(16):
            b=subset(q,ix,'cuda');b['frozen_local']=audio(b['audio_features'],b['valid'])['local']
            base.append(old_prediction(system,b,identities,noise[ix].cuda()).cpu())
        base=torch.cat(base);predictions[f'{seed}/base']=base
        mean=(base[...,cc]*valid[...,None]).sum(1)/valid.sum(1)[:,None]
        for name,mode in [('full','audio'),('static','static'),('reverse','reverse')]:
            values=[]
            for ix in ids.split(32):
                b=subset(q,ix,'cuda');delta=model(b['audio_features'],b['valid'],mode)['delta'].cpu()[...,cc]
                values.append(mean[ix,None]+delta)
            upper=torch.cat(values);pred=compose_upper_face(base,upper,valid)
            key=f'{seed}/{name}';predictions[key]=pred
            nonupper[key]=bool(torch.equal(pred[...,NOT_UPPER],base[...,NOT_UPPER]))
            mean_drift[key]=float(((pred[...,cc]*valid[...,None]).sum(1)/valid.sum(1)[:,None]-mean).abs().max())
        for name in ('base','full','static','reverse'):
            key=f'{seed}/{name}';pred=predictions[key];correct=0
            mouth[key]=protection_report(pred,base,q['motion'],valid,q['channel_mask'],q['emotion_id'])
            for ix in ids.split(32):
                b=subset(q,ix,'cuda');ident=batch_identity(identities,b)
                enc=system.encode_motion(torch.where(obs(b),pred[ix].cuda()-b['b0']-ident['baseline'][:,None],0.),b['valid'])
                correct+=int((enc['emotion_logits'].argmax(-1)==b['emotion_id']).sum())
            emotions[key]=correct/len(valid)
    curves={'clip_id':q['clip_id'],'target':q['motion'],'valid':valid,'times':q['times'],
            'channel_mask':q['channel_mask'],'b0':q['b0'],'predictions':predictions,'noise_seeds':list(SEEDS)}
    bench=report_generation(curves,data,out,'relative',SimpleNamespace(condition_mode='relative_audio',artifact_dir=out/'scores'))
    rows={k:json.loads((out/'scores'/'relative'/f'temporal_{k}.json').read_text()) for k in ('full','static','reverse')}
    temporal={control:{name:interval(rows['full'],rows[control],path) for name,path in
              [('centered_es',('joint_fair_es','centered')),('variogram',('variogram','aggregate'))]}
              for control in ('static','reverse')}
    em={name:float(np.mean([emotions[f'{s}/{name}'] for s in SEEDS])) for name in ('base','full','static','reverse')}
    checks={'mouth_preserved':all(v['passed'] for v in mouth.values()),'nonupper43_exact':all(nonupper.values()),
            'mean_preserved':max(mean_drift.values())<1e-5,
            'internal_emotion_no_large_drop':em['full']>=em['base']-.03,
            'mbe_no_worse':bench['full']['coefficient']['arkit_mbe']['value']<=bench['base']['coefficient']['arkit_mbe']['value'],
            'state_better_than_static':report['state_pairs']['static']['passed'],
            'state_better_than_reverse':report['state_pairs']['reverse']['passed']}
    report.update(checks=checks,passed=all(checks.values()),mouth_protection=mouth,mean_drift=mean_drift,
        generated_teacher_accuracy_nonindependent=emotions,temporal_pairs=temporal,
        independent_emotion_AV_visual_pending=True)
    save_json(out/'evaluation.json',report)
    manifest=json.loads((Path(data['_directory'])/'manifest.json').read_text())
    lengths={r['clip_id']:r['frames'] for r in manifest['roles']['val']['query']}
    save_checkpoint(out/'native_curves.pt',compact_curves(curves,lengths))
    return report


def main():
    p=argparse.ArgumentParser()
    for key in ('data','source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--selection-epochs',type=int,default=60)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    random.seed(47);np.random.seed(47);torch.manual_seed(47)
    set_status(a.output,status='preparing')
    data,system,audio,identities,_=load_context(a.data,a.source,'cuda',47);data['_directory']=str(a.data)
    q=data['splits']['train']
    if not q['channel_mask'][:,list(UPPER_INDICES)].all():raise ValueError('All upper channels must be observed')
    if [len(data['splits'][r]['valid']) for r in ('train','validation')]!=[4098,446]:raise ValueError('Full protocol required')
    frozen={'system':state_hash(system.state_dict()),'audio':state_hash(audio.state_dict())}
    root=Path(__file__).resolve().parents[1]
    protocol={'schema':'relative_audio_timing_v1','source_sha256':sha(a.source),'data':data['provenance'],
        'selection':'TRAIN-only speaker+sentence holdout; best epoch then all4098 refit',
        'global_mean':'frozen Stage4 per-sample mean; no learned mean replacement',
        'state_input':'clip-relative acoustic PCA24 + prosody4 + first differences; no identity/global code',
        'max_selection_epochs':a.selection_epochs,'seed':47,'smoke':a.smoke,'test_loaded':False,
        'sources':{n:sha(root/n) for n in ('scripts/train_relative_audio_timing.py','kinetalk_b0/models/relative_audio_timing.py')}}
    speakers=sorted(set(q['speaker']));held={s for i,s in enumerate(speakers) if i%5==0}
    held_ids=torch.tensor([i for i,s in enumerate(q['speaker']) if s in held])
    # Sentence-disjoint calibration as well as speaker-disjoint fitting.
    sentences=sorted({q['sentence_id'][i] for i in held_ids.tolist()})
    chosen=set(sentences[::4])
    cal=torch.tensor([i for i in held_ids.tolist() if q['sentence_id'][i] in chosen])
    fit=torch.tensor([i for i,s in enumerate(q['speaker']) if s not in held and q['sentence_id'][i] not in chosen])
    if a.smoke:
        fit,cal=fit[:32],cal[:16]
        data['splits']['validation']=subset(data['splits']['validation'],torch.arange(4),'cpu')
    protocol.update(selection_fit_ids=[q['clip_id'][i] for i in fit],selection_cal_ids=[q['clip_id'][i] for i in cal],
                    selection_held_speakers=sorted(held))
    save_json(a.output/'protocol.json',protocol)
    stats=fit_statistics(q,fit);torch.manual_seed(47);model=new_model(stats,data['target_scales'])
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02);rng=torch.Generator().manual_seed(47)
    best=float('inf');best_epoch=1;history=[]
    budget=1 if a.smoke else a.selection_epochs
    for epoch in range(1,budget+1):
        loss=train_epoch(model,q,fit,opt,rng)
        rec={'epoch':epoch,'loss':loss}
        if epoch==1 or epoch%5==0 or epoch==budget:
            prediction=predict(model,q,cal)
            score=state_metrics(prediction,q['target_centered_state'][cal],q['valid'][cal])
            rec['held_identity_state']=score
            if score['mse']<best:
                best=score['mse'];best_epoch=epoch
        history.append(rec);save_json(a.output/'selection_history.json',history)
        set_status(a.output,status='selecting_on_train_identities',epoch=epoch,epochs=budget,best_epoch=best_epoch,**{'loss':loss})
    save_json(a.output/'selection.json',{'chosen_epochs':best_epoch,'best_calibration_mse':best,
        'fit_clips':len(fit),'calibration_clips':len(cal),'development_used':False})
    del model,opt
    all_ids=torch.arange(len(q['valid'])) if not a.smoke else fit
    stats=fit_statistics(q,all_ids);torch.manual_seed(47);model=new_model(stats,data['target_scales'])
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02);rng=torch.Generator().manual_seed(47)
    for epoch in range(1,best_epoch+1):
        loss=train_epoch(model,q,all_ids,opt,rng)
        save_checkpoint(a.output/'last.pt',{'model':model.state_dict(),'epoch':epoch,'optimizer':opt.state_dict(),
            'protocol_sha256':canonical_hash(protocol),'selected_epochs':best_epoch})
        set_status(a.output,status='full_train_refit',epoch=epoch,epochs=best_epoch,clips=len(all_ids),loss=loss)
    save_checkpoint(a.output/'final.pt',{'model':model.state_dict(),'config':model.export_config(),
        'protocol':protocol,'protocol_sha256':canonical_hash(protocol),'selected_epochs':best_epoch})
    set_status(a.output,status='evaluating_full_development')
    report=final_evaluation(model.eval(),data,system,audio,identities,a.output)
    if state_hash(system.state_dict())!=frozen['system'] or state_hash(audio.state_dict())!=frozen['audio']:
        raise RuntimeError('Frozen source changed')
    set_status(a.output,status='complete',passed=report['passed'] and not a.smoke,
               selected_epochs=best_epoch,elapsed_seconds=time.monotonic()-started,
               independent_emotion_AV_visual_pending=True)


if __name__=='__main__':
    main()
