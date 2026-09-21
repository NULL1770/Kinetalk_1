"""Protected-mean stochastic motion with bottlenecked relative audio.

Exploratory generative continuation: deterministic state evidence is reported
separately, never converted into a pass by this runner. Matched audio/static
models are trained and final temporal/protection gates are mandatory.
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
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.relative_audio_timing import RelativeAudioTiming, relative_features
from kinetalk_b0.models.dc_protected_temporal_flow import DCProtectedTemporalFlow, project_temporal_dc
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from kinetalk_b0.models.audio_residual_flow import fair_trajectory_es
from scripts.train_isolated_audio_state import load_context, trim, old_prediction, center
from scripts.train_full_staged import subset, obs, batch_identity, NOT_UPPER
from scripts.train_formal_predictable_projection import save_json,save_checkpoint,canonical_hash,capture_rng,restore_rng
from scripts.train_predictable_renderer import state_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.paper_generation_report import report_generation
from scripts.compact_native_curves import compact_curves
from scripts.mouth_protection import protection_report
from scripts.run_calibrated_temporal_queue import interval
from scripts.run_isolated_state_queue import source_inventory

SEEDS=(42,123,2026)
MODES=('base','deterministic','full','static','reverse')


def status(out, **record):
    record.update(test_loaded=False,default_replaced=False)
    save_json(out/'status.json',record);print(json.dumps(record),flush=True)


def load_timing(path, data, source):
    ck=torch.load(path,map_location='cpu',weights_only=False);p=ck['protocol'];s=ck['model']
    if (p['schema']!='relative_audio_timing_v1' or p['source_sha256']!=sha(source)
            or p['data']['manifest_sha256']!=data['provenance']['manifest_sha256']
            or ck['protocol_sha256']!=canonical_hash(p) or p['test_loaded'] is not False):
        raise ValueError('Frozen timing provenance mismatch')
    m=RelativeAudioTiming(s['feature_mean'],s['feature_std'],s['projection'],s['channel_scales'],s['dynamic_scales'],**ck['config'])
    m.load_state_dict(s,strict=True);return m.cuda().requires_grad_(False).eval()


@torch.no_grad()
def cache_conditions(data,timing):
    for q in data['splits'].values():
        acoustic=[];state=[];delta=[]
        for ids in torch.arange(len(q['valid'])).split(32):
            b=subset(q,ids,'cuda');v=b['valid']
            f=relative_features(b['audio_features'],v,timing.feature_mean,timing.feature_std)
            acoustic.append(torch.cat((f[...,:1536]@timing.projection,f[...,-4:]),-1).cpu())
            out=timing(b['audio_features'],v)
            state.append(out['state'].cpu());delta.append(out['delta'][...,list(UPPER_INDICES)].cpu())
        q['relative_acoustic']=torch.cat(acoustic);q['relative_state']=torch.cat(state);q['relative_delta']=torch.cat(delta)


def prior_batch(q, ids):
    b=trim(q,ids,'cuda');t=b['valid'].shape[1]
    for key in ('relative_acoustic','relative_state','relative_delta'):b[key]=b[key][:,:t]
    return b


def conditions(b,adapter,flow,mode):
    v=b['valid'];x=b['relative_acoustic'].detach();state=b['relative_state'].detach()
    delta=b['relative_delta'].detach()
    if mode=='static':
        x=x*0.;state=state*0.;delta=delta*0.
    elif mode=='reverse':
        x,state,delta=x.clone(),state.clone(),delta.clone()
        for i in range(len(v)):
            ix=v[i].nonzero().flatten()
            for value in (x,state,delta):value[i,ix]=value[i,ix.flip(0)].clone()
    elif mode!='audio':raise ValueError(mode)
    local=torch.where(v[...,None],adapter(x),0.)
    cond={'valid':v,'h0':local.new_zeros(*v.shape,flow.content_dim),'channel_mask':b['channel_mask']}
    identity={'code':b['identity_code'].detach()}
    affect={'global':b['global_code'].detach(),'intensity_value':b['intensity_value'].detach()}
    return cond,identity,affect,local,state,delta


def prior_objective(flow,adapter,b,scales,mode,noise,time_value,other_noise=None):
    cond,identity,affect,local,state,delta=conditions(b,adapter,flow,mode)
    target=project_temporal_dc((b['motion'][...,list(UPPER_INDICES)].detach()-delta)/scales,b['valid'])
    loss=flow.flow_loss(target,cond,identity,affect,local,state,noise,time_value)
    if other_noise is not None:
        generated=torch.stack([flow.decode(cond,identity,affect,local,state,z,8)+delta/scales for z in (noise,other_noise)])
        # Proper trajectory score rewards fit and spread; no paired endpoint MSE.
        truth=project_temporal_dc(b['motion'][...,list(UPPER_INDICES)]/scales,b['valid'])
        loss=loss+.1*fair_trajectory_es(generated,truth,b['valid'],centered=True)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite prior objective')
    return loss


@torch.no_grad()
def evaluate(flow,adapter,data,system,audio,identities,out,arm):
    q=data['splits']['validation'];v=q['valid'];ids=torch.arange(len(v));cc=list(UPPER_INDICES)
    scales=data['target_scales'][cc].cuda();predictions={};mouth={};nonupper={};drift={};emotions={}
    for seed in SEEDS:
        noise=torch.randn(q['motion'].shape,generator=torch.Generator().manual_seed(seed))
        pieces={k:[] for k in MODES}
        for ix in ids.split(16):
            # Evaluation keeps the common full native storage clock used by
            # paired noise and curve arrays; only training batches are trimmed.
            b=subset(q,ix,'cuda');b['frozen_local']=audio(b['audio_features'],b['valid'])['local']
            base=old_prediction(system,b,identities,noise[ix].cuda())
            mean=torch.where(b['valid'][...,None],base[...,cc],0.).sum(1)/b['valid'].sum(1)[:,None]
            pieces['base'].append(base.cpu())
            det=compose_upper_face(base,mean[:,None]+b['relative_delta'],b['valid'])
            pieces['deterministic'].append(det.cpu())
            for key,mode in [('full',arm),('static','static'),('reverse','reverse')]:
                cond,identity,affect,local,state,delta=conditions(b,adapter,flow,mode)
                residual=flow.decode(cond,identity,affect,local,state,noise[ix][...,cc].cuda(),12)
                upper=mean[:,None]+delta+scales*residual
                pred=compose_upper_face(base,upper,b['valid']);pieces[key].append(pred.cpu())
        for key,values in pieces.items():predictions[f'{seed}/{key}']=torch.cat(values)
    for key,pred in predictions.items():
        base=predictions[key.split('/')[0]+'/base']
        mouth[key]=protection_report(pred,base,q['motion'],v,q['channel_mask'],q['emotion_id'])
        nonupper[key]=bool(torch.equal(pred[...,NOT_UPPER],base[...,NOT_UPPER]))
        drift[key]=float(((pred[...,cc]-base[...,cc])*v[...,None]).sum(1).div(v.sum(1)[:,None]).abs().max())
        correct=0
        for ix in ids.split(32):
            b=subset(q,ix,'cuda');ident=batch_identity(identities,b)
            e=system.encode_motion(torch.where(obs(b),pred[ix].cuda()-b['b0']-ident['baseline'][:,None],0.),b['valid'])
            correct+=int((e['emotion_logits'].argmax(-1)==b['emotion_id']).sum())
        emotions[key]=correct/len(v)
    curves={'clip_id':q['clip_id'],'target':q['motion'],'valid':v,'times':q['times'],
        'channel_mask':q['channel_mask'],'b0':q['b0'],'predictions':predictions,'noise_seeds':list(SEEDS)}
    bench=report_generation(curves,data,out,'prior',SimpleNamespace(condition_mode=arm,artifact_dir=out/'scores'))
    em={name:float(np.mean([emotions[f'{s}/{name}'] for s in SEEDS])) for name in MODES}
    checks={'mouth_preserved':all(x['passed'] for x in mouth.values()),'nonupper43_exact':all(nonupper.values()),
        'mean_preserved':max(drift.values())<1e-5,'internal_emotion_no_large_drop':em['full']>=em['base']-.03,
        'mbe_no_worse':bench['full']['coefficient']['arkit_mbe']['value']<=bench['base']['coefficient']['arkit_mbe']['value']}
    result={'clips':len(v),'checks':checks,'generated_teacher_accuracy_nonindependent':emotions,
        'max_mean_drift':drift,'mouth_protection':mouth,'protection_passed':all(checks.values()),
        'test_loaded':False,'independent_emotion_AV_visual_pending':True}
    save_json(out/'evaluation.json',result)
    manifest=json.loads((Path(data['_directory'])/'manifest.json').read_text())
    lengths={r['clip_id']:r['frames'] for r in manifest['roles']['val']['query']}
    save_checkpoint(out/'native_curves.pt',compact_curves(curves,lengths))
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ('data','source','timing','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.output.exists() and not a.resume:raise FileExistsError('Fresh paired prior output required')
    a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;random.seed(47);np.random.seed(47);torch.manual_seed(47)
    status(a.output,status='preparing')
    data,system,audio,identities,_=load_context(a.data,a.source,'cuda',47);data['_directory']=str(a.data)
    timing=load_timing(a.timing,data,a.source)
    if a.smoke:
        for key,q in data['splits'].items():data['splits'][key]=subset(q,torch.arange(16 if key=='train' else 4),'cpu')
    elif [len(data['splits'][r]['valid']) for r in ('train','validation')]!=[4098,446]:raise ValueError('Complete protocol required')
    frozen={k:state_hash(m.state_dict()) for k,m in [('system',system),('audio',audio),('timing',timing)]}
    cache_conditions(data,timing)
    cfg=copy.deepcopy(data['config']);cfg['model'].update(dit_dim=96,dit_depth=2,heads=4,dropout=.1)
    root=Path(__file__).resolve().parents[1]
    protocol={'schema':'relative_conditioned_motion_prior_v1','data':data['provenance'],'source_sha256':sha(a.source),
        'timing_sha256':sha(a.timing),'epochs':1 if a.smoke else a.epochs,'smoke':a.smoke,'seed':47,
        'model_config':cfg['model'],'objective':'zeroDC centered GT minus frozen state; FM + .1 two-draw fairES every8 batches',
        'condition':'relative PCA24+prosody4, state4, frozen global64/id128; zero h0 plus fixed native clock',
        'mean':'frozen Stage4 mean at inference','selection':'fixed final epoch; no development checkpoint selection',
        'exploratory_reason':'deterministic state not significant; assess conditional motion distribution separately',
        'frozen':frozen,'test_loaded':False,'default_replaced':False,
        'source_sha':sha(__file__),'sources':source_inventory(root)}
    if a.resume and json.loads((a.output/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol differs')
    save_json(a.output/'protocol.json',protocol)
    budget=protocol['epochs'];q=data['splits']['train'];scales=data['target_scales'][list(UPPER_INDICES)].cuda()
    for arm in ('audio','static'):
        out=a.output/arm;out.mkdir(exist_ok=True)
        if a.resume and (out/'complete.json').exists():
            marker=json.loads((out/'complete.json').read_text())
            if (marker['protocol_sha256']!=canonical_hash(protocol)
                    or marker['final_sha256']!=sha(out/'final.pt')
                    or marker.get('curves_sha256')!=sha(out/'native_curves.pt')
                    or marker.get('evaluation_sha256')!=sha(out/'evaluation.json')):
                raise ValueError('Completed arm provenance mismatch')
            continue
        torch.manual_seed(47);flow=DCProtectedTemporalFlow(cfg).cuda()
        adapter=nn.Linear(28,flow.emotion_dim).cuda()
        params=list(flow.parameters())+list(adapter.parameters())
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);rng=torch.Generator().manual_seed(47);first=0
        if a.resume and (out/'last.pt').exists():
            ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            if ck['protocol_sha256']!=canonical_hash(protocol):raise ValueError('Checkpoint protocol differs')
            flow.load_state_dict(ck['flow']);adapter.load_state_dict(ck['adapter']);opt.load_state_dict(ck['optimizer'])
            first=ck['epoch'];restore_rng(ck['rng'],rng)
        for epoch in range(first,budget):
            tick=time.monotonic();flow.train();adapter.train();losses=[]
            for bi,ids in enumerate(torch.randperm(len(q['valid']),generator=rng).split(16)):
                b=prior_batch(q,ids);shape=(*b['valid'].shape,9)
                noise=torch.randn(shape,generator=rng).cuda();tv=torch.rand(len(ids),generator=rng).cuda()
                second=torch.randn(shape,generator=rng).cuda() if bi%8==0 else None
                loss=prior_objective(flow,adapter,b,scales,arm,noise,tv,second)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                losses.append(float(loss.detach()))
            save_checkpoint(out/'last.pt',{'flow':flow.state_dict(),'adapter':adapter.state_dict(),'optimizer':opt.state_dict(),
                'epoch':epoch+1,'rng':capture_rng(rng),'protocol_sha256':canonical_hash(protocol)})
            rec={'arm':arm,'epoch':epoch+1,'epochs':budget,'clips':len(q['valid']),'loss':float(np.mean(losses)),
                'seconds':time.monotonic()-tick}
            save_json(out/f'epoch{epoch+1:03d}.json',rec);status(a.output,status='training',**rec)
        save_checkpoint(out/'final.pt',{'flow':flow.state_dict(),'adapter':adapter.state_dict(),'protocol':protocol,
            'protocol_sha256':canonical_hash(protocol)})
        status(a.output,status='evaluating',arm=arm)
        evaluate(flow.eval(),adapter.eval(),data,system,audio,identities,out,arm)
        save_json(out/'complete.json',{'protocol_sha256':canonical_hash(protocol),'final_sha256':sha(out/'final.pt'),
            'curves_sha256':sha(out/'native_curves.pt'),'evaluation_sha256':sha(out/'evaluation.json')})
        del flow,adapter,opt
    def read(arm,mode):return json.loads((a.output/arm/'scores'/'prior'/f'temporal_{mode}.json').read_text())
    full=read('audio','full');controls={'trained_static':read('static','full'),'same_static':read('audio','static'),'reverse':read('audio','reverse')}
    pairs={k:{name:interval(full,v,path) for name,path in [('centered_es',('joint_fair_es','centered')),
            ('variogram',('variogram','aggregate'))]} for k,v in controls.items()}
    checks={arm:json.loads((a.output/arm/'evaluation.json').read_text())['checks'] for arm in ('audio','static')}
    passed=all(v['passed'] for c in pairs.values() for v in c.values()) and all(all(c.values()) for c in checks.values()) and not a.smoke
    for name,m in [('system',system),('audio',audio),('timing',timing)]:
        if state_hash(m.state_dict())!=frozen[name]:raise RuntimeError('Frozen source changed')
    save_json(a.output/'acceptance.json',{'quantitative_passed':passed,'pairs':pairs,'protection':checks,
        'test_loaded':False,'independent_emotion_AV_visual_pending':True,'default_replaced':False})
    status(a.output,status='complete',quantitative_passed=passed,elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':main()
