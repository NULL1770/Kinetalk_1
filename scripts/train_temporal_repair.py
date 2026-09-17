"""Frozen-teacher local transfer and matched free upper-face flow experiments.

Additive follow-up to run12. No hard state lift, no target-conditioned decode
in deployable results, no renderer update in local transfer, no default edits.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.aligned_audio_local import AlignedAudioLocal
from kinetalk_b0.models.temporal_upper import TemporalUpperFlow, UPPER_INDICES, compose_upper_face
from scripts.diagnose_full_staged_transfer import _make_audio, sequence_metrics
from scripts.full_staged_data import load_training_inputs, sha
from scripts.train_full_staged import (
    subset, obs, mse, optimize, cache_current_base, identity_cache, batch_identity,
    teacher_affect, region_report, identity_report, NOT_UPPER,
)
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, canonical_hash, capture_rng, restore_rng
from scripts.train_predictable_renderer import state_hash

SCHEMA = 'aligned_local_free_upper_v1'
SEEDS = (42, 123, 2026)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'audio', 'targets', 'enrollment', 'native-root', 'trained-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--phase', choices=('align', 'direct', 'soft'), required=True)
    p.add_argument('--align-run', type=Path)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=53)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def matching_checkpoints(run, data):
    record=json.loads((run/'provenance.json').read_text()); recipe=record['recipe']
    if canonical_hash(recipe)!=record['recipe_sha256']:
        raise ValueError('Source recipe differs')
    if data['provenance']['input_sha256']!=recipe['data_provenance']['input_sha256']:
        raise ValueError('Data differs from run12')
    payload={};bindings={}
    for stage in ('teacher','audio'):
        path=run/stage/'final.pt';complete=json.loads(path.with_name('complete.json').read_text())
        digest=sha(path)
        if digest!=complete['final_sha256']:raise ValueError('Source checkpoint binding differs')
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value['config']!=data['config'] or value['recipe_sha256']!=record['recipe_sha256']:
            raise ValueError('Source configuration differs')
        if not torch.equal(value['scales'],data['target_scales']):raise ValueError('Source scales differ')
        payload[stage]=value;bindings[stage]={'path':str(path.resolve()),'sha256':digest}
    a={k:v for k,v in payload['teacher']['system'].items() if not k.startswith('renderer.')}
    b={k:v for k,v in payload['audio']['system'].items() if not k.startswith('renderer.')}
    if state_hash(a)!=state_hash(b):raise ValueError('Teacher/Audio frozen subsystem differs')
    return payload,bindings,int(recipe['args']['decode_steps']),int(recipe['stride_frames'])


@torch.no_grad()
def cache_conditions(system,audio,data,identities,device,batch_size):
    for role,q in data['splits'].items():
        values={k:[] for k in ('audio_global','audio_intensity','audio_local','audio_state',
                               'teacher_global','teacher_intensity','teacher_local','teacher_controls','control_weight')}
        for ix in torch.arange(len(q['valid'])).split(batch_size):
            b=subset(q,ix,device);ident=batch_identity(identities,b)
            a=audio(b['audio_features'],b['valid']);t=teacher_affect(system,b,ident)
            for key,val in (('audio_global',a['global']),('audio_intensity',a['intensity_value']),
                            ('audio_local',a['local']),('audio_state',a['state']),
                            ('teacher_global',t['global']),('teacher_intensity',t['intensity_value']),
                            ('teacher_local',t['local']),('teacher_controls',t['controls']),('control_weight',t['control_weight'])):
                values[key].append(val.cpu())
        q.update({k:torch.cat(v) for k,v in values.items()})


def projected_affect(system,adapter,b):
    global_affect={'global':b['audio_global'],'intensity_value':b['audio_intensity']}
    return system.project_affect({**global_affect,**adapter(b['audio_features'],b['valid'])},b['valid'])


def control_scale(q):
    weight=q['control_weight'][...,None]
    return ((q['teacher_controls'].square()*weight).sum((0,1))/weight.sum()).sqrt().clamp_min(.02)


def control_loss(pred,target,weight,scale):
    if pred.shape!=target.shape or weight.shape!=pred.shape[:2]:raise ValueError('Control target shape differs')
    error=((pred-target)/scale).square()*weight[...,None]
    return error.sum()/(weight.sum()*pred.shape[-1]).clamp_min(1)


def upper_target(b,scales):
    cc=list(UPPER_INDICES)
    if not (b['channel_mask'][:,cc]&b['anchor_valid'][:,cc]).all():raise ValueError('Upper target channels unavailable')
    return torch.where(b['valid'][...,None],(b['motion'][...,cc]-b['anchors'][:,None,cc])/scales[cc],0.)


def upper_decode(upper,local,b,ident,affect,scales,noise,steps,mode='full'):
    native=local(b['audio_features'],b['valid'])['local']
    state=b['audio_state']
    if mode=='static':
        native=(native.sum(1,keepdim=True)/b['valid'].sum(1)[:,None,None])*b['valid'][...,None]
        state=(state.sum(1,keepdim=True)/b['valid'].sum(1)[:,None,None])*b['valid'][...,None]
    elif mode=='reverse':
        native=native.clone();state=state.clone()
        for row in range(len(native)):
            valid=b['valid'][row].nonzero(as_tuple=True)[0]
            native[row,valid]=native[row,valid.flip(0)];state[row,valid]=state[row,valid.flip(0)]
    elif mode=='zero_state':state=torch.zeros_like(state)
    output=upper.decode(b['valid'],b['h0'],ident['code'],affect,native,state,noise[...,list(UPPER_INDICES)],steps=steps)
    return b['anchors'][:,None,list(UPPER_INDICES)]+scales[list(UPPER_INDICES)]*output


def populations(pred,b):
    result={'all':region_report(pred,b)}
    for name,choose in (('neutral',b['emotion_id']==0),('nonneutral',b['emotion_id']!=0)):
        ids=choose.nonzero(as_tuple=True)[0]
        if len(ids):result[name]=region_report(pred[ids],subset(b,ids,'cpu'))
    return result


@torch.no_grad()
def evaluate(system,adapter,upper,local,data,identities,args,scales,steps,*,full,role='validation'):
    q=data['splits'][role];count=len(q['valid']) if full else min(64,len(q['valid']))
    ids=torch.randperm(len(q['valid']),generator=torch.Generator().manual_seed(20260917))[:count].sort().values
    reference=subset(q,ids,'cpu');predictions={};metrics={};locals_pred=[];locals_true=[]
    seeds=SEEDS if full else (42,)
    for seed in seeds:
        noise=torch.randn(len(q['valid']),q['valid'].shape[1],52,generator=torch.Generator().manual_seed(seed))
        modes=['full']
        if full and seed==42:modes+=['original_local','zero_local','oracle_local'] if upper is None else ['base','static','reverse','zero_state']
        for mode in modes:
            out=[];readout=[]
            for ix in ids.split(args.batch_size):
                b=subset(q,ix,args.device);ident=batch_identity(identities,b);n=noise[ix].to(args.device)
                affect=projected_affect(system,adapter,b)
                if mode in ('original_local','zero_local','oracle_local'):
                    affect['local']=b['audio_local'] if mode=='original_local' else torch.zeros_like(b['audio_local']) if mode=='zero_local' else b['teacher_local']
                pred=system.generate(b['content'],b['valid'],ident,affect,initial_noise=n,steps=steps,base={'b0':b['b0'],'h0':b['h0']})['motion']
                if upper is not None and mode!='base':
                    base=pred
                    uv=upper_decode(upper,local,b,ident,affect,scales,n,steps,mode)
                    pred=compose_upper_face(base,uv,b['valid'])
                    if not torch.equal(pred[...,list(NOT_UPPER)],base[...,list(NOT_UPPER)]):raise RuntimeError('Nonupper output changed')
                if seed==42 and mode=='full':
                    locals_pred.append(affect['local'].cpu());locals_true.append(b['teacher_local'].cpu())
                t=system.encode_motion(torch.where(obs(b),pred-b['b0']-ident['baseline'][:,None],0.),b['valid'])
                readout.extend((t['emotion_logits'].argmax(-1)==b['emotion_id']).cpu().tolist())
                out.append(pred.cpu())
            key=f'{seed}/{mode}';predictions[key]=torch.cat(out)
            metrics[key]={'populations':populations(predictions[key],reference),
                          'generated_emotion_accuracy_nonindependent':sum(readout)/len(readout)}
    report={'schema':SCHEMA,'phase':args.phase,'clips':count,'role':role,'noise_seeds':list(seeds),
            'modes':metrics,'local_prediction':sequence_metrics(torch.cat(locals_pred),torch.cat(locals_true),reference['valid']),
            'oracle_local_is_target_conditioned':True,'test_loaded':False}
    curves={'schema':SCHEMA,'phase':args.phase,'clip_id':reference['clip_id'],'target':reference['motion'],
            'valid':reference['valid'],'channel_mask':reference['channel_mask'],'times':reference['times'],
            'emotion_id':reference['emotion_id'],'speaker_id':reference['speaker_id'],'b0':reference['b0'],
            'predictions':predictions,'noise_seeds':list(seeds),'decode_steps':steps}
    return report,curves


def main():
    args=arguments();started=time.monotonic()
    if args.output.exists() and not args.resume:raise FileExistsError('Fresh output required')
    if args.epochs<1 or args.batch_size<1:raise ValueError('Positive epochs/batch required')
    if args.phase!='align' and args.align_run is None:raise ValueError('Upper phases require completed align-run')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    data=load_training_inputs(args.source_run,args.audio,args.targets,args.enrollment,args.native_root)
    checkpoints,bindings,steps,stride=matching_checkpoints(args.trained_run,data)
    system=data['system'].to(args.device).eval().requires_grad_(False)
    system.load_state_dict(checkpoints['teacher']['system'])
    audio=_make_audio(checkpoints['audio']['audio'],stride,args.device)
    adapter=AlignedAudioLocal(audio.feature_mean,audio.feature_std,hidden=audio.input.out_features,
                              rank=system.motion_teacher.rank,stride=system.motion_teacher.stride).to(args.device)
    adapter.initialize_from_slow(audio).eval()
    align_binding=None
    if args.phase!='align':
        path=args.align_run/'final.pt';complete=json.loads((args.align_run/'complete.json').read_text())
        if sha(path)!=complete['final_sha256']:raise ValueError('Alignment checkpoint hash differs')
        aligned=torch.load(path,map_location='cpu',weights_only=False)
        if aligned['phase']!='align' or aligned['source_bindings']!=bindings:raise ValueError('Alignment lineage differs')
        if aligned['config']!=data['config']:raise ValueError('Alignment config differs')
        adapter.load_state_dict(aligned['adapter']);align_binding={'path':str(path.resolve()),'sha256':sha(path)}
    cache_current_base(system,data,args.device,args.batch_size);identities=identity_cache(system,data,args.device)
    cache_conditions(system,audio,data,identities,args.device,args.batch_size)
    scales=data['target_scales'].to(args.device);cscale=control_scale(data['splits']['train']).to(args.device)
    upper=local=None
    if args.phase=='align':
        adapter.requires_grad_(True);parameters=list(adapter.parameters());lr=1e-4
    else:
        adapter.requires_grad_(False)
        # Both arms begin identically; only the optional state feature differs.
        local=copy.deepcopy(audio).requires_grad_(False)
        local.input.load_state_dict(adapter.input.state_dict());local.blocks.load_state_dict(adapter.blocks.state_dict())
        torch.nn.init.zeros_(local.local_head.weight);torch.nn.init.zeros_(local.local_head.bias)
        for module in (local.input,local.blocks,local.local_head):module.requires_grad_(True)
        upper=TemporalUpperFlow(data['config'],use_state=args.phase=='soft').to(args.device).eval()
        parameters=list(upper.parameters())+[p for p in local.parameters() if p.requires_grad];lr=1e-4
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=1e-5)
    gen=torch.Generator().manual_seed(args.seed)
    freeze_modules={'system':system,'audio':audio}
    if upper is not None:freeze_modules['adapter']=adapter
    frozen={k:state_hash(m.state_dict()) for k,m in freeze_modules.items()}
    args.output.mkdir(parents=True,exist_ok=True)
    recipe={'schema':SCHEMA,'phase':args.phase,'epochs':args.epochs,'seed':args.seed,'smoke':args.smoke,
            'batch_size':args.batch_size,'source_bindings':bindings,'alignment':align_binding,
            'data_provenance':data['provenance'],'steps':steps,'frozen':frozen,
            'control_scale':cscale.cpu().tolist(),'target_scales':scales.cpu().tolist(),
            'objective':'train-scale-normalized teacher rank8 control MSE' if upper is None else 'free9 flow matching only',
            'upper_target':'(motion upper - independent neutral anchor) / train scale',
            'soft_state':'Frozen Stage4 audio state, same deployable prediction at fit and inference; no GT override',
            'time_interventions':'reverse/static affect the upper native local and soft state, not frozen content/global/base',
            'source_sha256':{f:sha(Path(__file__).resolve().parents[1]/f) for f in (
                'scripts/train_temporal_repair.py','scripts/full_staged_data.py','kinetalk_b0/models/aligned_audio_local.py',
                'kinetalk_b0/models/temporal_upper.py','kinetalk_b0/models/slow_state_affect.py','kinetalk_b0/models/dit.py')},
            'test_loaded':False,'default_replaced':False,'fixed_final_epoch':True}
    digest=canonical_hash(recipe);begin_epoch=0;total_steps=0
    if args.resume:
        saved=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        if saved['recipe_sha256']!=digest:raise ValueError('Resume recipe differs')
        adapter.load_state_dict(saved['adapter'])
        if upper is not None:upper.load_state_dict(saved['upper']);local.load_state_dict(saved['local'])
        optimizer.load_state_dict(saved['optimizer'])
        for values in optimizer.state.values():
            for k,v in values.items():
                if torch.is_tensor(v):values[k]=v.to(args.device)
        restore_rng(saved['rng'],gen);begin_epoch=saved['completed_epochs'];total_steps=saved['total_steps']
        if not 0 <= begin_epoch <= (1 if args.smoke else args.epochs):raise ValueError('Resume completed epoch is outside this run budget')
        # A completed checkpoint may resume only the final evaluation/export.
        # Keep its payload even when the training loop has no remaining epoch.
        payload=saved
    else:
        save_json(args.output/'provenance.json',{'recipe':recipe,'recipe_sha256':digest})
        save_checkpoint(args.output/'initial.pt',{'adapter':adapter.state_dict(),
            'upper':upper.state_dict() if upper is not None else None,'local':local.state_dict() if local is not None else None})
    n=len(data['splits']['train']['valid']);items=torch.arange(min(n,args.batch_size*2) if args.smoke else n)
    epochs=1 if args.smoke else args.epochs
    try:
        for epoch in range(begin_epoch,epochs):
            start=time.monotonic();sums=[];ordering=items[torch.randperm(len(items),generator=gen)]
            random_hash=hashlib.sha256()
            for bi,ix in enumerate(ordering.split(args.batch_size)):
                random_hash.update(ix.numpy().tobytes());b=subset(data['splits']['train'],ix,args.device)
                if upper is None:
                    out=adapter(b['audio_features'],b['valid'])
                    loss=control_loss(out['controls'],b['teacher_controls'],out['control_weight'],cscale)
                else:
                    ident=batch_identity(identities,b)
                    affect={'global':b['audio_global'],'intensity_value':b['audio_intensity']}
                    native=local(b['audio_features'],b['valid'])['local']
                    noise=torch.randn(len(ix),b['valid'].shape[1],9,generator=gen);ft=torch.rand(len(ix),generator=gen)
                    random_hash.update(noise.numpy().tobytes());random_hash.update(ft.numpy().tobytes())
                    loss=upper.flow_loss(upper_target(b,scales),b['valid'],b['h0'],ident['code'],affect,native,b['audio_state'],noise.to(args.device),ft.to(args.device))
                norm=optimize(loss,optimizer,parameters);sums.append(float(loss.detach()));total_steps+=1
                if bi%25==0:print(json.dumps({'event':'batch','phase':args.phase,'epoch':epoch+1,'batch':bi+1,'loss':sums[-1],'grad_norm':norm}),flush=True)
            row={'phase':args.phase,'epoch':epoch+1,'loss':sum(sums)/len(sums),'batches':len(sums),
                 'seconds':time.monotonic()-start,'batch_noise_time_sha256':random_hash.hexdigest(),'total_steps':total_steps}
            save_json(args.output/f'epoch{epoch+1:03d}.json',row)
            payload={'schema':SCHEMA,'recipe_sha256':digest,'phase':args.phase,'source_bindings':bindings,
                'adapter':adapter.state_dict(),'upper':upper.state_dict() if upper is not None else None,
                'local':local.state_dict() if local is not None else None,'config':data['config'],'scales':scales.cpu(),
                'control_scale':cscale.cpu(),'completed_epochs':epoch+1,'total_steps':total_steps,
                'optimizer':optimizer.state_dict(),'rng':capture_rng(gen)}
            save_checkpoint(args.output/'last.pt',payload);save_json(args.output/'status.json',{'status':'running',**row})
            print(json.dumps({'event':'epoch_complete',**row}),flush=True)
            if (epoch+1)%4==0 or args.smoke:
                report,_=evaluate(system,adapter,upper,local,data,identities,args,scales,steps,full=False)
                save_json(args.output/f'dev_epoch{epoch+1:03d}.json',report)
        for name,module in freeze_modules.items():
            if state_hash(module.state_dict())!=frozen[name]:raise RuntimeError('Frozen module changed: '+name)
            if any(p.grad is not None for p in module.parameters()):raise RuntimeError('Frozen module gradient: '+name)
        report,curves=evaluate(system,adapter,upper,local,data,identities,args,scales,steps,full=not args.smoke)
        report['identity']=identity_report(system,data,args.device)
        save_json(args.output/'evaluation.json',report);save_checkpoint(args.output/'curves.pt',curves)
        final={k:v for k,v in payload.items() if k not in ('optimizer','rng')}
        save_checkpoint(args.output/'final.pt',final)
        save_json(args.output/'complete.json',{'schema':SCHEMA,'phase':args.phase,'final_sha256':sha(args.output/'final.pt'),
            'curves_sha256':sha(args.output/'curves.pt'),'completed_epochs':epochs,'frozen':frozen})
        save_json(args.output/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,'phase':args.phase,'epochs':epochs})
        print('TEMPORAL_REPAIR_COMPLETE',flush=True)
    except BaseException as exc:
        save_json(args.output/'failure.json',{'phase':args.phase,'error':repr(exc),'recover_from':'last.pt'})
        raise


if __name__=='__main__':main()
