"""Matched fixed-final prefix continuation after the bounded receiver pilot.

Warmstarts both arms from the original shared history12 no-history model, never
the eight-clip pilot weights. The pilot gates mechanism exploration only.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_prefix_upper as p
from scripts.audit_temporal_repair import read, sha, load_pt, metadata_equal
from scripts.extend_prefix_pilot import corrected_gate
from scripts.evaluate_prefix_formal import evaluate_formal


def validate_pilot(pilot):
    status=read(pilot/'status.json')
    if status['status']!='complete' or status['total_epochs']!=30 or status['additional_epochs']!=15:
        raise ValueError('Require completed bounded pilot extension')
    curves={}; reports={}; bindings={}; selections={}; sources={}
    for arm in ('no_prefix','teacher_prefix'):
        path=pilot/arm; complete=read(path/'complete.json'); record=read(path/'provenance.json')
        if p.canonical_hash(record['recipe'])!=record['recipe_sha256'] or complete['recipe_sha256']!=record['recipe_sha256']:
            raise ValueError('Pilot extension recipe differs')
        for name in ('curves','final'):
            value=sha(path/(name+'.pt'))
            if value!=complete[name+'_sha256']:raise ValueError('Pilot binding changed')
            bindings[arm+'/'+name]=value
        curves[arm]=load_pt(path/'curves.pt'); reports[arm]=read(path/'evaluation.json')
        bindings[arm+'/evaluation']=sha(path/'evaluation.json')
        original=read(pilot.parent/'prefix_pilot15'/arm/'provenance.json')
        if (p.canonical_hash(original['recipe'])!=original['recipe_sha256'] or
            original['recipe_sha256']!=record['recipe']['source_recipe_sha256']):
            raise ValueError('Original pilot recipe binding differs')
        selections[arm]=[row['clip_id'] for row in record['recipe']['selection']['clips']]
        if selections[arm]!=original['recipe']['fit_ids'] or selections[arm]!=curves[arm]['clip_id']:
            raise ValueError('Pilot selected membership changed')
        sources[arm]=original['recipe']['source_sha256']
        # Recompute the gate's centered term from the hash-bound raw curves.
        for mode in ('full','oracle_history'):
            for group in ('brows','eyes_expression'):
                cc=list(p.GROUPS[group]);c=curves[arm]
                metrics=p.paired_metrics(c['predictions']['42/'+mode][...,cc],c['target'][...,cc],
                    c['valid'][...,None]&c['channel_mask'][:,None,cc])
                reports[arm]['modes']['42/'+mode]['metrics'][group]['centered_mse']=metrics['centered_mse']
    result=corrected_gate(curves,reports)
    if result!=read(pilot/'receiver_gate.json') or not result['passed']:
        raise ValueError('Receiver gate not passed or score changed')
    if not read(pilot/'matched_audit.json')['random_streams_equal']:
        raise ValueError('Pilot pairing failed')
    if selections['no_prefix']!=selections['teacher_prefix'] or sources['no_prefix']!=sources['teacher_prefix']:
        raise ValueError('Pilot arms used different membership or source')
    return {'path':str(pilot.resolve()),'files':bindings,'gate':result,'gate_sha256':sha(pilot/'receiver_gate.json'),
        'selected_fit_ids':selections['no_prefix'],'original_warmstart_sha256':sources['no_prefix'],
        'use':'Only mechanism gate. No pilot checkpoint is loaded as model initialization.'}


def training_batch(upper,b,ident,native,scales,generator,steps,probability,use_prefix):
    if b['valid'].shape[1] != 96 or not b['valid'].any() or not 0 <= probability <= 1:
        raise ValueError('Formal batch requires observed 96-frame sequences and a probability in [0,1]')
    target=p.h.normalized_target(b,scales); count=len(target)
    noise=torch.randn(count,6,24,9,generator=generator)
    synthetic_noise=torch.randn(target.shape,generator=generator)
    times=torch.rand(count,6,generator=generator); decisions=torch.rand(count,6,generator=generator)
    generated=p.rollout(upper,b,ident,native,synthetic_noise.to(target),steps=steps,use_prefix=use_prefix)
    loss=target.new_zeros(()); counts={key:0 for key in (
        'teacher_draw_count','valid_history_count','teacher_eligible_count','teacher_used_count')}
    for ci,start in enumerate(range(0,96,16)):
        ids=b['valid'][:,start:start+16].any(1).nonzero(as_tuple=True)[0]
        if not len(ids):continue
        choose=decisions[:,ci].to(target.device)<probability
        source_history=torch.where(choose[:,None,None],target,generated)
        has_history=b['valid'][ids,max(0,start-8):start].any(1)
        counts['teacher_draw_count']+=int(choose[ids].sum())
        counts['valid_history_count']+=int(has_history.sum())
        eligible=int((choose[ids]&has_history).sum())
        counts['teacher_eligible_count']+=eligible
        counts['teacher_used_count']+=eligible if use_prefix else 0
        part=p.patch_loss(upper,b,ident,native,target,source_history,ids,torch.full_like(ids,start),
            noise[ids.cpu(),ci].to(target),times[ids.cpu(),ci].to(target),empty=not use_prefix)
        loss=loss+part*b['valid'][ids,start:start+16].sum()
    return loss/b['valid'].sum(),counts,(noise,synthetic_noise,times,decisions)


def load_shared_warmstart(upper, state):
    expected_discarded = {
        'history_input.0.weight', 'history_input.0.bias',
        'history_input.2.weight', 'history_input.2.bias',
        'history_gru.weight_ih', 'history_gru.weight_hh',
        'history_gru.bias_ih', 'history_gru.bias_hh', 'history_projection.weight'}
    discarded = set(state)-set(upper.state_dict())
    if discarded != expected_discarded:
        raise ValueError('Warmstart contains unexpected discarded history keys')
    missing = upper.load_state_dict({key:value for key,value in state.items() if key not in discarded},strict=False)
    if set(missing.missing_keys)!={'known_embedding.weight'} or missing.unexpected_keys:
        raise ValueError('Warmstart shared keys differ')
    if torch.count_nonzero(upper.known_embedding.weight):
        raise ValueError('Fresh known/unknown embedding must be zero')
    return sorted(discarded)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','audio','targets','enrollment','native-root','trained-run','history-run','centered-run','pilot-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=12); parser.add_argument('--seed',type=int,default=79)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.pilot=False;args.pilot_selection=None;started=time.monotonic()
    if args.output.exists():raise FileExistsError('Fresh formal prefix output required')
    if args.epochs!=12 or args.batch_size!=16 or args.seed!=79:
        raise ValueError('Locked formal budget is 12 epochs, batch16, seed79')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    pilot_binding=validate_pilot(args.pilot_run)
    data,system,audio,local,identities,source,source_recipe,source_sha,steps,_,frozen=p.load_context(args)
    if steps != 12:raise ValueError('Formal protocol requires 12 solver steps')
    if (pilot_binding['original_warmstart_sha256']!=source_sha or
        not set(pilot_binding['selected_fit_ids'])<=set(data['splits']['train']['clip_id'])):
        raise ValueError('Pilot source or fit membership differs from formal data')
    if source_recipe['data_provenance']['input_sha256']!=data['provenance']['input_sha256']:
        raise ValueError('Warmstart data bindings differ')
    path=args.centered_run/'white/curves.pt'
    if sha(path)!=read(path.with_name('complete.json'))['curves_sha256'] or sha(path)!=source_recipe['baseline_curves_sha256']:
        raise ValueError('Original baseline curve binding differs')
    bases=load_pt(path);q=data['splits']['validation'];metadata_equal({**q,'target':q['motion']},bases)
    if len(data['splits']['train']['valid'])!=2315 or len(q['valid'])!=405:raise ValueError('Locked role counts changed')
    if args.smoke:
        data['splits']={role:p.r.subset(query,torch.arange(32),'cpu') for role,query in data['splits'].items()}
        bases={**bases,**p.r.subset({key:bases[key] for key in (
            'clip_id','target','valid','times','channel_mask','b0','emotion_id','speaker_id')},torch.arange(32),'cpu'),
            'predictions':{key:value[:32] for key,value in bases['predictions'].items()}}
    args.output.mkdir(parents=True);scales=source['scales'].to(args.device);matches={};epochs=1 if args.smoke else args.epochs
    for arm in ('no_prefix','scheduled_prefix'):
        torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
        upper=p.PrefixUpperFlow(data['config']).to(args.device).eval()
        discarded=load_shared_warmstart(upper,source['upper'])
        output=args.output/arm;output.mkdir();initial=p.state_hash(upper.state_dict())
        p.save_checkpoint(output/'initial.pt',{'upper':upper.state_dict()})
        parameters=list(upper.parameters());optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-5)
        recipe={'schema':'prefix_formal_v1','arm':arm,'epochs':epochs,'requested_epochs':12,'batch_size':16,'seed':args.seed,
            'smoke':args.smoke,'fixed_final_epoch':True,'test_loaded':False,'default_replaced':False,
            'chunk':16,'history':8,'decode_steps':steps,'config':data['config'],'scales':scales.cpu().tolist(),
            'initial':initial,'initial_file_sha256':sha(output/'initial.pt'),'frozen':frozen,'local_frozen':True,
            'explicit_discarded_source_keys':discarded,
            'warmstart_file':str((args.history_run/'no_history/final.pt').resolve()),'warmstart_sha256':source_sha,
            'warmstart_recipe_sha256':p.canonical_hash(source_recipe),'pilot_binding':pilot_binding,
            'data_provenance':data['provenance'],'baseline_curves_sha256':sha(path),
            'teacher_probabilities':[p.h.teacher_probability(e,12) for e in range(epochs)],
            'interpretation':'Paired previous-token context comparison; absent tokens also remove their acoustic context.',
            'code_sha256':{name:sha(Path(__file__).resolve().parents[1]/name) for name in (
                'scripts/train_prefix_formal.py','scripts/train_prefix_upper.py','scripts/evaluate_prefix_formal.py',
                'scripts/audit_history_upper.py','scripts/audit_temporal_repair.py','scripts/train_history_upper.py',
                'kinetalk_b0/models/prefix_upper_flow.py','kinetalk_b0/models/temporal_upper.py','docs/PREFIX_FORMAL_PROTOCOL_20260917.md')}}
        digest=p.canonical_hash(recipe);p.save_json(output/'provenance.json',{'recipe':recipe,'recipe_sha256':digest})
        generator=torch.Generator().manual_seed(args.seed);train=data['splits']['train'];draws=[];total_steps=0
        for epoch in range(epochs):
            t0=time.monotonic();losses=[];draw=hashlib.sha256();counts={key:0 for key in (
                'teacher_draw_count','valid_history_count','teacher_eligible_count','teacher_used_count')}
            probability=recipe['teacher_probabilities'][epoch]
            order=torch.randperm(len(train['valid']),generator=generator)
            for selected in order.split(16):
                b=p.r.subset(train,selected,args.device);ident=p.r.batch_identity(identities,b)
                loss,used,randoms=training_batch(upper,b,ident,b['prefix_local'],scales,generator,steps,probability,arm!='no_prefix')
                for tensor in (selected,*randoms):draw.update(tensor.numpy().tobytes())
                p.r.optimize(loss,optimizer,parameters);losses.append(float(loss.detach()));total_steps+=1
                for key,value in used.items():counts[key]+=value
            draws.append(draw.hexdigest());row={'arm':arm,'epoch':epoch+1,'loss':sum(losses)/len(losses),
                'seconds':time.monotonic()-t0,'total_steps':total_steps,'draw_sha256':draw.hexdigest(),
                'teacher_probability':probability,**counts,'no_prefix_ignores_history':arm=='no_prefix'}
            p.save_json(output/f'epoch{epoch+1:03d}.json',row);p.save_json(args.output/'status.json',{'status':'training',**row})
            payload={'schema':'prefix_formal_v1','arm':arm,'upper':upper.state_dict(),'scales':scales.cpu(),'recipe_sha256':digest,
                'completed_epochs':epoch+1,'total_steps':total_steps,'warmstart_sha256':source_sha,'frozen':frozen}
            p.save_checkpoint(output/'last.pt',{**payload,'optimizer':optimizer.state_dict(),'rng':p.capture_rng(generator)})
            print(p.json.dumps(row),flush=True)
        for name,module in [('system',system),('audio',audio),('local',local)]:
            if p.state_hash(module.state_dict())!=frozen[name] or any(param.grad is not None for param in module.parameters()):raise RuntimeError('Frozen module changed')
        p.save_json(args.output/'status.json',{'status':'evaluating','arm':arm,'completed_epochs':epochs})
        report,curves=evaluate_formal(upper,system,data['splits']['validation'],identities,scales,bases,args,steps,arm!='no_prefix')
        report.update(arm=arm,schema='prefix_formal_v1',smoke=args.smoke,recipe_sha256=digest)
        curves.update(arm=arm,formal_schema='prefix_formal_v1',recipe_sha256=digest)
        p.save_json(output/'evaluation.json',report);p.save_checkpoint(output/'curves.pt',curves);p.save_checkpoint(output/'final.pt',payload)
        p.save_json(output/'complete.json',{'final_sha256':sha(output/'final.pt'),'curves_sha256':sha(output/'curves.pt'),
            'completed_epochs':epochs,'recipe_sha256':digest,'frozen':frozen})
        matches[arm]={'initial':initial,'draws':draws}
    if matches['no_prefix']!=matches['scheduled_prefix']:raise RuntimeError('Formal pairing failed')
    p.save_json(args.output/'matched_audit.json',{'equal':True,'arms':matches})
    p.save_json(args.output/'status.json',{'status':'complete','smoke':args.smoke,'epochs_per_arm':epochs,'seconds':time.monotonic()-started})
    print('PREFIX_FORMAL_COMPLETE',flush=True)


if __name__=='__main__':main()
