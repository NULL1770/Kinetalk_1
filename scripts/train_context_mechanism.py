"""Fixed-budget diagnostic: absent past, teacher past, and one whole-window flow.

Preserves the old source and outputs. Teacher motion is training/explicit oracle
only; all normal inference uses generated past or no motion observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_prefix_upper as p
from scripts.train_prefix_formal import load_shared_warmstart
from scripts.audit_temporal_repair import read,sha,load_pt,metadata_equal
from scripts.evaluate_context_mechanism import evaluate

ARMS=('chunk_empty','chunk_teacher','whole')
MODES=('full','empty','reverse_history','static','reverse','oracle_history','oracle_reverse_history')
SCHEMA='context_mechanism_v1'


def whole_conditions(b,ident,native):
    valid=torch.nn.functional.pad(b['valid'],(8,0),value=False)
    h0=torch.nn.functional.pad(b['h0'],(0,0,8,0))
    local=torch.nn.functional.pad(native,(0,0,8,0))
    known=native.new_zeros(len(valid),valid.shape[1],9)
    mask=torch.zeros_like(valid)
    affect={'global':b['audio_global'],'intensity_value':b['audio_intensity']}
    return (valid,h0,ident['code'],affect,local),known,mask


@torch.no_grad()
def decode_context(upper,b,ident,native,noise,*,steps=12,mode='full',arm,oracle_target=None):
    if arm not in ARMS or mode not in MODES:raise ValueError('Unknown arm or mode')
    is_oracle=mode in ('oracle_history','oracle_reverse_history')
    if not is_oracle and oracle_target is not None:raise ValueError('Non-oracle mode rejects GT history')
    if arm=='chunk_teacher' and is_oracle and oracle_target is None:
        raise ValueError('Teacher oracle requires explicit target')
    effective=mode if mode in ('static','reverse') else 'full'
    local,content=p.time_intervention(native,b['h0'],b['valid'],effective)
    cb={**b,'h0':content}
    if arm=='whole':
        conditions,known,mask=whole_conditions(cb,ident,local)
        padded=torch.nn.functional.pad(noise,(0,0,8,0))
        return upper.decode_prefix(*conditions,padded,known=known,known_mask=mask,steps=steps)[:,8:]
    output=torch.zeros_like(noise)
    for start in range(0,noise.shape[1],16):
        stop=min(start+16,noise.shape[1]);ids=b['valid'][:,start:stop].any(1).nonzero(as_tuple=True)[0]
        if not len(ids):continue
        source=oracle_target if is_oracle and arm=='chunk_teacher' else output
        conditions,known,mask,_=p.window_batch(cb,ident,local,source,ids,torch.full_like(ids,start),
            empty=arm=='chunk_empty' or mode=='empty',reverse=mode in ('reverse_history','oracle_reverse_history'))
        initial=known.new_zeros(known.shape);initial[:,8:8+stop-start]=noise[ids,start:stop]
        result=upper.decode_prefix(*conditions,initial,known=known,known_mask=mask,steps=steps)
        if not torch.equal(result[mask],known[mask]):raise RuntimeError('Known prefix changed')
        output[ids,start:stop]=result[:,8:8+stop-start]
    return output


def context_batch(upper,b,ident,native,scales,generator,arm):
    if arm not in ARMS or b['valid'].shape[1]!=96 or not b['valid'].any(1).all():
        raise ValueError('Expected known arm and observed 96-frame clips')
    target=p.h.normalized_target(b,scales)
    noise=torch.randn(target.shape,generator=generator);times=torch.rand(len(target),generator=generator)
    if arm=='whole':
        conditions,known,mask=whole_conditions(b,ident,native)
        loss=upper.flow_loss_prefix(torch.nn.functional.pad(target,(0,0,8,0)),*conditions,
            torch.nn.functional.pad(noise.to(target),(0,0,8,0)),times.to(target),known=known,known_mask=mask)
    else:
        loss=target.new_zeros(())
        for start in range(0,96,16):
            ids=b['valid'][:,start:start+16].any(1).nonzero(as_tuple=True)[0]
            if not len(ids):continue
            current_noise=torch.nn.functional.pad(noise[ids.cpu(),start:start+16].to(target),(0,0,8,0))
            part=p.patch_loss(upper,b,ident,native,target,target,ids,torch.full_like(ids,start),
                current_noise,times[ids.cpu()].to(target),empty=arm=='chunk_empty')
            loss=loss+part*b['valid'][ids,start:start+16].sum()
        loss=loss/b['valid'].sum()
    return loss,(noise,times)


def fixed_fit_selection(q,count=128):
    """Metadata-only round-robin over speaker/emotion, lexicographic clip IDs."""
    groups={}
    for ix,cid in enumerate(q['clip_id']):
        key=(int(q['speaker_id'][ix]),int(q['emotion_id'][ix]))
        groups.setdefault(key,[]).append((cid,ix))
    for values in groups.values():values.sort()
    selected=[];depth=0
    while len(selected)<min(count,len(q['clip_id'])):
        for key in sorted(groups):
            if depth<len(groups[key]):selected.append(groups[key][depth][1])
            if len(selected)==min(count,len(q['clip_id'])):break
        depth+=1
    ids=torch.tensor(selected,dtype=torch.long)
    return ids,{'rule':'Round robin sorted (speaker_id,emotion_id) groups; lexicographic clip_id within group; fixed before inference',
        'clips':[{'clip_id':q['clip_id'][i],'original_fit_index':i,'speaker_id':int(q['speaker_id'][i]),'emotion_id':int(q['emotion_id'][i])} for i in selected],
        'selected_group_counts':{str(key):sum((int(q['speaker_id'][i]),int(q['emotion_id'][i]))==key for i in selected) for key in sorted(groups)},
        'selection_uses_motion':False,'count':len(ids)}


@torch.no_grad()
def fit_diagnostic(upper,q,identities,scales,args,steps,arm,*,position_probe=False):
    from scripts.evaluate_prefix_formal import actual_prefix_continuation,chunk_diagnostics
    from scripts.evaluate_context_mechanism import supplied_history_endpoints
    began=time.monotonic()
    noise=torch.randn(len(q['valid']),96,9,generator=torch.Generator().manual_seed(42))
    curves={};metrics={};normalized_outputs={};probe=[]
    modes=MODES if arm=='chunk_teacher' else ('full',)
    for mode in modes:
        outputs=[]
        for ix in torch.arange(len(q['valid'])).split(args.batch_size):
            b=p.r.subset(q,ix,args.device);ident=p.r.batch_identity(identities,b)
            condition={k:b[k] for k in ('valid','h0','audio_global','audio_intensity')}
            kwargs={'oracle_target':p.h.normalized_target(b,scales)} if mode.startswith('oracle_') else {}
            dynamic=decode_context(upper,condition,ident,b['prefix_local'],noise[ix].to(args.device),steps=steps,arm=arm,mode=mode,**kwargs)
            outputs.append((b['static_upper'][:,None]+dynamic*scales).cpu())
            if position_probe and mode=='full':
                # Same weights and local 16-frame content, but native positions0..15.
                no_padding=torch.zeros_like(dynamic)
                for start in range(0,96,16):
                    ids=b['valid'][:,start:start+16].any(1).nonzero(as_tuple=True)[0]
                    if not len(ids):continue
                    valid=b['valid'][ids,start:start+16];zero=dynamic.new_zeros(len(ids),16,9)
                    x=upper.decode_prefix(valid,b['h0'][ids,start:start+16],ident['code'][ids],
                        {'global':b['audio_global'][ids],'intensity_value':b['audio_intensity'][ids]},
                        b['prefix_local'][ids,start:start+16],noise[ix][ids.cpu(),start:start+16].to(dynamic),
                        known=zero,known_mask=torch.zeros_like(valid),steps=steps)
                    no_padding[ids,start:start+16]=x
                probe.append((b['static_upper'][:,None]+no_padding*scales).cpu())
        upper_values=torch.cat(outputs);pred=torch.zeros_like(q['motion']);pred[...,p.CC]=upper_values
        curves['42/'+mode]=pred;normalized_outputs[mode]=upper_values
        metrics[mode]={}
        for name in ('brows','eyes_expression'):
            cc=list(p.GROUPS[name]);mask=q['valid'][...,None]&q['channel_mask'][:,None,cc]
            metrics[mode][name]=p.paired_metrics(pred[...,cc],q['motion'][...,cc],mask)
        metrics[mode]['stitched_16frame_grid']=p.h.boundary_report(pred,q['motion'],q['valid'])
        metrics[mode]['grid_is_decoder_boundary']=arm!='whole'
        metrics[mode]['chunk_diagnostics']=chunk_diagnostics(pred,q['motion'],q['valid'],q['channel_mask'])
        scored=arm=='chunk_teacher' and mode!='empty'
        metrics[mode]['actual_supplied_prefix']=actual_prefix_continuation(pred,q['motion'],q['valid'],q['channel_mask'],
            supplied_history_endpoints(pred,q['motion'],q['valid'],mode)) if scored else None
        metrics[mode]['GT_was_input']=mode.startswith('oracle_') and arm=='chunk_teacher'
        metrics[mode]['same_GT_endpoint_DIAGNOSTIC']=actual_prefix_continuation(pred,q['motion'],q['valid'],q['channel_mask'],q['motion'])
    report={'role':'fixed_fit_reconstruction_diagnostic','clips':len(q['valid']),'seed':42,'arm':arm,'metrics':metrics,
        'nonupper_scored':False,'teacher_modes_are_oracle':True,'test_loaded':False,'seconds':time.monotonic()-began,
        'adjacent_prefix_boundary_pairs':int(((q['valid'][:,1:]&q['valid'][:,:-1])&
            ((torch.arange(1,96)%16)==0)[None]).sum())}
    if position_probe:
        values=torch.cat(probe);a=normalized_outputs['full'];valid=q['valid'][...,None].expand_as(a)
        report['position0_vs8_same_weights_same_noise']={'observed_upper_rms_difference':float((a[valid]-values[valid]).double().square().mean().sqrt()),
            'scope':'Only chunk_empty step0; positions and masked padding differ, no fitting or gain adjustment'}
    data={key:q[key] for key in ('clip_id','valid','times','channel_mask','b0','emotion_id','speaker_id')}
    data.update(target=q['motion'],predictions=curves,arm=arm,nonupper_placeholders=True)
    return report,data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','audio','targets','enrollment','native-root','trained-run','history-run','centered-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--seed',type=int,default=83)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.pilot=False;args.pilot_selection=None
    if args.output.exists():raise FileExistsError('Fresh mechanism output required')
    if (args.epochs,args.batch_size,args.seed)!=(12,16,83):raise ValueError('Locked budget 12epochs batch16 seed83')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;began=time.monotonic()
    data,system,audio,local,identities,source,source_recipe,source_sha,steps,_,frozen=p.load_context(args)
    if steps!=12 or len(data['splits']['train']['valid'])!=2315 or len(data['splits']['validation']['valid'])!=405:
        raise ValueError('Locked source dimensions or solver differ')
    basepath=args.centered_run/'white/curves.pt';bases=load_pt(basepath)
    if sha(basepath)!=read(basepath.with_name('complete.json'))['curves_sha256'] or sha(basepath)!=source_recipe['baseline_curves_sha256']:
        raise ValueError('Base binding differs')
    metadata_equal({**data['splits']['validation'],'target':data['splits']['validation']['motion']},bases)
    ids,selection=fixed_fit_selection(data['splits']['train'])
    fit=p.r.subset(data['splits']['train'],ids[:8] if args.smoke else ids,'cpu')
    args.output.mkdir(parents=True);p.save_json(args.output/'fit_selection.json',selection)
    if args.smoke:
        data['splits']={key:p.r.subset(value,torch.arange(32),'cpu') for key,value in data['splits'].items()}
        bases={**bases,**p.r.subset({key:bases[key] for key in ('clip_id','target','valid','times','channel_mask','b0','emotion_id','speaker_id')},torch.arange(32),'cpu'),
            'predictions':{key:value[:32] for key,value in bases['predictions'].items()}}
    train=data['splits']['train'];scales=source['scales'].to(args.device);epochs=1 if args.smoke else 12;matches={}
    for arm in ARMS:
        torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
        upper=p.PrefixUpperFlow(data['config']).to(args.device).eval();discarded=load_shared_warmstart(upper,source['upper'])
        output=args.output/arm;output.mkdir();initial=p.state_hash(upper.state_dict())
        p.save_checkpoint(output/'initial.pt',{'upper':upper.state_dict()})
        recipe={'schema':SCHEMA,'arm':arm,'epochs':epochs,'requested_epochs':12,'seed':83,'batch_size':16,'smoke':args.smoke,
            'initial':initial,'initial_file_sha256':sha(output/'initial.pt'),'warmstart_sha256':source_sha,
            'source_recipe_sha256':p.canonical_hash(source_recipe),'config':data['config'],'scales':scales.cpu().tolist(),
            'discarded_source_keys':discarded,'frozen':frozen,'data_provenance':data['provenance'],
            'baseline_curves_sha256':sha(basepath),'decode_steps':steps,'fit_selection_sha256':sha(args.output/'fit_selection.json'),
            'known_past':8,'chunk':16,'whole_total_tokens':104,'same_full_noise_and_clip_flow_time':True,
            'test_loaded':False,'default_replaced':False,'fixed_final_epoch':True,
            'code_sha256':{name:sha(Path(__file__).resolve().parents[1]/name) for name in (
                'scripts/train_context_mechanism.py','scripts/evaluate_context_mechanism.py','scripts/train_prefix_upper.py',
                'scripts/train_prefix_formal.py','scripts/evaluate_prefix_formal.py','scripts/train_history_upper.py',
                'scripts/audit_history_upper.py','scripts/audit_temporal_repair.py','scripts/train_temporal_repair.py',
                'kinetalk_b0/models/prefix_upper_flow.py','kinetalk_b0/models/temporal_upper.py','docs/CONTEXT_MECHANISM_PROTOCOL_20260917.md')}}
        digest=p.canonical_hash(recipe);p.save_json(output/'provenance.json',{'recipe':recipe,'recipe_sha256':digest})
        initial_report,initial_curves=fit_diagnostic(upper,fit,identities,scales,args,steps,arm,position_probe=arm=='chunk_empty')
        p.save_json(output/'step0_fit.json',initial_report);p.save_checkpoint(output/'step0_fit_curves.pt',initial_curves)
        parameters=list(upper.parameters());optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-5)
        generator=torch.Generator().manual_seed(83);draws=[];total_steps=0
        for epoch in range(epochs):
            t0=time.monotonic();losses=[];draw=hashlib.sha256()
            for selected in torch.randperm(len(train['valid']),generator=generator).split(16):
                b=p.r.subset(train,selected,args.device);ident=p.r.batch_identity(identities,b)
                loss,randoms=context_batch(upper,b,ident,b['prefix_local'],scales,generator,arm)
                for tensor in (selected,*randoms):draw.update(tensor.numpy().tobytes())
                p.r.optimize(loss,optimizer,parameters);losses.append(float(loss.detach()));total_steps+=1
            row={'arm':arm,'epoch':epoch+1,'loss':sum(losses)/len(losses),'seconds':time.monotonic()-t0,
                'total_steps':total_steps,'draw_sha256':draw.hexdigest(),'GT_past_training':arm=='chunk_teacher'}
            draws.append(draw.hexdigest());p.save_json(output/f'epoch{epoch+1:03d}.json',row)
            p.save_json(args.output/'status.json',{'status':'training',**row});print(json.dumps(row),flush=True)
            payload={'schema':SCHEMA,'arm':arm,'upper':upper.state_dict(),'scales':scales.cpu(),'recipe_sha256':digest,
                'completed_epochs':epoch+1,'total_steps':total_steps,'frozen':frozen}
            p.save_checkpoint(output/'last.pt',{**payload,'optimizer':optimizer.state_dict(),'rng':p.capture_rng(generator)})
        for name,module in (('system',system),('audio',audio),('local',local)):
            if p.state_hash(module.state_dict())!=frozen[name] or any(param.grad is not None for param in module.parameters()):
                raise RuntimeError('Frozen module changed')
        p.save_json(args.output/'status.json',{'status':'evaluating','arm':arm,'completed_epochs':epochs})
        report,curves=evaluate(upper,system,data['splits']['validation'],identities,scales,bases,args,steps,arm,decode_context)
        report.update(schema=SCHEMA,recipe_sha256=digest,smoke=args.smoke);curves.update(schema=SCHEMA,recipe_sha256=digest)
        p.save_json(output/'evaluation.json',report);p.save_checkpoint(output/'curves.pt',curves)
        fit_report,fit_curves=fit_diagnostic(upper,fit,identities,scales,args,steps,arm)
        p.save_json(output/'fit_evaluation.json',fit_report);p.save_checkpoint(output/'fit_curves.pt',fit_curves)
        p.save_checkpoint(output/'final.pt',payload)
        p.save_json(output/'complete.json',{'final_sha256':sha(output/'final.pt'),'curves_sha256':sha(output/'curves.pt'),
            'fit_curves_sha256':sha(output/'fit_curves.pt'),'completed_epochs':epochs,'recipe_sha256':digest,'frozen':frozen})
        matches[arm]={'initial':initial,'draws':draws}
    if any(value!=matches[ARMS[0]] for value in matches.values()):raise RuntimeError('Random pairing failed')
    p.save_json(args.output/'matched_audit.json',{'equal':True,'arms':matches})
    p.save_json(args.output/'status.json',{'status':'complete','smoke':args.smoke,'epochs_per_arm':epochs,'seconds':time.monotonic()-began})
    print('CONTEXT_MECHANISM_COMPLETE',flush=True)


if __name__=='__main__':main()
