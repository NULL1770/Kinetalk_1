"""Diverse independent neutral supports; frozen B0 and audio expression prior.

Original temporal style control versus a compact posture/response descriptor.
No query motion enters deployment; no reconstruction gradient enters the prior.
"""
from __future__ import annotations
import argparse,json,sys,time,shutil
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import ExpressionResponse,motion_objective
from scripts.refine_expression_prior import restore_model
from scripts.adapt_expression_receiver import inherited_budget
from scripts.train_expression_response import (configure,load_runtime,cache_base,development_fold,
    audit_data,reference_batch,quick_evaluate,state_digest,write,save,sha)


def support_pool(q,fit_ids,neutral):
    pools={}
    for i in fit_ids:
        if int(q['emotion_id'][i])==neutral:
            pools.setdefault(int(q['speaker_id'][i]),[]).append(i)
    for sid in {int(q['speaker_id'][i]) for i in fit_ids}:
        if len(pools.get(sid,[]))<3:raise ValueError('Insufficient independent neutral supports')
    return pools


def choose_supports(q,pools,query_ids,generator):
    pairs=[]
    for i in query_ids:
        sid=int(q['speaker_id'][i]);sentence=str(q['sentence_id'][i])
        eligible=[j for j in pools[sid] if q['clip_id'][j]!=q['clip_id'][i] and str(q['sentence_id'][j])!=sentence]
        if len(eligible)<2:raise ValueError('No independent same-person support pair')
        positions=torch.randperm(len(eligible),generator=generator)[:2].tolist()
        pairs.append([eligible[j] for j in positions])
    return torch.tensor(pairs,dtype=torch.long)


def sampled_references(q,pairs,device):
    batch=q.batch(pairs.flatten(),device,keys=('motion','b0','valid','channel_mask'))
    return {k:batch[k].reshape(len(pairs),2,*batch[k].shape[1:]) for k in ('motion','b0','valid','channel_mask')}


def candidate(parent,mode):
    if mode=='temporal':return parent
    if mode!='statistics':raise ValueError('Unknown style mode')
    cfg=replace(parent.cfg,reference_encoder='statistics')
    model=ExpressionResponse(cfg,parent.feature_mean,parent.feature_std,parent.scales).to(parent.scales.device)
    old=parent.state_dict();state=model.state_dict()
    # Keep the audio/posterior and common B0 receiver exactly. New style paths
    # start independently; no claim that old style dimensions are meaningful.
    for name in state:
        if not name.startswith('style.') and not name.startswith('decoder.bias.') and state[name].shape==old[name].shape:
            state[name]=old[name].clone()
    width=cfg.global_dim+cfg.local_dim
    for i in range(4):
        key=f'decoder.modulations.{i}.weight'
        state[key].zero_();state[key][:,:width]=old[key][:,:width]
    model.load_state_dict(state,strict=True)
    return model


def freeze_expression(model):
    model.eval().requires_grad_(False);model.zero_grad(set_to_none=True)
    model.style.requires_grad_(True);model.decoder.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def frozen_digests(model):
    return {name:state_digest(getattr(model,name)) for name in ('prior','posterior','emotion_head','intensity_head')}


def objective(model,b,refs):
    with torch.no_grad():p=model.audio_prior(b['audio_features'],b['valid'])
    s=model.encode_style(refs)['code']
    y=model.decode(b['b0'],p,s,b['valid'])
    return motion_objective(y,b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)


def balanced_smoke(q,ids):
    selected=[]
    for e in range(8):
        pool=[i for i in ids if int(q['emotion_id'][i])==e]
        selected.extend(pool[j] for j in np.linspace(0,len(pool)-1,min(4,len(pool)),dtype=int))
    return selected


def train(a):
    configure(a.seed);device=torch.device(a.device);out=Path(a.output)
    if (out/'complete.json').exists():raise FileExistsError('Completed run must not restart')
    if out.exists() and not a.resume:raise FileExistsError('Fresh run required')
    out.mkdir(parents=True,exist_ok=True)
    binding=json.loads(Path(a.binding).read_text())
    for n,h in binding['source_files'].items():assert sha(Path(__file__).resolve().parents[1]/n)==h,n
    data,base=load_runtime(binding,device);q=data['splits']['train']
    fold=development_fold(q,data['fit_sids']);audit=audit_data(data,fold)
    pools=support_pool(q,fold['train'],data['config']['data']['emotion_classes'].index('neutral'))
    ids=balanced_smoke(q,fold['train']) if a.smoke else fold['train']
    selected=set(ids)|{i for v in pools.values() for i in v}
    if not a.smoke:selected.update(fold['speaker_dev']+fold['sentence_dev'])
    canonical=sorted({j for i in selected for j in range(i//32*32,min((i//32+1)*32,len(q['_lengths'])))})
    cache_base(data,base,device,{'train':canonical,'validation':[]})
    parent,ck=restore_model(binding,device);neutral=state_digest(base.stage1)
    assert neutral==ck['neutral_digest']
    model=candidate(parent,a.mode);params=freeze_expression(model);frozen=frozen_digests(model)
    budget=inherited_budget(ck)
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    order_rng=torch.Generator().manual_seed(a.seed+51000)
    ref_rng=torch.Generator().manual_seed(a.seed+52000)
    protocol=dict(schema='phase51_diverse_neutral_reference_v1',mode=a.mode,
        args={k:v for k,v in vars(a).items() if k!='resume'},binding_sha256=sha(a.binding),
        data=audit,neutral_support_counts={str(k):len(v) for k,v in pools.items()},
        support_policy='TRAIN-fit neutral only; same actor; exclude query clip AND sentence; two without replacement',
        initial_model_digest=state_digest(model),frozen_digests=frozen,neutral_digest=neutral,
        config=model.checkpoint_config(),trainable_parameters=sum(p.numel() for p in params),
        loss='original position + .5 adjacent displacement; p_mean; no new loss',
        oracle_limit='Inherited motion posterior not recalibrated to changed style; not a selection metric',
        test_loaded=False,external_validation_used=False,**budget)
    start,step,history=0,0,[]
    if a.resume:
        assert json.loads((out/'protocol.json').read_text())==protocol
        z=torch.load(out/'last.pt',map_location=device,weights_only=False)
        assert z['binding_sha256']==sha(a.binding)
        model.load_state_dict(z['model']);opt.load_state_dict(z['optimizer'])
        order_rng.set_state(z['order_rng'].cpu());ref_rng.set_state(z['ref_rng'].cpu())
        torch.set_rng_state(z['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in z['cuda_rng']])
        start,step,history=z['epoch'],z['step'],z['history']
        assert frozen_digests(model)==frozen
    else:
        write(out/'protocol.json',protocol);write(out/'fold.json',fold)
    losses=[];began=time.time()
    for epoch in range(start,1 if a.smoke else a.epochs):
        if shutil.disk_usage(out).free<160*2**20:raise RuntimeError('Checkpoint disk floor reached')
        model.eval();model.style.train();model.decoder.train();tick=time.time();rows=[]
        order=torch.tensor(ids)[torch.randperm(len(ids),generator=order_rng)]
        batches=[order]*a.smoke_steps if a.smoke else order.split(a.batch_size)
        fixed_pairs=choose_supports(q,pools,order.tolist(),ref_rng) if a.smoke else None
        for index,sub in enumerate(batches):
            b=q.batch(sub,device)
            pairs=fixed_pairs if a.smoke else choose_supports(q,pools,sub.tolist(),ref_rng)
            refs=sampled_references(q,pairs,device)
            opt.zero_grad(set_to_none=True);loss,parts=objective(model,b,refs)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reference objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(('style.','decoder.')))
            assert all(p.grad is None for p in base.stage1.parameters())
            opt.step();step+=1
            row=dict(loss=float(loss.detach()),grad_norm=float(norm),**{k:float(v.detach()) for k,v in parts.items()})
            rows.append(row);losses.append(row['loss'])
            if index%100==0 or (a.smoke and index%20==0):
                v=dict(status='training',epoch=epoch+1,step=step,batch=index+1,seconds_per_update=(time.time()-tick)/(index+1),**row)
                write(out/'state.json',v);print(json.dumps(v),flush=True)
        duration=time.time()-tick
        entry=dict(epoch=epoch+1,updates=step,seconds=duration,train={k:float(np.mean([r[k] for r in rows])) for k in rows[0]})
        model.eval()
        if not a.smoke:
            for role in ('speaker_dev','sentence_dev'):
                v=fold[role];ix=np.linspace(0,len(v)-1,min(256,len(v)),dtype=int)
                entry[role]=quick_evaluate(model,data,[v[i] for i in ix],device)
        # Deployment contract: HuBERT portion is never inspected by expression path.
        b=q.batch(torch.tensor(ids[:2]),device);refs=reference_batch(data,b,device)
        with torch.no_grad():
            original=model.predict(b['audio_features'],b['b0'],b['valid'],refs)
            audio=b['audio_features'].clone();audio[...,:768]=float('nan')
            torch.testing.assert_close(original,model.predict(audio,b['b0'],b['valid'],refs),rtol=0,atol=0)
        assert frozen_digests(model)==frozen and state_digest(base.stage1)==neutral
        history.append(entry)
        state=dict(model=model.state_dict(),config=model.checkpoint_config(),reference_training='diverse_neutral_two',
            epoch=epoch+1,step=step,history=history,optimizer=opt.state_dict(),order_rng=order_rng.get_state(),
            ref_rng=ref_rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            neutral_digest=neutral,binding_sha256=sha(a.binding),frozen_digests=frozen,
            parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'],test_loaded=False,**budget)
        save(out/'last.pt',state);write(out/'history.json',history)
        eta=max(0,a.epochs-epoch-1)*np.mean([h['seconds'] for h in history[-3:]])
        write(out/'state.json',dict(status='training',epoch=epoch+1,updates=step,epoch_seconds=duration,remaining_training_seconds=float(eta),test_loaded=False))
        print(json.dumps(dict(event='epoch_complete',**entry,remaining_training_seconds=float(eta))),flush=True)
    final={k:v for k,v in state.items() if k not in ('optimizer','order_rng','ref_rng','torch_rng','cuda_rng')}
    save(out/'final.pt',final)
    if a.smoke:
        first,last=float(np.mean(losses[:10])),float(np.mean(losses[-10:]))
        passed=len(losses)==a.smoke_steps and last<.9*first
        write(out/'smoke.json',dict(passed=passed,first10=first,last10=last,ratio=last/first,updates=step,
            frozen_expression_exact=True,neutral_exact=True,hubert_nan_isolated=True,seconds=time.time()-began,test_loaded=False))
        if not passed:raise RuntimeError('Reference learnability smoke failed')
    write(out/'complete.json',dict(status='complete',final_sha256=sha(out/'final.pt'),epochs=state['epoch'],updates=step,test_loaded=False,**budget))
    write(out/'state.json',dict(status='training_complete',epochs=state['epoch'],updates=step,test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('temporal','statistics'),required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=47);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-steps',type=int,default=120);p.add_argument('--resume',action='store_true')
    train(p.parse_args())
