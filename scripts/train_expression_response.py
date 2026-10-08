"""Phase41 native response development run; frozen neutral B0, no sealed data.

One resumable optimizer, explicit q/p gradient responsibilities, independent
neutral supports, and a preregistered TRAIN speaker/sentence development fold.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import (
    ExpressionResponse,ResponseConfig,motion_objective,normalized_kl,semantic_objective)
from kinetalk_b0.models.model import Stage1Model
from scripts.packed_trainval_cache import load_packed
from scripts.train_full_staged import base_forward,sha,class_balanced_weights


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def save(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def state_digest(module):
    h=hashlib.sha256()
    for name,t in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def configure(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(False)


def load_runtime(binding,device):
    assert sha(binding['neutral_checkpoint']['path'])==binding['neutral_checkpoint']['sha256']
    ck=torch.load(binding['neutral_checkpoint']['path'],map_location='cpu',weights_only=False)
    data=load_packed(binding['data'],with_refs=True)
    assert data['provenance']['manifest_sha256']==binding['data_manifest_sha256']
    del data['system']  # Do not keep the obsolete full teacher/flow model active.
    b0=Stage1Model(ck['config']).to(device).eval().requires_grad_(False)
    b0.load_state_dict({k[len('stage1.'):]:v for k,v in ck['system'].items() if k.startswith('stage1.')},strict=True)
    support=ck['system'].get('motion_support',torch.tensor([True]*51+[False])).to(device).bool()
    base=SimpleNamespace(stage1=b0,motion_support=support)
    # Reuse only neutral B0. Prior, posterior, style and decoder start fresh.
    return data,base


def development_fold(q,fit_sids):
    held_sids=sorted(fit_sids)[-2:]
    held_sentence=lambda s:int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)%10==0
    train=[];speaker=[];sentence=[]
    for i,(sid,sen) in enumerate(zip(q['speaker_id'],q['sentence_id'])):
        if int(sid) in held_sids:speaker.append(i)
        elif held_sentence(sen):sentence.append(i)
        else:train.append(i)
    if not train or not speaker or not sentence:raise ValueError('Empty TRAIN internal fold')
    return {'train':train,'speaker_dev':speaker,'sentence_dev':sentence,'held_sids':held_sids}


@torch.no_grad()
def cache_base(data,base,device,selection=None):
    for role,q in data['splits'].items():
        chosen=selection.get(role,[]) if selection is not None else range(len(q['valid']))
        values=[torch.empty(0,52) for _ in q['_lengths']]
        for ids in torch.tensor(list(chosen),dtype=torch.long).split(32):
            if not len(ids):continue
            b=q.batch(ids,device,keys=('audio_features','valid'))
            y=base_forward(base,b['content'],b['valid'])['b0']
            for j,i in enumerate(ids.tolist()):values[i]=y[j,:int(q['_lengths'][i])].cpu()
        q.set_extra('b0',values)
        print(json.dumps({'event':'neutral_cache','role':role,'clips':len(list(chosen))}),flush=True)
    for sid,ref in data['refs'].items():
        valid=ref['valid'].to(device)
        content=ref['content'].to(device).float()
        ref['b0']=base_forward(base,content,valid)['b0'].cpu()


def audit_data(data,fold):
    q=data['splits']['train'];query_ids=set(q['clip_id'])|set(data['splits']['validation']['clip_id'])
    all_refs=[];same_sentence=0
    for sid,ref in data['refs'].items():
        assert len(ref['clip_id'])==2 and len(set(ref['clip_id']))==2
        assert not query_ids.intersection(ref['clip_id'])
        all_refs.extend(ref['clip_id'])
        for i in (q['speaker_id']==sid).nonzero(as_tuple=True)[0].tolist():
            same_sentence+=int(q['sentence_id'][i] in ref['sentence_id'])
    assert len(set(all_refs))==len(all_refs)
    return {'query_clips':len(q['valid']),'validation_clips':len(data['splits']['validation']['valid']),
            'fit_clips':len(fold['train']),'speaker_dev_clips':len(fold['speaker_dev']),
            'sentence_dev_clips':len(fold['sentence_dev']),'held_sids':fold['held_sids'],
            'references':len(all_refs),'support_query_clip_disjoint':True,
            'query_sentences_also_in_enrollment':same_sentence,
            'test_loaded':False,'validation_used_for_selection':False}


def reference_batch(data,batch,device,*,two_subsets=False,wrong_speaker=False):
    sids=batch['speaker_id'].tolist();refs=data['refs']
    if wrong_speaker:
        pool=sorted(set(sids));pool=sorted(data['dev_sids']) if set(sids)<=set(data['dev_sids']) else sorted(data['fit_sids'])
        sids=[pool[(pool.index(s)+1)%len(pool)] for s in sids]
    result={k:torch.stack([refs[s][k] for s in sids]).to(device) for k in ('motion','b0','valid','channel_mask')}
    width=int((result['valid']*torch.arange(1,result['valid'].shape[-1]+1,device=device)).amax())
    for k in ('motion','b0','valid'):result[k]=result[k][:,:,:width]
    if two_subsets:
        # [B,2,T,C] -> [2B,1,T,C], paired with repeat_interleave(query,2).
        result={k:v.flatten(0,1).unsqueeze(1) for k,v in result.items()}
    return result


def repeated_query(batch):
    return {k:v.repeat_interleave(2,0) if torch.is_tensor(v) else
            [x for item in v for x in (item,item)] for k,v in batch.items()}


def fit_statistics(data,ids):
    """Exclude internal held-out speakers/sentences even from normalization."""
    total=torch.zeros(772,dtype=torch.float64);squares=total.clone();count=0
    energy=torch.zeros(52,dtype=torch.float64);observations=energy.clone()
    q=data['splits']['train']
    for sub in torch.tensor(ids).split(64):
        b=q.batch(sub,keys=('audio_features','motion','valid','channel_mask','anchors','anchor_valid'))
        x=b['audio_features'][...,768:][b['valid']].double()
        total+=x.sum(0);squares+=x.square().sum(0);count+=len(x)
        obs=b['valid'][...,None]&b['channel_mask'][:,None]&b['anchor_valid'][:,None]
        delta=torch.where(obs,b['motion']-b['anchors'][:,None],0.).double()
        energy+=delta.square().sum((0,1));observations+=obs.sum((0,1))
    mean=total/count;std=(squares/count-mean.square()).clamp_min(0).sqrt().clamp_min(.001)
    scales=(energy/observations.clamp_min(1)).sqrt().clamp_min(.02)
    return mean.float(),std.float(),scales.float()


def objective(model,b,refs,*,beta,prior_weight,sample,generator,weights):
    style=model.encode_style(refs)['code']
    p=model.audio_prior(b['audio_features'],b['valid'])
    q=model.motion_posterior(b['motion'],b['b0'],style,b['valid'],b['channel_mask'],b['times'])
    y=model.decode(b['b0'],q,style,b['valid'],sample=sample,generator=generator)
    reconstruction,parts=motion_objective(y,b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)
    kl=normalized_kl(q,p)
    semantic=.5*(semantic_objective(model,q,b['emotion_id'],b['intensity_id'],b['intensity_valid'],weights)+
                 semantic_objective(model,p,b['emotion_id'],b['intensity_id'],b['intensity_valid'],weights))
    prior_reconstruction=reconstruction.new_zeros(())
    if prior_weight:
        detached={k:v.detach() for k,v in p.items()}
        predicted=model.decode(b['b0'],detached,style,b['valid'])
        prior_reconstruction,_=motion_objective(predicted,b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)
    total=reconstruction+beta*kl+.1*semantic+prior_weight*prior_reconstruction
    logs={'loss':total,'q_position':parts['position'],'q_velocity':parts['velocity'],
          'kl':kl,'semantic':semantic,'p_reconstruction':prior_reconstruction,
          'p_u_std':p['u_mean'][p['u_mask']].std(correction=0),
          'p_logvar_mean':p['u_logvar'][p['u_mask']].mean()}
    return total,logs


@torch.no_grad()
def quick_evaluate(model,data,ids,device):
    model.eval();records=[]
    for sub in torch.tensor(ids).split(16):
        b=data['splits']['train'].batch(sub,device);ref=reference_batch(data,b,device)
        s=model.encode_style(ref)['code'];p=model.audio_prior(b['audio_features'],b['valid'])
        q=model.motion_posterior(b['motion'],b['b0'],s,b['valid'],b['channel_mask'],b['times'])
        values={}
        for name,d in [('prior',p),('posterior',q)]:
            y=model.decode(b['b0'],d,s,b['valid'])
            _,part=motion_objective(y,b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)
            values[name+'_position']=float(part['position']);values[name+'_velocity']=float(part['velocity'])
        values['kl']=float(normalized_kl(q,p));records.append((len(sub),values))
    return {k:sum(n*v[k] for n,v in records)/sum(n for n,v in records) for k in records[0][1]}


def train(a):
    configure(a.seed);device=torch.device(a.device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    binding=json.loads(Path(a.binding).read_text());data,base=load_runtime(binding,device)
    frozen_digest=state_digest(base.stage1);fold=development_fold(data['splits']['train'],data['fit_sids'])
    audit=audit_data(data,fold);write(out/'data_audit.json',audit)
    write(out/'fold.json',{k:v for k,v in fold.items()})
    ids=fold['train']
    if a.smoke:
        chosen=[]
        for emotion in range(8):
            chosen.append(next(i for i in ids if int(data['splits']['train']['emotion_id'][i])==emotion))
        ids=chosen
        cache_base(data,base,device,{'train':ids,'validation':[]})
    else:cache_base(data,base,device)
    cfg=ResponseConfig(prior_variance=a.prior_variance)
    mean,std,scales=fit_statistics(data,ids)
    model=ExpressionResponse(cfg,mean,std,scales).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    latent_rng=torch.Generator(device=device).manual_seed(a.seed+10000)
    order_rng=torch.Generator().manual_seed(a.seed+20000)
    weights=class_balanced_weights(data['splits']['train']['emotion_id'][ids],8).to(device)
    protocol={'schema':'expression_response_v1','config':asdict(cfg),'args':vars(a),
              'binding_sha256':sha(a.binding),'data':audit,'neutral_digest':frozen_digest,
              'parameters':sum(p.numel() for p in model.parameters()),
              'loss':'q motion(position+.5 adjacent displacement) + annealed .01 KL + .1 global semantic + .5 detached prior motion after epoch4',
              'training_scope':'TRAIN internal speaker/sentence development split; not final all-data result',
              'default_replaced':False,'test_loaded':False}
    if a.resume:
        original=json.loads((out/'protocol.json').read_text())
        for key in ('seed','epochs','batch_size','lr','smoke','smoke_steps'):
            if original['args'][key]!=vars(a)[key]:raise ValueError('Resume configuration differs: '+key)
        if original['config'].get('prior_variance','learned')!=cfg.prior_variance:
            raise ValueError('Resume prior variance differs')
        if original['binding_sha256']!=protocol['binding_sha256']:raise ValueError('Resume source binding changed')
    else:write(out/'protocol.json',protocol)
    start_epoch=0;step=0;history=[]
    if a.resume:
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        assert ck['neutral_digest']==frozen_digest and ck['binding_sha256']==sha(a.binding)
        if ResponseConfig(**ck['config'])!=cfg:raise ValueError('Resume architecture differs')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        latent_rng.set_state(ck['latent_rng'].cpu());order_rng.set_state(ck['order_rng'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        start_epoch=ck['epoch'];step=ck['step'];history=ck['history']
    elif (out/'last.pt').exists():raise FileExistsError('Existing run: use explicit --resume')
    initial=None;smoke_losses=[];durations=[];started=time.time()
    for epoch in range(start_epoch,1 if a.smoke else a.epochs):
        if shutil.disk_usage(out).free<250*2**20:raise RuntimeError('Checkpoint safety disk floor reached')
        model.train();tick=time.time();logs=[]
        order=torch.tensor(ids)[torch.randperm(len(ids),generator=order_rng)]
        batches=[order]*a.smoke_steps if a.smoke else list(order.split(a.batch_size))
        for index,sub in enumerate(batches):
            b=data['splits']['train'].batch(sub,device);refs=reference_batch(data,b,device,two_subsets=True)
            b=repeated_query(b);optimizer.zero_grad(set_to_none=True)
            beta=.01 if a.smoke else .01*min(1.,(epoch+1)/4)
            prior_weight=0. if a.smoke or epoch<4 else .5
            loss,parts=objective(model,b,refs,beta=beta,prior_weight=prior_weight,
                sample=not a.smoke,generator=latent_rng,weights=weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            row={k:float(v.detach()) for k,v in parts.items()};row['grad_norm']=float(norm);logs.append(row)
            if initial is None:initial=row
            if a.smoke:smoke_losses.append(row['q_position'])
            if index%100==0 or (a.smoke and index%20==0):
                elapsed=time.time()-tick
                msg={'event':'update','epoch':epoch+1,'step':step,'batch':index+1,
                     'seconds_per_update':elapsed/(index+1),**row}
                print(json.dumps(msg),flush=True);write(out/'state.json',dict(status='training',**msg,test_loaded=False))
        duration=time.time()-tick;durations.append(duration)
        entry={'epoch':epoch+1,'updates':step,'seconds':duration,
               'train':{k:sum(x[k] for x in logs)/len(logs) for k in logs[0]}}
        if not a.smoke:
            # Fixed folds, no validation selection. Full lists persist in fold.json.
            for role in ('speaker_dev','sentence_dev'):
                chosen=fold[role];positions=np.linspace(0,len(chosen)-1,min(256,len(chosen)),dtype=int)
                entry[role]=quick_evaluate(model,data,[chosen[i] for i in positions],device)
        history.append(entry)
        assert state_digest(base.stage1)==frozen_digest and all(p.grad is None for p in base.stage1.parameters())
        checkpoint={'model':model.state_dict(),'config':asdict(cfg),'epoch':epoch+1,'step':step,
                    'optimizer':optimizer.state_dict(),'latent_rng':latent_rng.get_state(),
                    'order_rng':order_rng.get_state(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    'history':history,'neutral_digest':frozen_digest,'binding_sha256':sha(a.binding),
                    'test_loaded':False}
        save(out/'last.pt',checkpoint);write(out/'history.json',history)
        eta=max(0,a.epochs-epoch-1)*float(np.mean(durations[-3:]))
        write(out/'state.json',{'status':'training','epoch':epoch+1,'updates':step,'epoch_seconds':duration,
                               'remaining_training_seconds':eta,'last_checkpoint_sha256':sha(out/'last.pt'),'test_loaded':False})
        print(json.dumps({'event':'epoch_complete',**entry,'remaining_training_seconds':eta}),flush=True)
    model.eval()
    final={k:v for k,v in checkpoint.items() if k not in ('optimizer','latent_rng','order_rng','torch_rng','cuda_rng')}
    save(out/'final.pt',final)
    if a.smoke:
        last=float(np.mean(smoke_losses[-10:]));first=float(np.mean(smoke_losses[:10]))
        passed=last<.65*first and len(smoke_losses)==a.smoke_steps
        report={'passed':passed,'updates':step,'first10_position':first,'last10_position':last,
                'ratio':last/first,'frozen_neutral_unchanged':True,'seconds':time.time()-started,
                'parameters':protocol['parameters'],'test_loaded':False}
        write(out/'smoke.json',report);print(json.dumps(report),flush=True)
        if not passed:raise RuntimeError('Small-data learnability gate failed')
    write(out/'complete.json',{'status':'complete','updates':step,'epochs':epoch+1,
                             'final_sha256':sha(out/'final.pt'),'test_loaded':False})
    write(out/'state.json',{'status':'training_complete','updates':step,'epochs':epoch+1,'test_loaded':False})
    return model,data,base


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-steps',type=int,default=120);p.add_argument('--resume',action='store_true')
    p.add_argument('--prior-variance',choices=('learned','unit'),default='learned')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    try:train(args)
    except Exception as exc:
        write(Path(args.output)/'failure.json',{'type':type(exc).__name__,'message':str(exc),'time':time.time()})
        raise
