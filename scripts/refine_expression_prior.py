"""Frozen-teacher refinement of the 772D expression prior, without motion loss."""
from __future__ import annotations
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import (
    ExpressionResponse,ResponseConfig,normalized_kl,semantic_objective)
from scripts.train_expression_response import (configure,load_runtime,cache_base,development_fold,
    audit_data,reference_batch,training_reference_views,repeated_query,quick_evaluate,
    state_digest,write,save,sha,class_balanced_weights)


def distribution_match(model,q,p,valid,mode):
    """Mean fit in actual decoder coordinates; scale fit in original token space.

    This is not the joint Wasserstein distance after temporal projection.
    q is always a detached target; p variance never weights the mean term.
    """
    q={k:v.detach() for k,v in q.items()}
    if mode=='kl':return normalized_kl(q,p)
    if mode!='mean_scale':raise ValueError('Unknown prior matching mode')
    if not torch.equal(q['u_mask'],p['u_mask']):raise ValueError('Latent clocks differ')
    qg,qu=model.conditions(q,valid);pg,pu=model.conditions(p,valid)
    g_mean=(qg-pg).square().mean(-1)
    u_mean=torch.where(valid,(qu-pu).square().mean(-1),0.).sum(1)/valid.sum(1).clamp_min(1)
    scale=lambda key:(torch.exp(.5*q[key+'_logvar'])-torch.exp(.5*p[key+'_logvar'])).square().mean(-1)
    g_scale=scale('g');mask=p['u_mask']
    u_scale=torch.where(mask,scale('u'),0.).sum(1)/mask.sum(1).clamp_min(1)
    return .25*(g_mean+u_mean+g_scale+u_scale).mean()


def freeze_receiver(model):
    model.eval().requires_grad_(False)
    for module in (model.prior,model.emotion_head,model.intensity_head):module.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def frozen_digests(model):
    return {name:state_digest(getattr(model,name)) for name in ('posterior','style','decoder')}


def objective(model,b,refs,mode,weights=None):
    with torch.no_grad():
        style=model.encode_style(refs)['code']
        q=model.motion_posterior(b['motion'],b['b0'],style,b['valid'],b['channel_mask'],b['times'])
    p=model.audio_prior(b['audio_features'],b['valid'])
    matching=distribution_match(model,q,p,b['valid'],mode)
    semantic=semantic_objective(model,p,b['emotion_id'],b['intensity_id'],b['intensity_valid'],weights)
    loss=.01*matching+.1*semantic
    return loss,{'loss':loss,'matching':matching,'semantic':semantic}


def restore_model(binding,device):
    spec=binding['parent_checkpoint']
    if sha(spec['path'])!=spec['sha256']:raise ValueError('Parent checkpoint hash mismatch')
    ck=torch.load(spec['path'],map_location=device,weights_only=False)
    cfg=ResponseConfig(**ck['config'])
    model=ExpressionResponse(cfg,ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'],strict=True)
    return model,ck


def train(a):
    configure(a.seed);device=torch.device(a.device);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    # Fail before overwriting any run metadata.
    if not a.resume and (out/'protocol.json').exists():raise FileExistsError('Existing run; explicit --resume required')
    if (out/'complete.json').exists():raise FileExistsError('Completed run must not be restarted')
    binding=json.loads(Path(a.binding).read_text(encoding='utf8'))
    if a.matching!=binding['matching']:raise ValueError('Matching disagrees with frozen binding')
    data,base=load_runtime(binding,device);fold=development_fold(data['splits']['train'],data['fit_sids'])
    audit=audit_data(data,fold);ids=fold['train']
    if a.smoke:
        ids=[next(i for i in ids if int(data['splits']['train']['emotion_id'][i])==e) for e in range(8)]
        cache_base(data,base,device,{'train':ids,'validation':[]})
    else:cache_base(data,base,device)
    model,parent=restore_model(binding,device);reference_mode=parent.get('reference_training','single')
    if reference_mode!=binding['reference_training']:raise ValueError('Parent reference mode mismatch')
    params=freeze_receiver(model);frozen=frozen_digests(model);neutral_digest=state_digest(base.stage1)
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    order_rng=torch.Generator().manual_seed(a.seed+20000)
    weights=class_balanced_weights(data['splits']['train']['emotion_id'][ids],8).to(device)
    protocol={'schema':'phase44_frozen_expression_prior_v1','config':model.checkpoint_config(),
        'args':dict(vars(a),reference_training=reference_mode),'data':audit,'binding_sha256':sha(a.binding),
        'parent_checkpoint_sha256':binding['parent_checkpoint']['sha256'],
        'parent_epochs':parent['epoch'],'parent_updates':parent['step'],
        'initial_model_digest':state_digest(model),'frozen_digests':frozen,'neutral_digest':neutral_digest,
        'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in params),
        'loss':'.01 frozen q/p match + .1 audio global emotion/intensity; no motion reconstruction',
        'test_loaded':False,'default_replaced':False}
    start_epoch=0;step=0;history=[]
    if a.resume:
        old=json.loads((out/'protocol.json').read_text(encoding='utf8'))
        if old!=protocol:raise ValueError('Refinement protocol changed on resume')
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if ck['binding_sha256']!=sha(a.binding):raise ValueError('Resume source binding changed')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        order_rng.set_state(ck['order_rng'].cpu());torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        start_epoch=ck['epoch'];step=ck['step'];history=ck['history']
        if frozen_digests(model)!=frozen:raise ValueError('Frozen receiver changed in checkpoint')
    else:
        # Do not bind operational --resume into the scientific run configuration.
        write(out/'protocol.json',protocol);write(out/'data_audit.json',audit);write(out/'fold.json',fold)
    durations=[];all_match=[];began=time.time()
    for epoch in range(start_epoch,1 if a.smoke else a.epochs):
        if shutil.disk_usage(out).free<250*2**20:raise RuntimeError('Checkpoint safety disk floor reached')
        model.eval();model.prior.train();tick=time.time();rows=[]
        order=torch.tensor(ids)[torch.randperm(len(ids),generator=order_rng)]
        batches=[order]*a.smoke_steps if a.smoke else list(order.split(a.batch_size))
        for index,sub in enumerate(batches):
            b=data['splits']['train'].batch(sub,device)
            refs=training_reference_views(reference_batch(data,b,device),reference_mode,step)
            b=repeated_query(b);optimizer.zero_grad(set_to_none=True)
            loss,logs=objective(model,b,refs,a.matching,weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refinement objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            row={k:float(v.detach()) for k,v in logs.items()};row['grad_norm']=float(norm);rows.append(row)
            all_match.append(row['matching'])
            if index%100==0 or (a.smoke and index%20==0):
                entry=dict(event='update',epoch=epoch+1,step=step,batch=index+1,
                           seconds_per_update=(time.time()-tick)/(index+1),**row)
                print(json.dumps(entry),flush=True);write(out/'state.json',dict(status='training',**entry))
        duration=time.time()-tick;durations.append(duration)
        entry={'epoch':epoch+1,'updates':step,'seconds':duration,
               'train':{k:float(np.mean([row[k] for row in rows])) for k in rows[0]}}
        if not a.smoke:
            for role in ('speaker_dev','sentence_dev'):
                chosen=fold[role];positions=np.linspace(0,len(chosen)-1,min(256,len(chosen)),dtype=int)
                entry[role]=quick_evaluate(model,data,[chosen[i] for i in positions],device)
        history.append(entry)
        assert frozen_digests(model)==frozen and state_digest(base.stage1)==neutral_digest
        assert all(p.grad is None for part in (model.posterior,model.style,model.decoder,base.stage1) for p in part.parameters())
        ck={'model':model.state_dict(),'config':model.checkpoint_config(),'reference_training':reference_mode,
            'epoch':epoch+1,'step':step,'history':history,'optimizer':optimizer.state_dict(),
            'order_rng':order_rng.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'neutral_digest':neutral_digest,'binding_sha256':sha(a.binding),
            'parent_checkpoint_sha256':binding['parent_checkpoint']['sha256'],
            'parent_epochs':parent['epoch'],'parent_updates':parent['step'],'test_loaded':False}
        save(out/'last.pt',ck);write(out/'history.json',history)
        eta=max(0,a.epochs-epoch-1)*float(np.mean(durations[-3:]))
        write(out/'state.json',dict(status='training',epoch=epoch+1,updates=step,epoch_seconds=duration,
                                   remaining_training_seconds=eta,last_checkpoint_sha256=sha(out/'last.pt'),test_loaded=False))
        print(json.dumps(dict(event='epoch_complete',**entry,remaining_training_seconds=eta)),flush=True)
    model.eval();final={k:v for k,v in ck.items() if k not in ('optimizer','order_rng','torch_rng','cuda_rng')}
    save(out/'final.pt',final)
    if a.smoke:
        first=float(np.mean(all_match[:10]));last=float(np.mean(all_match[-10:]))
        passed=len(all_match)==a.smoke_steps and last<.9*first
        report={'passed':passed,'first10_match':first,'last10_match':last,'ratio':last/first,
                'updates':step,'frozen_receiver_exact':True,'seconds':time.time()-began,'test_loaded':False}
        write(out/'smoke.json',report)
        if not passed:raise RuntimeError('Frozen-target small-data learnability gate failed')
    write(out/'complete.json',{'status':'complete','updates':step,'epochs':epoch+1,
        'parent_epochs':parent['epoch'],'parent_updates':parent['step'],'final_sha256':sha(out/'final.pt'),'test_loaded':False})
    write(out/'state.json',{'status':'training_complete','updates':step,'epochs':epoch+1,'test_loaded':False})
    return model,data,base


def parser():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--matching',choices=('kl','mean_scale'),required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-steps',type=int,default=120)
    p.add_argument('--resume',action='store_true');return p


if __name__=='__main__':
    a=parser().parse_args()
    try:train(a)
    except Exception as exc:
        write(Path(a.output)/'failure.json',{'type':type(exc).__name__,'message':str(exc),'time':time.time()});raise
