"""Bounded v6 TRAIN continuation with resumable optimizer and frozen bootstrap.

The parent decoder is a TRAIN-only distillation target during a fixed bootstrap
interval. It is not saved in the deployed model and never receives dev/test input
for optimization. Same TRAIN query/reference order supports matched arms/seeds.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import motion_objective
from kinetalk_v2.unified_model import relation_style_loss
from kinetalk_v2.event_execution import event_execution_loss
from kinetalk_v4.unified_model_v4 import v4_losses,masked_mean
from kinetalk_v4.rollout_statistics import expression_mean_loss
from kinetalk_v6.factor_paths import FactorPathKineTalkV6,distribution_losses
from scripts.load_candidate_model import load_candidate_model
from scripts.train_expression_response import (
    configure,development_fold,load_runtime,save,sha,state_digest,write,
)
from scripts.train_unified_v4 import (
    attach_base,build_base_cache,evaluate,make_refs,motion_targets,reference_ids,select_balanced,
)
from scripts.train_v5_mixed import next_ids


def v6_objective(model,b,refs,targets,channels,*,parent=None,bootstrap=0.,neutral_id=0):
    out=model.training_forward(b,refs,targets)
    motion,parts=motion_objective(out['motion'],b['motion'],b['valid'],
        b['channel_mask'],b['times'],model.scales)
    legacy=v4_losses(model,out,targets,b)
    distribution=distribution_losses(model,out,targets,b)
    posterior,_=motion_objective(out['posterior_motion'],b['motion'],b['valid'],
        b['channel_mask'],b['times'],model.scales)
    qs=model.style(b['motion'],b['valid'],b['channel_mask'],model.scales)
    relation=relation_style_loss(torch.cat((out['style'],qs)),b['speaker_id'].repeat(2))
    consistency=F.smooth_l1_loss(out['style'],qs)
    # TRAIN query motion supplies another same-speaker performance for style
    # consistency only. It is never a deployment reference or state input.
    alternate=model.decode(b['h0'],b['b0'],b['valid'],out['conditioned'],qs.detach(),out['events'])['motion']
    supported=b['valid'][...,None]&b['channel_mask'][:,None]
    reference_consistency=masked_mean((alternate-out['motion'].detach()).abs()/model.scales,supported)
    execution=event_execution_loss(out['sampled_motion'],out['zero_event_motion'],
        out['sampled_schedule']['events'],out['conditioned']['state'].detach(),b['b0'].detach(),
        b['valid'],b['times'],b['channel_mask'],channels,
        quality_reference_motion=out['zero_event_motion'].detach())['total']
    cond=out['conditioned']
    zeros={k:torch.zeros_like(v) for k,v in cond.items()}
    content_only=model.decode(b['h0'],b['b0'],b['valid'],zeros,
        torch.zeros_like(out['style']),torch.zeros_like(out['events']))['motion']
    neutral_mask=b['valid'][...,None]&b['channel_mask'][:,None]&b['emotion_id'].eq(neutral_id)[:,None,None]
    content_anchor=masked_mean((content_only-b['motion']).abs()/model.scales,neutral_mask)
    # A posterior draw sees GT through q and is a legitimate reconstruction
    # path. The audio prior random draw is trained by event/statistic objectives.
    draw=model.distribution_sample(out['student'],b['valid'],b['times'],
        torch.Generator(device=b['motion'].device).manual_seed(148)) if model.cfg.distribution_weight else out['student']
    sampled_cond=model.condition(draw,b['valid'])
    rollout=model.decode(b['h0'],b['b0'],b['valid'],sampled_cond,out['style'],
                         out['sampled_schedule']['events'].detach())['motion']
    rollout_mean=expression_mean_loss(rollout,b['motion'],b['valid'],b['channel_mask'],model.scales)
    bootstrap_loss=motion*0.
    if parent is not None and bootstrap>0:
        with torch.no_grad():
            p=parent.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,
                             times=b['times'],mode='modal')
            target=parent.decode(b['h0'],b['b0'],b['valid'],p['conditioned'],
                                p['style'],targets['events'])['motion']
        mask=b['valid'][...,None]&b['channel_mask'][:,None]
        bootstrap_loss=masked_mean((out['motion']-target).square()/model.scales.square(),mask)
    loss=(motion+legacy['total']+model.cfg.posterior_weight*posterior+distribution['total']+
        .25*content_anchor+.05*relation+.05*consistency+.05*execution+.5*rollout_mean+
        bootstrap*bootstrap_loss+model.cfg.reference_weight*reference_consistency)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite v6 full objective')
    names=dict(loss=loss,motion=motion,position=parts['position'],posterior=posterior,
        prior_nll=distribution['prior_nll'],prior_kl=distribution['posterior_to_prior_kl'],
        posterior_kl=distribution['posterior_kl'],content_anchor=content_anchor,
        relation=relation,reference_consistency=reference_consistency,execution=execution,
        rollout_mean=rollout_mean,bootstrap=bootstrap_loss)
    return loss,out,{k:float(v.detach()) for k,v in names.items()}


def digest(value):return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def run(a):
    if min(a.updates,a.batch_size)<1:raise ValueError('positive budgets required')
    root=Path(a.output)
    if root.exists() and not a.resume:raise FileExistsError('fresh output required')
    root.mkdir(parents=True,exist_ok=bool(a.resume));configure(a.seed)
    device=torch.device(a.device);binding=json.loads(Path(a.binding).read_text())
    data,base=load_runtime(binding,device);frozen=state_digest(base.stage1)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    parent=load_candidate_model(ck,device).eval().requires_grad_(False)
    cfg_overrides=dict(teacher_guidance_mode='none',state_supervision_space='raw',
                       geometry_audio_mode='none')
    if a.arm=='direct':cfg_overrides.update(posterior_weight=0.,distribution_weight=0.)
    model=FactorPathKineTalkV6.from_parent(ck,device,**cfg_overrides)
    configure(a.seed);q=data['splits']['train'];fold=development_fold(q,data['fit_sids'])
    fit=list(fold['train']);held=select_balanced(q,fold['speaker_dev']+fold['sentence_dev'],a.eval_limit,47)
    pool={}
    for i in fit:pool.setdefault(int(q['speaker_id'][i]),[]).append(i)
    rng=random.Random(47)
    refs={i:reference_ids(q,[i],pool,rng)[0] for i in fit}
    order=list(fit);rng.shuffle(order);position=0
    # Cache the fixed required TRAIN/internal-dev inputs once per run.
    print(json.dumps(dict(event='cache_start',arm=a.arm,seed=a.seed,clips=len(set(fit+held)))),flush=True)
    cache=build_base_cache(q,base,sorted(set(fit+held)),device)
    def rebuild():
        order=list(fit);rng.shuffle(order);return order
    neutral_id=data['config']['data']['emotion_classes'].index('neutral')
    protocol=dict(schema='factor_paths_v6_bounded_v1',arm=a.arm,seed=a.seed,
        updates=a.updates,batch_size=a.batch_size,lr=a.lr,config=asdict(model.cfg),
        source_checkpoint_sha256=sha(a.checkpoint),binding_sha256=sha(a.binding),
        source_sha256=sha(__file__),model_source_sha256=sha(sys.modules[FactorPathKineTalkV6.__module__].__file__),
        neutral_digest=frozen,fit_indices_sha256=digest(fit),reference_sha256=digest(sorted(refs.items())),
        geometry_channels=ck['protocol']['geometry_channels'],hdtf_enabled=False,text_enabled=False,
        test_loaded=False,paper_result=False,bootstrap_updates=a.bootstrap_updates,
        parent_used='frozen TRAIN bootstrap only; absent from saved inference model',
        objective='complete observed reconstruction/geometry/state/class/event likelihood + posterior reconstruction/distribution KL,NLL + neutral content anchor + single-reference relation + execution/rollout mean')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    step=0;sequence=[];history=[]
    if a.resume:
        saved=torch.load(a.resume,map_location=device,weights_only=False)
        if saved['protocol']!=protocol:raise ValueError('resume protocol differs')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        step,sequence,history=saved['step'],saved['sequence'],saved['history']
        order,position=saved['order'],saved['position'];rng.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    else:write(root/'protocol.json',protocol)
    begin=time.monotonic();verified=False;model.train()
    while step<a.updates:
        ids,order,position=next_ids(order,position,a.batch_size,rebuild)
        if not set(ids).issubset(fit) or not set(refs[i] for i in ids).issubset(fit):
            raise ValueError('query/reference outside TRAIN internal fold')
        sequence.extend(ids);b=attach_base(q.batch(torch.tensor(ids),device),ids,cache)
        reference=make_refs(q,[refs[i] for i in ids],device);targets=motion_targets(b)
        optimizer.zero_grad(set_to_none=True)
        bootstrap=model.cfg.bootstrap_weight if step<a.bootstrap_updates else 0.
        loss,out,logs=v6_objective(model,b,reference,targets,protocol['geometry_channels'],
                                  parent=parent,bootstrap=bootstrap,neutral_id=neutral_id)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        if not verified:
            heads=[model.decoder.content_response,model.decoder.expression_response,
                   model.decoder.interaction_response,model.decoder.style_posture,model.student_logvar]
            for head in heads:
                grad=head.weight.grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum()>0:
                    raise RuntimeError('missing finite nonzero new-head gradient')
            verified=True
        optimizer.step();step+=1
        row=dict(step=step,gradient_norm=float(norm),**logs);history.append(row)
        if step==1 or step%20==0 or step==a.updates:
            state=dict(status='training',arm=a.arm,seed=a.seed,total_updates=a.updates,
                       elapsed_seconds=time.monotonic()-begin,**row)
            write(root/'state.json',state);print(json.dumps(state),flush=True)
        if step%64==0 or step==a.updates:
            if state_digest(base.stage1)!=frozen or any(p.grad is not None for p in base.stage1.parameters()):
                raise RuntimeError('frozen B0 changed')
            saved=dict(model=model.state_dict(),config=asdict(model.cfg),protocol=protocol,
                optimizer=optimizer.state_dict(),step=step,sequence=sequence,history=history,
                order=order,position=position,python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            save(root/'resume.pt',saved)
    internal=evaluate(model,q,held,cache,data,device,a.seed,protocol['geometry_channels'])
    save(root/'final.pt',{k:v for k,v in saved.items() if k!='optimizer'})
    write(root/'history.json',history)
    write(root/'complete.json',dict(status='complete',updates=step,internal_dev=internal,
        protocol=protocol,head_gradients_verified=verified,checkpoint_sha256=sha(root/'final.pt'),
        sequence_sha256=digest(sequence),elapsed_seconds=time.monotonic()-begin,paper_result=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--arm',choices=('direct','posterior'),default='posterior')
    p.add_argument('--updates',type=int,default=768);p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=47)
    p.add_argument('--eval-limit',type=int,default=128);p.add_argument('--bootstrap-updates',type=int,default=128)
    p.add_argument('--device',default='cuda');p.add_argument('--resume');run(p.parse_args())
