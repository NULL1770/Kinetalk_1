"""Independent reference-only training with output-level swap constraints.

Swapped-person motion has no framewise GT target. Its semantic label belongs
to the query, and articulation timing is compared to the same query's frozen
baseline geometry with offset/amplitude normalization. Proper-style paired
reconstruction remains active to resist loss of useful personal information.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import numpy as np
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.factor_rebuild import V5FactorRebuild
from kinetalk_v5.affect_evidence import AffectEvidence,affect_loss
from scripts.pilot_v5_factor_rebuild import raw_audio,reference,reconstruction,evaluate,contrastive_style
from scripts.train_v7_pilot import load,batch,sha
from scripts.train_expression_response import configure,load_runtime,write,save,state_digest
from kinetalk_v4.unified_model_v4 import affine_geometry
from kinetalk_b0.models.expression_response import masked_pool
from kinetalk_v2.unified_model import adjacency


def reference_fixed_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('reference_context.'):continue
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def timing_loss(model, swapped, baseline, batch):
    """Rig geometry temporal pattern, no forced person-specific mouth width."""
    j=model.core.decoder.geometry_jacobian;off=model.core.decoder.geometry_offset
    dependencies=j.abs().gt(1e-7)
    supported=(batch['channel_mask'][:,None]|~dependencies[None]).all(-1)
    p=affine_geometry(swapped,j,off);g=affine_geometry(baseline.detach(),j,off)
    valid=batch['valid']
    def normalize(x):
        mean=masked_pool(x,valid)[:,None]
        scale=masked_pool((x-mean).square(),valid).sqrt().clamp_min(.1)[:,None]
        return torch.where(valid[...,None],(x-mean)/scale,0.)
    p,g=normalize(p),normalize(g)
    obs=valid[...,None]&supported[:,None]
    edge=adjacency(valid,batch['times'])[...,None]&supported[:,None]
    error=F.smooth_l1_loss(p,g,reduction='none')
    velocity=F.smooth_l1_loss(p[:,1:]-p[:,:-1],g[:,1:]-g[:,:-1],reduction='none')
    return error[obs].mean()+.25*velocity[edge].mean()


def batch_donors(people):
    donor=[];use=[]
    for i,p in enumerate(people):
        ids=(people!=p).nonzero().flatten()
        donor.append(int(ids[0]) if len(ids) else i);use.append(bool(len(ids)))
    return torch.tensor(donor,device=people.device),torch.tensor(use,device=people.device)


def run(a):
    configure(a.seed);start=time.time();device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    if sha(a.checkpoint)!=a.sha256:raise ValueError('base mismatch')
    original=V5FactorRebuild.from_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=False))
    original.flow.sample_gain.zero_()
    with torch.no_grad():
        original.style_response.basis[-1].weight.zero_();original.style_response.basis[-1].bias.zero_()
    original=original.to(device).eval().requires_grad_(False);expected=reference_fixed_digest(original)
    evidence=AffectEvidence().to(device)
    evidence.load_state_dict(torch.load(a.evidence,map_location='cpu',weights_only=True),strict=True)
    evidence.eval().requires_grad_(False);evidence_before=state_digest(evidence)
    binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    fit=raw_audio(load(Path(a.data)/'fit.npz'),mean,std)
    rng=np.random.default_rng(a.seed)
    order=[rng.choice(len(fit['motion']),a.batch,replace=False).tolist() for _ in range(a.steps)]
    models={'frozen_start':original};history=[]
    for arm in ('paired_only','swap_preserve'):
        configure(a.seed);model=copy.deepcopy(original);model.reference_context.requires_grad_(True)
        params=list(model.reference_context.parameters());opt=torch.optim.AdamW(params,lr=a.lr)
        for step,ids in enumerate(order):
            b,refs=batch(fit,ids,device);r0,r1=reference(refs,0),reference(refs,1)
            with torch.no_grad():
                p=model.affect_audio(b['audio_features'],b['valid'])
                prior_s=original.encode_style(r0)
                baseline=original.decode(b,p,prior_s,mode='modal')['motion']
            opt.zero_grad(set_to_none=True)
            s0,s1=model.encode_style(r0),model.encode_style(r1)
            y=model.decode(b,p,s0,mode='modal')['motion']
            person=b['speaker_id']+100000*b['dataset_id']
            spread=prior_s.std(0,correction=0).clamp_min(.25).detach()
            loss=reconstruction(model,y,b)+.1*((s0-s1)/spread).square().mean()
            loss=loss+.02*contrastive_style(s0,s1,person)
            swap_sem=loss*0.;timing=loss*0.
            if arm=='swap_preserve':
                donor,use=batch_donors(person)
                swapped=model.decode(b,p,s0[donor],mode='modal')['motion']
                y1=model.decode(b,p,s1,mode='modal')['motion']
                # Factual emotion label only, including across reference swaps.
                swap_sem=(affect_loss(evidence,y,b)+affect_loss(evidence,y1,b)+
                           affect_loss(evidence,swapped,{**b,'dataset_id':
                             torch.where(use,b['dataset_id'],torch.ones_like(b['dataset_id']))}))/3.
                timing=timing_loss(model,swapped,baseline,b)
                loss=loss+a.semantic_weight*swap_sem+a.timing_weight*timing
            if not torch.isfinite(loss):raise RuntimeError('nonfinite style objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);opt.step()
            row=dict(arm=arm,step=step+1,loss=float(loss),swap_semantic=float(swap_sem),
                     timing_loss=float(timing),elapsed_seconds=time.time()-start)
            history.append(row)
            if (step+1)%32==0:write(out/'state.json',row);print(json.dumps(row),flush=True)
        model.eval().requires_grad_(False)
        if reference_fixed_digest(model)!=expected:raise RuntimeError('non-reference weights drifted')
        save(out/(arm+'.pt'),model.checkpoint());models[arm]=model
    if state_digest(evidence)!=evidence_before:raise RuntimeError('frozen observer changed')
    results={}
    for arm,model in models.items():
        dest=out/arm;dest.mkdir();results[arm]={}
        for role in ('held_sentence','unseen_speaker'):
            data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
            results[arm][role]=evaluate(model,data,device,binding,a.seed,dest,role)
    save(out/'frozen_start.pt',original.checkpoint());write(out/'history.json',history)
    write(out/'report.json',dict(schema='v5_reference_output_pilot_v1',elapsed_seconds=time.time()-start,
        base_checkpoint=a.checkpoint,base_sha256=a.sha256,evidence_sha256=sha(a.evidence),steps_per_arm=a.steps,
        semantic_weight=a.semantic_weight,timing_weight=a.timing_weight,
        frozen_routes_verified=True,independent_probes_used_in_training=False,sealed_test_loaded=False,
        results=results,paper_ready=False,limits=['TRAIN-real classifier gives supervision, four unused probes evaluate',
          'normalized geometric timing is a phonetic proxy, not certified content preservation',
          'style naturalness/target style transfer not proven by swap label stability',
          'reference correction retains native style bypass and may retain nonlinear nuisance']))
    print(json.dumps({'completed':str(out),'elapsed_seconds':time.time()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('checkpoint','sha256','binding','data','output','evidence'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--semantic-weight',type=float,default=.1)
    p.add_argument('--timing-weight',type=float,default=.1);run(p.parse_args())
