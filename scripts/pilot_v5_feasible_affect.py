"""Matched mean-head optimizer audit: ordinary vs guarded output supervision.

The guard projects the actual Adam parameter step, not a gradient before
Adam's preconditioning. It protects local derivatives for actual query mouth
reconstruction and TRAIN-only expression supervision; no held-data fitting.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.affect_trend import V5AffectTrend
from kinetalk_v5.affect_evidence import AffectEvidence, affect_loss, query_moment_loss
from kinetalk_v5.feasible_update import project_descent_step, flat_grad, flat_parameters, assign_parameters
from scripts.pilot_v5_affect_trend import fixed_digest
from scripts.pilot_v5_factor_rebuild import raw_audio, reference, reconstruction, evaluate
from scripts.train_v7_pilot import load, batch, sha
from scripts.train_expression_response import configure, load_runtime, write, save, state_digest


def run(a):
    configure(a.seed); started=time.time(); device=torch.device(a.device); out=Path(a.output)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    if sha(a.checkpoint)!=a.sha256:raise ValueError('starting checkpoint mismatch')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    original=V5AffectTrend.from_rebuild(ck).to(device).eval().requires_grad_(False)
    original.flow.sample_gain.zero_()
    with torch.no_grad():
        original.style_response.basis[-1].weight.zero_();original.style_response.basis[-1].bias.zero_()
    expected=fixed_digest(original)
    evidence=AffectEvidence().to(device)
    evidence.load_state_dict(torch.load(a.evidence,map_location='cpu',weights_only=True),strict=True)
    evidence.eval().requires_grad_(False); evidence_before=state_digest(evidence)
    binding=json.loads(Path(a.binding).read_text())
    runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    fit=raw_audio(load(Path(a.data)/'fit.npz'),mean,std)
    rng=np.random.default_rng(a.seed)
    order=[rng.choice(len(fit['motion']),a.batch,replace=False).tolist() for _ in range(a.steps)]
    cached=[]
    with torch.no_grad():
        for step,ids in enumerate(order):
            b,r=batch(fit,ids,device)
            p=super(V5AffectTrend,original).affect_audio(b['audio_features'],b['valid'])
            s=original.encode_style(reference(r,step%2));cached.append((b,p,s))
    models={'frozen_start':original};history=[]
    for arm in ('ordinary_output','guarded_output'):
        configure(a.seed);model=copy.deepcopy(original);model.audio_trend.requires_grad_(True)
        params=list(model.audio_trend.parameters());opt=torch.optim.AdamW(params,lr=a.lr)
        for step,(b,prior,style) in enumerate(cached):
            opt.zero_grad(set_to_none=True)
            correction=model.audio_trend(prior,b['valid'])
            p={**prior,'state':prior['state']+correction}
            y=model.decode(b,p,style,mode='modal')['motion']
            obs=b['valid'][...,None]&b['channel_mask'][:,None,14:40]
            mouth=((y[...,14:40]-b['motion'][...,14:40]).abs()/model.core.scales[14:40])[obs].mean()
            semantic=affect_loss(evidence,y,b)
            moments=query_moment_loss(y,b,model.core.scales)
            affect=a.semantic_weight*semantic+a.moment_weight*moments
            loss=reconstruction(model,y,b)+a.mouth_weight*mouth+affect+.002*correction[b['valid']].square().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
            lip_gradient=flat_grad(mouth,params);affect_gradient=flat_grad(affect,params)
            cosine=torch.nn.functional.cosine_similarity(lip_gradient[None],affect_gradient[None]).item()
            old=flat_parameters(params);loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);opt.step()
            proposed=flat_parameters(params)-old
            actual=project_descent_step(proposed,lip_gradient,affect_gradient) if arm=='guarded_output' else proposed
            if arm=='guarded_output':assign_parameters(params,old+actual)
            row=dict(arm=arm,step=step+1,loss=float(loss),mouth_loss=float(mouth),
                semantic_loss=float(semantic),moment_loss=float(moments),gradient_cosine=cosine,
                proposed_lip_derivative=float(proposed@lip_gradient),actual_lip_derivative=float(actual@lip_gradient),
                proposed_affect_derivative=float(proposed@affect_gradient),actual_affect_derivative=float(actual@affect_gradient),
                step_retained_fraction=float(actual.norm()/proposed.norm().clamp_min(1e-12)),
                elapsed_seconds=time.time()-started)
            history.append(row)
            if (step+1)%32==0:write(out/'state.json',row);print(json.dumps(row),flush=True)
        model.eval().requires_grad_(False)
        if fixed_digest(model)!=expected:raise RuntimeError('non-trend weights changed')
        save(out/(arm+'.pt'),model.checkpoint());models[arm]=model
    if state_digest(evidence)!=evidence_before:raise RuntimeError('frozen observer changed')
    del cached
    results={}
    for arm,model in models.items():
        dest=out/arm;dest.mkdir();results[arm]={}
        for role in ('held_sentence','unseen_speaker'):
            data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
            results[arm][role]=evaluate(model,data,device,binding,a.seed,dest,role)
    optimizer_audit={}
    for arm in ('ordinary_output','guarded_output'):
        rows=[r for r in history if r['arm']==arm]
        optimizer_audit[arm]=dict(conflicting_gradient_fraction=float(np.mean([r['gradient_cosine']<0 for r in rows])),
            lip_ascent_proposal_fraction=float(np.mean([r['proposed_lip_derivative']>1e-8 for r in rows])),
            lip_ascent_actual_fraction=float(np.mean([r['actual_lip_derivative']>1e-8 for r in rows])),
            mean_step_retained_fraction=float(np.mean([r['step_retained_fraction'] for r in rows])))
    write(out/'history.json',history)
    write(out/'report.json',dict(schema='v5_feasible_affect_pilot_v1',elapsed_seconds=time.time()-started,
        base_sha256=sha(a.checkpoint),evidence_sha256=sha(a.evidence),steps_per_arm=a.steps,
        semantic_weight=a.semantic_weight,moment_weight=a.moment_weight,mouth_weight=a.mouth_weight,
        results=results,optimizer_audit=optimizer_audit,frozen_routes_verified=True,
        independent_probes_used_in_training=False,sealed_test_loaded=False,
        flow_enabled=False,personal_response_enabled=False,paper_ready=False,
        limits=['first-order update constraint, not a finite-loss/generalization guarantee',
            'known constrained optimizer technique, not claimed research novelty',
            'native event sampling retained; no new stochastic dynamics validated',
            'no teacher or style-success claim from a mean-head optimizer ablation']))
    print(json.dumps({'completed':str(out),'elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('checkpoint','sha256','evidence','binding','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--steps',type=int,default=192);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--semantic-weight',type=float,default=.025)
    p.add_argument('--moment-weight',type=float,default=.2);p.add_argument('--mouth-weight',type=float,default=.5)
    run(p.parse_args())
