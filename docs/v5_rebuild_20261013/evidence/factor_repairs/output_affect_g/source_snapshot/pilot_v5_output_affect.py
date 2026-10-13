"""Mean adapter with output semantics: no evaluation-probe optimization."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.affect_trend import V5AffectTrend
from kinetalk_v5.affect_evidence import AffectEvidence,observed_statistics,affect_loss,query_moment_loss
from scripts.pilot_v5_affect_trend import fixed_digest
from scripts.pilot_v5_factor_rebuild import raw_audio,reference,reconstruction,evaluate
from scripts.train_v7_pilot import load,batch,sha
from scripts.train_expression_response import configure,load_runtime,write,save,state_digest


def run(a):
    configure(a.seed);start=time.time();device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    if sha(a.checkpoint)!=a.sha256:raise ValueError('starting checkpoint mismatch')
    original=V5AffectTrend.from_rebuild(torch.load(a.checkpoint,map_location='cpu',weights_only=False))
    original.flow.sample_gain.zero_()
    with torch.no_grad():
        original.style_response.basis[-1].weight.zero_();original.style_response.basis[-1].bias.zero_()
    original=original.to(device).eval().requires_grad_(False);expected=fixed_digest(original)
    binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    fit=raw_audio(load(Path(a.data)/'fit.npz'),mean,std)
    # Only known TRAIN real motion, no generated examples in observer fitting.
    known=fit['dataset_id'].eq(0)&fit['emotion_id'].ge(0)
    x=observed_statistics(fit['motion'][known],fit['valid'][known],fit['channel_mask'][known]).to(device)
    labels=fit['emotion_id'][known].to(device)
    evidence=AffectEvidence().to(device);evidence.fit_scales(x)
    opt=torch.optim.AdamW(evidence.parameters(),lr=.002)
    for step in range(a.observer_steps):
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(evidence.features(x),labels)
        loss.backward();opt.step()
    evidence.eval().requires_grad_(False);evidence_digest=state_digest(evidence)
    observer_fit_accuracy=float(evidence.features(x).argmax(-1).eq(labels).float().mean())
    save(out/'train_affect_evidence.pt',evidence.state_dict())
    rng=np.random.default_rng(a.seed)
    order=[rng.choice(len(fit['motion']),a.batch,replace=False).tolist() for _ in range(a.steps)]
    cached=[]
    with torch.no_grad():
        for step,ids in enumerate(order):
            b,r=batch(fit,ids,device)
            p=super(V5AffectTrend,original).affect_audio(b['audio_features'],b['valid'])
            s=original.encode_style(reference(r,step%2))
            cached.append((b,p,s))
    models={'frozen_start':original};history=[]
    for arm in ('modal_reconstruction','output_affect'):
        configure(a.seed);model=copy.deepcopy(original);model.audio_trend.requires_grad_(True)
        params=list(model.audio_trend.parameters());opt=torch.optim.AdamW(params,lr=a.lr)
        for step,(b,prior,style) in enumerate(cached):
            opt.zero_grad(set_to_none=True)
            correction=model.audio_trend(prior,b['valid'])
            p={**prior,'state':prior['state']+correction}
            y=model.decode(b,p,style,mode='modal')['motion']
            loss=reconstruction(model,y,b)+.002*correction[b['valid']].square().mean()
            sem=loss*0.;mom=loss*0.
            if arm=='output_affect':
                sem=affect_loss(evidence,y,b)
                mom=query_moment_loss(y,b,model.core.scales)
                loss=loss+a.semantic_weight*sem+a.moment_weight*mom
            if not torch.isfinite(loss):raise RuntimeError('nonfinite output affect objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);opt.step()
            row=dict(arm=arm,step=step+1,loss=float(loss),semantic_loss=float(sem),moment_loss=float(mom),
                     elapsed_seconds=time.time()-start);history.append(row)
            if (step+1)%32==0:write(out/'state.json',row);print(json.dumps(row),flush=True)
        model.eval().requires_grad_(False)
        if fixed_digest(model)!=expected:raise RuntimeError('frozen routes drifted')
        save(out/(arm+'.pt'),model.checkpoint());models[arm]=model
    if state_digest(evidence)!=evidence_digest:raise RuntimeError('real-motion observer changed in generator fitting')
    del cached
    results={};observer_diagnostic={}
    for role in ('held_sentence','unseen_speaker'):
        data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
        with torch.no_grad():
            logits=evidence(data['motion'].to(device),data['valid'].to(device),data['channel_mask'].to(device))
            observer_diagnostic[role]=float(logits.argmax(-1).cpu().eq(data['emotion_id']).float().mean())
        for arm,model in models.items():
            dest=out/arm;dest.mkdir(exist_ok=True);results.setdefault(arm,{})
            results[arm][role]=evaluate(model,data,device,binding,a.seed,dest,role)
    save(out/'frozen_start.pt',original.checkpoint());write(out/'history.json',history)
    report=dict(schema='v5_output_affect_pilot_v1',base_checkpoint=a.checkpoint,base_sha256=a.sha256,
        elapsed_seconds=time.time()-start,steps_per_arm=a.steps,semantic_weight=a.semantic_weight,
        moment_weight=a.moment_weight,observer_steps=a.observer_steps,
        real_train_observer_accuracy=observer_fit_accuracy,real_held_observer_accuracy=observer_diagnostic,
        training_observer_frozen_verified=True,independent_probes_used_in_training=False,sealed_test_loaded=False,
        frozen_routes_verified=True,flow_enabled=False,personal_response_enabled=False,
        results=results,paper_ready=False,limits=['training observer supervision is not independent evaluation',
            'mean native state not identified pure affect; teacher distillation disabled in this ablation',
            'modal schedule fixes train/deploy mismatch but uncertainty still requires innovation training',
            'style unchanged','small crop pilot does not establish paper contributions'])
    write(out/'report.json',report);print(json.dumps({'completed':str(out),'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('checkpoint','sha256','binding','data','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,default=47);p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=192);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--observer-steps',type=int,default=384);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--semantic-weight',type=float,default=.1);p.add_argument('--moment-weight',type=float,default=.2)
    run(p.parse_args())
