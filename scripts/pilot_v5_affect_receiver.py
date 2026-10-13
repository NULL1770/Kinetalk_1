"""Bounded emotion-receiver reopening, frozen style/content/events/teacher.

Joint output remains unrestricted. Compare two matched arms to test whether
native-state adaptation alone is an expression bottleneck. This ablation is
not a claimed new teacher, flow decoder, or independent disentanglement proof.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.joint_factor_repair import V5JointFactorRepair
from kinetalk_v5.affect_evidence import AffectEvidence,affect_loss,query_moment_loss
from scripts.pilot_v5_factor_rebuild import raw_audio,reference,reconstruction,evaluate
from scripts.train_v7_pilot import load,batch,sha
from scripts.train_expression_response import configure,load_runtime,write,save,state_digest


def fixed_digest(model):
    h=hashlib.sha256()
    for key,x in sorted(model.state_dict().items()):
        if key.startswith(('audio_trend.','core.decoder.affect.')):continue
        h.update(key.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(a):
    configure(a.seed);start=time.time();device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    model=V5JointFactorRepair.from_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=False)).to(device).eval().requires_grad_(False)
    if float(model.flow.sample_gain)!=0:raise ValueError('new flow must be disabled')
    expected=fixed_digest(model)
    evidence=AffectEvidence().to(device)
    evidence.load_state_dict(torch.load(a.evidence,map_location='cpu',weights_only=True),strict=True)
    evidence.eval().requires_grad_(False);edigest=state_digest(evidence)
    binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    data=raw_audio(load(Path(a.data)/'fit.npz'),mean,std)
    rng=np.random.default_rng(a.seed);order=[rng.choice(len(data['motion']),a.batch,replace=False).tolist() for _ in range(a.steps)]
    cached=[]
    with torch.no_grad():
        for step,ids in enumerate(order):
            b,r=batch(data,ids,device);s=model.encode_style(reference(r,step%2));cached.append((b,s))
    models={'frozen_start':model};history=[]
    for arm in ('state_only','state_receiver'):
        configure(a.seed);candidate=copy.deepcopy(model);candidate.audio_trend.requires_grad_(True)
        if arm=='state_receiver':candidate.core.decoder.affect.requires_grad_(True)
        parameters=[p for p in candidate.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(parameters,lr=a.lr)
        prior={k:x.detach().clone() for k,x in candidate.core.decoder.affect.state_dict().items()}
        for step,(b,s) in enumerate(cached):
            opt.zero_grad(set_to_none=True);p=candidate.affect_audio(b['audio_features'],b['valid'])
            y=candidate.decode(b,p,s,mode='modal')['motion']
            semantic=affect_loss(evidence,y,b);moments=query_moment_loss(y,b,candidate.core.scales)
            obs=b['valid'][...,None]&b['channel_mask'][:,None,14:40]
            mouth=((y[...,14:40]-b['motion'][...,14:40]).abs()/candidate.core.scales[14:40])[obs].mean()
            anchor=sum((x-prior[k]).square().mean() for k,x in candidate.core.decoder.affect.state_dict(keep_vars=True).items())
            loss=reconstruction(candidate,y,b)+.5*mouth+.05*semantic+.4*moments+.1*anchor
            if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,3.);opt.step()
            row=dict(arm=arm,step=step+1,loss=float(loss),semantic=float(semantic),moments=float(moments),mouth=float(mouth),elapsed=time.time()-start)
            history.append(row)
            if (step+1)%32==0:print(json.dumps(row),flush=True);write(out/'state.json',row)
        candidate.eval().requires_grad_(False)
        if fixed_digest(candidate)!=expected:raise RuntimeError('non-affect frozen tensor drift')
        models[arm]=candidate;save(out/(arm+'.pt'),candidate.checkpoint())
    if state_digest(evidence)!=edigest:raise RuntimeError('observer changed')
    results={}
    for arm,candidate in models.items():
        results[arm]={};dest=out/arm;dest.mkdir()
        for role in ('held_sentence','unseen_speaker'):
            q=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
            results[arm][role]=evaluate(candidate,q,device,binding,a.seed,dest,role)
    write(out/'history.json',history)
    write(out/'report.json',dict(schema='v5_affect_receiver_pilot_v1',elapsed_seconds=time.time()-start,
        checkpoint_sha256=sha(a.checkpoint),evidence_sha256=sha(a.evidence),steps_per_arm=a.steps,
        results=results,frozen_routes_verified=True,independent_probes_used_in_training=False,
        sealed_test_loaded=False,flow_enabled=False,paper_ready=False,
        limits=['matched short optimization diagnostic; no full paper innovation inference',
            'reopening affect receiver can affect style interaction despite frozen style tensors',
            'independent donor-direction and native dynamic audits still required']))
    print(json.dumps({'completed':str(out),'elapsed_seconds':time.time()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('checkpoint','evidence','binding','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=5e-5);run(p.parse_args())
