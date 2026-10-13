"""Matched mean-state distillation arms; no decoder/style or probe training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.affect_trend import V5AffectTrend
from scripts.pilot_v5_factor_rebuild import raw_audio, reference, reconstruction, evaluate
from scripts.train_v7_pilot import load, batch, sha
from scripts.train_expression_response import configure, load_runtime, write, save, state_digest
from kinetalk_v2.unified_model import teacher_targets


def fixed_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('audio_trend.'):
            continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(a):
    configure(a.seed); started=time.time(); device=torch.device(a.device); out=Path(a.output)
    if out.exists(): raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    if sha(a.checkpoint) != a.sha256: raise ValueError('starting checkpoint mismatch')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    original=V5AffectTrend.from_rebuild(ck).to(device).eval().requires_grad_(False)
    # The common anchor excludes the rejected response/flow, without touching
    # saved pilot_b. Original event sampling remains active at deployment.
    with torch.no_grad():
        original.flow.sample_gain.zero_()
        original.style_response.basis[-1].weight.zero_()
        original.style_response.basis[-1].bias.zero_()
    expected=fixed_digest(original)
    binding=json.loads(Path(a.binding).read_text())
    runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu()
    del runtime,base
    fit=raw_audio(load(Path(a.data)/'fit.npz'),mean,std)
    rng=np.random.default_rng(a.seed)
    order=[rng.choice(len(fit['motion']),min(a.batch,len(fit['motion'])),replace=False).tolist()
           for _ in range(a.steps)]
    # Audio feature extraction and teacher are frozen; caching avoids three
    # repeats and ensures every arm sees the same privileged supervision.
    cached=[]
    with torch.no_grad():
        for step,ids in enumerate(order):
            b,r=batch(fit,ids,device)
            p=super(V5AffectTrend,original).affect_audio(b['audio_features'],b['valid'])
            t=original.visual_teacher(b)
            s=original.encode_style(reference(r,step%2))
            targets=teacher_targets(b['motion'],b['b0'],b['valid'],b['channel_mask'],b['times'])
            cached.append((b,p,t,s,targets['events']))
    history=[]; models={'frozen_start':original}
    for arm in ('state_only','reconstruction_only','combined'):
        configure(a.seed); model=copy.deepcopy(original)
        model.audio_trend.requires_grad_(True)
        params=list(model.audio_trend.parameters()); opt=torch.optim.AdamW(params,lr=a.lr)
        for step,(b,prior,teacher,style,events) in enumerate(cached):
            opt.zero_grad(set_to_none=True)
            correction=model.audio_trend(prior,b['valid'])
            p={**prior,'state':prior['state']+correction}
            state_error=F.smooth_l1_loss(p['state'][b['valid']],teacher['state'][b['valid']])
            loss=state_error if arm=='state_only' else state_error*0.
            if arm != 'state_only':
                y=model.decode(b,p,style,events=events)['motion']
                loss=loss+reconstruction(model,y,b)
                if arm=='combined': loss=loss+a.state_weight*state_error
            # Bound correction norm softly, without changing teacher targets.
            loss=loss+.002*correction[b['valid']].square().mean()
            if not torch.isfinite(loss): raise RuntimeError('nonfinite trend objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,5.); opt.step()
            row=dict(arm=arm,step=step+1,loss=float(loss),state_loss=float(state_error),
                     elapsed_seconds=time.time()-started)
            history.append(row)
            if (step+1)%32==0: write(out/'state.json',row); print(json.dumps(row),flush=True)
        model.eval().requires_grad_(False)
        if fixed_digest(model)!=expected: raise RuntimeError('frozen route drifted')
        save(out/(arm+'.pt'),model.checkpoint()); models[arm]=model
    save(out/'frozen_start.pt',original.checkpoint()); del cached
    results={}
    for arm,model in models.items():
        dest=out/arm;dest.mkdir(); results[arm]={}
        for role in ('held_sentence','unseen_speaker'):
            data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
            results[arm][role]=evaluate(model,data,device,binding,a.seed,dest,role)
    report=dict(schema='v5_mean_trend_pilot_v1',elapsed_seconds=time.time()-started,base_checkpoint=a.checkpoint,
        base_sha256=a.sha256,steps_per_arm=a.steps,seed=a.seed,lr=a.lr,state_weight=a.state_weight,
        training_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        data_sha256={k:sha(Path(a.data)/(k+'.npz')) for k in ('fit','held_sentence','unseen_speaker')},
        frozen_routes_verified=True,common_frozen_digest=expected,flow_enabled=False,
        personal_response_enabled=False,independent_probes_used_in_training=False,sealed_test_loaded=False,
        results=results,paper_ready=False,limits=['deterministic conditional mean is not observed trajectory certainty',
           'existing event sampling retained, new stochastic innovation not yet trained',
           'teacher native state is not an identified pure emotion coordinate',
           'style encoder unchanged; no disentanglement claim','short crop/small speaker pilot'])
    write(out/'history.json',history); write(out/'report.json',report)
    print(json.dumps({'completed':str(out),'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('checkpoint','sha256','binding','data','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    p.add_argument('--steps',type=int,default=192);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--state-weight',type=float,default=1.)
    train(p.parse_args())
