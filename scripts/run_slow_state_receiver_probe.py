"""Oracle receiver probe using the exact slow-state target used by the student.

This is an upper-bound diagnostic: the target state comes from observed motion
through ``load_context`` and is never available at deployment.  It verifies
that the frozen Stage4 upper-only composition can consume the same four-state
representation that the audio student will predict.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.regional_intensity_gain import regional_envelope
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face, lift_slow_state
from scripts.train_full_staged import NOT_UPPER, subset
from scripts.train_isolated_audio_state import load_context, old_prediction
from scripts.train_formal_predictable_projection import save_json
from scripts.extract_emotion2vec_pilot import sha


def _center(x, valid):
    mask = valid[..., None]
    clean = torch.where(mask, x, 0.)
    mean = clean.sum(1, keepdim=True) / valid.sum(1, keepdim=True).to(x.dtype).clamp_min(1.)[..., None]
    return torch.where(mask, x - mean, 0.)


def _runs(mask):
    ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not ids: return []
    out=[]; left=prev=ids[0]
    for i in ids[1:]:
        if i != prev+1: out.append((left, prev+1)); left=i
        prev=i
    out.append((left, prev+1)); return out


def _corr(x, y, valid):
    values=[]
    for b in range(len(x)):
        for l,r in _runs(valid[b]):
            a=x[b,l:r]; c=y[b,l:r]; a=a-a.mean(); c=c-c.mean()
            den=(a.square().sum()*c.square().sum()).sqrt()
            if float(den)>1e-12: values.append(float((a*c).sum()/den))
    return float(np.mean(values)) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','source','output'): p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--smoke',action='store_true'); p.add_argument('--seed',type=int,default=47)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Fresh slow oracle output required')
    a.output.mkdir(parents=True); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data,system,audio,identities,_=load_context(a.data,a.source,a.device,a.seed)
    q=data['splits']['validation']
    if a.smoke: q=subset(q,torch.arange(min(32,len(q['valid']))),'cpu')
    valid=q['valid'].to(a.device)
    target_state=q['target_centered_state'].to(a.device)
    target_upper=lift_slow_state(target_state,data['target_scales'].to(a.device))[...,list(UPPER_INDICES)]
    target_upper=_center(target_upper,valid)
    target_env=regional_envelope(target_upper,valid,channel_scales=data['target_scales'][list(UPPER_INDICES)].to(a.device))
    ids=torch.arange(len(valid)); base=[]
    for ix in ids.split(16):
        b=subset(q,ix,a.device); b['frozen_local']=audio(b['audio_features'],b['valid'])['local']
        noise=torch.randn((*b['valid'].shape,52),generator=torch.Generator().manual_seed(a.seed+int(ix[0])))
        base.append(old_prediction(system,b,identities,noise.to(a.device)).cpu())
    base=torch.cat(base); mean=(base[...,list(UPPER_INDICES)]*q['valid'][...,None]).sum(1)/q['valid'].sum(1)[:,None]
    zero=torch.zeros_like(target_state); reverse=target_state.clone()
    for i in range(len(reverse)):
        ii=valid[i].nonzero().flatten(); reverse[i,ii]=reverse[i,ii.flip(0)]
    conditions={'zero':zero,'oracle':target_state,'reverse':reverse}
    scores={}
    for name,state in conditions.items():
        upper=mean.to(a.device)[:,None]+lift_slow_state(state,data['target_scales'].to(a.device))[...,list(UPPER_INDICES)]
        pred=compose_upper_face(base.to(a.device),upper,valid)
        centered=_center(pred[...,list(UPPER_INDICES)],valid)
        env=regional_envelope(centered,valid,channel_scales=data['target_scales'][list(UPPER_INDICES)].to(a.device))
        scores[name]={'state_mse':float(((state-target_state).square()[valid]).mean()),
                      'envelope_mse':float(((env-target_env).square()[valid]).mean()),
                      'envelope_corr':_corr(env,target_env,valid),
                      'nonupper43_exact':bool(torch.equal(pred[...,list(NOT_UPPER)].view(torch.int32),base.to(a.device)[...,list(NOT_UPPER)].view(torch.int32))),
                      'mean_drift':float((((pred[...,list(UPPER_INDICES)]*valid[...,None]).sum(1)/valid.sum(1)[:,None])-mean.to(a.device)).abs().max())}
    oracle=scores['oracle']; zero_score=scores['zero']; reverse_score=scores['reverse']
    result={'schema':'slow_state_receiver_probe_v1','oracle_only':True,'audio_student_started':False,
            'scores':scores,'receiver_passed':bool(oracle['state_mse']<1e-10 and oracle['envelope_mse']<zero_score['envelope_mse'] and oracle['envelope_corr'] is not None and reverse_score['envelope_corr'] is not None and oracle['envelope_corr']>reverse_score['envelope_corr'] and all(v['nonupper43_exact'] and v['mean_drift']<1e-5 for v in scores.values())),
            'data_manifest_sha256':data['provenance']['manifest_sha256'],'source_sha256':sha(a.source),'test_loaded':False,'default_replaced':False,'elapsed_seconds':time.monotonic()}
    save_json(a.output/'evaluation.json',result); save_json(a.output/'protocol.json',{'schema':'slow_state_receiver_probe_v1','target':'load_context target_centered_state','composition':'frozen Stage4 upper mean + lift_slow_state','oracle_only':True,'audio_student_started':False,'test_loaded':False,'default_replaced':False})
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
