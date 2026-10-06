"""Reference-only fixed identity-budget trial; frozen generator is not evaluated.

Restart the original identity optimizer from timing000, compare 12 vs 120
additional reference epochs for seeds47/48/49. No new loss or query fitting.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.diagnose_flow_sampling import sha
from scripts.packed_trainval_cache import load_packed
from scripts.train_full_staged import (base_forward, encode_ref, identity_pairs,
    identity_report, mse, obs, state_hash, style_contrastive, subset)


@torch.no_grad()
def query_shape(system,data,z,device):
    result={}
    regions=dict(all=list(range(51)),mouth=list(range(14,41)),brows=list(range(41,46)),jawOpen=[17])
    for role,sids in (('train',data['fit_sids']),('validation',data['dev_sids'])):
        selected=z[role+'__emotion']==0
        target=(z[role+'__gt_mean']-z[role+'__b0_mean'])[selected].astype(np.float64)
        speakers=z[role+'__speaker'][selected]
        baselines={}
        for sid in sids:
            ids=range(len(data['refs'][sid]['valid']))
            baselines[sid]=encode_ref(system,data,sid,ids,device)['baseline'][0].cpu().numpy()
        bias=np.stack([baselines[int(s)] for s in speakers]).astype(np.float64)
        error=(target-bias)**2
        result[role]=dict(clips=int(selected.sum()),
            **{name+'_mean_mse':float(error[:,cols].mean()) for name,cols in regions.items()})
    return result


def run(a):
    if a.output.exists():raise FileExistsError('Fresh identity-budget diagnostic required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    assert sha(a.checkpoint)==a.checkpoint_sha256
    source=json.loads((a.conditions/'report.json').read_text())
    assert source['checkpoint_sha256']==a.checkpoint_sha256 and not source['test_loaded']
    z=np.load(a.conditions/'per_clip_conditions.npz',allow_pickle=False)
    data=load_packed(a.data,materialize=False,with_refs=True)
    assert set(data['splits'])=={'train','validation'} and not data['provenance']['test_loaded']
    assert data['provenance']['manifest_sha256']==source['data_manifest_sha256']
    for role,q in data['splits'].items():
        np.testing.assert_array_equal(q['clip_id'],z[role+'__clip_id'])
        np.testing.assert_array_equal(q['speaker_id'].numpy(),z[role+'__speaker'])
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    system=NeutralAffectSystem(ck['config']).to(a.device).eval()
    system.load_state_dict(ck['system'],strict=True);system.requires_grad_(False)
    warm=copy.deepcopy(system.state_dict())
    for sid,q in data['refs'].items():
        with torch.no_grad():
            b=subset(q,torch.arange(len(q['valid'])),a.device)
            base=base_forward(system,b['content'],b['valid'])
            q['residual']=torch.where(obs(b),b['motion']-base['b0'],0.).cpu()
    pairs=identity_pairs(data['refs'],data['fit_sids'])
    assert len(pairs)==22
    with torch.no_grad():
        for role,sids in (('train',data['fit_sids']),('validation',data['dev_sids'])):
            for sid in sids:
                baseline=encode_ref(system,data,sid,range(len(data['refs'][sid]['valid'])),a.device)['baseline'][0].cpu().numpy()
                cached=z[role+'__identity_baseline'][z[role+'__speaker']==sid][0]
                np.testing.assert_allclose(baseline,cached,rtol=1e-5,atol=1e-5)
    frozen={k:v.detach().cpu().clone() for k,v in system.state_dict().items()
            if not k.startswith(('identity_encoder.','identity_bias.'))}
    report=dict(schema='identity_reference_budget_v1',test_loaded=False,
        source_checkpoint_sha256=a.checkpoint_sha256,data_manifest_sha256=source['data_manifest_sha256'],
        script_sha256=sha(Path(__file__)),source_npz_sha256=sha(a.conditions/'per_clip_conditions.npz'),
        reference_training_performed=True,generator_training_performed=False,generation_performed=False,
        fitting='Only 22 TRAIN independent complementary enrollment pairs; no query targets',
        diagnostic='TRAIN/validation neutral query means are evaluation only, using cached frozen B0',
        seeds=[47,48,49],additional_epochs=[12,120],batch_size=2,lr=.0001,weight_decay=.00001,
        original_loss='symmetric cross-reference MSE/(2*.25^2) + .05 style_contrastive',
        unchanged='B0, teacher, audio, renderer, flow source, all support and losses',
        decision_gate='All seeds: TRAIN/development reference MSE improves and TRAIN/validation neutral-query all/mouth/brow/jaw means do not regress',
        warm=dict(reference=identity_report(system,data,a.device),query=query_shape(system,data,z,a.device)),runs={})
    a.output.mkdir(parents=True)
    for seed in (47,48,49):
        first12={}
        for epochs in (12,120):
            system.load_state_dict(warm,strict=True);system.requires_grad_(False);system.eval();system.zero_grad(set_to_none=True)
            system.identity_encoder.requires_grad_(True);system.identity_bias.requires_grad_(True)
            parameters=list(system.identity_encoder.parameters())+list(system.identity_bias.parameters())
            optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-5)
            gen=torch.Generator().manual_seed(seed);last_loss=None
            for epoch in range(epochs):
                order=torch.randperm(len(pairs),generator=gen)
                sums=[]
                for ix in order.split(2):
                    av=[];bv=[]
                    for i in ix:
                        sid,aa,bb=pairs[int(i)]
                        av.append(encode_ref(system,data,sid,aa,a.device));bv.append(encode_ref(system,data,sid,bb,a.device))
                    ac=torch.cat([v['code'] for v in av]);bc=torch.cat([v['code'] for v in bv])
                    common=torch.cat([v['observed_channels']&w['observed_channels'] for v,w in zip(av,bv)])
                    base_loss=(mse(torch.cat([v['baseline'] for v in av]),torch.cat([v['neutral_mean'] for v in bv]),common)+
                        mse(torch.cat([v['baseline'] for v in bv]),torch.cat([v['neutral_mean'] for v in av]),common))/(2*.25**2)
                    contrast=style_contrastive(ac,bc,torch.tensor([pairs[int(i)][0] for i in ix],device=a.device))
                    loss=base_loss+.05*contrast
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    gradient=torch.nn.utils.clip_grad_norm_(parameters,1.)
                    assert torch.isfinite(gradient)
                    optimizer.step();sums.append(float(loss.detach()))
                last_loss=float(np.mean(sums))
                if epoch+1==12:
                    ident={k:v.detach().cpu() for k,v in system.state_dict().items() if k.startswith(('identity_encoder.','identity_bias.'))}
                    h=state_hash(ident)
                    if epochs==12:first12[seed]=h
                    else:assert h==first12[seed], 'Shared first12 trajectory must be exact'
            for k,v in frozen.items():assert torch.equal(system.state_dict()[k].detach().cpu(),v), 'Frozen state drift: '+k
            system.requires_grad_(False)
            key=f'seed{seed}_epochs{epochs}'
            entry=dict(seed=seed,additional_epochs=epochs,steps=epochs*11,last_reference_loss=last_loss,
                frozen_state_exact=True,first12_trajectory_exact=True,reference=identity_report(system,data,a.device),
                query=query_shape(system,data,z,a.device))
            ident={k:v.detach().cpu() for k,v in system.state_dict().items() if k.startswith(('identity_encoder.','identity_bias.'))}
            checkpoint=a.output/(key+'.pt')
            torch.save(dict(identity_state=ident,source_checkpoint_sha256=a.checkpoint_sha256,
                seed=seed,additional_epochs=epochs,diagnostic_only=True),checkpoint)
            entry['identity_checkpoint_sha256']=sha(checkpoint)
            report['runs'][key]=entry
            (a.output/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print(json.dumps(dict(event='identity_budget_result',run=key,**entry)),flush=True)
    decisions=[]
    for seed in (47,48,49):
        c=report['runs'][f'seed{seed}_epochs12'];e=report['runs'][f'seed{seed}_epochs120']
        passed=all(e['reference'][role]['cross_reference_baseline_mse']<c['reference'][role]['cross_reference_baseline_mse']
                   for role in ('fit','development'))
        passed=passed and all(e['query'][role][k]<=c['query'][role][k]
                             for role in ('train','validation') for k in ('all_mean_mse','mouth_mean_mse','brows_mean_mse','jawOpen_mean_mse'))
        decisions.append(dict(seed=seed,passes=passed))
    report['seed_gates']=decisions;report['passes_all_seeds']=all(r['passes'] for r in decisions)
    report['decision']='Needs separate full downstream retraining and final generated evaluation before any promotion' if report['passes_all_seeds'] else 'Reject longer identity-only fit as next repair; no adaptation or promotion'
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(event='complete',passes_all_seeds=report['passes_all_seeds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','conditions','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--device',default='cuda')
    run(p.parse_args())
