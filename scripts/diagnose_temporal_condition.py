"""Frozen u_a order ablation; global affect, content, identity and noise held fixed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_matched_motion_curves import REGIONS, RC, MC, region_values, means
from scripts.diagnose_flow_sampling import sha, clip_metrics
from scripts.packed_trainval_cache import load_packed
from scripts.phase1_condition_diagnostic import _identity_cache
from scripts.train_full_staged import audio_affect, base_forward, batch_identity
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features, classification_metrics

def temporal_variant(local,valid,mode,generator=None):
    if mode not in ('static','reverse','shuffle','zero'): raise ValueError(mode)
    out=torch.zeros_like(local)
    for j in range(len(local)):
        ix=valid[j].nonzero().flatten()
        if mode=='static': out[j,ix]=local[j,ix].mean(0)
        elif mode=='reverse': out[j,ix]=local[j,ix.flip(0)]
        elif mode=='shuffle':
            order=torch.randperm(len(ix),generator=generator).to(ix.device)
            out[j,ix]=local[j,ix[order]]
    return out

@torch.no_grad()
def run(a):
    if a.output.exists(): raise FileExistsError('Fresh diagnostic required')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device(a.device)
    assert sha(a.checkpoint)==a.checkpoint_sha256
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    provenance=json.loads((a.run_root/'provenance.json').read_text())
    assert ck['stage']=='audio' and ck['recipe_sha256']==provenance['recipe_sha256']
    recipe=provenance['recipe']
    assert not recipe['test_loaded'] and not recipe['args'].get('independent_probe_weight',0)
    data=load_packed(a.data,materialize=False,with_refs=True)
    assert set(data['splits'])=={'train','validation'} and not data['provenance']['test_loaded']
    manifest=data['provenance']['manifest_sha256']; assert manifest==ck['data_manifest_sha256']
    system=NeutralAffectSystem(ck['config']).to(device).eval()
    system.load_state_dict(ck['system'],strict=True); system.requires_grad_(False)
    audio=SlowStateAffect(ck['feature_stats']['mean'],ck['feature_stats']['std'],stride=recipe['args']['stride']).to(device).eval()
    audio.load_state_dict(ck['audio'],strict=True); audio.requires_grad_(False)
    identities=_identity_cache(system,data,device)
    probes=[];pm=[];support=None;names=data['config']['data']['emotion_classes']
    for path in a.probe:
        pc=torch.load(path,map_location='cpu',weights_only=False)
        assert pc['train_manifest_sha256']==manifest and pc['classes']==names
        assert not pc.get('test_used_for_selection') and not pc.get('generator_outputs_used_for_fitting')
        s=pc['channel_support'].bool()
        if support is not None: assert torch.equal(support,s)
        support=s
        p=MotionEmotionProbe(pc['feature_dim'],pc['hidden'],len(names)).eval()
        p.load_state_dict(pc['model'],strict=True)
        probes.append((p,pc.get('feature_mask',torch.ones(6*int(s.sum()),dtype=torch.bool))))
        pm.append({'sha256':sha(path),'kind':pc['kind']})
    q=data['splits']['validation']; assert q['channel_mask'][:,support].all()
    ids=torch.arange(len(q['_lengths']))
    if a.limit_per_class:
        selected=[]
        for k in range(len(names)):
            eligible=(q['emotion_id']==k).nonzero().flatten()
            selected.extend(eligible[torch.linspace(0,len(eligible)-1,min(len(eligible),a.limit_per_class)).long()].tolist())
        ids=torch.tensor(sorted(selected))
    noise_gen=torch.Generator().manual_seed(42); shuffle_gen=torch.Generator().manual_seed(20261005)
    modes=['audio','static','reverse','shuffle','zero']; stores={}
    for mode in ['GT',*modes]:
        stores[mode]={policy:{'metrics':[],'regions':[],'features':[],'delta_rms':[]} for policy in ('raw','clip_all')}
    labels=[];clipids=[];input_energy=[]
    for batch_no,ix in enumerate(ids.split(a.batch_size)):
        b=q.batch(ix,device);base=base_forward(system,b['content'],b['valid']);b.update(base)
        ident=batch_identity(identities,b);affect=audio_affect(audio,b['audio_features'],b['valid'])
        noise=torch.randn(b['motion'].shape,generator=noise_gen).to(device)
        predictions={'GT':b['motion']}
        for mode in modes:
            changed=affect if mode=='audio' else {**affect,'u_a':temporal_variant(affect['u_a'],b['valid'],mode,shuffle_gen)}
            predictions[mode]=system.generate(b['content'],b['valid'],ident,changed,initial_noise=noise,steps=12,base=base)['raw_motion']
        predictions={k:v.cpu() for k,v in predictions.items()}
        labels.extend(b['emotion_id'].cpu().tolist());clipids.extend(b['clip_id'])
        for j in range(len(ix)):
            v=b['valid'][j].cpu();ch=b['channel_mask'][j].cpu();t=b['motion'][j].cpu();tm=b['times'][j].cpu()
            local=affect['u_a'][j,b['valid'][j]].cpu()
            input_energy.append([float(local.square().mean()),float((local-local.mean(0)).square().mean())])
            for mode,preds in predictions.items():
              for policy in ('raw','clip_all'):
                p=preds[j] if policy=='raw' else preds[j].clamp(0,1)
                ref=predictions['audio'][j] if policy=='raw' else predictions['audio'][j].clamp(0,1)
                store=stores[mode][policy]
                store['metrics'].append(clip_metrics(p,t,v,ch,tm))
                store['regions'].append([region_values(p,t,v,ch,region) for region in REGIONS.values()])
                store['features'].append(motion_features(p[:,support],v).numpy())
                store['delta_rms'].append([float((p-ref)[v][:,[i for i in region if ch[i]]].square().mean().sqrt()) for region in REGIONS.values()])
        if batch_no%10==0: print(json.dumps({'event':'batch','clips':len(labels),'total':len(ids)}),flush=True)
    arrays={'labels':np.asarray(labels),'clip_id':np.asarray(clipids),'input_energy':np.asarray(input_energy)}
    report={'schema':'temporal_receiver_order_diagnostic_v1','test_loaded':False,'training_performed':False,
            'checkpoint_sha256':a.checkpoint_sha256,'data_manifest_sha256':manifest,'clips':len(labels),
            'classes':names,'probes':pm,'noise_seed':42,'shuffle_seed':20261005,'batch_size':a.batch_size,
            'steps':12,'sampler':'Euler','fixed':'global emotion/intensity/identity/content/h0/B0/noise unchanged',
            'changed':'only u_a; reverse/shuffle preserve each clip valid-frame distribution',
            'diagnostic_policy':'manipulated controls are diagnostics; no inference change selected',
            'input_energy_columns':['total_mean_square','within_clip_mean_square'],
            'input_energy_mean':np.asarray(input_energy).mean(0).tolist(),'region_columns':RC,'results':{}}
    yt=torch.tensor(labels)
    for mode in stores:
        report['results'][mode]={}
        for policy,store in stores[mode].items():
            arr={k:np.asarray(v) for k,v in store.items()}
            arrays.update({mode+'__'+policy+'__'+k:v for k,v in arr.items()})
            f=torch.from_numpy(arr['features'])
            report['results'][mode][policy]={'metrics':means(arr['metrics'],MC),
                'regions':{name:means(arr['regions'][:,j],RC) for j,name in enumerate(REGIONS)},
                'delta_rms_from_audio':dict(zip(REGIONS,arr['delta_rms'].mean(0).tolist())),
                'probes':[classification_metrics(yt,p(f[:,fm]).argmax(-1),names) for p,fm in probes]}
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'per_clip.npz',**arrays)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',type=Path,nargs='+',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--limit-per-class',type=int,default=0)
    run(p.parse_args())
