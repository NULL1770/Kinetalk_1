"""Decompose saved validation draw error; three-draw means are diagnostic only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_matched_motion_curves import REGIONS, region_values, means
from scripts.diagnose_flow_sampling import sha

COLUMNS=['draw_mse','mean_draw_mse','draw_variance','draw_variance_fraction',
         'draw_velocity_mse','mean_draw_velocity_mse','draw_velocity_variance',
         'velocity_variance_fraction','draw_temporal_rms','mean_draw_temporal_rms',
         'gt_temporal_rms','b0_temporal_rms','mean_draw_correlation','b0_correlation',
         'mean_draw_velocity_correlation','b0_velocity_correlation',
         'draw_offset_variance','draw_centered_variance','gt_speed_rms',
         'draw_speed_rms','mean_draw_speed_rms','b0_speed_rms']

def correlation(x,y):
    x=x-x.mean(0); y=y-y.mean(0)
    d=x.square().sum().sqrt()*y.square().sum().sqrt()
    return float((x*y).sum()/d) if d>1e-12 else float('nan')

def decompose(draws,target):
    """Exact finite-draw identity E||X-Y||² = ||mean(X)-Y||² + variance(X)."""
    mean=draws.mean(0)
    error=(draws-target).square().mean()
    bias=(mean-target).square().mean()
    variance=(draws-mean).square().mean()
    torch.testing.assert_close(error,bias+variance,rtol=2e-5,atol=1e-8)
    return float(error),float(bias),float(variance),float(variance/error) if error>1e-12 else 0.

def values(p,t,b,v,ch,ix):
    ix=[j for j in ix if ch[j]]
    x,y,base=p[:,v][:,:,ix],t[v][:,ix],b[v][:,ix]
    mean=x.mean(0); adj=v[1:] & v[:-1]
    dx=(p[:,1:]-p[:,:-1])[:,adj][:,:,ix]
    dy=(t[1:]-t[:-1])[adj][:,ix]; db=(b[1:]-b[:-1])[adj][:,ix]
    def rms(z): return float(z.square().mean().sqrt())
    dynamic=x-x.mean(1,keepdim=True)
    offset=x.mean(1)-x.mean(1).mean(0)
    centered_variance=(dynamic-dynamic.mean(0)).square().mean()
    return [*decompose(x,y),*decompose(dx,dy),rms(dynamic),
            rms(mean-mean.mean(0)),rms(y-y.mean(0)),rms(base-base.mean(0)),
            correlation(mean,y),correlation(base,y),correlation(dx.mean(0),dy),
            correlation(db,dy),float(offset.square().mean()),float(centered_variance),
            rms(dy),rms(dx),rms(dx.mean(0)),rms(db)]

@torch.no_grad()
def run(a):
    if a.output.exists(): raise FileExistsError('Fresh diagnostic required')
    torch.set_num_threads(2)
    r=json.loads((a.run_root/'provenance.json').read_text())
    recipe=r['recipe']
    assert not recipe['test_loaded']
    # The frozen timing000 recipe predates optional external-probe training.
    probe=recipe.get('independent_probe')
    if probe is None:
        assert 'independent_probe' not in recipe['args'] and 'independent_probe_weight' not in recipe['args']
    else:
        assert not probe['enabled'] and not recipe['args'].get('independent_probe')
    complete=json.loads((a.run_root/'audio/complete.json').read_text())
    path=a.run_root/'audio/curves.pt'
    assert sha(path)==complete['curves_sha256']
    ck=torch.load(a.run_root/'audio/final.pt',map_location='cpu',weights_only=False)
    assert ck['recipe_sha256']==r['recipe_sha256']
    assert ck['data_manifest_sha256']==recipe['data_provenance']['manifest_sha256']
    checkpoint_sha=sha(a.run_root/'audio/final.pt')
    assert checkpoint_sha==complete['final_sha256']
    del ck
    z=torch.load(path,map_location='cpu',weights_only=False)
    assert set(z['predictions'])=={'42/full','123/full','2026/full'}
    report={'schema':'finite_draw_noise_attribution_v1','test_loaded':False,'training_performed':False,
            'checkpoint_sha256':checkpoint_sha,'curves_sha256':sha(path),
            'data_manifest_sha256':recipe['data_provenance']['manifest_sha256'],
            'clips':len(z['clip_id']),'noise_seeds':z['noise_seeds'],'columns':COLUMNS,'regions':REGIONS,
            'scope':'exact decomposition for three saved draws, not population conditional variance',
            'mean_draw_policy':'finite-three mean diagnostic only; not chosen inference or replacement benchmark',
            'policies':'raw or clip each draw before decomposition; no lag/gain/threshold fitting','results':{}}
    arrays={'clip_id':np.asarray(z['clip_id'])}
    for policy in ('raw','clip_all'):
        pred=torch.stack(list(z['predictions'].values()))
        if policy=='clip_all': pred=pred.clamp(0,1)
        out=[]
        for i in range(len(z['clip_id'])):
            out.append([values(pred[:,i],z['target'][i],z['b0'][i],z['valid'][i],z['channel_mask'][i],ix)
                        for ix in REGIONS.values()])
        arr=np.asarray(out); arrays[policy]=arr
        report['results'][policy]={name:means(arr[:,j],COLUMNS) for j,name in enumerate(REGIONS)}
        print(json.dumps({'event':'policy_complete','policy':policy}),flush=True)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'per_clip_attribution.npz',**arrays)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
