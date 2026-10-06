"""Read-only full-validation B0/final jaw decomposition from saved curves."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def mean(x,valid):
    return torch.where(valid,x,0.).sum(1)/valid.sum(1)


def centered(x,valid):
    return torch.where(valid,x-mean(x,valid)[:,None],0.)


def corr(x,y,valid):
    a,b=centered(x,valid),centered(y,valid)
    d=(a.square().sum(1)*b.square().sum(1)).sqrt()
    return torch.where(d>1e-12,(a*b).sum(1)/d,torch.nan)


def jaw_scores(x,target,base,valid):
    adjacent=valid[:,1:]&valid[:,:-1]
    error=x.diff(dim=1)-target.diff(dim=1)
    be=base.diff(dim=1)-target.diff(dim=1)
    residual=x.diff(dim=1)-base.diff(dim=1)
    e=mean(error.square(),adjacent);bm=mean(be.square(),adjacent)
    rm=mean(residual.square(),adjacent);cross=2*mean(be*residual,adjacent)
    torch.testing.assert_close(e,bm+rm+cross,rtol=1e-12,atol=1e-12)
    columns=['correlation','pred_centered_rms','gt_centered_rms','mse','mean_bias_mse',
             'displacement_mse','base_displacement_mse','residual_displacement_energy','cross_term','range_q90_q10','gt_range_q90_q10']
    ranges=lambda v:torch.stack([torch.quantile(v[i,valid[i]],.9)-torch.quantile(v[i,valid[i]],.1) for i in range(len(v))])
    return columns,torch.stack([corr(x,target,valid),mean(centered(x,valid).square(),valid).sqrt(),
        mean(centered(target,valid).square(),valid).sqrt(),mean((x-target).square(),valid),
        (mean(x,valid)-mean(target,valid)).square(),e,bm,rm,cross,ranges(x),ranges(target)],dim=1).numpy()


@torch.no_grad()
def run(binding_path,output):
    if output.exists():raise FileExistsError('Fresh report required')
    torch.set_num_threads(2)
    binding=json.loads(binding_path.read_text())
    data=Path(binding['data'])
    labels={k:np.load(data/'validation'/f'{k}.npy',allow_pickle=False) for k in ('emotion_id','speaker_id','intensity_id')}
    for k in labels:
        name=f'validation/{k}.npy'
        assert sha(data/name)==binding['validation_array_sha256'][name]
    curves={}
    for model,spec in binding['models'].items():
        path=Path(spec['checkpoint_root'])/'audio/curves.pt'
        assert sha(path)==spec['files']['audio/curves.pt']
        curves[model]=torch.load(path,map_location='cpu',weights_only=False)
    a,b=curves.values()
    assert a['clip_id']==b['clip_id'] and len(a['clip_id'])==1367
    for k in ('target','valid','times','channel_mask'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    report=dict(schema='phase31_frozen_full_jaw_v1',clips=1367,test_loaded=False,training_performed=False,
        forward_performed=False,binding_sha256=sha(binding_path),script_sha256=sha(__file__),
        scope='Saved immutable validation curves only; equal clip means,3draw final. Emotional native GT is not the neutral B0 target, so nonneutral B0 errors are descriptive only.',models={})
    for model,z in curves.items():
        target=z['target'][:,:,17].double();base=z['b0'][:,:,17].double()
        valid=z['valid']&z['channel_mask'][:,None,17]
        results={}
        for policy in ('raw','clip_all'):
            bx=base if policy=='raw' else base.clamp(0,1)
            columns,bscore=jaw_scores(bx,target,bx,valid)
            finals=[]
            for draw in (42,123,2026):
                value=z['predictions'][f'{draw}/full'][:,:,17].double()
                if policy=='clip_all':value=value.clamp(0,1)
                _,score=jaw_scores(value,target,base,valid);finals.append(score)
            final=np.stack(finals).mean(0)
            for emotion in [None]+list(range(8)):
                mask=np.ones(1367,dtype=bool) if emotion is None else labels['emotion_id']==emotion
                def summary(x):
                    selected=x[mask]
                    return {name:float(selected[np.isfinite(selected[:,i]),i].mean()) if np.isfinite(selected[:,i]).any() else None
                            for i,name in enumerate(columns)}
                results[f'{policy}/'+('all' if emotion is None else str(emotion))]=dict(clips=int(mask.sum()),b0=summary(bscore),final=summary(final))
        report['models'][model]=dict(curves_sha256=binding['models'][model]['files']['audio/curves.pt'],results=results)
    for model,spec in binding['models'].items():assert sha(Path(spec['checkpoint_root'])/'audio/curves.pt')==spec['files']['audio/curves.pt']
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(report_sha256=sha(output),neutral={m:r['results']['clip_all/0'] for m,r in report['models'].items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.binding,a.output)
