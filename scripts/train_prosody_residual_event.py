"""Train a static event prior and a local-prosody residual on the fixed split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from kinetalk_b0.models.prosody_residual_schedule import StaticScheduleHead, ProsodyResidualSchedule
from scripts.audio_event_condition import derive_event_condition
from scripts.event_schedule_teacher import fit_teacher, extract_schedule, valid_runs
from scripts.probe_motion_condition_predictability import split_train_pool, sha
from scripts import train_continuous_motion_latent as common
from scripts.train_event_schedule_pipeline import target_arrays, process_nll, paired_delta

SCHEMA = 'prosody_residual_event_v1'; SEEDS = (42, 123, 2026)
def arr(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def prepare(clip, labels, stats):
    return {'clip_id':clip['clip_id'],'sentence':clip['sentence'],
      'condition':derive_event_condition(clip['features'].float(), clip['valid']).float(),
      'context':((clip['context']-stats['context_mean'])/stats['context_scale']).float(),
      'valid':clip['valid'].bool(), **dict(zip(('onset','risk','duration'),map(torch.from_numpy,target_arrays(labels[clip['clip_id']]))))}

def batch(rows, rng, device, frames=180, size=12):
    x=torch.zeros(size,frames,10,device=device); c=[]; v=torch.zeros(size,frames,dtype=torch.bool,device=device)
    y=torch.zeros(size,frames,4,device=device); r=torch.zeros_like(y,dtype=torch.bool); d=torch.zeros_like(y,dtype=torch.long)
    for i in range(size):
        row=rows[int(rng.integers(len(rows)))]; runs=valid_runs(arr(row['valid'])); left,right=runs[int(rng.integers(len(runs)))]
        a=int(rng.integers(left,right-frames+1)) if right-left>frames else left; n=min(frames,right-a)
        x[i,:n]=row['condition'][a:a+n].to(device); c.append(row['context']); v[i,:n]=True; y[i,:n]=row['onset'][a:a+n].to(device); r[i,:n]=row['risk'][a:a+n].to(device); d[i,:n]=row['duration'][a:a+n].to(device)
        if a>left:r[i,:min(15,n)]=False
        if a+n<right:r[i,max(0,n-15):n]=False
    return x,torch.stack(c).to(device),v,y,r,d

def train_static(rows, seed, steps, device):
    torch.manual_seed(seed); model=StaticScheduleHead(rows[0]['context'].numel()).to(device); opt=torch.optim.AdamW(model.parameters(),3e-4,weight_decay=.01); rng=np.random.default_rng(seed)
    for _ in range(steps):
        _,c,v,y,r,d=batch(rows,rng,device); loss=process_nll(model(c,v.shape[1]),y,r,d); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

def train_residual(base, rows, seed, steps, device):
    torch.manual_seed(seed); model=ProsodyResidualSchedule(base).to(device).freeze_base(); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],3e-4,weight_decay=.01); rng=np.random.default_rng(seed+10000)
    for _ in range(steps):
        x,c,v,y,r,d=batch(rows,rng,device); loss=process_nll(model(x,c,v),y,r,d); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

@torch.no_grad()
def evaluate(model, rows, device, residual):
    result=[]
    for row in rows:
        v=row['valid'][None].to(device); c=row['context'][None].to(device)
        out=model(row['condition'][None].to(device),c,v) if residual else model(c,len(row['condition']))
        p=out['onset_logits'][0].sigmoid().cpu().numpy(); lp=out['duration_logits'][0].log_softmax(-1).cpu().numpy(); onset,risk,dur=map(arr,(row['onset'],row['risk'],row['duration']))
        q=np.clip(p[risk],1e-7,1-1e-7); yy=onset[risk]; nll=float(-(yy*np.log(q)+(1-yy)*np.log1p(-q)).mean()); f,g=np.where((onset>0)&risk); dn=-lp[f,g,dur[f,g]]
        result.append({'clip_id':row['clip_id'],'sentence':row['sentence'],'brier':float(np.square(q-yy).mean()),'joint_nll':nll+float(dn.sum())/int(risk.sum()),'duration_nll':float(dn.mean()) if len(dn) else None})
    return result

def mean(rows):
    return {k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ('brier','joint_nll','duration_nll')}

def main(a):
    if a.output.exists(): raise FileExistsError(a.output)
    ref=json.loads((a.source_run/'protocol.json').read_text());
    if sha(a.dataset)!=ref['dataset_sha256']: raise ValueError('dataset mismatch')
    data=torch.load(a.dataset,weights_only=False,map_location='cpu',mmap=True); fit,valid=split_train_pool(data['clips'],ref); del data
    stats=common._load(a.source_run/'fit_stats.pt'); teacher=fit_teacher(fit); labels={c['clip_id']:extract_schedule(c['motion9'],c['valid'],teacher,motion_mask=c['motion_mask']) for c in fit+valid}; train=[prepare(c,labels,stats) for c in fit]; val=[prepare(c,labels,stats) for c in valid]
    a.output.mkdir(parents=True); common._write(a.output/'protocol.json',{'schema':SCHEMA,'dataset_sha256':ref['dataset_sha256'],'source_protocol_sha256':sha(a.source_run/'protocol.json'),'steps':a.steps,'seeds':list(SEEDS),'condition':'run-safe local prosody residual over frozen static context prior','default_replaced':False})
    reports={}
    for seed in (SEEDS[:1] if a.smoke else SEEDS):
        base=train_static(train,seed,a.steps,a.device); residual=train_residual(base,train,seed,a.steps,a.device); s=evaluate(base,val,a.device,False); r=evaluate(residual,val,a.device,True)
        reports[str(seed)]={'static':mean(s),'prosody_residual':mean(r),'paired_brier':paired_delta(r,s,'brier'),'paired_joint_nll':paired_delta(r,s,'joint_nll'),'paired_duration_nll':paired_delta(r,s,'duration_nll')}
        common._save(a.output/f'seed_{seed}.pt',{'static':base.state_dict(),'residual':residual.state_dict(),'steps':a.steps}); common._write(a.output/'results.json',reports)
    passed=all(v['paired_brier']['ci95'][1]<0 and v['paired_joint_nll']['ci95'][1]<0 for v in reports.values()); common._write(a.output/'decision.json',{'prosody_predictability_passed':passed,'default_replaced':False}); common._write(a.output/'status.json',{'schema':SCHEMA,'state':'complete','prosody_predictability_passed':passed,'default_replaced':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--source-run',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--steps',type=int,default=1500); p.add_argument('--smoke',action='store_true'); main(p.parse_args())
