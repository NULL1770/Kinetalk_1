"""Separate automatic deployment from explicitly supplied emotion control.

Held labels are used only to diagnose a declared control path. They never
enter fitted weights or the automatic path. Per-emotion native temporal
statistics and multi-seed upper-face diversity do not prove psychological
framewise emotion truth.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.joint_factor_repair import V5JointFactorRepair
from kinetalk_v4.unified_model_v4 import affine_geometry
from kinetalk_v2.unified_model import adjacency
from kinetalk_b0.models.expression_response import masked_pool
from scripts.pilot_v5_factor_rebuild import raw_audio,reference
from scripts.train_v7_pilot import load,batch,FixedProbe,sha
from scripts.train_expression_response import configure,load_runtime,write,state_digest
from kinetalk_b0.protocol import EMOTIONS


def native_statistics(y, valid, times, channels):
    use=valid[:,None]&channels[None]
    edge=adjacency(valid[None],times[None])[0]
    d=y[1:]-y[:-1]
    triple=edge[1:]&edge[:-1]
    if not use.any() or not edge.any() or not triple.any():raise ValueError('observed native run needed')
    x=y[valid][:,channels]
    return dict(mean=float(x.mean()),amplitude=float((x.quantile(.9,dim=0)-x.quantile(.1,dim=0)).mean()),
        speed=float(d[edge][:,channels].abs().mean()),acceleration=float((d[1:]-d[:-1])[triple][:,channels].abs().mean()))


def per_clip(y,b,j,model):
    v=b['valid'][j];cm=b['channel_mask'][j];times=b['times'][j]
    upper=cm.clone();upper[14:40]=False
    mouth=cm.clone();mouth[:14]=False;mouth[40:]=False
    obs=v[:,None]&cm[None]
    out=dict(full_mae=float((y-b['motion'][j]).abs()[obs].mean()),
        mouth_mae=float((y-b['motion'][j]).abs()[:,14:40][v[:,None]&cm[None,14:40]].mean()),
        upper=native_statistics(y,v,times,upper),mouth=native_statistics(y,v,times,mouth),
        gt_upper=native_statistics(b['motion'][j],v,times,upper),
        gt_mouth=native_statistics(b['motion'][j],v,times,mouth))
    g=affine_geometry(y,model.core.decoder.geometry_jacobian,model.core.decoder.geometry_offset)
    truth=affine_geometry(b['motion'][j],model.core.decoder.geometry_jacobian,model.core.decoder.geometry_offset)
    out['geometry_mae']=float((g-truth).abs()[v].mean())
    return out


@torch.no_grad()
def run(a):
    configure(47);started=time.time();device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    model=V5JointFactorRepair.from_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=False)).to(device).eval().requires_grad_(False)
    before=state_digest(model)
    if float(model.flow.sample_gain)!=0:raise ValueError('rejected new flow must be off')
    binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    probes=[FixedProbe(binding,i).to(device) for i in range(len(binding['probes']))]
    modes=['automatic','audio_temperature0.5','audio_argmax','declared_emotion','declared_emotion_level1','declared_emotion_level1.25','declared_emotion_level1.5']
    results={}
    for role in ('held_sentence','unseen_speaker'):
        data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std);records={m:[] for m in modes};examples={}
        for start in range(0,len(data['motion']),8):
            ids=list(range(start,min(start+8,len(data['motion']))));b,r=batch(data,ids,device)
            p=model.affect_audio(b['audio_features'],b['valid']);s=model.encode_style(reference(r,0))
            declared=F.one_hot(b['emotion_id'].clamp_min(0),model.cfg.classes).float()
            outputs={};sampled={}
            for mode in modes:
                emotion=None if mode=='automatic' else declared
                if mode=='audio_temperature0.5':emotion=(p['emotion_logits']/.5).softmax(-1)
                if mode=='audio_argmax':emotion=F.one_hot(p['emotion_logits'].argmax(-1),model.cfg.classes).float()
                strength=None if 'level' not in mode else p['state'].new_full((len(ids),),float(mode.split('level')[1]))
                y=model.decode(b,p,s,mode='modal',emotion=emotion,intensity=strength)['motion'];outputs[mode]=y
                if mode in ('automatic','declared_emotion'):
                    draws=[model.decode(b,p,s,seed=47+start+k*91,mode='sample',emotion=emotion,intensity=strength)['motion'] for k in range(4)]
                    sampled[mode]=draws
                classes=[probe.classes(y,b['valid']) for probe in probes]
                for k,i in enumerate(ids):
                    row=dict(row=i,emotion=int(b['emotion_id'][k]),person=int(b['speaker_id'][k]),
                        **per_clip(y[k],b,k,model),audio_emotion_correct=int(p['emotion_logits'][k].argmax()==b['emotion_id'][k]),
                        **{f'probe{ix}_correct':int(c[k]==b['emotion_id'][k]) for ix,c in enumerate(classes)})
                    if mode in sampled:
                        v=b['valid'][k];cm=b['channel_mask'][k];upper=cm.clone();upper[14:40]=False
                        changes=[(sampled[mode][ix][k]-sampled[mode][0][k]).abs() for ix in range(1,4)]
                        row['upper_seed_diversity']=float(torch.stack([d[v][:,upper].mean() for d in changes]).mean())
                        row['mouth_seed_diversity']=float(torch.stack([d[v,14:40][:,cm[14:40]].mean() for d in changes]).mean())
                        row['sample_upper']=native_statistics(sampled[mode][0][k],v,b['times'][k],upper)
                    records[mode].append(row)
            for k,i in enumerate(ids):
                e=int(b['emotion_id'][k])
                if e<0 or e in examples:continue
                examples[e]=i
                names=['GT',*modes,'automatic_sample','declared_emotion_sample']
                motions=[b['motion'][k],*[outputs[m][k] for m in modes],sampled['automatic'][0][k],sampled['declared_emotion'][0][k]]
                np.savez_compressed(out/f'{role}_emotion_{e}.npz',motions=torch.stack(motions).cpu().numpy(),mode_names=np.array(names),
                    times=b['times'][k].cpu().numpy(),valid=b['valid'][k].cpu().numpy(),channel_mask=b['channel_mask'][k].cpu().numpy(),
                    row=np.array(i),emotion=np.array(e))
        results[role]={}
        for mode,rows in records.items():
            summary={key:float(np.mean([r[key] for r in rows])) for key in ('full_mae','mouth_mae','geometry_mae','audio_emotion_correct',*[f'probe{i}_correct' for i in range(len(probes))])}
            summary['per_emotion']={EMOTIONS[e]:dict(n=len(rr),mouth_mae=float(np.mean([x['mouth_mae'] for x in rr])),
                **{f'probe{i}_correct':float(np.mean([x[f'probe{i}_correct'] for x in rr])) for i in range(len(probes))})
                for e in range(8) if (rr:=[r for r in rows if r['emotion']==e])}
            if 'upper_seed_diversity' in rows[0]:
                summary['upper_seed_diversity']=float(np.mean([r['upper_seed_diversity'] for r in rows]))
                summary['mouth_seed_diversity']=float(np.mean([r['mouth_seed_diversity'] for r in rows]))
                summary['upper_dynamics']={key:dict(gt=float(np.mean([r['gt_upper'][key] for r in rows])),
                    mean=float(np.mean([r['upper'][key] for r in rows])),sample=float(np.mean([r['sample_upper'][key] for r in rows])))
                    for key in ('amplitude','speed','acceleration')}
            results[role][mode]=summary
            write(out/f'{role}_{mode}_per_clip.json',rows)
    if before!=state_digest(model):raise RuntimeError('read-only audit changed weights')
    write(out/'report.json',dict(schema='v5_semantic_control_audit_v1',elapsed_seconds=time.time()-started,
        checkpoint_sha256=sha(a.checkpoint),read_only=True,sealed_test_loaded=False,
        fitted_from_held_labels=False,modes=modes,results=results,new_state_flow_enabled=False,
        limits=['automatic/audio_temperature/audio_argmax use no emotion label; latter two are declared inference calibrations',
            'declared emotion uses held labels to represent an explicitly supplied user condition, not automatic inference',
            'intensity levels are fixed controls, not GT fitted values',
            'upper-face dynamics includes eyes and cheeks, not a pure framewise emotion label',
            'seed diversity does not alone show natural plausible emotion dynamics']))
    print(json.dumps({'completed':str(out),'elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('checkpoint','binding','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');run(p.parse_args())
