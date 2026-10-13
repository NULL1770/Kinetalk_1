"""Reference changes: separate mouth scale from native temporal shape.

Uses a fixed shared-rig observer. Low-aperture rank agreement is NOT a
phoneme-labelled closure score, and size changes alone are NOT pure style.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.joint_factor_repair import V5JointFactorRepair
from kinetalk_v4.unified_model_v4 import affine_geometry
from scripts.pilot_v5_factor_rebuild import raw_audio,reference
from scripts.train_v7_pilot import load,batch,FixedProbe,sha
from scripts.train_expression_response import configure,load_runtime,write,state_digest
from scripts.diagnose_v7_factor_routes import cross_person_donors
from kinetalk_v2.unified_model import adjacency


def pattern_scores(own,swapped,valid,times):
    x,y=own[valid],swapped[valid]
    x0,y0=x-x.mean(),y-y.mean()
    denom=x0.norm()*y0.norm()
    correlation=float(x0@y0/denom) if float(denom)>1e-8 else None
    edge=adjacency(valid[None],times[None])[0]
    vx,vy=(own[1:]-own[:-1])[edge],(swapped[1:]-swapped[:-1])[edge]
    vv=vx.norm()*vy.norm()
    velocity_cosine=float(vx@vy/vv) if float(vv)>1e-8 else None
    # Each curve uses its own low quantile, so this measures rank/timing.
    # It cannot certify absolute lip contact after changing mouth offsets.
    lowx=x<=x.quantile(.2);lowy=y<=y.quantile(.2)
    low_f1=float(2*(lowx&lowy).sum()/(lowx.sum()+lowy.sum()).clamp_min(1))
    amp=float((y.quantile(.9)-y.quantile(.1))/(x.quantile(.9)-x.quantile(.1)).clamp_min(.05))
    return dict(aperture_pattern_correlation=correlation,aperture_velocity_cosine=velocity_cosine,
        low_aperture_rank_f1=low_f1,aperture_amplitude_ratio=amp)


@torch.no_grad()
def run(a):
    configure(47);start=time.time();device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    model=V5JointFactorRepair.from_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=False)).to(device).eval().requires_grad_(False)
    before=state_digest(model);binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    probes=[FixedProbe(binding,i).to(device) for i in range(len(binding['probes']))];result={}
    for role in ('held_sentence','unseen_speaker'):
        data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
        donors=cross_person_donors(data['speaker_id'],data['emotion_id']);codes=[]
        for starti in range(0,len(data['motion']),8):
            _,r=batch(data,list(range(starti,min(starti+8,len(data['motion'])))),device)
            codes.append(model.encode_style(reference(r,0)))
        codes=torch.cat(codes);rows=[];examples={}
        for starti in range(0,len(data['motion']),8):
            ids=list(range(starti,min(starti+8,len(data['motion']))));b,r=batch(data,ids,device)
            p=model.affect_audio(b['audio_features'],b['valid'])
            styles={'own_reference':codes[ids],'same_person_other_emotion_ref':model.encode_style(reference(r,1)),
                'different_person_ref':codes[donors[ids].to(device)]}
            ys={name:model.decode(b,p,s,mode='modal')['motion'] for name,s in styles.items()}
            geometries={name:affine_geometry(y,model.core.decoder.geometry_jacobian,model.core.decoder.geometry_offset) for name,y in ys.items()}
            classes={name:[probe.classes(y,b['valid']) for probe in probes] for name,y in ys.items()}
            for k,i in enumerate(ids):
                row=dict(row=i,emotion=int(b['emotion_id'][k]),donor=int(donors[i]),changes={})
                v=b['valid'][k];own=geometries['own_reference'][k]
                for mode in ('same_person_other_emotion_ref','different_person_ref'):
                    swapped=geometries[mode][k]
                    row['changes'][mode]=dict(**pattern_scores(own[:,2],swapped[:,2],v,b['times'][k]),
                        width_mean_change_mm=float((swapped[v,0]-own[v,0]).mean()),
                        width_abs_change_mm=float((swapped[v,0]-own[v,0]).abs().mean()),
                        aperture_mean_change_mm=float((swapped[v,2]-own[v,2]).mean()),
                        **{f'probe{ix}_class_preserved':int(classes[mode][ix][k]==classes['own_reference'][ix][k]) for ix in range(len(probes))})
                rows.append(row);e=int(b['emotion_id'][k])
                if e<0 or e in examples:continue
                examples[e]=i
                np.savez_compressed(out/f'{role}_emotion_{e}.npz',motions=torch.stack([b['motion'][k],*[ys[m][k] for m in styles]]).cpu().numpy(),
                    mode_names=np.array(['GT',*styles]),times=b['times'][k].cpu().numpy(),valid=v.cpu().numpy(),
                    channel_mask=b['channel_mask'][k].cpu().numpy(),row=np.array(i),emotion=np.array(e),donor=np.array(int(donors[i])))
        summary={}
        for mode in ('same_person_other_emotion_ref','different_person_ref'):
            rr=[r['changes'][mode] for r in rows]
            summary[mode]={key:float(np.mean([r[key] for r in rr if r[key] is not None])) for key in rr[0]}
        result[role]=summary;write(out/(role+'_per_clip.json'),rows)
    if state_digest(model)!=before:raise RuntimeError('read-only audit mutated weights')
    write(out/'report.json',dict(schema='v5_style_lip_audit_v1',elapsed_seconds=time.time()-start,
        checkpoint_sha256=sha(a.checkpoint),results=result,sealed_test_loaded=False,read_only=True,
        limits=['shared rig mouth scale, not subject-specific topology',
            'low aperture rank F1 is not a physiological or bilabial contact metric',
            'no supplied emotion labels in deployment; query labels only match evaluation donors',
            'high curve correlation plus amplitude change does not prove style disentanglement',
            'class preservation is relative to own-reference predictions, not necessarily true emotion']))
    print(json.dumps({'completed':str(out),'elapsed_seconds':time.time()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('checkpoint','binding','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda');run(p.parse_args())
