"""Read-only matched-content/emotion/intensity personal-style direction.

Real donor/source performances are not frame aligned. Only clip statistics
are compared, with label matching and an explicit performance-variation limit.
The target is NOT a training label or a ground truth counterfactual sequence.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.factor_rebuild import V5FactorRebuild
from kinetalk_v5.affect_trend import V5AffectTrend
from kinetalk_v5.joint_factor_repair import V5JointFactorRepair
from scripts.pilot_v5_factor_rebuild import raw_audio,reference
from scripts.pilot_v5_style_output import timing_loss
from scripts.train_v7_pilot import load,batch,FixedProbe,sha
from scripts.train_expression_response import configure,load_runtime,write,state_digest


def descriptor(y, valid, scale):
    rows=[]
    for x,v in zip(y,valid):
        x=x[v]
        if not len(x):raise ValueError('style descriptor needs observed frames')
        rows.append(torch.stack((x.mean(0),x.quantile(.1,dim=0),x.quantile(.9,dim=0)))/scale.clamp_min(.04))
    return torch.stack(rows)


def select_pairs(records, data):
    records={r['row']:r for r in records}
    pairs=[]
    for i in range(len(data['motion'])):
        if int(data['dataset_id'][i])!=0:continue
        if int(data['emotion_id'][i])<0 or not bool(data['valid'][i].any()):continue
        if 'emotion_valid' in data and not bool(data['emotion_valid'][i]):continue
        for j in range(len(data['motion'])):
            if i==j or int(data['speaker_id'][i])==int(data['speaker_id'][j]):continue
            if int(data['dataset_id'][j])!=0 or not bool(data['valid'][j].any()):continue
            if 'emotion_valid' in data and not bool(data['emotion_valid'][j]):continue
            if records[i]['sentence_id']!=records[j]['sentence_id']:continue
            if int(data['emotion_id'][i])!=int(data['emotion_id'][j]):continue
            if not bool(data['intensity_valid'][i]&data['intensity_valid'][j]):continue
            if not torch.isclose(data['intensity'][i],data['intensity'][j]):continue
            pairs.append((i,j))
    return pairs


@torch.no_grad()
def run(a):
    configure(a.seed);started=time.time();device=torch.device(a.device)
    binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    data=raw_audio(load(Path(a.data)/'held_sentence.npz'),mean,std)
    records=[r for r in json.loads((Path(a.data)/'provenance.json').read_text())['records'] if r['role']=='held_sentence']
    pairs=select_pairs(records,data)
    if not pairs:raise ValueError('no genuinely label/content-matched person pairs')
    results={}
    checkpoints=json.loads(Path(a.checkpoints).read_text())
    for name,path in checkpoints.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        cls={'v5_affect_trend_v1':V5AffectTrend,'v5_joint_factor_repair_v1':V5JointFactorRepair}.get(ck.get('schema'),V5FactorRebuild)
        model=cls.from_checkpoint(ck).to(device).eval().requires_grad_(False)
        if cls is V5FactorRebuild:
            model.flow.sample_gain.zero_()
            model.style_response.basis[-1].weight.zero_();model.style_response.basis[-1].bias.zero_()
        before=state_digest(model)
        probes=[FixedProbe(binding,i).to(device) for i in range(len(binding['probes']))]
        rows=[]
        for start in range(0,len(pairs),8):
            selected=pairs[start:start+8];ids=[i for i,j in selected];donors=[j for i,j in selected]
            b,r=batch(data,ids,device);d,dr=batch(data,donors,device)
            p=model.affect_audio(b['audio_features'],b['valid'])
            s=model.encode_style(reference(r,0));sd=model.encode_style(reference(dr,0))
            y=model.decode(b,p,s,mode='modal')['motion'];yd=model.decode(b,p,sd,mode='modal')['motion']
            got=descriptor(yd,b['valid'],model.core.scales)-descriptor(y,b['valid'],model.core.scales)
            wanted=descriptor(d['motion'],d['valid'],model.core.scales)-descriptor(b['motion'],b['valid'],model.core.scales)
            mask=(b['channel_mask']&d['channel_mask'])[:,None].expand_as(wanted)
            predicted=[probe.classes(y,b['valid']) for probe in probes]
            transferred=[probe.classes(yd,b['valid']) for probe in probes]
            for k,(i,j) in enumerate(selected):
                x,g=got[k][mask[k]],wanted[k][mask[k]]
                if not len(x):raise ValueError('matched people have no shared observed channels')
                one={key:value[k:k+1] for key,value in b.items()}
                timing=float(timing_loss(model,yd[k:k+1],y[k:k+1],one))
                active=g.abs().gt(.05)
                rows.append(dict(source_row=i,donor_row=j,
                    unchanged_style_target_error=float(g.abs().mean()),
                    transferred_style_target_error=float((x-g).abs().mean()),
                    achieved_style_change=float(x.abs().mean()),target_style_change=float(g.abs().mean()),
                    cosine=float(torch.nn.functional.cosine_similarity(x[None],g[None]).item()),
                    active_direction_agreement=float((x[active].sign()==g[active].sign()).float().mean()) if active.any() else None,
                    geometric_timing_proxy=timing,
                    **{f'probe{ix}_class_preserved':int(transferred[ix][k]==predicted[ix][k]) for ix in range(len(probes))},
                    **{f'probe{ix}_source_correct':int(predicted[ix][k]==b['emotion_id'][k]) for ix in range(len(probes))},
                    **{f'probe{ix}_transfer_correct':int(transferred[ix][k]==b['emotion_id'][k]) for ix in range(len(probes))}))
        if state_digest(model)!=before:raise RuntimeError('read-only audit mutated weights')
        aggregate={key:float(np.mean([r[key] for r in rows if r[key] is not None])) for key in rows[0] if key not in ('source_row','donor_row')}
        aggregate['target_error_gain']=aggregate['unchanged_style_target_error']-aggregate['transferred_style_target_error']
        results[name]=dict(checkpoint_sha256=sha(path),summary=aggregate,per_pair=rows)
    write(a.output,dict(schema='v5_matched_personal_direction_v1',read_only=True,sealed_test_loaded=False,
        elapsed_seconds=time.time()-started,pair_count=len(pairs),
        source_clip_count=len(set(i for i,j in pairs)),person_count=len(set(int(data['speaker_id'][i]) for i,j in pairs)),
        matching=['different_person','same_sentence','same_emotion','same_known_ordinal_intensity'],
        results=results,limits=['not frame aligned, descriptive clip-statistic direction only',
           'one performance difference is not ground truth intrinsic personal style',
           'directed pairs repeat clips, not independent observations','no novelty/paper-readiness inference']))
    print(json.dumps({'completed':a.output,'pairs':len(pairs),'elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('checkpoints','binding','data','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,default=47);p.add_argument('--device',default='cuda');run(p.parse_args())
