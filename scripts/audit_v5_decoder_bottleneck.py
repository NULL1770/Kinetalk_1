"""Frozen state/geometry interventions under one realized deployment schedule.

All *_oracle outputs use scoring motion or labels and are diagnostic only.
Deployable generation runs before query motion is loaded. Keeping the schedule,
content, reference and audio label heads fixed isolates decoder response from a
changed stochastic event history. This is not a trained ablation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v4.unified_model_v4 import KineTalkV4Config,KineTalkV4Model,affine_geometry
from kinetalk_v4.geometry_support import clean_geometry_jacobian,geometry_support
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features,classification_metrics
from scripts.train_expression_response import configure,load_runtime,state_digest,sha,write
from scripts.train_unified_v4 import select_balanced,build_base_cache,attach_base,motion_targets
from scripts.evaluate_expression_response import metrics,aggregate,finite


@torch.no_grad()
def run(a):
    configure(47); device=torch.device(a.device); out=Path(a.output)
    if out.exists(): raise FileExistsError('Fresh output required')
    out.mkdir(parents=True); binding=json.loads(Path(a.binding).read_text())
    data,base=load_runtime(binding,device)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); s=ck['model']
    model=KineTalkV4Model(KineTalkV4Config(**ck['config']),s['feature_mean'],s['feature_std'],
        s['scales'],s['decoder.geometry_jacobian'],s['decoder.geometry_offset'],s['decoder.geometry_scale']).to(device)
    model.load_state_dict(s,strict=True); model.eval().requires_grad_(False)
    frozen=(state_digest(model),state_digest(base.stage1)); q=data['splits']['validation']
    ids=select_balanced(q,list(range(len(q['_lengths']))),a.limit,47)
    display=set(binding['display_clips'].values())
    ids=sorted(set(ids)|{i for i,cid in enumerate(q['clip_id']) if cid in display})
    write(out/'selection.json',dict(indices=ids,test_loaded=False,seed=47))
    cache=build_base_cache(q,base,ids,device)
    if sha(binding['rig'])!=binding['rig_sha256']: raise ValueError('Rig changed')
    with np.load(binding['rig'],allow_pickle=False) as z: rig={k:z[k].copy() for k in z.files}
    names=data['config']['data']['emotion_classes']; label_ids=[]; scored={}; features={}; per_clip={}
    (out/'render_inputs').mkdir()
    support=torch.tensor([True]*51+[False]); J=clean_geometry_jacobian(model.decoder.geometry_jacobian)
    for start in range(0,len(ids),8):
        use=ids[start:start+8]
        gen=attach_base(q.batch(torch.tensor(use),device,
            keys=('audio_features','content','valid','channel_mask','times')),use,cache)
        if 'motion' in gen: raise ValueError('Query motion reached deployment input')
        enrolled=[data['refs'][int(q['speaker_id'][i])] for i in use]
        if any(q['clip_id'][i] in r['clip_id'] for i,r in zip(use,enrolled)): raise ValueError('Reference leakage')
        refs={k:torch.stack([r[k][0] for r in enrolled]).to(device) for k in ('motion','valid','channel_mask')}
        normal=model.predict(gen['audio_features'],gen['h0'],gen['b0'],gen['valid'],refs,
            times=gen['times'],seed=47+start)
        # Only after the deployable output is generated, load privileged scoring
        # targets and compute explicitly labelled decoder counterfactuals.
        score=attach_base(q.batch(torch.tensor(use),device),use,cache)
        target=motion_targets(score); teacher=model.motion_teacher(score)
        cond=normal['conditioned']; events=normal['schedule']['events']
        label_ids.extend(score['emotion_id'].cpu().tolist())
        probs=F.one_hot(score['emotion_id'],model.cfg.classes).float()
        levels=torch.linspace(.5,1.5,model.cfg.intensities,device=device)
        strength=torch.where(score['intensity_valid'],levels[score['intensity_id'].clamp(0,model.cfg.intensities-1)],cond['intensity'])
        quality=target['quality'].bool()&score['valid'][...,None]
        count=quality.sum(1).clamp_min(1)
        pooled=torch.where(quality,target['state'],0.).sum(1)/count
        anchor_frame=torch.where(quality,target['state'],cond['state'])
        anchor_mean=torch.where(quality,pooled[:,None],cond['state'])
        state_conditions={
            'label_oracle_fixed_schedule':model.condition(normal['student'],gen['valid'],emotion=probs,intensity=strength),
            'teacher_raw_oracle_fixed_schedule':dict(cond,state=teacher['state']),
            'teacher_transformed_oracle_fixed_schedule':model.condition(teacher,gen['valid'],emotion=cond['emotion'],intensity=cond['intensity']),
            'anchor_frame_oracle_fixed_schedule':dict(cond,state=anchor_frame),
            'anchor_mean_oracle_fixed_schedule':dict(cond,state=anchor_mean),
        }
        predictions={'audio_sample':normal['motion']}
        for name,c in state_conditions.items():
            predictions[name]=model.decode(gen['h0'],gen['b0'],gen['valid'],c,normal['style'],events,
                geometry_audio=normal['geometry_audio_context'])['motion']
        predictions['audio_zero_events']=model.decode(gen['h0'],gen['b0'],gen['valid'],cond,normal['style'],
            torch.zeros_like(events),geometry_audio=normal['geometry_audio_context'])['motion']
        mask=geometry_support(J,score['channel_mask'])[:,None].expand(-1,score['motion'].shape[1],-1)
        truth=affine_geometry(torch.where(score['channel_mask'][:,None],score['motion'],0.),J,model.decoder.geometry_offset)
        oracle_geometry=torch.where(mask,truth,normal['geometry_target'])
        predictions['geometry_oracle_fixed_schedule']=model.decoder._project(normal['proposal'],oracle_geometry,gen['valid'])
        for j,i in enumerate(use):
            for name,pred in predictions.items():
                motion=pred[j].clamp(0,1); row=metrics(motion,score,j,rig)
                geom=affine_geometry(motion,J,model.decoder.geometry_offset)
                for axis,k in enumerate(('width','lift','aperture','depth')):
                    m=score['valid'][j]&mask[j,:,axis]
                    row['geometry/'+k+'_MAE_mm']=float((geom[m,axis]-truth[j,m,axis]).abs().mean()) if m.any() else None
                    row['geometry/'+k+'_bias_mm']=float((geom[m,axis]-truth[j,m,axis]).mean()) if m.any() else None
                scored.setdefault(name,[]).append(row)
                features.setdefault(name,[]).append(motion_features(motion.cpu()[:,support],score['valid'][j].cpu()))
                per_clip.setdefault(name,[]).append(dict(clip_id=q['clip_id'][i],emotion=names[int(q['emotion_id'][i])],**row))
            if q['clip_id'][i] in display:
                length=int(q['_lengths'][i]); mode_names=['GT','B0']+list(predictions)
                np.savez_compressed(out/'render_inputs'/(q['clip_id'][i]+'.npz'),
                    motions=torch.stack([score['motion'][j],gen['b0'][j]]+[p[j] for p in predictions.values()]).cpu().numpy()[:,:length],
                    mode_names=np.asarray(mode_names),times=gen['times'][j,:length].cpu().numpy(),valid=gen['valid'][j,:length].cpu().numpy(),
                    channel_mask=score['channel_mask'][j].cpu().numpy(),channels=rig['channels'],clip_id=q['clip_id'][i],noise_seed=47+start)
        write(out/'state.json',dict(status='scoring',clips=start+len(use),total=len(ids)))
    probes=[]
    for spec in binding['probes']:
        if sha(spec['path'])!=spec['sha256']: raise ValueError('Probe changed')
        saved=torch.load(spec['path'],map_location='cpu',weights_only=False)
        net=MotionEmotionProbe(saved['feature_dim'],saved['hidden'],len(names)).eval();net.load_state_dict(saved['model'])
        probes.append((net,saved.get('feature_mask',torch.ones(saved['feature_dim'],dtype=torch.bool))))
    results={}
    for name,rows in scored.items():
        values=torch.stack(features[name]); labels=torch.tensor(label_ids)
        results[name]=dict(metrics=aggregate(rows),
            by_emotion={e:aggregate([r for r,raw in zip(rows,per_clip[name]) if raw['emotion']==e]) for e in names if any(raw['emotion']==e for raw in per_clip[name])},
            probes=[classification_metrics(labels,net(values[:,m]).argmax(-1),names) for net,m in probes])
    if frozen!=(state_digest(model),state_digest(base.stage1)): raise ValueError('Frozen model changed')
    report=dict(schema='v5_decoder_bottleneck_v1',checkpoint_sha256=sha(a.checkpoint),binding_sha256=sha(a.binding),source_sha256=sha(__file__),
        clips=len(ids),indices=ids,test_loaded=False,query_motion_loaded_for_deployment=False,
        fixed_schedule=True,results=results,training_protocol=ck['protocol'],
        limits=['All oracle conditions use scoring targets and cannot be claimed as deployable gains.',
            'Fixed-schedule interventions isolate input response; they are not trained ablations.',
            'State anchors and geometry observers are proxies; replacing them may move out of training support.',
            'One seed and development subset; no promotion, novelty claim or test read.'])
    write(out/'report.json',finite(report));write(out/'per_clip.json',finite(per_clip))
    write(out/'complete.json',dict(status='complete',clips=len(ids),checkpoint_sha256=sha(a.checkpoint),report_sha256=sha(out/'report.json'),test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,default=128)
    run(p.parse_args())
