"""Frozen geometry-head/projection audit on development, never a contact label.

The shared linear rig measures width, corner lift, aperture and depth. Query
motion is used exclusively for scoring; deployment generation receives audio,
B0 and one independent reference. No gain, lag or target geometry is fitted.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model, affine_geometry
from scripts.load_candidate_model import load_candidate_model
from kinetalk_v4.geometry_support import clean_geometry_jacobian, geometry_support
from scripts.train_expression_response import configure, load_runtime, sha, state_digest, write
from scripts.train_unified_v4 import select_balanced, build_base_cache, attach_base

NAMES = ('width_mm','corner_lift_mm','aperture_mm','corner_depth_mm')


def geometry_error(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Aligned observed geometry required')
    count = mask.sum(0)
    safe_p = np.where(mask,prediction,0.)
    safe_q = np.where(mask,target,0.)
    if not np.isfinite(safe_p).all() or not np.isfinite(safe_q).all():
        raise ValueError('Nonfinite observed geometry')
    p = safe_p.sum(0)/count.clip(1)
    q = safe_q.sum(0)/count.clip(1)
    difference = np.where(mask,prediction-target,0.)
    mae = np.abs(difference).sum(0)/count.clip(1)
    mse = np.square(difference).sum(0)/count.clip(1)
    bias = p-q
    centered = np.where(mask,(prediction-p)-(target-q),0.)
    centered_mse = np.square(centered).sum(0)/count.clip(1)
    return {name:dict(n=int(count[i]),prediction_mean=float(p[i]),target_mean=float(q[i]),
        MAE=float(mae[i]),MSE=float(mse[i]),bias=float(bias[i]),centered_MSE=float(centered_mse[i]))
        if count[i] else dict(n=0,prediction_mean=None,target_mean=None,MAE=None,MSE=None,bias=None,centered_MSE=None)
        for i,name in enumerate(NAMES)}


@torch.no_grad()
def main(args):
    configure(47); device=torch.device(args.device)
    root=Path(args.output)
    if root.exists():raise FileExistsError('Fresh geometry audit output required')
    root.mkdir(parents=True)
    binding=json.loads(Path(args.binding).read_text())
    data,base=load_runtime(binding,device);q=data['splits']['validation']
    ids=select_balanced(q,list(range(len(q['_lengths']))),args.limit,47)
    cache=build_base_cache(q,base,ids,device)
    class_names=data['config']['data']['emotion_classes']
    for path in args.checkpoint:
        ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['model']
        model=load_candidate_model(ck,device);model.eval().requires_grad_(False)
        before=(state_digest(model),state_digest(base.stage1))
        J=clean_geometry_jacobian(model.decoder.geometry_jacobian);offset=model.decoder.geometry_offset
        records=[]
        for start in range(0,len(ids),8):
            chosen=ids[start:start+8]
            gen=attach_base(q.batch(torch.tensor(chosen),device,
                keys=('audio_features','content','valid','channel_mask','times')),
                chosen,cache)
            assert 'motion' not in gen
            reference=[data['refs'][int(q['speaker_id'][i])] for i in chosen]
            if any(q['clip_id'][i] in r['clip_id'] for i,r in zip(chosen,reference)):
                raise ValueError('Query/reference clip overlaps')
            refs={k:torch.stack([r[k][0] for r in reference]).to(device)
                  for k in ('motion','valid','channel_mask')}
            output=model.predict(gen['audio_features'],gen['h0'],gen['b0'],gen['valid'],refs,
                times=gen['times'],seed=47+start)
            score=q.batch(torch.tensor(chosen),device,keys=('motion','valid','channel_mask'))
            observed=geometry_support(J,score['channel_mask'])[:,None]&score['valid'][...,None]
            truth=affine_geometry(torch.where(score['channel_mask'][:,None],score['motion'],0.),J,offset)
            variants={'B0':affine_geometry(gen['b0'],J,offset),
                'geometry_head':output['geometry_target'],
                'proposal_raw':affine_geometry(output['proposal'],J,offset),
                'projected_raw':affine_geometry(output['motion'],J,offset),
                'projected_clip_all':affine_geometry(output['motion'].clamp(0,1),J,offset)}
            for j,i in enumerate(chosen):
                mask=observed[j].cpu().numpy()
                # Geometry support is frame x observer; broadcast a static axis
                # support only after intersecting native frame validity.
                variants_np={k:v[j].cpu().numpy() for k,v in variants.items()}
                target=truth[j].cpu().numpy()
                records.append(dict(clip_id=q['clip_id'][i],emotion=class_names[int(q['emotion_id'][i])],
                    reference_id=reference[j]['clip_id'][0],query_motion_loaded_for_generation=False,
                    geometry={k:geometry_error(v,target,mask) for k,v in variants_np.items()}))
            write(root/'state.json',dict(status='scoring',clips=len(records),total=len(ids),checkpoint=sha(path)))
        summary={}
        for emotion in ['ALL']+class_names:
            selected=[r for r in records if emotion=='ALL' or r['emotion']==emotion]
            if not selected:continue
            summary[emotion]={}
            for mode in records[0]['geometry']:
                summary[emotion][mode]={}
                for name in NAMES:
                    supported=[r['geometry'][mode][name] for r in selected if r['geometry'][mode][name]['n']]
                    summary[emotion][mode][name]={
                        key:float(np.mean([v[key] for v in supported])) if supported else None
                        for key in ('MAE','MSE','bias','centered_MSE','prediction_mean','target_mean')}
                    summary[emotion][mode][name]['supported_clips']=len(supported)
        if before!=(state_digest(model),state_digest(base.stage1)):
            raise RuntimeError('Frozen geometry audit mutated weights')
        report=dict(schema='v5_geometry_head_projection_audit_v1',clips=len(ids),indices=ids,
            checkpoint_sha256=sha(path),binding_sha256=sha(args.binding),source_sha256=sha(__file__),
            geometry_jacobian_cleaned=True,coordinate_names=NAMES,test_loaded=False,promotion=False,
            query_motion_role='scoring only',summary=summary,records=records,
            limits=['Shared rig coordinates, not calibrated physiological contact or phoneme correctness.',
                    'Projection/head counterfactual comparison is not a trained ablation.',
                    'First reference/no best seed, no lag or gain fitting; balanced development subset only.'])
        write(root/(sha(path)[:12]+'.json'),report)
    write(root/'complete.json',dict(status='complete',checkpoints=len(args.checkpoint),clips=len(ids),test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--checkpoint',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=128)
    p.add_argument('--device',default='cuda');main(p.parse_args())
