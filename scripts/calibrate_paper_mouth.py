"""Fit a TRAIN-only reference offset while freezing recovered mouth timing."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_paper_full_data import load_paper_data
from scripts.train_full_staged import base_forward,subset,obs,MOUTH,sha
from scripts.recover_paper_mouth import predict,grouped,paired_interval
from scripts.mouth_protection import protection_report
from scripts.train_formal_predictable_projection import save_json,save_checkpoint
from scripts.train_predictable_renderer import state_hash
from kinetalk_b0.reference_mouth_calibration import fit_reference_mouth_calibration


def residual_means(pred,q):
    mask=obs(q)[...,list(MOUTH)];values=q['motion'][...,list(MOUTH)]-pred[...,list(MOUTH)]
    count=mask.sum(1)
    return torch.where(mask,values,0.).sum(1)/count.clamp_min(1),count>0


@torch.no_grad()
def reference_statistics(system,data,device,batch):
    result={}
    for sid,q in data['refs'].items():
        pred=predict(system,q,device,batch)
        means,valid=residual_means(pred,q)
        if not valid.all():raise ValueError('All reference mouth channels must be observed')
        result[sid]={'mean':means.mean(0),'per_reference':means,'clip_ids':q['clip_id'],
            'sentence_ids':q['sentence_id'],'query':q,'prediction':pred}
    return result


def add_reference_offset(pred,q,refs,calibration):
    gain=torch.tensor(calibration['gain'],dtype=pred.dtype);bias=torch.tensor(calibration['bias'],dtype=pred.dtype)
    offsets=torch.stack([gain*refs[int(s)]['mean']+bias for s in q['speaker_id']])
    value=pred.clone();cc=list(MOUTH)
    value[...,cc]=torch.where(q['valid'][...,None],pred[...,cc]+offsets[:,None],0.)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','original','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh calibration output required')
    a.output.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    save_json(a.output/'status.json',{'status':'loading','test_loaded':False})
    data=load_paper_data(a.data,seed=47);source=torch.load(a.source,map_location='cpu',weights_only=False)
    original=torch.load(a.original,map_location='cpu',weights_only=False)
    for payload in (source,original):
        if payload.get('data_manifest_sha256')!=data['provenance']['manifest_sha256'] or payload.get('stage')!='articulation':
            raise ValueError('Require articulation checkpoints matching train/val manifest')
    system=data['system'].to(a.device).eval();system.load_state_dict(source['system'],strict=True)
    system.set_motion_support(torch.tensor(source['config']['model']['motion_support'],device=a.device))
    system.set_residual_support(torch.tensor(source['config']['model']['residual_support'],device=a.device))
    frozen=state_hash(system.state_dict())
    if a.smoke:
        # Same scope contract, reduced eval only. Fitting still uses every
        # neutral training query and every independent training enrollment.
        q=data['splits']['validation'];ids=torch.cat([(q['emotion_id']==0).nonzero(as_tuple=True)[0][:8],(q['emotion_id']!=0).nonzero(as_tuple=True)[0][:8]])
        data['splits']['validation']=subset(q,ids,'cpu')
    train=data['splits']['train'];neutral=subset(train,(train['emotion_id']==0).nonzero(as_tuple=True)[0],'cpu')
    refs=reference_statistics(system,data,a.device,a.batch_size)
    neutral_pred=predict(system,neutral,a.device,a.batch_size);means,mean_mask=residual_means(neutral_pred,neutral)
    reference=torch.stack([refs[int(s)]['mean'] for s in neutral['speaker_id']])
    provenance={'split':'train','query_clip_ids':neutral['clip_id'],'query_sentence_ids':neutral['sentence_id'],
        'reference_clip_ids':[refs[int(s)]['clip_ids'] for s in neutral['speaker_id']],
        'reference_sentence_ids':[refs[int(s)]['sentence_ids'] for s in neutral['speaker_id']]}
    calibration=fit_reference_mouth_calibration(means,reference,neutral['speaker_id'],
        query_mask=mean_mask,reference_mask=torch.ones_like(mean_mask),provenance=provenance)
    save_json(a.output/'calibration.json',calibration)
    cfg=copy.deepcopy(source['config']);cfg['model']['mouth_reference_calibration']=calibration
    system.set_mouth_reference_calibration(calibration)
    # Confirm the deployment identity path matches the training-free offset.
    for sid,row in refs.items():
        q=row['query'];b=subset(q,torch.arange(len(q['valid'])),a.device)
        residual=torch.where(obs(b),b['motion']-row['prediction'].to(a.device),0.)
        ident=system.encode_identity(residual[None],b['valid'][None],reference_channel_mask=b['channel_mask'][None])
        expected=torch.tensor(calibration['gain'],device=a.device)*row['mean'].to(a.device)+torch.tensor(calibration['bias'],device=a.device)
        torch.testing.assert_close(ident['baseline'][0,list(MOUTH)],expected,rtol=2e-5,atol=2e-7)
    val=data['splits']['validation'];raw={mode:predict(system,val,a.device,a.batch_size,mode) for mode in ('real','static','reverse')}
    corrected={mode:add_reference_offset(pred,val,refs,calibration) for mode,pred in raw.items()}
    system.load_state_dict(original['system'],strict=True)
    old=predict(system,val,a.device,a.batch_size)
    system.load_state_dict(source['system'],strict=True)
    if state_hash(system.state_dict())!=frozen:raise RuntimeError('B0 or other weight drift during calibration')
    protection=protection_report(corrected['real'],old,val['motion'],val['valid'],val['channel_mask'],val['emotion_id'])
    before=grouped(raw['real'],val);after=grouped(corrected['real'],val)
    pairs=val['valid'][:,1:]&val['valid'][:,:-1]
    delta=torch.diff(corrected['real'][...,list(MOUTH)],dim=1)-torch.diff(raw['real'][...,list(MOUTH)],dim=1)
    timing_error=float(delta[pairs].abs().max())
    # Mean correction must not trade emotional-mouth fit for neutral fit.
    nonneutral_ok=after['nonneutral']['raw_mse']<=before['nonneutral']['raw_mse']*1.03
    paired={mode:paired_interval(corrected['real'],corrected[mode],val) for mode in ('static','reverse')}
    passed=bool(protection['passed'] and nonneutral_ok and timing_error<2e-6 and all(x['passed'] for x in paired.values()) and not a.smoke)
    report={'passed':passed,'protection':protection,'nonneutral_raw_within_3pct_of_recovered':nonneutral_ok,
        'max_mouth_displacement_change':timing_error,'before':before,'after':after,'paired':paired,
        'scope':'reference-calibrated mouth; not complete identity/emotion/upper-face acceptance','test_loaded':False}
    save_json(a.output/'evaluation.json',report)
    protocol={'schema':'paper_reference_mouth_calibration_v1','data':data['provenance'],
        'source_sha256':sha(a.source),'original_sha256':sha(a.original),'calibration':calibration,
        'selection':'one predefined fit, no validation fitting','query_gt_inference':False,
        'fit_neutral_clips':len(neutral['valid']),'fit_speakers':len(neutral['speaker_id'].unique()),
        'validation_clips':len(val['valid']),'test_loaded':False}
    root=Path(__file__).resolve().parents[1]
    names=['scripts/calibrate_paper_mouth.py','kinetalk_b0/reference_mouth_calibration.py','kinetalk_b0/models/neutral_affect.py']
    protocol['source_sha256_files']={name:sha(root/name) for name in names}
    save_json(a.output/'protocol.json',protocol)
    for name in names:
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes())
    save_checkpoint(a.output/'final.pt',{**source,'config':cfg,'calibration_protocol':protocol,
        'mouth_calibration_passed':passed,'recovery_passed':passed,'mouth_baseline_contract':'frozen recovered B0 + train-fitted independent reference static offset'})
    save_checkpoint(a.output/'curves.pt',{'clip_id':val['clip_id'],'target':val['motion'],'valid':val['valid'],
        'channel_mask':val['channel_mask'],'times':val['times'],'emotion_id':val['emotion_id'],
        'predictions':{'original':old,'recovered':raw['real'],'calibrated':corrected['real']}})
    save_json(a.output/'status.json',{'status':'complete' if passed else 'gate_rejected','passed':passed,
        'elapsed_seconds':time.monotonic()-started,'test_loaded':False})
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
