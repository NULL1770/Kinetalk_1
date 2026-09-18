"""Bounded-coordinate fixed-clock shape support with a frozen static baseline."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_clocked_motion_prior as c
from scripts import joint_motion_dictionary as original_dictionary
from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior, TemporalResidualPrior, categorical_energy_score

SCHEMA='clocked_bounded_static_residual_v1'


def coordinate_clips(clips,ids):
    result=list(clips); counts={}
    for i in ids:
        source=clips[i]; target=copy.copy(source); valid=source['valid'].numpy()
        target['upper']=np.zeros_like(source['upper'],dtype=np.float64)
        target['upper'][valid]=original_dictionary.logit(source['upper'][valid])
        target['anchor_upper']=original_dictionary.logit(source['anchor_upper'])
        result[i]=target
        counts[source['clip_id']]={'observed_values':int(source['upper'][valid].size),
            'clipped_values':int(((source['upper'][valid]<1e-4)|(source['upper'][valid]>1-1e-4)).sum())}
    return result,counts


def train_models(data,stats,dictionary,output,device,epochs):
    torch.manual_seed(c.SEED)
    base=ClockedMotionPrior(*stats,horizon=c.HORIZON,hidden=64,k=len(dictionary['shapes'])).to(device)
    tensors={k:v.to(device) for k,v in data.items()}
    distance=torch.tensor(c.shapes.pairwise_distances(dictionary['shapes'],dictionary['scales']),
                          dtype=torch.float32,device=device)
    histories={};wrapper=None
    for arm in ('static','residual'):
        if arm=='residual':
            torch.manual_seed(c.SEED)
            wrapper=TemporalResidualPrior(base)
        model=base if arm=='static' else wrapper
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01)
        rng=np.random.default_rng(c.SEED); history=[]; orderhash=hashlib.sha256()
        for epoch in range(epochs):
            model.train();order=rng.permutation(len(tensors['valid']));orderhash.update(order.tobytes())
            total,count=0.,0
            for start in range(0,len(order),128):
                ix=torch.as_tensor(order[start:start+128],device=device)
                if arm=='static':
                    logits=model(tensors['static'][ix],tensors['valid'][ix],tensors['global'][ix])
                else:
                    logits=model(tensors['temporal'][ix],tensors['valid'][ix],tensors['global'][ix],tensors['static'][ix])
                score=categorical_energy_score(logits.softmax(-1),tensors['distance'][ix],distance)
                loss=(score*tensors['weight'][ix]).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
                total+=float(loss.detach())*len(ix);count+=len(ix)
            history.append({'epoch':epoch+1,'energy_score':total/count,'updates':(len(order)+127)//128})
            c.old.save_json(output/(arm+'_losses.json'),history)
            c.old.save_json(output/'status.json',{'status':'training','arm':arm,'epoch':epoch+1,'epochs':epochs})
            print('EPOCH',arm,epoch+1,total/count,flush=True)
        model.eval();histories[arm]=history
        torch.save({'schema':SCHEMA,'state':model.state_dict(),'epochs':epochs,
            'order_sha256':orderhash.hexdigest(),'optimizer':optimizer.state_dict()},output/(arm+'_final.pt'))
        if arm=='static':
            before={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    if any(not torch.equal(v,before[k]) for k,v in base.cpu().state_dict().items()):
        raise RuntimeError('Frozen static baseline changed during residual training')
    wrapper.to(device)
    c.old.save_json(output/'matching.json',{'base_unchanged_exact':True,'epochs_each':epochs,
        'windows':len(tensors['valid']),'order_sha256':orderhash.hexdigest()})
    return wrapper,histories


@torch.no_grad()
def probability(model,features,static,valid,global_vec,device,scale):
    x,mask,locations=c.acoustic_windows(features,valid)
    sx,sm,sl=c.acoustic_windows(static,valid)
    if not torch.equal(mask,sm) or locations!=sl:
        raise RuntimeError('Paired acoustic window clock differs')
    g=torch.as_tensor(global_vec,dtype=torch.float32,device=device)[None].expand(len(x),-1)
    outputs=[]
    for start in range(0,len(x),128):
        logits=model(x[start:start+128].to(device),mask[start:start+128].to(device),
            g[start:start+128],sx[start:start+128].to(device),scale=scale)
        outputs.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(outputs),locations


def evaluate(clips,ids,dictionary,fitted,scales,low_cut,model,device,*,intervention='real',residual_scale=1.):
    rows,curves,donors=[],{},{}
    if intervention=='mismatch':
        for i in ids:
            candidates=[j for j in ids if clips[j]['speaker']==clips[i]['speaker']
                and clips[j]['emotion']==clips[i]['emotion'] and clips[j]['sentence']!=clips[i]['sentence']
                and len(c.old.runs(clips[j]['valid'].numpy()))==1]
            if candidates:donors[i]=min(candidates,key=lambda j:clips[j]['clip_id'])
        ids=[i for i in ids if i in donors]
    for i in ids:
        item=clips[i];valid=item['valid'].numpy()
        features=c.intervention_audio(item,intervention,clips[donors[i]] if i in donors else None)
        # Original clip static mean is held fixed even under donor replacement.
        static=c.static_acoustics(item['features'],item['valid'])
        probabilities,locations=probability(model,features,static,item['valid'],item['global'],device,residual_scale)
        level=c.equilibrium(item['global'],original_dictionary.logit(item['anchor_upper']),fitted)
        logits,tokens=c.sample_shapes(probabilities,locations,dictionary,level,valid,item['clip_id'])
        samples=np.where(valid[None,:,None],expit(logits),0.)
        score=c.metrics.score_clip(samples,item['upper'],valid,scales)
        energy=float(np.mean((c.old.center_runs(item['upper'],valid)[valid]/scales)**2))
        rows.append({'clip_id':item['clip_id'],'sentence':item['sentence'],'speaker':item['speaker'],
            'emotion':item['emotion'],'low_activity':energy<=low_cut,
            'acceleration':c.acceleration_stats(samples,item['upper'],valid,scales),**score})
        curves[item['clip_id']]={'samples':samples.astype(np.float32),'target':item['upper'],'valid':valid,
            'probabilities':probabilities,'tokens':tokens,'locations':locations,'logit_level':level}
    report=c.report_rows(rows);report.update(intervention=intervention,residual_scale=residual_scale,
        donor_mapping={clips[i]['clip_id']:clips[j]['clip_id'] for i,j in donors.items()})
    return report,curves


def eligible(report,baseline):
    s,b=report['summary'],baseline['summary']
    return (c.previous._relative_not_worse(s['joint_fair_es']['raw'],b['joint_fair_es']['raw']) and
            c.previous._relative_not_worse(s['variogram']['aggregate'],b['variogram']['aggregate']) and
            c.previous._relative_not_worse(s['covariance_distance']['velocity'],b['covariance_distance']['velocity']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audio','targets','native-root','native-manifest','delta-dir','audio-checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();started=time.monotonic()
    if args.output.exists():raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    c.old.save_json(args.output/'status.json',{'status':'loading','schema':SCHEMA})
    loading=copy.copy(args);loading.smoke=False
    clips,original,lineage=c.old.load_clips(loading);split=c.previous.split_inner(clips,original)
    if args.smoke:split={k:v[:12] for k,v in split.items()}
    ids=sorted(set(i for k in ('fit','calibration','confirmation') for i in split[k]))
    target=torch.load(args.targets,map_location='cpu',weights_only=False,mmap=True)['splits']['train']
    anchors={cid:row[c.previous.CC].numpy() for cid,row in zip(target['clip_id'],target['anchors'])}
    frozen=c.load_frozen_audio(args.audio_checkpoint,args.device);binding=c.assert_source_binding(frozen,lineage)
    contexts=c.encode_clips(frozen,[clips[i] for i in ids],batch_size=16,device=args.device)
    for i,context in zip(ids,contexts):
        clips[i]['global']=np.r_[context['global'].numpy(),context['intensity'].numpy()]
        clips[i]['anchor_upper']=anchors[clips[i]['clip_id']]
    del frozen,contexts
    root=Path(__file__).resolve().parents[1]
    files=['scripts/train_clocked_residual_prior.py','scripts/train_clocked_motion_prior.py',
        'scripts/clocked_motion_dictionary.py','scripts/joint_motion_dictionary.py',
        'kinetalk_b0/models/clocked_motion_prior.py','scripts/joint_motion_metrics.py',
        'scripts/train_joint_motion_prior.py','scripts/train_motion_process.py','scripts/joint_prior_audio_context.py',
        'scripts/full_native_context_data.py','scripts/full_staged_data.py','kinetalk_b0/models/slow_state_affect.py']
    protocol={'schema':SCHEMA,'source':lineage,'frozen_audio':binding,'smoke':args.smoke,
        'split':{k:[{m:clips[i][m] for m in ('clip_id','sentence','speaker','emotion')} for i in split[k]]
                 for k in ('fit','calibration','confirmation')},'development_only':True,
        'code_sha256':{f:c.old.sha(root/f) for f in files},
        'protocol_sha256':c.old.sha(root/'docs/CLOCKED_RESIDUAL_PROTOCOL_20260918.md'),'test_loaded':False}
    c.old.save_json(args.output/'protocol.json',protocol)
    for f in files:
        out=args.output/'source'/f;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((root/f).read_bytes())
    (args.output/'source/protocol.md').write_bytes((root/'docs/CLOCKED_RESIDUAL_PROTOCOL_20260918.md').read_bytes())
    fit_clips,clipping=coordinate_clips(clips,split['fit'])
    c.old.save_json(args.output/'fit_clipping.json',clipping)
    windows=c.shapes.extract_windows(fit_clips,split['fit'],horizon=c.HORIZON,hop=c.HOP)
    dictionary=c.shapes.fit_shape_dictionary(windows,k=16 if args.smoke else 128,seed=c.SEED)
    dictionary['coordinate_system']='logit raw minus observed training clip logit mean'
    torch.save(dictionary,args.output/'dictionary.pt')
    fitted=c.fit_equilibrium(fit_clips,split['fit']);torch.save(fitted,args.output/'equilibrium.pt')
    scales=c.previous.raw_scales(clips,split['fit']);c.old.save_json(args.output/'scales.json',scales.tolist())
    low_cut=float(np.quantile([np.mean((c.old.center_runs(clips[i]['upper'],clips[i]['valid'].numpy())
        [clips[i]['valid'].numpy()]/scales)**2) for i in split['fit']],.2))
    fm,fs=c.old.fit_feature_stats(clips,split['fit']);g=np.stack([clips[i]['global'] for i in split['fit']])
    stats=(fm,fs,torch.tensor(g.mean(0),dtype=torch.float32),torch.tensor(np.maximum(g.std(0),.01),dtype=torch.float32))
    data=c.build_training(clips,windows,dictionary)
    c.old.save_json(args.output/'training_data.json',{'windows':len(windows),'clips':len(split['fit']),
        'low_activity_train_quantile20':low_cut,'coordinate':'logit','clip_balanced':True})
    model,losses=train_models(data,stats,dictionary,args.output,args.device,2 if args.smoke else 30);del data
    choices=[];baseline=None
    for scale in (0.,.25,.5,1.):
        report,_=evaluate(clips,split['calibration'],dictionary,fitted,scales,low_cut,model,args.device,residual_scale=scale)
        c.old.save_json(args.output/('cal_scale_'+str(scale)+'.json'),report)
        if baseline is None:baseline=report
        choice={'scale':scale,'score':report['summary']['joint_fair_es']['centered'],'eligible':eligible(report,baseline)}
        choices.append(choice);print('CAL_SCALE',json.dumps(choice),flush=True)
    chosen=min([r for r in choices if r['eligible']],key=lambda r:r['score'])['scale']
    c.old.save_json(args.output/'selection.json',{'scale':chosen,'choices':choices,'scope':'199 calibration only'})
    for cell in ('calibration','confirmation'):
        reports={}
        for label,intervention,scale in [('static_trained','real',0.),('temporal','real',chosen),
                ('static','static',chosen),('reverse','reverse',chosen),('mismatch','mismatch',chosen)]:
            report,curves=evaluate(clips,split[cell],dictionary,fitted,scales,low_cut,model,args.device,
                intervention=intervention,residual_scale=scale)
            reports[label]=report;c.old.save_json(args.output/(cell+'_'+label+'.json'),report)
            torch.save(curves,args.output/(cell+'_'+label+'.pt'))
            print('EVAL',cell,label,report['summary']['joint_fair_es'] if report['summary'] else None,flush=True)
        gate=c.assessment(reports)
        if chosen==0.:gate['timing_passed']=False;gate['reason']='zero_residual_scale_selected'
        c.old.save_json(args.output/(cell+'_assessment.json'),gate)
        print('ASSESSMENT',cell,json.dumps(gate),flush=True)
    c.old.save_json(args.output/'status.json',{'schema':SCHEMA,'status':'complete','seconds':time.monotonic()-started,
        'epochs_per_stage':len(losses['static']),'smoke':args.smoke,'selected_scale':chosen,
        'timing_passed':gate['timing_passed'],'quality_passed':gate['quality_passed'],
        'development_only':True,'test_loaded':False,'generator_integrated':False,'default_replaced':False})


if __name__=='__main__':main()
