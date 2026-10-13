"""Staged bounded pilot with fixed observers and real factor interventions.

Run prepare_v7_pilot.py first. All fitting is on fit.npz; held_sentence and
unseen_speaker are diagnostic only. No test files or default model are touched.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v7.disentangled_model import V7Config,V7DisentangledModel,v7_loss,_seq_l1,_seq_mse
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features
from kinetalk_b0.models.expression_response import masked_pool
from scripts.train_expression_response import configure,state_digest,write,save

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:torch.from_numpy(z[k]) for k in z.files}

def batch(data,ids,device):
    ids=torch.tensor(ids,dtype=torch.long)
    b={k:v[ids].to(device) for k,v in data.items() if not k.startswith('ref_')}
    refs={k[4:]:v[ids].to(device) for k,v in data.items() if k.startswith('ref_')}
    return b,refs

def one_ref(refs,n):return {k:v[:,n:n+1] for k,v in refs.items()}

def meanabs(x,y,valid):return float(_seq_l1(x,y,valid))

def motion_stats(y,valid):
    edge=valid[:,1:]&valid[:,:-1]
    speed=meanabs(y[:,1:],y[:,:-1],edge)
    acc=(y[:,2:]-2*y[:,1:-1]+y[:,:-2])
    triple=valid[:,2:]&valid[:,1:-1]&valid[:,:-2]
    accel=float(acc[triple].abs().mean()) if triple.any() else 0.
    ranges=[]
    for x,m in zip(y,valid):
        if m.any():ranges.append((x[m].quantile(.9,dim=0)-x[m].quantile(.1,dim=0)).mean())
    return {'speed':speed,'acceleration':accel,'q90_q10_range':float(torch.stack(ranges).mean())}

class FixedProbe:
    def __init__(self,binding,index=0):
        saved=torch.load(binding['probes'][index]['path'],map_location='cpu',weights_only=False)
        if sha(binding['probes'][index]['path'])!=binding['probes'][index]['sha256']:raise ValueError('Probe binding differs')
        self.net=MotionEmotionProbe(saved['feature_dim'],saved['hidden'],8).eval().requires_grad_(False)
        self.net.load_state_dict(saved['model'])
        self.mask=saved.get('feature_mask',torch.ones(saved['feature_dim'],dtype=torch.bool))
    def to(self, device):
        self.net.to(device)
        self.mask=self.mask.to(device)
        return self
    def logits(self,y,valid):
        # Keep the trained observer frozen while retaining gradients through
        # differentiable motion statistics to the generated sequence.
        fs=torch.stack([motion_features(x[:,:51].clamp(0,1),m)
                        for x,m in zip(y,valid)])
        return self.net(fs[:,self.mask])
    @torch.no_grad()
    def classes(self,y,valid):
        device=next(self.net.parameters()).device
        return self.logits(y.to(device),valid.to(device)).argmax(-1).to(y.device)

@torch.no_grad()
def evaluate(model,data,device,probe,seed,output=None,role='held_sentence'):
    model.eval();metrics=[];records=[];n=len(data['motion']);examples={}
    from scripts.diagnose_v7_factor_routes import cross_person_donors
    # Enrollment is encoded across the whole role, before batching queries.
    # Sorted query batches each contain one speaker; they cannot supply valid
    # cross-speaker donors. Match donor query emotion where possible.
    donor_ids=cross_person_donors(data['speaker_id'],data['emotion_id'])
    enrolled=[]
    for start in range(0,n,8):
        _,refs=batch(data,list(range(start,min(start+8,n))),device)
        enrolled.append(model.encode_style(one_ref(refs,0))['mean'])
    enrolled=torch.cat(enrolled)
    for start in range(0,n,8):
        ids=list(range(start,min(start+8,n)));b,refs=batch(data,ids,device);valid=b['valid'].bool()
        r0,r1=one_ref(refs,0),one_ref(refs,1)
        controlled=b['emotion_id'].clamp_min(0)
        model_out=model(b,r0)
        c=model_out['content'];teach=model_out['teacher'];student=model_out['student'];s0=model_out['style']['mean']
        s1=model.encode_style(r1)['mean'];sz=torch.zeros_like(s0)
        # Change to another speaker within this balanced batch where possible.
        donor=[]
        for j,sid in enumerate(b['speaker_id']):
            other=torch.where(b['speaker_id']!=sid)[0]
            donor.append(int(other[0]) if len(other) else (j+1)%len(ids))
        sd=s0[torch.tensor(donor,device=device)]
        e=teach['trajectory']
        ys={name:model.decode(c,e,s,valid)['motion'] for name,s in [('teacher',s0),('teacher_zero',sz),('teacher_wrong',sd)]}
        ys['same_person_other_reference']=model.decode(c,e,s1,valid)['motion']
        modal=model.predict(b['audio_features'],valid,r0,h0=b['h0'],sample_flow=False)
        sample1=model.predict(b['audio_features'],valid,r0,h0=b['h0'],seed=seed+start,sample_flow=True)
        sample2=model.predict(b['audio_features'],valid,r0,h0=b['h0'],seed=seed+start+91,sample_flow=True)
        ys.update(audio_mean=modal['motion'],sample_1=sample1['motion'],sample_2=sample2['motion'])
        ys['controlled']=model.decode(c,model.class_control(controlled,valid,sample1['trajectory']),s0,valid)['motion']
        changed=(controlled+1)%model.cfg.classes
        ys['emotion_changed']=model.decode(c,model.class_control(changed,valid,sample1['trajectory']),s0,valid)['motion']
        teacher_class=probe.classes(ys['teacher'],valid)
        gt_class=probe.classes(b['motion'],valid)
        all_classes={k:probe.classes(ys[k],valid) for k in ('audio_mean','sample_1','controlled','emotion_changed','teacher_wrong')}
        known=b['emotion_id'].ge(0)
        for j,i in enumerate(ids):
            mask=valid[j:j+1];gt=b['motion'][j:j+1]
            row={'row':i,'speaker_id':int(b['speaker_id'][j]),'emotion_id':int(b['emotion_id'][j]),
                 'dataset_id':int(b['dataset_id'][j])}
            for name,y in ys.items():row[name+'_mae']=meanabs(y[j:j+1],gt,mask)
            mouth=torch.arange(14,40,device=device)
            for name in ('teacher','audio_mean','sample_1','controlled'):
                row[name+'_mouth_mae']=meanabs(ys[name][j:j+1,:,mouth],gt[:,:,mouth],mask)
            row['b0_mouth_mae']=meanabs(b['b0'][j:j+1,:,mouth],gt[:,:,mouth],mask)
            row['gt_probe_correct']=int(gt_class[j]==b['emotion_id'][j]) if known[j] else None
            row['teacher_probe_correct']=int(teacher_class[j]==b['emotion_id'][j]) if known[j] else None
            for name in all_classes:
                row[name+'_probe_correct']=int(all_classes[name][j]==b['emotion_id'][j]) if known[j] else None
            row['changed_emotion_probe_correct']=int(all_classes['emotion_changed'][j]==changed[j]) if known[j] else None
            row['same_person_style_code_delta']=float((s0[j]-s1[j]).square().mean().sqrt())
            row['different_person_style_code_delta']=float((s0[j]-sd[j]).square().mean().sqrt())
            row['reference_swap_motion_delta']=meanabs(ys['teacher'][j:j+1],ys['same_person_other_reference'][j:j+1],mask)
            row['style_swap_motion_delta']=meanabs(ys['teacher'][j:j+1],ys['teacher_wrong'][j:j+1],mask)
            row['emotion_swap_motion_delta']=meanabs(ys['controlled'][j:j+1],ys['emotion_changed'][j:j+1],mask)
            row['sample_diversity']=meanabs(ys['sample_1'][j:j+1],ys['sample_2'][j:j+1],mask)
            observer={k:model.content_observer(ys[k][j:j+1],mask) for k in ('teacher','teacher_wrong','controlled','emotion_changed')}
            row['style_swap_content_delta']=meanabs(observer['teacher'],observer['teacher_wrong'],mask)
            row['emotion_swap_content_delta']=meanabs(observer['controlled'],observer['emotion_changed'],mask)
            obs_gt=model.content_observer(gt,mask)
            row['observer_gt_h0_mse']=float(_seq_mse(obs_gt,b['h0'][j:j+1],mask))
            for name,y in [('gt',gt),('mean',ys['audio_mean'][j:j+1]),('sample',ys['sample_1'][j:j+1])]:
                for k,v in motion_stats(y,mask).items():row[name+'_'+k]=v
            metrics.append(row)
            if int(b['emotion_id'][j]) not in examples and int(b['dataset_id'][j])==0:
                emo=int(b['emotion_id'][j]);examples[emo]={'row':i,'motions':torch.stack([gt[0],b['b0'][j],ys['teacher'][j],ys['audio_mean'][j],ys['sample_1'][j],ys['controlled'][j]]).cpu().numpy(),
                     'times':b['times'][j].cpu().numpy(),'valid':mask[0].cpu().numpy(),'channel_mask':b['channel_mask'][j].cpu().numpy()}
    keys=sorted(set(k for r in metrics for k in r)-{'row','speaker_id','emotion_id','dataset_id'})
    result={k:float(np.mean([r[k] for r in metrics if r.get(k) is not None])) for k in keys}
    result['style_separation_ratio']=result['same_person_style_code_delta']/max(result['different_person_style_code_delta'],1e-8)
    if output:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        write(out/(role+'_per_clip.json'),metrics)
        for emo,item in examples.items():
            np.savez_compressed(out/(role+'_emotion_'+str(emo)+'.npz'),**item,
                                mode_names=np.array(['GT','B0','v7_teacher','v7_audio_mean','v7_sample','v7_controlled']))
    return result

def main(a):
    configure(a.seed);device=torch.device(a.device);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    fit=load(Path(a.data)/'fit.npz');held=load(Path(a.data)/'held_sentence.npz');unseen=load(Path(a.data)/'unseen_speaker.npz')
    speaker_values=sorted(set(fit['speaker_id'].tolist()));speaker_map={s:i for i,s in enumerate(speaker_values)}
    for data in (fit,held,unseen):data['speaker_class']=torch.tensor([speaker_map.get(s,-1) for s in data['speaker_id'].tolist()])
    cfg=V7Config(hidden=a.hidden,decoder_hidden=a.decoder_hidden,content_dim=fit['h0'].shape[-1],
                 style_dim=a.style_dim,speaker_classes=len(speaker_values),use_flow=not a.no_flow,use_response=not a.no_response)
    model=V7DisentangledModel(cfg).to(device)
    model.motion_scale.copy_(fit['motion'][fit['valid']].std(0).clamp_min(.025).to(device))
    rng=np.random.default_rng(a.seed);history=[];t0=time.time()
    def next_batch():return batch(fit,rng.choice(len(fit['motion']),a.batch,replace=False).tolist(),device)
    # Independent motion->audio-content anchor. Freeze it before generator fits.
    opt=torch.optim.AdamW(model.content_observer.parameters(),lr=a.lr)
    for step in range(a.observer_steps):
        model.content_observer.train();b,_=next_batch();opt.zero_grad(set_to_none=True)
        pred=model.content_observer(b['motion'],b['valid'].bool())
        loss=_seq_mse(pred,b['h0'],b['valid'].bool());loss.backward();opt.step()
    model.content_observer.eval().requires_grad_(False)
    observer_digest=state_digest(model.content_observer)
    # Stage 1: learn stable visual space and population/individual response.
    stage1_params=[p for n,p in model.named_parameters() if not n.startswith(('student.','flow.','content_observer.'))]
    opt=torch.optim.AdamW(stage1_params,lr=a.lr)
    first=None
    for step in range(a.teacher_steps):
        model.train();model.content_observer.eval();b,refs=next_batch();opt.zero_grad(set_to_none=True)
        pack=model(b,one_ref(refs,step%2),seed=a.seed+step)
        losses=v7_loss(model,pack,b,teacher_weight=1,student_weight=0,flow_weight=0,control_weight=0,style_weight=.005)
        selected=[k for k in losses if k not in ('total','student_emotion','student_intensity','student_recon','student_velocity','flow','trajectory_mean_distill','content_preserve')]
        loss=sum(losses[k] for k in selected)
        pop=pack['teacher_out']['population']
        observed=b['valid'][...,None].bool()&b['channel_mask'][:,None].bool()
        loss=loss+a.population_weight*_seq_l1(pop/model.motion_scale,b['motion']/model.motion_scale,observed)
        # Warm up the population expression before allowing a poorly fitted
        # reference likelihood to absorb the whole output into S.
        if step<a.population_warmup:
            loss=loss+2*_seq_l1(pop/model.motion_scale,b['motion']/model.motion_scale,observed)
        # Two legal references share the original query target; no fabricated swap GT.
        if step%4==0:
            other=model.encode_style(one_ref(refs,1-step%2))
            y=model.decode(pack['content'],pack['teacher']['trajectory'],other['mean'],b['valid'].bool())['motion']
            loss=loss+.5*_seq_l1(y/model.motion_scale,b['motion']/model.motion_scale,observed)
            loss=loss+.20*(pack['style']['mean']-other['mean']).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite teacher loss')
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(stage1_params,5));opt.step()
        row={'stage':'visual','step':step+1,'loss':float(loss),'teacher_recon':float(losses['teacher_recon']),'grad':grad}
        if first is None:first=row
        history.append(row)
        if (step+1)%32==0:write(out/'state.json',{**row,'elapsed_seconds':time.time()-t0});print(json.dumps(row),flush=True)
    save(out/'visual.pt',{'config':asdict(cfg),'model':model.state_dict(),'history':history})
    visual_digest=state_digest(model.teacher);decoder_digest=state_digest(model.decoder)
    # Stage 2: fixed teacher/decoder targets, audio mean + conditional trajectory flow.
    model.requires_grad_(False);model.student.requires_grad_(True);model.flow.requires_grad_(True)
    stage2_params=list(model.student.parameters())+list(model.flow.parameters())
    opt=torch.optim.AdamW(stage2_params,lr=a.lr)
    for step in range(a.student_steps):
        model.eval();model.student.train();model.flow.train();b,refs=next_batch();opt.zero_grad(set_to_none=True)
        pack=model(b,one_ref(refs,step%2),seed=a.seed+step)
        losses=v7_loss(model,pack,b,teacher_weight=0,student_weight=.25,flow_weight=1.,control_weight=0,leakage_weight=0)
        selected=('student_recon','student_velocity','flow','trajectory_mean_distill','student_emotion','student_intensity','student_output_emotion','content_preserve')
        loss=sum(losses[k] for k in selected)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite student loss')
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(stage2_params,5));opt.step()
        row={'stage':'audio_flow','step':step+1,'loss':float(loss),'student_recon':float(losses['student_recon'])*4,
             'flow':float(losses['flow']),'grad':grad};history.append(row)
        if (step+1)%32==0:write(out/'state.json',{**row,'elapsed_seconds':time.time()-t0});print(json.dumps(row),flush=True)
    if state_digest(model.teacher)!=visual_digest or state_digest(model.decoder)!=decoder_digest:raise RuntimeError('Frozen visual model changed')
    if state_digest(model.content_observer)!=observer_digest:raise RuntimeError('Fixed content observer changed')
    binding=json.loads(Path(a.binding).read_text());probe=FixedProbe(binding)
    result={role:evaluate(model,data,device,probe,a.seed,out/'evidence',role) for role,data in [('held_sentence',held),('unseen_speaker',unseen)]}
    h=result['held_sentence']
    gates={'finite_optimization':True,'teacher_reconstruction':h['teacher_mae']<.08,
           'flow_quality':h['sample_1_mae']<h['audio_mean_mae']*1.15,
           'flow_roughness':h['sample_acceleration']<h['gt_acceleration']*2,
           'style_separation':h['style_separation_ratio']<.8,
           'style_query_advantage':h['teacher_mae']<h['teacher_wrong_mae'],
           'style_effect':h['style_swap_motion_delta']>1e-3,
           'emotion_control':h['controlled_probe_correct']>=.5 and h['changed_emotion_probe_correct']>=.5,
           'content_observer_reliable':h['observer_gt_h0_mse']<float(fit['h0'][fit['valid']].var(0).mean())*.8,
           'mouth_vs_b0':h['audio_mean_mouth_mae']<=h['b0_mouth_mae'],
           'nontrivial_diversity':1e-4<h['sample_diversity']<.05}
    protocol={'data_provenance_sha256':sha(Path(a.data)/'provenance.json'),'test_loaded':False,
              'teacher_steps':a.teacher_steps,'student_steps':a.student_steps,'observer_steps':a.observer_steps,
              'frozen_teacher_verified':True,'frozen_decoder_verified':True,'frozen_observer_verified':True,
              'hdtf_missing_labels_masked':True,'one_reference_inference':True,
              'source_hashes':{str(p.relative_to(Path(__file__).resolve().parents[1])):sha(p) for p in [Path(__file__),Path(__file__).resolve().parents[1]/'kinetalk_v7/disentangled_model.py']}}
    report={'schema':'v7_multibatch_staged_pilot_v2','status':'complete','config':asdict(cfg),'results':result,'gates':gates,
            'long_training_ready':all(gates.values()),'elapsed_seconds':time.time()-t0,'protocol':protocol,'paper_result':False,
            'limits':['bounded crop pilot; not full-sequence or matched-budget paper result',
                      'h0 observer is an articulation anchor, not a validated viseme or sync evaluator',
                      'whole-clip emotion statistics probe is not framewise emotion truth',
                      'conditional posterior variance is not empirically calibrated',
                      'HDTF style evidence is buffered same-source only; no cross-emotion guarantee']}
    save(out/'final.pt',{'config':asdict(cfg),'model':model.state_dict(),'protocol':protocol})
    write(out/'history.json',history);write(out/'report.json',report);write(out/'complete.json',{'status':'complete','long_training_ready':all(gates.values()),'elapsed_seconds':time.time()-t0})
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--binding',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--hidden',type=int,default=48);p.add_argument('--decoder-hidden',type=int,default=64);p.add_argument('--style-dim',type=int,default=16)
    p.add_argument('--observer-steps',type=int,default=128);p.add_argument('--teacher-steps',type=int,default=256);p.add_argument('--student-steps',type=int,default=256)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--no-flow',action='store_true');p.add_argument('--no-response',action='store_true')
    p.add_argument('--population-weight',type=float,default=.75);p.add_argument('--population-warmup',type=int,default=192)
    main(p.parse_args())
