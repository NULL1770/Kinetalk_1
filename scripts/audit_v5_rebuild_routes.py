"""Read-only isolate teacher, reference response, flow and event routes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.factor_rebuild import V5FactorRebuild
from kinetalk_b0.models.expression_response import masked_pool
from scripts.pilot_v5_factor_rebuild import raw_audio,reference
from scripts.train_v7_pilot import load,batch,meanabs,motion_stats,FixedProbe
from scripts.train_expression_response import load_runtime,configure,write,state_digest
from kinetalk_v2.unified_model import adjacency


@torch.no_grad()
def main(a):
    configure(47);device=torch.device('cuda');binding=json.loads(Path(a.binding).read_text())
    model=V5FactorRebuild.from_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=False)).to(device).eval()
    before=state_digest(model)
    runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu()
    del runtime,base
    result={}
    for role in ('held_sentence','unseen_speaker'):
        data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std);rows=[]
        ys={name:[] for name in ('mean','flow_only','flow_only2','events_only','both','teacher')}
        gt=[];vm=[];labels=[]
        for start in range(0,len(data['motion']),8):
            b,r=batch(data,list(range(start,min(start+8,len(data['motion'])))),device)
            valid=b['valid'];s=model.encode_style(reference(r,0));sold=model.encode_style(reference(r,0),contextual=False)
            p=model.affect_audio(b['audio_features'],valid);t=model.visual_teacher(b)
            c=model.core.condition(p,valid)
            states=[model.flow.sample(p['state'],p['hidden'],c['emotion'],c['intensity'],valid,seed,model.rebuild_config.flow_steps,
                                     times=b['times']) for seed in (47+start,138+start)]
            zero=b['motion'].new_zeros(*valid.shape,model.cfg.event_groups*3)
            no=model.decode(b,p,s,events=zero)
            flows=[model.decode(b,{**p,'state':state},s,events=zero) for state in states]
            evt=model.decode(b,p,s,seed=47+start)
            both=model.decode(b,{**p,'state':states[0]},s,seed=47+start)
            teacher=model.decode(b,t,s,events=zero)
            oldctx=model.decode(b,p,sold,events=zero)
            obs=valid[...,None]&b['channel_mask'][:,None]
            err=lambda y:float((y-b['motion']).abs()[obs].mean())
            edge=adjacency(valid,b['times'])
            diff=t['state']-p['state'];fdiff=states[0]-p['state']
            rows.append(dict(response_mae=float(no['style_response'].abs()[obs].mean()),
                 response_real_recon_gain=err(no['motion']-no['style_response'])-err(no['motion']),
                 context_real_recon_gain=err(oldctx['motion'])-err(no['motion']),
                 teacher_noevent_recon_gain=err(no['motion'])-err(teacher['motion']),
                 teacher_correction_rms=float(diff[valid].square().mean().sqrt()),
                 flow_correction_rms=float(fdiff[valid].square().mean().sqrt()),
                 teacher_correction_step_rms=float((diff[:,1:]-diff[:,:-1])[edge].square().mean().sqrt()),
                 flow_correction_step_rms=float((fdiff[:,1:]-fdiff[:,:-1])[edge].square().mean().sqrt()),
                 flow_only_diversity=meanabs(flows[0]['motion'],flows[1]['motion'],valid)))
            outputs=dict(mean=no['motion'],flow_only=flows[0]['motion'],flow_only2=flows[1]['motion'],
                         events_only=evt['motion'],both=both['motion'],teacher=teacher['motion'])
            for k in ys:ys[k].append(outputs[k].cpu())
            gt.append(b['motion'].cpu());vm.append(valid.cpu());labels.append(b['emotion_id'].cpu())
        out={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        vm=torch.cat(vm);gt=torch.cat(gt);labels=torch.cat(labels);known=labels.ge(0)
        ys={k:torch.cat(v) for k,v in ys.items()};out['motions']={}
        for k,v in {'gt':gt,**ys}.items():out['motions'][k]={'mae':meanabs(v,gt,vm),**motion_stats(v,vm)}
        out['emotion_probes']={}
        for i in range(len(binding['probes'])):
            probe=FixedProbe(binding,i).to(device)
            out['emotion_probes'][str(i)]={k:float(probe.classes(v,vm).cpu()[known].eq(labels[known]).float().mean())
                                          for k,v in {'gt':gt,**ys}.items()}
        result[role]=out
    if state_digest(model)!=before:raise RuntimeError('read-only audit mutated weights')
    write(a.output,{'read_only':True,'results':result,'test_loaded':False,
                    'limits':['short-crop diagnostics','flow seeds do not prove natural dynamics',
                              'teacher reconstruction not deployment result']})
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--binding',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);main(p.parse_args())
