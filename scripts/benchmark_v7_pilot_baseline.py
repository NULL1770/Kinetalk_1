"""Frozen retained-v5 descriptive benchmark on precisely the v7 pilot crops."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.append('/root/kinetalk_v5_20261011/code')
from kinetalk_v4.unified_model_v4 import KineTalkV4Config,KineTalkV4Model
from scripts.train_expression_response import configure,load_runtime,state_digest,write
from scripts.train_v7_pilot import load,batch,sha,FixedProbe,meanabs,motion_stats

@torch.no_grad()
def main(a):
    configure(47);device=torch.device(a.device);binding=json.loads(Path(a.binding).read_text())
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if sha(a.checkpoint)!=a.sha256:raise ValueError('Retained v5 checkpoint changed')
    cfg=KineTalkV4Config(**ck['config']);state=ck['model']
    model=KineTalkV4Model(cfg,state['feature_mean'],state['feature_std'],state['scales'],
                         state['decoder.geometry_jacobian'],state['decoder.geometry_offset'],state['decoder.geometry_scale']).to(device)
    model.load_state_dict(state);model.eval();digest=state_digest(model)
    runtime,base=load_runtime(binding,device);probe=FixedProbe(binding)
    fmean=runtime['feature_stats']['mean'].to(device);fstd=runtime['feature_stats']['std'].to(device)
    provenance=json.loads((Path(a.data)/'provenance.json').read_text());rows=provenance['records']
    with np.load(binding['rig'],allow_pickle=False) as z:channels=z['channels'].copy()
    results={};out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for role in ('held_sentence','unseen_speaker'):
        data=load(Path(a.data)/(role+'.npz'));per=[];preds={}
        for start in range(0,len(data['motion']),8):
            ids=list(range(start,min(start+8,len(data['motion']))));b,_=batch(data,ids,device)
            values=[runtime['refs'][int(s)] for s in b['speaker_id']]
            refs={k:torch.stack([r[k][0] for r in values]).to(device) for k in ('motion','valid','channel_mask')}
            raw=b['audio_features']*fstd+fmean;raw=torch.where(b['valid'][...,None],raw,0.)
            pred=model.predict(raw,b['h0'],b['b0'],b['valid'].bool(),refs,times=b['times'],seed=47+start)['motion']
            classes=probe.classes(pred,b['valid'].bool())
            for j,i in enumerate(ids):
                mask=b['valid'][j:j+1].bool();gt=b['motion'][j:j+1]
                row={'row':i,'mae':meanabs(pred[j:j+1],gt,mask),
                     'mouth_mae':meanabs(pred[j:j+1,:,14:40],gt[:,:,14:40],mask),
                     'probe_correct':int(classes[j]==b['emotion_id'][j])}
                row.update(motion_stats(pred[j:j+1],mask));per.append(row);preds[i]=pred[j].cpu().numpy()
        results[role]={k:float(np.mean([r[k] for r in per])) for k in per[0] if k!='row'}
        write(out/(role+'_per_clip.json'),per)
        for path in sorted((Path(a.pilot)/'evidence').glob(role+'_emotion_*.npz')):
            with np.load(path,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
            i=int(v['row']);record=next(r for r in rows if r['role']==role and r['row']==i)
            # Deployable views only. Teacher GT reconstruction stays in original
            # evidence for diagnosing capacity and is clearly labelled there.
            v['motions']=np.stack([v['motions'][0],preds[i],v['motions'][3],v['motions'][4],v['motions'][5]])
            v['mode_names']=np.array(['GT','retained_v5','v7_audio_mean','v7_sample','v7_controlled'])
            v['channels']=channels;v['clip_id']=np.array(record['clip_id'])
            np.savez_compressed(out/path.name,**v)
    if state_digest(model)!=digest:raise ValueError('Baseline weights changed')
    report={'schema':'v7_crop_retained_v5_comparison_v1','results':results,'checkpoint_sha256':a.sha256,
            'test_loaded':False,'paper_result':False,'limits':['retained v5 has substantially more training; descriptive comparison only',
            'v5 uses default neutral enrollment; v7 uses independent cross-emotion reference; not a matched reference ablation',
            'native h0/B0 computed before crop; generation uses crop context','probe is whole-clip motion statistics']}
    write(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--pilot',required=True);p.add_argument('--output',required=True)
    p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--device',default='cuda');main(p.parse_args())
