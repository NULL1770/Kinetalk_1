"""Compare actual teacher-free v4 rollout with immutable Phase53 on same clips."""
import argparse,json,hashlib
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v4.unified_model_v4 import KineTalkV4Config,KineTalkV4Model
from kinetalk_b0.models.expression_response import ResponseConfig,ExpressionResponse
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features,classification_metrics
from scripts.train_expression_response import load_runtime,configure,sha,state_digest
from scripts.train_unified_v4 import build_base_cache,attach_base
from scripts.evaluate_expression_response import metrics,aggregate,finite


@torch.no_grad()
def evaluate(a):
    configure(47);device=torch.device(a.device);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    binding=json.loads(Path(a.binding).read_text());data,base=load_runtime(binding,device)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=KineTalkV4Model(KineTalkV4Config(**ck['config']),ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales'],ck['model']['decoder.geometry_jacobian'],ck['model']['decoder.geometry_offset'],ck['model']['decoder.geometry_scale']).to(device)
    model.load_state_dict(ck['model']);model.eval()
    if sha(a.baseline)!=a.baseline_sha256:raise ValueError('Phase53 baseline hash differs')
    old=torch.load(a.baseline,map_location='cpu',weights_only=False)
    baseline=ExpressionResponse(ResponseConfig(**old['config']),old['model']['feature_mean'],old['model']['feature_std'],old['model']['scales']).to(device)
    baseline.load_state_dict(old['model']);baseline.eval()
    before=state_digest(model);old_before=state_digest(baseline);q=data['splits']['validation']
    n=min(a.limit or len(q['_lengths']),len(q['_lengths']));ids=list(range(n))
    cache=build_base_cache(q,base,ids,device)
    support=torch.tensor([True]*51+[False]);names=data['config']['data']['emotion_classes']
    with np.load(binding['rig'],allow_pickle=False) as z:rig={k:z[k].copy() for k in z.files}
    scored={};features={};samples={};render_ids=set(binding['display_clips'].values())
    (out/'render_inputs').mkdir(exist_ok=True)
    for start in range(0,n,8):
        use=ids[start:start+8];b=attach_base(q.batch(torch.tensor(use),device),use,cache)
        values=[data['refs'][int(q['speaker_id'][i])] for i in use]
        if any(q['clip_id'][i] in r['clip_id'] for i,r in zip(use,values)):raise ValueError('Query/reference leakage')
        refs={k:torch.stack([r[k][0] for r in values]).to(device) for k in ('motion','valid','channel_mask')}
        old_refs={k:torch.stack([r[k] for r in values]).to(device) for k in ('motion','valid','channel_mask')}
        content=torch.stack([r['content'] for r in values]).to(device)
        from scripts.train_full_staged import base_forward
        flat=base_forward(base,content.flatten(0,1),old_refs['valid'].flatten(0,1))['b0']
        old_refs['b0']=flat.reshape(len(use),content.shape[1],content.shape[2],52)
        normal=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,times=b['times'],seed=47+start)
        modal=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,times=b['times'],mode='modal')
        p=baseline.audio_prior(b['audio_features'],b['valid'])
        all_style=baseline.encode_style(old_refs)['code']
        a_style=baseline.encode_style({k:v[:,:1] for k,v in old_refs.items()})['code']
        preds={'v4_sample':normal['motion'],'v4_modal':modal['motion'],
            'phase53_default':baseline.decode(b['b0'],p,all_style,b['valid']),
            'phase53_reference_A':baseline.decode(b['b0'],p,a_style,b['valid'])}
        for j,i in enumerate(use):
            for name,y in preds.items():
                for policy in ('raw','clip_all'):
                    key=name+'/'+policy;prediction=y[j] if policy=='raw' else y[j].clamp(0,1)
                    scored.setdefault(key,[]).append(metrics(prediction,b,j,rig))
                    features.setdefault(key,[]).append(motion_features(prediction.cpu()[:,support],b['valid'][j].cpu()))
            if q['clip_id'][i] in render_ids:
                length=int(q['_lengths'][i]);cid=q['clip_id'][i]
                np.savez_compressed(out/'render_inputs'/(cid+'.npz'),motions=torch.stack((b['motion'][j],b['b0'][j],preds['phase53_default'][j],preds['v4_sample'][j])).cpu().numpy()[:,:length],
                    mode_names=np.array(['GT','B0','Phase53','v4_sample']),times=b['times'][j,:length].cpu().numpy(),valid=b['valid'][j,:length].cpu().numpy(),channels=rig['channels'],channel_mask=b['channel_mask'][j].cpu().numpy(),clip_id=cid,noise_seed=47+start)
        state={'status':'scoring','clips':start+len(use),'total':n}
        (out/'state.json').write_text(json.dumps(state));print(json.dumps(state),flush=True)
    probe_specs=[]
    for spec in binding['probes']:
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Probe binding differs')
        saved=torch.load(spec['path'],map_location='cpu',weights_only=False)
        net=MotionEmotionProbe(saved['feature_dim'],saved['hidden'],len(names)).eval();net.load_state_dict(saved['model'])
        mask=saved.get('feature_mask',torch.ones(saved['feature_dim'],dtype=torch.bool));probe_specs.append((net,mask))
    result={}
    for key,rows in scored.items():
        f=torch.stack(features[key]);probes=[classification_metrics(q['emotion_id'][:n],net(f[:,mask]).argmax(-1),names) for net,mask in probe_specs]
        result[key]={'metrics':aggregate(rows),'probes':probes}
    if state_digest(model)!=before or state_digest(baseline)!=old_before:raise ValueError('Scoring changed model')
    report={'schema':'unified_v4_phase53_development_v1','clips':n,'test_loaded':False,'checkpoint_sha256':sha(a.checkpoint),
        'baseline_sha256':sha(a.baseline),'results':result,'training_protocol':ck['protocol'],
        'comparison_limits':['v4 training budget differs from Phase53; descriptive development comparison only',
        'motion probe F1 is whole-clip statistics, not temporal emotional truth','jaw closure is a coefficient proxy, not calibrated phonetic contact',
        'sample rollout and modal schedule are separately scored; no best-of-N selection','rig coordinates are provisional geometric observers; three contributions remain unverified']}
    (out/'report.json').write_text(json.dumps(finite(report),ensure_ascii=False,indent=2),encoding='utf8')
    (out/'complete.json').write_text(json.dumps({'status':'complete','clips':n,'checkpoint_sha256':sha(a.checkpoint)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline',default='/root/kinetalk_phase53_receiver_20261009/style_only/seed47/final.pt')
    p.add_argument('--baseline-sha256',default='80a7ca69b0f868bad6361381e1ca8e4e168ca227a0548221f8efa71b42169c41')
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,default=0);evaluate(p.parse_args())

