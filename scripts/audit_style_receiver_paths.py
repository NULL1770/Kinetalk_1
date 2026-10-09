"""Bounded TRAIN/internal-held interventions; zero paths are OOD diagnostics."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig,clean
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features,classification_metrics
from scripts.train_expression_response import configure,load_runtime,cache_base,development_fold,reference_batch,write,sha,state_digest


def decode_paths(model,base,g,u,style,valid,*,bias=True,modulation=True):
    d=model.decoder
    if not d.partitioned:raise ValueError('Statistical reference decoder required')
    base=clean(base.detach(),valid);h=clean(d.input(base/model.scales),valid)
    posture,response=style.chunk(2,-1)
    if not modulation:response=torch.zeros_like(response)
    condition=torch.cat((g[:,None].expand(-1,u.shape[1],-1),u,response[:,None].expand(-1,u.shape[1],-1)),-1)
    for block,mod in zip(d.blocks,d.modulations):
        gain,shift=mod(condition).chunk(2,-1)
        h=block(clean(h*(1+.1*gain.tanh())+.1*shift,valid),valid)
    offset=d.bias(posture)[:,None] if bias else 0.
    return clean(base+(d.output(h)+offset)*model.scales,valid)


def select(q,ids,count=12):
    chosen=[]
    for e in range(8):
        pool=[i for i in ids if int(q['emotion_id'][i])==e]
        chosen.extend(pool[j] for j in np.linspace(0,len(pool)-1,min(count,len(pool)),dtype=int))
    return sorted(chosen)


def measures(y,b,j,scales):
    mask=b['valid'][j];channels=b['channel_mask'][j];gt=b['motion'][j]
    delta=(y-gt)/scales;obs=mask[:,None]&channels[None]
    adj=mask[1:]&mask[:-1]&torch.isclose(b['times'][j,1:]-b['times'][j,:-1],torch.full_like(b['times'][j,1:],.04),rtol=1e-4,atol=1e-7)
    am=adj[:,None]&channels[None]
    jaw=y[mask,17];target=gt[mask,17];a=jaw-jaw.mean();c=target-target.mean()
    return dict(position=float(delta[obs].square().mean()),velocity=float((delta[1:]-delta[:-1])[am].square().mean()),
        jaw_range=float(torch.quantile(jaw,.95)-torch.quantile(jaw,.05)),jaw_gt_range=float(torch.quantile(target,.95)-torch.quantile(target,.05)),
        jaw_corr=float((a*c).sum()/(a.square().sum()*c.square().sum()).sqrt().clamp_min(1e-12)),
        jaw_mean=float(jaw.mean()),jaw_gt_mean=float(target.mean()),outside_fraction=float(((y<0)|(y>1))[obs].float().mean()))


@torch.no_grad()
def run(a):
    configure(47);device=torch.device(a.device);out=Path(a.output);out.mkdir(exist_ok=False)
    binding=json.loads(Path(a.binding).read_text());data,base=load_runtime(binding,device);q=data['splits']['train'];fold=development_fold(q,data['fit_sids'])
    chosen={r:select(q,fold[r]) for r in ('train','speaker_dev','sentence_dev')}
    selected=set(sum(chosen.values(),[]));canonical=sorted({j for i in selected for j in range(i//32*32,min((i//32+1)*32,len(q['_lengths'])))})
    cache_base(data,base,device,{'train':canonical,'validation':[]});names=data['config']['data']['emotion_classes'];support=torch.tensor([True]*51+[False]);probes=[]
    for spec in binding['probes']:
        assert sha(spec['path'])==spec['sha256'];z=torch.load(spec['path'],map_location='cpu',weights_only=False)
        assert z['classes']==names and torch.equal(z['channel_support'].bool(),support)
        net=MotionEmotionProbe(z['feature_dim'],z['hidden'],len(names)).eval();net.load_state_dict(z['model'])
        probes.append((net,z.get('feature_mask',torch.ones(z['feature_dim'],dtype=torch.bool))))
    report=dict(schema='phase53_receiver_path_diagnostic_v1',test_loaded=False,validation_inference=False,
        limits=['Zeroing trained paths is OOD; not a trained ablation or deployable candidate.','TRAIN probes can be optimistic; do not compare to full validation.','Jaw range is p95-p05; canonical B0 batch32.'],
        selected={r:[q['clip_id'][i] for i in ids] for r,ids in chosen.items()},results={})
    for label,spec in binding['diagnostic_checkpoints'].items():
        assert sha(spec['path'])==spec['sha256'];ck=torch.load(spec['path'],map_location=device,weights_only=False)
        m=ExpressionResponse(ResponseConfig(**ck['config']),ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(device)
        m.load_state_dict(ck['model'],strict=True);m.eval();before=state_digest(m);result={}
        for role,ids in chosen.items():
            rows={};features={};labels=[];ground=[];saturation=[]
            for sub in torch.tensor(ids).split(16):
                b=q.batch(sub,device);refs=reference_batch(data,b,device);s=m.encode_style(refs)['code'];p=m.audio_prior(b['audio_features'],b['valid']);g,u=m.conditions(p,b['valid'])
                y=m.decode(b['b0'],p,s,b['valid']);torch.testing.assert_close(y,decode_paths(m,b['b0'],g,u,s,b['valid']),rtol=0,atol=0)
                outputs={'normal':y,'no_bias':decode_paths(m,b['b0'],g,u,s,b['valid'],bias=False),
                    'no_style_modulation':decode_paths(m,b['b0'],g,u,s,b['valid'],modulation=False),
                    'neither_style_path':decode_paths(m,b['b0'],g,u,s,b['valid'],bias=False,modulation=False),
                    'zero_g':decode_paths(m,b['b0'],g*0,u,s,b['valid']),
                    'zero_u':decode_paths(m,b['b0'],g,u*0,s,b['valid'])}
                cond=torch.cat((g[:,None].expand(-1,u.shape[1],-1),u,s[:,None,m.cfg.style_dim//2:].expand(-1,u.shape[1],-1)),-1)
                for mod in m.decoder.modulations:
                    z=mod(cond)[...,:m.cfg.decoder_hidden][b['valid']]
                    st=F.linear(s[:,m.cfg.style_dim//2:],mod.weight[:,m.cfg.global_dim+m.cfg.local_dim:])
                    saturation.append(dict(gain_saturated=float((z.abs()>2).float().mean()),style_pre_tanh_rms=float(st.square().mean().sqrt())))
                labels.extend(b['emotion_id'].cpu().tolist())
                for j in range(len(sub)):
                    v=b['valid'][j].cpu();ground.append(motion_features(b['motion'][j].cpu()[:,support],v))
                    for mode,pred in outputs.items():
                        for policy in ('raw','clip'):
                            key=mode+'/'+policy;z=pred[j] if policy=='raw' else pred[j].clamp(0,1)
                            rows.setdefault(key,[]).append(measures(z,b,j,m.scales))
                            features.setdefault(key,[]).append(motion_features(z.cpu()[:,support],v))
            labels=torch.tensor(labels)
            def f1(fs):
                f=torch.stack(fs);return [classification_metrics(labels,n(f[:,mask]).argmax(-1),names) for n,mask in probes]
            def avg(items):return {k:float(np.mean([x[k] for x in items])) for k in items[0]}
            result[role]=dict(clips=len(ids),gt_probes=f1(ground),saturation=avg(saturation),
                modes={k:dict(metrics=avg(v),probes=f1(features[k]),by_emotion={names[e]:avg([r for r,c in zip(v,labels.tolist()) if c==e]) for e in range(8) if e in labels.tolist()}) for k,v in rows.items()})
            write(out/'progress.json',dict(checkpoint=label,role=role));print(json.dumps(dict(checkpoint=label,role=role,complete=True)),flush=True)
        assert state_digest(m)==before;report['results'][label]=dict(checkpoint=spec,roles=result)
        write(out/'report.json',report)
    write(out/'complete.json',dict(passed=True,report_sha256=sha(out/'report.json'),test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');run(p.parse_args())
