"""Frozen g/u attribution diagnosis; no training, no checkpoint selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig
from scripts.train_expression_response import configure,load_runtime,cache_base,write,sha,reference_batch,state_digest
from scripts.evaluate_expression_response import metrics,aggregate,finite
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features,classification_metrics


def swap(q,p,which):
    out={k:v for k,v in p.items()}
    if which=='q_g_p_u':
        for k in ('g_mean','g_logvar'):out[k]=q[k]
    elif which=='p_g_q_u':
        for k in ('u_mean','u_logvar','u_mask'):out[k]=q[k]
    else:raise ValueError(which)
    return out


def distribution_statistics(q,p,raw,lower=-6.,upper=2.):
    """Per-clip, per-dimension calibration; missing tokens never count."""
    result={}
    for part in ('g','u'):
        qm,pm=q[part+'_mean'],p[part+'_mean'];qv,pv=q[part+'_logvar'],p[part+'_logvar']
        mask=p['u_mask'] if part=='u' else torch.ones(pm.shape[:1],dtype=torch.bool,device=pm.device)
        if part=='g':qm,pm,qv,pv,mask=qm[:,None],pm[:,None],qv[:,None],pv[:,None],mask[:,None]
        weight=mask[...,None];count=weight.sum(1).clamp_min(1)
        reduce=lambda value:torch.where(weight,value,0.).sum(1)/count
        error=(qm-pm).square()
        raw_var=raw[part].chunk(2,-1)[1]
        if part=='g':raw_var=raw_var[:,None]
        result[part]={
            'mean_error_squared':reduce(error),
            'posterior_variance':reduce(qv.exp()),'prior_variance':reduce(pv.exp()),
            'kl_mean_component':reduce(.5*error*(-pv).exp()),
            'kl_variance_component':reduce(.5*(pv-qv+(qv-pv).exp()-1.)),
            'prior_raw_high_clamped_fraction':reduce((raw_var>upper).float()),
            'prior_raw_low_clamped_fraction':reduce((raw_var<lower).float()),
            'prior_logvar':reduce(pv),'posterior_logvar':reduce(qv),
            'coverage_95':reduce(((qm-pm).abs()<=1.96*pv.exp().sqrt()).float()),
            'prior_mean_squared':reduce(pm.square()),'posterior_mean_squared':reduce(qm.square())}
    return result


@torch.no_grad()
def run(binding,checkpoint,output,device='cuda'):
    configure(47);device=torch.device(device);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data,base=load_runtime(binding,device)
    cache_base(data,base,device,{'train':[],'validation':list(range(len(data['splits']['validation']['valid'])))})
    ck=torch.load(checkpoint,map_location=device,weights_only=False);cfg=ResponseConfig(**ck['config'])
    model=ExpressionResponse(cfg,ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'],strict=True);model.eval();before=state_digest(model)
    qdata=data['splits']['validation'];n=len(qdata['valid']);labels=qdata['emotion_id'];names=data['config']['data']['emotion_classes']
    support=torch.tensor([True]*51+[False]);modes=('prior_mean','posterior_oracle','q_g_p_u','p_g_q_u')
    rows={k:[] for k in modes};features={k:[] for k in modes};g_preds=[];stats={'g':{},'u':{}}
    raw={}
    hg=model.prior.global_head.register_forward_hook(lambda m,inp,out:raw.update(g=out))
    hu=model.prior.local_head.register_forward_hook(lambda m,inp,out:raw.update(u=out))
    probes=[]
    for spec in binding['probes']:
        assert sha(spec['path'])==spec['sha256']
        probe=torch.load(spec['path'],map_location='cpu',weights_only=False)
        assert probe['classes']==names and probe['train_manifest_sha256']==binding['data_manifest_sha256']
        assert not probe.get('test_used_for_selection') and not probe.get('generator_outputs_used_for_fitting')
        net=MotionEmotionProbe(probe['feature_dim'],probe['hidden'],len(names)).eval();net.load_state_dict(probe['model'])
        probes.append((net,probe.get('feature_mask',torch.ones(probe['feature_dim'],dtype=torch.bool))))
    for sub in torch.arange(n).split(16):
        b=qdata.batch(sub,device);refs=reference_batch(data,b,device);style=model.encode_style(refs)['code'];p=model.audio_prior(b['audio_features'],b['valid']);q=model.motion_posterior(b['motion'],b['b0'],style,b['valid'],b['channel_mask'],b['times'])
        g_preds.extend(model.emotion_head(p['g_mean']).argmax(-1).cpu().tolist())
        latent=distribution_statistics(q,p,raw,cfg.logvar_min,cfg.logvar_max)
        for part,values in latent.items():
            for key,value in values.items():stats[part].setdefault(key,[]).extend(value.cpu().tolist())
        dist={'prior_mean':p,'posterior_oracle':q,'q_g_p_u':swap(q,p,'q_g_p_u'),'p_g_q_u':swap(q,p,'p_g_q_u')}
        for mode,d in dist.items():
            pred=model.decode(b['b0'],d,style,b['valid'])
            for j,i in enumerate(sub.tolist()):
                value=pred[j].clamp(0,1).cpu()
                rows[mode].append(metrics(value,b,j));features[mode].append(motion_features(value[:,support],b['valid'][j].cpu()))
        write(output/'state.json',{'status':'evaluating','clips':int(sub[-1])+1,'total':n,'test_loaded':False})
        if int(sub[0])%160==0:print(json.dumps({'clips':int(sub[-1])+1,'total':n}),flush=True)
    report={'schema':'phase41_factor_swap_v1','checkpoint_sha256':sha(checkpoint),'clips':n,'test_loaded':False,'modes':{},
        'distribution':{part:{key:{'mean':float(np.mean(values)),'per_dim_mean':np.mean(values,axis=0).tolist(),
                                 'per_clip_mean_p90':float(np.quantile(np.mean(values,axis=1),.9))}
                            for key,values in values.items()} for part,values in stats.items()},
        'limits':['Mixed q/p conditions may be off-manifold; swaps localize sensitivity, not causal independence.',
                  'Posterior sees GT and can encode articulation errors. Oracle is never deployment.',
                  'Coverage is q-mean coverage by p variance, not human motion uncertainty calibration.']}
    for mode in modes:
        f=torch.stack(features[mode]);probes_out=[classification_metrics(labels,net(f[:,mask]).argmax(-1),names) for net,mask in probes]
        report['modes'][mode]={'metrics':aggregate(rows[mode]),'probes':probes_out,
            'by_emotion':{name:aggregate([rows[mode][i] for i in (labels==c).nonzero(as_tuple=True)[0].tolist()]) for c,name in enumerate(names)},
            'by_speaker':{str(int(sid)):aggregate([rows[mode][i] for i in (qdata['speaker_id']==sid).nonzero(as_tuple=True)[0].tolist()]) for sid in qdata['speaker_id'].unique()}}
    report['audio_global_probe']=classification_metrics(labels,torch.tensor(g_preds),names)
    assert state_digest(model)==before
    hg.remove();hu.remove()
    write(output/'per_clip.json',finite({'clip_ids':qdata['clip_id'],'modes':rows,'latents':stats}))
    write(output/'report.json',finite(report));write(output/'state.json',{'status':'complete','report_sha256':sha(output/'report.json'),'test_loaded':False,'weights_unchanged':True});print(json.dumps({'complete':True,'report':str(output/'report.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.binding).read_text()),a.checkpoint,a.output,a.device)
