"""Frozen development scoring and factor interventions for the response model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features,classification_metrics
from scripts.train_expression_response import (configure,load_runtime,cache_base,reference_batch,
                                              write,sha,state_digest)
from scripts.arkit_benchmark_report import score_fullface
from scripts.evaluate_vertex_lve import evaluate as vertex_evaluate
from scripts.audit_matched_motion_curves import region_values,REGIONS,RC,longest_valid_span
from scripts.build_validation_gap_table import COEFFICIENT,VERTEX
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
from kinetalk_b0.neutral_amplitude import apply_calibration_tensor


def finite(value):
    if isinstance(value,dict):return {str(k):finite(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [finite(v) for v in value]
    if isinstance(value,np.ndarray):return finite(value.tolist())
    if isinstance(value,np.generic):return finite(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def metrics(pred,b,index,rig=None):
    y=b['motion'][index].cpu();v=b['valid'][index].cpu();ch=b['channel_mask'][index].cpu()
    t=b['times'][index].cpu();pred=pred.cpu()
    report=score_fullface(pred.numpy()[None],{'target52':y.numpy(),'valid':v.numpy(),
                       'times':t.numpy(),'channel_mask':np.broadcast_to(ch.numpy(),y.shape)})
    row={k:report['metrics'][k]['value'] for k in COEFFICIENT}
    for name,ix in REGIONS.items():
        row.update({name+'/'+key:value for key,value in zip(RC,region_values(pred,y,v,ch,ix))})
    # Timing diagnostic in the fixed coefficient convention, not a phoneme score.
    closure=(pred[:,17]<.05)&v;true=(y[:,17]<.05)&v
    row['jaw_closure_frame_f1']=float(2*(closure&true).sum()/(closure.sum()+true.sum()).clamp_min(1))
    if rig is not None:
        vertex=vertex_evaluate(pred.numpy(),y.numpy(),v.numpy(),ch.numpy(),rig['neutral_vertices'],
            rig['blendshape_deltas'],lip_mask=rig['lip_mask'],expression_mask=rig['expression_mask'],
            fdd_mask=rig.get('fdd_mask'),coordinate_scale_to_mm=float(rig['coordinate_scale_to_mm']),
            coordinate_unit=str(rig['coordinate_unit'].item()),coefficient_support=np.array([True]*51+[False]))
        row.update({k:vertex[k] for k in VERTEX})
    return row


def aggregate(rows):
    return {k:float(np.mean([r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]))
            if any(r[k] is not None and np.isfinite(r[k]) for r in rows) else None for k in rows[0]}


def alter_local(p,mode,generator):
    result={k:x.clone() for k,x in p.items()}
    for i,mask in enumerate(p['u_mask']):
        z=p['u_mean'][i,mask]
        if mode=='static':z=z.mean(0,keepdim=True).expand_as(z)
        elif mode=='reverse':z=z.flip(0)
        elif mode=='shuffle':z=z[torch.randperm(len(z),generator=generator,device=z.device)]
        else:raise ValueError(mode)
        result['u_mean'][i,mask]=z
    return result


def export_render_input(path,*,motion,base,prior,posterior,valid,times,channels,clip_id):
    """Display only: keep the longest native observed span, never fill or retime.

    This is the same preregistered display policy as the Phase39 comparison.
    Full-sequence scoring is independent and retains all observed frames.
    """
    valid=valid.detach().cpu();times=times.detach().cpu()
    if valid.dtype!=torch.bool or valid.ndim!=1 or times.shape!=valid.shape:
        raise ValueError('Expected bool valid[T] and native times[T]')
    start,stop=longest_valid_span(valid.numpy());sl=slice(start,stop)
    clock=times[sl]
    if not torch.isfinite(clock).all() or (len(clock)>1 and not torch.isclose(
        clock[1:]-clock[:-1],torch.full_like(clock[1:],.04),rtol=1e-5,atol=1e-7).all()):
        raise ValueError('Display requires contiguous native 25fps observations')
    values=torch.stack([v.detach().cpu()[sl] for v in (motion,base,prior,posterior)])
    channels=channels.detach().cpu()
    if channels.dtype!=torch.bool or channels.shape!=(52,) or not channels.any():
        raise ValueError('Observed bool channel support[52] required')
    if values.shape!=(4,stop-start,52) or not torch.isfinite(values[...,channels]).all():
        raise ValueError('Observed render values must be finite')
    np.savez_compressed(path,channels=np.array(ARKIT_NAMES),times=clock.numpy(),
        valid=valid[sl].numpy(),channel_mask=channels.numpy(),clip_id=clip_id,noise_seed=-1,
        source_start_frame=start,source_stop_frame=stop,source_frames=len(valid),
        display_policy='longest_native_observed_span_earliest_tie_no_fill_no_retime',
        mode_names=np.array(['GT','Neutral_B0','Audio_prior_mean','Posterior_ORACLE']),
        motions=values.numpy())


@torch.no_grad()
def evaluate(binding,run,out,device='cuda',limit=None,runtime=None,neutral_calibration=None):
    device=torch.device(device);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(Path(run)/'final.pt',map_location=device,weights_only=False)
    complete=json.loads((Path(run)/'complete.json').read_text())
    assert sha(Path(run)/'final.pt')==complete['final_sha256']
    if runtime is None:
        data,base=load_runtime(binding,device)
        cache_base(data,base,device,{'train':[],'validation':list(range(limit or len(data['splits']['validation']['valid'])))})
        model=ExpressionResponse(ResponseConfig(**ck['config']),ck['model']['feature_mean'],
                    ck['model']['feature_std'],ck['model']['scales']).to(device)
        model.load_state_dict(ck['model'],strict=True)
    else:model,data,base=runtime
    model.eval();before=state_digest(model);q=data['splits']['validation'];n=limit or len(q['valid'])
    names=data['config']['data']['emotion_classes'];labels=q['emotion_id'][:n]
    support=torch.tensor([True]*51+[False]);probes=[]
    for spec in binding['probes']:
        assert sha(spec['path'])==spec['sha256']
        probe=torch.load(spec['path'],map_location='cpu',weights_only=False)
        assert probe['classes']==names and probe['train_manifest_sha256']==binding['data_manifest_sha256']
        assert not probe.get('test_used_for_selection') and not probe.get('generator_outputs_used_for_fitting')
        assert torch.equal(probe['channel_support'].bool(),support)
        net=MotionEmotionProbe(probe['feature_dim'],probe['hidden'],len(names)).eval()
        net.load_state_dict(probe['model']);probes.append((net,probe.get('feature_mask',torch.ones(probe['feature_dim'],dtype=torch.bool))))
    assert sha(binding['rig'])==binding['rig_sha256']
    with np.load(binding['rig'],allow_pickle=False) as z:rig={k:z[k].copy() for k in z.files}
    mode_names=['prior_mean','posterior_oracle','neutral_b0']
    scored={f'{m}/{policy}':[] for m in mode_names for policy in ('raw','clip_all')}
    features={k:[] for k in scored};gt_features=[];saved=[];g_preds=[]
    interventions={k:[] for k in ('normal','static_u','reverse_u','shuffle_u','wrong_reference','reference_A','reference_B','wrong_matched_audio','gt','gt_static','gt_reverse','gt_shift','gt_gain')}
    diagnostic_ids=set(np.linspace(0,n-1,min(96,n),dtype=int).tolist())
    rng=torch.Generator(device=device).manual_seed(2026)
    display=binding.get('display_clips',{})
    (out/'render_inputs').mkdir(exist_ok=True)
    for sub in torch.arange(n).split(16):
        b=q.batch(sub,device);refs=reference_batch(data,b,device)
        b0 = apply_calibration_tensor(b['b0'], neutral_calibration) if neutral_calibration is not None else b['b0']
        style=model.encode_style(refs)['code'];prior=model.audio_prior(b['audio_features'],b['valid'])
        post=model.motion_posterior(b['motion'],b0,style,b['valid'],b['channel_mask'],b['times'])
        preds={'prior_mean':model.decode(b0,prior,style,b['valid']),
               'posterior_oracle':model.decode(b0,post,style,b['valid']),'neutral_b0':b0}
        g_preds.extend(model.emotion_head(prior['g_mean']).argmax(-1).cpu().tolist())
        for j,i in enumerate(sub.tolist()):
            length=int(q['_lengths'][i]);v=b['valid'][j].cpu();gt=b['motion'][j].cpu()
            gt_features.append(motion_features(gt[:,support],v))
            for name,y in preds.items():
                for policy in ('raw','clip_all'):
                    p=y[j] if policy=='raw' else y[j].clamp(0,1)
                    key=f'{name}/{policy}';scored[key].append(metrics(p,b,j,rig))
                    features[key].append(motion_features(p.cpu()[:,support],v))
            saved.append({k:y[j,:length].cpu() for k,y in preds.items()})
            emotion=next((name for name,cid in display.items() if cid==q['clip_id'][i]),None)
            if emotion:
                export_render_input(out/'render_inputs'/f'{emotion}.npz',motion=gt[:length],
                    base=preds['neutral_b0'][j,:length],prior=preds['prior_mean'][j,:length],
                    posterior=preds['posterior_oracle'][j,:length],valid=v[:length],
                    times=b['times'][j,:length],channels=b['channel_mask'][j],clip_id=q['clip_id'][i])
        # Interventions are diagnostic subsets, never pooled with native scores.
        if diagnostic_ids.intersection(sub.tolist()):
            controls={'normal':preds['prior_mean']}
            for mode in ('static','reverse','shuffle'):
                controls[mode+'_u']=model.decode(b0,alter_local(prior,mode,rng),style,b['valid'])
            for name,ref in [('wrong_reference',reference_batch(data,b,device,wrong_speaker=True)),
                ('reference_A',{k:x[:,:1] for k,x in refs.items()}),('reference_B',{k:x[:,1:2] for k,x in refs.items()})]:
                controls[name]=model.decode(b0,prior,model.encode_style(ref)['code'],b['valid'])
            for j,i in enumerate(sub.tolist()):
                if i not in diagnostic_ids:continue
                for name,p in controls.items():interventions[name].append(metrics(p[j].clamp(0,1),b,j))
                true=b['motion'][j].clone();valid=b['valid'][j];ix=valid.nonzero(as_tuple=True)[0]
                for name,changed in [('gt',true[ix]),('gt_static',true[ix].mean(0,keepdim=True).expand(len(ix),-1)),
                    ('gt_reverse',true[ix].flip(0)),('gt_shift',true[ix].roll(5,0)),('gt_gain',true[ix]*1.25)]:
                    y=true.clone();y[ix]=changed;interventions[name].append(metrics(y,b,j))
                matches=[k for k in range(len(q['valid'])) if k!=i and
                    q['emotion_id'][k]==q['emotion_id'][i] and q['intensity_id'][k]==q['intensity_id'][i]
                    and q['speaker_id'][k]==q['speaker_id'][i] and q['sentence_id'][k]!=q['sentence_id'][i]]
                if matches:
                    k=min(matches,key=lambda k:abs(int(q['_lengths'][k])-int(q['_lengths'][i])))
                    other=q.batch(torch.tensor([k]),device)
                    pp=model.audio_prior(other['audio_features'],other['valid'])
                    # Intentional length mapping for negative control only; never native inference.
                    target={key:value[j:j+1].clone() for key,value in prior.items()}
                    target['g_mean']=pp['g_mean']
                    dest=target['u_mask'][0];source=pp['u_mean'][0,pp['u_mask'][0]].T[None]
                    target['u_mean'][0,dest]=F.interpolate(source,size=int(dest.sum()),mode='linear',align_corners=False)[0].T
                    y=model.decode(b0[j:j+1],target,style[j:j+1],b['valid'][j:j+1])
                    interventions['wrong_matched_audio'].append(metrics(y[0].clamp(0,1),b,j))
        write(out/'state.json',{'status':'scoring','clips':int(sub[-1])+1,'total':n,'test_loaded':False})
        print(json.dumps({'event':'evaluation','clips':int(sub[-1])+1,'total':n}),flush=True)
    def probe_results(fs):
        f=torch.stack(fs)
        return [classification_metrics(labels,net(f[:,mask]).argmax(-1),names) for net,mask in probes]
    report={'schema':'phase41_response_development_v1','clips':n,'test_loaded':False,'default_replaced':False,
            'neutral_calibration_enabled':neutral_calibration is not None,
            'reference_controls_use_same_query_scaffold':True,
            'neutral_scaffold_kind':'TRAIN_calibrated_frozen_B0' if neutral_calibration is not None else 'original_frozen_B0',
            'checkpoint_sha256':sha(Path(run)/'final.pt'),'data_manifest_sha256':binding['data_manifest_sha256'],
            'rig_sha256':binding['rig_sha256'],'training_scope':'TRAIN internal fit subset; compare budgets/data before paper claims',
            'stochastic_sampling':'not yet validated; main inference is prior mean',
            'gt_probes':probe_results(gt_features),
            'audio_global_classifier':classification_metrics(labels,torch.tensor(g_preds),names),
            'results':{k:{'metrics':aggregate(rows),'probes':probe_results(features[k])} for k,rows in scored.items()},
            'interventions':{k:{'n':len(rows),'metrics':aggregate(rows)} for k,rows in interventions.items() if rows},
            'diagnostic_indices':sorted(diagnostic_ids),
            'limits':['posterior_oracle uses query GT, never deployment','jaw closure threshold .05 is rig-specific, not phoneme accuracy',
                      'same-clock output and t-SNE do not prove content disentanglement','wrong reference difference alone does not prove correct identity transfer',
                      'independent phoneme readout and target-style transfer validation remain future stages']}
    assert state_digest(model)==before,'Evaluation mutated trained state'
    write(out/'report.json',finite(report));write(out/'per_clip.json',finite({'clip_ids':q['clip_id'][:n],'results':scored}))
    torch.save({'clip_id':q['clip_id'][:n],'predictions':saved,'test_loaded':False},out/'curves.pt')
    files={str(p.relative_to(out)):dict(sha256=sha(p),size=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name not in ('state.json','manifest.json')}
    write(out/'manifest.json',{'files':files,'test_loaded':False})
    write(out/'state.json',{'status':'complete','clips':n,'report_sha256':sha(out/'report.json'),'test_loaded':False})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--run',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda')
    p.add_argument('--neutral-calibration',type=Path)
    a=p.parse_args();configure(47)
    calibration=json.loads(a.neutral_calibration.read_text()) if a.neutral_calibration else None
    evaluate(json.loads(Path(a.binding).read_text()),a.run,a.output,a.device,a.limit,neutral_calibration=calibration)
