"""Validation-only factor interventions for the deployed unified v2 model.

This is a conditional sensitivity/controllability diagnostic.  It is not a
training ablation, style-reference reconstruction score, or phonetic contact
benchmark.  Query motion never enters generation.  Audio, B0, clip clock and a
per-query seed remain fixed; changed conditions may change sampled events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

SCHEMA = "v2-factor-interventions-validation-v1"
AXES = {
    "smile_minus_frown": {"mouthSmileLeft":.5,"mouthSmileRight":.5,"mouthFrownLeft":-.5,"mouthFrownRight":-.5},
    "mouth_tension": {"mouthStretchLeft":.25,"mouthStretchRight":.25,"mouthPressLeft":.25,"mouthPressRight":.25},
    "brow_lift": {"browInnerUp":.5,"browOuterUpLeft":.25,"browOuterUpRight":.25},
    "brow_contraction": {"browDownLeft":.5,"browDownRight":.5},
    "eye_open_minus_squint": {"eyeWideLeft":.5,"eyeWideRight":.5,"eyeSquintLeft":-.5,"eyeSquintRight":-.5},
    "cheek_nose": {"cheekPuff":.2,"cheekSquintLeft":.2,"cheekSquintRight":.2,"noseSneerLeft":.2,"noseSneerRight":.2},
    "dimple": {"mouthDimpleLeft":.5,"mouthDimpleRight":.5},
    "lip_shrug": {"mouthShrugLower":.5,"mouthShrugUpper":.5},
    "vertical_lip_deformation": {"mouthLowerDownLeft":.25,"mouthLowerDownRight":.25,"mouthUpperUpLeft":.25,"mouthUpperUpRight":.25},
    "rounding": {"mouthFunnel":.5,"mouthPucker":.5},
}
UPPER_PREFIXES=("eye","brow","cheek","nose")
EMOTION_ALIASES={"happy":("happy","happiness"),"sad":("sad","sadness"),"angry":("angry","anger")}


def finite(obj:Any)->Any:
    if isinstance(obj,dict):return {str(k):finite(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [finite(v) for v in obj]
    if isinstance(obj,np.ndarray):return finite(obj.tolist())
    if isinstance(obj,np.generic):return finite(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj


def write_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(finite(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def per_query_seed(clip_id:str,seed:int)->int:
    return (seed+int(hashlib.sha256(clip_id.encode()).hexdigest()[:8],16)) % (2**31-1)


def balanced_validation_ids(q,limit:int,seed:int)->list[int]:
    """Round-robin emotion then speaker; prevent first-speaker truncation bias."""
    rng=random.Random(seed);groups=defaultdict(lambda:defaultdict(list))
    for i in range(len(q['_lengths'])):
        groups[int(q['emotion_id'][i])][int(q['speaker_id'][i])].append(i)
    orders={}
    for emotion,speakers in groups.items():
        order=list(speakers);rng.shuffle(order);orders[emotion]=order
        for values in speakers.values():rng.shuffle(values)
    positions=defaultdict(int);chosen=[];emotions=sorted(groups)
    goal=min(limit or len(q['_lengths']),len(q['_lengths']))
    while len(chosen)<goal:
        added=False
        for emotion in emotions:
            order=orders[emotion]
            for _ in range(len(order)):
                sid=order[positions[emotion]%len(order)];positions[emotion]+=1
                if groups[emotion][sid]:
                    chosen.append(groups[emotion][sid].pop());added=True;break
            if len(chosen)>=goal:break
        if not added:break
    return chosen


def validate_reference(query_id:str,query_sentence:str,ref:dict,slot:int=0)->None:
    if query_id in ref['clip_id']:
        raise ValueError('Query/reference clip leakage')
    if query_sentence in ref.get('sentence_id',[]):
        raise ValueError('Query/reference sentence overlaps')
    if not ref['valid'][slot].any():raise ValueError('Independent reference has no valid frames')


def donor_speaker(sid:int,pool:list[int],seed:int,clip_id:str)->int:
    alternatives=[s for s in sorted(pool) if s!=sid]
    if not alternatives:raise ValueError('Cross-person reference intervention requires another speaker')
    return random.Random(per_query_seed(clip_id,seed)).choice(alternatives)


def emotion_indices(names:list[str])->dict[str,int]:
    lowered=[n.lower() for n in names];out={}
    for canonical,aliases in EMOTION_ALIASES.items():
        matches=[lowered.index(alias) for alias in aliases if alias in lowered]
        if len(set(matches))!=1:raise ValueError(f'Missing/ambiguous class for {canonical}: {names}')
        out[canonical]=matches[0]
    return out


def _runs(valid,times):
    from scripts.evaluate_v2_dynamics import contiguous_runs
    return contiguous_runs(np.asarray(valid,bool),np.asarray(times,float))[0]


def _corr(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)<3 or np.std(x)<1e-7 or np.std(y)<1e-7:return None
    return float(np.corrcoef(x,y)[0,1])


def motion_descriptors(motion,base,valid,times,channel_mask,channels)->dict:
    from scripts.evaluate_v2_dynamics import regional_dynamics
    x,b=np.asarray(motion,float),np.asarray(base,float);valid=np.asarray(valid,bool)
    mask=np.asarray(channel_mask,bool);names=[str(n).lower() for n in channels]
    if x.shape!=b.shape or x.ndim!=2 or x.shape[1]!=52 or valid.shape!=(len(x),) or mask.shape!=(52,):
        raise ValueError('Descriptor input requires aligned motion/base[T,52],valid[T],channel_mask[52]')
    if not valid.any():raise ValueError('No valid query frame')
    if not np.isfinite(x[valid][:,mask]).all():raise ValueError('Nonfinite observed output')
    residual=x-b
    upper=[i for i,n in enumerate(names) if n.startswith(UPPER_PREFIXES) and mask[i]]
    mouth=[i for i,n in enumerate(names) if n.startswith(('mouth','jaw')) and mask[i]]
    result={'axes':{},'asymmetry':{},'upper_residual_rms':float(np.sqrt((residual[valid][:,upper]**2).mean())) if upper else None,
            'mouth_residual_rms':float(np.sqrt((residual[valid][:,mouth]**2).mean())) if mouth else None,
            'upper_dynamics':regional_dynamics(x,valid,times,mask,upper),
            'mouth_dynamics':regional_dynamics(x,valid,times,mask,mouth)}
    for axis,weights in AXES.items():
        indices=[names.index(n.lower()) for n in weights]
        if not mask[indices].all():
            result['axes'][axis]={'observed':False};continue
        w=np.array(list(weights.values()))
        values=x[valid][:,indices]@w;r=residual[valid][:,indices]@w
        result['axes'][axis]={'observed':True,'mean':float(values.mean()),'residual_mean':float(r.mean()),
            'residual_rms':float(np.sqrt(np.square(r).mean())),
            'residual_range_p95_p05':float(np.percentile(r,95)-np.percentile(r,5))}
    for left,n in enumerate(names):
        if not n.endswith('left'):continue
        other=n[:-4]+'right'
        if other not in names:continue
        right=names.index(other)
        if not(mask[left] and mask[right]):continue
        delta=x[valid,left]-x[valid,right];r=residual[valid,left]-residual[valid,right]
        result['asymmetry'][n[:-4]]={'signed_mean':float(delta.mean()),'mean_abs':float(np.abs(delta).mean()),
                                 'residual_signed_mean':float(r.mean()),'residual_mean_abs':float(np.abs(r).mean())}
    return result


def content_change(a,b,valid,times,channel_mask,channels)->dict:
    """Paired phonetic proxies; amplitude is allowed to change legitimately."""
    a,b=np.asarray(a,float),np.asarray(b,float);valid=np.asarray(valid,bool);mask=np.asarray(channel_mask,bool)
    names=[str(n).lower() for n in channels];runs=_runs(valid,times)
    jaw=names.index('jawopen');mouth=[i for i,n in enumerate(names) if n.startswith(('jaw','mouth')) and mask[i]]
    if not mouth:return {'observed':False}
    result={'observed':True,'mouth_coefficient_pairwise_mae':float(np.abs(a[valid][:,mouth]-b[valid][:,mouth]).mean()),
            'jaw_observed':bool(mask[jaw]),'jaw_zero_lag_correlation':None,'jaw_derivative_correlation':None,
            'jaw_closure_proxy_flip_fraction_at_05':None,'jaw_range_ratio':None,
            'best_jaw_lag_frames_diagnostic':None}
    if not mask[jaw]:return result
    ja,jb=a[valid,jaw],b[valid,jaw]
    jaw_range_a=float(np.percentile(ja,95)-np.percentile(ja,5))
    jaw_range_b=float(np.percentile(jb,95)-np.percentile(jb,5))
    result.update(jaw_zero_lag_correlation=_corr(ja,jb),
        jaw_closure_proxy_flip_fraction_at_05=float(np.mean((ja<=.05)!=(jb<=.05))),
        jaw_range_ratio=jaw_range_b/jaw_range_a if jaw_range_a>1e-7 else None)
    da,db=[],[]
    for lo,hi in runs:
        if hi-lo>=2:
            da.extend(np.diff(a[lo:hi,jaw]));db.extend(np.diff(b[lo:hi,jaw]))
    result['jaw_derivative_correlation']=_corr(da,db)
    lags={}
    for lag in range(-3,4):
        xs,ys=[],[]
        for lo,hi in runs:
            if hi-lo<=abs(lag)+2:continue
            aa=a[lo:hi,jaw];bb=b[lo:hi,jaw]
            if lag<0:aa,bb=aa[-lag:],bb[:lag]
            elif lag>0:aa,bb=aa[:-lag],bb[lag:]
            xs.extend(aa);ys.extend(bb)
        score=_corr(xs,ys)
        if score is not None:lags[lag]=score
    if lags:result['best_jaw_lag_frames_diagnostic']=max(lags,key=lambda lag:(lags[lag],-abs(lag)))
    return result


def descriptor_delta(a:dict,b:dict)->dict:
    """Numeric post-minus-pre changes; this is not a reference style error."""
    out={}
    for key in ('upper_residual_rms','mouth_residual_rms'):
        if a.get(key) is not None and b.get(key) is not None:out[key+'_delta']=b[key]-a[key]
    out['axes']={axis:{key:b['axes'][axis][key]-a['axes'][axis][key]
                for key in ('mean','residual_mean','residual_rms','residual_range_p95_p05')}
                for axis in AXES if a['axes'][axis].get('observed') and b['axes'][axis].get('observed')}
    out['asymmetry']={axis:{key:b['asymmetry'][axis][key]-a['asymmetry'][axis][key] for key in a['asymmetry'][axis]}
                      for axis in a['asymmetry'] if axis in b['asymmetry']}
    return out


def schedule_change(a,b,valid)->dict:
    a,b=np.asarray(a,float).reshape(-1,4,3),np.asarray(b,float).reshape(-1,4,3);valid=np.asarray(valid,bool)
    return {'events_bitwise_equal':bool(np.array_equal(a,b)),
            'active_group_disagreement_fraction':float(np.mean((a[valid,:,0]>0)!=(b[valid,:,0]>0))),
            'event_tensor_pairwise_mae':float(np.abs(a[valid]-b[valid]).mean()),
            'same_seed_does_not_guarantee_same_conditional_event_schedule':True}


def _mean_nested(rows:list[dict])->dict:
    out={}
    if not rows:return out
    for key in sorted(set().union(*(r.keys() for r in rows))):
        vals=[r[key] for r in rows if key in r]
        dictionaries=[v for v in vals if isinstance(v,dict)]
        if dictionaries:out[key]=_mean_nested(dictionaries)
        else:
            booleans=[bool(v) for v in vals if isinstance(v,(bool,np.bool_))]
            if booleans:
                out[key]={'true_count':sum(booleans),'true_fraction':float(np.mean(booleans)),'n':len(booleans)}
                continue
            nums=[float(v) for v in vals if isinstance(v,(float,int,np.number)) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v)]
            if nums:out[key]={'mean':float(np.mean(nums)),'median':float(np.median(nums)),'n':len(nums)}
    return out


def flatten_numbers(value:dict,prefix='')->dict:
    out={}
    for key,v in value.items():
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(v,dict):out.update(flatten_numbers(v,name))
        elif isinstance(v,(int,float,np.number)) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v):out[name]=float(v)
    return out


def directional_control(delta:dict,emotion:str)->dict:
    axes=delta['axes']
    if emotion=='happy':choices={'smile_minus_frown_increase':('smile_minus_frown',1.)}
    elif emotion=='sad':choices={'frown_minus_smile_increase':('smile_minus_frown',-1.),'brow_lift_change':('brow_lift',1.)}
    elif emotion=='angry':choices={'brow_contraction_increase':('brow_contraction',1.),'mouth_tension_increase':('mouth_tension',1.)}
    else:raise ValueError('Unknown requested control')
    return {'diagnostic_only_not_emotion_accuracy':True,
            **{name:sign*axes[axis]['mean'] for name,(axis,sign) in choices.items() if axis in axes}}


@torch.no_grad()
def predict_variants(model,batch,reference_a,reference_b,controls:dict[str,int],*,seed:int,mode='sample')->dict:
    """No motion targets accepted; direct predict calls for all deployable cases."""
    kwargs={'times':batch['times'],'seed':seed,'mode':mode}
    positional=(batch['audio_features'],batch['h0'],batch['b0'],batch['valid'])
    outputs={'auto_A':model.predict(*positional,reference_a,**kwargs),
             'auto_other_reference':model.predict(*positional,reference_b,**kwargs)}
    for name,index in controls.items():
        emotion=F.one_hot(torch.tensor([index],device=batch['valid'].device),model.cfg.classes).float()
        outputs[name+'_A']=model.predict(*positional,reference_a,emotion=emotion,**kwargs)
        outputs[name+'_other_reference']=model.predict(*positional,reference_b,emotion=emotion,**kwargs)
    normal=outputs['auto_A'];student=normal['student'];style=normal['style'];events=normal['schedule']['events'];probs=student['emotion_logits'].softmax(-1)
    # Fixed-baseline-schedule decoder diagnostics isolate style from event
    # probability changes; no assertion that ordinary reference swaps fix events.
    fixed=model.decode(batch['h0'],batch['b0'],batch['valid'],student['state'],probs,
                       outputs['auto_other_reference']['style'],events)
    outputs['decoder_only_reference_swap_fixed_schedule']={**fixed,'student':student,
                     'style':outputs['auto_other_reference']['style'],'schedule':normal['schedule']}
    zero_events=torch.zeros_like(events)
    event_disabled=model.decode(batch['h0'],batch['b0'],batch['valid'],student['state'],probs,style,zero_events)
    outputs['inference_sensitivity_no_events']={**event_disabled,'student':student,'style':style,
                                               'schedule':{'events':zero_events}}
    zero_style=torch.zeros_like(style)
    style_disabled=model.decode(batch['h0'],batch['b0'],batch['valid'],student['state'],probs,zero_style,events)
    outputs['inference_sensitivity_zero_style_fixed_schedule']={**style_disabled,'student':student,
                                               'style':zero_style,'schedule':normal['schedule']}
    return outputs


@torch.no_grad()
def evaluate(args):
    from kinetalk_v2.unified_model import UnifiedConfig,UnifiedExpressionModel
    from scripts.train_expression_response import configure,load_runtime,sha,state_digest
    from scripts.train_unified_v2 import build_base_cache,attach_base
    configure(args.selection_seed);device=torch.device(args.device);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    binding=json.loads(Path(args.binding).read_text(encoding='utf-8'))
    data,base=load_runtime(binding,device);q=data['splits']['validation']
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('binding_sha256') and ck['binding_sha256']!=sha(args.binding):raise ValueError('Checkpoint/runtime binding differs')
    model=UnifiedExpressionModel(UnifiedConfig(**ck['config']),ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
    initial=(state_digest(model),state_digest(base.stage1))
    if sha(binding['rig'])!=binding['rig_sha256']:raise ValueError('Rig binding differs')
    with np.load(binding['rig'],allow_pickle=False) as rig:channels=[str(x) for x in rig['channels']]
    names=data['config']['data']['emotion_classes'];controls=emotion_indices(names)
    ids=balanced_validation_ids(q,args.limit,args.selection_seed)
    if not ids:raise ValueError('No validation selection')
    cache=build_base_cache(q,base,ids,device)
    pool=sorted(set(int(q['speaker_id'][i]) for i in range(len(q['_lengths']))))
    if set(pool)-set(data['refs']):raise ValueError('Validation reference pool incomplete')
    (root/'render_inputs').mkdir(exist_ok=True)
    rows=[]
    for ordinal,i in enumerate(ids):
        sid=int(q['speaker_id'][i]);cid=q['clip_id'][i];sentence=q['sentence_id'][i]
        donor=donor_speaker(sid,pool,args.selection_seed,cid)
        ra,rb=data['refs'][sid],data['refs'][donor]
        validate_reference(cid,sentence,ra);validate_reference(cid,sentence,rb)
        refs=[{k:r[k][0:1].to(device) for k in ('motion','valid','channel_mask')} for r in (ra,rb)]
        batch=attach_base(q.batch(torch.tensor([i]),device,keys=('audio_features','content','valid','channel_mask','times')), [i],cache)
        # The intervention generation batch intentionally never loads query GT motion.
        seed=per_query_seed(cid,args.seed)
        outputs=predict_variants(model,batch,*refs,controls,seed=seed,mode=args.mode)
        normal=outputs['auto_A'];valid=batch['valid'][0].cpu().numpy();times=batch['times'][0].cpu().numpy();cm=batch['channel_mask'][0].cpu().numpy();b0=batch['b0'][0].cpu().numpy()
        desc={name:motion_descriptors(out['motion'][0].cpu().numpy(),b0,valid,times,cm,channels) for name,out in outputs.items()}
        baseline_state=normal['student']['state'];baseline_logits=normal['student']['emotion_logits']
        comparisons={}
        for name,output in outputs.items():
            if name=='auto_A':continue
            # A requested emotion is compared to auto under the same reference.
            # Otherwise donor emotion effects would include the reference swap.
            baseline_name='auto_other_reference' if name.endswith('_other_reference') and name.split('_')[0] in controls else 'auto_A'
            comparison_baseline=outputs[baseline_name]
            state_change=float((output['student']['state']-baseline_state).abs().max())
            logit_change=float((output['student']['emotion_logits']-baseline_logits).abs().max())
            comparison={'kind':'inference_sensitivity_not_trained_ablation' if name.startswith('inference_sensitivity') else
                        ('decoder_diagnostic_fixed_baseline_schedule' if name.startswith('decoder_only') else 'deployable_condition_intervention'),
                'baseline_variant':baseline_name,
                'content_proxies':content_change(comparison_baseline['motion'][0].cpu().numpy(),output['motion'][0].cpu().numpy(),valid,times,cm,channels),
                'descriptor_delta':descriptor_delta(desc[baseline_name],desc[name]),
                'student_state_max_abs_change':state_change,'student_emotion_logits_max_abs_change':logit_change,
                'event_schedule':schedule_change(comparison_baseline['schedule']['events'][0].cpu().numpy(),output['schedule']['events'][0].cpu().numpy(),valid)}
            if name.split('_')[0] in controls:comparison['directional_response']=directional_control(comparison['descriptor_delta'],name.split('_')[0])
            comparisons[name]=comparison
        # Output response interaction: emotion effect under reference B minus
        # emotion effect under A. No paired reference-GT style reconstruction.
        interaction={};reference_effect={}
        for emotion in controls:
            da=descriptor_delta(desc['auto_A'],desc[emotion+'_A']);db=descriptor_delta(desc['auto_other_reference'],desc[emotion+'_other_reference'])
            fa,fb=flatten_numbers(da),flatten_numbers(db)
            interaction[emotion]={k:fb[k]-fa[k] for k in fa.keys()&fb.keys()}
            out_a,out_b=outputs[emotion+'_A'],outputs[emotion+'_other_reference']
            reference_effect[emotion]={'baseline_variant':emotion+'_A','intervention_variant':emotion+'_other_reference',
                'content_proxies':content_change(out_a['motion'][0].cpu().numpy(),out_b['motion'][0].cpu().numpy(),valid,times,cm,channels),
                'descriptor_delta':descriptor_delta(desc[emotion+'_A'],desc[emotion+'_other_reference']),
                'event_schedule':schedule_change(out_a['schedule']['events'][0].cpu().numpy(),out_b['schedule']['events'][0].cpu().numpy(),valid)}
        style_a,style_b=normal['style'][0],outputs['auto_other_reference']['style'][0]
        record={'clip_id':cid,'validation_index':i,'speaker_id':sid,'emotion_id':int(q['emotion_id'][i]),
            'seed':seed,'mode':args.mode,'reference_A':{'clip_id':ra['clip_id'][0],'speaker_id':sid},
            'other_reference':{'clip_id':rb['clip_id'][0],'speaker_id':donor},
            'query_motion_loaded_for_generation':False,'descriptors':desc,'comparisons':comparisons,
            'style_code_rms_change':float((style_b-style_a).square().mean().sqrt()),
            'style_code_cosine':float(F.cosine_similarity(style_a[None],style_b[None])[0]),
            'emotion_response_by_reference_interaction':interaction,
            'reference_effect_at_fixed_requested_emotion':reference_effect,
            'student_audio_state_invariant_across_all_conditions':all(c['student_state_max_abs_change']<=1e-7 and c['student_emotion_logits_max_abs_change']<=1e-7 for c in comparisons.values())}
        rows.append(record)
        if ordinal<args.save_curves:
            modes=list(outputs)
            np.savez_compressed(root/'render_inputs'/f'{cid}.npz',motion=np.stack([outputs[n]['motion'][0].cpu().numpy() for n in modes]),
                                mode_names=np.asarray(modes),b0=b0,times=times,valid=valid,channel_mask=cm,
                                channels=np.asarray(channels),seed=np.int64(seed),reference_ids=np.asarray([ra['clip_id'][0],rb['clip_id'][0]]))
        write_json(root/'state.json',{'status':'scoring','clips':len(rows),'total':len(ids)})
        print(json.dumps({'event':'factor_intervention_clip','clips':len(rows),'total':len(ids),'clip_id':cid}),flush=True)
    if (state_digest(model),state_digest(base.stage1))!=initial:raise ValueError('Intervention diagnostic changed frozen weights')
    methods=rows[0]['comparisons']
    summary={method:_mean_nested([r['comparisons'][method] for r in rows]) for method in methods}
    interaction={emotion:_mean_nested([r['emotion_response_by_reference_interaction'][emotion] for r in rows]) for emotion in controls}
    reference_effect={emotion:_mean_nested([r['reference_effect_at_fixed_requested_emotion'][emotion] for r in rows]) for emotion in controls}
    report={'schema':SCHEMA,'clips':len(rows),'selection_seed':args.selection_seed,'random_seed':args.seed,
        'checkpoint_sha256':sha(args.checkpoint),'binding_sha256':sha(args.binding),'rig_sha256':sha(binding['rig']),
        'script_sha256':sha(__file__),'checkpoint_protocol':ck.get('protocol',{}),'mode':args.mode,'test_loaded':False,
        'selection_policy':'validation round-robin emotion, rotated speakers; fixed independent enrollment reference A',
        'emotion_counts':dict(Counter(names[r['emotion_id']] for r in rows)),
        'speaker_counts':dict(Counter(r['speaker_id'] for r in rows)),
        'references_from_validation_speakers_only':True,'reference_query_clip_and_sentence_disjoint':True,
        'student_audio_state_invariant':all(r['student_audio_state_invariant_across_all_conditions'] for r in rows),
        'summary':summary,'emotion_response_reference_interaction':interaction,
        'reference_effect_at_fixed_requested_emotion':reference_effect,'records':rows,
        'limitations':['This is validation intervention sensitivity, not a retrained ablation or proof of disentanglement.',
          'No nonpaired reference reconstruction error is scored or called style ground truth.',
          'Mouth coefficients and amplitude may change with emotion/style; jaw correlation, lag and closure are phonetic proxies, not calibrated phoneme/contact truth.',
          'Same audio/B0/seed does not guarantee fixed events when the style conditions change. Fixed-schedule decoder diagnostics are separately named.',
          'Requested emotion enters predict emotion probabilities; the current architecture keeps acoustic state and event conditioning unchanged under this explicit label override.',
          'Signed ARKit descriptor changes test physical output sensitivity, not emotion accuracy or psychological intensity.',
          'No direction success is declared solely because the label/probability input changed; generated motion changes and descriptor signs are reported.',
          'Reference-response interactions indicate conditioning sensitivity; reference emotion/domain nuisance and natural variations require separate matched controls/independent style evaluation.']}
    write_json(root/'report.json',report)
    write_json(root/'complete.json',{'status':'complete','clips':len(rows),'checkpoint_sha256':sha(args.checkpoint),
               'student_audio_state_invariant':report['student_audio_state_invariant'],'test_loaded':False})


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,choices=(32,64),default=64)
    p.add_argument('--selection-seed',type=int,default=47);p.add_argument('--seed',type=int,default=47)
    p.add_argument('--mode',choices=('sample','modal'),default='sample');p.add_argument('--save-curves',type=int,default=4)
    return p


if __name__=='__main__':evaluate(parser().parse_args())
