"""Five real-epoch stages with live identity/global paths and restartable state.

An isolated warm-start experiment. Never replaces defaults or reads test.
The final upper-face stage uses a signed spline state plus projected flow;
all other output channels are copied from the completed audio-stage model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.full_staged_data import load_training_inputs
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, capture_rng, restore_rng, canonical_hash
from scripts.train_predictable_renderer import state_hash
from kinetalk_b0.semantic_losses import style_contrastive
from kinetalk_b0.models.slow_state_affect import (
    SlowStateAffect, UPPER_INDICES, readout_slow_state, masked_slow_state,
    lift_slow_state, compose_upper_face, project_upper_innovation, UpperInnovationFlow,
)
from kinetalk_b0.models.dit import ResidualDiT
from kinetalk_b0.models.audio_residual_flow import (
    AudioResidualFlow, fair_trajectory_es, normalized_mean, static_audio,
)

SCHEMA = 'full_staged_spline_innovation_v1'
STAGES = ('articulation', 'identity', 'teacher', 'audio', 'dynamics')
MOUTH = tuple(range(14, 41))
GROUPS = {'brows': (41,42,43,44,45), 'eyes_expression': (5,6,12,13), 'mouth': MOUTH}
NOT_UPPER = tuple(i for i in range(52) if i not in UPPER_INDICES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def subset(q, ids, device):
    result = {}
    for key, value in q.items():
        if torch.is_tensor(value):
            selected = value[ids]
            result[key] = selected.to(device=device, dtype=torch.float32 if selected.is_floating_point() and key != 'times' else selected.dtype)
        elif isinstance(value, list): result[key] = [value[int(i)] for i in ids]
    return result


def obs(q): return q['valid'][...,None] & q['channel_mask'][:,None]


def base_forward(system, content, valid, *, gradients=False):
    """Match true-length B0 evaluation while explicitly allowing Stage1 gradients."""
    last = (valid * torch.arange(1, valid.shape[1]+1, device=valid.device)).amax(1)
    output = {}
    with torch.set_grad_enabled(gradients):
        for length_tensor in last.unique():
            length = int(length_tensor)
            ids = (last == length).nonzero(as_tuple=True)[0]
            clean = torch.where(valid[ids,:length,None], content[ids,:length], 0.)
            values = system.stage1(clean, valid[ids,:length])
            for key in ('b0','h0'):
                padded = F.pad(values[key], (0,0,0,valid.shape[1]-length))
                if key not in output: output[key] = padded.new_zeros(len(content), valid.shape[1], padded.shape[-1])
                output[key] = output[key].index_copy(0, ids, padded)
    output = {k:torch.where(valid[...,None],v,0.) for k,v in output.items()}
    if hasattr(system, 'motion_support'):
        output['b0'] = torch.where(system.motion_support, output['b0'], 0.)
    return output


def unfreeze(module):
    module.requires_grad_(True)
    return list(module.parameters())


def mse(x,y,mask):
    if mask.shape != x.shape: mask = mask.expand_as(x)
    if not mask.any(): return x.sum()*0
    return (x[mask]-y[mask]).square().mean()


def huber(x,y,mask):
    if mask.shape != x.shape: mask=mask.expand_as(x)
    return F.smooth_l1_loss(x[mask],y[mask],beta=1.) if mask.any() else x.sum()*0


def semantics(a,q):
    v = F.cross_entropy(a['emotion_logits'],q['emotion_id'])
    good=q['intensity_valid'] & (q['intensity_id']>=0)
    if good.any(): v=v+F.cross_entropy(a['intensity_logits'][good],q['intensity_id'][good])
    return v


def articulation_selection(q, scope='neutral'):
    """Return articulation clips and an auditable emotion-count summary.

    The legacy neutral-only stage remains the default.  ``all-emotions``
    changes only this stage's membership; it does not alter the later teacher
    or audio objectives, their emotion cross-entropy masks, or any checkpoint
    loading behavior.  Splits and masks are supplied by the caller's approved
    train manifest, so no validation/test rows can enter here.
    """
    if scope not in ('neutral', 'all-emotions'):
        raise ValueError("articulation scope must be 'neutral' or 'all-emotions'")
    if not isinstance(q, dict) or not torch.is_tensor(q.get('emotion_id')):
        raise ValueError('training split must contain tensor emotion_id')
    labels = q['emotion_id']
    if labels.ndim != 1 or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('emotion_id must be a one-dimensional integer tensor')
    ids = torch.arange(len(labels), dtype=torch.long) if scope == 'all-emotions' else (labels == 0).nonzero(as_tuple=True)[0].long()
    counts = {str(int(label)): int((labels[ids] == label).sum())
              for label in torch.unique(labels[ids], sorted=True)}
    if not len(ids):
        raise ValueError('Articulation scope selected no training clips')
    return ids, {'requested_scope': scope, 'actual_scope': scope,
                 'clip_count': len(ids), 'emotion_counts': counts,
                 'all_train_clips_included': bool(len(ids) == len(labels)),
                 'train_emotion_ids': sorted(int(value) for value in torch.unique(labels, sorted=True))}


def optimize(loss,opt,params):
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
    opt.zero_grad(set_to_none=True); loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    opt.step()
    return float(norm)


def identity_pairs(refs, fit_sids):
    pairs=[]
    for sid in fit_sids:
        n=len(refs[sid]['valid']); half=n//2
        if half<1: raise ValueError('Two independent references required')
        for a in itertools.combinations(range(n),half):
            b=tuple(i for i in range(n) if i not in a)
            if len(a)==len(b) and a>b: continue
            pairs.append((sid,a,b))
    return pairs


@torch.no_grad()
def cache_current_base(system,data,device,batch_size=32):
    """Recompute after B0 changes. No old h0/b0 cache is used."""
    for role,q in data['splits'].items():
        out={'b0':[],'h0':[]}
        for ids in torch.arange(len(q['valid'])).split(batch_size):
            b=subset(q,ids,device); base=base_forward(system,b['content'],b['valid'])
            for k in out:out[k].append(base[k].cpu())
        q['b0']=torch.cat(out['b0']);q['h0']=torch.cat(out['h0'])
    for sid,q in data['refs'].items():
        b=subset(q,torch.arange(len(q['valid'])),device)
        base=base_forward(system,b['content'],b['valid'])
        q['residual']=torch.where(obs(b),b['motion']-base['b0'],0.).cpu()


def encode_ref(system,data,sid,ids,device):
    q=data['refs'][sid]
    result=system.encode_identity(q['residual'][list(ids)][None].to(device),q['valid'][list(ids)][None].to(device),
        reference_channel_mask=q['channel_mask'][list(ids)][None].to(device))
    result['observed_channels']=q['channel_mask'][list(ids)].all(0)[None].to(device)
    return result


def paper_condition(q, mode):
    if mode == 'audio': return q
    if mode != 'static': raise ValueError('Unknown condition mode')
    # Both explicit local acoustics and B0's audio-derived per-frame h0 must
    # lose temporal information in the independently trained static control.
    return {**q, 'audio_features':static_audio(q['audio_features'],q['valid']),
            'h0':static_audio(q['h0'],q['valid'])}


def reverse_valid(value,valid):
    output=torch.where(valid[...,None],value,0.).clone()
    for i in range(len(value)):
        ids=valid[i].nonzero(as_tuple=True)[0]
        output[i,ids]=value[i,ids.flip(0)]
    return output


def complete_dynamic_conditions(q,local_audio,mode='audio'):
    """Counterfactuals remove/reverse ALL frame-varying dynamic conditions.

    Global affect/identity and generation noise are held by the caller.
    Static pooling is after the encoder, so padding/TCN edge effects cannot
    recreate a frame-varying local condition. Native clock remains unchanged.
    """
    if mode not in ('audio','static','reverse'):raise ValueError('Unknown dynamic intervention')
    dynamic=local_audio(q['audio_features'],q['valid'])
    b=dict(q)
    if mode=='static':
        b['h0']=static_audio(q['h0'],q['valid'])
        dynamic={**dynamic,'state':static_audio(dynamic['state'],q['valid']),
                 'local':static_audio(dynamic['local'],q['valid'])}
    elif mode=='reverse':
        b['h0']=reverse_valid(q['h0'],q['valid'])
        dynamic={**dynamic,'state':reverse_valid(dynamic['state'],q['valid']),
                 'local':reverse_valid(dynamic['local'],q['valid'])}
    return b,dynamic


@torch.no_grad()
def identity_cache(system,data,device):
    return {sid:{k:v.detach() for k,v in encode_ref(system,data,sid,range(len(q['valid'])),device).items()
                 if k in ('code','baseline')} for sid,q in data['refs'].items()}


def batch_identity(identities,q):
    return {k:torch.cat([identities[int(sid)][k] for sid in q['speaker_id']],0) for k in ('code','baseline')}


def teacher_affect(system,q,identity):
    residual=torch.where(obs(q),q['motion']-q['b0']-identity['baseline'][:,None],0.)
    return system.encode_motion(residual,q['valid'])


def targets(q,scales,stride):
    if not (q['channel_mask'][:,list(UPPER_INDICES)] & q['anchor_valid'][:,list(UPPER_INDICES)]).all():
        raise ValueError('Projected flow requires all nine upper channels and independent anchors')
    raw,mask=readout_slow_state(q['motion'],obs(q)&q['anchor_valid'][:,None],q['anchors'],scales)
    state=masked_slow_state(raw,mask,stride=stride)
    normalized=torch.where(q['valid'][...,None],(q['motion'][...,list(UPPER_INDICES)]-q['anchors'][:,None,list(UPPER_INDICES)])/scales[list(UPPER_INDICES)],0.)
    return state,project_upper_innovation(normalized,q['valid'],stride=stride)


class UpperFlow(UpperInnovationFlow):
    """Batch-dictionary adapter around the independently tested flow module."""
    def flow_loss(self,target,q,identity,affect,local,state,noise,flow_time):
        return super().flow_loss(target,q['valid'],q['h0'],identity['code'],affect,local,state,noise,flow_time)

    def decode(self,q,identity,affect,local,state,noise,steps):
        return super().decode(q['valid'],q['h0'],identity['code'],affect,local,state,noise,steps=steps)


def upper_motion(q,scales,state,innovation):
    delta=lift_slow_state(state,scales)[...,list(UPPER_INDICES)]
    return q['anchors'][:,None,list(UPPER_INDICES)]+delta+scales[list(UPPER_INDICES)]*innovation


def region_report(pred,q):
    answer={}
    for name,cc in GROUPS.items():
        mask=obs(q)[...,list(cc)];x=pred[...,list(cc)].double();y=q['motion'][...,list(cc)].double()
        count=mask.sum(1,keepdim=True).clamp_min(1)
        xc=torch.where(mask,x-torch.where(mask,x,0.).sum(1,keepdim=True)/count,0.)
        yc=torch.where(mask,y-torch.where(mask,y,0.).sum(1,keepdim=True)/count,0.)
        energy=yc.square().sum();sse=(xc-yc).square().sum()
        pair=mask[:,1:]&mask[:,:-1]
        vals=x[mask]
        answer[name]={'raw_mse':float(mse(x,y,mask)), 'centered_mse':float(sse/mask.sum()),
            'centered_r2':float(1-sse/energy.clamp_min(1e-12)),
            'centered_correlation':float((xc*yc).sum()/(xc.square().sum()*energy).sqrt().clamp_min(1e-12)),
            'rms_ratio':float((xc.square().sum()/energy.clamp_min(1e-12)).sqrt()),
            'frame_displacement_mse':float(mse(x[:,1:]-x[:,:-1],y[:,1:]-y[:,:-1],pair)),
            'outside_fraction':float(((vals<0)|(vals>1)).double().mean())}
    return answer


@torch.no_grad()
def identity_report(system,data,device):
    result={}
    for role,sids in (('fit',data['fit_sids']),('development',data['dev_sids'])):
        a=[];b=[];errors=[]
        for sid in sids:
            n=len(data['refs'][sid]['valid']);cut=max(1,n//2)
            x=encode_ref(system,data,sid,range(cut),device)
            y=encode_ref(system,data,sid,range(cut,n),device)
            a.append(x['code']);b.append(y['code']);errors.append(mse(x['baseline'],y['neutral_mean'],x['observed_channels']&y['observed_channels']))
        similarity=F.normalize(torch.cat(a),dim=-1)@F.normalize(torch.cat(b),dim=-1).T
        result[role]={'speakers':len(sids),'reference_view_retrieval':float((similarity.argmax(-1)==torch.arange(len(sids),device=device)).float().mean()),
            'cross_reference_baseline_mse':float(torch.stack(errors).mean()),
            'scope':'fit reference views trained; development reference views not trained. Coefficient identity, not mesh geometry.'}
    return result


@torch.no_grad()
def evaluate(system,audio,upper,local_audio,data,identities,stage,args,*,full=False):
    q=data['splits']['validation'];limit=len(q['valid']) if full else min(64,len(q['valid']))
    # Fixed metadata-independent random subset; no favourable sample selection.
    ids=torch.randperm(len(q['valid']),generator=torch.Generator().manual_seed(20260917))[:limit].sort().values
    reference=subset(q,ids,'cpu');all_predictions={};class_correct=0;teacher_correct=0
    seeds=(42,123,2026) if full else (42,)
    for seed in seeds:
        random_noise=torch.randn(len(q['valid']),q['valid'].shape[1],52,generator=torch.Generator().manual_seed(seed))
        modes=['full']
        if stage=='dynamics' and full and (seed==42 or getattr(args,'paper_data',None)):modes+=['base','static_state','oracle_state','reverse_audio']
        for mode in modes:
            predictions=[];pred_states=[];targets_state=[];gen_emotions=[]
            for ix in ids.split(args.batch_size):
                b=subset(q,ix,args.device);ident=batch_identity(identities,b)
                n=random_noise[ix].to(args.device);base={'b0':b['b0'],'h0':b['h0']}
                # Before the teacher stage, neither motion teacher nor renderer
                # has been fitted in a fresh run. Evaluating their random
                # residual confounds the identity stage with untrained modules.
                teacher=None;affect=None
                if stage=='articulation':pred=base['b0']
                elif stage=='identity':
                    pred=torch.where(b['valid'][...,None],base['b0']+ident['baseline'][:,None],0.)
                else:
                    teacher=teacher_affect(system,b,ident)
                    affect=teacher if stage=='teacher' else audio(b['audio_features'],b['valid'])
                    pred=system.generate(b['content'],b['valid'],ident,affect,initial_noise=n,steps=args.decode_steps,base=base)['motion']
                if stage=='dynamics' and mode!='base':
                    if getattr(args,'temporal_upper',False):
                        intervention='static' if mode=='static_state' else 'reverse' if mode=='reverse_audio' else getattr(args,'condition_mode','audio')
                        dynamic_b,dynamic=complete_dynamic_conditions(b,local_audio,intervention)
                    else:
                        dynamic_b=paper_condition(b,getattr(args,'condition_mode','audio')) if getattr(args,'paper_data',None) else b
                        feat=dynamic_b['audio_features']
                        if mode=='reverse_audio':feat=reverse_valid(feat,b['valid'])
                        dynamic=local_audio(feat,b['valid'])
                    state=dynamic['state'];tgt,_=targets(b,data['target_scales'].to(args.device),args.stride)
                    if mode=='static_state' and not getattr(args,'temporal_upper',False):state=static_audio(state,b['valid'])
                    if mode=='oracle_state':state=tgt['state']
                    innovation=upper.decode(dynamic_b,ident,affect,dynamic['local'],state,n[...,list(UPPER_INDICES)],args.decode_steps)
                    uv=upper_motion(b,data['target_scales'].to(args.device),state,innovation)
                    composed=compose_upper_face(pred,uv,b['valid'])
                    if not torch.equal(composed[...,list(NOT_UPPER)],pred[...,list(NOT_UPPER)]):raise RuntimeError('Non-upper protection failed')
                    pred=composed;pred_states.append(dynamic['state'].cpu());targets_state.append(tgt['state'].cpu())
                if teacher is not None and seed==42 and mode=='full':
                    class_correct+=int((affect['emotion_logits'].argmax(-1)==b['emotion_id']).sum())
                    teacher_correct+=int((teacher['emotion_logits'].argmax(-1)==b['emotion_id']).sum())
                if teacher is not None:
                    gen_teacher=system.encode_motion(torch.where(obs(b),pred-b['b0']-ident['baseline'][:,None],0.),b['valid'])
                    gen_emotions.extend((gen_teacher['emotion_logits'].argmax(-1)==b['emotion_id']).cpu().tolist())
                predictions.append(pred.cpu())
            value=torch.cat(predictions)
            key=f'{seed}/{mode}'
            all_predictions[key]={'motion':value,'metrics':region_report(value,reference),
                'generated_teacher_emotion_accuracy_nonindependent':sum(gen_emotions)/len(gen_emotions) if gen_emotions else None}
            if pred_states:
                ps,ts=torch.cat(pred_states),torch.cat(targets_state)
                sm=reference['valid'][...,None]
                pc=torch.where(sm,ps-ps.sum(1,keepdim=True)/sm.sum(1,keepdim=True),0.)
                tc=torch.where(sm,ts-ts.sum(1,keepdim=True)/sm.sum(1,keepdim=True),0.)
                se=(pc-tc).square().sum();energy=tc.square().sum()
                all_predictions[key]['state_metrics']={'raw_mse':float(mse(ps,ts,sm)),
                    'centered_mse':float(mse(pc,tc,sm)),
                    'centered_correlation':float((pc*tc).sum()/(pc.square().sum()*energy).sqrt().clamp_min(1e-12)),
                    'centered_r2':float(1-se/energy.clamp_min(1e-12))}
    report={'stage':stage,'clips':limit,'noise_seeds':list(seeds),
        'condition_source':{'articulation':'audio_content_B0','identity':'audio_content_B0_plus_neutral_identity',
                            'teacher':'target_motion_oracle','audio':'audio_only','dynamics':'audio_only'}[stage],
        'audio_or_teacher_emotion_accuracy':class_correct/limit if stage not in ('articulation','identity') else None,
        'motion_teacher_emotion_accuracy':teacher_correct/limit if stage not in ('articulation','identity') else None,
        'emotion_readout_scope':'Teacher not yet trained; emotion readout unavailable.' if stage in ('articulation','identity') else 'Teacher trained in this run; generated emotion readout is not independent perceptual certification.',
        'modes':{k:{n:v for n,v in val.items() if n!='motion'} for k,val in all_predictions.items()},
        'test_loaded':False,'default_replaced':False,
        'scope':'paper development; complete native sequences' if getattr(args,'paper_data',None) else 'historical internal development',
        'condition_mode':getattr(args,'condition_mode','audio')}
    report['dynamic_intervention_contract']='all encoded local/state/h0; global/id/noise fixed' if getattr(args,'temporal_upper',False) else 'legacy partial-state/raw-local control'
    curves={'schema':SCHEMA,'stage':stage,'clip_id':reference['clip_id'],'target':reference['motion'],'valid':reference['valid'],
        'times':reference['times'],'channel_mask':reference['channel_mask'],'b0':reference['b0'],
        'predictions':{k:v['motion'] for k,v in all_predictions.items()},'noise_seeds':list(seeds)}
    return report,curves


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','audio','targets','enrollment','native-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paper-data',type=Path)
    p.add_argument('--start-stage',choices=STAGES,default='articulation')
    p.add_argument('--articulation-scope', choices=('neutral', 'all-emotions'), default='neutral',
                   help='Stage-1 B0 mouth/articulation membership; legacy neutral-only by default')
    p.add_argument('--stage-checkpoint',type=Path)
    p.add_argument('--condition-mode',choices=('audio','static'),default='audio')
    p.add_argument('--identity-epochs',type=int,default=None)
    p.add_argument('--end-stage',choices=STAGES,default='dynamics')
    p.add_argument('--protect-mouth',action='store_true',help='Fixed residual support excludes mouth; require validation protection at audio/dynamics endpoints')
    p.add_argument('--temporal-upper',action='store_true',help='Temporally coupled state/noise upper9 flow with complete counterfactuals')
    p.add_argument('--compact',action='store_true')
    p.add_argument('--artifact-dir',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--decode-steps',type=int,default=12)
    p.add_argument('--stride',type=int,default=16);p.add_argument('--seed',type=int,default=47)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if STAGES.index(args.end_stage)<STAGES.index(args.start_stage):raise ValueError('end-stage precedes start-stage')
    if not args.smoke and not 1<=args.epochs<=60:raise ValueError('Epoch budget must be 1–60')
    if args.batch_size<1 or args.decode_steps<1 or args.stride<1:raise ValueError('Positive batch/solver/stride required')
    if args.output.exists() and not args.resume:raise FileExistsError('Fresh run required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.paper_data:
        from scripts.prepare_paper_full_data import load_paper_data
        print(json.dumps({'event':'data_load_start','paper_data':str(args.paper_data.resolve()),
                          'smoke':bool(args.smoke)}), flush=True)
        data=load_paper_data(args.paper_data,seed=args.seed,smoke=args.smoke)
        print(json.dumps({'event':'data_load_complete','fit_clips':len(data['splits']['train']['valid']),
                          'development_clips':len(data['splits']['validation']['valid']),
                          'smoke':bool(args.smoke)}), flush=True)
    else:
        if any(getattr(args,k) is None for k in ('source_run','audio','targets','enrollment','native_root')):
            raise ValueError('Historical data requires all source paths')
        data=load_training_inputs(args.source_run,args.audio,args.targets,args.enrollment,args.native_root)
    if args.smoke and args.paper_data:
        # Metadata-independent first available clip per person/class. This
        # limits only smoke, never full training membership.
        for role,q in data['splits'].items():
            chosen=[];seen=set()
            for i,(sid,emo) in enumerate(zip(q['speaker_id'],q['emotion_id'])):
                key=(int(sid),int(emo))
                if key not in seen:seen.add(key);chosen.append(i)
            data['splits'][role]=subset(q,torch.tensor(chosen),'cpu')
    system=data['system'].to(args.device).eval();cfg=data['config']
    # Fixed deployment support is learned from TRAIN metadata only. Query
    # observation masks are used only to censor training targets and scores.
    train_support=data['splits']['train']['channel_mask'].any(0).bool()
    cfg['model']['motion_support']=train_support.tolist()
    system.set_motion_support(train_support.to(args.device))
    residual_support=train_support.clone()
    if args.protect_mouth:residual_support[list(MOUTH)]=False
    cfg['model']['residual_support']=residual_support.tolist()
    system.set_residual_support(residual_support.to(args.device))
    # Statistics are fitted on the authorized fit tensor only, not development.
    f=data['splits']['train']['audio_features'];valid=data['splits']['train']['valid']
    mean=data['feature_stats']['mean'];std=data['feature_stats']['std']
    audio=SlowStateAffect(mean,std,stride=args.stride).to(args.device).eval()
    if args.temporal_upper:
        if not args.paper_data:raise ValueError('Temporal upper experiment requires fixed paper data')
        from kinetalk_b0.models.temporal_audio_residual_flow import TemporalAudioResidualFlow
        upper=TemporalAudioResidualFlow(cfg,stride=args.stride).to(args.device).eval()
    else:upper=(AudioResidualFlow if args.paper_data else UpperFlow)(cfg,stride=args.stride).to(args.device).eval()
    local_audio=copy.deepcopy(audio)
    inherited_upper=False
    if args.stage_checkpoint:
        saved=torch.load(args.stage_checkpoint,map_location='cpu',weights_only=False)
        if saved.get('data_manifest_sha256')!=data['provenance'].get('manifest_sha256'):
            raise ValueError('Stage checkpoint belongs to another data protocol')
        for module,key in [(system,'system'),(audio,'audio')]:module.load_state_dict(saved[key],strict=True)
        calibration=saved.get('config',{}).get('model',{}).get('mouth_reference_calibration')
        if calibration is not None:
            cfg['model']['mouth_reference_calibration']=calibration
            system.set_mouth_reference_calibration(calibration)
        saved_support=saved.get('config',{}).get('model',{}).get('motion_support')
        if saved_support is not None and saved_support!=train_support.tolist():raise ValueError('Stage checkpoint train support differs')
        saved_residual=saved.get('config',{}).get('model',{}).get('residual_support')
        if saved_residual is not None and saved_residual!=residual_support.tolist():raise ValueError('Stage checkpoint residual support differs; use matching protection mode')
        architecture='temporal_audio_residual' if args.temporal_upper else 'unrestricted_audio_residual' if args.paper_data else 'Q_projected'
        if args.start_stage=='dynamics' and saved.get('stage')=='audio' and saved.get('upper_architecture')==architecture:
            # Preserve the pre-dynamics receiver exactly when continuing the
            # same architecture after a runtime-only memory fix.
            upper.load_state_dict(saved['upper'],strict=True)
            inherited_upper=True
        local_audio.load_state_dict(audio.state_dict())
        del saved
    elif args.start_stage!='articulation':raise ValueError('Starting later requires matching stage checkpoint')
    input_paths={k:str(getattr(args,k)) for k in ('source_run','audio','targets','enrollment','native_root')}
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),root/'scripts/full_staged_data.py',root/'kinetalk_b0/models/slow_state_affect.py',
        root/'kinetalk_b0/models/neutral_affect.py',root/'kinetalk_b0/models/dit.py',root/'kinetalk_b0/models/model.py',
        root/'kinetalk_b0/models/encoders.py',root/'kinetalk_b0/models/label_guided_affect.py',root/'kinetalk_b0/neutral_data.py',
        root/'kinetalk_b0/semantic_losses.py',root/'scripts/train_formal_predictable_projection.py']
    if args.paper_data:sources += [root/'scripts/prepare_paper_full_data.py',root/'kinetalk_b0/models/audio_residual_flow.py',root/'scripts/paper_generation_report.py']
    if args.protect_mouth:sources += [root/'scripts/mouth_protection.py',root/'kinetalk_b0/reference_mouth_calibration.py']
    if args.temporal_upper:sources += [root/'kinetalk_b0/models/temporal_audio_residual_flow.py']
    if args.paper_data:input_paths['paper_data']=str(args.paper_data.resolve())
    if args.artifact_dir:input_paths['artifact_dir']=str(args.artifact_dir.resolve())
    articulation_ids, articulation_scope_info = articulation_selection(data['splits']['train'], args.articulation_scope)
    recipe={'schema':SCHEMA,'args':{k:v for k,v in vars(args).items() if k not in ('resume','output') and not isinstance(v,Path)},
        'paths':input_paths,'data_provenance':data['provenance'],'source_sha256':{str(p.relative_to(root)):sha(p) for p in sources},
        'stages':list(STAGES[STAGES.index(args.start_stage):STAGES.index(args.end_stage)+1]),'epochs_per_stage':args.epochs,'stride_frames':args.stride,
        'condition':'four signed spline motion proxies + native audio + distilled global; no text/VA/activity/history',
        'warm_start':bool(args.stage_checkpoint) or not bool(args.paper_data),'stage_checkpoint_sha256':sha(args.stage_checkpoint) if args.stage_checkpoint else None,
        'new_audio_global':'1540D audio trained against updated motion-global teacher; does not reuse old global normalization',
        'identity_epoch':'One pass over disjoint complementary reference-view pairs from fit identities only',
        'articulation_epoch':'One shuffled pass over the selected articulation scope',
        'articulation_scope':articulation_scope_info,
        'other_epoch':'One shuffled pass over all fit queries',
        'trainable':'Stage-dependent; pretrained acoustic/content extractors remain frozen feature sources',
        'stage5_nonupper':'Exact frozen stage4 output copy, not a claim stage4 equals historical model',
        'motion_support':train_support.tolist(),'residual_support':residual_support.tolist(),
        'upper_rollout_activation_checkpointing':bool(args.paper_data and not args.temporal_upper),
        'upper_inherited_from_audio_checkpoint':inherited_upper,
        'mouth_reference_calibration':cfg['model'].get('mouth_reference_calibration'),
        'temporal_upper_config':getattr(upper,'temporal_config',None),
        'test_loaded':False,'default_replaced':False,'checkpoint_selection':'Fixed final epoch; intermediate validation not used for selection'}
    if args.paper_data:
        recipe.update(schema='paper_full_audio_residual_flow_v1',
          condition='native full audio + independent neutral identity + global affect + audio slow mean + unrestricted stochastic residual',
          dynamic_objective='FM + deterministic mean state Huber; every fourth batch two-draw raw+centered fair ES and domain penalty',
          starting_stage=args.start_stage,acoustic_extractors=('frozen pretrained; system/audio inherited from bound stage checkpoint; upper inherited' if inherited_upper else 'frozen pretrained; system/audio inherited from bound stage checkpoint; upper initialized afresh') if args.stage_checkpoint else 'frozen pretrained; all KineTalk modules initialized afresh')
    recipe_hash=canonical_hash(recipe);args.output.mkdir(parents=True,exist_ok=True)
    gen=torch.Generator().manual_seed(args.seed)
    start_stage=STAGES.index(args.start_stage);start_epoch=0;total_steps=0;elapsed_before=0.;resume_payload=None
    if args.resume:
        resume_payload=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        if resume_payload['recipe_sha256']!=recipe_hash:raise ValueError('Resume recipe/source/input binding differs')
        system.load_state_dict(resume_payload['system']);audio.load_state_dict(resume_payload['audio']);upper.load_state_dict(resume_payload['upper']);local_audio.load_state_dict(resume_payload['local_audio'])
        start_stage=resume_payload['stage_index'];start_epoch=resume_payload['completed_epochs'];total_steps=resume_payload['total_steps'];elapsed_before=resume_payload['elapsed_seconds']
        restore_rng(resume_payload['rng'],gen)
    else:
        save_json(args.output/'provenance.json',{'recipe':recipe,'recipe_sha256':recipe_hash})
        for source in sources:
            dest=args.output/'source'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    started=time.monotonic();scales=data['target_scales'].to(args.device)
    epochs=1 if args.smoke else args.epochs
    cache_current_base(system,data,args.device)
    identities=identity_cache(system,data,args.device)
    pairs=identity_pairs(data['refs'],data['fit_sids'])
    if not args.resume and not args.paper_data:
        save_json(args.output/'identity_initial.json',identity_report(system,data,args.device))
        initial_report,initial_curves=evaluate(system,audio,upper,local_audio,data,identities,'teacher',args,full=not args.smoke)
        initial_report['scope']='Original checkpoint motion-teacher oracle only; not historical audio baseline'
        save_json(args.output/'initial_teacher_oracle.json',initial_report)
        save_checkpoint(args.output/'initial_teacher_oracle_curves.pt',initial_curves)
    current_stage='setup'
    try:
        for stage_index,stage in enumerate(STAGES):
            if stage_index<start_stage:continue
            if stage_index>STAGES.index(args.end_stage):break
            current_stage=stage;stage_dir=args.output/stage;stage_dir.mkdir(exist_ok=True)
            epochs=1 if args.smoke else (args.identity_epochs if stage=='identity' and args.identity_epochs is not None else args.epochs)
            if epochs<1:raise ValueError('Stage epochs must be positive')
            if stage=='dynamics' and args.paper_data and not (resume_payload is not None and stage_index==start_stage):
                # Match fresh audio/static receiver data, noise and time draws
                # despite one run having performed the preceding stages.
                gen.manual_seed(args.seed+5000);torch.manual_seed(args.seed+5000)
            for module in (system,audio,upper,local_audio):
                module.requires_grad_(False);module.eval();module.zero_grad(set_to_none=True)
            if stage=='articulation':groups=[{'params':unfreeze(system.stage1),'lr':3e-4 if args.paper_data else 1e-5}]
            elif stage=='identity':groups=[{'params':unfreeze(system.identity_encoder)+unfreeze(system.identity_bias),'lr':1e-4}]
            elif stage=='teacher':groups=[{'params':unfreeze(system.motion_teacher)+unfreeze(system.local_projection),'lr':1e-4},{'params':unfreeze(system.renderer),'lr':1e-4 if args.paper_data else 3e-5}]
            elif stage=='audio':groups=[{'params':unfreeze(audio),'lr':1e-4},{'params':unfreeze(system.renderer),'lr':5e-5 if args.paper_data else 1e-5}]
            else:
                if not (resume_payload is not None and stage_index==start_stage):local_audio.load_state_dict(audio.state_dict())
                groups=[{'params':unfreeze(local_audio),'lr':1e-4},{'params':unfreeze(upper),'lr':1e-4}]
            parameters=[p for g in groups for p in g['params']]
            optimizer=torch.optim.AdamW(groups,weight_decay=1e-5)
            initial_epoch=start_epoch if stage_index==start_stage else 0
            if resume_payload is not None and stage_index==start_stage:
                optimizer.load_state_dict(resume_payload['optimizer'])
                for state in optimizer.state.values():
                    for key,value in state.items():
                        if torch.is_tensor(value):state[key]=value.to(args.device)
            if stage=='articulation':items=articulation_ids
            elif stage=='identity':items=torch.arange(len(pairs))
            else:items=torch.arange(len(data['splits']['train']['valid']))
            if args.smoke:items=items[:min(len(items),args.batch_size*2)]
            batches=math.ceil(len(items)/args.batch_size)
            if not batches:raise ValueError('Empty training stage: '+stage)
            trainable={name:sum(p.numel() for p in module.parameters() if p.requires_grad) for name,module in [('system',system),('audio',audio),('upper',upper),('local_audio',local_audio)]}
            if torch.device(args.device).type=='cuda':torch.cuda.reset_peak_memory_stats(args.device)
            print(json.dumps({'event':'stage_start','stage':stage,'epochs':epochs,'samples_per_epoch':len(items),'batches':batches,'trainable':trainable}),flush=True)
            frozen_before={name:state_hash(module.state_dict()) for name,module in [('system',system),('audio',audio),('upper',upper),('local_audio',local_audio)] if not any(p.requires_grad for p in module.parameters())}
            frozen_parameters={name:state_hash({k:v for k,v in module.named_parameters() if not v.requires_grad}) for name,module in [('system',system),('audio',audio),('upper',upper),('local_audio',local_audio)]}
            def checkpoint(epoch):
                payload={'schema':SCHEMA,'recipe_sha256':recipe_hash,'stage_index':stage_index,'stage':stage,'completed_epochs':epoch,'total_steps':total_steps,
                    'system':system.state_dict(),'audio':audio.state_dict(),'upper':upper.state_dict(),'local_audio':local_audio.state_dict(),
                    'optimizer':optimizer.state_dict(),'rng':capture_rng(gen),'elapsed_seconds':elapsed_before+time.monotonic()-started}
                save_checkpoint(args.output/'last.pt',payload)
            if initial_epoch==0:checkpoint(0)
            for epoch in range(initial_epoch,epochs):
                begin=time.monotonic();sums={};order=items[torch.randperm(len(items),generator=gen)]
                for bi,ix in enumerate(order.split(args.batch_size)):
                    if stage=='identity':
                        av=[];bv=[]
                        for index in ix:
                            sid,aa,bb=pairs[int(index)]
                            av.append(encode_ref(system,data,sid,aa,args.device));bv.append(encode_ref(system,data,sid,bb,args.device))
                        ac=torch.cat([a['code'] for a in av]);bc=torch.cat([b['code'] for b in bv])
                        common=torch.cat([a['observed_channels']&b['observed_channels'] for a,b in zip(av,bv)])
                        base_loss=(mse(torch.cat([a['baseline'] for a in av]),torch.cat([b['neutral_mean'] for b in bv]),common)+
                            mse(torch.cat([b['baseline'] for b in bv]),torch.cat([a['neutral_mean'] for a in av]),common))/(2*.25**2)
                        contrast=style_contrastive(ac,bc,torch.tensor([pairs[int(i)][0] for i in ix],device=args.device))
                        loss=base_loss+.05*contrast;values={'baseline':base_loss,'contrast':contrast}
                    else:
                        b=subset(data['splits']['train'],ix,args.device);identity=batch_identity(identities,b)
                        base={'b0':b['b0'],'h0':b['h0']};mask=obs(b)
                        noise=torch.randn(b['motion'].shape,generator=gen).to(args.device);ft=torch.rand(len(ix),generator=gen).to(args.device)
                        if stage=='articulation':
                            out=base_forward(system,b['content'],b['valid'],gradients=True)['b0']
                            cc=list(MOUTH);m=mask[...,cc];scale=scales[cc].clamp_min(.05)
                            raw=huber(out[...,cc]/scale,b['motion'][...,cc]/scale,m)
                            pair=m[:,1:]&m[:,:-1]
                            velocity=huber((out[:,1:,cc]-out[:,:-1,cc])/scale,(b['motion'][:,1:,cc]-b['motion'][:,:-1,cc])/scale,pair)
                            loss=raw+.1*velocity;values={'raw':raw,'velocity':velocity}
                        elif stage in ('teacher','audio'):
                            if stage=='teacher':affect=teacher_affect(system,b,identity);distill=affect['global'].sum()*0
                            else:
                                with torch.no_grad():teacher=teacher_affect(system,b,identity)
                                affect=audio(b['audio_features'],b['valid'])
                                distill=F.mse_loss(affect['global'],teacher['global'])
                            out=system.flow(b['motion'],b['content'],b['valid'],identity,affect,noise=noise,time=ft,base=base,observation_mask=b['channel_mask'])
                            flow=mse(out['prediction'],out['velocity_target'],out['observation_mask']);semantic=semantics(affect,b)
                            loss=flow+.1*semantic+(.5*distill if stage=='audio' else 0)
                            values={'flow':flow,'semantic':semantic,'global_distill':distill}
                            if stage=='audio':
                                truth,_=targets(b,scales,args.stride)
                                state_loss=huber(affect['state'],truth['state'],truth['state_mask'])
                                loss=loss+.2*state_loss;values['slow_state']=state_loss
                            # A genuine noise rollout every fourth batch protects output fit.
                            if bi%4==0:
                                generated=system.generate(b['content'],b['valid'],identity,affect,initial_noise=noise,steps=args.decode_steps,base=base)['motion']
                                reconstruction=torch.stack([huber(generated[...,list(cc)]/scales[list(cc)].clamp_min(.02),b['motion'][...,list(cc)]/scales[list(cc)].clamp_min(.02),mask[...,list(cc)]) for cc in GROUPS.values()]).mean()
                                domain=(F.relu(-generated[mask])+F.relu(generated[mask]-1)).mean()
                                loss=loss+.2*reconstruction+.1*domain;values.update(rollout_raw=reconstruction,domain=domain)
                        else:
                            with torch.no_grad():affect=audio(b['audio_features'],b['valid'])
                            if args.temporal_upper:dynamic_b,dynamic=complete_dynamic_conditions(b,local_audio,args.condition_mode)
                            else:
                                dynamic_b=paper_condition(b,args.condition_mode) if args.paper_data else b
                                dynamic=local_audio(dynamic_b['audio_features'],b['valid'])
                            truth,innovation=targets(b,scales,args.stride)
                            state_loss=huber(dynamic['state'],truth['state'],truth['state_mask'])
                            if args.paper_data:
                                cc=list(UPPER_INDICES)
                                normalized=(b['motion'][...,cc]-b['anchors'][:,None,cc])/scales[cc]
                                # Stop target gradients: the deterministic mean
                                # learns from its own supervision and ES, not
                                # by moving both sides of the FM regression.
                                innovation=torch.where(b['valid'][...,None],normalized-normalized_mean(dynamic['state']).detach(),0.)
                            flow=upper.flow_loss(innovation,dynamic_b,identity,affect,dynamic['local'],dynamic['state'],noise[...,list(UPPER_INDICES)],ft)
                            loss=flow+.5*state_loss;values={'flow':flow,'slow_state':state_loss}
                            if bi%4==0:
                                residual=upper.decode(dynamic_b,identity,affect,dynamic['local'],dynamic['state'],noise[...,list(UPPER_INDICES)],args.decode_steps)
                                uv=upper_motion(b,scales,dynamic['state'],residual)
                                domain=(F.relu(-uv[b['valid']])+F.relu(uv[b['valid']]-1)).mean()
                                loss=loss+.1*domain;values['domain']=domain
                                if args.paper_data:
                                    second_noise=torch.randn(noise.shape,generator=gen).to(args.device)[...,list(UPPER_INDICES)]
                                    second=upper.decode(dynamic_b,identity,affect,dynamic['local'],dynamic['state'],second_noise,args.decode_steps)
                                    draws=torch.stack((residual,second))+normalized_mean(dynamic['state'])[None]
                                    raw_es=fair_trajectory_es(draws,normalized,b['valid'])
                                    centered_es=fair_trajectory_es(draws,normalized,b['valid'],centered=True)
                                    loss=loss+.1*raw_es+.1*centered_es
                                    values.update(rollout_fair_es=raw_es,rollout_centered_fair_es=centered_es)
                    norm=optimize(loss,optimizer,parameters);total_steps+=1
                    for k,v in {'total':loss,**values}.items():sums.setdefault(k,[]).append(float(v.detach()))
                    if bi%25==0:
                        progress={'event':'batch','stage':stage,'epoch':epoch+1,'batch':bi+1,'batches':batches,'loss':float(loss.detach()),'grad_norm':norm}
                        if torch.device(args.device).type=='cuda':
                            progress.update(cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                                            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30)
                        save_json(args.output/'status.json',{**progress,'status':'running','completed_epochs':epoch,
                                  'total_steps':total_steps,'elapsed_seconds':elapsed_before+time.monotonic()-started})
                        print(json.dumps(progress),flush=True)
                elapsed=time.monotonic()-begin
                record={'stage':stage,'epoch':epoch+1,'batches':batches,'samples':len(items),'seconds':elapsed,'losses':{k:sum(v)/len(v) for k,v in sums.items()},'total_steps':total_steps}
                checkpoint(epoch+1);save_json(stage_dir/f'epoch{epoch+1:03d}.json',record)
                save_json(args.output/'status.json',{**record,'status':'running','stage_index':stage_index,'stage_count':len(STAGES),'epochs_per_stage':epochs,'elapsed_seconds':elapsed_before+time.monotonic()-started})
                print(json.dumps({'event':'epoch_complete',**record}),flush=True)
                if stage in ('teacher','audio','dynamics') and ((epoch+1)%4==0 or args.smoke):
                    report,_=evaluate(system,audio,upper,local_audio,data,identities,stage,args)
                    save_json(stage_dir/f'dev_epoch{epoch+1:03d}.json',report)
            if stage=='articulation':cache_current_base(system,data,args.device)
            if stage in ('articulation','identity'):identities=identity_cache(system,data,args.device)
            for name,want in frozen_before.items():
                if state_hash(dict(system=system,audio=audio,upper=upper,local_audio=local_audio)[name].state_dict())!=want:raise RuntimeError('Frozen module drift: '+name)
            for name,module in [('system',system),('audio',audio),('upper',upper),('local_audio',local_audio)]:
                actual=state_hash({k:v for k,v in module.named_parameters() if not v.requires_grad})
                if actual!=frozen_parameters[name]:raise RuntimeError('Frozen parameter subset drift: '+name)
            identity_result=identity_report(system,data,args.device);save_json(stage_dir/'identity.json',identity_result)
            report,curves=evaluate(system,audio,upper,local_audio,data,identities,stage,args,full=not args.smoke)
            if args.protect_mouth and stage in ('identity','audio','dynamics'):
                from scripts.mouth_protection import protection_report
                vq=data['splits']['validation'];lookup={cid:i for i,cid in enumerate(vq['clip_id'])}
                vb=subset(vq,torch.tensor([lookup[cid] for cid in curves['clip_id']]),args.device)
                vi=batch_identity(identities,vb)
                protected=torch.where(vb['valid'][...,None],vb['b0'] if stage=='identity' else vb['b0']+vi['baseline'][:,None],0.).cpu()
                gates={key:protection_report(pred,protected,curves['target'],curves['valid'],curves['channel_mask'],vb['emotion_id'].cpu()) for key,pred in curves['predictions'].items()}
                save_json(stage_dir/'mouth_protection.json',gates)
                if not all(g['passed'] for g in gates.values()):
                    save_json(stage_dir/'evaluation.json',report)
                    raise RuntimeError('Mouth protection failed; downstream stages not started')
            report['identity']=identity_result;save_json(stage_dir/'evaluation.json',report)
            if args.temporal_upper and stage=='dynamics':
                # Fixed training probe distinguishes failure to fit from
                # cross-identity validation degradation. This is not test.
                train_q=data['splits']['train']
                probe_ids=torch.randperm(len(train_q['valid']),generator=torch.Generator().manual_seed(20260921))[:64].sort().values
                probe_data={**data,'splits':{**data['splits'],'validation':subset(train_q,probe_ids,'cpu')}}
                probe_report,_=evaluate(system,audio,upper,local_audio,probe_data,identities,stage,args,full=True)
                probe_report['scope']='fixed 64 training queries diagnostic; no holdout claim'
                probe_report['clip_ids']=probe_data['splits']['validation']['clip_id']
                save_json(stage_dir/'training_probe.json',probe_report)
            if args.paper_data:
                from scripts.paper_generation_report import report_generation
                report_generation(curves,data,stage_dir,stage,args)
            if not args.compact or stage in ('audio','dynamics'):
                curve_dir=args.artifact_dir/stage if args.artifact_dir else stage_dir
                curve_dir.mkdir(parents=True,exist_ok=True);save_checkpoint(curve_dir/'curves.pt',curves)
            else:curve_dir=None
            save_checkpoint(stage_dir/'final.pt',{'schema':SCHEMA,'recipe_sha256':recipe_hash,'stage':stage,'completed_epochs':epochs,
                'system':system.state_dict(),'audio':audio.state_dict(),'upper':upper.state_dict(),'local_audio':local_audio.state_dict(),
                'config':cfg,'scales':data['target_scales'],'inference_only':True,
                'feature_stats':data['feature_stats'],'data_manifest_sha256':data['provenance'].get('manifest_sha256'),
                'upper_architecture':'temporal_audio_residual' if args.temporal_upper else 'unrestricted_audio_residual' if args.paper_data else 'Q_projected',
                'temporal_upper_config':getattr(upper,'temporal_config',None),
                'test_loaded':False})
            save_json(stage_dir/'complete.json',{'stage':stage,'completed_epochs':epochs,'final_sha256':sha(stage_dir/'final.pt'),
                'curves':str(curve_dir/'curves.pt') if curve_dir else None,'curves_sha256':sha(curve_dir/'curves.pt') if curve_dir else None})
            resume_payload=None;start_epoch=0
        save_json(args.output/'summary.json',{'schema':SCHEMA,'status':'complete','stages':recipe['stages'],'epochs_per_stage':epochs,'smoke':args.smoke,
            'total_steps':total_steps,'elapsed_seconds':elapsed_before+time.monotonic()-started,'default_replaced':False,'test_loaded':False})
        save_json(args.output/'status.json',{'status':'complete','elapsed_seconds':elapsed_before+time.monotonic()-started,'summary':'summary.json'})
        print('FULL_STAGED_TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        failure={'status':'failed','stage':current_stage,'exception':repr(exc),'recover_from':'last.pt','elapsed_seconds':elapsed_before+time.monotonic()-started}
        save_json(args.output/'failure.json',failure)
        save_json(args.output/'status.json',failure)
        raise


if __name__=='__main__':main()
