"""Joint real-trajectory motion support and bounded audio token reweighting."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_motion_process as old
from scripts.joint_prior_audio_context import load_frozen_audio, assert_source_binding, encode_clips
from scripts import joint_motion_dictionary as d
from scripts import joint_motion_metrics as metrics
from kinetalk_b0.models.joint_token_audio import BoundedTokenResidual, unit_audio_descriptor

SCHEMA = 'joint_empirical_motion_prior_v1'
SEEDS = old.SEEDS
CC = [41,42,43,44,45,5,6,12,13]


def split_inner(clips, original):
    sentences = sorted({clips[i]['sentence'] for i in original['fit']}, key=lambda s:
        hashlib.sha256(('joint_motion_20260918:sentence:'+s).encode()).hexdigest())
    if len(sentences) < 24:
        raise ValueError('At least24 fitting sentences required')
    confirmation, calibration = set(sentences[:10]), set(sentences[10:20])
    out = {'fit': [], 'calibration': [], 'confirmation': []}
    for i in original['fit']:
        s = clips[i]['sentence']
        out['confirmation' if s in confirmation else 'calibration' if s in calibration else 'fit'].append(i)
    out.update({'old_'+key: value for key, value in original.items() if key != 'fit'})
    return out


def raw_scales(clips, ids):
    energy = np.zeros(9); count = 0
    for i in ids:
        c = clips[i]; x = old.center_runs(c['upper'], c['valid'].numpy())
        energy += (x[c['valid'].numpy()]**2).sum(0); count += int(c['valid'].sum())
    return np.maximum(np.sqrt(energy/count), .005)


def join_history(raw, anchor):
    x = np.asarray(raw, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    x = np.clip(x, 1e-4, 1-1e-4); anchor = np.clip(anchor, 1e-4, 1-1e-4)
    return np.log(x/(1-x))-np.log(anchor/(1-anchor))


def local_input(global_vec, local, valid, start, history, dictionary):
    descriptor = unit_audio_descriptor(torch.as_tensor(local), torch.as_tensor(valid), int(start), duration=32)
    descriptor = descriptor.detach().cpu().numpy() if torch.is_tensor(descriptor) else np.asarray(descriptor)
    history = np.zeros((8,9)) if history is None else history
    return np.concatenate([global_vec, descriptor, (history/dictionary['scales']).reshape(-1)]).astype(np.float32)


def rollout(dictionary, global_vec, local, anchor, valid, *, seed, clip_key,
            history_weight, global_weight, mode='global', residual=None, stats=None, local_scale=0.):
    """Audio/reference-only API; target motion and teacher boundaries are absent."""
    valid = np.asarray(valid, bool); output = np.zeros((len(valid),9), np.float64); boundaries=[]; chosen=[]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:{clip_key}'.encode()).digest()[:8], 'little'))
    for run_start, run_end in old.runs(valid):
        t = run_start; history = None
        while t < run_end:
            logits = np.zeros(len(dictionary['medoids'])) if mode == 'uniform' else d.prior_logits(
                dictionary, global_vec, history, use_global=mode != 'history',
                history_weight=history_weight, global_weight=global_weight)
            if residual is not None and local_scale:
                x = local_input(global_vec, local, valid, t, history, dictionary)
                with torch.no_grad():
                    dev = next(residual.parameters()).device
                    adjustment = residual(torch.as_tensor((x-stats['mean'])/stats['std'], device=dev)[None])[0].cpu().numpy()
                logits = logits+local_scale*adjustment
            probability = np.exp(logits-np.max(logits)); probability /= probability.sum()
            token = min(np.searchsorted(np.cumsum(probability), rng.random()), len(probability)-1)
            previous = output[t-1] if t > run_start else None
            chunk = d.decode_unit(dictionary, int(token), anchor, previous_raw=previous, blend_frames=8)
            end = min(run_end, t+len(chunk)); output[t:end] = chunk[:end-t]
            if t > run_start:
                boundaries.append(t)
            chosen.append({'start': t, 'token': int(token), 'duration': len(chunk)})
            t = end
            if t < run_end:
                prefix = output[max(run_start,t-8):t]
                if len(prefix) < 8:
                    prefix = np.pad(prefix, ((8-len(prefix),0),(0,0)), mode='edge')
                history = join_history(prefix, anchor)
    return output, boundaries, chosen


def evaluate(clips, ids, dictionary, scales, setting, *, mode, residual=None, stats=None,
             local_scale=0., intervention='real', save_draws=False):
    rows=[]; curves={}; donors={}
    if intervention == 'mismatch':
        for i in ids:
            candidates=[j for j in ids if clips[j]['speaker']==clips[i]['speaker'] and clips[j]['emotion']==clips[i]['emotion']
                and clips[j]['sentence']!=clips[i]['sentence'] and len(old.runs(clips[j]['valid'].numpy()))==1]
            if candidates:
                donors[i]=min(candidates, key=lambda j: clips[j]['clip_id'])
        ids=[i for i in ids if i in donors]
    for i in ids:
        c=clips[i]; valid=c['valid'].numpy(); local=c['local'].numpy().copy()
        if intervention=='static':
            local[valid]=local[valid].mean(0)
        elif intervention=='reverse':
            for a,b in old.runs(valid):
                local[a:b]=local[a:b][::-1]
        elif intervention=='mismatch':
            donor=clips[donors[i]]; source=donor['local'][donor['valid']].numpy()
            for a,b in old.runs(valid):
                xx=np.linspace(0,len(source)-1,b-a)
                local[a:b]=np.stack([np.interp(xx,np.arange(len(source)),source[:,j]) for j in range(64)],-1)
        elif intervention!='real':
            raise ValueError('Unknown intervention')
        values=[]; boundaries=[]; tokens=[]
        for seed in SEEDS:
            value, boundary, token=rollout(dictionary,c['global'],local,c['anchor_upper'],valid,seed=seed,clip_key=c['clip_id'],
                history_weight=setting[0],global_weight=setting[1],mode=mode,residual=residual,stats=stats,local_scale=local_scale)
            values.append(value); boundaries.append(boundary); tokens.append(token)
        values=np.asarray(values)
        row=metrics.score_clip(values,c['upper'],valid,scales,boundaries=boundaries)
        rows.append({'clip_id':c['clip_id'],'speaker':c['speaker'],'sentence':c['sentence'],'emotion':c['emotion'],**row})
        if save_draws:
            curves[c['clip_id']]={'samples':values.astype(np.float32),'target':c['upper'],'valid':valid,
                'anchor':c['anchor_upper'],'boundaries':boundaries,'tokens':tokens}
    return {'summary':metrics.summarize(rows),'rows':rows,'mode':mode,'intervention':intervention,
        'setting':list(setting),'local_scale':local_scale,
        'donor_mapping':{clips[i]['clip_id']:clips[j]['clip_id'] for i,j in donors.items()}},curves


def quality_gate(report, uniform):
    s,b=report['summary'],uniform['summary']
    # Metrics contracts are explicit; missing/nonfinite data never pass.
    rms=np.asarray(s.get('rms_ratio', []), dtype=float)
    required = (s.get('variogram', {}).get('aggregate'), b.get('variogram', {}).get('aggregate'),
                s.get('covariance_distance', {}).get('velocity'), b.get('covariance_distance', {}).get('velocity'),
                s.get('speed', {}).get('seam'), s.get('speed', {}).get('within'))
    if (rms.shape != (4,) or not np.isfinite(rms).all()
            or required[0] is None or required[1] is None
            or required[2] is None or required[3] is None
            or required[4] is None or required[5] is None):
        return {'passed': False, 'reason': 'missing_temporal_support_or_nonfinite_metrics'}
    var=float(required[0]); basevar=float(required[1]); cov=float(required[2]); basecov=float(required[3])
    seam, within = required[4], required[5]
    if (seam.get('count', 0) <= 0 or within.get('count', 0) <= 0
            or not np.isfinite([var, basevar, cov, basecov, seam.get('sum_squares', np.nan),
                                within.get('sum_squares', np.nan)]).all()):
        return {'passed': False, 'reason': 'missing_temporal_support_or_nonfinite_metrics'}
    seam_rms=np.sqrt(seam['sum_squares']/seam['count'])
    within_rms=np.sqrt(within['sum_squares']/within['count'])
    ratio=float(seam_rms/max(within_rms,1e-12))
    passed=bool(np.isfinite(np.r_[rms,var,cov,ratio]).all() and ((rms>=.4)&(rms<=2.5)).all()
        and var<=1.1*max(basevar,1e-12) and cov<=1.1*max(basecov,1e-12) and ratio<=1.5)
    return {'passed':passed,'rms_ratio':rms.tolist(),'variogram_ratio':var/max(basevar,1e-12),
        'velocity_covariance_ratio':cov/max(basecov,1e-12),'seam_within_rms_ratio':ratio}


def local_training_allowed(motion_calibration_passed):
    """The expensive local audio fit is conditional on motion feasibility.

    Smoke mode exercises loading/evaluation, but must not silently train a
    local controller after the motion gate has failed.
    """
    return bool(motion_calibration_passed)


def final_status(*, smoke, seconds, local_trained, local_epochs,
                 motion_calibration_passed, motion_confirmation_passed):
    """Stable machine-readable terminal status written by the driver."""
    return {'schema': SCHEMA, 'status': 'complete', 'smoke': bool(smoke),
        'seconds': float(seconds), 'local_trained': bool(local_trained),
        'local_epochs': int(local_epochs),
        'motion_calibration_passed': bool(motion_calibration_passed),
        'motion_confirmation_passed': bool(motion_confirmation_passed),
        'generator_integrated': False, 'test_loaded': False, 'default_replaced': False}


def prior_coverage(clips, ids, dictionary, scales):
    units=d.extract_units(clips,ids)
    rows=[]
    for unit in units:
        match=d.oracle_nearest(dictionary,unit['prefix'],unit['future'])
        index=match[0] if isinstance(match,tuple) else match['index'] if isinstance(match,dict) else match
        # Teacher nearest selection and duration resampling are oracle only.
        med=dictionary['medoids'][int(index)]
        future=med['future']; grid=np.linspace(0,len(future)-1,len(unit['future']))
        reconstruction=np.stack([np.interp(grid,np.arange(len(future)),future[:,j]) for j in range(9)],-1)
        error=((reconstruction-unit['future'])/dictionary['scales'])**2
        rows.append({'clip_id':unit['clip_id'],'duration':unit['duration'],'normalized_mse':float(error.mean())})
    return {'scope':'GT nearest motion oracle only, not deployment','units':len(rows),
        'normalized_mse':float(np.mean([r['normalized_mse'] for r in rows])) if rows else None,'rows':rows}


def training_examples(clips, ids, dictionary, setting):
    units=d.extract_units(clips,ids); mapping={c['clip_id']:c for c in clips}; x=[]; logits=[]; targets=[]
    for unit in units:
        c=mapping[unit['clip_id']]
        match=d.oracle_nearest(dictionary,unit['prefix'],unit['future'])
        token=match[0] if isinstance(match,tuple) else match['index'] if isinstance(match,dict) else match
        x.append(local_input(c['global'],c['local'].numpy(),c['valid'].numpy(),unit['start'],unit['prefix'],dictionary))
        logits.append(d.prior_logits(dictionary,c['global'],unit['prefix'],use_global=True,
            history_weight=setting[0],global_weight=setting[1])); targets.append(int(token))
    return np.asarray(x,dtype=np.float32),np.asarray(logits,dtype=np.float32),np.asarray(targets,dtype=np.int64)


def dump_eval(output, name, result, curves):
    old.save_json(output/(name+'.json'),result)
    if curves:
        torch.save(curves,output/(name+'.pt'))


def bootstrap_gain(real, baseline):
    if [x['clip_id'] for x in real['rows']] != [x['clip_id'] for x in baseline['rows']]:
        raise ValueError('Paired IDs differ')
    pairs=[(r['sentence'],b['joint_fair_es']['centered']-r['joint_fair_es']['centered'])
        for r,b in zip(real['rows'],baseline['rows'])]
    groups={s:[v for ss,v in pairs if ss==s] for s in sorted({s for s,_ in pairs})}
    keys=list(groups); rng=np.random.default_rng(20260918)
    boot=[np.mean([v for j in rng.integers(len(keys),size=len(keys)) for v in groups[keys[j]]]) for _ in range(2000)]
    return {'gain':float(np.mean([v for _,v in pairs])),'ci95':np.quantile(boot,[.025,.975]).tolist()}


def paired_subset(result, clip_ids):
    """Return a report restricted to an explicitly paired clip-id support.

    The mismatch intervention can have fewer rows because a donor with the
    requested speaker/emotion stratum may not exist. Pairing is therefore
    performed before bootstrap, never by silently comparing different clips.
    """
    allowed = set(clip_ids)
    rows = [row for row in result['rows'] if row['clip_id'] in allowed]
    if not rows:
        raise ValueError('Empty paired report support')
    return {'summary': metrics.summarize(rows), **{key: value for key, value in result.items()
        if key not in ('summary', 'rows')}, 'rows': rows}


def _relative_not_worse(value, baseline, tolerance=0.05):
    """Return whether a lower-is-better metric did not regress beyond tolerance."""
    try:
        value, baseline = float(value), float(baseline)
    except (TypeError, ValueError):
        return False
    if not np.isfinite([value, baseline]).all():
        return False
    # A zero baseline has no meaningful relative scale; any positive regression
    # is therefore rejected while exact zeros remain admissible.
    if baseline <= 0:
        return value <= baseline
    return value <= (1.0 + tolerance) * baseline


def audio_gate(reports, motion_gate, scale):
    """Apply the pre-registered confirmation gate for local audio control.

    The local controller must beat the frozen global prior on paired sentence
    bootstrap, remain within five percent on absolute/temporal diagnostics,
    and beat static and same-speaker/emotion mismatched local interventions.
    This gate authorizes only the next experiment; it never integrates the
    controller into the renderer.
    """
    required = ('local', 'global', 'static', 'mismatch')
    if any(name not in reports for name in required) or not motion_gate.get('passed', False):
        return {'passed': False, 'reason': 'missing_motion_gate_or_intervention_report'}
    local = reports['local']['summary']; global_ = reports['global']['summary']
    static = reports['static']['summary']; mismatch = reports['mismatch']['summary']
    if not scale or float(scale) <= 0:
        return {'passed': False, 'reason': 'zero_local_scale'}
    # All paired gains use the intersection of the reports' clip IDs. The
    # mismatch arm is allowed to have lower support, but its support is made
    # explicit and must contain enough sentence clusters for a bootstrap.
    supports = [set(row['clip_id'] for row in reports[name]['rows']) for name in required]
    support = set.intersection(*supports)
    confirmation_support = len(set(row['clip_id'] for row in reports['local']['rows']))
    coverage = len(support) / max(confirmation_support, 1)
    if len(support) < 2:
        return {'passed': False, 'reason': 'insufficient_common_intervention_support',
                'support_clips': len(support), 'support_coverage': coverage}
    if coverage < 0.70:
        return {'passed': False, 'reason': 'insufficient_common_intervention_coverage',
                'support_clips': len(support), 'support_coverage': coverage,
                'required_coverage': 0.70}
    sentence_count = len({row['sentence'] for row in reports['mismatch']['rows'] if row['clip_id'] in support})
    if sentence_count < 2:
        return {'passed': False, 'reason': 'insufficient_common_intervention_sentences',
                'support_clips': len(support), 'support_sentences': sentence_count,
                'support_coverage': coverage}
    paired_reports = {name: paired_subset(reports[name], support) for name in required}
    gain_global = bootstrap_gain(paired_reports['local'], paired_reports['global'])
    gain_static = bootstrap_gain(paired_reports['static'], paired_reports['local'])
    gain_mismatch = bootstrap_gain(paired_reports['mismatch'], paired_reports['local'])
    local = paired_reports['local']['summary']; global_ = paired_reports['global']['summary']
    static = paired_reports['static']['summary']; mismatch = paired_reports['mismatch']['summary']
    local_es = local['joint_fair_es']['centered']; global_es = global_['joint_fair_es']['centered']
    static_es = static['joint_fair_es']['centered']; mismatch_es = mismatch['joint_fair_es']['centered']
    raw_ok = _relative_not_worse(local['joint_fair_es']['raw'], global_['joint_fair_es']['raw'])
    var_ok = _relative_not_worse(local['variogram']['aggregate'], global_['variogram']['aggregate'])
    cov_ok = _relative_not_worse(local['covariance_distance']['velocity'], global_['covariance_distance']['velocity'])
    try:
        finite = np.isfinite([local_es, global_es, static_es, mismatch_es]).all()
    except (TypeError, ValueError):
        finite = False
    passed = bool(finite and gain_global['ci95'][0] > 0
        and gain_global['gain'] >= .01 * max(float(global_es), 0.)
        and local_es < static_es and local_es < mismatch_es
        and raw_ok and var_ok and cov_ok)
    return {'passed': passed,
        'reason': 'passed' if passed else 'audio_confirmation_gate_failed',
        'global_gain': gain_global, 'static_gain': gain_static,
        'mismatch_gain': gain_mismatch,
        'centered_es': {'local': float(local_es), 'global': float(global_es),
                        'static': float(static_es), 'mismatch': float(mismatch_es)},
        'relative_checks': {'raw_es': raw_ok, 'variogram': var_ok, 'velocity_covariance': cov_ok},
        'motion_gate': motion_gate, 'support_clips': len(support),
        'support_coverage': coverage, 'support_sentences': sentence_count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audio','targets','native-root','native-manifest','delta-dir','audio-checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh experiment directory required')
    args.output.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20260918)
    old.save_json(args.output/'status.json',{'status':'loading','schema':SCHEMA,'smoke':args.smoke})
    # Always construct the complete metadata split first; smoke truncation follows it.
    loading=copy.copy(args);loading.smoke=False
    clips,original,lineage=old.load_clips(loading);split=split_inner(clips,original)
    if args.smoke:
        split={k:v[:8] for k,v in split.items()}
    # Only independent enrollment supplies origin. Nine anchors retained directly,
    # rather than reconstructing them from grouped means.
    target=torch.load(args.targets,map_location='cpu',weights_only=False,mmap=True)['splits']['train']
    anchors={cid:row[CC].numpy() for cid,row in zip(target['clip_id'],target['anchors'])}
    # old_* entries are descriptive historical references only; they are never
    # part of the frozen encoder context or any formal experiment split.
    selected=sorted(set(i for key in ('fit','calibration','confirmation') for i in split[key]))
    model=load_frozen_audio(args.audio_checkpoint,args.device);binding=assert_source_binding(model,lineage)
    encoded=encode_clips(model,[clips[i] for i in selected],batch_size=16,device=args.device)
    for i,context in zip(selected,encoded):
        c=clips[i];c['anchor_upper']=anchors[c['clip_id']]
        c['global']=np.concatenate([context['global'].numpy().reshape(-1),context['intensity'].numpy().reshape(-1)])
        c['local']=context['local']
        # Keep the full native clip clock; no target action is attached to this
        # audio context object after the frozen encoder returns.
        if c['local'].shape[:1] != c['features'].shape[:1]:
            raise ValueError('Frozen audio context length differs from native clip')
    del model
    code=('scripts/train_joint_motion_prior.py','scripts/joint_motion_dictionary.py','scripts/joint_motion_metrics.py',
          'scripts/joint_prior_audio_context.py','kinetalk_b0/models/joint_token_audio.py')
    protocol={'schema':SCHEMA,'smoke':args.smoke,'source':lineage,'frozen_audio':binding,
        'split':{k:[{m:clips[i][m] for m in ('clip_id','sentence','speaker','emotion')} for i in ids] for k,ids in split.items()},
        'protocol_sha256':old.sha(Path(__file__).resolve().parents[1]/'docs/JOINT_MOTION_PRIOR_PROTOCOL_20260918.md'),
        'code_sha256':{f:old.sha(Path(__file__).resolve().parents[1]/f) for f in code},'test_loaded':False,'dev405_indexed':False}
    old.save_json(args.output/'protocol.json',protocol)
    scales=raw_scales(clips,split['fit']);units=d.extract_units(clips,split['fit'])
    if len(units) < (16 if args.smoke else 128):
        raise ValueError('Insufficient fit units for the declared dictionary size')
    dictionary=d.fit_dictionary(units,k=16 if args.smoke else 128,seed=20260918)
    torch.save(dictionary,args.output/'dictionary.pt');old.save_json(args.output/'scales.json',scales.tolist())
    torch.save({clips[i]['clip_id']:{'global':clips[i]['global'],'local':clips[i]['local'],'anchor_upper':clips[i]['anchor_upper']}
        for i in selected},args.output/'frozen_context.pt')
    old.save_json(args.output/'coverage_calibration.json',prior_coverage(clips,split['calibration'],dictionary,scales))
    uniform,_=evaluate(clips,split['calibration'],dictionary,scales,(0.,0.),mode='uniform')
    dump_eval(args.output,'cal_uniform',uniform,{})
    trials=[];best=None
    settings=[(1.,1.)] if args.smoke else [(h,g) for h in (1.,4.,16.) for g in (1.,4.,16.)]
    for setting in settings:
        result,_=evaluate(clips,split['calibration'],dictionary,scales,setting,mode='global')
        gate=quality_gate(result,uniform)
        key='cal_global_h'+str(int(setting[0]))+'_g'+str(int(setting[1]))
        dump_eval(args.output,key,result,{})
        row={'setting':list(setting),'gate':gate,'joint_centered_es':result['summary']['joint_fair_es']['centered']}
        trials.append(row);print('PRIOR_CALIBRATION',json.dumps(row),flush=True)
        if gate['passed'] and (best is None or row['joint_centered_es']<best['joint_centered_es']):best=row
    old.save_json(args.output/'calibration_search.json',trials)
    chosen=best or min(trials,key=lambda x:x['joint_centered_es'])
    setting=chosen['setting']
    history_trials=[]
    for h in ((1.,) if args.smoke else (1.,4.,16.)):
        result,_=evaluate(clips,split['calibration'],dictionary,scales,(h,0.),mode='history')
        dump_eval(args.output,'cal_history_h'+str(int(h)),result,{})
        history_trials.append({'setting':[h,0.],'score':result['summary']['joint_fair_es']['centered'],'gate':quality_gate(result,uniform)})
    history_setting=min(history_trials,key=lambda x:(not x['gate']['passed'],x['score']))['setting']
    old.save_json(args.output/'selection.json',{'chosen':chosen,'calibration_passed':best is not None,'history_trials':history_trials,
        'selection_scope':'calibration only; fallback smallest ES is diagnostic if all fail'})
    residual=None;stats=None;scale=0.;train_losses=[]
    if local_training_allowed(best is not None):
        x,prior,y=training_examples(clips,split['fit'],dictionary,setting)
        stats={'mean':x.mean(0),'std':np.maximum(x.std(0),.01)}
        residual=BoundedTokenResidual(input_dim=x.shape[-1],hidden=64,k=len(dictionary['medoids'])).to(args.device)
        optimizer=torch.optim.AdamW(residual.parameters(),lr=.0003,weight_decay=.01)
        xx=torch.tensor((x-stats['mean'])/stats['std'],device=args.device)
        pp=torch.tensor(prior,device=args.device);yy=torch.tensor(y,device=args.device)
        rng=np.random.default_rng(20260918);orderhash=hashlib.sha256()
        for epoch in range(2 if args.smoke else 30):
            order=rng.permutation(len(x));orderhash.update(order.tobytes());values=[]
            for start in range(0,len(order),256):
                ix=torch.as_tensor(order[start:start+256],device=args.device)
                loss=torch.nn.functional.cross_entropy(pp[ix]+residual(xx[ix]),yy[ix])
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(residual.parameters(),1.);optimizer.step()
                values.append(float(loss.detach()))
            train_losses.append({'epoch':epoch+1,'loss':float(np.mean(values)),'updates':len(values)})
            old.save_json(args.output/'losses.json',train_losses)
            print('LOCAL_EPOCH',epoch+1,train_losses[-1]['loss'],flush=True)
        residual.eval();torch.save({'model':residual.state_dict(),'stats':stats,'dictionary_sha256':old.sha(args.output/'dictionary.pt'),
            'fit_clip_ids':[clips[i]['clip_id'] for i in split['fit']],'order_sha256':orderhash.hexdigest()},args.output/'local_final.pt')
        local_choices=[]
        for v in (0.,.25,.5,1.):
            report,_=evaluate(clips,split['calibration'],dictionary,scales,setting,mode='global',residual=residual,stats=stats,local_scale=v)
            dump_eval(args.output,'cal_local_'+str(v),report,{})
            local_choices.append({'scale':v,'score':report['summary']['joint_fair_es']['centered'],'gate':quality_gate(report,uniform)})
        feasible=[r for r in local_choices if r['gate']['passed']]
        scale=min(feasible,key=lambda r:r['score'])['scale'] if feasible else 0.
        old.save_json(args.output/'local_selection.json',{'choices':local_choices,'scale':scale})
    reports={}
    # Confirmation is first opened for scoring only after all choices are fixed.
    for label,mode,sett,net,weight in (('uniform','uniform',(0.,0.),None,0.),('history','history',history_setting,None,0.),
            ('global','global',setting,None,0.),('local','global',setting,residual,scale)):
        report,curves=evaluate(clips,split['confirmation'],dictionary,scales,sett,mode=mode,residual=net,stats=stats,local_scale=weight,save_draws=True)
        reports[label]=report;dump_eval(args.output,'confirmation_'+label,report,curves)
    # These controls preserve the same frozen global context and motion prior;
    # only the local sequence is removed or replaced.  They are required to
    # distinguish useful audio timing from generic stochastic token changes.
    if residual is not None and scale > 0:
        for intervention in ('static', 'reverse', 'mismatch'):
            report,curves=evaluate(clips,split['confirmation'],dictionary,scales,setting,mode='global',
                residual=residual,stats=stats,local_scale=scale,intervention=intervention,save_draws=True)
            reports[intervention]=report
            dump_eval(args.output,'confirmation_local_'+intervention,report,curves)
    gate=quality_gate(reports['global'],reports['uniform'])
    if residual is not None and scale > 0 and 'mismatch' in reports:
        audio_result=audio_gate(reports,gate,scale)
    else:
        audio_result={'passed':False,'reason':'local_controller_not_trained'}
    gain=audio_result.get('global_gain', bootstrap_gain(reports['local'],reports['global']))
    audio_passed=bool(audio_result.get('passed',False))
    old.save_json(args.output/'gate.json',{'motion_calibration_passed':best is not None,'motion_confirmation':gate,
        'audio_passed_preliminary':audio_passed,'audio_gate':audio_result,'audio_gain':gain,'local_scale':scale,
        'integration_authorized_by_gates':False,'note':'Additional intervention and joint quality audit required for integration even if preliminary gain passes'})
    old.save_json(args.output/'status.json', final_status(smoke=args.smoke,
        seconds=time.monotonic()-started, local_trained=residual is not None,
        local_epochs=len(train_losses), motion_calibration_passed=best is not None,
        motion_confirmation_passed=bool(best is not None and gate['passed'])))
    print('JOINT_PRIOR_COMPLETE',json.dumps(gate),flush=True)


if __name__=='__main__':main()
