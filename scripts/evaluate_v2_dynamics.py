"""Descriptive, mask/gap-aware v2 dynamics diagnostics on validation only.

These ARKit coefficient statistics are custom diagnostics, not SubtleTalk FDD.
Observed event-history likelihood is reported separately from teacher-free
sampled/modal generation. Every predeclared seed is retained; there is no
best-of-N selection. The motion parser makes weak labels, not emotional truth.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVENT_NAMES = ("eyelid_closure", "brow_emphasis", "cheek_nose_emphasis", "lip_tension")
STATE_NAMES = ("smile_minus_frown", "mouth_tension", "brow_lift", "brow_contraction",
               "eye_open_minus_squint", "cheek_nose", "dimple", "lip_shrug",
               "vertical_lip_deformation", "rounding")
REGION_CHANNELS = {
    "eyelid_blink": ("eyeBlinkLeft", "eyeBlinkRight"),
    "eye_open_squint": ("eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight"),
    "brow": ("browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "cheek_nose": ("cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight"),
}
SCHEMA = "v2-dynamics-validation-diagnostic-v1"


def finite(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite(obj.tolist())
    if isinstance(obj, np.generic):
        return finite(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def contiguous_runs(valid, times, step=.04):
    valid, times = np.asarray(valid, bool), np.asarray(times, float)
    if valid.ndim != 1 or times.shape != valid.shape:
        raise ValueError("Expected valid and source times [T]")
    labels = np.full(len(valid), -1, np.int64)
    runs = []
    start = 0
    while start < len(valid):
        if not valid[start]:
            start += 1; continue
        end = start + 1
        while end < len(valid) and valid[end] and np.isclose(times[end] - times[end-1], step, rtol=1e-4, atol=1e-7):
            end += 1
        labels[start:end] = len(runs); runs.append((start, end)); start = end
    return runs, labels


def regional_dynamics(motion, valid, times, channel_mask, channels):
    """Within-run population std and mean absolute native derivatives.

    No derivative bridges an invalid frame or clock discontinuity. The std is
    centered within each contiguous run, excluding an across-gap posture jump.
    Units are coefficient, coefficient/s and coefficient/s^2.
    """
    x = np.asarray(motion, float); valid = np.asarray(valid, bool)
    times = np.asarray(times, float); mask = np.asarray(channel_mask, bool)
    if x.ndim != 2 or x.shape[1] != 52 or mask.shape != (52,):
        raise ValueError("Expected motion[T,52] and channel_mask[52]")
    use = np.array([i for i in channels if mask[i]], dtype=int)
    empty = {"within_run_population_std": 0., "mean_abs_velocity_per_second": 0.,
             "mean_abs_acceleration_per_second2": 0., "observed_channel_frames": 0,
             "observed_channel_pairs": 0, "observed_channel_triples": 0}
    if not len(use):
        return empty
    if not np.isfinite(x[valid][:, use]).all():
        raise ValueError("Nonfinite observed regional coefficients")
    std_sum = std_count = velocity_sum = velocity_count = accel_sum = accel_count = 0.
    runs, _ = contiguous_runs(valid, times)
    for lo, hi in runs:
        run = x[lo:hi][:, use]; n = len(run)
        std_sum += float(run.std(axis=0, ddof=0).sum()) * n; std_count += n * len(use)
        if n >= 2:
            velocity = np.diff(run, axis=0) / np.diff(times[lo:hi])[:, None]
            velocity_sum += float(np.abs(velocity).sum()); velocity_count += velocity.size
            if n >= 3:
                # Adjacent .04s centers; no compaction or time rescaling.
                accel = np.diff(velocity, axis=0) / ((times[lo+2:hi] - times[lo:hi-2]) / 2)[:, None]
                accel_sum += float(np.abs(accel).sum()); accel_count += accel.size
    return {"within_run_population_std": std_sum / max(std_count, 1),
            "mean_abs_velocity_per_second": velocity_sum / max(velocity_count, 1),
            "mean_abs_acceleration_per_second2": accel_sum / max(accel_count, 1),
            "observed_channel_frames": int(std_count), "observed_channel_pairs": int(velocity_count),
            "observed_channel_triples": int(accel_count)}


def events_from_tensor(events, valid, times, *, duration_valid=None):
    """Parse complete native event runs from [T,4,3] active/phase/amplitude.

    Phase reset detects consecutive schedule events with no intervening idle
    frame. Events touching a run or gap boundary are censored, not counted as
    complete onsets/durations. The caller may supply parser duration_valid.
    """
    events = np.asarray(events, float)
    if events.ndim == 2 and events.shape[-1] == 12:
        events = events.reshape(-1, 4, 3)
    if events.shape != (len(valid), 4, 3):
        raise ValueError("Expected event tensor [T,4,3] or [T,12]")
    runs, _ = contiguous_runs(valid, times)
    result = [[] for _ in range(4)]
    censored = np.zeros(4, int)
    for rid, (lo, hi) in enumerate(runs):
        for group in range(4):
            t = lo
            while t < hi:
                if events[t, group, 0] <= .5:
                    t += 1; continue
                end = t + 1
                while end < hi and events[end, group, 0] > .5 and events[end, group, 1] >= events[end-1, group, 1] - 1e-7:
                    end += 1
                complete = t > lo and end < hi
                if duration_valid is not None:
                    complete = complete and bool(duration_valid[t, group])
                if complete:
                    result[group].append({"frame": int(t), "time": float(times[t]), "run": rid,
                                          "duration_seconds": (end - t) * .04,
                                          "amplitude": float(events[t:end, group, 2].max())})
                else:
                    censored[group] += 1
                t = end
    return result, censored


def match_events(reference: list[dict], predicted: list[dict], tolerance_seconds=.08):
    """Maximum ordered one-to-one onset matching inside the same native run."""
    if tolerance_seconds < 0:
        raise ValueError("event tolerance must be nonnegative")
    # DP maximizes match count, then minimizes total onset error. No one event
    # can explain two neighbors; no match can cross a missing-data gap.
    n, m = len(reference), len(predicted)
    dp = [[(0, 0.) for _ in range(m+1)] for _ in range(n+1)]
    action = [[None for _ in range(m+1)] for _ in range(n+1)]
    def better(a, b):
        return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1] - 1e-12)
    for i in range(1, n+1):
        for j in range(1, m+1):
            best, op = dp[i-1][j], "r"
            if better(dp[i][j-1], best): best, op = dp[i][j-1], "p"
            r, p = reference[i-1], predicted[j-1]
            err = abs(r["time"] - p["time"])
            if r["run"] == p["run"] and err <= tolerance_seconds + 1e-9:
                candidate = (dp[i-1][j-1][0]+1, dp[i-1][j-1][1]+err)
                if better(candidate, best): best, op = candidate, "match"
            dp[i][j], action[i][j] = best, op
    pairs = []; i, j = n, m
    while i and j:
        op = action[i][j]
        if op == "match": pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif op == "r": i -= 1
        else: j -= 1
    pairs.reverse()
    return pairs


def event_execution(schedule, realized, tolerance_seconds=.08):
    rows = {}
    for group, name in enumerate(EVENT_NAMES):
        pairs = match_events(schedule[group], realized[group], tolerance_seconds)
        tp, nr, npred = len(pairs), len(schedule[group]), len(realized[group])
        rows[name] = {"schedule_events": nr, "realized_events": npred, "matched_events": tp,
                      "onset_precision": tp/max(npred, 1), "onset_recall": tp/max(nr, 1),
                      "onset_f1": 2*tp/max(nr+npred, 1),
                      "matched_onset_abs_seconds_sum": sum(abs(schedule[group][i]["time"]-realized[group][j]["time"]) for i,j in pairs),
                      "matched_duration_abs_seconds_sum": sum(abs(schedule[group][i]["duration_seconds"]-realized[group][j]["duration_seconds"]) for i,j in pairs),
                      "matched_amplitude_abs_sum": sum(abs(schedule[group][i]["amplitude"]-realized[group][j]["amplitude"]) for i,j in pairs)}
    return rows


def probability_scores(hazard_logits, duration_logits, targets):
    """Weak-label calibration under observed past events, diagnostic only."""
    logits = torch.as_tensor(hazard_logits).double()
    onset = torch.as_tensor(targets['onset']).bool()
    onset_valid = torch.as_tensor(targets['onset_valid']).bool()
    duration_logits = torch.as_tensor(duration_logits).double()
    durations = torch.as_tensor(targets['duration']).long()
    duration_valid = torch.as_tensor(targets['duration_valid']).bool()
    logits = torch.where(onset_valid, logits, 0.)
    duration_logits = torch.where(duration_valid[...,None], duration_logits, 0.)
    durations = torch.where(duration_valid, durations, 0)
    if not torch.isfinite(logits).all() or not torch.isfinite(duration_logits).all():
        raise ValueError('Nonfinite observed conditional event probabilities')
    probabilities = logits.sigmoid()
    nll = F.binary_cross_entropy_with_logits(logits, onset.double(), reduction='none')
    duration_nll = F.cross_entropy(duration_logits.reshape(-1,duration_logits.shape[-1]),durations.flatten(),reduction='none').reshape_as(durations)
    result = {}
    for g, name in enumerate(EVENT_NAMES):
        valid = onset_valid[...,g]
        dvalid = duration_valid[...,g]
        count, dcount = int(valid.sum()), int(dvalid.sum())
        result[name] = {"onset_count": count, "positive_onsets": int((onset[...,g]&valid).sum()),
                       "onset_probability_sum": float(torch.where(valid,probabilities[...,g],0.).sum()),
                       "brier_sum": float(torch.where(valid,(probabilities[...,g]-onset[...,g].double()).square(),0.).sum()),
                       "binary_nll_sum": float(torch.where(valid,nll[...,g],0.).sum()),
                       "duration_count": dcount,
                       "duration_conditional_nll_sum": float(torch.where(dvalid,duration_nll[...,g],0.).sum())}
    return result


def expression_direction_error(pred_state, target_state, quality, epsilon=.01):
    pred = np.asarray(pred_state, float); target = np.asarray(target_state, float); quality = np.asarray(quality, bool)
    if pred.shape != target.shape or quality.shape != target.shape or pred.shape[-1] != 10:
        raise ValueError("Expected expression state and quality [T,10]")
    error = np.where(quality, np.abs(pred-target), 0.)
    axes = []
    for k, name in enumerate(STATE_NAMES):
        directional = quality[:,k] & (np.abs(target[:,k]) > epsilon)
        axes.append({"axis": name, "mae": float(error[:,k].sum()/max(int(quality[:,k].sum()),1)),
                     "observations": int(quality[:,k].sum()), "directional_observations": int(directional.sum()),
                     "nonzero_target_sign_error": float(np.sum(directional & (np.sign(pred[:,k]) != np.sign(target[:,k])))/max(int(directional.sum()),1))})
    good = quality.all(-1) & (np.linalg.norm(target,axis=-1) > epsilon)
    pnorm = np.linalg.norm(pred,axis=-1); tnorm = np.linalg.norm(target,axis=-1)
    cosine = np.sum(pred*target,axis=-1) / np.maximum(pnorm*tnorm,1e-12)
    return {"axis_metrics": axes, "mean_axis_mae": float(error.sum()/max(int(quality.sum()),1)),
            "direction_cosine_error": float(np.where(good,1.-cosine,0.).sum()/max(int(good.sum()),1)),
            "direction_frames": int(good.sum()), "epsilon": epsilon}


def wasserstein_1d(a, b):
    """Exact empirical 1-Wasserstein distance; undefined for empty samples."""
    a, b = np.sort(np.asarray(a,float)), np.sort(np.asarray(b,float))
    if not len(a) or not len(b): return None
    support = np.unique(np.r_[a,b])
    if len(support) < 2: return 0.
    left = support[:-1]
    return float(np.sum(np.abs(np.searchsorted(a,left,side='right')/len(a)-np.searchsorted(b,left,side='right')/len(b))*np.diff(support)))


def summarize_event_distributions(groups, seconds, gt_groups=None, gt_seconds=None):
    out = {}
    for g, name in enumerate(EVENT_NAMES):
        duration = [r['duration_seconds'] for r in groups[g]]
        amplitude = [r['amplitude'] for r in groups[g]]
        row = {'complete_events':len(groups[g]),'complete_events_per_minute':len(groups[g])/max(seconds,1e-12)*60,
               'duration_seconds_mean':float(np.mean(duration)) if duration else None,
               'amplitude_mean':float(np.mean(amplitude)) if amplitude else None}
        if gt_groups is not None:
            target = gt_groups[g]
            row.update(complete_events_per_minute_abs_error=abs(row['complete_events_per_minute']-len(target)/max(gt_seconds,1e-12)*60),
                       duration_seconds_wasserstein1=wasserstein_1d(duration,[r['duration_seconds'] for r in target]),
                       amplitude_wasserstein1=wasserstein_1d(amplitude,[r['amplitude'] for r in target]))
        out[name] = row
    return out


def seed_diversity(motions, events, valid, channel_mask, upper_indices):
    motions = np.asarray(motions,float); events = np.asarray(events,float).reshape(len(motions),len(valid),4,3)
    mask = np.asarray(valid,bool)[:,None] & np.asarray(channel_mask,bool)[None,:]
    region = np.zeros(52,bool); region[list(upper_indices)] = True; observed = mask & region[None]
    motion_errors, event_errors = [], []
    for i,j in itertools.combinations(range(len(motions)),2):
        motion_errors.append(float(np.where(observed,np.abs(motions[i]-motions[j]),0.).sum()/max(int(observed.sum()),1)))
        event_errors.append(float(np.abs(events[i,:, :,0]-events[j,:,:,0])[valid].mean()) if np.any(valid) else 0.)
    return {'seed_pairs':len(motion_errors),'upper_coefficient_pairwise_mae':float(np.mean(motion_errors)) if motion_errors else 0.,
            'schedule_active_pairwise_disagreement':float(np.mean(event_errors)) if event_errors else 0.}


def aggregate_numeric(rows):
    if not rows: return {}
    result = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
        if vals: result[key] = float(np.mean(vals))
    return result


def finalize_probability(rows):
    result = {}
    for name in EVENT_NAMES:
        summed = {k:sum(r[name][k] for r in rows) for k in rows[0][name]} if rows else {}
        n, d = summed.get('onset_count',0), summed.get('duration_count',0)
        result[name] = {**summed,'brier':summed.get('brier_sum',0)/max(n,1),
                        'binary_nll':summed.get('binary_nll_sum',0)/max(n,1),
                        'weak_onset_prevalence':summed.get('positive_onsets',0)/max(n,1),
                        'mean_onset_probability':summed.get('onset_probability_sum',0)/max(n,1),
                        'duration_conditional_nll':summed.get('duration_conditional_nll_sum',0)/max(d,1)}
    return result


def aggregate_direction(rows):
    return {'mean_axis_mae':float(np.mean([r['mean_axis_mae'] for r in rows])) if rows else 0.,
            'direction_cosine_error':float(np.mean([r['direction_cosine_error'] for r in rows])) if rows else 0.,
            'axis_metrics':[{"axis": name, **aggregate_numeric([r['axis_metrics'][k] for r in rows])} for k,name in enumerate(STATE_NAMES)]}


def finalize_execution(rows):
    result = {}
    for name in EVENT_NAMES:
        sums={key:sum(r[name][key] for r in rows) for key in rows[0][name] if key not in ('onset_precision','onset_recall','onset_f1')} if rows else {}
        tp,nr,npred=sums.get('matched_events',0),sums.get('schedule_events',0),sums.get('realized_events',0)
        result[name]={**sums,'onset_precision':tp/max(npred,1),'onset_recall':tp/max(nr,1),'onset_f1':2*tp/max(nr+npred,1),
                      'matched_onset_abs_seconds':sums.get('matched_onset_abs_seconds_sum',0)/max(tp,1),
                      'matched_duration_abs_seconds':sums.get('matched_duration_abs_seconds_sum',0)/max(tp,1),
                      'matched_amplitude_abs':sums.get('matched_amplitude_abs_sum',0)/max(tp,1)}
    return result


@torch.no_grad()
def evaluate(args):
    from kinetalk_v2.unified_model import UnifiedConfig, UnifiedExpressionModel, teacher_targets
    from kinetalk_b0.models.expression_response import ResponseConfig, ExpressionResponse
    from scripts.train_expression_response import configure, load_runtime, sha, state_digest
    from scripts.train_unified_v2 import build_base_cache, attach_base, select_balanced
    from scripts.train_full_staged import base_forward
    configure(args.selection_seed); device=torch.device(args.device); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    binding=json.loads(Path(args.binding).read_text(encoding='utf-8'))
    data,base=load_runtime(binding,device); q=data['splits']['validation']
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=UnifiedExpressionModel(UnifiedConfig(**ck['config']),ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'],strict=True); model.eval().requires_grad_(False)
    if sha(args.baseline)!=args.baseline_sha256: raise ValueError('Phase53 hash differs from immutable binding')
    old=torch.load(args.baseline,map_location='cpu',weights_only=False)
    baseline=ExpressionResponse(ResponseConfig(**old['config']),old['model']['feature_mean'],old['model']['feature_std'],old['model']['scales']).to(device)
    baseline.load_state_dict(old['model'],strict=True); baseline.eval().requires_grad_(False)
    initial=(state_digest(model),state_digest(baseline),state_digest(base.stage1))
    ids=select_balanced(q,list(range(len(q['_lengths']))),args.limit,args.selection_seed)
    cache=build_base_cache(q,base,ids,device)
    if sha(binding['rig'])!=binding['rig_sha256']: raise ValueError('Rig hash differs')
    with np.load(binding['rig'],allow_pickle=False) as rig: channels=[str(x).lower() for x in rig['channels']]
    regions={name:[channels.index(s.lower()) for s in names] for name,names in REGION_CHANNELS.items()}
    upper=sorted(set(i for ix in regions.values() for i in ix))
    methods=['GT','phase53_default','v2_modal']+[f'v2_sample_seed{s}' for s in args.seeds]
    regional={m:{r:[] for r in regions} for m in methods}; state_rows={m:[] for m in methods if m!='GT'}
    distribution={m:[[] for _ in range(4)] for m in methods}; censored={m:np.zeros(4,int) for m in methods}
    execution={m:[] for m in methods if m.startswith('v2_')}; probability=[]; student_direction=[]; diversity=[]; refs_record=[]
    clip_records=[]; total_seconds=0.
    for start in range(0,len(ids),args.batch_size):
        use=ids[start:start+args.batch_size]
        batch=attach_base(q.batch(torch.tensor(use),device),use,cache)
        support=[data['refs'][int(q['speaker_id'][i])] for i in use]
        for i,ref in zip(use,support):
            if q['clip_id'][i] in ref['clip_id'] or q['sentence_id'][i] in ref['sentence_id']:
                raise ValueError('Query and fixed independent enrollment reference overlap')
        refs={k:torch.stack([r[k][0] for r in support]).to(device) for k in ('motion','valid','channel_mask')}
        old_refs={k:torch.stack([r[k] for r in support]).to(device) for k in ('motion','valid','channel_mask')}
        ref_content=torch.stack([r['content'] for r in support]).to(device)
        ref_base=base_forward(base,ref_content.flatten(0,1),old_refs['valid'].flatten(0,1))['b0']
        old_refs['b0']=ref_base.reshape(*ref_content.shape[:3],52)
        old_prior=baseline.audio_prior(batch['audio_features'],batch['valid'])
        old_style=baseline.encode_style(old_refs)['code']
        old_y=baseline.decode(batch['b0'],old_prior,old_style,batch['valid'])
        targets=teacher_targets(batch['motion'],batch['b0'],batch['valid'],batch['channel_mask'],batch['times'])
        student=model.audio_state(batch['audio_features'],batch['valid']); style=model.encode_style(refs)
        # The model observes GT past events only in this explicitly separate
        # conditional-likelihood diagnostic. No observed-mode decoded motion is
        # evaluated or passed to the deployment generation methods below.
        conditional=model.events(student['hidden'],student['state'],style,batch['valid'],targets['events'],mode='observed',times=batch['times'])
        probability.append(probability_scores(conditional['hazard_logits'],conditional['duration_logits'],targets))
        predictions={'GT':{'motion':batch['motion']},'phase53_default':{'motion':old_y},
                     'v2_modal':model.predict(batch['audio_features'],batch['h0'],batch['b0'],batch['valid'],refs,times=batch['times'],mode='modal')}
        for seed in args.seeds:
            predictions[f'v2_sample_seed{seed}']=model.predict(batch['audio_features'],batch['h0'],batch['b0'],batch['valid'],refs,times=batch['times'],seed=seed+start)
        parsed={m:targets if m=='GT' else teacher_targets(v['motion'],batch['b0'],batch['valid'],batch['channel_mask'],batch['times']) for m,v in predictions.items()}
        for j,i in enumerate(use):
            n=int(q['_lengths'][i]); valid=batch['valid'][j,:n].cpu().numpy(); times=batch['times'][j,:n].cpu().numpy(); mask=batch['channel_mask'][j].cpu().numpy()
            total_seconds+=int(valid.sum())*.04
            refs_record.append({'clip_id':q['clip_id'][i],'single_reference_id':support[j]['clip_id'][0],
                                'phase53_default_reference_ids':list(support[j]['clip_id'])})
            student_direction.append(expression_direction_error(student['state'][j,:n].cpu().numpy(),targets['state'][j,:n].cpu().numpy(),targets['quality'][j,:n].cpu().numpy()))
            row={'clip_id':q['clip_id'][i],'index':i,'valid_frames':int(valid.sum()),'events':{},'execution':{}}
            for name,pred in predictions.items():
                y=pred['motion'][j,:n].cpu().numpy(); p=parsed[name]
                for region,ix in regions.items(): regional[name][region].append(regional_dynamics(y,valid,times,mask,ix))
                event_list,event_censored=events_from_tensor(p['events'][j,:n].cpu().numpy(),valid,times,duration_valid=p['duration_valid'][j,:n].cpu().numpy())
                for g in range(4): distribution[name][g].extend(event_list[g])
                censored[name]+=event_censored
                row['events'][name]=[len(v) for v in event_list]
                if name!='GT': state_rows[name].append(expression_direction_error(p['state'][j,:n].cpu().numpy(),targets['state'][j,:n].cpu().numpy(),targets['quality'][j,:n].cpu().numpy()))
                if name.startswith('v2_'):
                    scheduled=pred['schedule']['events'][j,:n].reshape(n,4,3)
                    scheduled=torch.where(targets['event_valid'][j,:n,:,None],scheduled,0.)
                    schedule,_=events_from_tensor(scheduled.cpu().numpy(),valid,times)
                    matched=event_execution(schedule,event_list,args.tolerance_frames*.04)
                    execution[name].append(matched);row['execution'][name]=matched
            seed_keys=[f'v2_sample_seed{s}' for s in args.seeds]
            diversity.append(seed_diversity(np.stack([predictions[k]['motion'][j,:n].cpu().numpy() for k in seed_keys]),
                                           np.stack([predictions[k]['schedule']['events'][j,:n].cpu().numpy() for k in seed_keys]),valid,mask,upper))
            clip_records.append(row)
        status={'status':'diagnosing','clips':start+len(use),'total':len(ids)}
        write_json(out/'state.json',status); print(json.dumps(status),flush=True)
    result={}
    for method in methods:
        result[method]={'regional_dynamics_macro_clip':{r:aggregate_numeric(regional[method][r]) for r in regions},
                        'realized_event_distributions':summarize_event_distributions(distribution[method],total_seconds,
                            None if method=='GT' else distribution['GT'],total_seconds),
                        'censored_event_counts':dict(zip(EVENT_NAMES,censored[method].tolist()))}
        if method!='GT': result[method]['expression_direction']=aggregate_direction(state_rows[method])
        if method in execution: result[method]['schedule_execution_consistency']=finalize_execution(execution[method])
    if initial!=(state_digest(model),state_digest(baseline),state_digest(base.stage1)): raise ValueError('Diagnostic mutated frozen model')
    report={'schema':SCHEMA,'scope':'descriptive_development_only','validation_universe':len(q['_lengths']),
            'clips':len(ids),'selected_indices':ids,'clip_ids':[q['clip_id'][i] for i in ids],'selection_seed':args.selection_seed,
            'selection_policy':'deterministic balanced round-robin speaker/emotion from validation universe',
            'sample_seeds':args.seeds,'batch_size':args.batch_size,'seed_policy':'requested seed plus deterministic batch start; no sample selection',
            'tolerance_frames':args.tolerance_frames,'prediction_policy':'raw coefficients, no clamp or smoothing',
            'expression_state_axes':list(STATE_NAMES),'event_groups':list(EVENT_NAMES),
            'coefficient_regions':{k:list(v) for k,v in REGION_CHANNELS.items()},
            'conditional_probability_mask':'teacher_targets.onset_valid; duration_valid at complete positive onsets; same weak-label mask as training',
            'checkpoint_sha256':sha(args.checkpoint),'baseline_sha256':sha(args.baseline),'binding_sha256':sha(args.binding),
            'training_protocol':ck.get('protocol',{}),'evaluation_phonetic_constraints_enabled':False,
            'test_loaded':False,'observed_history_probability_diagnostic':finalize_probability(probability),
            'student_expression_direction':aggregate_direction(student_direction),'seed_diversity_macro_clip':aggregate_numeric(diversity),
            'results':result,'references':refs_record,'valid_seconds':total_seconds,
            'limitations':['Custom coefficient run-std/velocity/acceleration are not SubtleTalk FDD or established perceptual dynamic scores.',
                'Motion parser thresholds/anchors are shared with training: weak pseudo labels are not an independent external dynamic or emotional ground truth.',
                'Observed-history Brier/NLL measures conditional probability given observed past events, not teacher-free rollout likelihood or generated-motion accuracy.',
                'No best-of-N, ground-truth event inputs, or query-motion teacher inputs enter sampled/modal generation.',
                'Eyelid closure is a motion event, not emotional accuracy. Expression direction is an ARKit-coordinate diagnostic, not psychological VA.',
                'Only complete within-run events enter distributions/matching; boundary events are censored. Rare-event empty Wasserstein distances are null.',
                'Phase53 default uses its original independent reference set; v2 uses fixed first independent reference. Training budgets/architectures differ.',
                'No phonetic constraints are supplied by this evaluator; it evaluates the current unconstrained implementation explicitly.']}
    write_json(out/'report.json',report);write_json(out/'per_clip.json',clip_records)
    write_json(out/'complete.json',{'status':'complete','clips':len(ids),'checkpoint_sha256':sha(args.checkpoint),'sample_seeds':args.seeds})


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline',default='/root/kinetalk_phase53_receiver_20261009/style_only/seed47/final.pt')
    p.add_argument('--baseline-sha256',default='80a7ca69b0f868bad6361381e1ca8e4e168ca227a0548221f8efa71b42169c41')
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--selection-seed',type=int,default=47)
    p.add_argument('--seeds',type=int,nargs='+',default=[47,91,137]);p.add_argument('--tolerance-frames',type=int,default=2)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.limit<0 or args.batch_size<1 or args.tolerance_frames<0 or len(set(args.seeds))!=len(args.seeds):
        raise SystemExit('Invalid limit/batch/tolerance or repeated sample seeds')
    evaluate(args)
