"""Reference-relative intensity and learned upper9 execution, TRAIN selection.

This experimental path releases the old upper mean and never changes the other
43 coefficients. The two-stage oracle/student distinction remains explicit.
No labels, query motion, or query-derived mean enter deployment conditions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.reference_intensity_decoder import (
    ReferenceIntensityDecoder, ReferenceIntensityStudent, neutral_relative_intensity)
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from kinetalk_b0.models.temporal_motion_carrier import smooth_motion_carrier
from scripts.reference_decoder_data import load_reference_context
from scripts.train_audio_regional_envelope import (
    _fold, _diverse_ids, _feature_reverse, _feature_shuffle, _static_envelope,
    _envelope_metrics, _paired_cluster_ci, _upper_motion_diagnostics)
from scripts.train_full_staged import subset, NOT_UPPER
from scripts.train_isolated_audio_state import old_prediction
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, canonical_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

SOURCE_SHA = 'be975208b978309e7eff328fb07418e413e49a8fe7f76b9bac67c341083e0fbb'
UPPER = list(UPPER_INDICES)
SCHEMA = 'reference_intensity_decoder_v1'
WINDOW = 5


def mean_valid(x, valid):
    return torch.where(valid[..., None], x, 0.).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def equal_clip_loss(error, valid):
    return (torch.where(valid[..., None], error, 0.).sum((1, 2)) /
            (valid.sum(1) * error.shape[-1]).clamp_min(1)).mean()


@torch.no_grad()
def fit_statistics(q, ids, device, batch_size=32, pca_frames=12000):
    if len(ids) < 1 or pca_frames < 2 or not q['valid'][ids].any(1).all():
        raise ValueError('Statistics require nonempty clips and at least two PCA frames')
    if not (q['channel_mask'][ids][:, UPPER] & q['anchor_valid'][ids][:, UPPER]).all():
        raise ValueError('Statistics require observed upper9 and independent anchors')
    width = q['audio_features'].shape[-1]
    total = torch.zeros(width, dtype=torch.float64)
    squares = total.clone(); motion_sq = torch.zeros(9, dtype=torch.float64); count = 0
    samples = []
    per_clip = max(4, pca_frames // len(ids))
    for ix in ids.split(batch_size):
        valid = q['valid'][ix]
        x = q['audio_features'][ix]
        observed = x[valid].double()
        if not torch.isfinite(observed).all():
            raise ValueError('Nonfinite observed audio')
        total += observed.sum(0); squares += observed.square().sum(0); count += len(observed)
        delta = torch.where(valid[..., None], q['motion'][ix][..., UPPER] - q['anchors'][ix][:, None, UPPER], 0.)
        motion_sq += delta.double().square().sum((0, 1))
        for row in range(len(ix)):
            positions = valid[row].nonzero(as_tuple=True)[0]
            chosen = torch.linspace(0, len(positions) - 1, min(per_clip, len(positions))).round().long()
            samples.append(x[row, positions[chosen], :1536])
    feature_mean = (total / count).float()
    feature_std = (squares / count - (total / count).square()).clamp_min(0).sqrt().clamp_min(1e-4).float()
    sample = torch.cat(samples)
    if len(sample) > pca_frames:
        sample = sample[torch.linspace(0, len(sample)-1, pca_frames).round().long()]
    normalized = ((sample - feature_mean[:1536]) / feature_std[:1536]).to(device)
    if len(normalized) < 2:
        raise ValueError('At least two observed fit frames are needed for PCA')
    torch.manual_seed(20260922)
    rank = min(24, len(normalized)-1, normalized.shape[-1])
    _, _, projection = torch.pca_lowrank(normalized, q=rank, center=False, niter=3)
    projection = projection.cpu()
    reduced = []
    for ix in ids.split(batch_size):
        x = (q['audio_features'][ix][q['valid'][ix]] - feature_mean) / feature_std
        reduced.append(torch.cat((x[:, :1536] @ projection, x[:, 1536:]), -1))
    values = torch.cat(reduced)
    result = {'feature_mean': feature_mean, 'feature_std': feature_std, 'projection': projection,
              'reduced_mean': values.mean(0), 'reduced_std': values.std(0, unbiased=False).clamp_min(1e-4),
              'scales9': (motion_sq / count).sqrt().clamp_min(.02).float()}
    for field in ('global_code', 'identity_code'):
        values = q[field][ids]
        result[field+'_mean'] = values.mean(0)
        result[field+'_std'] = values.std(0, unbiased=False).clamp_min(.05)
    return result


@torch.no_grad()
def prepare_cache(q, stats, batch_size=32):
    if not (q['channel_mask'][:, UPPER] & q['anchor_valid'][:, UPPER]).all():
        raise ValueError('Observed upper9 and independent reference anchors are required')
    reduced, intensity = [], []
    for ids in torch.arange(len(q['valid'])).split(batch_size):
        v = q['valid'][ids]
        clean = torch.where(v[..., None], q['audio_features'][ids], stats['feature_mean'])
        x = (clean - stats['feature_mean']) / stats['feature_std']
        x = torch.cat((x[..., :1536] @ stats['projection'], x[..., 1536:]), -1)
        reduced.append(torch.where(v[..., None], (x-stats['reduced_mean'])/stats['reduced_std'], 0.))
        intensity.append(neutral_relative_intensity(q['motion'][ids][..., UPPER], q['anchors'][ids][:, UPPER],
                                                   stats['scales9'], v, WINDOW))
    result = {'features': torch.cat(reduced), 'intensity': torch.cat(intensity),
              'upper': q['motion'][..., UPPER], 'anchor': q['anchors'], 'valid': q['valid'],
              'clip_id': q['clip_id'], 'speaker': q['speaker'], 'sentence_id': q['sentence_id']}
    for field in ('global_code', 'identity_code'):
        result[field] = (q[field]-stats[field+'_mean'])/stats[field+'_std']
    return result


def batch(cache, ids, device):
    return {key: cache[key][ids].to(device) for key in
            ('features', 'intensity', 'upper', 'anchor', 'valid', 'global_code', 'identity_code')}


def objective(upper, b, scales, predicted_intensity=None):
    valid = b['valid']
    # Clear unobserved values before arithmetic, not after squaring: otherwise
    # NaN padding can enter gradients despite a later zero-valued mask.
    clean_upper = torch.where(valid[..., None], upper, 0.)
    clean_target = torch.where(valid[..., None], b['upper'], 0.)
    error = (clean_upper-clean_target) / scales
    recon = equal_clip_loss(error.square(), valid)
    actual = neutral_relative_intensity(upper, b['anchor'][:, UPPER], scales, valid, WINDOW)
    clean_intensity_target = torch.where(valid[..., None], b['intensity'], 0.)
    intensity = equal_clip_loss((actual-clean_intensity_target).square(), valid)
    adjacent = valid[:, 1:] & valid[:, :-1]
    velocity = equal_clip_loss((error[:, 1:]-error[:, :-1]).square(), adjacent)
    loss = recon + .25 * intensity + .1 * velocity
    if predicted_intensity is not None:
        clean_prediction = torch.where(valid[..., None], predicted_intensity, 0.)
        clean_intensity_target = torch.where(valid[..., None], b['intensity'], 0.)
        loss = loss + .5 * equal_clip_loss((clean_prediction-clean_intensity_target).square(), valid)
    return loss, {'reconstruction': recon, 'output_intensity': intensity, 'velocity': velocity}


def new_models(stats, device, seed):
    torch.manual_seed(seed)
    dimensions = {'global_dim': len(stats['global_code_mean']), 'identity_dim': len(stats['identity_code_mean'])}
    decoder = ReferenceIntensityDecoder(hidden=64, **dimensions).to(device)
    student = ReferenceIntensityStudent(input_dim=len(stats['reduced_mean']), hidden=48, **dimensions).to(device)
    return decoder, student


def epoch(decoder, student, cache, ids, stats, optimizer, stage, device, batch_size, seed):
    decoder.train(stage != 'student'); student.train(stage != 'oracle')
    order = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))]
    scales = stats['scales9'].to(device); total = 0.
    for ix in order.split(batch_size):
        b = batch(cache, ix, device)
        inp = b['intensity'] if stage == 'oracle' else student(
            b['features'], b['valid'], b['global_code'], b['identity_code'], b['anchor'])['intensity']
        generated = decoder(inp, b['valid'], b['global_code'], b['identity_code'], b['anchor'])
        loss, _ = objective(generated, b, scales, None if stage == 'oracle' else inp)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 1., error_if_nonfinite=True)
        optimizer.step(); total += float(loss.detach()) * len(ix)
    return total/len(ids)


def optimizer_for(decoder, student, stage):
    if stage not in ('oracle', 'student', 'joint'):
        raise ValueError('Unknown training stage')
    decoder.zero_grad(set_to_none=True); student.zero_grad(set_to_none=True)
    decoder.requires_grad_(stage != 'student'); student.requires_grad_(stage != 'oracle')
    if stage == 'oracle':
        groups = [{'params': decoder.parameters(), 'lr': 5e-4}]
    elif stage == 'student':
        groups = [{'params': student.parameters(), 'lr': 3e-4}]
    else:
        groups = [{'params': decoder.parameters(), 'lr': 5e-5}, {'params': student.parameters(), 'lr': 1e-4}]
    return torch.optim.AdamW(groups, weight_decay=.02)


@torch.no_grad()
def predict(decoder, student, cache, ids, device, batch_size=32, mode='full'):
    decoder.eval(); student.eval(); upper, requested = [], []
    donors = _evaluation_donors(cache, ids) if mode == 'mismatch' else None
    for ix in ids.split(batch_size):
        b = batch(cache, ix, device)
        if mode.startswith('oracle'):
            intensity = b['intensity']
            if mode == 'oracle_static': intensity = _static_envelope(intensity, b['valid'])
            elif mode == 'oracle_reverse': intensity = _feature_reverse(intensity, b['valid'])
            elif mode == 'oracle_zero': intensity = torch.zeros_like(intensity)
            elif mode != 'oracle': raise ValueError(mode)
        else:
            features = b['features']
            if mode == 'reverse': features = _feature_reverse(features, b['valid'])
            elif mode == 'shuffle': features = _feature_shuffle(features, b['valid'], ix, seed=20260922)
            elif mode == 'mismatch':
                features = mismatch_features(cache, ix, donors).to(device)
            elif mode not in ('full', 'static', 'static_intensity'):
                raise ValueError(mode)
            intensity = student(features, b['valid'], b['global_code'], b['identity_code'], b['anchor'],
                                temporal_mode='static' if mode == 'static' else 'full')['intensity']
            if mode == 'static_intensity': intensity = _static_envelope(intensity, b['valid'])
        upper.append(decoder(intensity, b['valid'], b['global_code'], b['identity_code'], b['anchor']).cpu())
        requested.append(intensity.cpu())
    return torch.cat(upper), torch.cat(requested)


def _evaluation_donors(cache, eligible_ids):
    """Mismatch uses only clips in this evaluation population, never fit donors."""
    from scripts.train_audio_regional_envelope import _donors
    pool = {'valid': cache['valid'][eligible_ids],
            'speaker': [cache['speaker'][int(i)] for i in eligible_ids],
            'sentence_id': [cache['sentence_id'][int(i)] for i in eligible_ids]}
    local = _donors(pool)
    return {int(dst): int(eligible_ids[int(src)]) for dst, src in zip(eligible_ids, local)}


def mismatch_features(cache, ids, donors):
    out = torch.zeros_like(cache['features'][ids])
    for row, i in enumerate(ids.tolist()):
        donor = int(donors[i])
        if donor == i:
            raise ValueError('Mismatch donor cannot be the query clip')
        dst, src = cache['valid'][i], cache['valid'][donor]
        x = cache['features'][donor, src].T[None]
        out[row, dst] = torch.nn.functional.interpolate(x, size=int(dst.sum()), mode='linear', align_corners=False)[0].T
    return out


def scores(upper, intensity, cache, ids, stats, *, light=True):
    valid = cache['valid'][ids]; target = cache['upper'][ids]
    scales = stats['scales9']; b = batch(cache, ids, 'cpu')
    loss, parts = objective(upper, b, scales)
    delta = torch.where(valid[..., None], upper, 0.)-torch.where(valid[..., None], target, 0.)
    centered = delta-mean_valid(delta, valid)[:, None]
    per_mse = torch.where(valid[..., None], (delta/scales).square(), 0.).sum((1, 2))/(valid.sum(1)*9)
    per_centered = torch.where(valid[..., None], (centered/scales).square(), 0.).sum((1, 2))/(valid.sum(1)*9)
    per_mean = (mean_valid(delta, valid)/scales).square().mean(-1)
    actual = neutral_relative_intensity(upper, cache['anchor'][ids][:, UPPER], scales, valid, WINDOW)
    actual_metrics = _envelope_metrics(actual, cache['intensity'][ids], valid, light=light)
    pred_metrics = _envelope_metrics(intensity, cache['intensity'][ids], valid, light=light)
    return {'selection_score': float(loss), 'normalized_upper_mse': float(per_mse.mean()),
            'normalized_centered_upper_mse': float(per_centered.mean()),
            'normalized_mean_upper_mse': float(per_mean.mean()),
            'raw_upper_mse': float(equal_clip_loss(delta.square(), valid)),
            'parts': {k: float(v) for k,v in parts.items()},
            'output_intensity': {k:v for k,v in actual_metrics.items() if k!='per_clip'},
            'predicted_intensity': {k:v for k,v in pred_metrics.items() if k!='per_clip'},
            'per_clip_normalized_mse': per_mse.tolist(), 'per_clip_centered_mse': per_centered.tolist(),
            'per_clip_mean_mse': per_mean.tolist(),
            'per_clip_output_intensity_mse': [r['mse'] for r in actual_metrics['per_clip']]}


def select_stage(decoder, student, cache, fit, cal, stats, stage, budget, args, out):
    optimizer = optimizer_for(decoder, student, stage)
    if stage == 'joint':
        u, i = predict(decoder, student, cache, cal, args.device, args.batch_size)
        best_score = scores(u, i, cache, cal, stats)['selection_score']
    else:
        best_score = float('inf')
    chosen = 0; best = (copy.deepcopy(decoder.state_dict()), copy.deepcopy(student.state_dict()))
    history = []
    for number in range(1, budget+1):
        train_loss = epoch(decoder, student, cache, fit, stats, optimizer, stage, args.device, args.batch_size, args.seed+number)
        u, i = predict(decoder, student, cache, cal, args.device, args.batch_size,
                       'oracle' if stage=='oracle' else 'full')
        report = scores(u,i,cache,cal,stats)
        record = {'stage':stage,'epoch':number,'train_loss':train_loss,
                  'calibration_score':report['selection_score'], 'calibration_upper_mse':report['raw_upper_mse'],
                  'calibration_output_corr':report['output_intensity']['correlation'],
                  'calibration_predicted_corr':report['predicted_intensity']['correlation']}
        history.append(record); print(json.dumps(record),flush=True)
        if report['selection_score'] < best_score:
            best_score,chosen = report['selection_score'],number
            best = (copy.deepcopy(decoder.state_dict()),copy.deepcopy(student.state_dict()))
        save_json(out/(stage+'_history.json'),history)
    decoder.load_state_dict(best[0]);student.load_state_dict(best[1])
    return {'epochs':chosen,'score':best_score,'budget':budget}


@torch.no_grad()
def evaluate_modes(decoder, student, cache, ids, stats, args, modes):
    reports, predictions = {},{}
    for mode in modes:
        u,i=predict(decoder,student,cache,ids,args.device,args.batch_size,mode)
        reports[mode]=scores(u,i,cache,ids,stats,light=False);predictions[mode]=u
    comparisons = paired_comparisons(reports, cache, ids)
    report = {'modes':reports,'paired_comparisons':comparisons,
              'oracle_scope':'target-informed receiver diagnostic; never audio-prediction evidence',
              'mismatch_scope':'evaluation-population donor local features; frozen recipient global/reference conditions retained',
              'static_scope':'inference interventions, not separately trained matched-static ablations',
              'mean_dynamic_scope':'raw upper reconstruction decomposed into mean and centered errors; mean improvement is not dynamic improvement',
              'paper_success_established':False}
    if 'mismatch' in modes:
        report['mismatch_donors']={cache['clip_id'][dst]:cache['clip_id'][src]
                                   for dst,src in _evaluation_donors(cache,ids).items()}
    if 'full' not in reports and 'oracle' in reports:
        report['receiver_acceptance_diagnostic'] = receiver_acceptance(reports, comparisons)
    return report,predictions


def receiver_acceptance(reports, comparisons):
    """Inspect final decoder output, never oracle-input self-correlation."""
    oracle = reports['oracle']
    checks = {'final_output_intensity_correlation_positive':
              (oracle['output_intensity']['correlation'] or -1.) > 0.}
    for control in ('oracle_static', 'oracle_reverse'):
        for metric in ('centered_upper_mse', 'intensity_mse'):
            pair = comparisons.get(control, {}).get(metric)
            checks[f'{metric}_beats_{control}_both_cluster_ci'] = bool(pair and all(
                pair[group+'_cluster']['improvement_supported'] for group in ('speaker', 'sentence')))
    return {'checks': checks, 'passed': all(checks.values()),
            'based_on': 'final decoded upper/output intensity relative to raw-motion supervision',
            'oracle_requested_intensity_self_correlation_used': False,
            'student_may_run_fixed_budget': True,
            'continuation_scope': 'budgeted student feasibility experiment; continuation does not assert receiver success',
            'paper_success_established': False}


def paired_comparisons(reports, cache, ids):
    main='full' if 'full' in reports else 'oracle'
    speakers=[cache['speaker'][int(i)] for i in ids];sentences=[cache['sentence_id'][int(i)] for i in ids]
    comparisons={name:{key:_paired_cluster_ci(reports[main][field],r[field],speakers,sentences)
                        for key,field in [('upper_mse','per_clip_normalized_mse'),('centered_upper_mse','per_clip_centered_mse'),
                                          ('mean_upper_mse','per_clip_mean_mse'),
                                          ('intensity_mse','per_clip_output_intensity_mse')]}
                 for name,r in reports.items() if name!=main}
    return comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--oracle-epochs',type=int,default=20);parser.add_argument('--student-epochs',type=int,default=30)
    parser.add_argument('--joint-epochs',type=int,default=10);parser.add_argument('--seed',type=int,default=53)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--selection-only',action='store_true')
    args=parser.parse_args()
    if min(args.batch_size,args.oracle_epochs,args.student_epochs)<1 or args.joint_epochs<0:parser.error('Invalid budget')
    if args.output.exists():raise FileExistsError('Fresh output directory required')
    args.output.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    save_json(args.output/'status.json',{'status':'loading_train','test_loaded':False,'development_loaded':False})
    data,system,audio,identities=load_reference_context(args.data,args.source,args.device,role='train',seed=args.seed,
                                                      expected_source_sha256=SOURCE_SHA)
    q=data['splits']['train'];fit,cal,fold=_fold(q)
    if args.smoke:
        fit=_diverse_ids(q,fit,32,args.seed);cal=_diverse_ids(q,cal,32,args.seed+1)
    stats=fit_statistics(q,fit,args.device,args.batch_size)
    cache=prepare_cache(q,stats,args.batch_size)
    decoder,student=new_models(stats,args.device,args.seed)
    sources={name:sha(ROOT/name) for name in ('scripts/train_reference_intensity_decoder.py',
        'scripts/reference_decoder_data.py','kinetalk_b0/models/reference_intensity_decoder.py')}
    protocol={'schema':SCHEMA,'source_sha256':SOURCE_SHA,'data':data['provenance'],'fold':fold,
        'fit_ids':fit.tolist(),'calibration_ids':cal.tolist(),'source_files':sources,
        'targets':'two-region mean absolute deviation from independent neutral enrollment, fit-only channel RMS scales, five-frame smoothing',
        'output':'bounded complete upper9; previous upper mean deliberately released; nonupper43 copied',
        'inputs':'fit-only PCA(audio1536)+prosody4, frozen audio global, neutral identity, raw anchor9; no query motion/labels',
        'condition_dimensions':{'global':decoder.global_dim,'identity':decoder.identity_dim},
        'loss':{'normalized_upper_mse':1.,'output_intensity_mse':.25,'normalized_velocity_mse':.1,'student_intensity_mse':.5},
        'selection_metric':'calibration normalized upper MSE + .25 output intensity MSE + .1 normalized velocity MSE',
        'pca_rank':len(stats['reduced_mean'])-4,'selection_only':args.selection_only,'smoke':args.smoke,
        'upper_mean_protected':False,'test_loaded':False,'source_seen_full_train':True,
        'statistics_scope':'all feature/channel/PCA/global/identity statistics fit on selection-fit only; refit recomputes on full TRAIN; frozen Stage4 source already saw full TRAIN',
        'window':WINDOW,'budget':{'oracle':args.oracle_epochs,'student':args.student_epochs,'joint':args.joint_epochs},
        'seed':args.seed,'deterministic_upper':True,'naturalness_validated':False,'medtalk_parity_established':False}
    protocol.update(oracle_scope='target-informed upper decoder training/diagnostic; oracle and student share identical anchor/scales/window/compose target',
        mismatch_scope='only local precomputed acoustic features swapped within evaluation population; recipient global audio/reference held fixed',
        reverse_scope='reverse precomputed local acoustic features, not waveform; recipient global/reference held fixed',
        static_scope='two inference interventions, no independently trained static baseline',
        postprocessing='learned sigmoid output bound; no post-hoc clamp/smoothing on decoder outputs; separate smooth_prior uses window5',
        paper_success_established=False)
    save_json(args.output/'protocol.json',protocol)
    selection={}
    selection['oracle']=select_stage(decoder,student,cache,fit,cal,stats,'oracle',2 if args.smoke else args.oracle_epochs,args,args.output)
    oracle_report,_=evaluate_modes(decoder,student,cache,cal,stats,args,('oracle','oracle_static','oracle_reverse','oracle_zero'))
    save_json(args.output/'oracle_evaluation.json',oracle_report)
    selection['student']=select_stage(decoder,student,cache,fit,cal,stats,'student',2 if args.smoke else args.student_epochs,args,args.output)
    selection['joint']=select_stage(decoder,student,cache,fit,cal,stats,'joint',1 if args.smoke else args.joint_epochs,args,args.output)
    selection_report,_=evaluate_modes(decoder,student,cache,cal,stats,args,('full','static','static_intensity','reverse','shuffle','mismatch'))
    save_json(args.output/'selection_evaluation.json',selection_report);save_json(args.output/'selection.json',selection)
    save_checkpoint(args.output/'selection.pt',{'schema':SCHEMA,'decoder':decoder.state_dict(),'student':student.state_dict(),
        'stats':stats,'decoder_config':decoder.export_config(),'student_config':student.export_config(),
        'protocol':protocol,'protocol_sha256':canonical_hash(protocol),'selection':selection})
    if args.smoke or args.selection_only:
        save_json(args.output/'status.json',{'status':'complete','scope':'TRAIN selection only','development_loaded':False,
            'test_loaded':False,'default_replaced':False,'elapsed_seconds':time.monotonic()-started})
        return
    del cache
    all_ids=torch.arange(len(q['valid']));stats=fit_statistics(q,all_ids,args.device,args.batch_size)
    cache=prepare_cache(q,stats,args.batch_size);decoder,student=new_models(stats,args.device,args.seed+100000)
    history=[]
    for stage in ('oracle','student','joint'):
        optimizer=optimizer_for(decoder,student,stage)
        for number in range(1,selection[stage]['epochs']+1):
            loss=epoch(decoder,student,cache,all_ids,stats,optimizer,stage,args.device,args.batch_size,args.seed+number)
            record={'stage':stage,'epoch':number,'refit_loss':loss};history.append(record);print(json.dumps(record),flush=True)
            save_json(args.output/'refit_history.json',history)
    save_checkpoint(args.output/'final.pt',{'schema':SCHEMA,'decoder':decoder.state_dict(),'student':student.state_dict(),
        'stats':stats,'decoder_config':decoder.export_config(),'student_config':student.export_config(),
        'protocol':protocol,'protocol_sha256':canonical_hash(protocol),'selection':selection})
    del cache,q,data,system,audio,identities
    if torch.cuda.is_available():torch.cuda.empty_cache()
    save_json(args.output/'status.json',{'status':'evaluating_development','test_loaded':False,'development_loaded':True})
    data,system,audio,identities=load_reference_context(args.data,args.source,args.device,role='validation',seed=args.seed,
                                                      expected_source_sha256=SOURCE_SHA)
    q=data['splits']['validation'];cache=prepare_cache(q,stats,args.batch_size);ids=torch.arange(len(q['valid']))
    report,uppers=evaluate_modes(decoder,student,cache,ids,stats,args,('full','static','static_intensity','reverse','shuffle','mismatch'))
    bases=[];rng=torch.Generator().manual_seed(48)
    for ix in ids.split(args.batch_size):
        b=subset(q,ix,args.device);noise=torch.randn((*b['valid'].shape,52),generator=rng).to(args.device)
        bases.append(old_prediction(system,b,identities,noise).cpu())
    base=torch.cat(bases);smooth=smooth_motion_carrier(base,q['valid'],window=5)['smoothed']
    predictions={'original_prior':base,'smooth_prior':smooth}
    predictions.update({name:compose_upper_face(base,u,q['valid']) for name,u in uppers.items()})
    for name in ('original_prior','smooth_prior'):
        u=predictions[name][...,UPPER]
        intensity=neutral_relative_intensity(u,q['anchors'][:,UPPER],stats['scales9'],q['valid'],WINDOW)
        report['modes'][name]=scores(u,intensity,cache,ids,stats,light=False)
    report['paired_comparisons']=paired_comparisons(report['modes'],cache,ids)
    report['nonupper43_bit_exact']={name:torch.equal(v[...,NOT_UPPER].contiguous().view(torch.int32),base[...,NOT_UPPER].contiguous().view(torch.int32)) for name,v in predictions.items()}
    report['motion_diagnostics']={name:_upper_motion_diagnostics(v,q['valid']) for name,v in predictions.items() if name in ('original_prior','smooth_prior','full','static')}
    report['target_motion_diagnostics']=_upper_motion_diagnostics(q['motion'],q['valid'])
    report.update(test_loaded=False,default_replaced=False,medtalk_parity_established=False,
        generated_emotion_identity_naturalness_pending=True,prior_noise_seed=48,
        upper_generator_deterministic=True,scope='development once after TRAIN selection and full TRAIN refit',
        mismatch_scope='local acoustic features swapped; frozen recipient global/reference conditions retained',
        global_audio_accuracy=float((q['emotion_logits'].argmax(-1)==q['emotion_id']).float().mean()))
    save_json(args.output/'evaluation.json',report)
    save_checkpoint(args.output/'curves.pt',{'schema':SCHEMA+'_native_curves','clip_id':q['clip_id'],
        'speaker':q['speaker'],'sentence_id':q['sentence_id'],'times':q['times'],'valid':q['valid'],
        'channel_mask':q['channel_mask'],'target':q['motion'],'predictions':predictions,'seed':48,
        'native_rate':True,'postprocessed':False,'decoder_output_bound':'learned sigmoid, not post-hoc clamp',
        'condition_postprocessing':{name:{'smoothed':name=='smooth_prior','window':5 if name=='smooth_prior' else 1,
            'display_clamped':False} for name in predictions}, 'display_clamp_applied':False,
        'source_sha256':SOURCE_SHA,'selected_checkpoint_sha256':sha(args.output/'final.pt'),
        'protocol_sha256':canonical_hash(protocol)})
    previews=[]
    for label in sorted(set(q['emotion_id'].tolist())):
        options=[i for i in range(len(ids)) if int(q['emotion_id'][i])==label]
        i=min(options,key=lambda j:q['clip_id'][j]);clip=q['clip_id'][i]
        length=int(q['valid'][i].nonzero(as_tuple=True)[0][-1])+1
        names=['gt','original_prior','smooth_prior','full','static']
        motions=torch.stack([q['motion'][i,:length]]+[predictions[n][i,:length] for n in names[1:]])
        np.savez_compressed(args.output/(clip+'_comparison.npz'),channels=np.array(ARKIT_NAMES),
            mode_names=np.array(names),motions=motions.numpy(),times=q['times'][i,:length].numpy(),
            valid=q['valid'][i,:length].numpy(),channel_mask=q['channel_mask'][i].numpy(),clip_id=clip,noise_seed=48)
        np.savez_compressed(args.output/(clip+'_input.npz'),prior=base[i,:length].numpy(),
            audio_features=q['audio_features'][i,:length].numpy(),global_code=q['global_code'][i].numpy(),
            identity_code=q['identity_code'][i].numpy(),anchor=q['anchors'][i].numpy(),
            times=q['times'][i,:length].numpy(),valid=q['valid'][i,:length].numpy(),
            channel_mask=q['channel_mask'][i].numpy(),clip_id=clip,noise_seed=48)
        previews.append({'clip_id':clip,'emotion_id':label,'frames':length})
    save_json(args.output/'preview_manifest.json',previews)
    save_json(args.output/'status.json',{'status':'complete','test_loaded':False,'development_loaded':True,
        'default_replaced':False,'medtalk_parity_established':False,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'status':'complete','output':str(args.output)}),flush=True)


if __name__=='__main__':main()
