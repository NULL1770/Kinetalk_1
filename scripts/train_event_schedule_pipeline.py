"""Fixed-split event timing predictor and separately trained conditional prior.

Runs a control-path experiment and audio-prediction experiment automatically.
Oracle schedules are labeled explicitly and never passed as audio inference.
This driver does not promote weights or access historical outer target tensors.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from kinetalk_b0.models.audio_event_schedule import AudioEventSchedulePredictor, DURATIONS
from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE, ContinuousLatentFlow
from kinetalk_b0.models.event_conditioned_flow import EventConditionedFlow, shift_schedule
from scripts.event_schedule_teacher import fit_teacher, extract_schedule, valid_runs
from scripts.probe_motion_condition_predictability import split_train_pool, sha
from scripts import train_continuous_motion_latent as common
from scripts.evaluate_continuous_motion_latent import (
    compose_full, _summary_scores, _numerics, _seed, _metadata, _write_artifacts, UPPER,
)
from scripts.joint_motion_metrics import summarize

SCHEMA = 'event_schedule_pipeline_v1'
SEEDS = (42, 123, 2026)
SOURCES = ('scripts/train_event_schedule_pipeline.py', 'scripts/event_schedule_teacher.py',
           'kinetalk_b0/models/audio_event_schedule.py', 'kinetalk_b0/models/event_conditioned_flow.py',
           'kinetalk_b0/models/continuous_upper_motion.py', 'kinetalk_b0/models/motion_process_prior.py',
           'scripts/evaluate_continuous_motion_latent.py', 'scripts/arkit_benchmark_report.py',
           'scripts/evaluate_arkit_literature_metrics.py', 'scripts/train_continuous_motion_latent.py',
           'scripts/probe_motion_condition_predictability.py', 'scripts/joint_motion_metrics.py',
           'scripts/prepare_continuous_motion_dataset.py', 'scripts/train_prior_audio_adapter.py')


def arr(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def schedule_for(clip, teacher):
    return extract_schedule(clip['motion9'], clip['valid'], teacher, motion_mask=clip['motion_mask'])


def target_arrays(label):
    active = label['schedule'][:, :4] > .5
    onset = label['onset']
    risk = label['known'] & (~active | onset)
    duration = np.zeros_like(onset, dtype=np.int64)
    for event in label['events']:
        duration[event['start'], event['group_index']] = int(np.argmin(
            np.abs(np.array(DURATIONS)-(event['end']-event['start']))))
    return onset.astype(np.float32), risk, duration


def acoustic_input(clip, stats, mode='real'):
    """Only deployable inputs; static/reverse operate on complete audio runs."""
    native = arr(clip['valid']); raw = arr(clip['features']).copy()
    for a, b in valid_runs(native):
        if mode == 'static': raw[a:b] = raw[a:b].mean(0, keepdims=True)
        elif mode == 'reverse': raw[a:b] = raw[a:b][::-1]
        elif mode != 'real': raise ValueError('Unknown acoustic intervention')
    value = np.zeros_like(raw, dtype=np.float32)
    value[native] = (raw[native]-arr(stats['audio_mean']))/arr(stats['audio_scale'])
    context = (arr(clip['context'])-arr(stats['context_mean']))/arr(stats['context_scale'])
    return torch.from_numpy(value), torch.as_tensor(context, dtype=torch.float32), torch.as_tensor(native)


def prepare_audio(clips, labels, stats, mode):
    rows = []
    for clip in clips:
        feature, context, native = acoustic_input(clip, stats, mode)
        onset, risk, duration = target_arrays(labels[clip['clip_id']])
        if risk.any():
            rows.append({'clip_id': clip['clip_id'], 'feature': feature, 'context': context, 'valid': native,
                         'onset': torch.from_numpy(onset), 'risk': torch.from_numpy(risk),
                         'duration': torch.from_numpy(duration)})
    return rows


def predictor_batch(rows, rng, device, batch_size=12, frames=180):
    """Crop on audio support; discard truncated receptive-field likelihood.

    Default TCN dilations (1,2,4,8) give a 15-frame radius. Original native-run
    edges keep the same replicated boundary as deployment; artificial crop
    edges exclude the first/last15 targets while retaining them as audio halo.
    """
    picks = rng.integers(len(rows), size=batch_size)
    x = torch.zeros(batch_size, frames, rows[0]['feature'].shape[-1], device=device)
    v = torch.zeros(batch_size, frames, dtype=torch.bool, device=device)
    y = torch.zeros(batch_size, frames, 4, device=device)
    risk = torch.zeros_like(y, dtype=torch.bool); duration = torch.zeros_like(y, dtype=torch.long)
    contexts = []
    for b, index in enumerate(picks):
        row = rows[index]
        runs = valid_runs(arr(row['valid']))
        run_start, end = runs[int(rng.integers(len(runs)))]; a = run_start
        if end-a > frames: a = int(rng.integers(a, end-frames+1))
        stop = min(a+frames, end); n = stop-a
        for dest, key in ((x, 'feature'), (y, 'onset'), (risk, 'risk'), (duration, 'duration')):
            dest[b, :n] = row[key][a:stop].to(device)
        if a > run_start: risk[b, :min(15, n)] = False
        if stop < end: risk[b, max(0, n-15):n] = False
        v[b, :n] = True; contexts.append(row['context'])
    return x, torch.stack(contexts).to(device), v, y, risk, duration


def process_nll(out, onset, risk, duration):
    onset_loss = F.binary_cross_entropy_with_logits(out['onset_logits'], onset, reduction='none')
    duration_loss = F.cross_entropy(out['duration_logits'].flatten(0, 2), duration.flatten(),
                                    reduction='none').reshape_as(onset)
    # Joint event-process likelihood: duration term exists only when an onset
    # occurs, with the SAME risk-time denominator, not an arbitrary large weight.
    error = onset_loss*risk + duration_loss*(onset.bool() & risk)
    count = risk.sum((1, 2)); good = count > 0
    return (error.sum((1, 2))[good]/count[good]).mean() if good.any() else error.sum()*0


def initialize_rates(model, rows):
    # Equal clips in initialization; training risk remains clip/crop balanced.
    prevalence = np.mean([arr(r['onset']).sum(0)/np.maximum(arr(r['risk']).sum(0), 1) for r in rows], 0)
    prevalence = np.clip(prevalence, 1e-4, .25)
    counts = np.ones((4, len(DURATIONS)))
    for row in rows:
        for g in range(4):
            selected = arr(row['duration'])[:, g][arr(row['onset'])[:, g] > 0]
            counts[g] += np.bincount(selected, minlength=len(DURATIONS))/max(len(rows), 1)
    with torch.no_grad():
        model.onset_head.weight.zero_()
        model.onset_head.bias.copy_(torch.from_numpy(np.log(prevalence/(1-prevalence))).to(model.onset_head.bias))
        model.duration_head.weight.zero_()
        model.duration_head.bias.copy_(torch.from_numpy(np.log(counts/counts.sum(1, keepdims=True))).flatten().to(model.duration_head.bias))


def train_predictor(rows, seed, steps, device, output, status):
    torch.manual_seed(seed)
    config = dict(feature_dim=1540, context_dim=len(rows[0]['context']), hidden=64)
    model = AudioEventSchedulePredictor(**config).to(device)
    initialize_rates(model, rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(seed); losses = []
    for step in range(1, steps+1):
        x, c, valid, y, risk, duration = predictor_batch(rows, rng, device)
        out = model(x, c, valid); loss = process_nll(out, y, risk, duration)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite event prediction loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); losses.append(float(loss.detach()))
        if step == 1 or step % 250 == 0 or step == steps:
            status('training_predictor', seed=seed, arm=output.stem, step=step, steps=steps, loss=losses[-1])
            common._save(output.with_name(output.stem+'_last.pt'), {
                'state': model.state_dict(), 'config': config, 'step': step, 'seed': seed,
                'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state, 'losses': losses})
    common._save(output, {'state': model.state_dict(), 'config': config, 'seed': seed, 'steps': steps, 'losses': losses})
    return model.eval()


@torch.no_grad()
def evaluate_predictor(model, clips, labels, stats, mode, device):
    rows = []
    for clip in clips:
        x, c, v = acoustic_input(clip, stats, mode)
        out = model(x[None].to(device), c[None].to(device), v[None].to(device))
        probability = out['onset_logits'][0].sigmoid().cpu().numpy()
        log_duration = out['duration_logits'][0].log_softmax(-1).cpu().numpy()
        onset, risk, duration = target_arrays(labels[clip['clip_id']])
        if not risk.any(): continue
        p = np.clip(probability[risk], 1e-7, 1-1e-7); y = onset[risk]
        onset_nll = float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
        event_frame, event_group = np.where((onset > 0) & risk)
        event_duration_nll = -log_duration[event_frame, event_group, duration[event_frame, event_group]]
        duration_sum = float(event_duration_nll.sum())
        rows.append({'clip_id': clip['clip_id'], 'sentence': clip['sentence'],
                     'brier': float(np.square(p-y).mean()),
                     'onset_nll': onset_nll,
                     'joint_nll': onset_nll+duration_sum/int(risk.sum()),
                     'duration_nll': float(event_duration_nll.mean()) if len(event_frame) else None,
                     'duration_nll_sum': duration_sum, 'scored_onsets': int(len(event_frame)),
                     'onsets': int(onset.sum()), 'risk_positions': int(risk.sum()),
                     'probability': probability, 'duration_log_probability': log_duration,
                     'truth_onset': onset, 'risk': risk, 'truth_duration_bin': duration})
    return rows


def summarize_predictor(rows):
    if not rows: raise ValueError('No supported predictor validation clips')
    result = {k: float(np.mean([r[k] for r in rows])) for k in ('brier', 'onset_nll', 'joint_nll')}
    durations = [r['duration_nll'] for r in rows if r['duration_nll'] is not None]
    count = sum(r['scored_onsets'] for r in rows)
    result.update(clips=len(rows), duration_scored_clips=len(durations), scored_onsets=count,
                  duration_nll=float(np.mean(durations)) if durations else None,
                  duration_nll_per_event=sum(r['duration_nll_sum'] for r in rows)/count if count else None)
    return result


def paired_delta(first, second, metric='brier'):
    lookup = {r['clip_id']: r for r in second}; grouped = {}; omitted = []
    if set(lookup) != {r['clip_id'] for r in first}: raise ValueError('Predictor arm membership differs')
    for row in first:
        reference = lookup[row['clip_id']]
        if (row[metric] is None) != (reference[metric] is None):
            raise ValueError('Paired metric support differs')
        if row[metric] is None:
            omitted.append(row['clip_id']); continue
        grouped.setdefault(row['sentence'], []).append(row[metric]-reference[metric])
    if not grouped:
        return {'audio_minus_control': None, 'ci95': None, 'sentences': 0,
                'sentences_audio_better': 0, 'metric': metric, 'omitted_clips': omitted}
    values = np.array([np.mean(grouped[s]) for s in sorted(grouped)])
    samples = np.random.default_rng(20260919).choice(values, (4096, len(values))).mean(1)
    return {'audio_minus_control': float(values.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'sentences': len(values), 'sentences_audio_better': int((values < 0).sum()), 'metric': metric,
            'omitted_clips': omitted}


def receiver_segments(clips, labels, stats, ae, device):
    result = []
    for clip in clips:
        label = labels[clip['clip_id']]
        support = label['known'].all(1) & arr(clip['motion_mask']).all(1) & arr(clip['valid'])
        context = (clip['context']-stats['context_mean'])/stats['context_scale']
        for a, end in valid_runs(support):
            for start in range(a, end, 200):
                stop = min(start+200, end)
                if stop-start < 10: continue
                residual = (clip['motion9'][start:stop]-clip['b9'])/stats['residual_scale']
                with torch.no_grad():
                    z, _ = ae.encode(residual[None].to(device), torch.ones(1, stop-start, dtype=torch.bool, device=device))
                result.append({'z': z[0].cpu(), 'audio': torch.from_numpy(label['schedule'][start:stop]),
                               'context': context, 'metadata': {'clip_id': clip['clip_id'], 'start': start, 'end': stop}})
    return result


@torch.no_grad()
def generate_with_schedule(flow, ae, clip, stats, schedule, device, seed):
    valid = arr(clip['valid']); baseline = arr(clip['baseline52'])
    values = baseline[:, UPPER].copy()
    context = ((clip['context']-stats['context_mean'])/stats['context_scale'])[None].to(device)
    for run, (a, b) in enumerate(valid_runs(valid)):
        n = b-a; length = (n+4)//5*5
        condition = torch.zeros(1, length, 12, device=device)
        condition[0, :n] = torch.as_tensor(schedule[a:b], device=device)
        native = torch.ones(1, n, dtype=torch.bool, device=device)
        frame_mask = torch.arange(length, device=device)[None] < n
        blocks = condition.reshape(1, -1, 5, 12); mask = frame_mask.reshape(1, -1, 5)
        noise = torch.randn(1, blocks.shape[1], ae.latent_dim, device=device,
                            generator=torch.Generator(device=device).manual_seed(_seed(clip['clip_id'], run, seed)))
        z = flow.sample(mask.any(-1), context, blocks, noise, steps=24, audio_frame_valid=mask)
        decoded = ae.decode(z*stats['latent_scale'].to(device)+stats['latent_mean'].to(device), native)
        values[a:b] = (decoded[0]*stats['residual_scale'].to(device)+clip['b9'].to(device)).cpu().numpy()
    return values


def evaluate_receiver(flow, ae, clips, stats, schedules, output, device, mode, teacher):
    curves, scores, numerics, event_rows, truth_rows = {}, [], [], [], []
    flow.eval(); ae.eval()
    for clip in clips:
        cid = clip['clip_id']; schedule_draws = schedules[cid]
        seeds = (42, 123, 2026, 77)
        samples = np.stack([generate_with_schedule(flow, ae, clip, stats,
                            schedule_draws[min(i, len(schedule_draws)-1)], device, seed) for i, seed in enumerate(seeds)])
        metric, score_mask = _summary_scores(samples, clip, stats)
        if metric is not None: scores.append(metric)
        full = compose_full(samples, arr(clip['baseline52']))
        numerics.append({'clip_id': cid, **_numerics(samples, full, arr(clip['baseline52']), arr(clip['valid']))})
        curves[cid] = {'metadata': _metadata(clip), 'samples': samples, 'target': arr(clip['motion9']),
                       'native_valid': arr(clip['valid']), 'score_mask': score_mask, 'generated_mask': arr(clip['valid']),
                       'seeds': list(seeds), 'run_records': [], 'raw_coefficient_prediction': True}
        # Same-label event agreement is diagnostic, not a naturalness certificate.
        activities = []
        for i, sample in enumerate(samples):
            predicted = extract_schedule(sample, arr(clip['valid']), teacher)
            activities.append(predicted['schedule'][:, :4])
            wanted = schedule_draws[min(i, len(schedule_draws)-1)][:, :4]
            support = np.broadcast_to(arr(clip['valid'])[:, None], wanted.shape)
            event_rows.append({'clip_id': cid, 'seed': seeds[i], 'scored_positions': int(support.sum()),
                               'prediction_known_fraction': float(predicted['known'][support].mean()),
                               'agreement_mse': float(np.square(predicted['schedule'][:, :4]-wanted)[support].mean()),
                               'generated_events': len(predicted['events']),
                               'requested_active_fraction': float(wanted[arr(clip['valid'])].mean())})
        # Shared target support, fixed across oracle, empty, shifted and null.
        # Unsupported predicted events count as inactive, never erase errors.
        truth_rows.append(shared_event_score(clip, teacher, np.stack(activities)))
    result = {'schema': SCHEMA, 'mode': mode, 'scope': 'inner development; oracle arms are not audio inference',
              'summary': summarize(scores) if scores else None, 'per_clip_scores': scores, 'numerics': numerics,
              'numerical_gate': {'passed': all(x['nonfinite_values'] == 0 and x['other43_exact'] and x['native_invalid_exact_baseline'] for x in numerics)},
              'selected_clip_ids': [c['clip_id'] for c in clips], 'event_condition_diagnostics': event_rows,
              'shared_truth_event_scores': truth_rows,
              'naturalness_certified': False, 'default_replaced': False}
    _write_artifacts(clips, curves, output, result, mode)
    common._save(output/'conditions.pt', schedules)
    return result


def shared_event_score(clip, teacher, activities):
    label = schedule_for(clip, teacher)
    known = label['known'] & arr(clip['valid'])[:, None]
    target = label['schedule'][:, :4]
    probability = np.asarray(activities).mean(0)
    active = known & (target > .5)
    return {'clip_id': clip['clip_id'], 'sentence': clip['sentence'],
            'brier': float(np.square(probability-target)[known].mean()) if known.any() else None,
            'target_known_positions': int(known.sum()), 'target_active_positions': int(active.sum()),
            'mean_probability_on_true_events': float(probability[active].mean()) if active.any() else None}


def receiver_control_gate(results, directory):
    rows = {name: [r for r in result['shared_truth_event_scores'] if r['brier'] is not None]
            for name, result in results.items()}
    if not rows['oracle'] or not any(r['target_active_positions'] for r in rows['oracle']):
        return {'passed': False, 'reason': 'No shared supported validation events'}
    comparisons = {name: paired_delta(rows['oracle'], rows[name]) for name in ('null', 'empty')}
    oracle = common._load(directory/'oracle'/'curves.pt')['clips']
    response = {}
    for name in ('empty', 'shifted'):
        other = common._load(directory/name/'curves.pt')['clips']
        response[name] = float(np.mean([np.abs(oracle[cid]['samples'][:, oracle[cid]['native_valid']]-
                   other[cid]['samples'][:, oracle[cid]['native_valid']]).mean() for cid in oracle]))
    numerically_ok = all(r['numerical_gate']['passed'] for r in results.values())
    passed = numerically_ok and all(c['audio_minus_control'] < 0 for c in comparisons.values()) and all(v > 1e-5 for v in response.values())
    return {'passed': passed, 'oracle_vs_controls': comparisons, 'paired_noise_motion_mae_response': response,
            'numerical_protection_passed': numerically_ok,
            'scope': 'Engineering control-path gate only; small subset, no naturalness or audio timing certification',
            'unknown_prediction_policy': 'Unsupported predicted events score as inactive on fixed GT-known support'}


def run(args):
    started = time.monotonic(); torch.set_num_threads(4)
    if args.output.exists(): raise FileExistsError('Fresh experiment directory required')
    if min(args.receiver_steps, args.predictor_steps) < 1: raise ValueError('Positive training budgets required')
    if args.smoke and max(args.receiver_steps, args.predictor_steps) > 100: raise ValueError('Smoke budget must not exceed 100')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True)
    def status(state, **data):
        common._write(args.output/'status.json', {'schema': SCHEMA, 'state': state, 'updated': time.time(), **data})
    status('loading')
    reference = json.loads((args.source_run/'protocol.json').read_text(encoding='utf8'))
    if sha(args.dataset) != reference['dataset_sha256']: raise ValueError('Source dataset mismatch')
    payload = torch.load(args.dataset, weights_only=False, mmap=True, map_location='cpu')
    if payload.get('schema') != 'continuous_motion_dataset_v1': raise ValueError('Unexpected dataset schema')
    train, valid = split_train_pool(payload['clips'], reference); del payload
    # Reject discontinuous native runs rather than silently joining motion clocks.
    for clip in train+valid:
        times = arr(clip['times'])
        for a, b in valid_runs(arr(clip['valid'])):
            if not np.allclose(np.diff(times[a:b]), 1/25, rtol=1e-4, atol=1e-5):
                raise ValueError('Discontinuous native run: '+clip['clip_id'])
    if args.smoke: train, valid = train[:12], valid[:4]
    stats = common._load(args.source_run/'fit_stats.pt')
    if not args.smoke and set(stats['train_clip_ids']) != {c['clip_id'] for c in train}:
        raise ValueError('Prior statistics were not fitted to exact training members')
    device = torch.device(args.device)
    ae_ck = common._load(args.source_run/'ae_final.pt'); prior_ck = common._load(args.source_run/'prior_final.pt')
    expected_protocol = common._value_sha(reference)
    if any(ck.get('binding', {}).get('protocol_sha256') != expected_protocol for ck in (ae_ck, prior_ck)):
        raise ValueError('Source checkpoint protocol binding mismatch')
    if prior_ck['binding'].get('ae_sha256') != common._value_sha(ae_ck['state']) or prior_ck['binding'].get('stats_sha256') != common._value_sha(stats):
        raise ValueError('Source prior AE/statistics binding mismatch')
    ae = ContinuousUpperAE(**ae_ck['config']).to(device); ae.load_state_dict(ae_ck['state']); ae.eval().requires_grad_(False)
    prior = ContinuousLatentFlow(**prior_ck['config']).to(device); prior.load_state_dict(prior_ck['state']); prior.eval()
    code = Path(__file__).resolve().parents[1]
    protocol = {'schema': SCHEMA, 'dataset_sha256': reference['dataset_sha256'],
                'source_protocol_sha256': sha(args.source_run/'protocol.json'),
                'prior_sha256': sha(args.source_run/'prior_final.pt'), 'ae_sha256': sha(args.source_run/'ae_final.pt'),
                'fit_stats_sha256': sha(args.source_run/'fit_stats.pt'),
                'source_sha256': {p: sha(code/p) for p in SOURCES}, 'smoke': args.smoke,
                'train_ids': [c['clip_id'] for c in train], 'validation_ids': [c['clip_id'] for c in valid],
                'receiver_steps_per_arm': args.receiver_steps, 'predictor_steps_per_arm': args.predictor_steps,
                'predictor_seeds': SEEDS[:1] if args.smoke else SEEDS, 'receiver_seed': 2026091909,
                'selection': 'fixed final checkpoints; never choose a best seed',
                'condition': '4 active / 4 phase / 4 duration seconds; no motion amplitude',
                'outer_tensor_indexed': False, 'outer_archive_mapped': True,
                'candidate_policy': 'always export fixed audio candidate diagnostic; no default promotion',
                'predictor_gate': 'all seeds onset Brier and joint-process NLL audio<matched static with sentence-bootstrap CI upper<0; joint NLL audio<reverse; separate receiver control remains required',
                'receiver_gate': 'oracle event Brier below null and empty on shared GT-known support; paired-noise motion MAE response to empty and shifted >1e-5; all numerical protection checks pass; engineering gate only',
                'scope': 'development with inherited upstream exposure, not sealed test'}
    common._write(args.output/'protocol.json', protocol)
    for path in SOURCES:
        dest = args.output/'source'/path; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((code/path).read_bytes())
    teacher = fit_teacher(train)
    common._write(args.output/'teacher.json', teacher)
    labels = {c['clip_id']: schedule_for(c, teacher) for c in train+valid}
    coverage = {c['clip_id']: {'split': 'fit' if c['clip_id'] in set(protocol['train_ids']) else 'inner_validation',
                 'events': len(labels[c['clip_id']]['events']),
                 'known_fraction': float(labels[c['clip_id']]['known'][arr(c['valid'])].mean()),
                 'events_per_group': np.bincount([e['group_index'] for e in labels[c['clip_id']]['events']], minlength=4).tolist()}
                for c in train+valid}
    common._write(args.output/'teacher_coverage.json', coverage)
    quantization = {}
    for split, clips in [('fit', train), ('inner_validation', valid)]:
        durations = np.array([e['end']-e['start'] for c in clips for e in labels[c['clip_id']]['events']])
        quantization[split] = {'events': len(durations), 'longer_than_max_bin': int((durations > max(DURATIONS)).sum()),
            'nearest_bin_mae_frames': float(np.abs(durations[:, None]-np.array(DURATIONS)).min(1).mean()) if len(durations) else None,
            'max_event_frames': int(durations.max()) if len(durations) else None, 'bins_frames': list(DURATIONS)}
    common._write(args.output/'duration_quantization.json', quantization)
    fit_events = sum(len(labels[c['clip_id']]['events']) for c in train)
    if fit_events < (1 if args.smoke else 40):
        status('stopped', reason='Too few supported training events', fit_events=fit_events); return
    common._save(args.output/'event_labels.pt', labels)
    # Evaluate a fixed metadata-only subset for costly flow control diagnostics.
    from scripts.train_prior_audio_adapter import diverse_diagnostics
    selected = diverse_diagnostics(valid, 4 if args.smoke else 24)
    segments = receiver_segments(train, labels, stats, ae, device)
    if not segments: status('stopped', reason='No jointly known receiver segments'); return
    common._write(args.output/'receiver_coverage.json', {'segments': len(segments),
                     'frames': sum(len(s['audio']) for s in segments), 'clips': len({s['metadata']['clip_id'] for s in segments}),
                     'active_frames_per_group': torch.cat([s['audio'][:, :4] for s in segments]).sum(0).tolist()})
    null_segments = [{**s, 'audio': torch.zeros_like(s['audio'])} for s in segments]
    torch.manual_seed(2026091909); initialized = EventConditionedFlow.from_prior(prior)
    binding = {'protocol_sha256': sha(args.output/'protocol.json'), 'teacher_sha256': sha(args.output/'teacher.json')}
    receivers = {}
    for name, items in [('event', segments), ('null', null_segments)]:
        status('training_receiver', arm=name)
        model = copy.deepcopy(initialized)
        common._train_stage(model, items, stats, stage='receiver_'+name, budget=args.receiver_steps,
                            batch_size=24, device=device, output=args.output, binding=binding,
                            seed=2026091909, use_audio=True)
        receivers[name] = model.eval()
    oracle = {c['clip_id']: [labels[c['clip_id']]['schedule']] for c in selected}
    empty = {c['clip_id']: [np.zeros((len(c['valid']), 12), np.float32)] for c in selected}
    shifted = {c['clip_id']: [shift_schedule(oracle[c['clip_id']][0], arr(c['valid']), 12)] for c in selected}
    control_results = {}
    for name, model, schedules in [('oracle', receivers['event'], oracle), ('empty', receivers['event'], empty),
                                   ('shifted', receivers['event'], shifted), ('null', receivers['null'], empty)]:
        status('evaluating_receiver', arm=name)
        control_results[name] = evaluate_receiver(model, ae, selected, stats, schedules,
                                                   args.output/'control'/name, device, name, teacher)
    control_gate = receiver_control_gate(control_results, args.output/'control')
    common._write(args.output/'receiver_gate.json', control_gate)
    predictor_rows = {mode: prepare_audio(train, labels, stats, mode) for mode in ('real', 'static')}
    common._write(args.output/'predictor_coverage.json', {
        'included_fit_ids': [r['clip_id'] for r in predictor_rows['real']],
        'excluded_no_risk_fit_ids': [c['clip_id'] for c in train if c['clip_id'] not in {r['clip_id'] for r in predictor_rows['real']}],
        'validation_no_risk_ids': [c['clip_id'] for c in valid if not target_arrays(labels[c['clip_id']])[1].any()]})
    if not predictor_rows['real']: raise ValueError('No supported predictor training clips')
    predictor_reports = {}; primary_audio = None; primary_static = None
    for seed in (SEEDS[:1] if args.smoke else SEEDS):
        models = {}
        for name, mode in [('audio', 'real'), ('matched_static', 'static')]:
            models[name] = train_predictor(predictor_rows[mode], seed, args.predictor_steps, device,
                                          args.output/f'predictor_{seed}_{name}.pt', status)
        reports = {}
        for name, model, mode in [('audio', models['audio'], 'real'), ('matched_static', models['matched_static'], 'static'),
                                  ('reverse', models['audio'], 'reverse'), ('own_static', models['audio'], 'static')]:
            reports[name] = evaluate_predictor(model, valid, labels, stats, mode, device)
        common._save(args.output/f'predictor_{seed}_predictions.pt', reports)
        predictor_reports[str(seed)] = {'summary': {k: summarize_predictor(v) for k, v in reports.items()},
                                       'audio_vs_static': paired_delta(reports['audio'], reports['matched_static']),
                                       'audio_vs_static_joint_nll': paired_delta(reports['audio'], reports['matched_static'], 'joint_nll'),
                                       'audio_vs_static_duration_nll': paired_delta(reports['audio'], reports['matched_static'], 'duration_nll'),
                                       'audio_vs_reverse': paired_delta(reports['audio'], reports['reverse'], 'joint_nll')}
        common._write(args.output/'predictor_results.json', predictor_reports)
        if primary_audio is None: primary_audio, primary_static = models['audio'], models['matched_static']
    gate = all(r['audio_vs_static']['ci95'][1] < 0 and r['audio_vs_static_joint_nll']['ci95'][1] < 0 and r['audio_vs_reverse']['audio_minus_control'] < 0
               for r in predictor_reports.values())
    # Audio inference below has no access to label schedules or motion masks.
    candidates = {}
    for name, model, intervention in [('audio', primary_audio, 'real'), ('matched_static', primary_static, 'static'),
                                     ('reverse', primary_audio, 'reverse')]:
        schedules = {}
        for clip in selected:
            x, c, v = acoustic_input(clip, stats, intervention)
            schedules[clip['clip_id']] = []
            for seed in (42, 123, 2026, 77):
                result = model.sample_schedule(x[None].to(device), c[None].to(device), v[None].to(device),
                         generator=torch.Generator(device=device).manual_seed(_seed(clip['clip_id'], 0, seed)+17))
                schedules[clip['clip_id']].append(result['condition'][0].cpu().numpy())
        status('evaluating_audio_generation', arm=name)
        candidates[name] = evaluate_receiver(receivers['event'], ae, selected, stats, schedules,
                                               args.output/'generation'/name, device, name, teacher)
    common._write(args.output/'decision.json', {'audio_predictability_passed': gate, 'default_replaced': False,
                     'receiver_control_path_passed': control_gate['passed'],
                     'control_path_requires_review': True, 'audio_generation_is_fixed_seed_diagnostic': True,
                     'outer_evaluated': False, 'smoke': args.smoke})
    status('complete', seconds=time.monotonic()-started, audio_predictability_passed=gate,
           default_replaced=False, needs_visual_review=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source-run', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--receiver-steps', type=int, default=2000); p.add_argument('--predictor-steps', type=int, default=1500)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try: run(args)
    except Exception as exc:
        if not isinstance(exc, FileExistsError) and args.output.exists():
            common._write(args.output/'status.json', {'schema': SCHEMA, 'state': 'failed', 'error': repr(exc), 'updated': time.time()})
        raise
