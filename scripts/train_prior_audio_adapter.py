"""Sentence-isolated, fixed-budget test of audio corrections to a frozen prior.

Only a new inner split inside the historical 819 train pool selects a candidate.
The old 64 diagnostic clips are evaluated once, after the candidate is locked.
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

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from kinetalk_b0.models.prior_audio_adapter import BoundedPriorAudioAdapter
from scripts import train_continuous_motion_latent as common
from scripts.evaluate_continuous_motion_latent import evaluate_ae, evaluate_generation
from scripts.prepare_continuous_motion_dataset import SCHEMA, contiguous_runs, fit_statistics, prepare_segments
from scripts.audit_prior_audio_adapter import audit

SEED = 2026091902
SOURCES = tuple(dict.fromkeys((*common.SOURCE_FILES, 'scripts/train_prior_audio_adapter.py',
    'kinetalk_b0/models/prior_audio_adapter.py', 'scripts/audit_prior_audio_adapter.py')))


def inner_split(clips, valid_sentences=5):
    """Hash only sentence metadata; no targets, predictions or audio statistics."""
    fit = [c for c in clips if c['split'] == 'train']
    sentences = sorted({c['sentence'] for c in fit}, key=lambda s: hashlib.sha256(
        ('bounded_audio_inner_v1:'+s).encode()).hexdigest())
    if not 2 <= valid_sentences < len(sentences):
        raise ValueError('Need disjoint nonempty inner training/validation sentences')
    held = set(sentences[:valid_sentences])
    train = [{**c, 'split': 'train'} for c in fit if c['sentence'] not in held]
    valid = [{**c, 'split': 'inner_validation'} for c in fit if c['sentence'] in held]
    outer = [c for c in clips if c['split'] == 'holdout']
    if {c['sentence'] for c in outer} & set(sentences):
        raise ValueError('Original sentence isolation violated')
    return train, valid, outer, sorted(held)


def diverse_diagnostics(clips, count=24):
    """Round robin sentences; within each sentence interleave speaker/emotion."""
    cells = {}
    for clip in sorted(clips, key=lambda c: (str(c['speaker']), str(c['emotion']), c['clip_id'])):
        cells.setdefault(clip['sentence'], []).append(clip)
    chosen = []; depth = 0
    while len(chosen) < min(count, len(clips)):
        for sentence in sorted(cells):
            if depth < len(cells[sentence]): chosen.append(cells[sentence][depth])
            if len(chosen) >= count: break
        depth += 1
    return chosen


def static_features(clips):
    """Static full native audio runs, before supervised chunking or target mask."""
    result = []
    for c in clips:
        features = c['features'].clone()
        for start, stop in contiguous_runs(c['valid']):
            features[start:stop] = features[start:stop].mean(0, keepdim=True)
        result.append({**c, 'features': features})
    return result


def evaluate_cached(ae, model, clips, stats, path, device, *, use_audio=True,
                    intervention='real', resume=False):
    # Common evaluator preserves complete native audio runs and stage artifacts.
    result = common._evaluate(ae, model, clips, stats, path, device,
        use_audio=use_audio, intervention=intervention, resume=resume)
    pointer = Path(path).with_name(Path(path).name+'_artifact.json')
    actual = Path(json.loads(pointer.read_text(encoding='utf8'))['directory']) if pointer.exists() else Path(path)
    result.setdefault('artifact_directory', str(actual))
    if not (Path(result['artifact_directory'])/'curves.pt').is_file():
        raise RuntimeError('Evaluation artifact pointer missing curves')
    return result


def select_candidate(records):
    passing = [r for r in records if r['accepted']]
    # Fixed earliest passing point, no best-score scan.
    return min(passing, key=lambda r: r['step']) if passing else None


def run(args):
    if min(args.ae_steps, args.prior_steps, args.adapter_steps, args.batch_size) < 1:
        raise ValueError('Positive budgets required')
    milestones = sorted(set(args.milestones or [1000, args.adapter_steps]))
    if not milestones or milestones[0] < 1 or milestones[-1] != args.adapter_steps:
        raise ValueError('Milestones must end at the fixed adapter budget')
    if args.skip_quality_gates and (not args.smoke or max(args.ae_steps, args.prior_steps, args.adapter_steps) > 100):
        raise ValueError('Quality gate override allowed only for <=100-step smoke')
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(args.device)
    if device.type == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists() and not args.resume:
        raise FileExistsError('Existing protocol; explicit resume required')
    if args.resume and not (output/'protocol.json').exists():
        raise FileNotFoundError('Resume requires protocol')
    payload = common._load(args.dataset)
    if payload.get('schema') != SCHEMA: raise ValueError('Dataset schema differs')
    train, valid, outer, held = inner_split(payload['clips'], args.valid_sentences)
    # Discard inherited 819-clip statistics. AE/prior/adapter all fit inner train.
    stats = fit_statistics(train)
    diagnostic = diverse_diagnostics(train)
    if args.smoke:
        valid = diverse_diagnostics(valid, 4); diagnostic = diagnostic[:4]
        outer = diverse_diagnostics(outer, 4)
    root = Path(__file__).resolve().parents[1]
    protocol = {'schema': 'bounded_audio_experiment_v1', 'dataset_sha256': common._file_sha(args.dataset),
        'source_sha256': {p: common._file_sha(root/p) for p in SOURCES},
        'seed': SEED, 'inner_train_ids': [c['clip_id'] for c in train],
        'inner_validation_ids': [c['clip_id'] for c in valid], 'validation_sentences': held,
        'outer_ids': [c['clip_id'] for c in outer], 'train_diagnostic_ids': [c['clip_id'] for c in diagnostic],
        'fit_stats_sha256': common._value_sha(stats), 'budgets': [args.ae_steps, args.prior_steps, args.adapter_steps],
        'milestones': milestones, 'max_delta': args.max_delta, 'batch_size': args.batch_size,
        'device_type': device.type, 'smoke': args.smoke, 'skip_quality_gates': args.skip_quality_gates,
        'selection': 'earliest fixed milestone passing inner validation; no outer selection',
        'loss': 'one flow matching objective; structural frozen prior plus bounded velocity correction',
        'matched_static': 'same adapter init, batches/noise/time/budget; full native-run static features before chunking',
        'outer_scope': 'historically exposed internal diagnostic only; not sealed or independent test',
        'upstream_exposure': 'frozen base/feature extractors historically exposed; new stats/AE/prior exclude inner validation',
        'default_replaced': False}
    path = output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != protocol:
            raise ValueError('Resume protocol/code/data/budget mismatch')
    else: common._write(path, protocol)
    binding = {'protocol_sha256': common._value_sha(protocol)}
    segments, report = prepare_segments(train, stats, 'train', 200)
    common._write(output/'segment_report.json', report)
    total = args.ae_steps+args.prior_steps+2*args.adapter_steps
    def status(state, **more):
        common._write(output/'status.json', {'state': state, 'updated': time.time(),
            'naturalness_certified': False, 'needs_visual_review': True, **more})
    ae = ContinuousUpperAE().to(device)
    common._train_stage(ae, segments, stats, stage='ae', budget=args.ae_steps,
        batch_size=args.batch_size, device=device, output=output, binding=binding,
        seed=SEED+1, resume=args.resume, total_updates=total)
    ae_train = evaluate_cached(ae, None, diagnostic, stats, output/'ae_reconstruction/train_diagnostic', device, resume=args.resume)
    ae_valid = evaluate_cached(ae, None, valid, stats, output/'ae_reconstruction/inner_validation', device, resume=args.resume)
    ae_gate = common._ae_gate(ae_train, ae_train['artifact_directory'])
    common._write(output/'ae_gate.json', ae_gate)
    if not ae_gate['passed'] and not args.skip_quality_gates:
        status('stopped', reason='AE train reconstruction failed', stage='ae'); return
    ae.eval(); ae.requires_grad_(False)
    encoded = common._encode_segments(ae, segments, device)
    stats.update(common._latent_stats(encoded)); common._save(output/'fit_stats.pt', stats)
    latent_binding = {**binding, 'ae_sha256': common._value_sha(ae.state_dict()), 'stats_sha256': common._value_sha(stats)}
    torch.manual_seed(SEED+2)
    prior = ContinuousLatentFlow(context_dim=int(segments[0]['context'].numel())).to(device)
    torch.nn.init.zeros_(prior.audio_block_projection.weight); torch.nn.init.zeros_(prior.audio_block_projection.bias)
    common._train_stage(prior, encoded, stats, stage='prior', budget=args.prior_steps,
        batch_size=args.batch_size, device=device, output=output, binding=latent_binding,
        seed=SEED+2, use_audio=False, resume=args.resume,
        completed_before=args.ae_steps, total_updates=total)
    prior_train = evaluate_cached(ae, prior, diagnostic, stats, output/'prior_generation/train_diagnostic', device,
                                  use_audio=False, resume=args.resume)
    prior_valid = evaluate_cached(ae, prior, valid, stats, output/'prior_generation/inner_validation', device,
                                  use_audio=False, resume=args.resume)
    prior_gate = common._prior_gate(prior_train); common._write(output/'prior_gate.json', prior_gate)
    if not prior_gate['passed'] and not args.skip_quality_gates:
        status('stopped', reason='Prior train engineering gate failed', stage='prior'); return
    prior_sha = common._value_sha(prior.state_dict())
    static_segments, static_report = prepare_segments(static_features(train), stats, 'train', 200)
    if [s['metadata'] for s in static_segments] != [s['metadata'] for s in segments]:
        raise RuntimeError('Matched static segment clocks differ')
    encoded_static = [{**s, 'audio': static_segments[i]['audio']} for i,s in enumerate(encoded)]
    torch.manual_seed(SEED+3)
    initial = BoundedPriorAudioAdapter(copy.deepcopy(prior), max_delta=args.max_delta).to(device)
    records = []; final_models = {}
    # Both arms are trained at each milestone before its diagnostics are read.
    for milestone in milestones:
        arm_results = {}
        for arm, data, intervention in [('audio', encoded, 'real'), ('matched_static', encoded_static, 'static')]:
            model = copy.deepcopy(initial)
            progress = common._train_stage(model, data, stats, stage=arm, budget=args.adapter_steps,
                batch_size=args.batch_size, device=device, output=output,
                binding={**latent_binding, 'prior_sha256': prior_sha}, seed=SEED+4,
                use_audio=True, resume=args.resume or milestone != milestones[0], stop_after=milestone,
                completed_before=args.ae_steps+args.prior_steps, total_updates=total)
            if common._value_sha(model.prior.state_dict()) != prior_sha:
                raise RuntimeError('Frozen prior changed')
            point = output/f'point_{milestone}'; point.mkdir(exist_ok=True)
            # Preserve only the two fixed endpoints, not a checkpoint sweep.
            checkpoint = point/(arm+'.pt')
            if not checkpoint.exists(): common._save(checkpoint, progress)
            else:
                saved = common._load(checkpoint)
                if saved['step'] != milestone: raise ValueError('Wrong milestone checkpoint')
                model.load_state_dict(saved['state'])
            results = evaluate_cached(ae, model, valid, stats, point/(arm+'_generation')/'inner_validation', device,
                intervention=intervention, resume=args.resume)
            arm_results[arm] = results
            final_models[arm] = model
        ac, sc = (common._load(point/(a+'.pt')) for a in ('audio', 'matched_static'))
        pairing = {k: ac[k] == sc[k] for k in ('initial_state_sha256', 'order_sha256', 'step')}
        if not all(pairing.values()): raise RuntimeError('Matched arm training pairing failed')
        common._write(point/'pairing.json', pairing)
        static_result = evaluate_cached(ae, final_models['audio'], valid, stats, point/'audio_static/inner_validation', device,
            intervention='static', resume=args.resume)
        acceptance = audit(prior_valid['artifact_directory'], arm_results['audio']['artifact_directory'],
            static_result['artifact_directory'], output=point/'acceptance.json',
            matched_static_dir=arm_results['matched_static']['artifact_directory'])
        records.append({'step': milestone, 'accepted': acceptance['accepted'], 'audit': str(point/'acceptance.json')})
        status('inner_evaluated', step=milestone, accepted=acceptance['accepted'])
    chosen = select_candidate(records)
    decision = {'records': records, 'chosen': chosen,
        'outer_evaluation': 'selected endpoint only' if chosen else 'fixed final diagnostic only; no promotion',
        'selection_data': 'inner_validation only', 'default_replaced': False}
    common._write(output/'decision.json', decision)
    # Whole payload is resident; outer motion targets are not scored or fitted
    # until this decision has been locked.
    point = output/f'point_{chosen["step"] if chosen else milestones[-1]}'
    for arm in ('audio','matched_static'):
        final_models[arm] = copy.deepcopy(initial)
        final_models[arm].load_state_dict(common._load(point/(arm+'.pt'))['state'])
    if outer:
        evaluate_cached(ae, prior, outer, stats, output/'outer/prior_generation/holdout', device,
                        use_audio=False, resume=args.resume)
        for arm, intervention in [('audio','real'),('matched_static','static')]:
            evaluate_cached(ae, final_models[arm], outer, stats, output/f'outer/{arm}_generation/holdout', device,
                            intervention=intervention, resume=args.resume)
        for intervention in ('static', 'reverse', 'mismatch'):
            evaluate_cached(ae, final_models['audio'], outer, stats, output/f'outer/audio_{intervention}/holdout', device,
                            intervention=intervention, resume=args.resume)
    status('complete', total_updates=total, accepted_inner=bool(chosen), decision=decision,
           quality_status='needs_visual_review', default_replaced=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--ae-steps', type=int, default=4000)
    p.add_argument('--prior-steps', type=int, default=14000); p.add_argument('--adapter-steps', type=int, default=3000)
    p.add_argument('--milestones', type=int, nargs='+', default=None); p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--valid-sentences', type=int, default=5); p.add_argument('--max-delta', type=float, default=.35)
    p.add_argument('--resume', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--skip-quality-gates', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try: run(args)
    except Exception as exc:
        common._write(args.output/'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'updated':time.time()})
        raise
