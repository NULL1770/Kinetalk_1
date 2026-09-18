"""Fixed-budget resumable latent-motion experiment; gates do not certify quality."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from scripts.evaluate_continuous_motion_latent import evaluate_ae, evaluate_generation
from scripts.prepare_continuous_motion_dataset import SCHEMA as DATA_SCHEMA, prepare_segments

SEED = 20260919
SCHEMA = 'continuous_motion_training_v2'
SOURCE_FILES = ('scripts/train_continuous_motion_latent.py', 'scripts/prepare_continuous_motion_dataset.py',
                'scripts/evaluate_continuous_motion_latent.py', 'scripts/joint_motion_metrics.py',
                'kinetalk_b0/models/continuous_upper_motion.py')


def _write(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def _save(path, payload):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary); temporary.replace(path)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''): h.update(block)
    return h.hexdigest()


def _value_sha(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x): h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: visit(item)
        else: h.update(json.dumps(x, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def _finite_state(model):
    return all(torch.isfinite(value).all().item() for value in model.state_dict().values())


def _pad_batch(items, key, device):
    width = items[0][key].shape[-1]; length = max(len(x[key]) for x in items)
    values = torch.zeros(len(items), length, width, device=device)
    valid = torch.zeros(len(items), length, dtype=torch.bool, device=device)
    for row, item in enumerate(items):
        n = len(item[key]); values[row, :n] = item[key].to(device); valid[row, :n] = True
    return values, valid


def _pad_audio(items, device, block_size):
    length = max(len(item['audio']) for item in items)
    length = ((length+block_size-1)//block_size)*block_size
    width = items[0]['audio'].shape[-1]
    audio = torch.zeros(len(items), length, width, device=device)
    mask = torch.zeros(len(items), length, dtype=torch.bool, device=device)
    for row, item in enumerate(items):
        n = len(item['audio']); audio[row, :n] = item['audio'].to(device); mask[row, :n] = True
    return audio.reshape(len(items), -1, block_size, width), mask.reshape(len(items), -1, block_size)


def _clip_groups(items):
    groups = {}
    for index, item in enumerate(items): groups.setdefault(item['metadata']['clip_id'], []).append(index)
    if not groups: raise ValueError('No training segments')
    return [groups[key] for key in sorted(groups)]


def _sample_indices(groups, size, rng):
    result = []
    for _ in range(size):
        group = groups[int(rng.integers(len(groups)))]
        result.append(group[int(rng.integers(len(group)))])
    return result


def _encode_segments(ae, segments, device):
    ae.eval(); result = []
    with torch.no_grad():
        for item in segments:
            values, valid = _pad_batch([item], 'residual', device)
            z, zvalid = ae.encode(values, valid)
            result.append({'z': z[0, :int(zvalid.sum())].cpu(), 'audio': item['audio'].cpu(),
                           'context': item['context'].cpu(), 'metadata': item['metadata']})
    return result


def _latent_stats(encoded):
    values = torch.cat([item['z'] for item in encoded]).double()
    return {'latent_mean': values.mean(0).float(), 'latent_scale': values.std(0, unbiased=False).clamp_min(.05).float(),
            'latent_train_segments': len(encoded), 'latent_weighting': 'equal latent tokens, not equal clips',
            'latent_source': 'AE fit train segments only'}


def _flow_batch(model, items, stats, device, seed, step):
    target, valid = _pad_batch(items, 'z', device)
    mean, scale = stats['latent_mean'].to(device), stats['latent_scale'].to(device)
    target = torch.where(valid[..., None], (target-mean)/scale, 0.)
    context = torch.stack([item['context'] for item in items]).to(device)
    audio, frame_valid = _pad_audio(items, device, model.block_size)
    if audio.shape[1] != target.shape[1] or not torch.equal(frame_valid.any(-1), valid):
        raise RuntimeError('Latent/audio block clocks disagree')
    generator = torch.Generator(device=device).manual_seed(seed*1000003+step)
    noise = torch.randn(target.shape, generator=generator, device=device)
    fraction = torch.rand(len(items), generator=generator, device=device)
    return target, valid, context, audio, noise, fraction, frame_valid


def _check_checkpoint(checkpoint, binding, model, stage, budget, init_sha):
    expected = {'binding': binding, 'config': model.config, 'stage': stage, 'budget': budget,
                'initial_state_sha256': init_sha}
    for key, value in expected.items():
        if checkpoint.get(key) != value: raise ValueError(f'Resume mismatch in {stage}: {key}')
    if not 0 <= checkpoint['step'] <= budget: raise ValueError('Invalid saved step')


def _train_stage(model, items, stats, *, stage, budget, batch_size, device, output,
                 binding, seed, use_audio=None, resume=False, stop_after=None,
                 completed_before=0, total_updates=None):
    """Exact fixed-budget recovery; stop_after only provides an interruption test hook."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    initial_sha = _value_sha(model.state_dict())
    final_path, last_path = output/(stage+'_final.pt'), output/(stage+'_last.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed); first = 0; losses = []; chain = '0'*64
    training_seconds = 0.; profile = None
    if final_path.exists() or last_path.exists():
        if not resume: raise FileExistsError(f'{stage} checkpoint exists; explicit --resume required')
        saved = _load(final_path if final_path.exists() else last_path)
        _check_checkpoint(saved, binding, model, stage, budget, initial_sha)
        model.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['batch_rng']; first = saved['step']
        losses, chain = saved['losses'], saved['order_sha256']
        training_seconds, profile = saved['training_seconds'], saved.get('profile')
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda' and saved.get('cuda_rng') is not None: torch.cuda.set_rng_state(saved['cuda_rng'], device)
        if final_path.exists():
            if first != budget: raise ValueError('Final checkpoint does not finish fixed budget')
            return saved
    groups = _clip_groups(items); model.train(); end_step = budget if stop_after is None else min(budget, stop_after)
    def checkpoint(step):
        return {'schema': SCHEMA, 'stage': stage, 'step': step, 'budget': budget,
                'binding': binding, 'config': model.config, 'state': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'stats': stats, 'batch_rng': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                'initial_state_sha256': initial_sha, 'order_sha256': chain, 'losses': losses,
                'training_seconds': training_seconds, 'profile': profile, 'use_audio': use_audio,
                'sampling': 'uniform clip then uniform supervised segment',
                'loss_weighting': 'equal examples' if use_audio is None else 'equal valid latent token coordinates'}
    for step in range(first+1, end_step+1):
        if device.type == 'cuda': torch.cuda.synchronize(device)
        tick = time.perf_counter()
        indices = _sample_indices(groups, batch_size, rng)
        chain = hashlib.sha256(bytes.fromhex(chain)+np.asarray(indices, dtype='<i8').tobytes()).hexdigest()
        batch = [items[index] for index in indices]
        if use_audio is None:
            values, valid = _pad_batch(batch, 'residual', device)
            error = torch.where(valid[..., None], model(values, valid)-values, 0.)
            loss = (error.square().sum((1, 2))/(valid.sum(1)*values.shape[-1])).mean()
        else:
            target, valid, context, audio, noise, fraction, frames = _flow_batch(model, batch, stats, device, seed, step)
            loss = model.flow_loss(target, valid, context, audio, noise, fraction,
                                   use_audio=use_audio, audio_frame_valid=frames)
        if not torch.isfinite(loss): raise FloatingPointError(f'{stage} nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): raise FloatingPointError(f'{stage} nonfinite gradient')
        optimizer.step()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        training_seconds += time.perf_counter()-tick; losses.append(float(loss.detach()))
        if step == 100 or (step == end_step and profile is None):
            profile = {'updates': step, 'actual_seconds': training_seconds, 'seconds_per_update': training_seconds/step,
                       'scope': 'training updates only; evaluation/export time excluded'}
        if step == 1 or step == 100 or step % 250 == 0 or step == end_step:
            remaining = max(0, (total_updates or budget)-completed_before-step)
            progress = {'stage': stage, 'step': step, 'steps': budget, 'loss': losses[-1],
                        'mean_loss': float(np.mean(losses[-100:])), 'training_seconds': training_seconds,
                        'profile': profile, 'estimated_remaining_training_seconds': training_seconds/step*remaining,
                        'state': 'training', 'updated': time.time(), 'needs_visual_review': True}
            _write(output/(stage+'_progress.json'), progress); _write(output/'status.json', progress)
            _save(last_path, checkpoint(step))
    saved = checkpoint(end_step)
    if end_step == budget:
        if not _finite_state(model): raise FloatingPointError(f'{stage} parameters nonfinite')
        _save(final_path, saved)
    return saved


def _diagnostic_members(clips):
    train = sorted([c for c in clips if c['split'] == 'train'], key=lambda c: c['clip_id'])
    hold = sorted([c for c in clips if c['split'] == 'holdout'], key=lambda c: c['clip_id'])
    cells = {}
    for clip in train: cells.setdefault((str(clip['speaker']), str(clip['emotion'])), []).append(clip)
    selected = []; index = 0
    while len(selected) < min(16, len(train)):
        for key in sorted(cells):
            if index < len(cells[key]): selected.append(cells[key][index])
            if len(selected) >= 16: break
        index += 1
    return {'train_diagnostic': selected, 'holdout': hold[:64]}


def _evaluate(ae, flow, clips, stats, destination, device, *, use_audio=False, intervention='real', resume=False):
    destination = Path(destination)
    canonical = destination
    pointer = canonical.with_name(canonical.name+'_artifact.json')
    if resume and pointer.exists():
        destination = Path(json.loads(pointer.read_text(encoding='utf8'))['directory'])
    if (destination/'result.json').exists() and (destination/'curves.pt').exists():
        if not resume: raise FileExistsError('Evaluation already exists')
        return json.loads((destination/'result.json').read_text(encoding='utf8'))
    if destination.exists() and any(destination.iterdir()):
        index = 1
        while destination.with_name(destination.name+f'_recovery{index}').exists(): index += 1
        destination = destination.with_name(destination.name+f'_recovery{index}')
    result = (evaluate_ae(ae, clips, stats, destination, device) if flow is None else
              evaluate_generation(flow, ae, clips, stats, destination, device, use_audio=use_audio,
                                  intervention=intervention, seeds=(42, 123, 2026, 77), steps=24))
    result['artifact_directory'] = str(destination); _write(destination/'result.json', result)
    _write(pointer, {'directory': str(destination)})
    return result


def _ae_gate(result, directory):
    curves = _load(Path(directory)/'curves.pt')['clips']
    error = variance = raw_error = constant_error = 0.
    for row in curves.values():
        mask = np.asarray(row['score_mask'], bool); truth = np.asarray(row['target'], float)[mask]
        prediction = np.asarray(row['samples'], float)[0, mask]
        if not len(truth): continue
        tc, pc = truth-truth.mean(0), prediction-prediction.mean(0)
        error += float(np.square(pc-tc).sum()); variance += float(np.square(tc).sum())
        raw_error += float(np.square(prediction-truth).sum()); constant_error += float(np.square(tc).sum())
    r2 = 1-error/variance if variance > 0 else None
    passed = bool(result['numerical_gate']['passed'] and r2 is not None and r2 > 0 and raw_error < constant_error)
    return {'passed': passed, 'scope': 'fixed train diagnostics only; capacity gate, not naturalness',
            'centered_pooled_r2': r2, 'raw_sse': raw_error, 'same_clip_oracle_constant_sse': constant_error,
            'numerical_gate': result['numerical_gate'], 'needs_visual_review': True}


def _prior_gate(result):
    summary = result.get('summary') or {}; failures = []
    if not result['numerical_gate']['passed']: failures.append('nonfinite or protected output mismatch')
    speed = summary.get('speed', {}); pred = (speed.get('all') or {}).get('rms'); ref = (speed.get('reference_all') or {}).get('rms')
    ratio = pred/ref if pred is not None and ref is not None and ref > 0 else None
    if ratio is None or not np.isfinite(ratio) or ratio > 5 or ratio < .02: failures.append('speed ratio outside [.02,5] or unavailable')
    rms = summary.get('rms_ratio')
    if not rms or any(x is None or not np.isfinite(x) for x in rms): failures.append('unavailable group RMS')
    elif all(x < .05 for x in rms) or any(x > 5 for x in rms): failures.append('collapsed or excessive group RMS')
    checks = result['numerics']; count = sum(x['native_generated_values'] for x in checks)
    outside = sum(x['raw_upper_out_of_range_count'] for x in checks); rate = outside/count if count else None
    minima = [x['raw_upper_min'] for x in checks if x['raw_upper_min'] is not None]
    maxima = [x['raw_upper_max'] for x in checks if x['raw_upper_max'] is not None]
    if not minima or min(minima) < -1 or max(maxima) > 2: failures.append('raw values outside [-1,2]')
    if rate is None or rate > .35: failures.append('raw out-of-range fraction above .35')
    return {'passed': not failures, 'failures': failures, 'speed_ratio': ratio, 'group_rms_ratio': rms,
            'raw_out_of_range_fraction': rate, 'scope': 'fixed train diagnostics; broad catastrophic continuation gate only',
            'needs_visual_review': True, 'naturalness_certified': False}


def _protocol(args, payload, members):
    root = Path(__file__).resolve().parents[1]
    return {'schema': SCHEMA, 'dataset_sha256': _file_sha(args.dataset),
            'source_sha256': {name: _file_sha(root/name) for name in SOURCE_FILES},
            'dataset_schema': payload.get('schema'), 'data_stats_sha256': _value_sha(payload['stats']),
            'members': [{'clip_id': c['clip_id'], 'split': c['split'], 'sentence': c['sentence']} for c in payload['clips']],
            'diagnostic_members': {key: [c['clip_id'] for c in value] for key, value in members.items()},
            'budgets': {'ae': args.ae_steps, 'prior': args.prior_steps, 'matched_global': args.adapt_steps, 'audio': args.adapt_steps},
            'batch_size': args.batch_size, 'seed': SEED, 'device_type': torch.device(args.device).type,
            'smoke': args.smoke, 'skip_quality_gates': args.skip_quality_gates,
            'training_max_native_frames': 200, 'evaluation_native_runs_full_length': True,
            'base_outputs_frozen': True, 'va_used': False, 'default_replaced': False,
            'ae_gate': 'train centered pooled R2>0 and raw SSE<same-clip oracle constant SSE; finite/protection',
            'prior_gate': 'train finite/protection; speed [.02,5]; not all RMS<.05 nor any>5; raw [-1,2]; oob<=.35',
            'sampling': 'uniform clip then uniform segment; AE equal examples; FM valid token coordinates',
            'latent_statistics': 'equal latent tokens in training segments',
            'paired_adaptation': 'same prior init, same batch seed, explicit noise/time seed, equal update budget',
            'audio_initialization': 'audio_block_projection weight/bias zero before prior; both clones inherit zero',
            'quality_claim': 'engineering continuation only; needs_visual_review'}


def run(args):
    for key in ('ae_steps', 'prior_steps', 'adapt_steps', 'batch_size'):
        if getattr(args, key) < 1: raise ValueError('Positive training budgets/batch required')
    if args.skip_quality_gates and (not args.smoke or max(args.ae_steps, args.prior_steps, args.adapt_steps) > 100):
        raise ValueError('--skip-quality-gates requires --smoke and every stage <=100 updates')
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device(args.device)
    if device.type == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); protocol_path = output/'protocol.json'
    if protocol_path.exists() and not args.resume: raise FileExistsError('Training protocol exists; use --resume')
    if args.resume and not protocol_path.exists(): raise FileNotFoundError('Resume requires an existing training protocol')
    if not protocol_path.exists() and (any(output.glob('*_last.pt')) or any(output.glob('*_final.pt'))):
        raise ValueError('Unbound checkpoints exist')
    payload = _load(args.dataset)
    if payload.get('schema') != DATA_SCHEMA: raise ValueError('Unexpected dataset schema')
    clips, stats = payload['clips'], dict(payload['stats'])
    train_sentences = {c['sentence'] for c in clips if c['split'] == 'train'}
    hold_sentences = {c['sentence'] for c in clips if c['split'] == 'holdout'}
    if train_sentences & hold_sentences: raise ValueError('Train/holdout sentence overlap')
    if set(stats['train_clip_ids']) != {c['clip_id'] for c in clips if c['split'] == 'train'}:
        raise ValueError('Fit statistics membership mismatch')
    members = _diagnostic_members(clips)
    if args.smoke:
        members = {split: subset[:4] for split, subset in members.items()}
    if not all(members.values()): raise ValueError('Train diagnostic and holdout members required')
    protocol = _protocol(args, payload, members)
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf8')) != protocol: raise ValueError('Resume protocol/code/dataset/budget mismatch')
    else: _write(protocol_path, protocol)
    binding = {'protocol_sha256': _value_sha(protocol), 'dataset_sha256': protocol['dataset_sha256']}
    segments, report = prepare_segments(clips, stats, 'train', max_frames=200)
    _write(output/'segment_report.json', report)
    if not segments: raise ValueError('No supervised training segments')
    total = args.ae_steps+args.prior_steps+2*args.adapt_steps
    def status(state, **extra):
        _write(output/'status.json', {'schema': SCHEMA, 'state': state, 'updated': time.time(),
                                     'needs_visual_review': True, 'naturalness_certified': False, **extra})
    def evaluate_splits(stage, ae, flow=None, use_audio=False, intervention='real'):
        results = {}
        for split, subset in members.items():
            destination = output/stage/split
            results[split] = _evaluate(ae, flow, subset, stats, destination, device, use_audio=use_audio,
                                       intervention=intervention, resume=args.resume)
            results[split].setdefault('artifact_directory', str(destination))
        return results
    def gate_stage(stage, gate):
        gate['continued_under_smoke_override'] = bool(args.skip_quality_gates and not gate['passed'])
        _write(output/(stage+'_gate.json'), gate)
        if not gate['passed'] and not args.skip_quality_gates:
            status('stopped', stage=stage, reason='Fixed engineering gate failed', gate=gate); return False
        return True
    ae = ContinuousUpperAE().to(device)
    _train_stage(ae, segments, stats, stage='ae', budget=args.ae_steps, batch_size=args.batch_size,
                 device=device, output=output, binding=binding, seed=SEED+1, resume=args.resume,
                 completed_before=0, total_updates=total)
    ae_results = evaluate_splits('ae_reconstruction', ae)
    ae_gate = _ae_gate(ae_results['train_diagnostic'], ae_results['train_diagnostic']['artifact_directory'])
    if not gate_stage('ae', ae_gate): return 0
    encoded = _encode_segments(ae, segments, device)
    stats.update(_latent_stats(encoded)); _save(output/'fit_stats.pt', stats)
    flow_binding = {**binding, 'ae_state_sha256': _value_sha(ae.state_dict()), 'stats_sha256': _value_sha(stats)}
    torch.manual_seed(SEED+2)
    flow = ContinuousLatentFlow(context_dim=int(segments[0]['context'].numel())).to(device)
    torch.nn.init.zeros_(flow.audio_block_projection.weight); torch.nn.init.zeros_(flow.audio_block_projection.bias)
    _train_stage(flow, encoded, stats, stage='prior', budget=args.prior_steps, batch_size=args.batch_size,
                 device=device, output=output, binding=flow_binding, seed=SEED+2, use_audio=False,
                 resume=args.resume, completed_before=args.ae_steps, total_updates=total)
    prior_results = evaluate_splits('prior_generation', ae, flow, False)
    if not gate_stage('prior', _prior_gate(prior_results['train_diagnostic'])): return 0
    initial_sha = _value_sha(flow.state_dict()); final_models = {}
    for index, (stage, use_audio) in enumerate((('matched_global', False), ('audio', True))):
        adapted = copy.deepcopy(flow)
        _train_stage(adapted, encoded, stats, stage=stage, budget=args.adapt_steps, batch_size=args.batch_size,
                     device=device, output=output, binding=flow_binding, seed=SEED+3, use_audio=use_audio,
                     resume=args.resume, completed_before=args.ae_steps+args.prior_steps+index*args.adapt_steps,
                     total_updates=total)
        results = evaluate_splits(stage+'_generation', ae, adapted, use_audio)
        _write(output/(stage+'_gate.json'), _prior_gate(results['train_diagnostic']))
        final_models[stage] = adapted
    global_ck, audio_ck = _load(output/'matched_global_final.pt'), _load(output/'audio_final.pt')
    pairing = {'initial_state_identical': global_ck['initial_state_sha256'] == audio_ck['initial_state_sha256'] == initial_sha,
               'batch_order_identical': global_ck['order_sha256'] == audio_ck['order_sha256'],
               'updates_identical': global_ck['step'] == audio_ck['step'] == args.adapt_steps,
               'noise_time_identical': True, 'noise_seed': SEED+3}
    if not all(pairing[key] for key in ('initial_state_identical', 'batch_order_identical', 'updates_identical')):
        raise RuntimeError('Matched adaptation pairing failed')
    _write(output/'paired_training.json', pairing)
    for intervention in ('static', 'reverse', 'mismatch'):
        evaluate_splits('audio_'+intervention, ae, final_models['audio'], True, intervention)
    status('complete', total_updates=total, quality_status='needs_visual_review', smoke=args.smoke,
           skip_quality_gates=args.skip_quality_gates, stages=['ae', 'prior', 'matched_global', 'audio'])
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--ae-steps', type=int, default=4000)
    parser.add_argument('--prior-steps', type=int, default=8000); parser.add_argument('--adapt-steps', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=24); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--skip-quality-gates', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    arguments = parse_args()
    try:
        raise SystemExit(run(arguments))
    except Exception as exc:
        _write(arguments.output/'status.json', {'schema': SCHEMA, 'state': 'failed',
               'error': f'{type(exc).__name__}: {exc}', 'updated': time.time()})
        raise
