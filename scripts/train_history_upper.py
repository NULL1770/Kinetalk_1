"""Matched chunk flow with absent versus scheduled previous-motion context.

Deployment rolls out generated past chunks only. Ground-truth past is restricted
to training and a separately named oracle evaluation. No full-clip GT mean is
ever used in history normalization or in an inference condition.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_temporal_repair as r
from scripts.audit_temporal_repair import summarize, metadata_equal
from scripts.train_centered_temporal_prior import time_intervention, temporal_stats
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, capture_rng, canonical_hash
from scripts.train_predictable_renderer import state_hash
from kinetalk_b0.models.history_upper_flow import HistoryUpperFlow
from kinetalk_b0.models.slow_state_affect import lift_slow_state, compose_upper_face

SCHEMA = 'scheduled_history_upper_v1'
CC = list(r.UPPER_INDICES)
CHUNK = 16
HISTORY = 8


def teacher_probability(epoch, epochs):
    """Epoch is zero based; final third of budget uses generated history only."""
    end = max(1, (2*epochs)//3-1)
    return .75*max(0., 1.-epoch/end)


@torch.no_grad()
def cache_audio(system, audio, data, device, batch_size):
    for q in data['splits'].values():
        cache = {key: [] for key in ('audio_global', 'audio_intensity', 'audio_local', 'static_upper')}
        if not (q['channel_mask'][:, CC] & q['anchor_valid'][:, CC]).all():
            raise ValueError('Both roles require nine observed channels and independent anchors')
        for ix in torch.arange(len(q['valid'])).split(batch_size):
            b = r.subset(q, ix, device)
            a = audio(b['audio_features'], b['valid'])
            state = torch.where(b['valid'][..., None], a['state'], 0.).sum(1, keepdim=True)/b['valid'].sum(1)[:, None, None]
            static = b['anchors'][:, None]+lift_slow_state(state, data['target_scales'].to(device))
            for key, value in [('audio_global', a['global']), ('audio_intensity', a['intensity_value']),
                               ('audio_local', a['local']), ('static_upper', static[:, 0, CC])]:
                cache[key].append(value.cpu())
        q.update({key: torch.cat(values) for key, values in cache.items()})


def normalized_target(b, scales):
    # The baseline is predicted from audio and independent enrollment, never a
    # query mean. Teacher history at t therefore contains no future GT statistic.
    return torch.where(b['valid'][..., None], (b['motion'][..., CC]-b['static_upper'][:, None])/scales, 0.)


def select_conditions(b, identity, local, content, ids, start, stop):
    return (b['valid'][ids, start:stop], content[ids, start:stop], identity['code'][ids],
            {'global': b['audio_global'][ids], 'intensity_value': b['audio_intensity'][ids]}, local[ids, start:stop])


def history_at(source, valid, ids, start, *, mode='generated'):
    left = max(0, start-HISTORY)
    history = source[ids, left:start]
    mask = valid[ids, left:start]
    if mode == 'empty':
        return history[:, :0], mask[:, :0]
    if mode == 'reverse_history':
        history = history.clone()
        for row in range(len(ids)):
            good = mask[row].nonzero(as_tuple=True)[0]
            history[row, good] = history[row, good.flip(0)]
    return history, mask


@torch.no_grad()
def rollout(upper, b, identity, local, noise, *, steps=12, mode='full', oracle_target=None):
    """Autoregressive motion history; audio features may use offline context."""
    if mode not in ('full', 'empty', 'reverse_history', 'static', 'reverse', 'oracle_history'):
        raise ValueError('Unknown history rollout mode')
    if mode == 'oracle_history' and oracle_target is None:
        raise ValueError('Oracle mode requires explicit target, never an implicit fallback')
    if mode != 'oracle_history' and oracle_target is not None:
        raise ValueError('Deployable modes must not receive target history')
    native, content = time_intervention(local, b['h0'], b['valid'], mode)
    output = torch.zeros_like(noise)
    for start in range(0, noise.shape[1], CHUNK):
        stop = min(start+CHUNK, noise.shape[1])
        ids = b['valid'][:, start:stop].any(1).nonzero(as_tuple=True)[0]
        if not len(ids):
            continue
        source = oracle_target if mode == 'oracle_history' else output
        history, history_valid = history_at(source, b['valid'], ids, start, mode=mode)
        conditions = select_conditions(b, identity, native, content, ids, start, stop)
        prediction = upper.decode_history(*conditions, noise[ids, start:stop], history=history,
                                           history_valid=history_valid, steps=steps)
        output[ids, start:stop] = prediction
    return output


def chunk_training_loss(upper, b, identity, native, target, generated, noise, times, teacher_draw, probability):
    total = target.new_zeros(())
    used_teacher, available = 0, 0
    for chunk_index, start in enumerate(range(0, target.shape[1], CHUNK)):
        stop = min(start+CHUNK, target.shape[1])
        ids = b['valid'][:, start:stop].any(1).nonzero(as_tuple=True)[0]
        if not len(ids):
            continue
        gt, history_valid = history_at(target, b['valid'], ids, start)
        synthetic, _ = history_at(generated.detach(), b['valid'], ids, start)
        use_teacher = teacher_draw[ids, chunk_index] < probability
        history = torch.where(use_teacher[:, None, None], gt, synthetic)
        has_history = history_valid.any(1)
        used_teacher += int((use_teacher & has_history).sum()); available += int(has_history.sum())
        conditions = select_conditions(b, identity, native, b['h0'], ids, start, stop)
        loss = upper.flow_loss_history(target[ids, start:stop], *conditions, noise[ids, start:stop],
            times[ids, chunk_index], history=history, history_valid=history_valid)
        total = total+loss*conditions[0].sum()
    return total/b['valid'].sum(), used_teacher, available


def boundary_report(pred, target, valid):
    out = {}
    pair_mask = valid[:, 1:] & valid[:, :-1]
    boundary = torch.arange(1, valid.shape[1], device=valid.device).remainder(CHUNK) == 0
    for name, cc in [('brows', CC[:5]), ('eyes_expression', CC[5:])]:
        region = {}
        for tag, select in [('chunk_boundary', pair_mask & boundary[None]), ('inside_chunk', pair_mask & ~boundary[None])]:
            p = (pred[:, 1:, cc]-pred[:, :-1, cc])[select].double()
            t = (target[:, 1:, cc]-target[:, :-1, cc])[select].double()
            if not p.numel():
                region[tag] = None; continue
            region[tag] = {'pairs': int(select.sum()), 'pred_rms': float(p.square().mean().sqrt()),
                'gt_rms': float(t.square().mean().sqrt()), 'displacement_mse': float((p-t).square().mean())}
        out[name] = region
    return out


@torch.no_grad()
def evaluate(system, upper, local, data, identities, args, scales, bases, steps):
    q = data['splits']['validation']; count = min(32, len(q['valid'])) if args.smoke else len(q['valid'])
    ids = torch.arange(count); reference = r.subset(q, ids, 'cpu'); predictions = {}; records = {}
    seeds = (42,) if args.smoke else r.SEEDS
    for seed in seeds:
        noise = torch.randn(len(q['valid']), q['valid'].shape[1], 9, generator=torch.Generator().manual_seed(seed))
        modes = ['full']+(['empty', 'reverse_history', 'static', 'reverse', 'oracle_history'] if seed == 42 else [])
        for mode in modes:
            results, emotions = [], []
            for ix in ids.split(args.batch_size):
                b = r.subset(q, ix, args.device); ident = r.batch_identity(identities, b)
                native = local(b['audio_features'], b['valid'])['local']
                kwargs = {'oracle_target': normalized_target(b, scales)} if mode == 'oracle_history' else {}
                dynamic = rollout(upper, b, ident, native, noise[ix].to(args.device), steps=steps, mode=mode, **kwargs)
                baseline = bases['predictions'][f'{seed}/base'][ix].to(args.device)
                upper_values = b['static_upper'][:, None]+dynamic*scales
                pred = compose_upper_face(baseline, upper_values, b['valid'])
                if not torch.equal(pred[..., list(r.NOT_UPPER)], baseline[..., list(r.NOT_UPPER)]):
                    raise RuntimeError('Nonupper baseline changed')
                readout = system.encode_motion(torch.where(r.obs(b), pred-b['b0']-ident['baseline'][:, None], 0.), b['valid'])
                emotions.extend((readout['emotion_logits'].argmax(-1) == b['emotion_id']).cpu().tolist())
                results.append(pred.cpu())
            key = f'{seed}/{mode}'; predictions[key] = torch.cat(results)
            records[key] = {'populations': r.populations(predictions[key], reference),
                'temporal': temporal_stats(predictions[key], reference['motion'], reference['valid']),
                'boundaries': boundary_report(predictions[key], reference['motion'], reference['valid']),
                'generated_emotion_accuracy_nonindependent': sum(emotions)/len(emotions),
                'oracle_target_history': mode == 'oracle_history'}
    curves = {key: reference[key] for key in ('clip_id', 'motion', 'valid', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    curves['target'] = curves.pop('motion')
    curves.update(schema=SCHEMA, predictions=predictions, noise_seeds=list(seeds), decode_steps=steps)
    report = {'schema': SCHEMA, 'clips': count, 'modes': records, 'nonupper_exact': True,
        'deploy_history': 'Generated previous chunks only, no query motion or query mean',
        'test_loaded': False, 'default_replaced': False}
    if not args.smoke:
        report['distribution'] = summarize(curves, reference['emotion_id'])
    return report, curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'audio', 'targets', 'enrollment', 'native-root', 'trained-run', 'align-run', 'centered-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=67); p.add_argument('--device', default='cuda'); p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise FileExistsError('Fresh history experiment required')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Positive budget required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    data = r.load_training_inputs(args.source_run, args.audio, args.targets, args.enrollment, args.native_root)
    checkpoints, bindings, steps, stride = r.matching_checkpoints(args.trained_run, data)
    system = data['system'].to(args.device).eval().requires_grad_(False)
    system.load_state_dict(checkpoints['teacher']['system'])
    audio = r._make_audio(checkpoints['audio']['audio'], stride, args.device)
    alignpath = args.align_run/'final.pt'
    if r.sha(alignpath) != json.loads((args.align_run/'complete.json').read_text())['final_sha256']:
        raise ValueError('Alignment binding differs')
    aligned = torch.load(alignpath, map_location='cpu', weights_only=False)
    if aligned['source_bindings'] != bindings or aligned['phase'] != 'align' or aligned['config'] != data['config']:
        raise ValueError('Alignment lineage differs')
    for query in data['splits'].values():
        delta = query['times'][:, 1:]-query['times'][:, :-1]
        if not torch.allclose(delta, torch.full_like(delta, .04), atol=1e-7, rtol=1e-5):
            raise ValueError('History ages require the locked native 25Hz clock')
    r.cache_current_base(system, data, args.device, args.batch_size)
    identities = r.identity_cache(system, data, args.device)
    cache_audio(system, audio, data, args.device, args.batch_size)
    basepath = args.centered_run/'white/curves.pt'
    complete = json.loads((args.centered_run/'white/complete.json').read_text())
    recipe_old = json.loads((args.centered_run/'white/provenance.json').read_text())['recipe']
    if r.sha(basepath) != complete['curves_sha256'] or recipe_old['source_bindings'] != bindings:
        raise ValueError('Frozen baseline binding differs')
    bases = torch.load(basepath, map_location='cpu', weights_only=False, mmap=True)
    q = data['splits']['validation']; metadata_equal({**q, 'target': q['motion']}, bases)
    train = data['splits']['train']
    residual = torch.where(train['valid'][..., None], train['motion'][..., CC]-train['static_upper'][:, None], 0.)
    scales = (residual.square().sum((0, 1))/train['valid'].sum()).sqrt().clamp_min(.02).to(args.device)
    frozen = {name: state_hash(module.state_dict()) for name, module in [('system', system), ('audio', audio)]}
    args.output.mkdir(parents=True)
    initial_hashes, draw_hashes = {}, {}
    for arm in ('no_history', 'scheduled_history'):
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
        local = copy.deepcopy(audio).requires_grad_(False)
        for name in ('input', 'blocks'):
            getattr(local, name).load_state_dict({key.removeprefix(name+'.'): value for key, value in aligned['adapter'].items() if key.startswith(name+'.')})
        torch.nn.init.zeros_(local.local_head.weight); torch.nn.init.zeros_(local.local_head.bias)
        for module in (local.input, local.blocks, local.local_head): module.requires_grad_(True)
        upper = HistoryUpperFlow(data['config'], use_history=arm == 'scheduled_history').to(args.device).eval()
        params = list(upper.parameters())+[param for param in local.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
        initial_hashes[arm] = {'upper': state_hash(upper.state_dict()), 'local': state_hash(local.state_dict())}
        output = args.output/arm; output.mkdir()
        epochs = 1 if args.smoke else args.epochs
        recipe = {'schema': SCHEMA, 'arm': arm, 'epochs': epochs, 'seed': args.seed, 'batch_size': args.batch_size,
            'smoke': args.smoke, 'chunk': CHUNK, 'history': HISTORY, 'decode_steps': steps,
            'source_bindings': bindings, 'alignment_sha256': r.sha(alignpath), 'baseline_curves_sha256': r.sha(basepath),
            'initial': initial_hashes[arm], 'frozen': frozen, 'scales': scales.cpu().tolist(), 'config': data['config'],
            'data_provenance': data['provenance'], 'teacher_probability': [teacher_probability(e, epochs) for e in range(epochs)],
            'objective': 'Full normalized upper residual relative to predicted static mean; flow matching only; no per-chunk centering',
            'generated_history': 'Before update, full model self-rollout with generated past only, detached. Same 12-step solver at fit and deploy.',
            'fit_target_mean': 'Never centered using query statistics; GT past cannot leak future via target centering',
            'test_loaded': False, 'default_replaced': False, 'fixed_final_epoch': True,
            'source_sha256': {path: r.sha(Path(__file__).resolve().parents[1]/path) for path in (
                'scripts/train_history_upper.py', 'kinetalk_b0/models/history_upper_flow.py', 'kinetalk_b0/models/temporal_upper.py',
                'docs/HISTORY_CONTEXT_PROTOCOL_20260917.md')}}
        digest = canonical_hash(recipe); save_json(output/'provenance.json', {'recipe': recipe, 'recipe_sha256': digest})
        generator = torch.Generator().manual_seed(args.seed)
        ids = torch.arange(min(32, len(train['valid'])) if args.smoke else len(train['valid']))
        draw_hashes[arm] = []; total_steps = 0
        for epoch in range(epochs):
            t0 = time.monotonic(); losses = []; draws = hashlib.sha256(); teacher_used = history_available = 0
            probability = teacher_probability(epoch, epochs)
            for ix in ids[torch.randperm(len(ids), generator=generator)].split(args.batch_size):
                b = r.subset(train, ix, args.device); ident = r.batch_identity(identities, b)
                target = normalized_target(b, scales); chunks = (target.shape[1]+CHUNK-1)//CHUNK
                noise = torch.randn(target.shape, generator=generator)
                rollout_noise = torch.randn(target.shape, generator=generator)
                flow_times = torch.rand(len(ix), chunks, generator=generator)
                decisions = torch.rand(len(ix), chunks, generator=generator)
                for tensor in (ix, noise, rollout_noise, flow_times, decisions): draws.update(tensor.numpy().tobytes())
                with torch.no_grad():
                    local_detached = local(b['audio_features'], b['valid'])['local']
                    generated = rollout(upper, b, ident, local_detached, rollout_noise.to(args.device), steps=steps)
                native = local(b['audio_features'], b['valid'])['local']
                loss, used, available = chunk_training_loss(upper, b, ident, native, target, generated,
                    noise.to(args.device), flow_times.to(args.device), decisions.to(args.device), probability)
                r.optimize(loss, optimizer, params); losses.append(float(loss.detach())); total_steps += 1
                teacher_used += used; history_available += available
            draw_hashes[arm].append(draws.hexdigest())
            row = {'arm': arm, 'epoch': epoch+1, 'loss': sum(losses)/len(losses), 'seconds': time.monotonic()-t0,
                'teacher_probability': probability, 'teacher_selected_count': teacher_used,
                'history_available_count': history_available, 'no_history_arm_ignores_all_history': arm == 'no_history',
                'batch_noise_time_decision_sha256': draws.hexdigest(), 'total_steps': total_steps}
            save_json(output/f'epoch{epoch+1:03d}.json', row)
            payload = {'schema': SCHEMA, 'arm': arm, 'upper': upper.state_dict(), 'local': local.state_dict(),
                'scales': scales.cpu(), 'recipe_sha256': digest, 'completed_epochs': epoch+1, 'total_steps': total_steps,
                'optimizer': optimizer.state_dict(), 'rng': capture_rng(generator)}
            save_checkpoint(output/'last.pt', payload); save_json(args.output/'status.json', {'status': 'training', **row})
            print(json.dumps(row), flush=True)
        for name, module in [('system', system), ('audio', audio)]:
            if state_hash(module.state_dict()) != frozen[name] or any(param.grad is not None for param in module.parameters()):
                raise RuntimeError('Frozen subsystem changed: '+name)
        report, curves = evaluate(system, upper, local, data, identities, args, scales, bases, steps)
        save_json(output/'evaluation.json', report); save_checkpoint(output/'curves.pt', curves)
        save_checkpoint(output/'final.pt', {k: v for k, v in payload.items() if k not in ('optimizer', 'rng')})
        save_json(output/'complete.json', {'final_sha256': r.sha(output/'final.pt'), 'curves_sha256': r.sha(output/'curves.pt'),
            'completed_epochs': epochs, 'recipe_sha256': digest, 'frozen': frozen})
    if initial_hashes['no_history'] != initial_hashes['scheduled_history'] or draw_hashes['no_history'] != draw_hashes['scheduled_history']:
        raise RuntimeError('Pair matching differs')
    save_json(args.output/'matched_audit.json', {'initial_equal': True, 'random_streams_equal': True,
        'initial_sha256': initial_hashes, 'epoch_draw_sha256': draw_hashes})
    save_json(args.output/'status.json', {'status': 'complete', 'seconds': time.monotonic()-started,
        'epochs_per_arm': epochs, 'smoke': args.smoke})
    print('HISTORY_UPPER_COMPLETE', flush=True)


if __name__ == '__main__': main()
