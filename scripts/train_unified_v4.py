"""Resumable KineTalk v4 training runner.

This runner is intentionally isolated from the v2/v3 experiments.  It uses
the frozen neutral B0/content front-end and trains the v4 geometry-aware,
event-conditioned and factor-interaction model on the existing TRAIN fold.
No validation/test arrays are loaded as training targets.  The first 768
audio dimensions are content features; the final 772 dimensions are the
deployable affect student input.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import (
    load_runtime, development_fold, configure, state_digest, write, save,
)
from scripts.train_full_staged import base_forward
from kinetalk_b0.models.expression_response import motion_objective
from kinetalk_v2.unified_model import (
    teacher_targets, relation_style_loss,
)
from kinetalk_v2.event_execution import event_execution_loss
from kinetalk_v4.geometry import load_rig_geometry
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model, v4_losses


def select_balanced(q, ids, limit, seed):
    if not limit or len(ids) <= limit:
        return list(ids)
    rng = random.Random(seed)
    groups = {}
    for i in ids:
        groups.setdefault((int(q['speaker_id'][i]), int(q['emotion_id'][i])), []).append(i)
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < limit:
        for key in sorted(groups):
            if groups[key]:
                chosen.append(groups[key].pop())
            if len(chosen) >= limit:
                break
    return chosen


def reference_ids(q, query_ids, pool, rng):
    refs = []
    for i in query_ids:
        candidates = [j for j in pool[int(q['speaker_id'][i])]
                      if q['clip_id'][j] != q['clip_id'][i]
                      and q['sentence_id'][j] != q['sentence_id'][i]]
        cross = [j for j in candidates if q['emotion_id'][j] != q['emotion_id'][i]]
        if not candidates:
            raise ValueError('No independent reference for ' + str(q['clip_id'][i]))
        refs.append(rng.choice(cross or candidates))
    return refs


@torch.no_grad()
def build_base_cache(q, base, ids, device):
    cached = {}
    for start in range(0, len(ids), 16):
        use = ids[start:start + 16]
        b = q.batch(torch.tensor(use), device, keys=('content', 'valid'))
        result = base_forward(base, b['content'], b['valid'])
        for j, i in enumerate(use):
            n = int(q['_lengths'][i])
            cached[i] = {k: result[k][j, :n].cpu() for k in ('b0', 'h0')}
    return cached


def attach_base(b, ids, cache):
    width = b['valid'].shape[1]
    for key in ('b0', 'h0'):
        chunks = [cache[int(i)][key] for i in ids]
        b[key] = torch.stack([F.pad(x, (0, 0, 0, width - len(x))) for x in chunks]).to(b['valid'].device)
    return b


def make_refs(q, ids, device):
    return q.batch(torch.tensor(ids), device,
                   keys=('motion', 'valid', 'channel_mask', 'clip_id', 'sentence_id'))


def motion_targets(b):
    # The weak parser scans source-clock runs. CPU parsing avoids a GPU
    # synchronization for every frame while returning the same native labels.
    args = [b[k].detach().cpu() for k in ('motion', 'b0', 'valid', 'channel_mask', 'times')]
    return {k: v.to(b['motion'].device) for k, v in teacher_targets(*args).items()}


def pair_batches(q, ids, batch_size, rng):
    groups = {}
    for i in ids:
        groups.setdefault(int(q['speaker_id'][i]), []).append(i)
    units = []
    for values in groups.values():
        values = list(values); rng.shuffle(values)
        units.extend(values[j:j + 2] for j in range(0, len(values), 2))
    rng.shuffle(units)
    order = [i for unit in units for i in unit]
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


@torch.no_grad()
def fit_geometry_scale(q, ids, cache, geometry_jacobian, geometry_offset, device):
    """Fit observer scales using TRAIN motion only (robust RMS)."""
    jac = geometry_jacobian.to(device)
    offset = geometry_offset.to(device)
    dependency = jac.abs().gt(1e-7)
    sums = torch.zeros(4, dtype=torch.float64)
    squares = sums.clone(); counts = sums.clone()
    for start in range(0, len(ids), 16):
        use = ids[start:start + 16]
        b = attach_base(q.batch(torch.tensor(use), device,
                                keys=('motion', 'valid', 'channel_mask')), use, cache)
        del b['b0'], b['h0']
        obs = b['valid'][..., None] & b['channel_mask'][:, None]
        geom = torch.einsum('btc,gc->btg', torch.where(obs, b['motion'], 0.), jac) + offset
        complete = (b['channel_mask'][:, None] | ~dependency[None]).all(-1)
        mask = b['valid'][..., None] & complete[:, None]
        x = torch.where(mask, geom, 0.).double().cpu()
        sums += x.sum((0, 1)); squares += x.square().sum((0, 1)); counts += mask.sum((0, 1)).double().cpu()
    if not counts.any():
        return torch.ones(4)
    mean = sums / counts.clamp_min(1)
    scale = (squares / counts.clamp_min(1) - mean.square()).clamp_min(0).sqrt().clamp_min(.05)
    return scale.float()


@torch.no_grad()
def fit_training_statistics(q, ids, cache, device):
    """Fit affect-feature and residual scales on selected TRAIN clips only."""
    fsum = torch.zeros(772, dtype=torch.float64); fsq = fsum.clone(); fcount = 0
    esum = torch.zeros(52, dtype=torch.float64); esq = esum.clone(); ecount = torch.zeros(52, dtype=torch.float64)
    for start in range(0, len(ids), 16):
        use = ids[start:start + 16]
        b = attach_base(q.batch(torch.tensor(use), device,
                                keys=('audio_features', 'motion', 'valid', 'channel_mask')), use, cache)
        features = b['audio_features'][..., 768:][b['valid']].double().cpu()
        fsum += features.sum(0); fsq += features.square().sum(0); fcount += len(features)
        obs = b['valid'][..., None] & b['channel_mask'][:, None]
        residual = torch.where(obs, b['motion'] - b['b0'], 0.).double().cpu()
        obs_cpu = obs.cpu()
        esum += residual.sum((0, 1)); esq += residual.square().sum((0, 1))
        ecount += obs_cpu.sum((0, 1)).double()
        del b
    if fcount < 1:
        raise ValueError('No valid TRAIN audio frames')
    mean = fsum / fcount
    std = (fsq / fcount - mean.square()).clamp_min(0).sqrt().clamp_min(.001)
    scales = (esq / ecount.clamp_min(1)).clamp_min(0).sqrt().clamp_min(.02)
    return mean.float(), std.float(), scales.float()


@torch.no_grad()
def evaluate(model, q, ids, cache, data, device, seed, channels):
    model.eval(); rng = random.Random(seed); totals = {}; count = 0
    pool = {}
    for i in ids:
        pool.setdefault(int(q['speaker_id'][i]), []).append(i)
    for start in range(0, len(ids), 8):
        use = ids[start:start + 8]
        b = attach_base(q.batch(torch.tensor(use), device), use, cache)
        enrollment = [data['refs'][int(q['speaker_id'][i])] for i in use]
        refs = {k: torch.stack([r[k][0] for r in enrollment]).to(device)
                for k in ('motion', 'valid', 'channel_mask')}
        if any(q['clip_id'][i] == r['clip_id'][0] for i, r in zip(use, enrollment)):
            raise ValueError('Query overlaps independent enrollment')
        targets = motion_targets(b)
        observed = model.training_forward(b, refs, targets)
        rollout = model.predict(b['audio_features'], b['h0'], b['b0'], b['valid'], refs,
                                times=b['times'], seed=seed + start)
        for name, out in (('observed', observed), ('rollout', rollout)):
            _, parts = motion_objective(out['motion'], b['motion'], b['valid'],
                                        b['channel_mask'], b['times'], model.scales)
            for key, value in parts.items():
                totals[name + '_' + key] = totals.get(name + '_' + key, 0.) + float(value) * len(use)
        known = b['emotion_id'].ge(0)
        correct = rollout['student']['emotion_logits'].argmax(-1).eq(b['emotion_id']) & known
        totals['emotion_correct'] = totals.get('emotion_correct', 0) + int(correct.sum())
        totals['emotion_known'] = totals.get('emotion_known', 0) + int(known.sum())
        for name, events in (('teacher', targets['events']), ('rollout', rollout['events'])):
            active = events.reshape(len(use), -1, 4, 3)[..., 0].bool()
            totals[name + '_active'] = totals.get(name + '_active', 0) + int((active & b['valid'][..., None]).sum())
        totals['valid_group_frames'] = totals.get('valid_group_frames', 0) + int(b['valid'].sum()) * 4
        count += len(use)
    result = {k: v / count for k, v in totals.items()
              if k.endswith('_position') or k.endswith('_velocity')}
    result.update(clips=count,
                  emotion_accuracy=totals['emotion_correct'] / max(totals['emotion_known'], 1),
                  teacher_event_occupancy=totals['teacher_active'] / max(totals['valid_group_frames'], 1),
                  rollout_event_occupancy=totals['rollout_active'] / max(totals['valid_group_frames'], 1))
    model.train(); return result


def objective(model, b, refs, targets, *, warmup, channels):
    out = model.training_forward(b, refs, targets, teacher_state=warmup)
    motion, parts = motion_objective(out['motion'], b['motion'], b['valid'],
                                     b['channel_mask'], b['times'], model.scales)
    supervision = v4_losses(model, out, targets, b, teacher_warmup=warmup)
    query_style = model.style(b['motion'], b['valid'], b['channel_mask'], model.scales)
    relation = relation_style_loss(torch.cat((out['style'], query_style)),
                                   b['speaker_id'].repeat(2))
    consistency = F.smooth_l1_loss(out['style'], query_style)
    # Counterfactual execution is kept weak until the event parser has enough
    # observed events. It trains the sampled path without treating one sampled
    # performance as the unique framewise target.
    execution = event_execution_loss(
        out['sampled_motion'], out['zero_event_motion'], out['sampled_schedule']['events'],
        out['conditioned']['state'].detach(), b['b0'].detach(), b['valid'], b['times'],
        b['channel_mask'], channels, quality_reference_motion=out['zero_event_motion'].detach())['total']
    loss = motion + supervision['total'] + .05 * relation + .05 * consistency + .05 * execution
    return loss, out, {'motion': float(motion), 'position': float(parts['position']),
                       'teacher_anchor': float(supervision['teacher_anchor']),
                       'student_state': float(supervision['state_direct']),
                       'hazard': float(supervision['hazard']),
                       'execution': float(execution), 'relation': float(relation)}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def train(a):
    configure(a.seed); device = torch.device(a.device)
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    if not a.resume and ((root / 'protocol.json').exists() or (root / 'final.pt').exists()):
        raise FileExistsError('Existing v4 run: choose a fresh output or explicit resume')
    binding = json.loads(Path(a.binding).read_text())
    data, base = load_runtime(binding, device)
    frozen = state_digest(base.stage1)
    q = data['splits']['train']; fold = development_fold(q, data['fit_sids'])
    fit = select_balanced(q, fold['train'], a.limit, a.seed)
    if a.smoke and len(fit) > a.smoke_clips:
        fit = select_balanced(q, fold['train'], a.smoke_clips, a.seed)
    held = select_balanced(q, fold['speaker_dev'] + fold['sentence_dev'], a.eval_limit, a.seed)
    pool = {}
    for i in fold['train']:
        pool.setdefault(int(q['speaker_id'][i]), []).append(i)
    rng = random.Random(a.seed)
    ref_by_query = {i: reference_ids(q, [i], pool, rng)[0] for i in fit}
    required = sorted(set(fit + held + list(ref_by_query.values())))
    print(json.dumps({'event': 'cache_start', 'clips': len(required)}), flush=True)
    cache = build_base_cache(q, base, required, device)

    geometry_path = Path(a.geometry_asset)
    gj, go, _, geometry_channels, geometry_meta = load_rig_geometry(geometry_path)
    geometry_scale = fit_geometry_scale(q, fit, cache, gj, go, device)
    content_dim = cache[fit[0]]['h0'].shape[-1]
    cfg = KineTalkV4Config(content_dim=content_dim, hidden=a.hidden,
                           decoder_hidden=a.decoder_hidden,
                           use_geometry=not a.no_geometry,
                           use_interactions=not a.no_interactions,
                           use_teacher=not a.no_teacher,
                           use_style=not a.no_style)
    feature_mean, feature_std, target_scales = fit_training_statistics(q, fit, cache, device)
    geometry_meta = dict(geometry_meta, scale_is_train_fitted=True)
    model = KineTalkV4Model(cfg, feature_mean, feature_std,
                             target_scales, gj, go, geometry_scale).to(device)
    protocol = {
        'schema': 'kinetalk_v4_training_v1', 'seed': a.seed,
        'epochs': a.epochs, 'fit_clips': len(fit), 'internal_dev_clips': len(held),
        'binding_sha256': file_sha(a.binding), 'neutral_digest': frozen,
        'geometry_asset_sha256': file_sha(geometry_path),
        'geometry_scale_train_fitted': geometry_scale.tolist(),
        'geometry_metadata': geometry_meta, 'geometry_channels': list(geometry_channels),
        'audio_input': '1540D: first 768D frozen content/B0, last 772D affect student',
        'teacher': 'privileged motion teacher; deployment uses audio student',
        'event_objective': 'observed likelihood plus weak sampled counterfactual execution',
        'text_enabled': False, 'hdtf_enabled': False, 'test_loaded': False,
        'phase53_untouched': True, 'source_sha256': file_sha(__file__),
        'flags': {k: v for k, v in vars(a).items() if k != 'resume'},
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    start_epoch = 0; step = 0; history = []; order_rng = random.Random(a.seed)
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        if ck['protocol'] != protocol:
            raise ValueError('Resume protocol differs')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        start_epoch, step, history = ck['epoch'], ck['step'], ck['history']
        order_rng.setstate(ck['python_rng']); torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    write(root / 'protocol.json', protocol)
    write(root / 'fold.json', {'fit_ids': fit, 'held_ids': held, 'reference_by_query': ref_by_query})
    initial = None
    run_epochs = 1 if a.smoke else a.epochs
    for epoch in range(start_epoch, run_epochs):
        warm = epoch < a.warmup_epochs
        model.train(); begin = time.monotonic(); sums = {}; batches = pair_batches(q, fit, a.batch_size, order_rng)
        if a.smoke and a.smoke_steps > len(batches):
            batches = [batches[i % len(batches)] for i in range(a.smoke_steps)]
        for index, ids in enumerate(batches):
            if a.smoke:
                warm = index < a.smoke_steps // 2 and cfg.use_teacher
            elif not cfg.use_teacher:
                warm = False
            b = attach_base(q.batch(torch.tensor(ids), device), ids, cache)
            refs = make_refs(q, [ref_by_query[i] for i in ids], device)
            if any(q['sentence_id'][i] == q['sentence_id'][j] for i, j in zip(ids, [ref_by_query[i] for i in ids])):
                raise ValueError('Reference sentence overlap')
            targets = motion_targets(b)
            optimizer.zero_grad(set_to_none=True)
            loss, out, logs = objective(model, b, refs, targets, warmup=warm,
                                       channels=geometry_channels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite v4 loss')
            loss.backward()
            if not warm and (model.student.state.weight.grad is None or
                             model.student.state.weight.grad.abs().sum() == 0):
                raise RuntimeError('Student gradient missing')
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); step += 1
            logs.update(loss=float(loss), grad_norm=float(grad))
            for key, value in logs.items():
                if isinstance(value, (int, float)): sums[key] = sums.get(key, 0.) + value
            if initial is None: initial = float(loss)
            if index % 20 == 0:
                state = {'status': 'training', 'epoch': epoch + 1, 'step': step,
                         'batch': index + 1, 'batches': len(batches),
                         'teacher_warmup': warm, **logs}
                write(root / 'state.json', state); print(json.dumps(state), flush=True)
        dev = evaluate(model, q, held, cache, data, device, a.seed, geometry_channels)
        if state_digest(base.stage1) != frozen:
            raise RuntimeError('Frozen neutral B0 changed')
        entry = {'epoch': epoch + 1, 'step': step, 'seconds': time.monotonic() - begin,
                 'train': {k: v / len(batches) for k, v in sums.items()},
                 'internal_dev': dev}
        history.append(entry); write(root / 'history.json', history)
        ck = {'config': asdict(cfg), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
              'protocol': protocol, 'epoch': epoch + 1, 'step': step, 'history': history,
              'python_rng': order_rng.getstate(), 'torch_rng': torch.get_rng_state(),
              'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        save(root / 'resume.pt', ck)
        if epoch == run_epochs - 1:
            save(root / 'final.pt', {k: v for k, v in ck.items() if k != 'optimizer'})
        write(root / 'state.json', {'status': 'epoch_complete', **entry})
        print(json.dumps({'event': 'epoch_complete', **entry}), flush=True)
    write(root / 'complete.json', {'status': 'complete', 'epochs': run_epochs,
        'updates': step, 'initial_loss': initial, 'final_loss': history[-1]['train']['loss'],
        'neutral_frozen_sha256': frozen, 'protocol': protocol,
        'final_internal_dev': history[-1]['internal_dev'], 'paper_result': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True); p.add_argument('--geometry-asset', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=47); p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--warmup-epochs', type=int, default=1); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--eval-limit', type=int, default=256); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--decoder-hidden', type=int, default=192); p.add_argument('--resume')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--smoke-clips', type=int, default=32)
    p.add_argument('--smoke-steps', type=int, default=120)
    p.add_argument('--no-geometry', action='store_true'); p.add_argument('--no-interactions', action='store_true')
    p.add_argument('--no-teacher', action='store_true'); p.add_argument('--no-style', action='store_true')
    train(p.parse_args())
