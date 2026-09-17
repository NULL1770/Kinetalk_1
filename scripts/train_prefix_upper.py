"""Explicit native-clock past-motion tokens for conditional flow continuation.

Pilot fits fixed metadata-selected training clips only. Full training is a
separate, fresh run: identical no-prefix/scheduled-prefix arms with generated
past at deployment. No current/future target is used as an inference condition.
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
from scripts import train_history_upper as h
from scripts.audit_temporal_repair import metadata_equal, summarize, paired_metrics, GROUPS
from scripts.train_centered_temporal_prior import temporal_stats, time_intervention
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, capture_rng, canonical_hash
from scripts.train_predictable_renderer import state_hash
from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from kinetalk_b0.models.slow_state_affect import compose_upper_face

SCHEMA = 'explicit_prefix_upper_v1'
CHUNK, HISTORY, CC = h.CHUNK, h.HISTORY, h.CC


def window_batch(b, identity, local, source, ids, starts, *, empty=False, reverse=False):
    """Fixed H+CHUNK clock. Missing history is invalid, never time-compressed.

    `source` is only indexed at strictly preceding positions. Current/future
    source values are never gathered, even as temporary known placeholders.
    """
    if starts.shape != ids.shape or starts.dtype != torch.long or (starts < 0).any():
        raise ValueError('Nonnegative starts must match row indices')
    frames = b['valid'].shape[1]
    clock = starts[:, None] + torch.arange(-HISTORY, CHUNK, device=starts.device)[None]
    exists = (clock >= 0) & (clock < frames)
    index = clock.clamp(0, frames-1)
    valid = b['valid'][ids[:, None], index] & exists
    past = torch.arange(HISTORY+CHUNK, device=starts.device)[None] < HISTORY
    known_mask = valid & past
    if empty:
        valid = valid & ~past; known_mask = torch.zeros_like(valid)
    known = local.new_zeros(len(ids), HISTORY+CHUNK, 9)
    if not empty:
        prefix_clock = index[:, :HISTORY]
        prefix_valid = known_mask[:, :HISTORY]
        prefix = local.new_zeros(len(ids), HISTORY, 9)
        rows, cols = prefix_valid.nonzero(as_tuple=True)
        prefix[rows, cols] = source[ids[rows], prefix_clock[rows, cols]]
        if reverse:
            prefix = prefix.clone()
            for row in range(len(ids)):
                good = prefix_valid[row].nonzero(as_tuple=True)[0]
                prefix[row, good] = prefix[row, good.flip(0)]
        known[:, :HISTORY] = prefix
    content = torch.where(valid[..., None], b['h0'][ids[:, None], index], 0.)
    native = torch.where(valid[..., None], local[ids[:, None], index], 0.)
    conditions = (valid, content, identity['code'][ids],
        {'global': b['audio_global'][ids], 'intensity_value': b['audio_intensity'][ids]}, native)
    return conditions, known, known_mask, index


@torch.no_grad()
def rollout(upper, b, identity, local, noise, *, steps=12, mode='full', oracle_target=None, use_prefix=True):
    if mode not in ('full', 'empty', 'reverse_history', 'static', 'reverse', 'oracle_history'):
        raise ValueError('Unknown prefix rollout mode')
    if mode == 'oracle_history' and oracle_target is None:
        raise ValueError('Oracle requires explicit past target')
    if mode != 'oracle_history' and oracle_target is not None:
        raise ValueError('Deployable modes reject target history')
    native, content = time_intervention(local, b['h0'], b['valid'], mode)
    condition_batch = {**b, 'h0': content}; output = torch.zeros_like(noise)
    for start in range(0, noise.shape[1], CHUNK):
        stop = min(start+CHUNK, noise.shape[1])
        ids = b['valid'][:, start:stop].any(1).nonzero(as_tuple=True)[0]
        if not len(ids): continue
        starts = torch.full_like(ids, start)
        source = oracle_target if mode == 'oracle_history' else output
        conditions, known, known_mask, _ = window_batch(condition_batch, identity, native, source, ids, starts,
            empty=not use_prefix or mode == 'empty', reverse=mode == 'reverse_history')
        initial = known.new_zeros(known.shape)
        initial[:, HISTORY:HISTORY+stop-start] = noise[ids, start:stop]
        result = upper.decode_prefix(*conditions, initial, known=known, known_mask=known_mask, steps=steps)
        if not torch.equal(result[known_mask], known[known_mask]):
            raise RuntimeError('Known prefix changed during decoding')
        output[ids, start:stop] = result[:, HISTORY:HISTORY+stop-start]
    return output


def patch_loss(upper, b, identity, local, target, source, ids, starts, noise, times, *, empty=False):
    conditions, known, known_mask, index = window_batch(b, identity, local, source.detach(), ids, starts, empty=empty)
    unknown = conditions[0] & ~known_mask
    supervision = torch.where(unknown[..., None], target[ids[:, None], index], 0.)
    return upper.flow_loss_prefix(supervision, *conditions, noise, times, known=known, known_mask=known_mask)


@torch.no_grad()
def evaluate(upper, b, identity, local, scales, *, steps, use_prefix, baseline=None, system=None):
    predictions = {}; records = {}; target = b['motion']; valid = b['valid']
    normalized = h.normalized_target(b, scales)
    for seed in r.SEEDS:
        noise = torch.randn(len(valid), valid.shape[1], 9, generator=torch.Generator().manual_seed(seed)).to(local)
        for mode in ['full']+(['empty', 'reverse_history', 'static', 'reverse', 'oracle_history'] if seed == 42 else []):
            kwargs = {'oracle_target': normalized} if mode == 'oracle_history' else {}
            upper_motion = rollout(upper, b, identity, local, noise, steps=steps, mode=mode, use_prefix=use_prefix, **kwargs)
            values = b['static_upper'][:, None] + upper_motion*scales
            # Pilot copied channels use zero placeholders, not GT. They are not
            # scored. Formal uses the exact previous Stage3 base for every seed.
            base = torch.zeros_like(target) if baseline is None else baseline[f'{seed}/base'].to(local)
            pred = compose_upper_face(base, values, valid)
            if not torch.equal(pred[..., list(r.NOT_UPPER)], base[..., list(r.NOT_UPPER)]) or not torch.equal(pred[~valid], base[~valid]):
                raise RuntimeError('Frozen baseline protection failed')
            key = f'{seed}/{mode}'; predictions[key] = pred.cpu()
            metrics = {}
            for name in ('brows', 'eyes_expression'):
                cc = list(GROUPS[name]); mask = valid[..., None] & b['channel_mask'][:, None, cc]
                metrics[name] = paired_metrics(pred[..., cc], target[..., cc], mask)
            record = {'metrics': metrics, 'temporal': temporal_stats(pred, target, valid),
                'boundaries': h.boundary_report(pred, target, valid), 'oracle_target_history': mode == 'oracle_history'}
            if system is not None:
                readout = system.encode_motion(torch.where(r.obs(b), pred-b['b0']-identity['baseline'][:, None], 0.), valid)
                record['motion_teacher_emotion_accuracy_NONINDEPENDENT'] = float((readout['emotion_logits'].argmax(-1)==b['emotion_id']).float().mean())
            records[key] = record
    curves = {key: b[key].cpu() if torch.is_tensor(b[key]) else b[key] for key in ('clip_id', 'valid', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    curves.update(schema=SCHEMA, target=target.cpu(), predictions=predictions, noise_seeds=list(r.SEEDS), decode_steps=steps)
    report = {'clips': len(valid), 'modes': records, 'nonupper_scored': baseline is not None, 'test_loaded': False}
    if baseline is not None: report['distribution'] = summarize(curves, curves['emotion_id'])
    return report, curves


def load_context(args):
    data = r.load_training_inputs(args.source_run, args.audio, args.targets, args.enrollment, args.native_root)
    checkpoints, bindings, steps, stride = r.matching_checkpoints(args.trained_run, data)
    system = data['system'].to(args.device).eval().requires_grad_(False)
    system.load_state_dict(checkpoints['teacher']['system'])
    audio = r._make_audio(checkpoints['audio']['audio'], stride, args.device)
    sourcepath = args.history_run/'no_history/final.pt'
    complete = json.loads(sourcepath.with_name('complete.json').read_text())
    source_recipe = json.loads(sourcepath.with_name('provenance.json').read_text())['recipe']
    if r.sha(sourcepath) != complete['final_sha256'] or source_recipe['source_bindings'] != bindings:
        raise ValueError('History source binding differs')
    source = torch.load(sourcepath, map_location='cpu', weights_only=False)
    if (source['schema'] != h.SCHEMA or source['arm'] != 'no_history' or source_recipe['smoke'] or
        source['recipe_sha256'] != canonical_hash(source_recipe) or source_recipe['config'] != data['config']):
        raise ValueError('History warmstart recipe differs')
    selection = None
    if args.pilot:
        selection = json.loads(args.pilot_selection.read_text(encoding='utf8'))
        ids = [row['clip_id'] for row in selection['clips']]
        q = data['splits']['train']
        if len(ids) != 8 or len(set(ids)) != 8 or not set(ids) <= set(q['clip_id']):
            raise ValueError('Pilot must be eight explicit current-fit clips')
        data['splits'] = {'train': r.subset(q, torch.tensor([q['clip_id'].index(cid) for cid in ids]), 'cpu')}
        selected_sids = set(map(int, data['splits']['train']['speaker_id']))
        data['refs'] = {sid: value for sid, value in data['refs'].items() if sid in selected_sids}
    for q in data['splits'].values():
        if q['valid'].shape[1] != 96 or not torch.allclose(q['times'][:, 1:]-q['times'][:, :-1], torch.full_like(q['times'][:, 1:], .04), atol=1e-7, rtol=1e-5):
            raise ValueError('Locked 96 frame native clock required')
    r.cache_current_base(system, data, args.device, args.batch_size)
    identities = r.identity_cache(system, data, args.device)
    h.cache_audio(system, audio, data, args.device, args.batch_size)
    local = copy.deepcopy(audio).eval().requires_grad_(False); local.load_state_dict(source['local'])
    for q in data['splits'].values():
        q['prefix_local'] = torch.cat([local(r.subset(q, ix, args.device)['audio_features'], r.subset(q, ix, args.device)['valid'])['local'].detach().cpu()
            for ix in torch.arange(len(q['valid'])).split(args.batch_size)])
    frozen = {'system': state_hash(system.state_dict()), 'audio': state_hash(audio.state_dict()), 'local': state_hash(local.state_dict())}
    if {key: frozen[key] for key in ('system', 'audio')} != source_recipe['frozen']:
        raise ValueError('Frozen source hashes differ')
    return data, system, audio, local, identities, source, source_recipe, r.sha(sourcepath), steps, selection, frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'audio', 'targets', 'enrollment', 'native-root', 'trained-run', 'history-run', 'centered-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pilot', action='store_true'); p.add_argument('--pilot-selection', type=Path)
    p.add_argument('--epochs', type=int, default=12); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=79)
    args = p.parse_args(); started = time.monotonic(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.output.exists(): raise FileExistsError('Fresh prefix run required')
    if not 1 <= args.epochs <= 15 or args.batch_size < 1 or (args.pilot and args.pilot_selection is None):
        raise ValueError('Explicit pilot selection and budget <=15 required')
    if not args.pilot:
        raise ValueError('Formal training disabled until pilot receiver gate is passed and evaluated')
    data, system, audio, local, identities, source, source_recipe, source_sha, steps, selection, frozen = load_context(args)
    scales = source['scales'].to(args.device); train = data['splits']['train']
    arms = ('no_prefix', 'teacher_prefix') if args.pilot else ('no_prefix', 'scheduled_prefix')
    bases = None
    if not args.pilot:
        path = args.centered_run/'white/curves.pt'; complete = json.loads(path.with_name('complete.json').read_text())
        if r.sha(path) != complete['curves_sha256'] or r.sha(path) != source_recipe['baseline_curves_sha256']:
            raise ValueError('Frozen baseline differs')
        bases = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        metadata_equal({**data['splits']['validation'], 'target': data['splits']['validation']['motion']}, bases)
    args.output.mkdir(parents=True)
    matches = {}; reports = {}
    for arm in arms:
        torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
        upper = PrefixUpperFlow(data['config']).to(args.device).eval()
        compatible = {key: value for key, value in source['upper'].items() if key in upper.state_dict()}
        incompatible = upper.load_state_dict(compatible, strict=False)
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != {'known_embedding.weight'}:
            raise ValueError('Unexpected prefix warmstart missing tensors')
        output = args.output/arm; output.mkdir()
        initial = state_hash(upper.state_dict()); save_checkpoint(output/'initial.pt', {'upper': upper.state_dict()})
        params = list(upper.parameters()); optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
        recipe = {'schema': SCHEMA, 'arm': arm, 'pilot': args.pilot, 'epochs': args.epochs, 'batch_size': args.batch_size,
            'seed': args.seed, 'chunk': CHUNK, 'history': HISTORY, 'decode_steps': steps, 'config': data['config'],
            'source_sha256': source_sha, 'source_recipe_sha256': canonical_hash(source_recipe),
            'scales': scales.cpu().tolist(), 'initial': initial, 'frozen': frozen, 'fit_ids': train['clip_id'],
            'pilot_selection': selection, 'pilot_selection_sha256': r.sha(args.pilot_selection) if args.pilot else None,
            'data_provenance': data['provenance'], 'fixed_final_epoch': True, 'test_loaded': False, 'default_replaced': False,
            'objective': 'Unknown-only flow matching, exact clean prefix clamp in joint motion-token attention; no seam loss or smoothing',
            'local_frozen': True, 'teacher_probability': [1. if args.pilot else h.teacher_probability(e, args.epochs) for e in range(args.epochs)],
            'code_sha256': {name: r.sha(Path(__file__).resolve().parents[1]/name) for name in (
                'scripts/train_prefix_upper.py', 'kinetalk_b0/models/prefix_upper_flow.py', 'kinetalk_b0/models/temporal_upper.py',
                'scripts/train_history_upper.py', 'docs/PREFIX_CONTINUATION_PROTOCOL_20260917.md')}}
        digest = canonical_hash(recipe); save_json(output/'provenance.json', {'recipe': recipe, 'recipe_sha256': digest})
        generator = torch.Generator().manual_seed(args.seed); draws = []; total_steps = 0
        if args.pilot:
            b = r.subset(train, torch.arange(len(train['valid'])), args.device); ident = r.batch_identity(identities, b)
            native = b['prefix_local']; target = h.normalized_target(b, scales)
            pairs = [(i, start) for i in range(len(b['valid'])) for start in range(0, 96, CHUNK) if b['valid'][i, start:start+CHUNK].any()]
            initial_report, _ = evaluate(upper, b, ident, native, scales, steps=steps, use_prefix=arm != 'no_prefix')
            save_json(output/'initial_evaluation.json', initial_report)
        for epoch in range(args.epochs):
            t0 = time.monotonic(); losses = []; draw = hashlib.sha256(); selected_count = 0
            probability = recipe['teacher_probability'][epoch]
            order = torch.randperm(len(pairs) if args.pilot else len(train['valid']), generator=generator)
            for selected in order.split(args.batch_size):
                if args.pilot:
                    pair_tensor = torch.tensor([pairs[i] for i in selected], device=args.device)
                    ids, starts = pair_tensor[:, 0], pair_tensor[:, 1]
                    noise = torch.randn(len(ids), HISTORY+CHUNK, 9, generator=generator)
                    times = torch.rand(len(ids), generator=generator)
                    for tensor in (selected, noise, times): draw.update(tensor.numpy().tobytes())
                    loss = patch_loss(upper, b, ident, native, target, target, ids, starts, noise.to(args.device), times.to(args.device), empty=arm=='no_prefix')
                else:
                    b = r.subset(train, selected, args.device); ident = r.batch_identity(identities, b); native = b['prefix_local']
                    target = h.normalized_target(b, scales)
                    noise = torch.randn(len(selected), 6, HISTORY+CHUNK, 9, generator=generator)
                    synthetic_noise = torch.randn(target.shape, generator=generator)
                    times = torch.rand(len(selected), 6, generator=generator); decisions = torch.rand(len(selected), 6, generator=generator)
                    for tensor in (selected, noise, synthetic_noise, times, decisions): draw.update(tensor.numpy().tobytes())
                    generated = rollout(upper, b, ident, native, synthetic_noise.to(args.device), steps=steps, use_prefix=arm!='no_prefix')
                    loss = target.new_zeros(())
                    for ci, start in enumerate(range(0, 96, CHUNK)):
                        ids = b['valid'][:, start:start+CHUNK].any(1).nonzero(as_tuple=True)[0]
                        if not len(ids): continue
                        choose = decisions[:, ci].to(args.device) < probability
                        source_history = torch.where(choose[:, None, None], target, generated)
                        selected_count += int(choose[ids].sum()) if start else 0
                        part = patch_loss(upper, b, ident, native, target, source_history, ids, torch.full_like(ids, start),
                            noise[ids.cpu(), ci].to(args.device), times[ids.cpu(), ci].to(args.device), empty=arm=='no_prefix')
                        loss = loss+part*b['valid'][ids, start:start+CHUNK].sum()
                    loss = loss/b['valid'].sum()
                r.optimize(loss, optimizer, params); losses.append(float(loss.detach())); total_steps += 1
            draws.append(draw.hexdigest())
            row = {'arm': arm, 'epoch': epoch+1, 'loss': sum(losses)/len(losses), 'seconds': time.monotonic()-t0,
                'total_steps': total_steps, 'draw_sha256': draw.hexdigest(), 'teacher_probability': probability,
                'teacher_draw_count': selected_count, 'no_prefix_ignores_history': arm=='no_prefix'}
            save_json(output/f'epoch{epoch+1:03d}.json', row); save_json(args.output/'status.json', {'status': 'training', **row})
            payload = {'schema': SCHEMA, 'arm': arm, 'upper': upper.state_dict(), 'scales': scales.cpu(),
                'source_sha256': source_sha, 'recipe_sha256': digest, 'completed_epochs': epoch+1, 'total_steps': total_steps}
            save_checkpoint(output/'last.pt', {**payload, 'optimizer': optimizer.state_dict(), 'rng': capture_rng(generator)})
            print(json.dumps(row), flush=True)
        for name, module in [('system', system), ('audio', audio), ('local', local)]:
            if state_hash(module.state_dict()) != frozen[name] or any(param.grad is not None for param in module.parameters()):
                raise RuntimeError('Frozen module changed: '+name)
        if args.pilot:
            report, curves = evaluate(upper, b, ident, native, scales, steps=steps, use_prefix=arm!='no_prefix')
        else:
            # Evaluate in batches to keep GPU memory bounded; noise is sliced
            # from globally seeded streams by evaluate_full below.
            report, curves = evaluate_full(upper, system, data['splits']['validation'], identities, scales, bases, args, steps, arm)
        report.update(schema=SCHEMA, pilot=args.pilot, evaluation_role='training_reconstruction' if args.pilot else 'internal_development',
            default_replaced=False, deploy_history='Generated preceding chunks only', local_frozen=True)
        reports[arm] = report; save_json(output/'evaluation.json', report); save_checkpoint(output/'curves.pt', curves)
        save_checkpoint(output/'final.pt', payload)
        save_json(output/'complete.json', {'final_sha256': r.sha(output/'final.pt'), 'curves_sha256': r.sha(output/'curves.pt'),
            'completed_epochs': args.epochs, 'recipe_sha256': digest, 'frozen': frozen})
        matches[arm] = {'initial': initial, 'draws': draws}
    if matches[arms[0]] != matches[arms[1]]: raise RuntimeError('Paired initialization/random streams differ')
    save_json(args.output/'matched_audit.json', {'equal': True, 'arms': matches})
    if args.pilot:
        checks = {}
        for region in ('brows', 'eyes_expression'):
            base_record = reports['no_prefix']['modes']['42/full']
            oracle_record = reports['teacher_prefix']['modes']['42/oracle_history']
            base_boundary = base_record['boundaries'][region]['chunk_boundary']
            oracle_boundary = oracle_record['boundaries'][region]['chunk_boundary']
            checks[region] = {
                'oracle_boundary_rms_ratio': oracle_boundary['pred_rms']/max(oracle_boundary['gt_rms'], 1e-12),
                'boundary_error_ratio_to_no_prefix': oracle_boundary['displacement_mse']/max(base_boundary['displacement_mse'], 1e-12),
                'centered_mse_not_worse': oracle_record['metrics'][region]['centered_mse'] <= base_record['metrics'][region]['centered_mse']}
        passed = all(row['oracle_boundary_rms_ratio'] <= 2 and row['boundary_error_ratio_to_no_prefix'] <= .5
            and row['centered_mse_not_worse'] for row in checks.values())
        save_json(args.output/'receiver_gate.json', {'passed': passed, 'checks': checks,
            'predeclared_protocol': recipe['code_sha256']['docs/PREFIX_CONTINUATION_PROTOCOL_20260917.md'],
            'scope': 'Eight fit reconstruction clips. Passing is not audio prediction or generalization success.'})
    save_json(args.output/'status.json', {'status': 'complete', 'pilot': args.pilot, 'epochs': args.epochs, 'seconds': time.monotonic()-started})
    print('PREFIX_UPPER_COMPLETE', flush=True)


def evaluate_full(*args, **kwargs):
    raise NotImplementedError('Formal continuation requires successful receiver pilot and a reviewed evaluator')


if __name__ == '__main__': main()
