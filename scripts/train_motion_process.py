"""Isolated variable-duration motion teacher and paired acoustic-prior pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.full_native_context_data import NativeContextStore, sha
from scripts.full_staged_data import _renderer_cache_binding
from scripts.motion_process_representation import (read_group_states, fit_dynamic_scales,
    fit_motion_process, render_motion_process)
from kinetalk_b0.models.motion_process_prior import MotionProcessPrior, joint_event_nll, initial_nll

SCHEMA = 'persistent_motion_process_pilot_v1'
DURATIONS = (4, 8, 12, 20, 32, 48, 64)
SEEDS = (42, 123, 2026, 77, 91, 301, 509, 997)
GROUPS = ('brow_up', 'brow_down', 'eye_squint', 'eye_wide')
CELLS = ('fit', 'sentence', 'speaker', 'joint')


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def fixed_split(speakers, sentences):
    if len(speakers) != len(sentences) or not len(speakers):
        raise ValueError('Matching nonempty speaker/sentence metadata required')
    def held(values, kind, count):
        return set(sorted(set(map(str, values)), key=lambda x: hashlib.sha256(
            f'motion_process_20260918:{kind}:{x}'.encode()).hexdigest())[:count])
    hs, ht = held(speakers, 'speaker', 4), held(sentences, 'sentence', 13)
    result = {key: [] for key in CELLS}
    for i, (speaker, sentence) in enumerate(zip(speakers, sentences)):
        key = ('joint' if str(sentence) in ht else 'speaker') if str(speaker) in hs else (
            'sentence' if str(sentence) in ht else 'fit')
        result[key].append(i)
    if any(not v for v in result.values()):
        raise ValueError('All four fixed split cells must be nonempty')
    return result


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist()))


def load_clips(args):
    archive = torch.load(args.audio, map_location='cpu', weights_only=False, mmap=True)
    cache_path = Path(archive['provenance']['renderer_cache'])
    if sha(cache_path) != _renderer_cache_binding(archive):
        raise ValueError('Renderer/audio binding mismatch')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
    q, audio = copy.copy(cache['splits']['train']['q']), archive['splits']['train']
    target = torch.load(args.targets, map_location='cpu', weights_only=False, mmap=True)
    if _renderer_cache_binding(target) != sha(cache_path):
        raise ValueError('Independent enrollment target binding mismatch')
    anchors = target['splits']['train']
    if q['clip_id'] != audio['clip_id'] or q['clip_id'] != anchors['clip_id']:
        raise ValueError('Train membership mismatch')
    for key in ('times', 'valid'):
        if not torch.equal(q[key], audio[key]):
            raise ValueError('Native audio/target clock mismatch')
    if list(q['sentence_id']) != list(audio['sentence_id']):
        raise ValueError('Audio sentence mismatch')
    q.update(content=audio['features'][..., :768], audio_features=audio['features'],
             anchors=anchors['anchors'], anchor_valid=anchors['anchor_valid'])
    store = NativeContextStore(q, args.native_root, args.native_manifest, args.delta_dir, role='train')
    split = fixed_split(q['speaker_id'].tolist(), q['sentence_id'])
    if args.smoke:
        split = {k: v[:8] for k, v in split.items()}
    selected = sorted(i for ids in split.values() for i in ids)
    mapping, clips = {}, []
    for i in selected:
        raw = store.clip(i)
        observed = raw['valid'][:, None] & raw['channel_mask'][None]
        state, groupmask = read_group_states(raw['motion'].numpy(), observed.numpy())
        if not np.array_equal(groupmask, np.repeat(raw['valid'].numpy()[:, None], 4, axis=1)):
            raise ValueError('Four fully observed groups required by pilot')
        neutral, nm = read_group_states(raw['anchors'][None].numpy(), raw['anchor_valid'][None].numpy())
        if not nm.all():
            raise ValueError('Independent neutral anchors unavailable')
        mapping[i] = len(clips)
        clips.append({'clip_id': q['clip_id'][i], 'sentence': str(q['sentence_id'][i]),
            'speaker': int(q['speaker_id'][i]), 'emotion': int(q['emotion_id'][i]),
            'features': raw['audio_features'].float(), 'valid': raw['valid'],
            'state': state, 'groupmask': groupmask, 'anchor': neutral[0],
            'upper': raw['motion'][:, [41,42,43,44,45,5,6,12,13]].numpy(),
            'native_sha256': raw['metadata']['artifact_sha256']})
        store._cache.pop(i, None)
        if len(clips) % 300 == 0:
            print('NATIVE_LOADED', len(clips), flush=True)
    split = {k: [mapping[i] for i in ids] for k, ids in split.items()}
    lineage = {'cache': sha(cache_path), 'audio': sha(args.audio), 'targets': sha(args.targets),
               'delta_manifest': sha(Path(args.delta_dir)/'manifest.json'), 'store': store.provenance}
    return clips, split, lineage


def fit_feature_stats(clips, ids):
    total, sq, count = torch.zeros(1540, dtype=torch.float64), torch.zeros(1540, dtype=torch.float64), 0
    for i in ids:
        x = clips[i]['features'][clips[i]['valid']].double()
        total += x.sum(0)
        sq += x.square().sum(0)
        count += len(x)
    mean = total / count
    return mean.float(), (sq/count-mean.square()).clamp_min(1e-6).sqrt().float()


def reconstruction_summary(clips, ids, key):
    numerator = np.zeros(4); target_e = np.zeros(4); pred_e = np.zeros(4); cross = np.zeros(4)
    segs = 0; framecount = 0; run_count = 0
    channel_error = np.zeros(9); channel_count = 0
    for i in ids:
        c = clips[i]; valid = c['valid'].numpy()
        y = center_runs(c['state'], valid)[valid]; p = center_runs(c[key], valid)[valid]
        numerator += ((p-y)**2).sum(0); target_e += (y*y).sum(0)
        pred_e += (p*p).sum(0); cross += (p*y).sum(0)
        segs += sum(sum(s['duration'] > 0 for s in group) for group in c['plan']['segments'])
        framecount += valid.sum()
        run_count += len(runs(valid))
        if 'upper' in c:
            lifted = center_runs(c[key][:, [1,1,0,0,0,2,3,2,3]], valid)[valid]
            truth = center_runs(c['upper'], valid)[valid]
            channel_error += ((lifted-truth)**2).sum(0); channel_count += valid.sum()
    return {'groups': {name: {'centered_r2': float(1-numerator[g]/max(target_e[g], 1e-15)),
        'correlation': float(cross[g]/max(math.sqrt(target_e[g]*pred_e[g]), 1e-15)),
        'rms_ratio': float(math.sqrt(pred_e[g]/max(target_e[g], 1e-15)))} for g, name in enumerate(GROUPS)},
        'segments_per_second': float(segs/(framecount/25)), 'segments': int(segs),
        'valid_frames': int(framecount), 'scalar_delta_rate': float(segs/(framecount/25)),
        'initial_scalar_rate': float(4*run_count/(framecount/25)),
        'duration_symbols_per_second': float(segs/(framecount/25)),
        'upper9_centered_mse': (channel_error/max(channel_count, 1)).tolist()}


def center_runs(value, valid):
    result = np.zeros_like(value)
    for start, end in runs(valid):
        result[..., start:end, :] = value[..., start:end, :]-value[..., start:end, :].mean(-2, keepdims=True)
    return result


def uniform_reconstruction(c):
    output = np.zeros_like(c['state'])
    for group in range(4):
        for start, end in runs(c['valid'].numpy()):
            if end-start == 1:
                output[start, group] = c['state'][start, group]
                continue
            count = sum(s['duration'] > 0 and start <= s['start'] < end
                        for s in c['plan']['segments'][group])
            knots = np.unique(np.round(np.linspace(start, end-1, count+1)).astype(int))
            output[start:end, group] = np.interp(np.arange(start, end), knots, c['state'][knots, group])
    return output


def prepare_plans(clips, split):
    scales = fit_dynamic_scales([(clips[i]['state'], clips[i]['groupmask']) for i in split['fit']], floor=.005)
    for c in clips:
        plan = fit_motion_process(c['state'], c['groupmask'], scales,
                                  durations=DURATIONS, complexity_penalty=1.0)
        c['plan'] = plan
        c['reconstructed'] = render_motion_process(plan)
        c['uniform'] = uniform_reconstruction(c)
        c['mean'] = np.zeros_like(c['state'])
        for start, end in runs(c['valid'].numpy()):
            c['mean'][start:end] = c['state'][start:end].mean(0)
        events = []
        for group, segments in enumerate(plan['segments']):
            pd, pl, previous_end = 0., 0., -1
            for s in segments:
                start, duration = s['start'], s['duration']
                if start != previous_end:
                    pd, pl = 0., 0.
                delta = (s['end_value']-s['start_value'])/scales[group]
                if duration in DURATIONS and not s['right_censored']:
                    events.append([start, group, (s['start_value']-c['anchor'][group])/scales[group],
                                   pd, pl, DURATIONS.index(duration), delta])
                pd, pl, previous_end = delta, float(duration), start+duration
        c['events'] = np.asarray(events, dtype=np.float32).reshape(-1, 7)
        c['initial'] = np.asarray([(c['state'][start]-c['anchor'])/scales
            for start, _ in runs(c['valid'].numpy())], dtype=np.float32)
    report = {cell: {key: reconstruction_summary(clips, ids, key)
        for key in ('reconstructed', 'uniform', 'mean')} for cell, ids in split.items()}
    passed = all(row['segments_per_second'] <= 12 and all(
        g['centered_r2'] >= .5 and g['correlation'] >= .8 and .75 <= g['rms_ratio'] <= 1.25
        for g in row['groups'].values()) for cell in ('sentence', 'speaker', 'joint')
        for row in [report[cell]['reconstructed']])
    return scales, report, passed


def batch(clips, ids, mean, std, device):
    n = max(len(clips[i]['valid']) for i in ids)
    features = torch.zeros(len(ids), n, 1540)
    valid = torch.zeros(len(ids), n, dtype=torch.bool)
    events, initials = [], []
    for b, i in enumerate(ids):
        c = clips[i]; size = len(c['valid'])
        features[b, :size] = torch.where(c['valid'][:, None], (c['features']-mean)/std, 0.)
        valid[b, :size] = c['valid']
        if len(c['events']):
            events.append(np.column_stack((np.full(len(c['events']), b), c['events'])))
        initials.extend([(b, row) for row in c['initial']])
    ev = torch.tensor(np.concatenate(events) if events else np.empty((0, 8)), dtype=torch.float32, device=device)
    initial_ix = torch.tensor([b for b, _ in initials], device=device)
    initial_y = torch.tensor(np.stack([row for _, row in initials]), device=device)
    return features.to(device), valid.to(device), ev, initial_ix, initial_y


def likelihood(model, hidden, pooled, ev, initial_ix, initial_y):
    if len(ev):
        b, t, group = ev[:, 0].long(), ev[:, 1].long(), ev[:, 2].long()
        out = model.event_distribution(hidden[b, t], pooled[b], group, ev[:, 3], ev[:, 4], ev[:, 5])
        nll = joint_event_nll(out, ev[:, 6].long(), ev[:, 7])
        prob = out['duration_logits'].softmax(-1)
        brier = (prob-torch.nn.functional.one_hot(ev[:, 6].long(), 7)).square().sum(-1)
    else:
        nll = pooled.new_empty(0); brier = nll
    ini = initial_nll(model.initial_distribution(pooled[initial_ix]), initial_y)
    loss = (nll.mean() if len(nll) else pooled.sum()*0) + ini.mean()/4
    return loss, nll, brier, ini


@torch.no_grad()
def sample_rollout(model, hidden, pooled, valid, seed, clip_keys=None):
    """No motion, teacher boundaries, clip mean, or teacher state in this API."""
    b, frames = valid.shape
    clip_keys = list(range(b)) if clip_keys is None else clip_keys
    if len(clip_keys) != b:
        raise ValueError('One noise key per clip required')
    # Every clip/group/event owns its random draw. Different predicted durations,
    # padding and batch composition cannot consume another process's randomness.
    uniform = hidden.new_empty(b, 4, frames+1)
    normal = torch.empty_like(uniform); initial_noise = torch.empty_like(uniform)
    for i, key in enumerate(clip_keys):
        for group in range(4):
            draw_seed = int.from_bytes(hashlib.sha256(f'{seed}:{key}:{group}'.encode()).digest()[:8], 'little') % (2**63-1)
            # A fixed number of slots supports every native clip in this dataset;
            # stream prefixes do not depend on padded frame count.
            uniform[i, group] = torch.as_tensor(np.random.default_rng(draw_seed).random(frames+1), device=hidden.device)
            normal[i, group] = torch.as_tensor(np.random.default_rng(draw_seed ^ 0xAAAA).standard_normal(frames+1), device=hidden.device)
            initial_noise[i, group] = torch.as_tensor(np.random.default_rng(draw_seed ^ 0x5555).standard_normal(frames+1), device=hidden.device)
    init = model.initial_distribution(pooled)
    current = init['loc'] + init['log_scale'].exp()*initial_noise[..., 0]
    run_index = torch.zeros(b, dtype=torch.long, device=hidden.device)
    seen_valid = torch.zeros(b, dtype=torch.bool, device=hidden.device)
    event_index = torch.zeros(b, 4, dtype=torch.long, device=hidden.device)
    remaining = torch.zeros(b, 4, dtype=torch.long, device=hidden.device)
    slope = torch.zeros_like(current)
    previous_delta = torch.zeros_like(current); previous_duration = torch.zeros_like(current)
    result = hidden.new_zeros(b, frames, 4)
    for t in range(frames):
        active = valid[:, t]
        if t:
            reset = active & ~valid[:, t-1]
            if reset.any():
                run_index[reset & seen_valid] += 1
                z = initial_noise[reset].gather(2, run_index[reset, None, None].expand(-1, 4, 1)).squeeze(-1)
                current[reset] = init['loc'][reset]+init['log_scale'][reset].exp()*z
                remaining[reset] = 0; previous_delta[reset] = 0; previous_duration[reset] = 0
        result[:, t] = torch.where(active[:, None], current, 0.)
        due = active[:, None] & (remaining == 0)
        bi, gi = due.nonzero(as_tuple=True)
        if len(bi):
            out = model.event_distribution(hidden[bi, t], pooled[bi], gi, current[bi, gi],
                previous_delta[bi, gi], previous_duration[bi, gi])
            ei = event_index[bi, gi]
            u = uniform[bi, gi, ei]
            index = (u[:, None] > out['duration_logits'].softmax(-1).cumsum(-1)).sum(-1).clamp_max(6)
            duration = model.durations[index]
            delta = out['loc'].gather(1, index[:, None]).squeeze(-1)+out['log_scale'].gather(
                1, index[:, None]).squeeze(-1).exp()*normal[bi, gi, ei]
            remaining[bi, gi] = duration
            slope[bi, gi] = delta/duration
            previous_delta[bi, gi] = delta
            previous_duration[bi, gi] = duration.to(previous_duration.dtype)
            event_index[bi, gi] += 1
        current = current + torch.where(active[:, None], slope, 0.)
        remaining = torch.where(active[:, None], (remaining-1).clamp_min(0), remaining)
        seen_valid |= active
    return result


def distribution_metrics(samples, target, valid):
    """Fair ES on scale-normalized trajectories; all draws, not best-of-N."""
    y = target[valid].astype(np.float64); x = samples[:, valid].astype(np.float64)
    k = len(x)
    out = {}
    for centered in (False, True):
        xx = center_runs(samples.astype(np.float64), valid)[:, valid] if centered else x
        yy = center_runs(target.astype(np.float64), valid)[valid] if centered else y
        d = np.sqrt(((xx-yy)**2).mean(1)).mean(0)
        spread = sum(np.sqrt(((xx[i]-xx[j])**2).mean(0)) for i in range(k) for j in range(i))/(k*(k-1))
        out['centered_es' if centered else 'raw_es'] = (d-spread).tolist()
    variogram = np.zeros(4); count = 0; by_lag = {}
    for lag in (1, 4, 16, 32):
        support = np.array([valid[t:t+lag+1].all() for t in range(len(valid)-lag)])
        if support.any():
            xd = np.abs(samples[:, lag:]-samples[:, :-lag])[:, support]**.5
            yd = np.abs(target[lag:]-target[:-lag])[support]**.5
            error = ((xd.mean(0)-yd)**2)
            variogram += error.sum(0); count += int(support.sum())
            by_lag[str(lag)] = error.mean(0).tolist()
    out['variogram'] = (variogram/max(count, 1)).tolist()
    xc = center_runs(samples.astype(np.float64), valid)[:, valid]
    yc = center_runs(target.astype(np.float64), valid)[valid]
    te = (yc**2).sum(0); pe = (xc**2).sum(1)
    out['rms_ratio'] = np.sqrt(pe/np.maximum(te, 1e-12)).mean(0).tolist()
    out['correlation'] = ((xc*yc).sum(1)/np.maximum(np.sqrt(pe*te), 1e-12)).mean(0).tolist()
    out['mean_error'] = ((x.mean(1)-y.mean(0))**2).mean(0).tolist()
    # Aggregate gate retains its fixed four-lag score; detailed values are separate.
    return out


def aggregate(rows):
    metrics = ('event_nll', 'initial_segment_nll', 'interior_segment_nll', 'initial_nll', 'duration_brier', 'centered_es', 'raw_es', 'variogram',
               'rms_ratio', 'correlation', 'mean_error', 'out_of_domain')
    return {key: np.mean([r[key] for r in rows if r.get(key) is not None], axis=0).tolist()
        for key in metrics if any(r.get(key) is not None for r in rows)}


@torch.no_grad()
def evaluate(model, clips, ids, mean, std, scales, args, local_enabled, output, mode='real'):
    rows = []; examples = {}
    donors = {}
    if mode == 'mismatch':
        for i in ids:
            options = [j for j in ids if clips[j]['speaker'] == clips[i]['speaker'] and
                clips[j]['emotion'] == clips[i]['emotion'] and clips[j]['sentence'] != clips[i]['sentence']
                and len(runs(clips[j]['valid'].numpy())) == 1]
            if options:
                donors[i] = sorted(options, key=lambda j: clips[j]['clip_id'])[0]
        ids = [i for i in ids if i in donors]
    for start in range(0, len(ids), 16):
        selected = ids[start:start+16]
        features, valid, ev, ini_ix, ini_y = batch(clips, selected, mean, std, args.device)
        hidden, pooled = model.forward_context(features, valid, local_enabled=local_enabled)
        if mode == 'static':
            hidden = (hidden*valid[..., None]).sum(1, keepdim=True)/valid.sum(1)[:, None, None]
            hidden = hidden.expand(-1, valid.shape[1], -1)
        elif mode == 'reverse':
            hidden = hidden.clone()
            for b in range(len(selected)):
                for left, right in runs(valid[b].cpu().numpy()):
                    hidden[b, left:right] = hidden[b, left:right].flip(0)
        elif mode == 'mismatch':
            df, dv, _, _, _ = batch(clips, [donors[i] for i in selected], mean, std, args.device)
            dh, _ = model.forward_context(df, dv, local_enabled=True)
            hidden = hidden.clone()
            for b in range(len(selected)):
                source = dh[b, dv[b]].T[None]
                for left, right in runs(valid[b].cpu().numpy()):
                    hidden[b, left:right] = torch.nn.functional.interpolate(
                        source, size=right-left, mode='linear', align_corners=False)[0].T
        elif mode != 'real':
            raise ValueError('Unknown intervention')
        _, nll, brier, ini = likelihood(model, hidden, pooled, ev, ini_ix, ini_y)
        draws = torch.stack([sample_rollout(model, hidden, pooled, valid, seed,
            [clips[i]['clip_id'] for i in selected]) for seed in SEEDS]).cpu().numpy()
        for b, i in enumerate(selected):
            c = clips[i]; n = len(c['valid']); mask = c['valid'].numpy()
            target = (c['state']-c['anchor'])/scales
            samples = draws[:, b, :n]
            em = ev[:, 0].long() == b; im = ini_ix == b
            raw = samples*scales+c['anchor']
            row = {'clip_id': c['clip_id'], 'speaker': c['speaker'], 'sentence': c['sentence'], 'emotion': c['emotion'],
                'event_count': int(em.sum()), 'event_nll': float(nll[em].mean()) if em.any() else None,
                'initial_segment_nll': float(nll[em & (ev[:, 5] == 0)].mean()) if (em & (ev[:, 5] == 0)).any() else None,
                'interior_segment_nll': float(nll[em & (ev[:, 5] != 0)].mean()) if (em & (ev[:, 5] != 0)).any() else None,
                'duration_brier': float(brier[em].mean()) if em.any() else None, 'initial_nll': float(ini[im].mean()),
                'out_of_domain': ((raw[:, mask] < 0)|(raw[:, mask] > 1)).mean((0, 1)).tolist(),
                **distribution_metrics(samples, target, mask)}
            rows.append(row)
            if mode == 'real':
                examples[c['clip_id']] = {'samples': raw.astype(np.float32), 'target': c['state'],
                    'teacher': c['reconstructed'], 'valid': mask, 'anchor': c['anchor']}
    report = {'summary': aggregate(rows), 'rows': rows,
        'by_speaker': {str(g): aggregate([r for r in rows if r['speaker'] == g]) for g in sorted({r['speaker'] for r in rows})},
        'by_emotion': {str(g): aggregate([r for r in rows if r['emotion'] == g]) for g in sorted({r['emotion'] for r in rows})},
        'intervention': mode, 'donor_mapping': {clips[i]['clip_id']: clips[j]['clip_id'] for i, j in donors.items()}}
    save_json(output, report)
    if mode == 'real':
        torch.save(examples, output.with_suffix('.pt'))
    return report


def audio_gate(reports):
    gains = {}
    for cell in ('sentence', 'speaker', 'joint'):
        a = reports['global_only'][cell]['rows']; b = reports['local_audio'][cell]['rows']
        if [r['clip_id'] for r in a] != [r['clip_id'] for r in b]:
            raise ValueError('Paired evaluation IDs mismatch')
        paired = [(x['sentence'], x['event_nll']-y['event_nll']) for x, y in zip(a, b)
                  if x['event_nll'] is not None and y['event_nll'] is not None]
        if not paired or not np.isfinite([g for _, g in paired]).all():
            return {'passed': False, 'reason': 'missing_or_nonfinite_event_scores', 'cells': gains}
        for arm in ('global_only', 'local_audio'):
            summary = reports[arm][cell]['summary']
            for key in ('centered_es', 'raw_es', 'variogram', 'rms_ratio', 'out_of_domain'):
                if key not in summary or not np.isfinite(summary[key]).all():
                    return {'passed': False, 'reason': 'missing_or_nonfinite_free_rollout_scores', 'cells': gains}
        cluster = {s: [g for key, g in paired if key == s] for s in sorted({s for s, _ in paired})}
        rng = np.random.default_rng(20260918)
        keys = list(cluster)
        boot = [np.mean([g for ix in rng.integers(len(keys), size=len(keys)) for g in cluster[keys[ix]]]) for _ in range(2000)]
        gains[cell] = {'nll_gain': float(np.mean([g for _, g in paired])),
            'sentence_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'distribution_ratios': {key: (np.asarray(reports['local_audio'][cell]['summary'][key])/
                np.maximum(np.asarray(reports['global_only'][cell]['summary'][key]), 1e-12)).tolist()
                for key in ('centered_es', 'raw_es', 'variogram')}}
    passed = (gains['sentence']['nll_gain'] >= .01 and gains['sentence']['sentence_bootstrap_ci95'][0] > 0
        and gains['speaker']['nll_gain'] > 0 and all(max(row) <= 1.05
        for cell in ('sentence', 'speaker') for row in gains[cell]['distribution_ratios'].values()))
    passed = passed and all(.25 <= rms <= 2 for cell in ('sentence', 'speaker')
        for rms in reports['local_audio'][cell]['summary']['rms_ratio']) and all(value <= .05
        for cell in ('sentence', 'speaker') for value in reports['local_audio'][cell]['summary']['out_of_domain'])
    return {'passed': bool(passed), 'cells': gains}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--representation-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh pilot output')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260918); np.random.seed(20260918)
    started = time.monotonic()
    save_json(args.output/'status.json', {'status': 'loading', 'schema': SCHEMA, 'smoke': args.smoke})
    clips, split, lineage = load_clips(args)
    protocol = {'schema': SCHEMA, 'source': lineage, 'protocol_sha256': sha(Path(__file__).resolve().parents[1]/'docs/MOTION_PROCESS_PROTOCOL_20260918.md'),
        'split': {cell: [{k: clips[i][k] for k in ('clip_id', 'speaker', 'sentence', 'emotion')} for i in ids] for cell, ids in split.items()},
        'smoke': args.smoke, 'epochs': 2 if args.smoke else 30, 'test_loaded': False, 'dev_indexed': False,
        'code_sha256': {f: sha(Path(__file__).resolve().parents[1]/f) for f in ('scripts/train_motion_process.py',
            'scripts/motion_process_representation.py', 'kinetalk_b0/models/motion_process_prior.py')}}
    save_json(args.output/'protocol.json', protocol)
    scales, representation, passed = prepare_plans(clips, split)
    save_json(args.output/'representation.json', {'passed': passed, 'scales': scales.tolist(), 'cells': representation})
    torch.save({c['clip_id']: c['plan'] for c in clips}, args.output/'teacher_plans.pt')
    print('REPRESENTATION_GATE', passed, json.dumps(representation), flush=True)
    if (not passed and not args.smoke) or args.representation_only:
        save_json(args.output/'status.json', {'status': 'representation_complete' if passed else 'representation_gate_failed',
            'seconds': time.monotonic()-started, 'smoke': args.smoke, 'generator_started': False})
        return
    mean, std = fit_feature_stats(clips, split['fit'])
    torch.save({'mean': mean, 'std': std, 'scales': scales, 'fit_clip_ids': [clips[i]['clip_id'] for i in split['fit']]},
               args.output/'normalization.pt')
    prototype = MotionProcessPrior(feature_dim=1540, hidden=96, groups=4, durations=DURATIONS)
    initial = copy.deepcopy(prototype.state_dict())
    reports, matching = {}, {}
    for arm, local_enabled in (('global_only', False), ('local_audio', True)):
        folder = args.output/arm; folder.mkdir()
        model = MotionProcessPrior(feature_dim=1540, hidden=96, groups=4, durations=DURATIONS).to(args.device)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
        rng = np.random.default_rng(20260918); order_hash = hashlib.sha256(); losses = []
        for epoch in range(protocol['epochs']):
            tick = time.monotonic(); model.train(); total, updates = 0., 0
            order = rng.permutation(split['fit']).tolist(); order_hash.update(np.asarray(order, dtype=np.int64).tobytes())
            for start in range(0, len(order), 24):
                ix = order[start:start+24]
                f, v, ev, ini_ix, ini_y = batch(clips, ix, mean, std, args.device)
                hidden, pooled = model.forward_context(f, v, local_enabled=local_enabled)
                loss, _, _, _ = likelihood(model, hidden, pooled, ev, ini_ix, ini_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite likelihood')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); total += float(loss.detach()); updates += 1
            row = {'epoch': epoch+1, 'loss': total/updates, 'updates': updates, 'seconds': time.monotonic()-tick}
            losses.append(row); save_json(folder/'losses.json', losses)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                        'schema': SCHEMA, 'arm': arm}, folder/'last.tmp')
            (folder/'last.tmp').replace(folder/'last.pt')
            save_json(args.output/'status.json', {'status': 'training', 'arm': arm, **row, 'smoke': args.smoke})
            print(arm, json.dumps(row), flush=True)
        model.eval(); reports[arm] = {}
        for cell in ('sentence', 'speaker', 'joint'):
            reports[arm][cell] = evaluate(model, clips, split[cell], mean, std, scales, args,
                local_enabled, folder/(cell+'.json'))
        if local_enabled:
            interventions = {}
            for mode in ('static', 'reverse', 'mismatch'):
                interventions[mode] = evaluate(model, clips, split['sentence'], mean, std, scales, args, True, folder/(mode+'.json'), mode)
            support = set(interventions['mismatch']['donor_mapping'])
            save_json(folder/'common_interventions.json', {'clip_ids': sorted(support), 'summary': {
                mode: aggregate([r for r in report['rows'] if r['clip_id'] in support]) for mode, report in
                {'real': reports[arm]['sentence'], **interventions}.items()}})
        torch.save({'model': model.cpu().state_dict(), 'schema': SCHEMA, 'arm': arm, 'epochs': protocol['epochs'],
                    'normalization_sha256': sha(args.output/'normalization.pt')}, folder/'final.pt')
        matching[arm] = {'order_sha256': order_hash.hexdigest(), 'updates': sum(r['updates'] for r in losses)}
    if matching['global_only'] != matching['local_audio']:
        raise ValueError('Training budget/order not matched')
    gate = audio_gate(reports); save_json(args.output/'audio_gate.json', gate)
    save_json(args.output/'matching.json', matching)
    save_json(args.output/'status.json', {'status': 'complete', 'smoke': args.smoke,
        'representation_passed': passed, 'audio_passed': gate['passed'], 'seconds': time.monotonic()-started,
        'generator_started': False, 'default_replaced': False, 'test_loaded': False})
    print('MOTION_PROCESS_COMPLETE', json.dumps(gate), flush=True)


if __name__ == '__main__':
    main()
