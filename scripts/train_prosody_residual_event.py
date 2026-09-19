"""Source-bound local-prosody event ablation; smoke never certifies success."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from kinetalk_b0.models.prosody_residual_schedule import StaticScheduleHead, ProsodyResidualSchedule
from scripts.audio_event_condition import derive_event_condition
from scripts.event_schedule_teacher import valid_runs
from scripts.probe_motion_condition_predictability import split_train_pool, sha
from scripts import train_continuous_motion_latent as common
from scripts.train_event_schedule_pipeline import target_arrays, process_nll, paired_delta, initialize_rates, SOURCES as EVENT_SOURCES

SCHEMA = 'prosody_residual_event_v2'
SEEDS = (42, 123, 2026)
SOURCES = tuple(dict.fromkeys(EVENT_SOURCES + ('scripts/train_prosody_residual_event.py',
    'scripts/audio_event_condition.py', 'kinetalk_b0/models/prosody_residual_schedule.py')))


def arr(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def acoustic_condition(clip, stats, mode='real'):
    """Deployment inputs only; no target motion or supervision mask."""
    if mode not in ('real', 'reverse', 'null'): raise ValueError('Unsupported intervention')
    valid = clip['valid'].bool(); raw = clip['features'].float()
    prosody = ((raw[:, 1536:1540]-stats['audio_mean'][1536:1540]) /
               stats['audio_scale'][1536:1540])[valid].mean(0)
    context = torch.cat((((clip['context']-stats['context_mean'])/stats['context_scale']).float(), prosody))
    condition = torch.zeros(len(valid), 10)
    if mode != 'null':
        intervened = raw.clone()
        if mode == 'reverse':
            for left, right in valid_runs(arr(valid)): intervened[left:right] = intervened[left:right].flip(0)
        condition = derive_event_condition(intervened, valid).float()
        for left, right in valid_runs(arr(valid)):
            section = condition[left:right]
            condition[left:right] = section-section.mean(0, keepdim=True)
    return condition, context, valid


def prepare(clip, labels, stats, mode='real'):
    condition, context, valid = acoustic_condition(clip, stats, mode)
    return {'clip_id': clip['clip_id'], 'sentence': clip['sentence'], 'condition': condition,
            'context': context, 'valid': valid,
            **dict(zip(('onset', 'risk', 'duration'), map(torch.from_numpy, target_arrays(labels[clip['clip_id']]))))}


def batch(rows, rng, device, frames=180, size=12):
    """Uniform clip/run/crop; target-independent sampling, with audio halo."""
    x = torch.zeros(size, frames, 10, device=device)
    c = []; v = torch.zeros(size, frames, dtype=torch.bool, device=device)
    y = torch.zeros(size, frames, 4, device=device)
    r = torch.zeros_like(y, dtype=torch.bool); d = torch.zeros_like(y, dtype=torch.long)
    for i in range(size):
        row = rows[int(rng.integers(len(rows)))]; runs = valid_runs(arr(row['valid']))
        left, right = runs[int(rng.integers(len(runs)))]
        a = int(rng.integers(left, right-frames+1)) if right-left > frames else left
        n = min(frames, right-a)
        x[i, :n] = row['condition'][a:a+n].to(device); c.append(row['context']); v[i, :n] = True
        y[i, :n] = row['onset'][a:a+n].to(device); r[i, :n] = row['risk'][a:a+n].to(device)
        d[i, :n] = row['duration'][a:a+n].to(device)
        if a > left: r[i, :min(15, n)] = False
        if a+n < right: r[i, max(0, n-15):n] = False
    return x, torch.stack(c).to(device), v, y, r, d


def train(model, rows, seed, steps, device, residual, output, status):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 3e-4, weight_decay=.01)
    rng = np.random.default_rng(seed); losses = []; started = time.monotonic()
    risk_seen = events_seen = empty_batches = 0
    for step in range(1, steps+1):
        x, c, v, y, r, d = batch(rows, rng, device)
        out = model(x, c, v) if residual else model(c, v.shape[1])
        loss = process_nll(out, y, r, d)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite event likelihood')
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
        opt.step(); losses.append(float(loss.detach()))
        risk_seen += int(r.sum()); events_seen += int((y.bool() & r).sum()); empty_batches += int(not r.any())
        if step == 1 or step % 250 == 0 or step == steps:
            elapsed = time.monotonic()-started
            status('training', arm=output.stem, seed=seed, step=step, steps=steps,
                   loss=losses[-1], elapsed_seconds=elapsed, seconds_per_step=elapsed/step)
            common._save(output.with_name(output.stem+'_last.pt'), {
                'state': model.state_dict(), 'optimizer': opt.state_dict(), 'step': step, 'seed': seed,
                'losses': losses, 'batch_rng': rng.bit_generator.state, 'risk_positions_seen': risk_seen,
                'events_seen': events_seen, 'empty_batches': empty_batches, 'seconds': elapsed})
    common._save(output, {'state': model.state_dict(), 'seed': seed, 'steps': steps, 'losses': losses,
             'risk_positions_seen': risk_seen, 'events_seen': events_seen, 'empty_batches': empty_batches,
             'seconds': time.monotonic()-started, 'context_dim': rows[0]['context'].numel(), 'hidden': 64})
    return model.eval()


@torch.no_grad()
def evaluate(model, rows, device, residual):
    result = []
    for row in rows:
        v = row['valid'][None].to(device); c = row['context'][None].to(device)
        out = model(row['condition'][None].to(device), c, v) if residual else model(c, len(row['condition']))
        p = out['onset_logits'][0].sigmoid().cpu().numpy()
        lp = out['duration_logits'][0].log_softmax(-1).cpu().numpy()
        onset, risk, dur = map(arr, (row['onset'], row['risk'], row['duration']))
        if not risk.any(): continue
        if not np.isfinite(p).all() or not np.isfinite(lp).all(): raise FloatingPointError('Nonfinite prediction')
        q = np.clip(p[risk], 1e-7, 1-1e-7); yy = onset[risk]
        nll = float(-(yy*np.log(q)+(1-yy)*np.log1p(-q)).mean())
        f, g = np.where((onset > 0) & risk); dn = -lp[f, g, dur[f, g]]
        result.append({'clip_id': row['clip_id'], 'sentence': row['sentence'],
                       'brier': float(np.square(q-yy).mean()), 'onset_nll': nll,
                       'joint_nll': nll+float(dn.sum())/int(risk.sum()),
                       'duration_nll': float(dn.mean()) if len(dn) else None,
                       'risk_positions': int(risk.sum()), 'scored_onsets': len(dn),
                       'probability': p, 'duration_log_probability': lp,
                       'truth_onset': onset, 'truth_duration': dur, 'risk': risk})
    return result


def mean(rows):
    if not rows: raise ValueError('No supported event targets')
    result = {}
    for k in ('brier', 'onset_nll', 'joint_nll', 'duration_nll'):
        observed = [r[k] for r in rows if r[k] is not None]
        result[k] = float(np.mean(observed)) if observed else None
    result.update(clips=len(rows), scored_onsets=sum(r['scored_onsets'] for r in rows))
    return result


def main(a):
    if a.output.exists(): raise FileExistsError(a.output)
    if a.steps < 1 or (a.smoke and a.steps > 100): raise ValueError('Invalid budget')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    a.output.mkdir(parents=True); started = time.monotonic()
    def status(state, **detail):
        common._write(a.output/'status.json', {'schema': SCHEMA, 'state': state, 'updated': time.time(),
                     'smoke': a.smoke, 'default_replaced': False, **detail})
    status('loading')
    ref = json.loads((a.source_run/'protocol.json').read_text())
    source = json.loads((a.event_run/'protocol.json').read_text())
    if source['smoke'] or source['dataset_sha256'] != ref['dataset_sha256']: raise ValueError('Formal source mismatch')
    if sha(a.dataset) != ref['dataset_sha256']: raise ValueError('Dataset mismatch')
    if source['source_protocol_sha256'] != sha(a.source_run/'protocol.json'): raise ValueError('Prior protocol mismatch')
    if source['fit_stats_sha256'] != sha(a.source_run/'fit_stats.pt'): raise ValueError('Fit statistics mismatch')
    data = torch.load(a.dataset, weights_only=False, map_location='cpu', mmap=True)
    if data.get('schema') != 'continuous_motion_dataset_v1': raise ValueError('Unexpected dataset')
    fit, valid = split_train_pool(data['clips'], ref); del data
    if source['train_ids'] != [c['clip_id'] for c in fit] or source['validation_ids'] != [c['clip_id'] for c in valid]:
        raise ValueError('Exact event split membership differs')
    stats = common._load(a.source_run/'fit_stats.pt')
    if set(stats['train_clip_ids']) != {c['clip_id'] for c in fit}: raise ValueError('Fit statistics membership mismatch')
    labels = common._load(a.event_run/'event_labels.pt')
    if set(labels) != {c['clip_id'] for c in fit+valid}: raise ValueError('Event label membership differs')
    code = Path(__file__).resolve().parents[1]
    common._write(a.output/'protocol.json', {'schema': SCHEMA, 'dataset_sha256': ref['dataset_sha256'],
        'source_protocol_sha256': sha(a.source_run/'protocol.json'), 'event_protocol_sha256': sha(a.event_run/'protocol.json'),
        'fit_stats_sha256': sha(a.source_run/'fit_stats.pt'), 'teacher_sha256': sha(a.event_run/'teacher.json'),
        'labels_sha256': sha(a.event_run/'event_labels.pt'), 'sources': {p: sha(code/p) for p in SOURCES},
        'steps_per_stage': a.steps, 'seeds': list(SEEDS[:1] if a.smoke else SEEDS), 'smoke': a.smoke,
        'fit_ids': [c['clip_id'] for c in fit], 'validation_ids': [c['clip_id'] for c in valid],
        'condition': '202 normalized context + 4 fit-normalized acoustic mean; local10 centered within native runs',
        'residual': 'zero initialized difference tanh(f(condition))-tanh(f(zero)); logit correction bounded [-2,2]',
        'sampling': 'uniform clip then audio run/crop; crop likelihood halo15; no event balancing',
        'gate': 'all3seeds residual-vs-static Brier/jointNLL/durationNLL sentence-bootstrap CIupper<0; residual-vs-reverse jointNLL CIupper<0',
        'selection': 'fixed final step; never select seed; smoke never passes scientific gate',
        'scope': 'inner development; inherited upstream exposure; same event targets; outer tensors not indexed',
        'default_replaced': False})
    for p in SOURCES:
        dest = a.output/'source'/p; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((code/p).read_bytes())
    train_rows = [prepare(c, labels, stats) for c in fit]; val_rows = [prepare(c, labels, stats) for c in valid]
    reverse_rows = [prepare(c, labels, stats, 'reverse') for c in valid]
    null_rows = [prepare(c, labels, stats, 'null') for c in valid]
    excluded = [r['clip_id'] for r in train_rows if not r['risk'].any()]
    train_rows = [r for r in train_rows if r['risk'].any()]
    if not train_rows or train_rows[0]['context'].numel() != 206: raise ValueError('Missing training support or wrong context')
    common._write(a.output/'coverage.json', {'fit_clips': len(train_rows), 'validation_clips': len(val_rows),
        'excluded_no_risk_fit': excluded, 'excluded_no_risk_validation': [r['clip_id'] for r in val_rows if not r['risk'].any()]})
    reports = {}
    for seed in (SEEDS[:1] if a.smoke else SEEDS):
        torch.manual_seed(seed)
        base = StaticScheduleHead(206).to(a.device); initialize_rates(base, train_rows)
        base = train(base, train_rows, seed, a.steps, a.device, False, a.output/f'static_{seed}.pt', status)
        static_state_hash = common._value_sha(base.state_dict()); base.zero_grad(set_to_none=True)
        s = evaluate(base, val_rows, a.device, False); st = evaluate(base, train_rows, a.device, False)
        torch.manual_seed(seed+10000); residual = ProsodyResidualSchedule(base).to(a.device)
        residual = train(residual, train_rows, seed, a.steps, a.device, True, a.output/f'residual_{seed}.pt', status)
        if common._value_sha(base.state_dict()) != static_state_hash: raise RuntimeError('Static prior was modified')
        if any(p.grad is not None for p in base.parameters()): raise RuntimeError('Frozen prior retained gradients')
        r = evaluate(residual, val_rows, a.device, True); rt = evaluate(residual, train_rows, a.device, True)
        reverse = evaluate(residual, reverse_rows, a.device, True); null = evaluate(residual, null_rows, a.device, True)
        if len(s) != len(null): raise RuntimeError('Static/null evaluation membership differs')
        for x, y in zip(s, null):
            if x['clip_id'] != y['clip_id'] or any(not np.array_equal(x[k], y[k]) for k in ('probability', 'duration_log_probability')):
                raise RuntimeError('Zero condition does not preserve exact static probabilities')
        reports[str(seed)] = {'static': mean(s), 'prosody_residual': mean(r), 'reverse': mean(reverse),
            'fit_static': mean(st), 'fit_prosody_residual': mean(rt), 'static_hash_preserved': True, 'null_exact_static': True,
            'paired_brier': paired_delta(r, s, 'brier'), 'paired_joint_nll': paired_delta(r, s, 'joint_nll'),
            'paired_duration_nll': paired_delta(r, s, 'duration_nll'), 'paired_reverse_nll': paired_delta(r, reverse, 'joint_nll')}
        common._save(a.output/f'predictions_{seed}.pt', {'static': s, 'residual': r, 'reverse': reverse, 'fit_static': st, 'fit_residual': rt})
        common._write(a.output/'results.json', reports)
    passed = not a.smoke and len(reports) == 3 and all(
        v[k]['ci95'] is not None and v[k]['ci95'][1] < 0 for v in reports.values()
        for k in ('paired_brier', 'paired_joint_nll', 'paired_duration_nll', 'paired_reverse_nll'))
    common._write(a.output/'decision.json', {'prosody_predictability_passed': passed, 'default_replaced': False,
        'smoke': a.smoke, 'scope': 'predictability ablation only; not full generation acceptance'})
    status('complete', seconds=time.monotonic()-started, prosody_predictability_passed=passed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dataset', 'source-run', 'event-run', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    try: main(args)
    except Exception as exc:
        if args.output.exists() and not isinstance(exc, FileExistsError):
            common._write(args.output/'status.json', {'schema': SCHEMA, 'state': 'failed', 'error': repr(exc), 'updated': time.time()})
        raise
