"""Frozen-prior multiscale audio test with automatic short-run decision.

Uses the established 613/206 sentence split and complete joint observation
support. Only adapter weights are optimized; native generation never reads GT.
The inner set is repeatedly inspected development, not an untouched test set.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from kinetalk_b0.models.prior_audio_multiscale import MultiScalePriorAudioResidual
from scripts import train_continuous_motion_latent as common
from scripts import audit_prior_audio_adapter as audit
from scripts.evaluate_continuous_motion_latent import evaluate_generation, _mismatched_audio
from scripts.prepare_continuous_motion_dataset import prepare_segments, contiguous_runs
from scripts.probe_motion_condition_predictability import split_train_pool
from scripts.train_prior_audio_adapter import static_features, diverse_diagnostics

SCHEMA = 'multiscale_prior_audio_v2'
SEED = 2026091914
SEEDS = (42, 123, 2026, 77)
SOURCES = tuple(dict.fromkeys((*common.SOURCE_FILES,
    'kinetalk_b0/models/prior_audio_multiscale.py',
    'scripts/train_multiscale_prior_audio.py', 'scripts/launch_multiscale_prior_audio.py', 'scripts/audit_prior_audio_adapter.py',
    'scripts/probe_motion_condition_predictability.py', 'scripts/train_prior_audio_adapter.py')))


def smoke_population(clips):
    """Two metadata-selected sentences per emotion where available."""
    selected = []
    for emotion in sorted({str(c['emotion']) for c in clips}):
        rows = sorted([c for c in clips if str(c['emotion']) == emotion], key=lambda c: c['clip_id'])
        first = rows[0]
        other = [c for c in rows if c['sentence'] != first['sentence']]
        selected.append(first)
        if other:
            selected.append(min(other, key=lambda c: (c['speaker'] != first['speaker'], c['clip_id'])))
    return selected


def prepare_training(fit, stats, ae, device):
    segments, coverage = prepare_segments(fit, stats, 'train', 200)
    encoded = common._encode_segments(ae, segments, device)
    static, static_coverage = prepare_segments(static_features(fit), stats, 'train', 200)
    if coverage != static_coverage or [x['metadata'] for x in segments] != [x['metadata'] for x in static]:
        raise RuntimeError('Static training support differs')
    matched = [{**row, 'audio': static[i]['audio']} for i, row in enumerate(encoded)]
    coverage['retained_clip_ids'] = sorted({x['metadata']['clip_id'] for x in encoded})
    return encoded, matched, coverage


class DonorBank:
    """Fit-only audio donors; no donor motion or motion-mask access.

    Prefer same speaker/emotion, different sentence. Fall back to same emotion
    across speakers and record the rate. Skip if no different sentence exists.
    Resample longest native donor run onto the query supervised segment clock.
    """
    def __init__(self, fit, stats):
        self.clips = {c['clip_id']: c for c in fit}
        self.candidates, self.same_speaker, self.audio = {}, {}, {}
        for c in fit:
            cid = c['clip_id']
            eligible = [d for d in fit if d['sentence'] != c['sentence'] and d['emotion'] == c['emotion']]
            same = [d for d in eligible if d['speaker'] == c['speaker']]
            self.candidates[cid] = sorted(d['clip_id'] for d in (same or eligible))
            self.same_speaker[cid] = bool(same)
            left, right = max(contiguous_runs(c['valid']), key=lambda p: (p[1]-p[0], -p[0]))
            self.audio[cid] = ((c['features'][left:right]-stats['audio_mean'])/stats['audio_scale']).float()

    def batch(self, items, reference, frame_valid, *, seed, step, static=False):
        swapped = torch.zeros_like(reference)
        eligible = torch.zeros(len(items), dtype=torch.bool, device=reference.device)
        rng = np.random.default_rng(seed*1000003+step)
        donors = []; same = 0
        for i, item in enumerate(items):
            cid = item['metadata']['clip_id']; candidates = self.candidates[cid]
            if not candidates:
                swapped[i] = reference[i]; donors.append(None); continue
            donor = candidates[int(rng.integers(len(candidates)))]; donors.append(donor)
            x = self.audio[donor]
            if static:
                x = x.mean(0, keepdim=True)
            n = int(frame_valid[i].sum())
            value = F.interpolate(x.T[None], size=n, mode='linear', align_corners=False)[0].T
            swapped[i].reshape(-1, reference.shape[-1])[:n] = value.to(reference.device)
            eligible[i] = True; same += int(self.same_speaker[cid])
        return swapped, eligible, donors, same


def objective(model, target, valid, context, audio, noise, fraction, frames,
              swapped, eligible, swap_weight=.25, margin=.001):
    time = fraction[:, None, None]
    noisy = (1-time)*noise+time*target
    desired = target-noise
    with torch.no_grad():
        base = model.prior.velocity(noisy, fraction, valid, context, audio,
                                    use_audio=False, audio_frame_valid=frames)
    real = base + model.residual(noisy, fraction, valid, context, audio, audio_frame_valid=frames)
    wrong = base + model.residual(noisy, fraction, valid, context, swapped, audio_frame_valid=frames)
    def per_example(v):
        return torch.where(valid[..., None], (v-desired).square(), 0.).sum((1, 2))/(valid.sum(1)*target.shape[-1])
    a, b = per_example(real), per_example(wrong)
    # FM uses the source token-coordinate weighting; the auxiliary contrast
    # gives each eligible query equal weight with shared target/noise/time.
    fm = (a*valid.sum(1)).sum()/valid.sum()
    ranking = F.relu(margin+a[eligible]-b[eligible]).mean() if eligible.any() else fm*0
    loss = fm+swap_weight*ranking
    return loss, {'fm': float(fm.detach()), 'ranking': float(ranking.detach()),
                  'real_minus_swap_fm': float((a-b)[eligible].mean().detach()) if eligible.any() else None}


def adapter_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('prior.')}


def load_adapter(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {k for k in model.state_dict() if k.startswith('prior.')}
    if set(missing) != expected or unexpected:
        raise ValueError('Incomplete or unexpected adapter checkpoint')


def train_arm(model, items, stats, bank, args, arm, stop, binding):
    path = args.output/(arm+'_last.pt')
    optimizer = torch.optim.AdamW(model.residual_parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed+1); first = 0; losses = []; chain = '0'*64; donor_chain = '0'*64
    seconds = 0.; paired_count = same_count = total_count = 0
    initial = common._value_sha(adapter_state(model))
    if path.exists():
        saved = common._load(path)
        if saved['binding'] != binding or saved['initial_sha256'] != initial or saved['config'] != model.config:
            raise ValueError('Resume model or protocol binding differs')
        load_adapter(model, saved['state']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['batch_rng']; first = saved['step']; losses = saved['losses']
        chain, donor_chain = saved['order_sha256'], saved['donor_sha256']
        seconds = saved['training_seconds']
        paired_count, same_count, total_count = saved['donor_counts']
        if first > stop:
            raise ValueError('Saved update exceeds requested endpoint')
    prior_sha = common._value_sha(model.prior.state_dict())
    groups = common._clip_groups(items); model.train(); tick = time.monotonic()
    for step in range(first+1, stop+1):
        indices = common._sample_indices(groups, args.batch_size, rng)
        chain = hashlib.sha256(bytes.fromhex(chain)+np.asarray(indices, dtype='<i8').tobytes()).hexdigest()
        batch = [items[i] for i in indices]
        values = common._flow_batch(model, batch, stats, torch.device(args.device), args.seed+2, step)
        target, valid, context, audio, noise, fraction, frames = values
        swapped, eligible, donors, same = bank.batch(batch, audio, frames, seed=args.seed+3,
                                                   step=step, static=arm == 'matched_static')
        donor_chain = hashlib.sha256(bytes.fromhex(donor_chain)+json.dumps(donors).encode()).hexdigest()
        loss, detail = objective(model, *values, swapped, eligible)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(model.residual_parameters()), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        paired_count += int(eligible.sum()); same_count += same; total_count += len(batch)
        losses.append({'step': step, 'loss': float(loss.detach()), **detail})
        if step == 1 or step % 100 == 0 or step == stop:
            if str(args.device).startswith('cuda'): torch.cuda.synchronize()
            elapsed = seconds+time.monotonic()-tick
            status = {'state': 'training', 'arm': arm, 'step': step, 'endpoint': stop,
                      'seconds_per_step': elapsed/step, 'training_seconds': elapsed,
                      'updated': time.time(), **losses[-1]}
            common._write(args.output/'status.json', status); print(json.dumps(status), flush=True)
            common._save(path, {'schema': SCHEMA, 'binding': binding, 'config': model.config,
                'state': adapter_state(model), 'optimizer': optimizer.state_dict(), 'step': step,
                'initial_sha256': initial, 'order_sha256': chain, 'donor_sha256': donor_chain,
                'batch_rng': rng.bit_generator.state, 'losses': losses, 'training_seconds': elapsed,
                'donor_counts': [paired_count, same_count, total_count]})
    if common._value_sha(model.prior.state_dict()) != prior_sha:
        raise RuntimeError('Frozen prior changed')
    return common._load(path)


def evaluate_point(models, ae, validation, stats, args, point):
    results = {}
    specs = [('prior', models['audio'].prior, False, 'real'),
             ('audio', models['audio'], True, 'real'),
             ('matched_static', models['matched_static'], True, 'static'),
             *[(mode, models['audio'], True, mode) for mode in ('static', 'reverse', 'mismatch')]]
    for name, model, local, intervention in specs:
        common._write(args.output/'status.json', {'state': 'evaluating', 'arm': name,
                      'point': str(point), 'clips': len(validation), 'updated': time.time()})
        destination = point/name
        if all((destination/name).exists() for name in ('result.json', 'curves.pt', 'arkit_benchmark.json')):
            results[name] = json.loads((destination/'result.json').read_text(encoding='utf8'))
            continue
        if destination.exists() and any(destination.iterdir()):
            index = 1
            while (point/(name+f'_interrupted_{index}')).exists(): index += 1
            recovered = point/(name+f'_interrupted_{index}')
            if destination.resolve().parent != point.resolve() or recovered.resolve().parent != point.resolve():
                raise ValueError('Evaluation recovery escaped point directory')
            destination.rename(recovered)
        results[name] = evaluate_generation(model, ae, validation, stats, destination, args.device,
                          use_audio=local, intervention=intervention, seeds=SEEDS, steps=24)
    result = audit.audit(point/'prior', point/'audio', point/'static',
                         matched_static_dir=point/'matched_static')
    # Compare the same clips, masks, explicit noise draws and metrics. The
    # mismatch evaluator can exclude no-donor clips. Explicitly declare that
    # scope and compare the exact same subset; retain all clips in main arms.
    loaded = {name: audit._load(point/name, name) for name in results}
    for name in loaded:
        reference = loaded['prior']
        if name == 'mismatch':
            excluded = set(results[name]['mismatch_excluded_no_donor'])
            expected_ids = set(reference['rows'])-excluded
            if set(loaded[name]['rows']) != expected_ids:
                raise ValueError('Mismatch subset differs from declared no-donor exclusions')
            reference = {**reference, 'rows': {cid:r for cid,r in reference['rows'].items() if cid in expected_ids}}
        audit._match(reference, loaded[name], name)
    ids = sorted(loaded['audio']['rows']); comparisons = {}
    for control in ('prior', 'matched_static', 'static', 'reverse', 'mismatch'):
        comparisons[control] = {}
        for metric in ('centered', 'variogram'):
            def value(row):
                m = row['metric']
                return m['joint_fair_es']['centered'] if metric == 'centered' else m['variogram']['aggregate']
            supported = [cid for cid in ids if cid in loaded[control]['rows'] and value(loaded['audio']['rows'][cid]) is not None
                         and value(loaded[control]['rows'][cid]) is not None]
            comparisons[control][metric] = audit.sentence_cluster_bootstrap(
                [value(loaded['audio']['rows'][cid])-value(loaded[control]['rows'][cid]) for cid in supported],
                [str(loaded['audio']['rows'][cid]['metadata']['sentence']) for cid in supported])
    reports = {name: json.loads((point/name/'arkit_benchmark.json').read_text(encoding='utf8'))['summary']
               for name in results}
    quality = result['metrics']; a, p = quality['arms']['audio'], quality['arms']['prior']
    protection = {
        'numerics_and_other43': all(r['numerical_gate']['passed'] for r in results.values()),
        'raw_es': a['joint_fair_es']['raw'] <= 1.02*p['joint_fair_es']['raw'],
        'diversity': quality['spread_ratio_to_prior'] is not None and quality['spread_ratio_to_prior'] >= .8,
        'speed': a['speed_ratio_to_reference'] is not None and .5 <= a['speed_ratio_to_reference'] <= 1.5,
        'group_amplitude': all(x is not None and .5 <= x <= 1.5 for x in a['rms_ratio']),
        'raw_oob': all(x <= y+.005 for x, y in zip(a['raw_oob'], p['raw_oob'])),
        'mbe': reports['audio']['arkit_mbe']['value'] <= 1.02*reports['prior']['arkit_mbe']['value'],
        'lbe_exact': reports['audio']['arkit_lbe']['value'] == reports['prior']['arkit_lbe']['value'],
    }
    screening = all(protection.values()) and all(
        c['centered']['mean_delta'] < 0 and c['variogram']['mean_delta'] <= 0 for c in comparisons.values())
    confirmed = screening and all(c['centered']['ci95'][1] < 0 for c in comparisons.values())
    result['original_adapter_gate'] = {k: result[k] for k in ('accepted', 'failures', 'gate_contract')}
    result['failures'] = ([name+' protection failed' for name, passed in protection.items() if not passed]
        + [control+' centered ES CI does not establish improvement' for control, c in comparisons.items()
           if c['centered']['ci95'][1] >= 0]
        + [control+' variogram does not improve' for control, c in comparisons.items()
           if c['variogram']['mean_delta'] > 0])
    result['gate_contract'] = {'protections_required': list(protection),
        'short_screen': 'negative centered-ES mean and nonpositive variogram mean vs all five controls',
        'final': 'short screen plus negative sentence-cluster centered-ES CI upper vs all five controls'}
    result.update(schema=SCHEMA, comparisons=comparisons, protections=protection,
                  mismatch_excluded_no_donor=results['mismatch']['mismatch_excluded_no_donor'],
                  continue_long_training=screening, accepted=confirmed, arkit=reports,
                  default_replaced=False, naturalness_certified=False,
                  acceptance_note='Short screening uses paired mean signs; final evidence also requires centered-ES sentence CI below zero for every control. Neither certifies perception.')
    common._write(point/'acceptance.json', result)
    return result


def run(args):
    if not (0 < args.pilot_steps <= args.final_steps and args.batch_size > 0):
        raise ValueError('Require positive increasing budgets')
    if args.smoke and args.final_steps > 100:
        raise ValueError('Smoke budget exceeds 100')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a fresh output or explicit --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < 1024**3:
        raise RuntimeError('At least 1 GiB available space required')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    source = json.loads((args.source_run/'protocol.json').read_text(encoding='utf8'))
    dataset_sha = common._file_sha(args.dataset)
    if source['dataset_sha256'] != dataset_sha:
        raise ValueError('Source dataset differs')
    payload = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('schema') != 'continuous_motion_dataset_v1': raise ValueError('Wrong dataset schema')
    fit, validation = split_train_pool(payload['clips'], source)
    validation = [{**c, 'split': 'inner_validation'} for c in validation]
    stats = common._load(args.source_run/'fit_stats.pt')
    ae_ck, prior_ck = [common._load(args.source_run/name) for name in ('ae_final.pt', 'prior_final.pt')]
    if set(stats['train_clip_ids']) != {c['clip_id'] for c in fit}:
        raise ValueError('Source fit statistics membership differs')
    if any(ck['binding']['protocol_sha256'] != common._value_sha(source) for ck in (ae_ck, prior_ck)):
        raise ValueError('Source checkpoint protocol differs')
    if (prior_ck['binding']['ae_sha256'] != common._value_sha(ae_ck['state']) or
            prior_ck['binding']['stats_sha256'] != common._value_sha(stats)):
        raise ValueError('Source AE/statistics checkpoint binding differs')
    ae = ContinuousUpperAE(**ae_ck['config']).to(args.device)
    ae.load_state_dict(ae_ck['state']); ae.eval().requires_grad_(False)
    prior = ContinuousLatentFlow(**prior_ck['config']).to(args.device)
    prior.load_state_dict(prior_ck['state']); prior.eval().requires_grad_(False)
    real, static, coverage = prepare_training(fit, stats, ae, args.device)
    if not args.smoke and (len(fit) != 613 or len(validation) != 206 or
                          len(coverage['retained_clip_ids']) != 613 or coverage['totals']['kept_frames'] != 66272):
        raise ValueError('Expected full support: 613 fit, 66272 frames, 206 validation')
    if args.smoke: validation = smoke_population(validation)
    _, _, missing_donors = _mismatched_audio(validation)
    bank = DonorBank(fit, stats)
    torch.manual_seed(args.seed)
    initial = MultiScalePriorAudioResidual(prior).to(args.device)
    root = Path(__file__).resolve().parents[1]
    protocol = {'schema': SCHEMA, 'seed': args.seed, 'smoke': args.smoke,
        'source_run': str(args.source_run.resolve()), 'dataset_sha256': dataset_sha,
        'source_files': {name: common._file_sha(args.source_run/name) for name in
                         ('protocol.json', 'ae_final.pt', 'prior_final.pt', 'fit_stats.pt')},
        'source_sha256': {p: common._file_sha(root/p) for p in SOURCES},
        'model_config': initial.config, 'budgets': [args.pilot_steps, args.final_steps],
        'batch_size': args.batch_size, 'optimizer': {'name': 'AdamW', 'lr': .0003, 'weight_decay': .0001},
        'loss': {'real_fm': 1., 'swap_margin': .001, 'swap_weight': .25,
                 'null_preservation': 'exact architecture H(audio)-H(zero); no redundant null loss'},
        'coverage': coverage, 'fit_ids': [c['clip_id'] for c in fit],
        'validation_ids': [c['clip_id'] for c in validation],
        'mismatch_excluded_no_donor': missing_donors,
        'mismatch_scope': 'same-emotion different-sentence eligible subset; all clips retained in other arms',
        'inference': 'native audio runs, frozen global/identity context; no target/motion mask',
        'scope': 'historically exposed internal sentence development; no outer evaluation',
        'selection': 'fixed pilot then conditional fixed final; no epoch sweep',
        'seeds': list(SEEDS), 'flow_steps': 24, 'default_replaced': False}
    if (args.output/'protocol.json').exists():
        if json.loads((args.output/'protocol.json').read_text(encoding='utf8')) != protocol:
            raise ValueError('Resume protocol differs')
    else:
        common._write(args.output/'protocol.json', protocol)
        for p in SOURCES:
            dest = args.output/'source'/p; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/p, dest)
    binding = {'protocol_sha256': common._file_sha(args.output/'protocol.json')}
    frozen = (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict()), common._value_sha(stats))
    points = sorted(set((args.pilot_steps, args.final_steps)))
    for step in points:
        point = args.output/f'point_{step}'
        if (point/'acceptance.json').exists():
            decision = json.loads((point/'acceptance.json').read_text(encoding='utf8'))
        else:
            models = {}; checkpoints = {}
            for arm, items in [('audio', real), ('matched_static', static)]:
                model = copy.deepcopy(initial)
                ck = train_arm(model, items, stats, bank, args, arm, step, binding)
                models[arm] = model; checkpoints[arm] = ck
                point.mkdir(exist_ok=True)
                common._save(point/(arm+'.pt'), {k: v for k, v in ck.items() if k not in ('optimizer', 'losses', 'batch_rng')})
            pair_keys = ('initial_sha256', 'order_sha256', 'donor_sha256', 'step', 'donor_counts')
            pairing = {k: checkpoints['audio'][k] == checkpoints['matched_static'][k] for k in pair_keys}
            if not all(pairing.values()): raise RuntimeError('Matched training draws differ')
            common._write(point/'pairing.json', pairing)
            if frozen != (common._value_sha(ae.state_dict()), common._value_sha(prior.state_dict()), common._value_sha(stats)):
                raise RuntimeError('Frozen source or statistics changed')
            decision = evaluate_point(models, ae, validation, stats, args, point)
        if not decision['continue_long_training']:
            break
    common._write(args.output/'status.json', {'schema': SCHEMA, 'state': 'complete',
        'last_endpoint': step, 'planned_final': args.final_steps,
        'long_training_stopped_by_gate': step < args.final_steps,
        'accepted': decision['accepted'] and not args.smoke, 'smoke': args.smoke,
        'default_replaced': False, 'naturalness_certified': False, 'updated': time.time()})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('dataset', 'source-run', 'output'): p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--pilot-steps', type=int, default=1000)
    p.add_argument('--final-steps', type=int, default=6000); p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--resume', action='store_true'); p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try: run(args)
    except Exception as exc:
        common._write(args.output/'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'updated': time.time()})
        raise
