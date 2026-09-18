"""Fixed-clock four-group activity predictability before motion integration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import activity_condition_core as core
from scripts import train_clocked_motion_prior as c

SCHEMA = 'four_group_activity_condition_v1'
SEEDS = (20260918, 20260919, 20260920)
ARMS = ('static_trained', 'real', 'static', 'reverse', 'shift', 'mismatch')
CELLS = ('fit', 'calibration', 'confirmation')
GROUPS = ('brow_up', 'brow_down', 'eye_squint', 'eye_wide')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def metadata(clips, ids):
    return [{k: clips[i][k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')} for i in ids]


def prepare_targets(clips, ids):
    result = {}
    for i in ids:
        item = clips[i]
        locations = core.window_locations(item['valid'])
        if not locations:
            continue
        energy = np.stack([core.activity_energy(item['upper'][start:start+count]) for start, count in locations])
        raw_energy = []
        for start, _ in locations:
            delta = np.diff(item['upper'][start+11:start+20], axis=0)*25.
            raw_energy.append([float(np.sqrt(np.mean(delta[:, group]**2))) for group in core.GROUPS])
        result[i] = {'starts': [start for start, _ in locations], 'energy': energy,
                     'raw_energy': np.asarray(raw_energy), 'coverage': core.window_coverage(item['valid'])}
    return result


def fit_statistics(clips, ids, targets):
    ids = [i for i in ids if i in targets]
    if not ids:
        raise ValueError('No fitting clips with a complete H32 window')
    energies = np.concatenate([targets[i]['energy'] for i in ids])
    weights = np.concatenate([np.full(len(targets[i]['starts']), 1./len(targets[i]['starts'])) for i in ids])
    thresholds = core.fit_thresholds(energies, weights)
    static = torch.stack([torch.cat((torch.as_tensor(clips[i]['global']),
        core.prosody_mean(clips[i]['features'], clips[i]['valid']))) for i in ids]).double()
    acoustic = torch.cat([clips[i]['features'][clips[i]['valid']][:, 1536:1540] for i in ids]).double()
    return {'thresholds': thresholds, 'static_mean': static.mean(0).float(),
            'static_std': static.std(0, unbiased=False).clamp_min(.01).float(),
            'prosody_std': acoustic.std(0, unbiased=False).clamp_min(.01).float(),
            'fit_clip_ids': [clips[i]['clip_id'] for i in ids]}


def inputs(item, starts, stats, mode='real', donor=None):
    """Only acoustics/global/valid are read; no motion/energy/labels used."""
    mean = core.prosody_mean(item['features'], item['valid'])
    features = core.intervene_prosody(item['features'], item['valid'], mode,
        donor_features=None if donor is None else donor['features'],
        donor_valid=None if donor is None else donor['valid'])
    windows = torch.stack([features[start:start+32] for start in starts])
    descriptor = core.prosody_descriptor(windows, mean, stats['prosody_std'])
    context = (torch.cat((torch.as_tensor(item['global']), mean))-stats['static_mean'])/stats['static_std']
    return context[None].expand(len(starts), -1).float(), descriptor.float()


def training_data(clips, ids, targets, stats):
    arrays = {k: [] for k in ('context', 'descriptor', 'target', 'weight')}
    for i in ids:
        if i not in targets:
            continue
        target = targets[i]; count = len(target['starts'])
        context, descriptor = inputs(clips[i], target['starts'], stats)
        arrays['context'].append(context); arrays['descriptor'].append(descriptor)
        arrays['target'].append(torch.as_tensor(core.activity_targets(target['energy'], stats['thresholds']), dtype=torch.float32))
        arrays['weight'].append(torch.full((count,), 1./count))
    data = {k: torch.cat(v) for k, v in arrays.items()}
    data['weight'] /= data['weight'].mean()
    return data


def train_pair(data, seed, epochs, output, device):
    torch.manual_seed(seed)
    base = core.StaticActivityHead().to(device)
    tensors = {k: v.to(device) for k, v in data.items()}
    histories, orders = {}, {}
    for arm in ('static', 'temporal'):
        if arm == 'static':
            model = base
            parameters = list(base.parameters())
        else:
            before = {k: v.detach().clone() for k, v in base.state_dict().items()}
            torch.manual_seed(seed)
            model = core.FrozenActivityResidual(base).to(device)
            parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.01)
        rng = np.random.default_rng(seed); rows = []; order_hash = hashlib.sha256()
        for epoch in range(epochs):
            model.train()
            order = rng.permutation(len(tensors['target'])); order_hash.update(order.tobytes())
            total, count = 0., 0
            for start in range(0, len(order), 256):
                ix = torch.as_tensor(order[start:start+256], device=device)
                logits = model(tensors['context'][ix]) if arm == 'static' else model(
                    tensors['context'][ix], tensors['descriptor'][ix])
                score = (logits.sigmoid()-tensors['target'][ix]).square().mean(-1)
                loss = (score*tensors['weight'][ix]).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
                total += float(loss.detach())*len(ix); count += len(ix)
            rows.append({'epoch': epoch+1, 'brier': total/count, 'updates': (len(order)+255)//256})
            c.old.save_json(output/f'{seed}_{arm}_losses.json', rows)
            c.old.save_json(output/'status.json', {'schema': SCHEMA, 'status': 'training',
                'seed': seed, 'arm': arm, 'epoch': epoch+1, 'epochs': epochs})
        histories[arm] = rows; orders[arm] = order_hash.hexdigest()
        model.eval()
        torch.save({'schema': SCHEMA, 'seed': seed, 'arm': arm, 'epochs': epochs,
            'state': model.state_dict(), 'order_sha256': orders[arm], 'optimizer': optimizer.state_dict()},
            output/f'{seed}_{arm}.pt')
        print('TRAINED', seed, arm, rows[-1]['brier'], flush=True)
    if orders['static'] != orders['temporal']:
        raise RuntimeError('Matched training orders differ')
    if any(not torch.equal(v, before[k]) for k, v in model.base.state_dict().items()):
        raise RuntimeError('Frozen base changed')
    if any(p.grad is not None or p.requires_grad for p in model.base.parameters()):
        raise RuntimeError('Static base received gradients')
    return model.eval(), {'order_sha256': orders['static'], 'base_unchanged_exact': True,
                         'trainable_temporal_parameters': sum(p.numel() for p in parameters)}


def donor_map(clips, ids, targets):
    result = {}
    for i in ids:
        if i not in targets:
            continue
        candidates = [j for j in ids if j in targets and clips[j]['speaker'] == clips[i]['speaker']
            and clips[j]['emotion'] == clips[i]['emotion'] and clips[j]['sentence'] != clips[i]['sentence']
            and len(c.old.runs(clips[j]['valid'].numpy())) == 1]
        if candidates:
            result[i] = min(candidates, key=lambda j: clips[j]['clip_id'])
    return result


@torch.no_grad()
def evaluate(clips, ids, targets, stats, models, device):
    donors = donor_map(clips, ids, targets); rows = []
    for i in ids:
        if i not in targets:
            continue
        item, target = clips[i], targets[i]
        row = {k: item[k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')}
        row.update(target, target=core.activity_targets(target['energy'], stats['thresholds']),
            donor_id=clips[donors[i]]['clip_id'] if i in donors else None, probabilities={})
        for arm in ARMS:
            if arm == 'mismatch' and i not in donors:
                continue
            context, descriptor = inputs(item, target['starts'], stats,
                mode='static' if arm == 'static_trained' else arm, donor=clips[donors[i]] if arm == 'mismatch' else None)
            context, descriptor = context.to(device), descriptor.to(device)
            values = []
            for model in models:
                logits = model.base(context) if arm == 'static_trained' else model(context, descriptor)
                values.append(logits.sigmoid().cpu().numpy())
            row['probabilities'][arm] = np.stack(values)
        if not np.array_equal(row['probabilities']['static_trained'], row['probabilities']['static']):
            raise RuntimeError('Static intervention must exactly equal frozen static baseline')
        rows.append(row)
    return rows


def score_rows(rows, arm, seed_index=None):
    scored = []
    for row in rows:
        if arm not in row['probabilities']:
            continue
        samples = np.asarray(row['probabilities'][arm], dtype=np.float64)
        p = samples.mean(0) if seed_index is None else samples[seed_index]
        y = np.asarray(row['target'], dtype=np.float64)
        brier = ((p-y)**2).mean(0)
        q = np.clip(p, 1e-7, 1-1e-7)
        bce = -(y*np.log(q)+(1-y)*np.log1p(-q)).mean(0)
        scored.append({**{k: row[k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')},
            'brier': brier.tolist(), 'bce': bce.tolist(), 'prevalence': y.mean(0).tolist()})
    if not scored:
        return {'clips': 0, 'rows': [], 'brier': None, 'group_brier': None}
    matrix = np.asarray([row['brier'] for row in scored])
    return {'clips': len(scored), 'rows': scored, 'brier': float(matrix.mean()),
        'group_brier': matrix.mean(0).tolist(),
        'bce': float(np.mean([row['bce'] for row in scored])),
        'prevalence': np.mean([row['prevalence'] for row in scored], axis=0).tolist()}


def bootstrap_gain(real, control, groups=(0, 1, 2, 3)):
    if [r['clip_id'] for r in real['rows']] != [r['clip_id'] for r in control['rows']]:
        raise ValueError('Paired metadata differ')
    diff = np.asarray([r['brier'] for r in control['rows']])-np.asarray([r['brier'] for r in real['rows']])
    values = diff[:, list(groups)].mean(-1)
    sentences = sorted({row['sentence'] for row in real['rows']})
    clusters = [np.where([row['sentence'] == sentence for row in real['rows']])[0] for sentence in sentences]
    rng = np.random.default_rng(20260918)
    samples = [float(values[np.concatenate([clusters[j] for j in rng.integers(len(clusters), size=len(clusters))])].mean())
        for _ in range(2000)]
    return {'gain': float(values.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist()}


def assessment(rows):
    common = [row for row in rows if all(arm in row['probabilities'] for arm in ARMS)]
    all_scores = {arm: score_rows(rows, arm) for arm in ARMS}
    result = {'all': all_scores, 'support': len(common),
        'support_fraction': len(common)/max(len(rows), 1),
        'support_sentences': len({row['sentence'] for row in common}), 'development_only': True}
    if not common:
        return {**result, 'passed': False, 'reason': 'no_common_support'}
    scores = {arm: score_rows(common, arm) for arm in ARMS}
    gains = {arm: bootstrap_gain(scores['real'], scores[arm]) for arm in ARMS if arm != 'real'}
    brow = bootstrap_gain(scores['real'], scores['static_trained'], (0, 1))
    base = scores['static_trained']; real = scores['real']
    per_seed = []
    for seed_index in range(len(common[0]['probabilities']['real'])):
        b = score_rows(common, 'static_trained', seed_index)
        r = score_rows(common, 'real', seed_index)
        per_seed.append({'seed_index': seed_index, 'static_brier': b['brier'], 'real_brier': r['brier'],
                         'gain': b['brier']-r['brier']})
    checks = {'support': result['support_fraction'] >= .8 and result['support_sentences'] >= 8,
        'static_gain': gains['static_trained']['gain'] >= .01*base['brier'] and gains['static_trained']['ci95'][0] > 0,
        'interventions': all(gains[arm]['ci95'][0] > 0 for arm in ('reverse', 'shift', 'mismatch')),
        'brow_gain': brow['gain'] >= .01*np.mean(base['group_brier'][:2]) and brow['ci95'][0] > 0,
        'group_protection': bool(np.all(np.asarray(real['group_brier']) <= 1.02*np.asarray(base['group_brier']))),
        'nondegenerate_labels': bool(np.all((np.asarray(real['prevalence']) >= .05) & (np.asarray(real['prevalence']) <= .95))),
        'training_seed_consistency': all(row['gain'] > 0 for row in per_seed)}
    slices = {}
    for key in ('emotion', 'speaker', 'sentence'):
        slices[key] = {str(value): {arm: score_rows([row for row in common if row[key] == value], arm)
            for arm in ('static_trained', 'real')} for value in sorted({row[key] for row in common})}
    checks = {key: bool(value) for key, value in checks.items()}
    return {**result, 'paired': scores, 'gains': gains, 'brow_gain': brow, 'per_seed': per_seed,
            'slices': slices, 'checks': checks, 'passed': all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir', 'audio-checkpoint', 'source-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    c.old.save_json(args.output/'status.json', {'schema': SCHEMA, 'status': 'loading'})
    source = read(args.source_run/'protocol.json'); source_status = read(args.source_run/'status.json')
    if source.get('schema') != 'clocked_bounded_static_residual_v1' or source_status.get('status') != 'complete' or source_status.get('smoke') is not False:
        raise ValueError('Completed formal bounded source required for pinned lineage')
    root = Path(__file__).resolve().parents[1]
    for relative, digest in source['code_sha256'].items():
        if c.old.sha(root/relative) != digest:
            raise ValueError('Source dependency differs: '+relative)
    loading = copy.copy(args); loading.smoke = False
    clips, original, lineage = c.old.load_clips(loading)
    split = c.previous.split_inner(clips, original)
    if lineage != source['source'] or any(metadata(clips, split[cell]) != source['split'][cell] for cell in CELLS):
        raise ValueError('Source lineage/membership differs')
    split = {cell: split[cell][:12] if args.smoke else split[cell] for cell in CELLS}
    ids = sorted(set(i for values in split.values() for i in values))
    frozen = c.load_frozen_audio(args.audio_checkpoint, args.device)
    binding = c.assert_source_binding(frozen, lineage)
    if binding != source['frozen_audio']:
        raise ValueError('Frozen audio binding differs')
    contexts = c.encode_clips(frozen, [clips[i] for i in ids], device=args.device)
    for i, context in zip(ids, contexts):
        clips[i]['global'] = np.r_[context['global'].numpy(), context['intensity'].numpy()]
    del frozen, contexts
    seeds = SEEDS[:1] if args.smoke else SEEDS; epochs = 2 if args.smoke else 30
    files = sorted(set(source['code_sha256']) | {'scripts/activity_condition_core.py', 'scripts/train_activity_condition.py'})
    protocol_file = root/'docs/ACTIVITY_CONDITION_PROTOCOL_20260918.md'
    protocol = {'schema': SCHEMA, 'source': lineage, 'frozen_audio': binding,
        'source_protocol_sha256': c.old.sha(args.source_run/'protocol.json'), 'smoke': args.smoke,
        'split': {cell: metadata(clips, split[cell]) for cell in CELLS},
        'epochs_each': epochs, 'seeds': list(seeds), 'test_loaded': False, 'development_only': True,
        'code_sha256': {f: c.old.sha(root/f) for f in files}, 'protocol_sha256': c.old.sha(protocol_file),
        'generator_integrated': False, 'default_replaced': False, 'objective': 'clip-balanced Brier',
        'groups': list(GROUPS), 'horizon': 32, 'hop': 8, 'scale_fixed': 1.}
    c.old.save_json(args.output/'protocol.json', protocol)
    for relative in files:
        destination = args.output/'source'/relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, destination)
    shutil.copyfile(protocol_file, args.output/'source/protocol.md')
    targets = prepare_targets(clips, ids)
    stats = fit_statistics(clips, split['fit'], targets)
    torch.save(stats, args.output/'statistics.pt')
    c.old.save_json(args.output/'thresholds.json', {'thresholds': stats['thresholds'].tolist(),
        'fit_clip_ids': stats['fit_clip_ids'], 'quantile': .65, 'weights': 'equal clip, equal window within clip',
        'threshold_floor_reached': (stats['thresholds'] <= core.THRESHOLD_FLOOR).tolist()})
    diagnostics = {}
    for cell in CELLS:
        available = [i for i in split[cell] if i in targets]
        diagnostics[cell] = {'source_clips': len(split[cell]), 'scored_clips': len(available),
            'excluded': [clips[i]['clip_id'] for i in split[cell] if i not in targets],
            'windows': sum(len(targets[i]['starts']) for i in available),
            'prevalence': np.mean([core.activity_targets(targets[i]['energy'], stats['thresholds']).mean(0) for i in available], axis=0).tolist(),
            'mean_energy': np.mean([targets[i]['energy'].mean(0) for i in available], axis=0).tolist(),
            'raw_mean_energy': np.mean([targets[i]['raw_energy'].mean(0) for i in available], axis=0).tolist(),
            'zero_energy_clip_fraction': np.mean([np.all(targets[i]['energy'] <= 1e-8, axis=0) for i in available], axis=0).tolist(),
            'coverage': {clips[i]['clip_id']: core.window_coverage(clips[i]['valid']) for i in split[cell]}}
    c.old.save_json(args.output/'target_diagnostics.json', diagnostics)
    data = training_data(clips, split['fit'], targets, stats); models = []; matching = {}
    for seed in seeds:
        model, match = train_pair(data, seed, epochs, args.output, args.device)
        models.append(model); matching[str(seed)] = match
    c.old.save_json(args.output/'matching.json', matching)
    reports, predictions = {}, {}
    for cell in CELLS:
        predictions[cell] = evaluate(clips, split[cell], targets, stats, models, args.device)
        reports[cell] = assessment(predictions[cell])
        print('ASSESSMENT', cell, json.dumps({k: reports[cell][k] for k in ('passed', 'support', 'support_fraction')}), flush=True)
    torch.save({'schema': SCHEMA, 'seeds': list(seeds), 'cells': predictions}, args.output/'predictions.pt')
    c.old.save_json(args.output/'reports.json', reports)
    passed = all(reports[cell]['passed'] for cell in ('calibration', 'confirmation')) and not args.smoke
    c.old.save_json(args.output/'status.json', {'schema': SCHEMA, 'status': 'complete',
        'seconds': time.monotonic()-started, 'smoke': args.smoke, 'epochs_each': epochs, 'seeds': list(seeds),
        'activity_passed': passed, 'development_only': True, 'test_loaded': False,
        'generator_integrated': False, 'default_replaced': False})


if __name__ == '__main__':
    main()
