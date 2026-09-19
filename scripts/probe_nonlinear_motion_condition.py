"""Fixed-budget nonlinear follow-up to the audited linear condition probe.

Uses the same fit/validation clips, windows, descriptors and equal-clip risk.
No outer clip tensors enter fitting or model selection. Not a generator.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts import probe_motion_condition_predictability as base

SEEDS = (42, 123, 2026)


class Predictor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net(x)


def fit(x, y, weight, *, seed, steps, device):
    mean, scale = base.weighted_moments(x, weight, .01)
    torch.manual_seed(seed)
    model = Predictor(x.shape[1]).to(device)
    values = torch.as_tensor((x-mean)/scale, dtype=torch.float32, device=device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    probability = torch.as_tensor(weight/weight.sum(), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(seed+10000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    history = []
    for step in range(steps):
        indices = torch.multinomial(probability, 512, replacement=True, generator=generator)
        loss = (model(values[indices])-targets[indices]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fitting loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step+1) % 100 == 0 or step == 0:
            history.append({'step': step+1, 'loss': float(loss.detach())})
    model.eval()
    return {'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'input_mean': mean, 'input_scale': scale, 'history': history,
            'seed': seed, 'steps': steps, 'parameter_count': sum(p.numel() for p in model.parameters())}


def predict(checkpoint, x, device='cpu'):
    model = Predictor(x.shape[1]).to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    values = torch.as_tensor((x-checkpoint['input_mean'])/checkpoint['input_scale'],
                             dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(values).cpu().numpy().astype(np.float64)


def score(rows, models, target_stats, device):
    scores = {arm: [] for arm in base.ARMS}
    predictions = []
    for row in rows:
        truth = (row['y']-target_stats['mean'])/target_stats['scale']
        preds = {'train_mean': np.zeros_like(truth),
                 'matched_static': predict(models['matched_static'], row['static'], device),
                 'audio': predict(models['audio'], row['real'], device),
                 'audio_static': predict(models['audio'], row['static'], device),
                 'audio_reverse': predict(models['audio'], row['reverse'], device)}
        for arm in base.ARMS:
            scores[arm].append({'clip_id': row['clip_id'], 'sentence': row['sentence'],
                               'windows': len(truth), **base.clip_scores(preds[arm], truth)})
        predictions.append({'clip_id': row['clip_id'], 'locations': row['locations'],
                            'truth_normalized': truth, 'predictions_normalized': preds})
    return {arm: {'clip_equal': base.aggregate(scores[arm]),
                  'sentence_equal': base.aggregate(scores[arm], sentence_equal=True),
                  'clips': scores[arm]} for arm in base.ARMS}, predictions


def run(args):
    torch.set_num_threads(4)
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    reference = json.loads(args.reference_protocol.read_text(encoding='utf8'))
    if base.sha(args.dataset) != reference['dataset_sha256']:
        raise ValueError('Dataset differs from fixed protocol')
    data = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    if data['schema'] != base.DATASET_SCHEMA:
        raise ValueError('Wrong dataset schema')
    train, valid = base.split_train_pool(data['clips'], reference)
    del data
    args.output.mkdir(parents=True)
    protocol = {'schema': 'nonlinear_motion_condition_probe_v1', 'seeds': SEEDS,
                'steps_per_model': args.steps, 'hidden': [64, 64], 'batch_size': 512,
                'learning_rate': .0003, 'weight_decay': .01, 'dropout': 0,
                'dataset_sha256': reference['dataset_sha256'],
                'reference_protocol_sha256': base.sha(args.reference_protocol),
                'source_sha256': base.sha(__file__), 'base_source_sha256': base.sha(base.__file__),
                'train_ids': [c['clip_id'] for c in train], 'validation_ids': [c['clip_id'] for c in valid],
                'windows': base.WINDOWS, 'targets': base.TARGET_NAMES,
                'input': 'Same projected audio window descriptors/context as linear probe; no time position input.',
                'normalization': 'Fit only, independently per arm; scale floor .01; target scale floor .001',
                'training': 'Equal clips then equal supported windows via weighted sampling; fixed final step only.',
                'control': 'Same architecture, seed, batch-index draws, budget; local audio descriptors zero for static.',
                'scope': 'Development follow-up selected after failed linear probe; no outer tensor indexing; no generator changed.',
                'preregistered_integration_screen': 'All three seeds must have mean audio-minus-static MSE <0 and its sentence bootstrap upper bound <0; audio must also beat its own reversed input. Passing motivates separate generator test, never automatic default replacement.',
                'device': args.device}
    base.write_json(args.output/'protocol.json', protocol)
    (args.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    stats = base.audio_statistics(train)
    results = {str(seed): {} for seed in SEEDS}
    checkpoints, predictions = {}, {}
    for window in base.WINDOWS:
        rows, coverage = base.make_rows(train, stats, window)
        validation, vc = base.make_rows(valid, stats, window)
        y, weights = base.concatenate(rows, 'y')
        mean, scale = base.weighted_moments(y, weights, .001)
        target_stats = {'mean': mean, 'scale': scale}
        y = (y-mean)/scale
        for seed in SEEDS:
            models = {}
            for arm, key in [('audio', 'real'), ('matched_static', 'static')]:
                x, w = base.concatenate(rows, key)
                np.testing.assert_array_equal(w, weights)
                models[arm] = fit(x, y, weights, seed=seed, steps=args.steps, device=args.device)
            result, pred = score(validation, models, target_stats, args.device)
            results[str(seed)][str(window)] = {'inner_validation': result, 'coverage': vc}
            key = f'{seed}_{window}'
            checkpoints[key] = {'models': models, 'target_stats': target_stats}
            predictions[key] = pred
            print(json.dumps({'seed': seed, 'window': window, 'mse': {
                a: result[a]['sentence_equal']['mean_normalized_mse'] for a in base.ARMS}}), flush=True)
    comparisons = {s: {c: base.primary_comparison(r, c) for c in ('matched_static', 'audio_reverse')}
                   for s, r in results.items()}
    passed = all(v['matched_static']['all']['exploratory_sentence_bootstrap_95ci'][1] < 0
                 and v['audio_reverse']['all']['audio_minus_audio_reverse'] < 0
                 for v in comparisons.values())
    report = {'protocol_sha256': base.sha(args.output/'protocol.json'), 'results': results,
              'comparisons': comparisons, 'integration_screen_passed': passed,
              'generator_modified': False, 'default_replaced': False,
              'elapsed_seconds': time.monotonic()-started}
    base.write_json(args.output/'report.json', report)
    torch.save({'audio_statistics': stats, 'models': checkpoints, 'protocol': protocol}, args.output/'models.pt')
    torch.save(predictions, args.output/'predictions.pt')
    base.write_json(args.output/'manifest.json', {p.name: base.sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({'complete': True, 'integration_screen_passed': passed,
                      'seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'reference-protocol', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
