"""Fixed, fit-only probe of coarse motion conditions from native audio.

This is an internal linear diagnostic, not a generator or a new acceptance gate.
The historical outer holdout is excluded before audio/motion tensors are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

SCHEMA = 'motion_condition_predictability_v1'
DATASET_SCHEMA = 'continuous_motion_dataset_v1'
SEED = 2026091907
FPS = 25
WINDOWS = (10, 25, 50)
GROUP_NAMES = ('up', 'down', 'squint', 'wide')
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
TARGET_NAMES = tuple('log_speed_' + g for g in GROUP_NAMES) + tuple('signed_change_' + g for g in GROUP_NAMES)
PROJECTION_DIM = 64
RIDGE = 1.0
ARMS = ('train_mean', 'matched_static', 'audio', 'audio_static', 'audio_reverse')


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def split_train_pool(clips, reference=None):
    """Same fixed hash split as bounded_audio_inner_v1; outer tensors untouched."""
    pool = [c for c in clips if c['split'] == 'train']
    sentences = sorted({c['sentence'] for c in pool}, key=lambda s: hashlib.sha256(
        ('bounded_audio_inner_v1:' + s).encode()).hexdigest())
    if len(sentences) <= 5:
        raise ValueError('Need more than five fit-pool sentences')
    held = set(sentences[:5])
    fit = [c for c in pool if c['sentence'] not in held]
    valid = [c for c in pool if c['sentence'] in held]
    if reference is not None:
        if reference.get('schema') != 'bounded_audio_experiment_v1':
            raise ValueError('Wrong reference protocol schema')
        if [c['clip_id'] for c in fit] != reference['inner_train_ids'] or [c['clip_id'] for c in valid] != reference['inner_validation_ids']:
            raise ValueError('Exact nested split membership/order mismatch')
        if sorted(held) != sorted(reference['validation_sentences']):
            raise ValueError('Reference sentence membership mismatch')
    return fit, valid


def native_runs(valid):
    valid = numpy(valid)
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError('Boolean native audio mask required')
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def locations(valid, window):
    """Fixed run-start phase, full nonoverlapping windows, no padding or gaps."""
    if window not in WINDOWS:
        raise ValueError('Unsupported fixed window')
    return [(int(a), int(a + window), int(left), int(right))
            for left, right in native_runs(valid) for a in range(left, right - window + 1, window)]


def audio_statistics(fit):
    first, second = [], []
    for clip in fit:
        raw = numpy(clip['features']).astype(np.float64)
        valid = numpy(clip['valid'])
        native_runs(valid)
        if raw.ndim != 2 or raw.shape != (len(valid), 1540) or not valid.any() or not np.isfinite(raw[valid]).all():
            raise ValueError('Finite [T,1540] native-valid features required')
        x = raw[valid]
        first.append(x.mean(0)); second.append(np.square(x).mean(0))
    if not first:
        raise ValueError('Nonempty fitting clips required')
    mean = np.mean(first, 0)
    scale = np.sqrt(np.maximum(np.mean(second, 0) - mean ** 2, 0)).clip(.01)
    rng = np.random.default_rng(SEED)
    projection = rng.normal(size=(1536, PROJECTION_DIM)) / np.sqrt(1536.)
    return {'audio_mean': mean, 'audio_scale': scale, 'projection': projection}


def projected_native_audio(clip, stats):
    raw = numpy(clip['features']).astype(np.float64)
    valid = numpy(clip['valid'])
    native_runs(valid)
    if raw.shape != (len(valid), 1540) or not np.isfinite(raw[valid]).all():
        raise ValueError('Invalid native audio')
    normalized = np.zeros_like(raw)
    normalized[valid] = (raw[valid] - stats['audio_mean']) / stats['audio_scale']
    projected = np.c_[normalized[:, :1536] @ stats['projection'], normalized[:, 1536:]]
    return projected


def audio_windows(clip, projected, window, intervention='real'):
    """Input extraction never reads motion/motion masks or clips the time axis."""
    if intervention not in ('real', 'static', 'reverse'):
        raise ValueError('Unknown audio intervention')
    context = numpy(clip['context']).astype(np.float64).reshape(-1)
    if not np.isfinite(context).all() or projected.shape != (len(clip['valid']), PROJECTION_DIM + 4):
        raise ValueError('Finite context and projected native audio required')
    loc = locations(clip['valid'], window)
    run_audio = {}
    for left, right in native_runs(clip['valid']):
        raw = projected[left:right]
        run_audio[(int(left), int(right))] = raw[::-1] if intervention == 'reverse' else raw
    rows = []
    for left, right, run_left, run_right in loc:
        native = run_audio[(run_left, run_right)]
        run_mean = native.mean(0)
        if intervention == 'static':
            local = np.zeros(3 * projected.shape[1])
        else:
            block = native[left-run_left:right-run_left]
            quarter = max(1, window // 4)
            local = np.r_[block.mean(0)-run_mean, block.std(0), block[-quarter:].mean(0)-block[:quarter].mean(0)]
        rows.append(np.r_[context, run_mean, local])
    width = len(context) + 4 * projected.shape[1]
    return np.asarray(rows, dtype=np.float64).reshape(-1, width), loc


def motion_descriptor(motion):
    """Four log(1+speed) envelopes and four signed endpoint changes.

    Speed is raw coefficient/s after a within-window three-frame box filter.
    Signed change compares smoothed first/last quarter group means; no centering
    using the full target clip, target thresholds, or baseline-motion inputs.
    """
    motion = numpy(motion).astype(np.float64)
    if motion.ndim != 2 or motion.shape[1] != 9 or len(motion) not in WINDOWS or not np.isfinite(motion).all():
        raise ValueError('Finite full [fixed_window,9] motion required')
    smooth = (motion[:-2] + motion[1:-1] + motion[2:]) / 3.
    speed = np.diff(smooth, axis=0) * FPS
    quarter = max(1, len(smooth) // 4)
    energy = [np.log1p(np.sqrt(np.square(speed[:, list(g)]).mean())) for g in GROUPS]
    signed = [smooth[-quarter:, list(g)].mean() - smooth[:quarter, list(g)].mean() for g in GROUPS]
    return np.asarray(energy + signed)


def make_rows(clips, stats, window):
    """Build audio on every deployment window, then apply target eligibility."""
    rows, coverage = [], []
    for clip in clips:
        projected = projected_native_audio(clip, stats)
        inputs = {}; loc = None
        for arm in ('real', 'static', 'reverse'):
            inputs[arm], current = audio_windows(clip, projected, window, arm)
            if loc is not None and current != loc:
                raise RuntimeError('Intervention changes native window clock')
            loc = current
        motion = numpy(clip['motion9'])
        mask = numpy(clip['motion_mask'])
        if motion.shape != (len(clip['valid']), 9) or mask.shape != motion.shape or mask.dtype != np.bool_:
            raise ValueError('Expected native motion9 and Boolean observation mask')
        keep, truth = [], []
        for index, (left, right, _, _) in enumerate(loc):
            if mask[left:right].all():
                keep.append(index); truth.append(motion_descriptor(motion[left:right]))
        audio_frames = int(numpy(clip['valid']).sum())
        tail_frames = sum((right-left) % window for left, right in native_runs(clip['valid']))
        coverage.append({'clip_id': clip['clip_id'], 'native_valid_frames': audio_frames,
            'audio_windows': len(loc), 'scored_windows': len(keep),
            'omitted_audio_tail_frames': int(tail_frames),
            'unobserved_window_frames': (len(loc)-len(keep))*window})
        if keep:
            rows.append({'clip_id': clip['clip_id'], 'sentence': clip['sentence'],
                'locations': [loc[i] for i in keep], 'y': np.asarray(truth),
                **{arm: value[keep] for arm, value in inputs.items()}})
    if not rows:
        raise ValueError('No fully observed windows')
    return rows, coverage


def weighted_moments(values, weight, floor):
    w = np.asarray(weight, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or w.shape != (len(x),) or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Finite matrix with positive per-row weights required')
    w = w / w.sum()
    mean = w @ x
    scale = np.sqrt(w @ np.square(x-mean)).clip(floor)
    return mean, scale


def concatenate(rows, key):
    return np.concatenate([r[key] for r in rows]), np.concatenate([np.full(len(r[key]), 1. / len(r[key])) for r in rows])


def ridge_fit(x, y, weights):
    """Unpenalized intercept, fixed lambda on mean equal-clip empirical risk."""
    mean, scale = weighted_moments(x, weights, .01)
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    z = (x-mean) / scale
    intercept = w @ y
    lhs = z.T @ (w[:, None] * z)
    lhs.flat[::len(lhs)+1] += RIDGE
    coefficients = np.linalg.solve(lhs, z.T @ (w[:, None] * (y-intercept)))
    return {'input_mean': mean, 'input_scale': scale, 'coefficients': coefficients, 'intercept': intercept}


def ridge_predict(model, x):
    return ((x-model['input_mean']) / model['input_scale']) @ model['coefficients'] + model['intercept']


def clip_scores(pred, truth):
    error = np.square(pred-truth).mean(0)
    p = pred - pred.mean(0); t = truth - truth.mean(0)
    return {'mse': error.tolist(), 'train_mean_mse': np.square(truth).mean(0).tolist(),
            'centered_mse': np.square(p-t).mean(0).tolist(),
            'centered_truth_energy': np.square(t).mean(0).tolist(),
            'predicted_centered_energy': np.square(p).mean(0).tolist(),
            'centered_cross': (p*t).mean(0).tolist()}


def aggregate(scores, *, sentence_equal=False):
    fields = ('mse', 'train_mean_mse', 'centered_mse', 'centered_truth_energy', 'predicted_centered_energy', 'centered_cross')
    values = {}
    for field in fields:
        if sentence_equal:
            grouped = {}
            for row in scores:
                grouped.setdefault(row['sentence'], []).append(row[field])
            values[field] = np.mean([np.mean(v, 0) for v in grouped.values()], 0)
        else:
            values[field] = np.mean([s[field] for s in scores], 0)
    correlation = values['centered_cross'] / np.maximum(np.sqrt(values['predicted_centered_energy']*values['centered_truth_energy']), 1e-15)
    r2 = 1-values['mse']/np.maximum(values['train_mean_mse'], 1e-15)
    cr2 = 1-values['centered_mse']/np.maximum(values['centered_truth_energy'], 1e-15)
    return {**{k: v.tolist() for k, v in values.items()}, 'r2_vs_train_mean': r2.tolist(),
        'centered_r2_vs_static': cr2.tolist(), 'centered_correlation': correlation.tolist(),
        'mean_normalized_mse': float(values['mse'].mean()),
        'activity_normalized_mse': float(values['mse'][:4].mean()),
        'direction_normalized_mse': float(values['mse'][4:].mean())}


def score_rows(rows, models, target_stats):
    scores = {arm: [] for arm in ARMS}; predictions = []
    for row in rows:
        truth = (row['y']-target_stats['mean']) / target_stats['scale']
        pred = {'train_mean': np.zeros_like(truth),
            'matched_static': ridge_predict(models['matched_static'], row['static']),
            'audio': ridge_predict(models['audio'], row['real']),
            'audio_static': ridge_predict(models['audio'], row['static']),
            'audio_reverse': ridge_predict(models['audio'], row['reverse'])}
        for arm in ARMS:
            scores[arm].append({'clip_id': row['clip_id'], 'sentence': row['sentence'],
                               'windows': len(truth), **clip_scores(pred[arm], truth)})
        predictions.append({'clip_id': row['clip_id'], 'sentence': row['sentence'],
            'locations': row['locations'], 'truth_normalized': truth,
            'predictions_normalized': pred})
    return {arm: {'clip_equal': aggregate(scores[arm]), 'sentence_equal': aggregate(scores[arm], sentence_equal=True),
                  'clips': scores[arm]} for arm in ARMS}, predictions


def primary_comparison(results, comparator='matched_static', resamples=4096):
    """Equal sentence, equal fixed scale, equal target component primary score."""
    by_sentence = {}
    for window in WINDOWS:
        arms = results[str(window)]['inner_validation']
        reference = {r['clip_id']: r for r in arms[comparator]['clips']}
        for row in arms['audio']['clips']:
            other = reference[row['clip_id']]
            delta = np.asarray(row['mse'])-np.asarray(other['mse'])
            by_sentence.setdefault(row['sentence'], {}).setdefault(window, []).append(delta)
    # All predefined scales must be represented in each sentence; never drop a
    # difficult scale silently to improve the primary composite.
    if any(set(scales) != set(WINDOWS) for scales in by_sentence.values()):
        raise ValueError('Primary comparison missing a fixed sentence/scale cell')
    sentence_vectors = {s: np.mean([np.mean(v[w], 0) for w in WINDOWS], 0) for s, v in by_sentence.items()}
    vectors = np.stack([sentence_vectors[s] for s in sorted(sentence_vectors)])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(vectors), (resamples, len(vectors)))
    rows = {}
    for name, subset in (('all', slice(None)), ('activity', slice(0, 4)), ('direction', slice(4, 8))):
        values = vectors[:, subset].mean(-1)
        sample = values[indices].mean(-1)
        rows[name] = {'audio_minus_' + comparator: float(values.mean()),
            'exploratory_sentence_bootstrap_95ci': np.quantile(sample, [.025, .975]).tolist(),
            'sentences_audio_better': int((values < 0).sum()), 'sentence_count': len(values),
            'per_sentence_delta': dict(zip(sorted(sentence_vectors), values.tolist()))}
    return rows


def run(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    if args.output.exists():
        raise FileExistsError('Fresh output required; no validation-driven reruns into an existing directory')
    reference = json.loads(args.reference_protocol.read_text(encoding='utf8'))
    dataset_hash = sha(args.dataset)
    if reference['dataset_sha256'] != dataset_hash:
        raise ValueError('Dataset does not match fixed nested protocol')
    payload = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('schema') != DATASET_SCHEMA:
        raise ValueError('Unexpected source schema')
    fit, valid = split_train_pool(payload['clips'], reference)
    # Drop payload references before tensor extraction; only chosen fit/validation
    # clips remain. torch.load maps the whole archive; do not call that sealed.
    del payload
    protocol = {'schema': SCHEMA, 'dataset_sha256': dataset_hash, 'reference_protocol_sha256': sha(args.reference_protocol),
        'source_sha256': sha(__file__), 'seed': SEED, 'fps': FPS, 'windows_frames': WINDOWS,
        'fit_ids': [c['clip_id'] for c in fit], 'validation_ids': [c['clip_id'] for c in valid],
        'fit_sentences': sorted({c['sentence'] for c in fit}), 'validation_sentences': sorted({c['sentence'] for c in valid}),
        'targets': TARGET_NAMES, 'group_order': GROUP_NAMES, 'groups_in_upper9': GROUPS,
        'ridge_lambda_mean_risk': RIDGE, 'feature_projection': 'fixed seeded Gaussian 1536->64, plus unprojected4 prosody',
        'features': 'context + full native-run audio mean + window deviation/std/first-last-quarter change',
        'static': 'same context and native-run mean; all three local descriptor blocks zero before fit scaling',
        'target_transform': 'log1p RMS raw-coefficient/s after3-frame box; signed first/last smoothed quarter group change',
        'target_scale_floor': .001, 'audio_scale_floor': .01, 'ridge_feature_scale_floor': .01,
        'weighting': 'fit equal clips then equal supported windows; primary equal sentences/equal fixed scales/equal8 targets',
        'primary': 'audio minus separately trained matched_static normalized MSE over all3 scales and8 targets',
        'diagnostics': 'activity/direction components; within-clip centered scores; own-static/reverse interventions; train descriptive',
        'input_support': 'full nonoverlapping windows from native audio runs; fixed run-start phase; no target-mask input',
        'score_support': 'all9 target channels observed at every window frame; report excluded windows and native tails',
        'bootstrap': {'resamples': args.bootstrap, 'seed': SEED, 'clusters': 'five sentences; exploratory, not publication significance'},
        'outer_tensor_indexed': False, 'outer_archive_mapped': True,
        'scope': 'historically exposed internal development; upstream frozen extractors may be exposed; not independent test',
        'interpretation': 'linear family diagnostic only; no generator change, no nonlinear impossibility claim, no threshold selection',
        'versions': {'numpy': np.__version__, 'torch': torch.__version__}}
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', protocol)
    stats = audio_statistics(fit)
    results, checkpoints, all_predictions = {}, {}, {}
    for window in WINDOWS:
        train_rows, train_coverage = make_rows(fit, stats, window)
        valid_rows, valid_coverage = make_rows(valid, stats, window)
        y, weights = concatenate(train_rows, 'y')
        mean, scale = weighted_moments(y, weights, .001)
        target_stats = {'mean': mean, 'scale': scale}
        y = (y-mean) / scale
        models = {}
        for arm, features in (('audio', 'real'), ('matched_static', 'static')):
            x, input_weights = concatenate(train_rows, features)
            if not np.array_equal(input_weights, weights):
                raise RuntimeError('Matched arm fitting support differs')
            models[arm] = ridge_fit(x, y, weights)
        train_result, train_predictions = score_rows(train_rows, models, target_stats)
        valid_result, valid_predictions = score_rows(valid_rows, models, target_stats)
        key = str(window)
        results[key] = {'fit_descriptive': train_result, 'inner_validation': valid_result,
            'coverage': {'fit': train_coverage, 'inner_validation': valid_coverage},
            'target_fit_mean': mean.tolist(), 'target_fit_scale': scale.tolist()}
        checkpoints[key] = {'target_stats': target_stats, 'models': models}
        all_predictions[key] = {'fit_descriptive': train_predictions, 'inner_validation': valid_predictions}
        print(json.dumps({'window_frames': window, 'validation_mse': {a: valid_result[a]['sentence_equal']['mean_normalized_mse'] for a in ARMS}}), flush=True)
    comparisons = {name: primary_comparison(results, name, args.bootstrap) for name in ('train_mean', 'matched_static', 'audio_static', 'audio_reverse')}
    report = {'schema': SCHEMA, 'protocol_sha256': sha(args.output/'protocol.json'), 'results': results,
        'primary_comparisons': comparisons, 'elapsed_seconds': time.monotonic()-started,
        'generator_modified': False, 'default_replaced': False}
    torch.save({'protocol': protocol, 'protocol_sha256': report['protocol_sha256'], 'audio_statistics': stats,
                'window_models': checkpoints}, args.output/'coefficients.pt')
    torch.save({'protocol_sha256': report['protocol_sha256'], 'windows': all_predictions}, args.output/'predictions.pt')
    write_json(args.output/'report.json', report)
    lines = ['# Coarse motion condition predictability', '',
        'Internal fixed linear probe; five validation sentences. No generator changed. Lower normalized MSE is better.', '',
        '| Window | Train mean | Trained static | Audio | Own static | Reversed audio |', '|---|---:|---:|---:|---:|---:|']
    for window in WINDOWS:
        value = results[str(window)]['inner_validation']
        lines.append('| '+str(window/FPS)+'s | '+' | '.join(f"{value[a]['sentence_equal']['mean_normalized_mse']:.6f}" for a in ARMS)+' |')
    lines += ['', 'Equal-sentence/equal-scale/equal-target primary audio minus trained static:',
        '```json', json.dumps(comparisons['matched_static'], ensure_ascii=False, indent=2), '```', '',
        'Bootstrap intervals are exploratory with only five sentence clusters. Centered scores are scoring diagnostics using target centering, never prediction inputs.',
        'A positive result would motivate a separate generator-conditioning test; a negative result does not prove nonlinear audio prediction impossible.', '']
    (args.output/'report.md').write_text('\n'.join(lines), encoding='utf8')
    write_json(args.output/'manifest.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print('MOTION_CONDITION_PROBE_COMPLETE', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'reference-protocol', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--bootstrap', type=int, default=4096)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
