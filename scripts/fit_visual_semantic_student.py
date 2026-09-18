"""Sentence-OOF fixed-ridge audio students for frozen visual semantic targets.

No holdout tuning. Each fitting subset owns its standardization and PCA.
Deployment consumes audio/masks only; target labels are used for fitting and
scoring, never to construct static or reversed prediction conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'visual_semantic_ridge_student_v1'
ALPHA = 100.
PCA_DIM = 64
WINDOW = 5
MODES = ('actual', 'static', 'reverse')


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _runs(valid):
    boundaries = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)))


def audio_windows(clip):
    """Native valid runs, five-frame averages, no padding or crossing a gap."""
    features, valid = _numpy(clip['features']), _numpy(clip['valid'])
    if (features.ndim != 2 or features.shape[1] != 1540 or valid.shape != (len(features),)
            or valid.dtype != bool or not valid.any() or not np.isfinite(features[valid]).all()):
        raise ValueError('finite observed features[T,1540] and nonempty Boolean valid[T] required')
    selected = features[:, 768:1540].astype(np.float64)
    spans, run_ids, values = [], [], []
    for ri, (start, stop) in enumerate(_runs(valid)):
        for left in range(start, stop, WINDOW):
            right = min(left+WINDOW, stop)
            spans.append((int(left), int(right))); run_ids.append(ri)
            values.append(selected[left:right].mean(0))
    return {'values': np.stack(values), 'spans': spans, 'run_ids': np.asarray(run_ids),
            'static': selected[valid].mean(0), 'valid': valid.copy()}


def _targets(clip, windows):
    valid = windows['valid']; mask = _numpy(clip['semantic_valid'])
    va, posterior = _numpy(clip['va']).astype(np.float64), _numpy(clip['posterior']).astype(np.float64)
    if (mask.shape != valid.shape or mask.dtype != bool or (mask & ~valid).any()
            or va.shape != (len(valid), 2) or posterior.shape != (len(valid), 8)
            or not np.isfinite(va[mask]).all() or not np.isfinite(posterior[mask]).all()
            or (np.abs(va[mask]) > 1.00001).any() or (posterior[mask] < 0).any()
            or not np.allclose(posterior[mask].sum(1), 1., atol=1e-4, rtol=0)):
        raise ValueError('valid bounded VA and normalized posterior on semantic_valid are required')
    rows, known = [], []
    for left, right in windows['spans']:
        observed = mask[left:right]
        known.append(bool(observed.any()))
        if observed.any():
            v = va[left:right][observed].mean(0)
            p = posterior[left:right][observed].mean(0)
            rows.append(np.r_[v, np.log(np.maximum(p, 1e-4))])
        else:
            rows.append(np.zeros(10))
    return np.stack(rows), np.asarray(known)


def _fit_preprocess(windows):
    x = np.concatenate([w['values'] for w in windows])
    mean = x.mean(0); scale = x.std(0).clip(1e-6)
    z = (x-mean)/scale
    covariance = z.T @ z / len(z)
    eig, vectors = np.linalg.eigh(covariance)
    dimension = min(PCA_DIM, len(x)-1, x.shape[1])
    if dimension < 1:
        raise ValueError('at least two training acoustic windows required')
    components = vectors[:, -dimension:][:, ::-1].copy()
    # Canonical column sign, for stable serialized fits in the same runtime.
    pivots = np.argmax(np.abs(components), axis=0)
    components *= np.where(components[pivots, np.arange(dimension)] < 0, -1., 1.)
    return {'mean': mean, 'scale': scale, 'components': components,
            'explained_eigenvalues': eig[-dimension:][::-1].copy(), 'fit_window_count': len(x)}


def _design(windows, preprocessing, mode):
    if mode not in MODES:
        raise ValueError('unknown audio intervention')
    project = lambda x: ((x-preprocessing['mean'])/preprocessing['scale']) @ preprocessing['components']
    values = project(windows['values'])
    run_ids = windows['run_ids']
    if mode == 'static':
        values = np.broadcast_to(project(windows['static']), values.shape).copy()
    elif mode == 'reverse':
        values = values.copy()
        for ri in np.unique(run_ids):
            ids = np.flatnonzero(run_ids == ri)
            values[ids] = values[ids[::-1]]
    left = np.arange(len(values)); right = left.copy()
    for index in range(len(values)):
        if index and run_ids[index-1] == run_ids[index]: left[index] = index-1
        if index+1 < len(values) and run_ids[index+1] == run_ids[index]: right[index] = index+1
    return np.concatenate((values[left], values, values[right]), axis=1)


def _ridge(x, y, weights):
    weights = weights / weights.mean()
    mean_x = np.average(x, axis=0, weights=weights)
    mean_y = np.average(y, axis=0, weights=weights)
    a, b = (x-mean_x)*np.sqrt(weights[:, None]), (y-mean_y)*np.sqrt(weights[:, None])
    coefficient = np.linalg.solve(a.T @ a + ALPHA*np.eye(x.shape[1]), a.T @ b)
    return {'coefficient': coefficient, 'intercept': mean_y-mean_x@coefficient, 'alpha': ALPHA}


def fit_model(clips):
    """Fit only supplied clips; all label-dependent arithmetic lives here."""
    if not clips: raise ValueError('nonempty training clips required')
    windows = [audio_windows(clip) for clip in clips]
    preprocessing = _fit_preprocess(windows)
    targets = [_targets(clip, window) for clip, window in zip(clips, windows)]
    if any(not known.any() for _, known in targets):
        raise ValueError('every fitting clip needs at least one observed semantic window')
    y = np.concatenate([value[known] for value, known in targets])
    weights = np.concatenate([np.full(int(known.sum()), 1./known.sum()) for _, known in targets])
    models = {}
    for mode in ('actual', 'static'):
        x = np.concatenate([_design(window, preprocessing, mode)[known]
                            for window, (_, known) in zip(windows, targets)])
        models[mode] = _ridge(x, y, weights)
    return {'schema': SCHEMA, 'preprocessing': preprocessing, 'heads': models,
            'fit_clip_ids': [clip['clip_id'] for clip in clips],
            'fit_sentences': sorted({clip['sentence'] for clip in clips}),
            'target_order': ['va2', 'log_posterior8'], 'pca_max_dimensions': PCA_DIM}


def predict_audio(model, clip):
    """Inference API reads only features and valid; targets may be absent."""
    windows = audio_windows(clip)
    output = {'audio_valid': torch.from_numpy(windows['valid']), 'va': {}, 'posterior': {}}
    for mode in MODES:
        head = model['heads']['static' if mode == 'static' else 'actual']
        x = _design(windows, model['preprocessing'], mode)
        raw = x @ head['coefficient'] + head['intercept']
        va = np.clip(raw[:, :2], -1., 1.)
        logits = raw[:, 2:]; probabilities = np.exp(logits-logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        for name, values, width in (('va', va, 2), ('posterior', probabilities, 8)):
            native = np.zeros((len(windows['valid']), width), np.float32)
            for index, (left, right) in enumerate(windows['spans']):
                native[left:right] = values[index]
            output[name][mode] = torch.from_numpy(native)
    return output


def _scores(predictions, clips):
    report = {}
    for split in ('train', 'holdout'):
        report[split] = {}
        indices = [i for i, c in enumerate(clips) if c['split'] == split]
        for target in ('va', 'posterior'):
            report[split][target] = {}
            for mode in MODES:
                rows = []
                for i in indices:
                    mask = _numpy(clips[i]['semantic_valid']) & _numpy(clips[i]['valid'])
                    if not mask.any(): continue
                    truth = _numpy(clips[i][target])[mask].astype(float)
                    predicted = _numpy(predictions[i][target][mode])[mask].astype(float)
                    error = predicted-truth
                    centered = (predicted-predicted.mean(0))-(truth-truth.mean(0))
                    # Brier uses the sum across categories; VA uses dimension mean.
                    factor = truth.shape[1] if target == 'posterior' else 1.
                    rows.append({'clip_id': clips[i]['clip_id'], 'frames': int(mask.sum()),
                                 'raw': float(np.mean(error**2)*factor),
                                 'centered': float(np.mean(centered**2)*factor)})
                report[split][target][mode] = {'metric': 'brier' if target == 'posterior' else 'mse',
                    'clip_equal_raw': float(np.mean([r['raw'] for r in rows])) if rows else None,
                    'clip_equal_centered': float(np.mean([r['centered'] for r in rows])) if rows else None,
                    'clip_count': len(rows), 'per_clip': rows}
    return report


def fit_dataset(data):
    clips = data['clips']
    if not clips or len({c['clip_id'] for c in clips}) != len(clips):
        raise ValueError('nonempty unique clip IDs required')
    if any(c['split'] not in ('train', 'holdout') or not isinstance(c['sentence'], str) for c in clips):
        raise ValueError('train/holdout split and string sentence required')
    train = [c for c in clips if c['split'] == 'train']
    holdout = [c for c in clips if c['split'] == 'holdout']
    sentences = sorted({c['sentence'] for c in train})
    if len(sentences) < 3 or set(sentences) & {c['sentence'] for c in holdout}:
        raise ValueError('three training sentences and sentence-disjoint holdout required')
    # Validate holdout masks/labels for scoring without including them in any
    # fit, preprocessing statistic, model selection or deployable condition.
    for clip in holdout:
        _targets(clip, audio_windows(clip))
    # Metadata-only deterministic folds, never selected with semantic values.
    fold_by_sentence = {sentence: index % 3 for index, sentence in enumerate(sentences)}
    all_model = fit_model(train)
    fold_models = {fold: fit_model([c for c in train if fold_by_sentence[c['sentence']] != fold]) for fold in range(3)}
    predictions = []
    for clip in clips:
        if clip['split'] == 'train':
            fold = fold_by_sentence[clip['sentence']]; model = fold_models[fold]
            source = 'sentence_oof'
            assert clip['sentence'] not in model['fit_sentences']
        else:
            fold = None; model = all_model; source = 'all_train'
        pred = predict_audio(model, clip)
        pred.update(clip_id=clip['clip_id'], sentence=clip['sentence'], split=clip['split'],
                    fold=fold, prediction_source=source,
                    semantic_valid=torch.from_numpy(_numpy(clip['semantic_valid']).copy()))
        predictions.append(pred)
    return {'schema': SCHEMA, 'clips': predictions, 'fold_by_sentence': fold_by_sentence}, {
        'all_train': all_model, 'oof': fold_models}, _scores(predictions, clips)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('fresh output required')
    data = torch.load(args.dataset, map_location='cpu', weights_only=False)
    predictions, models, scores = fit_dataset(data)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, args.output/'predictions.pt'); torch.save(models, args.output/'models.pt')
    (args.output/'report.json').write_text(json.dumps(scores, indent=2, allow_nan=False)+'\n', encoding='utf8')
    provenance = {'schema': SCHEMA, 'dataset_sha256': _sha(args.dataset), 'code_sha256': _sha(__file__),
                  'alpha': ALPHA, 'pca_max_dimensions': PCA_DIM, 'window_frames': WINDOW,
                  'audio_feature_slice': [768, 1540], 'context_windows': [-1, 0, 1],
                  'context_is_noncausal': True, 'fit_scope': 'train only; all preprocessing refit per OOF fold',
                  'deployment_uses_motion_or_visual_targets': False,
                  'static': 'separate same-alpha ridge trained on broadcast observed audio clip means',
                  'reverse': 'actual model with within-run acoustic window order reversed',
                  'regression_weight': 'equal clips, equal observed windows inside each clip',
                  'pca_weight': 'equal acoustic windows from fitting clips only',
                  'scores': 'clip-equal; train predictions are sentence-OOF; holdout is already-development evidence',
                  'outputs': {name: _sha(args.output/name) for name in ('predictions.pt', 'models.pt', 'report.json')}}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'schema': SCHEMA, 'clips': len(predictions['clips']), 'output': str(args.output)}))


if __name__ == '__main__': main()
