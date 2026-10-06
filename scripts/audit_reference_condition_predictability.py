"""Fixed real-train code ridge audit; no generation, model change or test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def fit_predict(x, y, query):
    mean = x.mean(0)
    scale = x.std(0).clip(1e-4)
    target_mean = y.mean(0)
    z = (x - mean) / scale
    w = np.linalg.solve(z.T @ z / len(z) + .001 * np.eye(z.shape[1]),
                        z.T @ (y - target_mean) / len(z))
    return (query - mean) / scale @ w + target_mean


def inputs(z, role, reference):
    x = z[role + '__audio_global'].astype(np.float64)
    if not reference:
        return x
    identity = z[role + '__identity_baseline'].astype(np.float64)
    # Audio-predicted category is available at inference; true emotion is not.
    onehot = np.eye(8)[z[role + '__audio_emotion']]
    interaction = (onehot[:, :, None] * identity[:, None, :]).reshape(len(x), -1)
    return np.concatenate([x, interaction], axis=1)


def score(pred, target):
    error = pred - target
    return {'mse': float(np.mean(error ** 2)),
            'mean_bias_mse': float(np.mean(error.mean(0) ** 2))}


def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    z = np.load(a.source / 'per_clip_conditions.npz', allow_pickle=False)
    source = json.loads((a.source / 'report.json').read_text())
    assert source['test_loaded'] is False and source['training_performed'] is False
    y = z['train__teacher_global'].astype(np.float64)
    vy = z['validation__teacher_global'].astype(np.float64)
    speaker = z['train__speaker']
    folds = speaker % 5
    report = {'schema': 'fixed_reference_code_predictability_v1', 'test_loaded': False,
              'generator_training_performed': False, 'generation_performed': False,
              'real_train_ridge_fitted': True, 'ridge_lambda': .001,
              'std_floor': 1e-4, 'folds': 'train speaker_id modulo5, fixed',
              'scope': 'Extra code mapping only; frozen encoders already trained on all train speakers',
              'input': 'audio64, plus independentneutralidentity baseline52 x audio-predicted emotion8',
              'source_checkpoint_sha256': source['checkpoint_sha256'],
              'source_npz_sha256': hashlib.sha256((a.source / 'per_clip_conditions.npz').read_bytes()).hexdigest(),
              'data_manifest_sha256': source['data_manifest_sha256'],
              'results': {'unchanged_audio': {'train': score(z['train__audio_global'], y),
                                             'validation': score(z['validation__audio_global'], vy)}}}
    predictions = {}
    for reference in (False, True):
        name = 'audio_and_neutral_reference' if reference else 'audio_only_ridge'
        x, vx = inputs(z, 'train', reference), inputs(z, 'validation', reference)
        out = np.empty_like(y)
        rows = []
        for fold in range(5):
            selected = folds == fold
            out[selected] = fit_predict(x[~selected], y[~selected], x[selected])
            rows.append({'fold': fold, 'query_speakers': np.unique(speaker[selected]).tolist(),
                         'query_clips': int(selected.sum()), **score(out[selected], y[selected])})
        vp = fit_predict(x, y, vx)
        report['results'][name] = {'speaker_fold_out_of_fit_train': score(out, y),
                                   'fold_rows': rows, 'all_train_fit_validation': score(vp, vy)}
        predictions[name + '__train_folds'] = out
        predictions[name + '__validation'] = vp
        print(json.dumps({'condition': name, **report['results'][name]}), flush=True)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'predicted_codes.npz', **predictions)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
