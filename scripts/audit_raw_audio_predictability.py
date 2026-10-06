"""Fixed TRAIN speaker-fold diagnostic for retained pretrained emotion features.

Compares existing hidden mean128 with hidden mean128 plus raw emotion mean768.
No generator training, candidate deployment, validation fitting or sweep.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_reference_condition_predictability import fit_predict, score
from scripts.diagnose_flow_sampling import sha
from scripts.packed_trainval_cache import load_packed


def masked_emotion_mean(features, valid):
    if features.ndim != 3 or features.shape[-1] != 1540 or valid.shape != features.shape[:2]:
        raise ValueError('Expected native shared1540 audio and matching mask')
    if valid.dtype != torch.bool or not valid.any(1).all():
        raise ValueError('Every clip must contain observed frames')
    emotion = features[..., 768:1536]
    if not torch.isfinite(emotion[valid]).all():
        raise ValueError('Observed audio must be finite')
    return torch.where(valid[..., None], emotion, 0.).sum(1) / valid.sum(1, keepdim=True)


@torch.no_grad()
def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh diagnostic required')
    torch.set_num_threads(2)
    cached = json.loads((a.source/'report.json').read_text())
    pool_report = json.loads((a.pooling/'report.json').read_text())
    assert cached['checkpoint_sha256'] == pool_report['source_checkpoint_sha256'] == a.checkpoint_sha256
    assert not cached['test_loaded'] and not pool_report['test_loaded']
    assert cached['training_performed'] is False and pool_report['generator_training_performed'] is False
    z = np.load(a.source/'per_clip_conditions.npz', allow_pickle=False)
    pool = np.load(a.pooling/'features_predictions.npz', allow_pickle=False)
    data = load_packed(a.data, materialize=False, with_refs=False)
    manifest = data['provenance']['manifest_sha256']
    assert manifest == cached['data_manifest_sha256'] == pool_report['data_manifest_sha256']
    assert set(data['splits']) == {'train', 'validation'} and not data['provenance']['test_loaded']
    arrays = {}
    for role, q in data['splits'].items():
        ids = np.asarray(q['clip_id'])
        np.testing.assert_array_equal(ids, z[role+'__clip_id'])
        np.testing.assert_array_equal(ids, pool[role+'_clip_id'])
        np.testing.assert_array_equal(q['speaker_id'].numpy(), z[role+'__speaker'])
        rows = []
        for batch_no, ix in enumerate(torch.arange(len(q['_lengths'])).split(16)):
            b = q.batch(ix, 'cpu', keys=('audio_features', 'valid'))
            rows.append(masked_emotion_mean(b['audio_features'], b['valid']).numpy())
            if batch_no % 100 == 0:
                print(json.dumps(dict(event='raw_emotion_mean', role=role, clips=int(ix[-1])+1)), flush=True)
        arrays[role+'__raw_emotion_mean'] = np.concatenate(rows)
    report = dict(schema='raw_emotion_predictability_v1', test_loaded=False,
        generator_training_performed=False, generation_performed=False,
        real_train_ridge_fitted=True, ridge_lambda=.001, feature_std_floor=1e-4,
        source_checkpoint_sha256=a.checkpoint_sha256, data_manifest_sha256=manifest,
        source_npz_sha256=sha(a.source/'per_clip_conditions.npz'),
        pooling_npz_sha256=sha(a.pooling/'features_predictions.npz'),
        script_sha256=sha(Path(__file__)), results={},
        scope='Additional readouts; frozen student has already seen all TRAIN speakers',
        folds='TRAIN speaker_id modulo5; fixed all-TRAIN fit reported separately on validation',
        input='existing last-hidden mean128 plus shared pretrained emotion mean768',
        decision_gate='Both total MSE and each of five TRAIN fold MSEs must improve before a structural trial')
    y = z['train__teacher_global'].astype(np.float64)
    vy = z['validation__teacher_global'].astype(np.float64)
    speaker = z['train__speaker']; folds = speaker % 5
    for extra in (False, True):
        name = 'hidden_mean' if not extra else 'hidden_plus_raw_emotion'
        def features(role):
            x = pool[role+'__mean']
            if extra: x = np.concatenate([x, arrays[role+'__raw_emotion_mean']], 1)
            return x.astype(np.float64)
        x, vx = features('train'), features('validation')
        out = np.empty_like(y); rows = []
        for fold in range(5):
            query = folds == fold
            out[query] = fit_predict(x[~query], y[~query], x[query])
            rows.append(dict(fold=fold, query_speakers=np.unique(speaker[query]).tolist(),
                             query_clips=int(query.sum()), **score(out[query], y[query])))
        vp = fit_predict(x, y, vx)
        report['results'][name] = dict(features=x.shape[1], fold_rows=rows,
            speaker_fold_out_of_fit_train=score(out, y), all_train_fit_validation=score(vp, vy))
        arrays[name+'__train_prediction'] = out; arrays[name+'__validation_prediction'] = vp
        if not extra:
            np.testing.assert_allclose(out, pool['mean_only__train_fold_prediction'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(vp, pool['mean_only__validation_prediction'], rtol=1e-10, atol=1e-10)
        print(json.dumps(dict(event='ridge_result', method=name, **report['results'][name])), flush=True)
    base, candidate = (report['results'][k] for k in ('hidden_mean','hidden_plus_raw_emotion'))
    passed = (candidate['speaker_fold_out_of_fit_train']['mse'] < base['speaker_fold_out_of_fit_train']['mse']
        and all(e['mse'] < c['mse'] for c, e in zip(base['fold_rows'], candidate['fold_rows'])))
    report['passes_train_gate'] = passed
    report['decision'] = 'Consider a single structural bypass trial; no adoption' if passed else 'Do not train this bypass based on validation alone'
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'features_predictions.npz', **arrays)
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(event='complete', output=str(a.output), passes_train_gate=passed)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('data','source','pooling','output'): p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    run(p.parse_args())
