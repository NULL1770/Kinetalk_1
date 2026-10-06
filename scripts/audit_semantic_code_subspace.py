"""Fixed teacher-head rowspace error attribution; no generation or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def semantic_projection(emotion_weight, intensity_weight):
    heads = [np.asarray(w, dtype=np.float64) for w in (emotion_weight, intensity_weight)]
    matrix = np.concatenate([w - w.mean(0) for w in heads])
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int((singular > singular[0] * 1e-10).sum())
    basis = vh[:rank]
    projection = basis.T @ basis
    np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)
    return projection, singular, rank


def error_parts(audio, teacher, projection):
    difference = np.asarray(audio, dtype=np.float64) - teacher
    projected = difference @ projection
    remainder = difference - projected
    total, semantic, other = [float(np.mean(v ** 2)) for v in (difference, projected, remainder)]
    np.testing.assert_allclose(total, semantic + other, atol=1e-12)
    return {'mse': total, 'head_rowspace_mse': semantic, 'other_directions_mse': other,
            'head_rowspace_error_fraction': semantic / total if total else 0.}


def probabilities(code, weight, bias):
    logits = code @ weight.T + bias
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    source_report = json.loads((a.source / 'report.json').read_text())
    assert not source_report['test_loaded'] and not source_report['training_performed']
    heads_meta = json.loads(a.heads.with_suffix('.json').read_text())
    assert heads_meta['checkpoint_sha256'] == source_report['checkpoint_sha256']
    assert heads_meta['data_manifest_sha256'] == source_report['data_manifest_sha256']
    assert hashlib.sha256(a.heads.read_bytes()).hexdigest() == heads_meta['npz_sha256']
    heads = np.load(a.heads, allow_pickle=False)
    codes = np.load(a.source / 'per_clip_conditions.npz', allow_pickle=False)
    projection, singular, rank = semantic_projection(heads['emotion_weight'], heads['intensity_weight'])
    mean = codes['train__teacher_global'].astype(np.float64).mean(0)
    report = {'schema': 'fixed_teacher_head_direction_error_v1', 'test_loaded': False,
              'training_performed': False, 'generation_performed': False,
              'checkpoint_sha256': heads_meta['checkpoint_sha256'],
              'data_manifest_sha256': heads_meta['data_manifest_sha256'],
              'source_npz_sha256': hashlib.sha256((a.source / 'per_clip_conditions.npz').read_bytes()).hexdigest(),
              'heads_npz_sha256': heads_meta['npz_sha256'], 'rank_relative_tolerance': 1e-10,
              'rank': rank, 'singular_values': singular.tolist(), 'results': {},
              'interpretation_limit': 'Other directions may carry expression needed by the renderer; attribution is not a deployment projection.'}
    for role in ('train', 'validation'):
        audio = codes[role + '__audio_global'].astype(np.float64)
        teacher = codes[role + '__teacher_global'].astype(np.float64)
        row = error_parts(audio, teacher, projection)
        variance = teacher - mean
        projected_variance = variance @ projection
        row['teacher_variance_retained_fraction'] = float(
            np.mean(projected_variance ** 2) / np.mean(variance ** 2))
        row['probability_preservation_max_error'] = {}
        for name, value in (('audio', audio), ('teacher', teacher)):
            projected = mean + (value - mean) @ projection
            for head in ('emotion', 'intensity'):
                w, b = heads[head + '_weight'], heads[head + '_bias']
                before, after = probabilities(value, w, b), probabilities(projected, w, b)
                np.testing.assert_allclose(before, after, atol=1e-10, rtol=1e-10)
                row['probability_preservation_max_error'][name + '_' + head] = float(np.max(np.abs(before-after)))
        report['results'][role] = row
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'projection.npz', projection=projection, train_teacher_mean=mean)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--heads', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
