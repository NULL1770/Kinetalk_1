"""Independently rescore sealed ten-arm semantic-student transfer curves.

Reuses the separately implemented NumPy primary metrics from the six-arm audit,
not the trainer/evaluator scoring implementation. No dataset or video is read.
The experiment is a frozen-receiver distribution-transfer diagnostic, not a
matched receiver retraining or an unbiased final-generalization experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_visual_semantic_pilot import independent_primary, sha, write_json

SCHEMA = 'semantic_student_transfer_audit_v1'
SOURCE_SCHEMA = 'semantic_student_transfer_v1'
KINDS = ('va', 'posterior')
MODES = ('old', 'new', 'static', 'reverse', 'constant')
ARMS = tuple(kind+'_'+mode for kind in KINDS for mode in MODES)
SEEDS = [42, 123, 2026, 77]
UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]
BOOTSTRAP_SEED = 20260919


def _difference(actual, expected):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError('primary report undefined-value mismatch')
        return 0.
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('primary report structure/nonfinite mismatch')
    return float(np.max(np.abs(a-b))) if a.size else 0.


def sentence_bootstrap(rows, metric, draws=2000, seed=BOOTSTRAP_SEED):
    """Resample sentence clusters while retaining the clip-equal estimand."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['sentence'], []).append(float(row[metric]))
    if not grouped or type(draws) is not int or draws < 1:
        raise ValueError('nonempty paired rows and positive bootstrap draws required')
    sentences = sorted(grouped)
    totals = np.asarray([sum(grouped[s]) for s in sentences])
    counts = np.asarray([len(grouped[s]) for s in sentences])
    selected = np.random.default_rng(seed).integers(0, len(sentences), size=(draws, len(sentences)))
    samples = totals[selected].sum(1)/counts[selected].sum(1)
    return {'mean': float(totals.sum()/counts.sum()),
            'q025': float(np.quantile(samples, .025)), 'q975': float(np.quantile(samples, .975)),
            'draws': draws, 'seed': seed, 'sentences': len(sentences), 'clips': int(counts.sum()),
            'estimand': 'clip-equal mean paired difference; sentence-cluster percentile bootstrap',
            'single_sentence_interval_is_degenerate': len(sentences) == 1}


def _validate_curves(packed):
    if packed.get('schema') != SOURCE_SCHEMA or packed['protocol'].get('sample_seeds') != SEEDS:
        raise ValueError('unexpected transfer schema or fixed sample seeds')
    if packed['protocol'].get('modes') != list(MODES):
        raise ValueError('unexpected transfer modes')
    clips = packed['clips']
    if not clips:
        raise ValueError('empty packaged curves')
    roles = {'holdout': [], 'fit_examples': []}
    for cid, clip in clips.items():
        meta = clip['metadata']; valid = np.asarray(clip['valid']); native = np.asarray(clip['native_valid'])
        target = np.asarray(clip['target']); baseline = np.asarray(clip['baseline52'])
        t = len(valid)
        if (meta['clip_id'] != cid or meta['split'] not in ('train', 'holdout')
                or not isinstance(meta['sentence'], str) or valid.dtype != bool or native.dtype != bool
                or valid.shape != (t,) or native.shape != (t,) or not valid.any() or (valid & ~native).any()
                or target.shape != (t, 9) or baseline.shape != (t, 52)
                or not np.isfinite(target[valid]).all() or not np.isfinite(baseline).all()
                or clip['seeds'] != SEEDS or set(clip['samples']) != set(ARMS)):
            raise ValueError('invalid common curve support, shape, membership or seeds: '+cid)
        times = np.asarray(clip['times'])
        target52 = np.asarray(clip['target52'])
        if (times.shape != (t,) or not np.allclose(times, np.arange(t)/25, atol=1e-7, rtol=0)
                or target52.shape != (t, 52) or not np.array_equal(target, target52[:, UPPER], equal_nan=True)):
            raise ValueError('target channel mapping or native clock differs: '+cid)
        for arm in ARMS:
            samples = np.asarray(clip['samples'][arm])
            if (samples.shape != (4, t, 9) or not np.isfinite(samples).all()
                    or not np.array_equal(samples[:, ~native], np.broadcast_to(baseline[~native][:, UPPER], (4, int((~native).sum()), 9)))):
                raise ValueError('invalid draw shape, finite values or protected native-invalid frames: '+cid+'/'+arm)
        roles['holdout' if meta['split'] == 'holdout' else 'fit_examples'].append(cid)
    return clips, {role: sorted(ids) for role, ids in roles.items()}


def _compare_report(result, reported, ids, clips):
    summary = reported['summary']; rows = reported['rows']
    if len(rows) != len(ids) or {row['clip_id'] for row in rows} != set(ids):
        raise ValueError('report and curve clip membership differs')
    differences = [_difference(result['joint_fair_es'][metric], summary['joint_fair_es'][metric])
                   for metric in ('raw', 'centered')]
    differences += [_difference(a, b) for a, b in zip(result['rms_ratio'], summary['rms_ratio'])]
    if len(summary['rms_ratio']) != 4: raise ValueError('report group membership differs')
    differences += [_difference(result['speed_rms'], summary['speed']['all']['rms']),
                    _difference(result['target_speed_rms'], summary['speed']['reference_all']['rms'])]
    by_id = {row['clip_id']: row for row in rows}
    for row in result['rows']:
        reference = by_id[row['clip_id']]
        if reference['sentence'] != clips[row['clip_id']]['metadata']['sentence']:
            raise ValueError('report sentence membership differs')
        differences.extend(_difference(row[metric], reference['joint_fair_es'][metric]) for metric in ('raw', 'centered'))
    maximum = max(differences)
    if maximum > 1e-9:
        raise ValueError('independent primary report mismatch: '+str(maximum))
    return maximum


def audit(run, output, bootstrap_draws=2000):
    run, output = Path(run), Path(output)
    if output.exists(): raise FileExistsError('fresh audit output required')
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError('write audit outside sealed run directory')
    names = ('predictions.pt', 'reports.json', 'va_scales.pt', 'posterior_scales.pt')
    for name in names:
        if not (run/name).is_file(): raise ValueError('missing audit input: '+name)
    hashes = {name: sha(run/name) for name in names}
    manifest_verified = []
    if (run/'manifest.json').is_file():
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf8'))
        for name in names:
            if manifest[name]['sha256'] != hashes[name] or manifest[name]['bytes'] != (run/name).stat().st_size:
                raise ValueError('sealed run manifest mismatch: '+name)
            manifest_verified.append(name)
    if (run/'status.json').is_file() and json.loads((run/'status.json').read_text(encoding='utf8'))['status'] != 'complete':
        raise ValueError('transfer run is not complete')
    packed = torch.load(run/'predictions.pt', map_location='cpu', weights_only=False)
    clips, roles = _validate_curves(packed)
    reported = json.loads((run/'reports.json').read_text(encoding='utf8'))
    if set(reported) != set(ARMS): raise ValueError('expected all ten report arms')
    results = {}; paired = {role: {} for role in roles}; maximum = 0.
    for kind in KINDS:
        scale = np.asarray(torch.load(run/(kind+'_scales.pt'), map_location='cpu', weights_only=False)['metric_scale'], dtype=float)
        if scale.shape != (9,) or not np.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('invalid '+kind+' metric scale')
        for mode in MODES:
            arm = kind+'_'+mode; results[arm] = {}
            for role, ids in roles.items():
                if not ids: continue
                result = independent_primary(clips, ids, arm, scale)
                result['max_report_difference'] = _compare_report(result, reported[arm][role], ids, clips)
                maximum = max(maximum, result['max_report_difference']); results[arm][role] = result
        for role, ids in roles.items():
            if not ids: continue
            new = {row['clip_id']: row for row in results[kind+'_new'][role]['rows']}
            for mode in ('old', 'static', 'reverse', 'constant'):
                reference = {row['clip_id']: row for row in results[kind+'_'+mode][role]['rows']}
                differences = [{'clip_id': cid, 'sentence': clips[cid]['metadata']['sentence'],
                                **{metric: new[cid][metric]-reference[cid][metric] for metric in ('raw', 'centered')}} for cid in ids]
                paired[role][kind+'_new_minus_'+mode] = {
                    'raw': sentence_bootstrap(differences, 'raw', draws=bootstrap_draws),
                    'centered': sentence_bootstrap(differences, 'centered', draws=bootstrap_draws),
                    'clip_differences': differences, 'negative_favors_new': True}
    result = {'schema': SCHEMA, 'run': str(run.resolve()), 'input_sha256': hashes,
              'audit_code_sha256': sha(__file__),
              'independent_metric_module_sha256': sha(Path(__file__).with_name('audit_visual_semantic_pilot.py')),
              'manifest_verified_inputs': manifest_verified, 'sample_seeds': SEEDS,
              'clip_counts': {role: len(ids) for role, ids in roles.items()},
              'max_report_difference': maximum, 'independent_primary': results, 'paired': paired,
              'scope': {'used_dataset': False, 'selected_draws': False,
                        'recomputed': 'raw/centered trajectory fair ES, four-group centered RMS ratios, adjacent-frame speed; all ten arms',
                        'not_recomputed': 'other ancillary report metrics, receiver replay, student training and source membership',
                        'bootstrap': 'fixed seed, paired by clip, resample whole sentence clusters with clip-equal mean',
                        'interpretation': 'frozen-receiver distribution transfer on previously used development clips; no naturalness or final generalization certification'}}
    output.parent.mkdir(parents=True, exist_ok=True); write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.run, args.output)
    print(json.dumps({'schema': SCHEMA, 'max_report_difference': result['max_report_difference'],
                      'clip_counts': result['clip_counts'], 'output': str(args.output)}))
