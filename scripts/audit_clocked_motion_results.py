"""Independently rescore saved fixed-clock curves without opening any dataset.

Reads only a completed experiment directory. No trainer, model, native loader,
or production scoring helper is imported. The only output is ``audit.json``.
Saved predictions are float32 whereas the original scorer received float64;
comparisons use rtol=1e-5, atol=1e-8 and report all failures rather than silently
changing precision, interventions, samples or acceptance thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch


SCHEMA = 'clocked_motion_saved_curve_audit_v1'
ARMS = ('temporal', 'static_trained', 'static', 'reverse', 'mismatch', 'uniform')
PAIRED_ARMS = ('temporal', 'static_trained', 'static', 'reverse', 'mismatch')
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
CELLS = ('calibration', 'confirmation')
RTOL, ATOL = 1e-5, 1e-8


def json_default(value):
    """Convert only NumPy containers/scalars; nonfinite floats still fail JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def runs(valid):
    edge = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def centered(value, spans):
    output = np.zeros_like(value, dtype=np.float64)
    for left, right in spans:
        part = value[..., left:right, :]-value[..., left:left+1, :]
        output[..., left:right, :] = part-part.mean(axis=-2, keepdims=True)
    return output


def energy_score(draws, truth):
    # Fair ensemble ES: mean distance to observation minus the unbiased
    # pairwise draw-distance half expectation. Every saved draw participates.
    x, y = draws.reshape(len(draws), -1), truth.reshape(-1)
    first = np.linalg.norm(x-y, axis=1).mean()/np.sqrt(len(y))
    second = sum(np.linalg.norm(x[i]-x[j])/np.sqrt(len(y))
                 for i in range(len(x)) for j in range(i))
    return float(first-second/(len(x)*(len(x)-1)))


def group_sum(vector):
    return np.asarray([np.sum(vector[list(group)]) for group in GROUPS])


def ratios(numerator, denominator, unchanged_zero=False):
    return [float(np.sqrt(max(a, 0.)/b)) if b > 0 else
            1. if unchanged_zero and a == 0 else None for a, b in zip(numerator, denominator)]


def rescore(curve, scales):
    raw = numpy(curve['samples'], np.float64)
    target = numpy(curve['target'], np.float64)
    valid = numpy(curve['valid'])
    if (raw.ndim != 3 or raw.shape[0] < 2 or raw.shape[-1] != 9
            or target.shape != raw.shape[1:] or valid.dtype != np.bool_
            or valid.shape != target.shape[:1] or not valid.any()
            or not np.isfinite(raw[:, valid]).all() or not np.isfinite(target[valid]).all()):
        raise ValueError('Invalid saved samples, target or native mask')
    spans = runs(valid)
    raw = np.where(valid[None, :, None], raw, 0.)
    target = np.where(valid[:, None], target, 0.)
    x, y = raw/scales, target/scales
    xc, yc = centered(x, spans), centered(y, spans)
    joint, group = {}, {}
    for key, xx, yy in (('raw', x[:, valid], y[valid]),
                        ('centered', xc[:, valid], yc[valid])):
        joint[key] = energy_score(xx, yy)
        group[key] = [energy_score(xx[..., list(g)], yy[..., list(g)]) for g in GROUPS]
    pred_energy = group_sum(np.square(xc[:, valid]).sum(axis=1).mean(axis=0))
    target_energy = group_sum(np.square(yc[valid]).sum(axis=0))
    clipped = centered(np.clip(raw, 0., 1.)/scales, spans)
    clip_energy = group_sum(np.square(clipped[:, valid]).sum(axis=1).mean(axis=0))
    outside = ((raw[:, valid] < 0.) | (raw[:, valid] > 1.)).sum(axis=(0, 1))
    outside = group_sum(outside)
    raw_count = np.array([len(x)*valid.sum()*len(g) for g in GROUPS])
    speed = {'sum_squares': 0., 'count': 0, 'reference_sum_squares': 0., 'reference_count': 0}
    acceleration = dict(speed)
    vx, vy, transitions = np.zeros((9, 9)), np.zeros((9, 9)), 0
    var_sum, var_pairs = 0., 0
    for left, right in spans:
        dx, dy = np.diff(x[:, left:right], axis=1), np.diff(y[left:right], axis=0)
        speed['sum_squares'] += float(np.square(dx).mean(axis=-1).sum())
        speed['count'] += int(np.prod(dx.shape[:2]))
        speed['reference_sum_squares'] += float(np.square(dy).mean(axis=-1).sum())
        speed['reference_count'] += len(dy)
        if len(dy):
            dx0, dy0 = dx-dx.mean(axis=1, keepdims=True), dy-dy.mean(axis=0, keepdims=True)
            vx += np.einsum('kti,ktj->ij', dx0, dx0)
            vy += dy0.T@dy0
            transitions += len(dy)
        if right-left >= 3:
            ax, ay = np.diff(dx, axis=1), np.diff(dy, axis=0)
            acceleration['sum_squares'] += float(np.square(ax).sum())
            acceleration['count'] += ax.size
            acceleration['reference_sum_squares'] += float(np.square(ay).sum())
            acceleration['reference_count'] += ay.size
        for lag in (1, 4, 16, 32):
            if right-left > lag:
                xd = np.sqrt(np.abs(x[:, left+lag:right]-x[:, left:right-lag])).mean(axis=0)
                yd = np.sqrt(np.abs(y[left+lag:right]-y[left:right-lag]))
                var_sum += float(np.square(xd-yd).sum())
                var_pairs += len(yd)
    covariance = float(np.sqrt(np.square(vx/(len(x)*transitions)-vy/transitions).mean())) if transitions else None
    return {'sample_count': len(x), 'valid_frames': int(valid.sum()), 'valid_runs': len(spans),
        'joint_fair_es': joint, 'group_fair_es': group,
        'rms_ratio': ratios(pred_energy, target_energy),
        'raw_oob': (outside/raw_count).tolist(),
        'clamp_rms_retention': ratios(clip_energy, pred_energy, unchanged_zero=True),
        'pooled': {'prediction_energy': pred_energy, 'target_energy': target_energy,
            'clipped_prediction_energy': clip_energy, 'oob_count': outside, 'raw_count': raw_count},
        'speed': speed, 'acceleration': acceleration,
        'variogram': var_sum/(9*var_pairs) if var_pairs else None,
        'velocity_covariance': covariance,
        'low_activity_energy': float(np.square(yc[valid]).mean()),
        'short_runs': sum(right-left < 32 for left, right in spans),
        'short_frames': sum(right-left for left, right in spans if right-left < 32)}


def sum_stats(rows, key):
    value = {k: sum(row[key][k] for row in rows) for k in rows[0][key]}
    a, b = value['count'], value['reference_count']
    value['rms'] = float(np.sqrt(value['sum_squares']/a)) if a else None
    value['reference_rms'] = float(np.sqrt(value['reference_sum_squares']/b)) if b else None
    value['rms_ratio'] = (value['rms']/value['reference_rms']
                          if a and b and value['reference_rms'] > 0 else None)
    return value


def summarize(rows):
    if not rows:
        return None
    pooled = {key: np.sum([row['pooled'][key] for row in rows], axis=0)
              for key in rows[0]['pooled']}
    mean_supported = lambda key: float(np.mean([row[key] for row in rows if row[key] is not None])) if any(row[key] is not None for row in rows) else None
    return {'clips': len(rows),
        'joint_fair_es': {key: float(np.mean([row['joint_fair_es'][key] for row in rows])) for key in ('raw', 'centered')},
        'group_fair_es': {key: np.mean([row['group_fair_es'][key] for row in rows], axis=0).tolist() for key in ('raw', 'centered')},
        'rms_ratio': ratios(pooled['prediction_energy'], pooled['target_energy']),
        'raw_oob': (pooled['oob_count']/pooled['raw_count']).tolist(),
        'clamp_rms_retention': ratios(pooled['clipped_prediction_energy'], pooled['prediction_energy'], unchanged_zero=True),
        'speed': sum_stats(rows, 'speed'), 'acceleration': sum_stats(rows, 'acceleration'),
        'variogram': mean_supported('variogram'), 'velocity_covariance': mean_supported('velocity_covariance'),
        'short_runs': sum(row['short_runs'] for row in rows),
        'short_frames': sum(row['short_frames'] for row in rows)}


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []

    def equal(self, label, actual, expected):
        self.count += 1
        if isinstance(actual, (list, tuple, np.ndarray)) or isinstance(expected, (list, tuple, np.ndarray)):
            actual, expected = np.asarray(actual, dtype=object), np.asarray(expected, dtype=object)
            ok = actual.shape == expected.shape
            if ok:
                # None denotes unavailable support, not numerical NaN.
                mask_a = np.equal(actual, None)
                mask_b = np.equal(expected, None)
                ok = np.array_equal(mask_a, mask_b)
                if ok:
                    a, b = actual[~mask_a].astype(float), expected[~mask_b].astype(float)
                    ok = np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=RTOL, atol=ATOL)
        elif actual is None or expected is None:
            ok = actual is None and expected is None
        elif isinstance(actual, (float, int, np.number)) and not isinstance(actual, (bool, np.bool_)):
            ok = np.isfinite(actual) and np.isfinite(expected) and np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
        else:
            ok = actual == expected
        if not ok:
            def safe(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                return value
            self.failures.append({'check': label, 'actual': safe(actual), 'expected': safe(expected)})

    def truth(self, label, value):
        self.equal(label, bool(value), True)


def compare_metrics(checks, label, score, recorded, aggregate=False):
    for key in ('joint_fair_es', 'group_fair_es'):
        for kind in ('raw', 'centered'):
            checks.equal(label+'/'+key+'/'+kind, score[key][kind], recorded[key][kind])
    for key in ('rms_ratio', 'raw_oob', 'clamp_rms_retention'):
        checks.equal(label+'/'+key, score[key], recorded[key])
    checks.equal(label+'/variogram', score['variogram'], recorded['variogram']['aggregate'])
    checks.equal(label+'/velocity_covariance', score['velocity_covariance'], recorded['covariance_distance']['velocity'])
    for prefix, kind in (('', 'all'), ('reference_', 'reference_all')):
        for key in ('count', 'sum_squares'):
            checks.equal(label+'/speed/'+prefix+key, score['speed'][prefix+key], recorded['speed'][kind][key])
    if not aggregate:
        for key in ('sample_count', 'valid_frames', 'valid_runs'):
            checks.equal(label+'/'+key, score[key], recorded[key])
        for key in ('count', 'sum_squares', 'reference_count', 'reference_sum_squares'):
            checks.equal(label+'/acceleration/'+key, score['acceleration'][key], recorded['acceleration'][key])


def bootstrap(real, baseline, support):
    # Match the declared estimand: clip-equal gain and sentence-cluster
    # resampling, retaining all clips in each sampled sentence occurrence.
    groups = {}
    for cid in sorted(support):
        sentence = real[cid]['sentence']
        groups.setdefault(sentence, []).append(baseline[cid]['joint_fair_es']['centered']-real[cid]['joint_fair_es']['centered'])
    keys = sorted(groups)
    totals = np.array([np.sum(groups[key]) for key in keys])
    counts = np.array([len(groups[key]) for key in keys])
    rng = np.random.default_rng(20260918)
    draws = rng.integers(len(keys), size=(2000, len(keys)))
    boot = totals[draws].sum(1)/counts[draws].sum(1)
    return {'gain': float(totals.sum()/counts.sum()), 'ci95': np.quantile(boot, [.025, .975]).tolist()}


def not_worse(value, baseline):
    return (value is not None and baseline is not None and np.isfinite([value, baseline]).all()
            and value <= ((1.05*baseline) if baseline > 0 else baseline))


def audit(root):
    hashes = {}
    def read_json(name):
        path = root/name
        hashes[name] = sha(path)
        return json.loads(path.read_text(encoding='utf8'))
    def read_curves(name):
        path = root/name
        hashes[name] = sha(path)
        return torch.load(path, map_location='cpu', weights_only=False)
    protocol, status = read_json('protocol.json'), read_json('status.json')
    scales = np.asarray(read_json('scales.json'), dtype=np.float64)
    if scales.shape != (9,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Expected finite positive saved scales[9]')
    checks = Checks()
    checks.equal('completed_run', status.get('status'), 'complete')
    source_schema = protocol.get('schema')
    if source_schema not in ('clocked_motion_shape_development_v1', 'clocked_bounded_static_residual_v1'):
        raise ValueError('Unsupported fixed-clock result schema: '+str(source_schema))
    residual = source_schema == 'clocked_bounded_static_residual_v1'
    arms = PAIRED_ARMS if residual else ARMS
    level_key = 'logit_level' if residual else 'level'
    selection = read_json('selection.json') if residual else None
    checks.equal('protocol_schema', source_schema, status.get('schema'))
    checks.equal('test_not_loaded', protocol.get('test_loaded'), False)
    low_cut = read_json('training_data.json')['low_activity_train_quantile20']
    result = {'schema': SCHEMA, 'root': str(root.resolve()),
        'scope': 'Saved curves only; no native/dev405/sealed dataset access, no inference or training',
        'tolerance': {'rtol': RTOL, 'atol': ATOL,
            'reason': 'Saved predictions float32; original scoring before save float64'},
        'development_only': True, 'experiment_schema': source_schema, 'arms': list(arms),
        'selected_scale': selection['scale'] if residual else None, 'cells': {}}
    for cell in CELLS:
        reference = {row['clip_id']: row for row in protocol['split'][cell]}
        scores, summaries, curves_by_arm, reports = {}, {}, {}, {}
        cell_report = {'arms': {}}
        for arm in arms:
            label = cell+'_'+arm
            report = read_json(label+'.json')
            curves = read_curves(label+'.pt')
            ids = [row['clip_id'] for row in report['rows']]
            checks.truth(label+'/unique_ids', len(ids) == len(set(ids)))
            checks.truth(label+'/curve_ids_match_rows', set(ids) == set(curves))
            checks.truth(label+'/ids_in_protocol', set(ids) <= set(reference))
            if arm != 'mismatch':
                checks.truth(label+'/full_protocol_membership', set(ids) == set(reference))
            local = {}
            for row in report['rows']:
                cid = row['clip_id']
                item = rescore(curves[cid], scales)
                for key in ('sentence', 'speaker', 'emotion'):
                    checks.equal(label+'/'+cid+'/'+key, row[key], reference[cid][key])
                    item[key] = row[key]
                item['clip_id'] = cid
                item['low_activity'] = item['low_activity_energy'] <= low_cut
                checks.equal(label+'/'+cid+'/low_activity', item['low_activity'], row['low_activity'])
                checks.equal(label+'/'+cid+'/draws', item['sample_count'], 8)
                compare_metrics(checks, label+'/'+cid, item, row)
                local[cid] = item
            summary = summarize(list(local.values()))
            if summary is None:
                checks.equal(label+'/empty_summary', report['summary'], None)
            else:
                compare_metrics(checks, label+'/summary', summary, report['summary'], aggregate=True)
                for key in ('count', 'sum_squares', 'reference_count', 'reference_sum_squares', 'rms_ratio'):
                    checks.equal(label+'/acceleration/'+key, summary['acceleration'][key], report['acceleration'][key])
            low = summarize([row for row in local.values() if row['low_activity']])
            if low is not None:
                compare_metrics(checks, label+'/low_activity', low, report['low_activity'], aggregate=True)
            else:
                checks.equal(label+'/empty_low_activity', report['low_activity'], None)
            speakers = {str(s): summarize([row for row in local.values() if row['speaker'] == s])
                        for s in sorted({row['speaker'] for row in local.values()})}
            for speaker, speaker_summary in speakers.items():
                compare_metrics(checks, label+'/speaker/'+speaker, speaker_summary, report['by_speaker'][speaker], aggregate=True)
            scores[arm], summaries[arm], curves_by_arm[arm], reports[arm] = local, summary, curves, report
            cell_report['arms'][arm] = {'summary': summary, 'low_activity': low,
                                        'by_speaker': speakers}
        for arm in arms[1:]:
            for cid in set(curves_by_arm['temporal']) & set(curves_by_arm[arm]):
                left, right = curves_by_arm['temporal'][cid], curves_by_arm[arm][cid]
                for key in ('target', 'valid', level_key):
                    checks.truth(cell+'/'+arm+'/'+cid+'/same_'+key,
                        np.array_equal(numpy(left[key]), numpy(right[key]), equal_nan=True))
                checks.truth(cell+'/'+arm+'/'+cid+'/same_clock', left['locations'] == right['locations'])
        if residual:
            # Static input cancels the residual at every scale, not merely at
            # initialization; common random keys must then give exact equality.
            for cid in set(curves_by_arm['static']) & set(curves_by_arm['static_trained']):
                for key in ('samples', 'probabilities', 'tokens'):
                    checks.truth(cell+'/'+cid+'/static_cancellation/'+key,
                        np.array_equal(numpy(curves_by_arm['static'][cid][key]),
                                       numpy(curves_by_arm['static_trained'][cid][key])))
            if selection['scale'] == 0:
                for arm in PAIRED_ARMS:
                    for cid in set(curves_by_arm[arm]) & set(curves_by_arm['static_trained']):
                        checks.truth(cell+'/'+arm+'/'+cid+'/zero_scale_exact',
                            np.array_equal(numpy(curves_by_arm[arm][cid]['samples']),
                                           numpy(curves_by_arm['static_trained'][cid]['samples'])))
        support = set.intersection(*(set(scores[key]) for key in PAIRED_ARMS))
        sentences = {scores['temporal'][cid]['sentence'] for cid in support}
        coverage = len(support)/max(len(scores['temporal']), 1)
        assessment = read_json(cell+'_assessment.json')
        paired = {'clips': len(support), 'sentences': len(sentences), 'coverage': coverage}
        if len(support) >= 2 and len(sentences) >= 2 and coverage >= .70:
            gains = {arm: bootstrap(scores['temporal'], scores[arm], support) for arm in PAIRED_ARMS[1:]}
            for arm, gain in gains.items():
                for key in ('gain', 'ci95'):
                    checks.equal(cell+'/gain/'+arm+'/'+key, gain[key], assessment['gains_baseline_minus_real'][arm][key])
            real = summarize([scores['temporal'][cid] for cid in sorted(support)])
            base = summarize([scores['static_trained'][cid] for cid in sorted(support)])
            relative = {'raw_es': bool(not_worse(real['joint_fair_es']['raw'], base['joint_fair_es']['raw'])),
                        'variogram': bool(not_worse(real['variogram'], base['variogram'])),
                        'velocity_covariance': bool(not_worse(real['velocity_covariance'], base['velocity_covariance']))}
            summary = summaries['temporal']
            rms = np.asarray(summary['rms_ratio'], dtype=float)
            speed = summary['speed']['rms_ratio']
            acc = summary['acceleration']
            quality = {'group_rms': bool(rms.shape == (4,) and np.isfinite(rms).all() and ((rms >= .65) & (rms <= 1.5)).all()),
                'speed': bool(speed is not None and np.isfinite(speed) and speed <= 1.5),
                'acceleration': bool(acc['count'] > 0 and acc['reference_count'] > 0 and acc['reference_sum_squares'] > 0
                    and acc['rms_ratio'] is not None and np.isfinite(acc['rms_ratio']) and acc['rms_ratio'] <= 1.5),
                'raw_domain': bool(max(summary['raw_oob']) <= .01),
                'clamp_retention': bool(min(summary['clamp_rms_retention']) >= .95)}
            timing = bool(gains['static_trained']['ci95'][0] > 0
                and gains['static_trained']['gain'] >= .01*base['joint_fair_es']['centered']
                and all(gains[key]['gain'] > 0 for key in ('static', 'reverse', 'mismatch'))
                and all(relative.values()))
            if residual and selection['scale'] == 0:
                timing = False
            for key, value in relative.items():
                checks.equal(cell+'/relative/'+key, value, assessment['relative_protection'][key])
            for key, value in quality.items():
                checks.equal(cell+'/quality/'+key, value, assessment['quality_checks'][key])
            checks.equal(cell+'/timing_passed', timing, assessment['timing_passed'])
            checks.equal(cell+'/quality_passed', all(quality.values()), assessment['quality_passed'])
            checks.equal(cell+'/support', len(support), assessment['support'])
            checks.equal(cell+'/support_fraction', coverage, assessment['support_fraction'])
            paired.update(gains_baseline_minus_real=gains, timing_passed=timing,
                          quality_passed=all(quality.values()), quality_checks=quality,
                          failed_quality_checks=[key for key, value in quality.items() if not value],
                          relative_protection=relative)
        else:
            checks.equal(cell+'/unsupported_timing', assessment['timing_passed'], False)
            paired['reason'] = 'Insufficient shared intervention support for timing assessment'
        cell_report['paired_assessment'] = paired
        result['cells'][cell] = cell_report
        print('AUDITED', cell, len(reference), 'clips;', len(support), 'paired;', len(checks.failures), 'failures', flush=True)
    result.update(checks=checks.count, failures=checks.failures,
                  all_checks_passed=not checks.failures, input_sha256=hashes,
                  audit_script_sha256=sha(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    try:
        result = audit(root)
    except Exception as exc:
        result = {'schema': SCHEMA, 'root': str(root), 'all_checks_passed': False,
                  'error': repr(exc), 'traceback': traceback.format_exc(),
                  'audit_script_sha256': sha(__file__)}
    (root/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
        allow_nan=False, default=json_default), encoding='utf8')
    print('CLOCKED_AUDIT', json.dumps({'passed': result['all_checks_passed'],
          'checks': result.get('checks'), 'failures': len(result.get('failures', [])),
          'error': result.get('error')}), flush=True)
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
