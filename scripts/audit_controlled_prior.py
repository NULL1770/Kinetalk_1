"""Replay saved controlled-prior conditions and audit results without data loads.

Generation is explicitly separated from scoring: only saved prior parameters,
independent reference style, predicted logit level, validity mask and seeded
key are passed to replay. Query GT is read solely after replay for metrics.
No fitting, checkpoint selection, tuning or source-result mutation occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import controlled_motion_process as motion
from scripts import joint_motion_metrics as metrics

SCHEMA = 'controlled_prior_replay_audit_v1'
ARMS = ('medoid', 'process', 'reference_swap')
GROUP_NAMES = ('up', 'down', 'squint', 'wide')
GAINS = (0., .5, 1., 1.5)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []

    def equal(self, name, actual, expected, exact=False, rtol=1e-6, atol=1e-8):
        if isinstance(expected, dict):
            self.equal(name+'/keys', sorted(actual), sorted(expected), exact=True)
            for key in expected:
                self.equal(name+'/'+str(key), actual.get(key), expected[key], exact, rtol, atol)
            return
        self.count += 1
        if expected is None or isinstance(expected, (str, bool)):
            okay = actual == expected
        elif isinstance(expected, list) and any(v is None or isinstance(v, (str, dict, list)) for v in expected):
            okay = len(actual) == len(expected)
            if okay:
                for index, (value, reference) in enumerate(zip(actual, expected)):
                    self.equal(name+'/'+str(index), value, reference, exact, rtol, atol)
                return
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            okay = a.shape == b.shape and (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=rtol, atol=atol))
        if not okay:
            self.failures.append({'check': name, 'detail': 'mismatch', 'exact': exact})


def runs(valid):
    boundaries = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)))


def replay(level, style, fitted, valid, seed, key, gain=1.):
    """No target, reference trajectory, audio or query-motion argument exists."""
    valid = np.asarray(valid, bool)
    output = np.zeros((len(valid), 9))
    for left, right in runs(valid):
        state = motion.initialize(level, style, fitted, seed, f'{key}:run:{left}', activity_gain=gain)
        output[left:right] = motion.sample(state, right-left)
    return output


def group_energy(raw, valid):
    values = np.asarray(raw, dtype=np.float64)
    energies = np.zeros(4); counts = np.zeros(4)
    for left, right in runs(valid):
        part = values[left:right]-values[left:left+1]
        part -= part.mean(0)
        for index, group in enumerate(motion.GROUPS):
            energies[index] += np.square(part[:, group]).sum()
            counts[index] += len(part)*len(group)
    return np.sqrt(energies/np.maximum(counts, 1))


def acceleration(samples, target, valid, scales):
    prediction, reference = [], []
    for left, right in runs(valid):
        if right-left >= 3:
            prediction.extend(np.square(np.diff(samples[:, left:right]/scales, n=2, axis=1)).reshape(-1))
            reference.extend(np.square(np.diff(target[left:right]/scales, n=2, axis=0)).reshape(-1))
    return {'sum_squares': float(np.sum(prediction)), 'count': len(prediction),
        'reference_sum_squares': float(np.sum(reference)), 'reference_count': len(reference)}


def trajectory_summary(raw):
    x = np.asarray(raw, np.float64); centered = x-x.mean(0)
    velocity = np.diff(x, axis=0)*25.; acceleration = np.diff(x, n=2, axis=0)*625.
    power = np.abs(np.fft.rfft(centered, axis=0))**2
    frequency = np.fft.rfftfreq(len(x), d=.04); total = power[1:].sum()
    autocorrelations = []
    for lag in (1, 4, 8, 16):
        first, second = centered[:-lag], centered[lag:]
        denominator = float(np.sqrt(np.sum(first*first)*np.sum(second*second)))
        autocorrelations.append(float(np.sum(first*second)/denominator) if denominator > 1e-12 else None)
    activity = np.linalg.norm(velocity, axis=1) > .05
    edge = np.flatnonzero(np.diff(np.r_[False, activity, False]))
    return {'group_rms': [float(np.sqrt(np.mean(centered[:, group]**2))) for group in motion.GROUPS],
        'velocity_p95': float(np.quantile(np.abs(velocity), .95)),
        'acceleration_p95': float(np.quantile(np.abs(acceleration), .95)),
        'above_3hz_energy_fraction': float(power[frequency > 3].sum()/total) if total > 1e-14 else 0.,
        'autocorrelation_lags_1_4_8_16': autocorrelations,
        'activity_fraction_speed_norm_gt_005': float(activity.mean()),
        'activity_run_seconds': ((edge[1::2]-edge[::2])*.04).tolist(),
        'finite': bool(np.isfinite(x).all()), 'in_domain': bool(((x >= 0)&(x <= 1)).all())}


def audit(root):
    root = Path(root).resolve()
    files = ('protocol.json', 'status.json', 'predictions.pt', 'prior.pt', 'references.json', 'reports.json',
             'long_controls.json', 'fit_dynamics_reference.json', 'query_selection.json')
    hashes = {name: sha(root/name) for name in files}
    protocol, status, reports = read(root/'protocol.json'), read(root/'status.json'), read(root/'reports.json')
    references, long_saved = read(root/'references.json'), read(root/'long_controls.json')
    fit_reference = read(root/'fit_dynamics_reference.json')
    selected = read(root/'query_selection.json')['queries']
    saved = torch.load(root/'predictions.pt', map_location='cpu', weights_only=False)
    fitted = torch.load(root/'prior.pt', map_location='cpu', weights_only=False)
    checks, replay_checks = Checks(), Checks()
    checks.equal('completed', status['status'], 'complete', exact=True)
    checks.equal('formal', status['smoke'], False, exact=True)
    checks.equal('schema', saved['schema'], protocol['schema'], exact=True)
    for filename in ('scripts/controlled_motion_process.py', 'scripts/joint_motion_metrics.py'):
        checks.equal('source_hash/'+filename, sha(Path(__file__).resolve().parents[1]/filename), protocol['code_sha256'][filename], exact=True)
    checks.equal('selection', [row['clip_id'] for row in selected], protocol['query_clip_ids'], exact=True)
    checks.equal('curves', sorted(saved['curves']), sorted(protocol['query_clip_ids']), exact=True)
    checks.equal('fit_membership', fitted['fit_clip_ids'], protocol['fit_clip_ids'], exact=True)
    if checks.failures:
        raise ValueError('Source/schema/selection binding failed before replay: '+str(checks.failures))
    seeds = protocol['seeds']; metric_rows = {arm: [] for arm in ARMS}
    saved_rows = {arm: {row['clip_id']: row for row in reports[arm]['rows']} for arm in ARMS}
    gain_rows, native_exact, max_native_error, speakers = [], True, 0., {}
    for metadata in selected:
        cid = metadata['clip_id']; curve = saved['curves'][cid]
        level, valid = np.asarray(curve['level']), np.asarray(curve['valid'])
        styles = [np.asarray(row['style']) for row in references[cid]['references']]
        source_ids = references[cid]['reference_ids']
        checks.equal(cid+'/references_fit', all(value in fitted['fit_clip_ids'] for value in source_ids), True, exact=True)
        for reference in references[cid]['reference_metadata']:
            checks.equal(cid+'/reference_speaker/'+reference['clip_id'], reference['speaker'], metadata['speaker'], exact=True)
            checks.equal(cid+'/reference_sentence/'+reference['clip_id'], reference['sentence'] != metadata['sentence'], True, exact=True)
            checks.equal(cid+'/reference_not_query/'+reference['clip_id'], reference['clip_id'] != cid, True, exact=True)
        generated = {}
        for arm, style in zip(('process', 'reference_swap'), styles):
            generated[arm] = np.stack([replay(level, style, fitted, valid, seed, cid) for seed in seeds])
            actual = generated[arm].astype(np.float32); expected = curve['samples'][arm]
            equal = np.array_equal(actual, expected); native_exact &= equal
            max_native_error = max(max_native_error, float(np.max(np.abs(actual-expected))))
            replay_checks.equal(cid+'/'+arm+'/replay_float32', actual, expected, exact=True)
        per_gain = {}
        for seed in seeds:
            per_gain[seed] = np.stack([replay(level, styles[0], fitted, valid, seed, cid, gain) for gain in GAINS])
            rms = np.stack([group_energy(value, valid) for value in per_gain[seed]])
            order = np.all(np.diff(rms, axis=0) >= -1e-12, axis=0)
            gain_rows.append({'clip_id': cid, 'seed': seed, 'group_rms_by_gain': rms.tolist(),
                'nondecreasing_each_group': order.tolist()})
            if seed == 42:
                for gain, values in zip(GAINS, per_gain[seed]):
                    replay_checks.equal(cid+'/gain42/'+str(gain), values.astype(np.float32), curve['gain_controls_seed42'][str(gain)], exact=True)
        # Only now load target for scoring. It was not passed to any replay.
        target = np.asarray(curve['target'])
        for arm in ARMS:
            scales = np.asarray(reports[arm]['summary']['scales'])
            samples = np.asarray(curve['samples'][arm], dtype=np.float64)
            row = {**metadata, 'low_activity': saved_rows[arm][cid]['low_activity'],
                'acceleration': acceleration(samples, target, valid, scales),
                **metrics.score_clip(samples, target, valid, scales)}
            metric_rows[arm].append(row)
            checks.equal(cid+'/'+arm+'/metrics', row, saved_rows[arm][cid], rtol=1e-5, atol=1e-8)
        speakers.setdefault(str(metadata['speaker']), cid)
        print('AUDIT_REPLAY', cid, flush=True)
    for arm in ARMS:
        summary = metrics.summarize(metric_rows[arm])
        checks.equal(arm+'/summary', summary, reports[arm]['summary'], rtol=1e-5, atol=1e-8)
    long_rows, long_exact = [], True
    for speaker, cid in speakers.items():
        curve = saved['curves'][cid]; level = np.asarray(curve['level'])
        styles = [np.asarray(row['style']) for row in references[cid]['references']]
        lookup = {(row['seed'], row['kind'], row.get('gain')): row for row in long_saved[speaker]}
        for seed in seeds:
            for gain in GAINS:
                state = motion.initialize(level, styles[0], fitted, seed, cid+':long', activity_gain=gain)
                raw = motion.sample(state, 1500)
                row = {'seed': seed, 'gain': gain, 'kind': 'stationary_60s', **trajectory_summary(raw),
                    'quarter_group_rms': [trajectory_summary(part)['group_rms'] for part in np.array_split(raw, 4)]}
                replay_checks.equal(cid+f'/long/{seed}/{gain}', row, lookup[(seed, 'stationary_60s', gain)], rtol=1e-6, atol=1e-8)
                long_rows.append({'speaker': speaker, **row})
                if seed == 42:
                    actual = raw.astype(np.float32); expected = saved['long'][cid]['gain_'+str(gain)]
                    long_exact &= np.array_equal(actual, expected)
                    replay_checks.equal(cid+'/savedlong/'+str(gain), actual, expected, exact=True)
                    actual_stats = trajectory_summary(expected.astype(np.float64))
                    # Float32 serialization of an exactly constant gain0
                    # series keeps exact velocities; centering may leave tiny
                    # offsets. Null autocorrelation is not a replay claim.
                    for metric in ('velocity_p95', 'acceleration_p95', 'above_3hz_energy_fraction',
                                   'activity_fraction_speed_norm_gt_005', 'finite', 'in_domain'):
                        checks.equal(cid+'/savedlong_metrics/'+str(gain)+'/'+metric, actual_stats[metric],
                            lookup[(seed, 'stationary_60s', gain)][metric], rtol=1e-5, atol=1e-6)
            state = motion.initialize(level, styles[0], fitted, seed, cid+':controls')
            raw = np.concatenate([motion.sample(state, 128), motion.sample(state, 64, 'hold'),
                motion.sample(state, 64, 'release'), motion.sample(state, 128, 'run'),
                motion.sample(state, 128, 'run', style=styles[1])])
            hold, release = raw[136:192], raw[208:256]
            edge = np.unique(np.concatenate([np.arange(t-8, t+8) for t in (128, 192, 256, 384)]))
            row = {'seed': seed, 'kind': 'run_hold_release_resume_swap',
                'hold_speed_max': float(np.abs(np.diff(hold, axis=0)).max()),
                'release_equilibrium_max_error': float(np.abs(release-expit(level)).max()),
                'boundary_velocity_p95': float(np.quantile(np.abs(np.diff(raw, axis=0)*25)[edge], .95)),
                'boundary_acceleration_p95': float(np.quantile(np.abs(np.diff(raw, n=2, axis=0)*625)[edge], .95)),
                'control_times_frames': [128, 192, 256, 384], **trajectory_summary(raw)}
            row['boundary_to_fit_median_ratio'] = {key: row['boundary_'+key]/max(
                fit_reference['quantiles'][key]['0.5'], 1e-12) for key in ('velocity_p95', 'acceleration_p95')}
            replay_checks.equal(cid+f'/controls/{seed}', row, lookup[(seed, 'run_hold_release_resume_swap', None)], rtol=1e-6, atol=1e-8)
            long_rows.append({'speaker': speaker, **row})
            if seed == 42:
                actual = raw.astype(np.float32); expected = saved['long'][cid]['controls']
                long_exact &= np.array_equal(actual, expected)
                replay_checks.equal(cid+'/savedcontrols', actual, expected, exact=True)
                checks.equal(cid+'/saved_hold_exact', np.abs(np.diff(expected[136:192], axis=0)).max(), 0., exact=True)
                checks.equal(cid+'/saved_release', expected[208:256],
                    np.broadcast_to(expit(level).astype(np.float32), (48, 9)), exact=True)
    monotonic = np.asarray([row['nondecreasing_each_group'] for row in gain_rows])
    local_controls = [row for row in long_rows if row['kind'] == 'run_hold_release_resume_swap']
    source_long = [row for rows in long_saved.values() for row in rows]
    controls = [row for row in source_long if row['kind'] == 'run_hold_release_resume_swap']
    unit_long = [row for row in source_long if row['kind'] == 'stationary_60s' and row['gain'] == 1.]
    boundary_ratios = {}
    for key in ('velocity_p95', 'acceleration_p95'):
        benchmark = fit_reference['quantiles'][key]['0.95']
        ratios = [row['boundary_'+key]/benchmark for row in controls]
        boundary_ratios[key] = {'fit_q95_of_clipmean_window_p95': benchmark,
            'ratios': ratios, 'maximum': float(max(ratios)), 'fraction_above_1_5': float(np.mean(np.asarray(ratios)>1.5)),
            'scope': 'Scale-mismatched diagnostic: long-control boundary p95 versus fit q95 of clip-mean H32 p95'}
    metrics_summary = {arm: reports[arm]['summary'] for arm in ARMS}
    result = {'schema': SCHEMA, 'status': 'complete', 'input_sha256': hashes, 'audit_script_sha256': sha(__file__),
        'checks': checks.count, 'failures': checks.failures, 'all_checks_passed': not checks.failures and not replay_checks.failures,
        'saved_metrics_checks_passed': not checks.failures,
        'replay_checks': replay_checks.count, 'replay_failures': replay_checks.failures,
        'replay_checks_passed': not replay_checks.failures,
        'native_replay_float32_exact': native_exact, 'native_replay_max_abs_error': max_native_error,
        'long_replay_float32_exact': long_exact, 'queries': len(selected), 'seeds': seeds,
        'generation_inputs': ['saved prior fit statistics', 'saved predicted logit level', 'independent reference style5',
                              'validity clock', 'seed/clip key', 'explicit activity gain'],
        'query_motion_used_for_generation': False, 'target_used_for_scoring_only': True,
        'metrics_recomputed_from': 'All metrics independently recomputed from actual saved float32 curves; original run scored float64 process/swap before saving.',
        'metric_tolerance': {'rtol': 1e-5, 'atol': 1e-8},
        'gain_monotonicity': {'gains': list(GAINS), 'group_order': list(GROUP_NAMES),
            'clip_seed_pairs': len(gain_rows), 'fraction_nondecreasing_by_group': monotonic.mean(0).tolist(),
            'all_groups_fraction': float(monotonic.all(1).mean()), 'rows': gain_rows,
            'scope': 'Local regenerated seeds; crosses platform and may differ from stored paths because eigenvector noise factors are nonunique'},
        'saved_gain_seed42': {'rows': [{'clip_id': cid,
            'group_rms_by_gain': [group_energy(curve['gain_controls_seed42'][str(g)], np.asarray(curve['valid'])).tolist() for g in GAINS],
            'nondecreasing_each_group': np.all(np.diff(np.stack([group_energy(curve['gain_controls_seed42'][str(g)],
                np.asarray(curve['valid'])) for g in GAINS]), axis=0)>=-1e-12, axis=0).tolist()}
            for cid, curve in saved['curves'].items()], 'scope': 'Actual saved native gain controls; only seed42 was serialized'},
        'hold_release': {'cases': len(controls), 'maximum_hold_speed': max(row['hold_speed_max'] for row in controls),
            'maximum_release_error': max(row['release_equilibrium_max_error'] for row in controls)},
        'boundary_diagnostics': boundary_ratios, 'metrics': metrics_summary,
        'unit_gain_60s': {'cases': len(unit_long),
            'activity_fraction_mean': float(np.mean([row['activity_fraction_speed_norm_gt_005'] for row in unit_long])),
            'above_3hz_energy_fraction_mean': float(np.mean([row['above_3hz_energy_fraction'] for row in unit_long])),
            'velocity_p95_mean': float(np.mean([row['velocity_p95'] for row in unit_long])),
            'acceleration_p95_mean': float(np.mean([row['acceleration_p95'] for row in unit_long]))},
        'style_time_constant': {'fit_lower': float(fitted['style_lower'][4]), 'fit_upper': float(fitted['style_upper'][4]),
            'reference_tau_values': sorted({float(row['style'][4]) for value in references.values() for row in value['references']})},
        'limits': 'Replay is platform-sensitive because eigenvector factor signs/order are not canonical. Saved-curve metric audit is separate. low_activity labels copied because original threshold artifact is not in this delivery. Query set is32 selected development examples; no audio timing, naturalness or whole-face certificate.',
        'source_results_modified': False, 'model_trained': False, 'native_data_loaded': False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh audit output required')
    result = audit(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({key: result[key] for key in ('all_checks_passed', 'checks', 'native_replay_float32_exact',
                                                 'long_replay_float32_exact', 'hold_release')}, ensure_ascii=False))
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
