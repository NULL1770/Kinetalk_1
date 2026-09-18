"""Rescore saved activity probabilities without datasets or model inference.

This audit reads only a completed run's saved predictions and JSON metadata.
It independently verifies target thresholding, probability ranges, exact
static cancellation, shared intervention membership, and clip-balanced Brier
scores. It does not fit thresholds, rerun inference, or reinterpret bootstrap
confidence intervals as independent-test evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch


SCHEMA = 'activity_condition_saved_prediction_audit_v1'
ARMS = ('static_trained', 'real', 'static', 'reverse', 'shift', 'mismatch')
CELLS = ('fit', 'calibration', 'confirmation')
RTOL, ATOL = 1e-6, 1e-8


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(rows, seeds, thresholds):
    """Validate stored labels and arrays; do not infer labels from predictions."""
    thresholds = numpy(thresholds).astype(np.float64)
    if thresholds.shape != (4,) or not np.isfinite(thresholds).all() or (thresholds < 1e-6).any():
        raise ValueError('Four finite fitted thresholds >=1e-6 required')
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Unique integer seeds required')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty cell rows required')
    ids = [row['clip_id'] for row in rows]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError('Unique nonempty clip IDs required')
    for row in rows:
        if not isinstance(row['sentence'], str) or not row['sentence']:
            raise ValueError('Nonempty sentence ID required')
        starts = numpy(row['starts'])
        energy, target = numpy(row['energy']), numpy(row['target'])
        if (starts.ndim != 1 or not len(starts) or not np.issubdtype(starts.dtype, np.integer)
                or (starts < 0).any() or (np.diff(starts) <= 0).any()
                or energy.shape != (len(starts), 4) or not np.isfinite(energy).all()
                or (energy < 0).any() or target.shape != energy.shape or target.dtype != np.bool_):
            raise ValueError('Original sorted starts and finite nonnegative [N,4] energies with Boolean targets required')
        if not np.array_equal(target, energy > thresholds):
            raise ValueError('Saved target differs from strict energy > fitted threshold')
        if 'raw_energy' in row:
            raw = numpy(row['raw_energy'])
            if raw.shape != energy.shape or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError('Finite nonnegative raw energies matching targets required')
        probabilities = row['probabilities']
        if not set(ARMS[:-1]) <= set(probabilities) or not set(probabilities) <= set(ARMS):
            raise ValueError('All five primary arms and only declared intervention arms required')
        donor = row.get('donor_id')
        if (donor is not None) != ('mismatch' in probabilities):
            raise ValueError('Mismatch arm and donor ID support disagree')
        if donor is not None and (not isinstance(donor, str) or not donor or donor == row['clip_id']):
            raise ValueError('Mismatch needs a different nonempty donor ID')
        for arm, value in probabilities.items():
            p = numpy(value)
            if (p.shape != (len(seeds), len(starts), 4) or not np.isfinite(p).all()
                    or (p < 0).any() or (p > 1).any()):
                raise ValueError('Finite probability [seed,window,4] in [0,1] required: '+arm)
        if not np.array_equal(numpy(probabilities['static']), numpy(probabilities['static_trained'])):
            raise ValueError('Static input failed exact cancellation')


def clip_score(row, arm):
    p = numpy(row['probabilities'][arm]).astype(np.float64)
    y = numpy(row['target']).astype(np.float64)
    ensemble = p.mean(0)
    group = np.square(ensemble-y).mean(0)
    seed_group = np.square(p-y[None]).mean(1)
    # BCE is supplementary only. Clipping is numerical log protection, never
    # used by the proper Brier objective or by validation of probability range.
    q = np.clip(ensemble, 1e-7, 1-1e-7)
    return {'clip_id': row['clip_id'], 'sentence': row['sentence'],
            'speaker': row['speaker'], 'emotion': row['emotion'], 'windows': len(y),
            'ensemble_brier': float(group.mean()), 'group_brier': group.tolist(),
            'per_seed_brier': seed_group.mean(1).tolist(),
            'per_seed_group_brier': seed_group.tolist(),
            'event_rate': y.mean(0).tolist(),
            'group_bce': (-(y*np.log(q)+(1-y)*np.log1p(-q)).mean(0)).tolist(),
            'ensemble_bce': float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean())}


def summarize(rows, arm):
    """Every clip has equal weight irrespective of number of windows."""
    scored = [clip_score(row, arm) for row in rows if arm in row['probabilities']]
    if not scored:
        return None
    mean = lambda key: np.mean([row[key] for row in scored], axis=0)
    return {'clips': len(scored), 'sentences': len({row['sentence'] for row in scored}),
            'windows': sum(row['windows'] for row in scored),
            'ensemble_brier': float(mean('ensemble_brier')),
            'group_brier': mean('group_brier').tolist(),
            'per_seed_brier': mean('per_seed_brier').tolist(),
            'per_seed_group_brier': mean('per_seed_group_brier').tolist(),
            'event_rate': mean('event_rate').tolist(),
            'group_bce': mean('group_bce').tolist(),
            'ensemble_bce': float(mean('ensemble_bce'))}


def rescore(rows, seeds, thresholds):
    validate_rows(rows, seeds, thresholds)
    common = [row for row in rows if all(arm in row['probabilities'] for arm in ARMS)]
    return {'all': {arm: summarize(rows, arm) for arm in ARMS},
            'common': {arm: summarize(common, arm) for arm in ARMS},
            'common_ids': sorted(row['clip_id'] for row in common),
            'per_clip': {arm: [clip_score(row, arm) for row in rows if arm in row['probabilities']]
                         for arm in ARMS},
            'coverage': len(common)/len(rows)}


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []

    def equal(self, label, actual, expected):
        self.count += 1
        if isinstance(actual, (int, float, list, tuple, np.ndarray)) and not isinstance(actual, bool):
            a, b = np.asarray(actual), np.asarray(expected)
            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                ok = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=RTOL, atol=ATOL)
            else:
                ok = np.array_equal(a, b)
        else:
            ok = actual == expected
        if not ok:
            self.failures.append({'check': label, 'actual': actual, 'expected': expected})


def compare_summary(checks, label, actual, recorded, clip_rows=None):
    if actual is None:
        checks.equal(label+'/clips', 0, recorded['clips'])
        checks.equal(label+'/rows', [], recorded['rows'])
        checks.equal(label+'/brier', None, recorded['brier'])
        checks.equal(label+'/group_brier', None, recorded['group_brier'])
        return
    for own, report in (('clips', 'clips'), ('ensemble_brier', 'brier'),
                        ('group_brier', 'group_brier'), ('ensemble_bce', 'bce'), ('event_rate', 'prevalence')):
        checks.equal(label+'/'+report, actual[own], recorded[report])
    if clip_rows is not None:
        checks.equal(label+'/clip_order', [r['clip_id'] for r in clip_rows], [r['clip_id'] for r in recorded['rows']])
        for own, report in zip(clip_rows, recorded['rows']):
            for key in ('clip_id', 'sentence', 'speaker', 'emotion'):
                checks.equal(label+'/'+own['clip_id']+'/'+key, own[key], report[key])
            for own_key, report_key in (('group_brier', 'brier'), ('group_bce', 'bce'), ('event_rate', 'prevalence')):
                checks.equal(label+'/'+own['clip_id']+'/'+report_key, own[own_key], report[report_key])


def fit_thresholds_saved(rows):
    """Independent inverse-CDF weighted quantile from saved fitting energies."""
    values = np.concatenate([numpy(row['energy']).astype(np.float64) for row in rows])
    weight = np.concatenate([np.full(len(row['energy']), 1./len(row['energy'])) for row in rows])
    weight /= weight.max()
    result = []
    for group in range(4):
        order = np.argsort(values[:, group], kind='stable')
        index = min(np.searchsorted(np.cumsum(weight[order]), .65*weight.sum(), side='left'), len(order)-1)
        result.append(max(float(values[order[index], group]), 1e-6))
    return result


def compare_diagnostics(checks, cell, rows, reference, recorded):
    excluded = recorded['excluded']
    ids = {row['clip_id'] for row in rows}
    checks.equal(cell+'/diagnostics/source_clips', len(reference), recorded['source_clips'])
    checks.equal(cell+'/diagnostics/scored_clips', len(rows), recorded['scored_clips'])
    checks.equal(cell+'/diagnostics/excluded', sorted(set(reference)-ids), sorted(excluded))
    checks.equal(cell+'/diagnostics/no_duplicate_excluded', len(excluded), len(set(excluded)))
    checks.equal(cell+'/diagnostics/windows', sum(len(row['starts']) for row in rows), recorded['windows'])
    checks.equal(cell+'/diagnostics/coverage_membership', sorted(reference), sorted(recorded['coverage']))
    for key, value in (('prevalence', np.mean([numpy(r['target']).mean(0) for r in rows], axis=0)),
                       ('mean_energy', np.mean([numpy(r['energy']).mean(0) for r in rows], axis=0)),
                       ('raw_mean_energy', np.mean([numpy(r['raw_energy']).mean(0) for r in rows], axis=0)),
                       ('zero_energy_clip_fraction', np.mean([np.all(numpy(r['energy']) <= 1e-8, axis=0) for r in rows], axis=0))):
        checks.equal(cell+'/diagnostics/'+key, value.tolist(), recorded[key])
    for row in rows:
        checks.equal(cell+'/'+row['clip_id']+'/coverage_binding', row['coverage'], recorded['coverage'][row['clip_id']])
        checks.equal(cell+'/'+row['clip_id']+'/coverage_windows', len(row['starts']), row['coverage']['windows'])


def audit(root):
    """Report bindings are intentionally explicit, with no inferred aliases."""
    root = Path(root)
    hashes = {}
    def read(name):
        hashes[name] = sha(root/name)
        return json.loads((root/name).read_text(encoding='utf8'))
    protocol, status = read('protocol.json'), read('status.json')
    thresholds, reports = read('thresholds.json'), read('reports.json')
    diagnostics = read('target_diagnostics.json')
    hashes['predictions.pt'] = sha(root/'predictions.pt')
    saved = torch.load(root/'predictions.pt', map_location='cpu', weights_only=False)
    checks = Checks()
    checks.equal('completed_run', status['status'], 'complete')
    checks.equal('schema_binding', protocol['schema'], status['schema'])
    checks.equal('test_not_loaded', protocol['test_loaded'], False)
    checks.equal('seed_binding', saved['seeds'], protocol['seeds'])
    checks.equal('evaluation_cells', sorted(saved['cells']), sorted(CELLS))
    checks.equal('prediction_schema', saved['schema'], protocol['schema'])
    checks.equal('fitted_quantile', thresholds['quantile'], .65)
    checks.equal('fitted_thresholds', fit_thresholds_saved(saved['cells']['fit']), thresholds['thresholds'])
    checks.equal('fitted_threshold_membership', sorted(r['clip_id'] for r in saved['cells']['fit']),
                 sorted(thresholds['fit_clip_ids']))
    result = {'schema': SCHEMA, 'scope': 'Saved prediction arrays only; no native data, model inference or training',
              'root': str(root.resolve()), 'tolerance': {'rtol': RTOL, 'atol': ATOL}, 'cells': {}}
    for cell in CELLS:
        rows = saved['cells'][cell]
        computed = rescore(rows, saved['seeds'], thresholds['thresholds'])
        reference = {row['clip_id']: row for row in protocol['split'][cell]}
        checks.equal(cell+'/unique_protocol_ids', len(reference), len(protocol['split'][cell]))
        for row in rows:
            if row['clip_id'] not in reference:
                raise ValueError('Prediction outside declared cell membership: '+row['clip_id'])
            for key in ('sentence', 'speaker', 'emotion'):
                checks.equal(cell+'/'+row['clip_id']+'/'+key, row[key], reference[row['clip_id']][key])
            donor = row.get('donor_id')
            if donor is not None:
                if donor not in reference:
                    raise ValueError('Donor outside declared cell membership: '+donor)
                for key in ('speaker', 'emotion'):
                    checks.equal(cell+'/'+row['clip_id']+'/donor_'+key, row[key], reference[donor][key])
                checks.equal(cell+'/'+row['clip_id']+'/donor_different_sentence', row['sentence'] != reference[donor]['sentence'], True)
        compare_diagnostics(checks, cell, rows, reference, diagnostics[cell])
        report = reports[cell]
        common = [row for row in rows if row['clip_id'] in set(computed['common_ids'])]
        checks.equal(cell+'/support', len(common), report['support'])
        checks.equal(cell+'/support_fraction', computed['coverage'], report['support_fraction'])
        checks.equal(cell+'/support_sentences', len({r['sentence'] for r in common}), report['support_sentences'])
        for arm in ARMS:
            compare_summary(checks, cell+'/all/'+arm, computed['all'][arm], report['all'][arm], computed['per_clip'][arm])
            if common:
                common_scored = [clip_score(row, arm) for row in common]
                compare_summary(checks, cell+'/paired/'+arm, computed['common'][arm], report['paired'][arm], common_scored)
        if common:
            base, real = computed['common']['static_trained'], computed['common']['real']
            for arm in ARMS:
                if arm != 'real':
                    checks.equal(cell+'/gain/'+arm, computed['common'][arm]['ensemble_brier']-real['ensemble_brier'],
                                 report['gains'][arm]['gain'])
            checks.equal(cell+'/brow_gain', float(np.mean(base['group_brier'][:2])-np.mean(real['group_brier'][:2])), report['brow_gain']['gain'])
            checks.equal(cell+'/per_seed_count', len(saved['seeds']), len(report['per_seed']))
            for index, row in enumerate(report['per_seed']):
                checks.equal(cell+'/seed_index/'+str(index), index, row['seed_index'])
                checks.equal(cell+'/seed_static/'+str(index), base['per_seed_brier'][index], row['static_brier'])
                checks.equal(cell+'/seed_real/'+str(index), real['per_seed_brier'][index], row['real_brier'])
                checks.equal(cell+'/seed_gain/'+str(index), base['per_seed_brier'][index]-real['per_seed_brier'][index], row['gain'])
            for key in ('emotion', 'speaker', 'sentence'):
                values = {row[key] for row in common}
                checks.equal(cell+'/slices/'+key, sorted(map(str, values)), sorted(report['slices'][key]))
                for value in values:
                    part = [row for row in common if row[key] == value]
                    for arm in ('static_trained', 'real'):
                        compare_summary(checks, cell+'/slices/'+key+'/'+str(value)+'/'+arm,
                            summarize(part, arm), report['slices'][key][str(value)][arm], [clip_score(row, arm) for row in part])
            # Verify bookkeeping and deterministic parts of the gate, while
            # leaving the bootstrap computation explicitly outside this audit.
            direct = {'support': computed['coverage'] >= .8 and len({r['sentence'] for r in common}) >= 8,
                      'group_protection': bool(np.all(np.asarray(real['group_brier']) <= 1.02*np.asarray(base['group_brier']))),
                      'nondegenerate_labels': bool(np.all((np.asarray(real['event_rate']) >= .05) & (np.asarray(real['event_rate']) <= .95))),
                      'training_seed_consistency': all(b-r > 0 for b, r in zip(base['per_seed_brier'], real['per_seed_brier']))}
            for key, value in direct.items():
                checks.equal(cell+'/checks/'+key, value, report['checks'][key])
            checks.equal(cell+'/passed_binding', all(report['checks'].values()), report['passed'])
        else:
            checks.equal(cell+'/no_common_support', report['passed'], False)
            checks.equal(cell+'/no_common_reason', report['reason'], 'no_common_support')
        result['cells'][cell] = computed
    checks.equal('final_activity_passed', all(reports[c]['passed'] for c in ('calibration', 'confirmation')) and not status['smoke'],
                 status['activity_passed'])
    result.update(checks=checks.count, failures=checks.failures, all_checks_passed=not checks.failures,
                  bootstrap_recomputed=False, target_from_raw_motion_recomputed=False,
                  input_sha256=hashes, audit_script_sha256=sha(__file__))
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
                  'error': repr(exc), 'traceback': traceback.format_exc(), 'audit_script_sha256': sha(__file__)}
    (root/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('ACTIVITY_AUDIT', json.dumps({'passed': result['all_checks_passed'], 'checks': result.get('checks'),
          'failures': len(result.get('failures', [])), 'error': result.get('error')}), flush=True)
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
