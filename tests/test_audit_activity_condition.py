"""Saved-probability auditing contracts; no real-data or model quality claim."""
import copy
import json

import numpy as np
import pytest
import torch

from scripts import audit_activity_condition as audit


def fixture():
    rows = []
    for index, count in enumerate((1, 3, 2)):
        target = np.full((count, 4), bool(index % 2))
        probability = np.stack([np.full((count, 4), .1+.2*seed+.1*index) for seed in range(3)])
        values = {arm: probability.copy() for arm in audit.ARMS}
        donor = 'other' if index < 2 else None
        if donor is None:
            del values['mismatch']
        rows.append({'clip_id': f'c{index}', 'sentence': f's{index}', 'speaker': 0, 'emotion': 1,
                     'starts': list(range(0, count*8, 8)), 'energy': np.where(target, .2, .1),
                     'target': target, 'probabilities': values, 'donor_id': donor})
    return rows, [1, 2, 3], [.1]*4


def test_ensemble_averages_probabilities_before_brier_and_clips_equally():
    rows, seeds, thresholds = fixture()
    out = audit.rescore(rows, seeds, thresholds)
    # Clip probabilities are .3,.4,.5; targets 0,1,0. Count weighting must
    # not turn these three equally weighted clips into six equal windows.
    expected = np.mean([.3**2, .6**2, .5**2])
    np.testing.assert_allclose(out['all']['real']['ensemble_brier'], expected)
    assert out['all']['real']['ensemble_brier'] < np.mean(out['all']['real']['per_seed_brier'])
    assert not np.isclose(expected, np.average([.3**2, .6**2, .5**2], weights=[1, 3, 2]))
    assert out['common_ids'] == ['c0', 'c1'] and out['coverage'] == 2/3
    assert out['common']['real']['clips'] == out['common']['mismatch']['clips'] == 2
    np.testing.assert_allclose(out['all']['real']['event_rate'], [1/3]*4)


def test_target_threshold_is_strict_and_static_cancellation_exact():
    rows, seeds, thresholds = fixture()
    audit.validate_rows(rows, seeds, thresholds)
    bad = copy.deepcopy(rows); bad[0]['target'][0, 0] = True
    with pytest.raises(ValueError, match='strict energy'):
        audit.validate_rows(bad, seeds, thresholds)
    bad = copy.deepcopy(rows); bad[0]['probabilities']['static'][0, 0, 0] += 1e-12
    with pytest.raises(ValueError, match='exact cancellation'):
        audit.validate_rows(bad, seeds, thresholds)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.01, 1.01])
def test_reject_invalid_probabilities(value):
    rows, seeds, thresholds = fixture(); rows[0]['probabilities']['real'][0, 0, 0] = value
    with pytest.raises(ValueError, match='probability'):
        audit.validate_rows(rows, seeds, thresholds)


def test_reject_duplicate_ids_wrong_seed_count_or_donor_support():
    rows, seeds, thresholds = fixture()
    bad = copy.deepcopy(rows); bad[1]['clip_id'] = bad[0]['clip_id']
    with pytest.raises(ValueError, match='Unique nonempty clip'):
        audit.validate_rows(bad, seeds, thresholds)
    with pytest.raises(ValueError, match='probability'):
        audit.validate_rows(rows, seeds[:2], thresholds)
    bad = copy.deepcopy(rows); bad[0]['donor_id'] = None
    with pytest.raises(ValueError, match='support disagree'):
        audit.validate_rows(bad, seeds, thresholds)


def test_empty_shared_support_remains_unavailable():
    rows, seeds, thresholds = fixture()
    for row in rows:
        row['donor_id'] = None; row['probabilities'].pop('mismatch', None)
    out = audit.rescore(rows, seeds, thresholds)
    assert out['coverage'] == 0 and out['common_ids'] == []
    assert all(value is None for value in out['common'].values())


def test_reject_energy_nan_or_invalid_window_clock():
    rows, seeds, thresholds = fixture()
    bad = copy.deepcopy(rows); bad[1]['starts'] = [0, 8, 8]
    with pytest.raises(ValueError, match='sorted starts'):
        audit.validate_rows(bad, seeds, thresholds)
    bad = copy.deepcopy(rows); bad[0]['energy'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite nonnegative'):
        audit.validate_rows(bad, seeds, thresholds)


def test_weighted_threshold_does_not_overweight_long_clip():
    rows = [{'energy': np.full((1, 4), .1)}, {'energy': np.full((100, 4), .2)},
            {'energy': np.full((1, 4), .3)}]
    np.testing.assert_allclose(audit.fit_thresholds_saved(rows), [.2]*4)


def test_completed_saved_run_audit_and_report_tampering(tmp_path):
    # Driver is used only to produce a synthetic on-disk report fixture. The
    # audit implementation itself never imports driver/model/dataset code.
    from scripts import train_activity_condition as driver
    rows, seeds, _ = fixture()
    thresholds = audit.fit_thresholds_saved(rows)
    for row in rows:
        row['target'] = row['energy'] > thresholds
        row['raw_energy'] = row['energy']*2
        row['coverage'] = {'windows': len(row['starts'])}
    for row in rows[:2]:
        row['donor_id'] = rows[1-int(row['clip_id'][1:])]['clip_id']
    cells = {cell: copy.deepcopy(rows) for cell in audit.CELLS}
    protocol = {'schema': driver.SCHEMA, 'test_loaded': False, 'seeds': seeds,
                'split': {cell: [{k: row[k] for k in ('clip_id', 'sentence', 'speaker', 'emotion')} for row in rows]
                          for cell in audit.CELLS}}
    reports = {cell: driver.assessment(value) for cell, value in cells.items()}
    diagnostics = {}
    for cell, cell_rows in cells.items():
        diagnostics[cell] = {'source_clips': len(rows), 'scored_clips': len(rows), 'excluded': [],
            'windows': sum(len(row['starts']) for row in rows),
            'prevalence': np.mean([r['target'].mean(0) for r in rows], axis=0).tolist(),
            'mean_energy': np.mean([r['energy'].mean(0) for r in rows], axis=0).tolist(),
            'raw_mean_energy': np.mean([r['raw_energy'].mean(0) for r in rows], axis=0).tolist(),
            'zero_energy_clip_fraction': np.mean([np.all(r['energy'] <= 1e-8, axis=0) for r in rows], axis=0).tolist(),
            'coverage': {r['clip_id']: r['coverage'] for r in rows}}
    status = {'schema': driver.SCHEMA, 'status': 'complete', 'smoke': True, 'activity_passed': False}
    for name, value in [('protocol.json', protocol), ('status.json', status), ('reports.json', reports),
                        ('target_diagnostics.json', diagnostics), ('thresholds.json', {'thresholds': thresholds,
                            'quantile': .65, 'fit_clip_ids': [r['clip_id'] for r in rows]})]:
        (tmp_path/name).write_text(json.dumps(value), encoding='utf8')
    torch.save({'schema': driver.SCHEMA, 'seeds': seeds, 'cells': cells}, tmp_path/'predictions.pt')
    checked = audit.audit(tmp_path)
    assert checked['all_checks_passed'], checked['failures']
    assert checked['checks'] > 300
    assert not checked['bootstrap_recomputed'] and not checked['target_from_raw_motion_recomputed']
    reports['confirmation']['paired']['real']['brier'] += .01
    (tmp_path/'reports.json').write_text(json.dumps(reports), encoding='utf8')
    rejected = audit.audit(tmp_path)
    assert not rejected['all_checks_passed']
    assert any(row['check'] == 'confirmation/paired/real/brier' for row in rejected['failures'])
