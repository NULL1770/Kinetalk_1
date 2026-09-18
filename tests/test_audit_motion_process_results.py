"""Independent corrupt-artifact regressions for the saved process-prior audit."""
import copy

import numpy as np
import pytest
import torch

from scripts import audit_motion_process_results as audit
from scripts import train_motion_process as p
from scripts.motion_process_representation import fit_motion_process, render_motion_process


def fixture():
    valid = np.r_[np.ones(12, bool), np.zeros(3, bool), np.ones(26, bool)]
    target = np.tile((.35 + .08*np.sin(np.arange(41)*.2))[:, None], (1, 4))
    target[~valid] = 0.
    scale = np.array([.1, .2, .15, .12])
    anchor = np.array([.2, .15, .21, .11])
    plan = fit_motion_process(target, valid, scale)
    rng = np.random.default_rng(721)
    normalized = ((target-anchor)/scale)[None] + rng.normal(0, .04, (8, 41, 4))
    normalized[:, ~valid] = 0.
    raw = normalized*scale+anchor
    saved = {'samples': raw.astype(np.float32), 'target': target, 'teacher': render_motion_process(plan),
             'valid': valid, 'anchor': anchor}
    row = {'clip_id': 'one', 'speaker': 1, 'sentence': 'hello', 'emotion': 2,
        'event_nll': .2, 'initial_nll': 1., 'duration_brier': .4,
        'out_of_domain': ((raw[:, valid] < 0)|(raw[:, valid] > 1)).mean((0, 1)).tolist(),
        **p.distribution_metrics(normalized, (target-anchor)/scale, valid)}
    return saved, scale, plan, row


def test_inverse_anchor_and_scale_restore_all_eight_draw_metrics():
    saved, scales, plan, row = fixture()
    audit.validate_draws(saved, scales, plan, 'fixture')
    rescored, error, ambiguous = audit.rescore_draw(saved, scales, row, 'fixture')
    assert error < audit.METRIC_ATOL
    assert max(ambiguous) == 0
    assert rescored['clip_id'] == 'one'
    bad = copy.deepcopy(row)
    bad['raw_es'] = p.distribution_metrics(saved['samples'], (saved['target']-saved['anchor'])/scales,
                                          saved['valid'])['raw_es']
    with pytest.raises(ValueError, match='raw_es'):
        audit.rescore_draw(saved, scales, bad, 'wrong_coordinates')


@pytest.mark.parametrize('corruption', ['seed_count', 'teacher', 'plan_scale', 'invalid_draw', 'nan'])
def test_draw_validation_detects_corrupt_payloads(corruption):
    saved, scales, plan, row = fixture()
    if corruption == 'seed_count': saved['samples'] = saved['samples'][:7]
    elif corruption == 'teacher': saved['teacher'][0, 0] += .01
    elif corruption == 'plan_scale': plan['scales'][0] += .01
    elif corruption == 'invalid_draw': saved['samples'][:, 12] += .01
    elif corruption == 'nan': saved['samples'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError): audit.validate_draws(saved, scales, plan, 'bad')


def test_out_of_domain_rounding_exception_only_applies_exact_zero_or_one():
    saved, scales, plan, row = fixture()
    row['out_of_domain'][0] = .1
    with pytest.raises(ValueError, match='Out-of-domain'):
        audit.rescore_draw(saved, scales, row, 'bad')
    saved['samples'][0, 0, 0] = 1.
    corrected = p.distribution_metrics((saved['samples'].astype(float)-saved['anchor'])/scales,
                                        (saved['target']-saved['anchor'])/scales, saved['valid'])
    row.update(corrected)
    row['out_of_domain'][0] = 1/(8*saved['valid'].sum())
    _, _, ambiguity = audit.rescore_draw(saved, scales, row, 'rounded_boundary')
    assert ambiguity[0] == row['out_of_domain'][0]


def test_budget_replays_actual_fit_indices_and_exact_epochs():
    fit = [3, 5, 9, 18, 22]
    losses = [{'epoch': epoch, 'loss': .3, 'updates': 1, 'seconds': .1} for epoch in range(1, 31)]
    matching = {'order_sha256': audit.expected_order(fit, 30), 'updates': 30}
    assert audit.validate_budget(losses, matching, fit, 30) == matching
    for mutation in ('order', 'updates', 'epoch'):
        bad_loss, bad_match = copy.deepcopy(losses), matching.copy()
        if mutation == 'order': bad_match['order_sha256'] = audit.expected_order(range(5), 30)
        elif mutation == 'updates': bad_match['updates'] -= 1
        else: bad_loss[3]['epoch'] = 3
        with pytest.raises(ValueError): audit.validate_budget(bad_loss, bad_match, fit, 30)


def test_normalization_fit_ids_must_match_exact_membership_and_order():
    meta = lambda cid, speaker, sentence: {'clip_id': cid, 'speaker': speaker, 'sentence': sentence, 'emotion': 0}
    split = {'fit': [meta('a', 0, 'train'), meta('b', 0, 'train')],
        'sentence': [meta('c', 0, 'held')], 'speaker': [meta('d', 1, 'train')], 'joint': [meta('e', 1, 'held')]}
    protocol = {'split': split, 'smoke': True}
    norm = {'fit_clip_ids': ['a', 'b']}
    assert audit.validate_split(protocol, norm, ['a', 'c', 'd', 'b', 'e']) == [0, 3]
    for ids in (['b', 'a'], ['a', 'c'], ['a'], ['a', 'b', 'c']):
        with pytest.raises(ValueError, match='Normalization'):
            audit.validate_split(protocol, {'fit_clip_ids': ids}, ['a', 'c', 'd', 'b', 'e'])


def test_aggregate_audit_checks_grouped_metrics_not_only_overall_summary():
    _, _, _, row = fixture()
    report = {'rows': [row], 'summary': p.aggregate([row]), 'by_speaker': {'1': p.aggregate([row])},
              'by_emotion': {'2': p.aggregate([row])}}
    audit.validate_aggregates(report, 'report')
    report['by_speaker']['1']['event_nll'] += .1
    with pytest.raises(ValueError, match='by_speaker'):
        audit.validate_aggregates(report, 'bad')


def test_representation_gate_requires_all_groups_and_holdouts():
    cells = {cell: {'reconstructed': {'segments_per_second': 8., 'groups': {
        name: {'centered_r2': .8, 'correlation': .9, 'rms_ratio': 1.} for name in p.GROUPS}}}
        for cell in audit.HOLDS}
    assert audit.representation_gate(cells)
    cells['joint']['reconstructed']['groups']['eye_wide']['correlation'] = .79
    assert not audit.representation_gate(cells)


def test_comparator_rejects_missing_structure_and_nonfinite_values():
    with pytest.raises(ValueError): audit.compare({'x': [1.]}, {'x': [1.], 'y': 2})
    with pytest.raises(ValueError): audit.compare({'x': [1.]}, {'x': [float('nan')]})
    with pytest.raises(ValueError): audit.compare({'x': 1}, {'x': True})
