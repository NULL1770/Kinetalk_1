"""Meaningful numeric and immutable-target contracts for post-hoc diagnosis."""
import copy
import json
import math

import numpy as np
import pytest
import torch

from scripts import diagnose_motion_process_failure as d
from scripts.motion_process_representation import fit_motion_process
from kinetalk_b0.models.motion_process_prior import initial_nll


def test_event_decomposition_and_direction_probabilities_match_known_gaussians():
    logits = torch.tensor([[0., 0.], [math.log(3), 0.]], dtype=torch.float64)
    loc = torch.tensor([[0., 1.], [-1., 2.]], dtype=torch.float64)
    log_scale = torch.zeros_like(loc)
    index = torch.tensor([0, 1])
    target = torch.tensor([1., -1.], dtype=torch.float64)
    result = d.event_components({'duration_logits': logits, 'loc': loc, 'log_scale': log_scale}, index, target)
    torch.testing.assert_close(result['duration_ce'], torch.tensor([math.log(2), math.log(4)], dtype=torch.float64))
    expected_nll = torch.tensor([.5, 4.5], dtype=torch.float64)+.5*math.log(2*math.pi)
    torch.testing.assert_close(result['gaussian_nll'], expected_nll)
    torch.testing.assert_close(result['joint_nll'], expected_nll+result['duration_ce'])
    torch.testing.assert_close(result['standardized_squared_error'], torch.tensor([1., 9.], dtype=torch.float64))
    torch.testing.assert_close(result['coverage_2std'], torch.tensor([1., 0.], dtype=torch.float64))
    normal = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))
    positive = normal.cdf(loc)
    conditional = positive[torch.arange(2), index]
    marginal = (positive*logits.softmax(-1)).sum(-1)
    torch.testing.assert_close(result['direction_brier_conditional'], (conditional-torch.tensor([1., 0.])).square())
    torch.testing.assert_close(result['direction_brier_marginal'], (marginal-torch.tensor([1., 0.])).square())


def test_saved_plan_reuse_never_refits_excludes_terminal_and_verifies_sources(monkeypatch):
    state = np.tile(np.sin(np.arange(67.)*.6)[:, None], (1, 4))
    valid = np.ones(67, bool); valid[24:28] = False
    mask = np.repeat(valid[:, None], 4, axis=1)
    scales = np.array([.4, .5, .6, .7])
    plan = fit_motion_process(state, mask, scales)
    clip = {'clip_id': 'clip', 'valid': torch.tensor(valid), 'state': state,
        'groupmask': mask, 'anchor': np.array([.1, .2, .3, .4])}
    before = copy.deepcopy(plan)
    def forbidden(*args, **kwargs):
        raise AssertionError('Diagnosis must not refit segmentation or normalization')
    monkeypatch.setattr(d.p, 'prepare_plans', forbidden)
    monkeypatch.setattr(d.p, 'fit_dynamic_scales', forbidden)
    monkeypatch.setattr(d.p, 'fit_feature_stats', forbidden)
    d.attach_saved_targets([clip], {'clip': plan}, scales)
    assert len(clip['events']) == sum(not s['right_censored'] for ss in plan['segments'] for s in ss)
    np.testing.assert_allclose(clip['initial'], (state[[0, 28]]-clip['anchor'])/scales, rtol=2e-7)
    assert plan['segments'] == before['segments']
    bad = copy.deepcopy(plan); bad['scales'][0] *= 2
    with pytest.raises(ValueError, match='scale'):
        d.attach_saved_targets([clip], {'clip': bad}, scales)
    changed = copy.deepcopy(clip); changed['state'][0, 0] += .1
    with pytest.raises(ValueError, match='endpoint'):
        d.attach_saved_targets([changed], {'clip': plan}, scales)


def test_summary_sign_excludes_exact_holds_without_changing_likelihood_support():
    table = {'zero_delta': np.array([1., 0., 0.]), 'direction_brier_conditional': np.array([1., .2, .4]),
        'joint_nll': np.array([2., 3., 4.]), '_group': np.array([0, 1, 2])}
    result = d.summarize(table)
    assert result['count'] == 3 and result['direction_count'] == 2
    assert result['direction_brier_conditional'] == pytest.approx(.3)
    assert result['joint_nll'] == 3
    only_hold = d.summarize(table, np.array([True, False, False]))
    assert only_hold['direction_brier_conditional'] is None
    assert only_hold['joint_nll'] == 2


def test_initial_replay_preserves_original_fp32_formula_and_reduction(tmp_path):
    loc = torch.tensor([[23.117, -4.19, 5.13, .5], [-1.2, 2.3, 20.5, -18.]], dtype=torch.float32)
    target = torch.tensor([[-8.313, 1.29, 1.33, -.2], [6.1, -4.7, -10.5, 14.]], dtype=torch.float32)
    log_std = torch.tensor([[-3., -.22, 1.13, -.71], [-1.87, -.8, -.03, -2.5]], dtype=torch.float32)
    exact = initial_nll({'loc': loc, 'log_scale': log_std}, target)
    parts = d.normal_components(loc, log_std, target)
    assert torch.equal(parts['gaussian_nll'].sum(-1), exact)
    recorded = float(exact.mean())
    path = tmp_path/'saved.json'
    path.write_text(json.dumps({'rows': [{'clip_id': 'c', 'event_count': 1,
        'event_nll': 2., 'initial_nll': recorded}]}))
    diagnostic = {'rows': [{'clip_id': 'c', 'events': {'count': 1, 'joint_nll': 2.},
        # The display aggregate is intentionally rounded; replay must use the
        # untouched original GPU-style sum/mean pathway rather than this field.
        'initial': {'gaussian_nll': recorded/4+.1},
        'exact_original_reduction': {'event_nll': 2., 'initial_nll': recorded}}]}
    assert d.verify_saved_nll(diagnostic, path) == {
        'max_event_nll_abs_error': 0., 'max_initial_nll_abs_error': 0.}
    diagnostic['rows'][0]['exact_original_reduction']['initial_nll'] += .01
    with pytest.raises(ValueError, match='differs'):
        d.verify_saved_nll(diagnostic, path)
