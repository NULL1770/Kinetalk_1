import copy

import numpy as np
import pytest

from scripts.motion_process_representation import (
    DEFAULT_DURATIONS, PersistentMotionDecoder, fit_dynamic_scales,
    fit_motion_process, quantize_motion_process, read_group_states,
    reconstruction_metrics, render_motion_process,
    uniform_motion_process,
)


def test_raw_groups_partial_observation_and_no_clipping():
    x = np.full((3, 52), np.nan); observed = np.zeros_like(x, dtype=bool)
    x[0, 43] = -.2; observed[0, 43] = True
    x[0, 44] = 1.4; observed[0, 44] = True
    x[1, 41] = 1.7; observed[1, 41] = True
    states, valid = read_group_states(x, observed)
    assert states[0, 0] == .6 and states[1, 1] == 1.7
    assert not valid[2].any() and (states[~valid] == 0).all()
    with pytest.raises(ValueError):
        read_group_states(x, np.ones_like(observed))


def test_dynamic_scales_pool_only_supplied_runs_and_ignore_gap_offset():
    x = np.array([[0.], [2.], [np.nan], [100.], [102.]])
    valid = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(fit_dynamic_scales([(x, valid)]), [1.])
    flat = np.ones((4, 1))*900
    np.testing.assert_array_equal(fit_dynamic_scales([(flat, np.ones(4, bool))]), [.005])


def test_variable_knots_encode_hold_return_and_independent_overlap():
    t = np.arange(41.)
    a = np.interp(t, [0, 8, 20, 40], [0., .8, .8, 0.])
    b = np.interp(t, [0, 12, 20, 40], [.2, .2, .6, .6])
    target = np.stack([a, b], -1)
    plan = fit_motion_process(target, np.ones(41, bool), np.ones(2), complexity_penalty=.01)
    result = render_motion_process(plan)
    np.testing.assert_allclose(result, target, atol=1e-12)
    assert [s['duration'] for s in plan['segments'][0]] == [8, 12, 20]
    assert any(s['start_value'] == s['end_value'] for s in plan['segments'][0])
    assert plan['segments'][0][-1]['end_value'] < plan['segments'][0][-1]['start_value']
    assert [s['start'] for s in plan['segments'][0]] != [s['start'] for s in plan['segments'][1]]


def test_gap_hard_boundary_short_tail_censoring_and_padding_invariance():
    target = np.arange(21.)[:, None]*.01
    valid = np.ones((21, 1), bool); valid[8:11] = False
    target[~valid] = np.nan
    plan = fit_motion_process(target, valid, [.1])
    output = render_motion_process(plan)
    assert (output[~valid] == 0).all()
    assert plan['segments'][0][0]['duration'] == 7
    assert all(s['right_censored'] for s in plan['segments'][0])
    padded = fit_motion_process(np.vstack((target, [[np.nan]]*9)),
                                np.vstack((valid, [[False]]*9)), [.1])
    assert plan['segments'] == padded['segments']
    np.testing.assert_array_equal(render_motion_process(padded)[:21], output)
    malformed = copy.deepcopy(plan); malformed['segments'][0][0]['duration'] = 10
    with pytest.raises(ValueError, match='gap'):
        render_motion_process(malformed)


def test_all_invalid_and_singleton_are_not_events():
    target = np.zeros((8, 2)); valid = np.zeros_like(target, bool)
    valid[3, 0] = True; target[3, 0] = .7
    plan = fit_motion_process(target, valid, [.1, .1])
    assert plan['segments'][0][0]['duration'] == 0
    assert plan['segments'][0][0]['right_censored']
    assert plan['segments'][1] == []
    np.testing.assert_array_equal(render_motion_process(plan), np.where(valid, target, 0.))


def test_every_interior_duration_is_in_grid_terminal_is_censored():
    target = np.sin(np.arange(117.)*.3)[:, None]
    plan = fit_motion_process(target, np.ones(117, bool), [.5])
    assert all(s['duration'] in DEFAULT_DURATIONS for s in plan['segments'][0][:-1])
    assert not any(s['right_censored'] for s in plan['segments'][0][:-1])
    assert plan['segments'][0][-1]['right_censored']
    assert np.isfinite(render_motion_process(plan)).all()


def test_dp_reconstruction_error_agrees_with_metrics_and_rate_is_explicit():
    target = np.sin(np.arange(83.)*.05)[:, None]
    valid = np.ones(83, bool)
    plan = fit_motion_process(target, valid, [.4], complexity_penalty=.02)
    reconstructed = render_motion_process(plan)
    row = reconstruction_metrics(target, valid, reconstructed, [.4], plan)['groups'][0]
    assert row['raw_mse'] == pytest.approx(np.mean((target-reconstructed)**2))
    assert row['normalized_mse'] == pytest.approx(row['raw_mse']/.16)
    assert row['centered_corr'] > .99
    assert row['segments_per_100_frames'] == 100*len(plan['segments'][0])/83


def test_quantized_delta_accumulates_error_instead_of_resetting_to_gt_endpoint():
    target = np.interp(np.arange(17.), [0, 4, 8, 12, 16], [0., .3, 0., .3, 0.])[:, None]
    plan = fit_motion_process(target, np.ones(17, bool), [1.], durations=(4,), complexity_penalty=0)
    quantized = quantize_motion_process(plan, [-.25, 0., .5])
    assert quantized['segments'][0][1]['start_value'] == .5
    assert quantized['segments'][0][-1]['end_value'] == .5
    assert render_motion_process(quantized)[-1, 0] == .5
    assert plan['segments'][0][-1]['end_value'] == 0.


def test_uniform_control_keeps_counts_and_gaps_but_changes_interior_knots():
    target = np.interp(np.arange(41.), [0, 8, 20, 40], [0., .8, .8, 0.])[:, None]
    plan = fit_motion_process(target, np.ones(41, bool), [1.], complexity_penalty=.01)
    uniform = uniform_motion_process(plan, target)
    assert len(uniform['segments'][0]) == len(plan['segments'][0]) == 3
    assert [s['start'] for s in uniform['segments'][0]] == [0, 13, 27]
    assert np.isfinite(render_motion_process(uniform)).all()
    assert [s['start'] for s in plan['segments'][0]] == [0, 8, 20]


def test_persistent_state_chunk_invariance_hold_and_resume():
    whole = PersistentMotionDecoder([0., .5])
    whole.start_segment(0, 20, 1.)
    whole.start_segment(1, 8, -.3)
    expected = whole.advance(32)
    chunked = PersistentMotionDecoder([0., .5])
    chunked.start_segment(0, 20, 1.); chunked.start_segment(1, 8, -.3)
    first = chunked.advance(7)
    restored = PersistentMotionDecoder.from_state_dict(chunked.state_dict())
    actual = np.vstack((first, restored.advance(9), restored.advance(16)))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[20:], np.tile([1., -.3], (12, 1)))
    restored.start_segment(0, 12, 0.)
    assert restored.advance(12)[-1, 0] == 0.
    with pytest.raises(ValueError, match='unfinished'):
        active = PersistentMotionDecoder([0.]); active.start_segment(0, 4, 1.); active.start_segment(0, 4, 2.)


def test_invalid_duration_scale_and_checkpoint_fail_explicitly():
    with pytest.raises(ValueError):
        fit_motion_process(np.zeros((4, 1)), np.ones(4, bool), [0.])
    with pytest.raises(ValueError):
        fit_motion_process(np.zeros((4, 1)), np.ones(4, bool), [1.], durations=(4, 4))
    decoder = PersistentMotionDecoder([0.]); state = decoder.state_dict(); state['age'][0] = 3
    with pytest.raises(ValueError):
        PersistentMotionDecoder.from_state_dict(state)
