"""Sparse brow lifecycle/control contracts, independent of empirical quality."""
import copy
import json

import numpy as np
import pytest
from scipy.special import expit

from scripts import sparse_brow_event_process as m


def model_fixture(random=True, mark_space="logit_delta"):
    peak = [0.12, .10, .7, .6, .55]
    returned = [.03, .02, .08, .05, .06]
    if mark_space == "coefficient_delta":
        peak = [.01, .008, .08, .06, .06]
        returned = [.002, .002, .008, .005, .006]
    loc = np.r_[peak, returned, np.log(8), np.log1p(3), np.log(9)]
    scale = np.diag([.001] * 10 + [.01] * 3) if random else np.zeros((13, 13))
    return {"schema_version": m.SCHEMA, "fps": 25, "mark_space": mark_space,
            "components": [{"weight": 1., "df": 5., "loc": loc.tolist(), "scale": scale.tolist()}],
            "wait_hazard": [.2, .4, .7], "wait_bin_frames": 12,
            "minimum_hold_frames": 4,
            "max_peak_offset": [.3] * 5 if mark_space == "coefficient_delta" else [2.] * 5,
            "max_return_offset": [.1] * 5 if mark_space == "coefficient_delta" else [.5] * 5,
            "duration_bounds": {"onset": [4, 30], "apex": [0, 12], "release": [4, 30]}}


def process(seed=7, *, model=None, style=None, level=None):
    return m.SparseBrowEventProcess(model or model_fixture(),
        np.linspace(-2., -.5, 9) if level is None else level, seed, "fixed_query", style)


def concatenate(outputs):
    return {name: np.concatenate([row[name] for row in outputs]) for name in outputs[0]}


def assert_outputs_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_automatic_hold_is_exact_and_all_four_stages_are_emitted():
    state = process(model=model_fixture(False))
    out = state.sample(500)
    assert set(out["phase"]) == {"HOLD", "ONSET", "APEX", "RELEASE"}
    both_hold = (out["phase"][1:] == "HOLD") & (out["phase"][:-1] == "HOLD")
    assert both_hold.sum() > 50
    np.testing.assert_array_equal(np.diff(out["logits"], axis=0)[both_hold], np.zeros((both_hold.sum(), 9)))
    assert all(row["durations"] == [8, 3, 9] for row in state.events)
    for event in state.events:
        if event["status"] == "complete":
            assert event["end_frame"] - event["start_frame"] == 20
    assert state.state_dict()["query_motion_used"] is False
    assert state.state_dict()["reference_trajectory_used"] is False


def test_same_runtime_chunk_invariance_and_json_resume():
    first, second = process(), process()
    a = first.sample(1000)
    b = concatenate([second.sample(n) for n in [0, 1, 2, 31, 7, 113, 409, 437]])
    assert_outputs_equal(a, b)
    assert first.state_dict() == second.state_dict()
    saved = json.loads(json.dumps(first.state_dict(), allow_nan=False))
    restored = m.SparseBrowEventProcess.from_state_dict(model_fixture(), saved)
    assert first.state_dict() == restored.state_dict()
    assert_outputs_equal(first.sample(333), restored.sample(333))
    assert first.state_dict() == restored.state_dict()


@pytest.mark.parametrize("space", ["logit_delta", "coefficient_delta"])
def test_amplitude_zero_exact_frozen_level_and_unchanged_clock(space):
    model = model_fixture(mark_space=space)
    level = np.linspace(-12., -.5, 9)
    states = [process(model=model, level=level, style={"amplitude": amplitude}) for amplitude in [0., .5, 1., 1.5]]
    outs = [state.sample(900) for state in states]
    np.testing.assert_array_equal(outs[0]["logits"], np.broadcast_to(level, (900, 9)))
    for out in outs[1:]:
        np.testing.assert_array_equal(out["phase"], outs[0]["phase"])
        np.testing.assert_array_equal(out["event_index"], outs[0]["event_index"])
    marks = [[event["raw_mark"] for event in state.events] for state in states]
    assert all(mark == marks[0] for mark in marks[1:])
    if space == "logit_delta":
        for amplitude, out in zip([0., .5, 1., 1.5], outs):
            np.testing.assert_allclose(out["logits"] - level, amplitude * (outs[2]["logits"] - level), atol=2e-14)


def test_coefficient_endpoint_preserves_physical_amplitude_at_low_level():
    model = model_fixture(False, "coefficient_delta")
    level = np.full(9, -7.)
    low = process(model=model, level=level, style={"amplitude": .5})
    high = process(model=model, level=level, style={"amplitude": 1.})
    low.sample(200)
    high.sample(200)
    for a, b in zip(low.events, high.events):
        observed_low = expit(a["peak_logit"]) - expit(a["start_logit"])
        observed_high = expit(b["peak_logit"]) - expit(b["start_logit"])
        np.testing.assert_allclose(observed_high, b["peak_offset"], atol=2e-16)
        np.testing.assert_allclose(observed_high, 2 * observed_low, atol=2e-16)
        assert not any(b["peak_boundary_clipped"])
    assert high.events[0]["peak_offset"][2] == pytest.approx(.08)


def test_coefficient_boundary_clipping_is_reported_and_zero_control_ignores_eps():
    model = model_fixture(False, "coefficient_delta")
    level = np.full(9, 12.)
    state = process(model=model, level=level)
    out = state.sample(150)
    assert any(any(row["peak_boundary_clipped"]) for row in state.events)
    assert np.isfinite(out["values"]).all()
    zero = process(model=model, level=level, style={"amplitude": 0})
    np.testing.assert_array_equal(zero.sample(150)["logits"], np.broadcast_to(level, (150, 9)))
    assert not any(any(row["peak_boundary_clipped"]) for row in zero.events)


def test_unsupported_zero_cap_channels_never_snap_to_coefficient_epsilon():
    model = model_fixture(False, "coefficient_delta")
    model["max_peak_offset"][:2] = [0., 0.]
    model["max_return_offset"][:2] = [0., 0.]
    level = np.full(9, -12.)
    state = process(model=model, level=level)
    out = state.sample(200)
    np.testing.assert_array_equal(out["logits"][:, :2], np.broadcast_to(level[:2], (200, 2)))
    assert all(not any(row["peak_boundary_clipped"][:2]) for row in state.events)


def test_activity_changes_count_without_changing_same_index_joint_marks():
    states = [process(style={"activity": activity}) for activity in [.5, 1., 2.]]
    for state in states:
        state.sample(1500)
    counts = [len(state.events) for state in states]
    assert counts[0] < counts[1] < counts[2]
    common = min(counts)
    assert [[row["raw_mark"] for row in state.events[:common]] for state in states].count(
        [row["raw_mark"] for row in states[0].events[:common]]) == 3
    assert [[row["durations"] for row in state.events[:common]] for state in states].count(
        [row["durations"] for row in states[0].events[:common]]) == 3


def test_zero_activity_is_exact_static_hold_and_can_resume():
    state = process(style={"activity": 0})
    out = state.sample(100)
    assert len(state.events) == 0 and set(out["phase"]) == {"HOLD"}
    np.testing.assert_array_equal(out["logits"], np.broadcast_to(state.level, (100, 9)))
    restored = m.SparseBrowEventProcess.from_state_dict(model_fixture(), state.state_dict())
    state.set_style({"activity": 1.})
    restored.set_style({"activity": 1.})
    assert_outputs_equal(state.sample(300), restored.sample(300))
    assert len(state.events) > 0


def test_reference_swap_only_changes_future_event_and_preserves_current_curve():
    a, b = process(), process()
    while a.phase != "ONSET":
        a.sample(1)
        b.sample(1)
    saved = copy.deepcopy(a.active_event)
    b.set_style({"amplitude": [2., .25], "activity": 2.})
    current_remaining = sum(saved["durations"]) - a.stage_age
    assert_outputs_equal(a.sample(current_remaining), b.sample(current_remaining))
    np.testing.assert_array_equal(a.x, b.x)
    a.sample(300)
    b.sample(300)
    assert a.events[0] == b.events[0]
    assert a.events[1]["raw_mark"] == b.events[1]["raw_mark"]
    assert a.events[1]["amplitude"] != b.events[1]["amplitude"]


def test_hold_and_release_are_c2_from_actual_state_with_exact_terminal_values():
    state = process(model=model_fixture(False))
    while state.phase != "ONSET" or state.stage_age < 3:
        state.sample(1)
    initial = [state.x.copy(), state.v.copy(), state.a.copy()]
    stop_endpoint = state.x + state.v * (8 / 25) / 2
    state.command("HOLD")
    at_start = m.evaluate_quintic(state.coefficients, 0., 8 / 25)
    for expected, actual in zip(initial, at_start):
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    at_end = m.evaluate_quintic(state.coefficients, 1., 8 / 25)
    np.testing.assert_allclose(at_end[0], stop_endpoint, atol=1e-13)
    np.testing.assert_allclose(at_end[1], 0, atol=2e-12)
    np.testing.assert_allclose(at_end[2], 0, atol=2e-11)
    out = state.sample(50)
    np.testing.assert_array_equal(state.x, stop_endpoint)
    np.testing.assert_array_equal(out["logits"][7:], np.broadcast_to(out["logits"][7], (43, 9)))
    assert state.events[-1]["status"] == "aborted"
    state.command("RELEASE")
    out = state.sample(100)
    np.testing.assert_array_equal(out["logits"][15:], np.broadcast_to(state.level, (85, 9)))
    np.testing.assert_array_equal(state.v, np.zeros(5))
    np.testing.assert_array_equal(state.a, np.zeros(5))


def test_mid_transition_resume_and_command_chunk_invariance():
    a, b = process(), process()
    a.sample(31)
    b.sample(31)
    for mode, total in [("HOLD", 3), ("RUN", 37), ("RELEASE", 4), ("RUN", 111), ("HOLD", 20)]:
        a.command(mode)
        b.command(mode)
        expected = a.sample(total)
        actual = concatenate([b.sample(1), b.sample(total-1)])
        assert_outputs_equal(expected, actual)
        assert a.state_dict() == b.state_dict()


def test_long_run_has_no_cumulative_return_drift_and_other_channels_are_fixed():
    state = process(model=model_fixture(False))
    out = state.sample(6000)
    assert len(state.events) > 50
    expected_return = state.level[:5] + np.asarray(state.events[0]["return_level_offset"])
    for event in state.events:
        np.testing.assert_array_equal(event["return_logit"], expected_return)
    np.testing.assert_array_equal(out["logits"][:, 5:], np.broadcast_to(state.level[5:], (6000, 4)))
    np.testing.assert_array_equal(out["values"][:, 5:], np.broadcast_to(expit(state.level[5:]), (6000, 4)))
    assert np.max(np.abs(out["logits"][:, :5] - state.level[:5])) < 1.


def test_survival_table_terminal_tail_not_truncated():
    model = m.validate_model(model_fixture(False))
    model["wait_hazard"] = [0., .5]
    model["wait_bin_frames"] = 10
    model["minimum_hold_frames"] = 4
    assert m.sample_wait_frames(model, 0.) == 15
    assert m.sample_wait_frames(model, .5) == 25
    assert m.sample_wait_frames(model, 1.-2.**-40) > 300
    model["wait_hazard"] = [1.]
    assert m.sample_wait_frames(model, .99) == 5


def test_student_t_joint_covariance_and_shared_component_selection():
    model = model_fixture(False)
    factor = np.zeros((13, 13))
    factor[0, 0] = .12
    factor[2, 0] = .09
    factor[5, 1] = .04
    model["components"][0]["scale"] = (factor @ factor.T).tolist()
    state = process(model=model)
    # Exercise joint mark generation directly to avoid 100k rendering ticks.
    marks = []
    for _ in range(2500):
        state._start_event()
        marks.append(state.active_event["raw_mark"])
        state._abort()
    centered = np.asarray(marks) - np.asarray(model["components"][0]["loc"])
    np.testing.assert_allclose(centered[:, 2], .75 * centered[:, 0], atol=3e-15)
    expected_variance = .12**2 * 5/3
    assert np.var(centered[:, 0]) == pytest.approx(expected_variance, rel=.17)


def test_mixture_chooses_one_joint_component_not_independent_channels():
    model = model_fixture(False)
    first = model["components"][0]
    first["weight"] = .3
    second = copy.deepcopy(first)
    second["weight"] = .7
    second["loc"][:10] = (-np.asarray(first["loc"][:10])).tolist()
    model["components"].append(second)
    state = process(model=model)
    for _ in range(1000):
        state._start_event()
        event = state.active_event
        assert event["raw_mark"] == model["components"][event["component"]]["loc"]
        state._abort()
    assert np.mean([row["component"] == 0 for row in state.events]) == pytest.approx(.3, abs=.04)


def test_zero_apex_and_mark_clipping_have_explicit_provenance():
    model = model_fixture(False)
    model["components"][0]["loc"] = [100.] * 5 + [-100.] * 5 + [-100., 0., 100.]
    state = process(model=model)
    out = state.sample(500)
    assert "APEX" not in out["phase"]
    for event in state.events:
        assert event["durations"] == [4, 0, 30]
        assert all(event["peak_clipped"]) and all(event["return_clipped"])
        assert event["duration_clipped"] == [True, False, True]
        if event["status"] == "complete":
            assert event["end_frame"] - event["start_frame"] == 34
    assert np.isfinite(out["values"]).all()


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x.update(wait_hazard=[.2, 0.]), "hazard"),
    (lambda x: x.update(components=[]), "components"),
    (lambda x: x["components"][0].update(df=4), "df=5"),
    (lambda x: x["components"][0]["scale"][0].__setitem__(0, -1.), "semidefinite"),
    (lambda x: x.update(mark_space="arbitrary"), "mark_space"),
    (lambda x: x.update(duration_bounds={"onset": [0, 4], "apex": [0, 3], "release": [4, 8]}), "integer"),
])
def test_invalid_models_rejected(mutation, match):
    model = model_fixture()
    mutation(model)
    with pytest.raises(ValueError, match=match):
        process(model=model)


def test_invalid_inputs_rejected_and_model_hash_protects_restore():
    state = process()
    for invalid in [-1, True, .5]:
        with pytest.raises(ValueError, match="integer"):
            state.sample(invalid)
    for invalid in [-1., float("nan"), True]:
        with pytest.raises(ValueError, match="activity"):
            state.set_style({"activity": invalid})
    with pytest.raises(ValueError, match="only"):
        state.set_style({"reference_sequence": np.ones((10, 5))})
    model = model_fixture()
    model["wait_hazard"][0] = .25
    with pytest.raises(ValueError, match="hash"):
        m.SparseBrowEventProcess.from_state_dict(model, state.state_dict())
