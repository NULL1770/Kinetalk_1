import copy
import numpy as np
import pytest
from scripts import extract_brow_states_v3 as s
from scripts.audit_brow_states_v3 import validate_arrays


def data(signal, group="raise", valid=None):
    n = len(signal); raw = np.full((n, 5), .3)
    raw[:, s.v2.GROUPS[group]] += np.asarray(signal)[:, None]
    valid = np.ones(n, bool) if valid is None else valid.copy()
    eligible = np.zeros(n, bool)
    for a, b in s.v2.valid_runs(valid):
        if b-a > 4: eligible[a+2:b-2] = True
    return dict(raw5=raw, smooth5=raw.copy(), valid=valid, eligible=eligible)


def pulse(sign=1):
    return sign*np.r_[np.zeros(12), np.linspace(0, .2, 10), np.full(12, .2), np.linspace(.2, 0, 10), np.zeros(12)]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("group", ["raise", "down"])
def test_complete_episode_direction_and_plateau(sign, group):
    row, a = s.build_clip({"source_id": "x"}, data(pulse(sign), group)); validate_arrays(a)
    events = row["groups"][group]["episodes"]; assert len(events) == 1
    e = events[0]
    assert e["direction"] == sign and not e["left_censored"] and not e["right_censored"]
    j = s.GROUP_NAMES.index(group)
    assert {s.HOLD, s.ONSET, s.ACTIVE, s.RELEASE}.issubset(set(a["labels"][:, j]))
    assert np.sum(a["labels"][:, j] == s.ACTIVE) >= 4


def test_static_high_expression_is_not_event_or_hold():
    row, a = s.build_clip({"source_id": "x"}, data(np.full(70, .5)))
    assert not row["groups"]["raise"]["episodes"]
    assert (a["labels"][2:-2] == s.STATIC).all(); assert not a["phase_known"].any(); validate_arrays(a)


def test_boundary_ramp_has_signed_trend_but_ambiguous_phase():
    row, a = s.build_clip({"source_id": "x", "candidates": []}, data(np.linspace(0, .5, 50)))
    tr = row["groups"]["raise"]["transitions"]
    assert len(tr) == 1 and tr[0]["left_censored"] and tr[0]["right_censored"]
    assert tr[0]["duration_type"] == "lower_bound"
    assert (a["labels"][2:-2, 0] == s.TRANSITION).all()
    assert not a["phase_known"][:, 0].any(); assert (a["trend"][2:-2, 0] == 1).all()


def test_spike_not_promoted_to_transition():
    y = np.zeros(60); y[30] = .4
    row, a = s.build_clip({"source_id": "x"}, data(y))
    assert not row["groups"]["raise"]["transitions"]
    assert a["labels"][30, 0] == s.UNSUPPORTED and not a["known_mask"][30, 0]


def test_gap_and_half_open_edges_never_labelled():
    y = pulse(); v = np.ones(len(y), bool); v[26:29] = False
    row, a = s.build_clip({"source_id": "x"}, data(y, valid=v)); validate_arrays(a)
    assert not a["known_mask"][~a["eligible"]].any()
    for group in row["groups"].values():
        for kind in ("transitions", "episodes", "stable_segments"):
            for e in group[kind]: assert v[e["start"]:e["stop"]].all()


def test_order_independence_and_input_immutability():
    arrays = data(pulse()); before = copy.deepcopy(arrays)
    report = {"source_id": "x", "candidates": [{"left_censored": True}, {"right_censored": True}]}
    r1, a1 = s.build_clip(report, arrays); report["candidates"].reverse(); r2, a2 = s.build_clip(report, arrays)
    assert r1 == r2
    for k in a1: np.testing.assert_array_equal(a1[k], a2[k])
    for k in arrays: np.testing.assert_array_equal(arrays[k], before[k])


def test_two_groups_independent_and_asynchronous():
    arrays = data(pulse()); down = np.roll(-pulse(), 9); arrays["smooth5"][:, :2] += down[:, None]
    row, a = s.build_clip({"source_id": "x"}, arrays); validate_arrays(a)
    assert a["labels"].shape == (len(down), 2)
    assert np.any(a["labels"][:, 0] != a["labels"][:, 1])
    assert row["groups"]["down"]["episodes"][0]["direction"] == -1


def test_partial_episode_carries_duration_censoring():
    row, a = s.build_clip({"source_id": "x"}, data(pulse()[14:]))
    e = row["groups"]["raise"]["episodes"][0]
    assert e["left_censored"] and not e["right_censored"]
    assert e["onset_duration_type"] == "lower_bound" and e["left_boundary_kind"] == "clip_start_with_sg_guard"
    assert a["censored_mask"][:, 0].any()


def test_all_invalid_and_bad_eligible():
    arrays = data(np.zeros(30), valid=np.zeros(30, bool)); _, a = s.build_clip({"source_id": "x"}, arrays)
    validate_arrays(a); assert not a["known_mask"].any(); arrays["eligible"][3] = True
    with pytest.raises(ValueError, match="eligible"): s.build_clip({"source_id": "x"}, arrays)


def test_slow_drift_not_chain_of_false_static_windows():
    row, a = s.build_clip({"source_id": "x"}, data(np.linspace(0, .12, 80)))
    assert not row["groups"]["raise"]["stable_segments"]
    assert not (a["labels"][:, 0] == s.STATIC).any()
