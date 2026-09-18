"""Independent behavioral and provenance checks for observed brow-state v3."""
import copy
import numpy as np
import pytest

from scripts import audit_brow_states_v3 as audit
from scripts import extract_brow_states_v3 as s


def make_arrays(signal, valid=None, group="raise"):
    signal = np.asarray(signal, float)
    n = len(signal)
    value = np.full((n, 5), .3)
    value[:, s.v2.GROUPS[group]] += signal[:, None]
    valid = np.ones(n, bool) if valid is None else np.asarray(valid).copy()
    eligible = np.zeros(n, bool)
    for left, right in s.v2.valid_runs(valid):
        if right - left > 4:
            eligible[left + 2:right - 2] = True
    return {"raw5": value.copy(), "smooth5": value, "valid": valid, "eligible": eligible}


def plateau_pulse():
    return np.r_[np.zeros(12), np.linspace(0, .2, 10), np.full(12, .2),
                 np.linspace(.2, 0, 10), np.zeros(12)]


def zero_apex_pulse():
    return np.r_[np.zeros(12), np.linspace(0, .2, 10), np.linspace(.2, 0, 10)[1:], np.zeros(12)]


@pytest.mark.parametrize("signal", [plateau_pulse(), zero_apex_pulse()])
@pytest.mark.parametrize("sign", [-1, 1])
def test_plateau_and_zero_apex_have_same_directional_phase_contract(signal, sign):
    row, arrays = s.build_clip({"source_id": "isolated"}, make_arrays(sign * signal))
    audit.validate_arrays(arrays)
    episodes = row["groups"]["raise"]["episodes"]
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["direction"] == sign
    assert not episode["left_censored"] and not episode["right_censored"] and not episode["conflict"]
    labels = arrays["labels"][:, 0]
    assert np.all(labels[episode["start"]:episode["peak"]] == s.ONSET)
    assert np.all(labels[episode["peak"]:episode["release"] + 1] == s.ACTIVE)
    assert np.all(labels[episode["release"] + 1:episode["stop"]] == s.RELEASE)
    if len(signal) == len(zero_apex_pulse()):
        assert episode["peak"] == episode["release"]
        assert np.count_nonzero(labels == s.ACTIVE) == 1
    else:
        assert episode["release"] - episode["peak"] >= 4


@pytest.mark.parametrize("left,right", [(True, False), (False, True), (True, True)])
def test_censoring_identifies_observed_boundary_and_lower_bound_durations(left, right):
    signal = plateau_pulse()
    signal = signal[14 if left else 0:len(signal) - (14 if right else 0)]
    row, arrays = s.build_clip({"source_id": "crop"}, make_arrays(signal))
    audit.validate_arrays(arrays)
    episode = row["groups"]["raise"]["episodes"][0]
    assert episode["left_censored"] is left and episode["right_censored"] is right
    assert episode["onset_duration_type"] == ("lower_bound" if left else "complete")
    assert episode["release_duration_type"] == ("lower_bound" if right else "complete")
    if left:
        assert episode["start"] == 2
        assert episode["left_boundary_kind"] == "clip_start_with_sg_guard"
    if right:
        assert episode["stop"] == len(signal) - 2
        assert episode["right_boundary_kind"] == "clip_end_with_sg_guard"
    assert arrays["censored_mask"][episode["start"]:episode["stop"], 0].all()
    assert not arrays["known_mask"][~arrays["eligible"]].any()


def test_boundary_censoring_at_missing_gap_is_not_recorded_as_clip_censoring():
    fragment = plateau_pulse()[14:]
    signal = np.r_[np.zeros(6), fragment]
    valid = np.ones(len(signal), bool)
    valid[5] = False
    row, arrays = s.build_clip({"source_id": "gap"}, make_arrays(signal, valid))
    audit.validate_arrays(arrays)
    episode = row["groups"]["raise"]["episodes"][0]
    assert episode["left_censored"]
    assert episode["left_boundary_kind"] == "invalid_gap_left_with_sg_guard"
    assert episode["start"] == 8
    assert not arrays["phase_known"][4:8].any()


def test_composite_alternative_phase_claims_are_excluded_instead_of_order_selected():
    pulse = plateau_pulse()
    row, arrays = s.build_clip({"source_id": "two_pulses"}, make_arrays(np.r_[pulse, pulse[12:]]))
    audit.validate_arrays(arrays)
    conflict = arrays["conflict"][:, 0]
    assert conflict.any()
    assert not arrays["phase_known"][conflict, 0].any()
    assert not arrays["known_mask"][conflict, 0].any()
    assert not arrays["trend_known"][conflict, 0].any()
    assert np.isnan(arrays["reconstruction"][conflict, 0]).all()
    for episode in row["groups"]["raise"]["episodes"]:
        assert episode["conflict"] == bool(conflict[episode["start"]:episode["stop"]].any())
    # The unaffected constant group must not inherit another group's conflicts.
    assert not arrays["conflict"][:, 1].any()


def test_noisy_internal_support_failure_is_not_boundary_censoring():
    # A gap with excessive backtracking fails both directed flanks despite a peak.
    signal = np.r_[np.zeros(12), [.04, .01, .1, .04, .17, .08, .2, .1, .17, .04, .1, .01, .04], np.zeros(12)]
    row, arrays = s.build_clip({"source_id": "unstable"}, make_arrays(signal))
    assert not row["groups"]["raise"]["episodes"]
    assert not arrays["censored_mask"][:, 0].any()
    assert not arrays["phase_known"][:, 0].any()
    assert np.any(arrays["labels"][:, 0] == s.UNSUPPORTED)


@pytest.mark.parametrize("n", [0, 1, 4, 5, 6, 7])
def test_short_runs_respect_exact_guard_and_do_not_invent_episodes(n):
    original = make_arrays(np.zeros(n))
    before = copy.deepcopy(original)
    row, arrays = s.build_clip({"source_id": "short"}, original)
    audit.validate_arrays(arrays)
    assert arrays["eligible"].sum() == max(0, n - 4)
    assert not arrays["phase_known"].any()
    assert not row["groups"]["raise"]["episodes"]
    for key in original:
        np.testing.assert_array_equal(before[key], original[key])


def test_audit_rejects_directionless_identified_dynamic_phase():
    _, arrays = s.build_clip({"source_id": "x"}, make_arrays(plateau_pulse()))
    dynamic = np.isin(arrays["labels"], [s.ONSET, s.ACTIVE, s.RELEASE]) & arrays["phase_known"]
    arrays["direction"][dynamic] = 0
    with pytest.raises(ValueError):
        audit.validate_arrays(arrays)


def test_plateau_apex_uses_same_stationary_trend_convention_as_zero_apex():
    for signal in (plateau_pulse(), zero_apex_pulse()):
        _, arrays = s.build_clip({"source_id": "apex"}, make_arrays(signal))
        stationary = np.isin(arrays["labels"], [s.HOLD, s.STATIC, s.ACTIVE]) & arrays["known_mask"]
        assert np.all(arrays["trend"][stationary] == 0)


@pytest.mark.parametrize("corruption", ["shape", "nonfinite"])
def test_audit_rejects_invalid_source_curve_contract(corruption):
    _, arrays = s.build_clip({"source_id": "x"}, make_arrays(plateau_pulse()))
    if corruption == "shape":
        arrays["smooth5"] = arrays["smooth5"][:, :4]
    else:
        arrays["smooth5"][8, 2] = np.nan
    with pytest.raises(ValueError):
        audit.validate_arrays(arrays)


def test_audit_requires_every_consumed_array_to_be_in_manifest(tmp_path):
    states = tmp_path / "states"
    (states / "arrays").mkdir(parents=True)
    row, arrays = s.build_clip({"source_id": "omitted"}, make_arrays(plateau_pulse()))
    s.v2.write_json(states / "clips.json", [row])
    s.v2.write_json(states / "protocol.json", {"schema": s.SCHEMA})
    np.savez_compressed(states / "arrays" / "omitted.npz", **arrays)
    # A manifest's existence is not sufficient: the actual read array is absent.
    s.v2.write_json(states / "manifest.json", {
        name: {"sha256": s.v2.sha(states / name), "bytes": (states / name).stat().st_size}
        for name in ("clips.json", "protocol.json")
    })
    with pytest.raises(ValueError):
        audit.audit(states, tmp_path / "audit.json")
