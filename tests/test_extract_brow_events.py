"""Synthetic contracts for fixed event teacher; no original datasets touched."""
import numpy as np
import pytest

from scripts import extract_brow_events as teacher


THRESHOLDS = {"raise": .035, "down": .035}


def pulse(sign=1., channel_group="raise", apex_frames=0):
    q = np.linspace(0, 1, 17)
    q = q**3*(10-15*q+6*q*q)
    signal = np.r_[np.zeros(12), q, np.ones(apex_frames), q[-2::-1], np.zeros(12)]
    raw = np.full((len(signal), 5), .3)
    raw[:, teacher.GROUPS[channel_group]] += sign*.2*signal[:, None]
    return raw, np.ones(len(raw), bool)


def test_static_and_single_frame_spike_are_not_events():
    raw = np.full((80, 5), .2)
    result, _ = teacher.extract_clip(raw, np.ones(80, bool), THRESHOLDS, "static")
    assert result["events"] == []
    raw[40, 2:] += .5
    result, _ = teacher.extract_clip(raw, np.ones(80, bool), THRESHOLDS, "spike")
    assert result["events"] == []
    assert result["summary"]["unsupported_candidate_count"] > 0


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("group", ["raise", "down"])
def test_complete_signed_pulse(direction, group):
    raw, valid = pulse(direction, group)
    result, arrays = teacher.extract_clip(raw, valid, THRESHOLDS, "synthetic")
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event["groups"][0]["direction"] == direction
    assert event["groups"][0]["group"] == group
    assert event["d_onset"] >= 4 and event["d_release"] >= 4
    assert event["start"] >= 2 and event["end"] < len(raw)-2
    np.testing.assert_array_equal(event["baseline5"], arrays["smooth5"][event["start"]])
    np.testing.assert_allclose(np.array(event["p5"])+event["baseline5"], event["peak5"])
    np.testing.assert_allclose(np.array(event["r5"])+event["baseline5"], event["terminal5"])
    assert not event["verified_semantic_ground_truth"]


def test_gap_does_not_join_motion_or_smoothing():
    raw, valid = pulse()
    middle = len(raw)//2
    valid[middle-2:middle+3] = False
    result, arrays = teacher.extract_clip(raw, valid, THRESHOLDS, "gapped")
    assert not result["events"]
    assert result["summary"]["censored_candidate_count"] > 0
    for c in result["candidates"]:
        assert valid[c["start"]:c["end"]+1].all()
    for left, right in teacher.valid_runs(valid):
        expected = teacher.savgol_filter(raw[left:right], 5, 2, axis=0)
        np.testing.assert_allclose(arrays["smooth5"][left:right], expected)


def test_boundary_motion_preserved_as_censored():
    raw = np.full((50, 5), .2)
    raw[:, 2:] += np.linspace(0, .3, 50)[:, None]
    result, _ = teacher.extract_clip(raw, np.ones(50, bool), THRESHOLDS, "ramp")
    assert not result["events"]
    assert result["summary"]["censored_candidate_count"] >= 1
    assert result["summary"]["unsupported_energy_fraction"] > .8
    assert all(not w["supported_hold"] for w in result["waits"])


def test_fit_thresholds_use_noise_floor_and_no_mutation():
    raw, valid = pulse()
    saved = raw.copy()
    thresholds = teacher.fit_noise_thresholds([(raw, valid)])
    assert all(v["threshold"] >= .035 for v in thresholds.values())
    assert all(v["fit_clip_count"] == 1 for v in thresholds.values())
    assert all(not v["noise_is_independent_measurement"] for v in thresholds.values())
    np.testing.assert_array_equal(raw, saved)


def test_joint_synchronous_groups_merge():
    raw, valid = pulse()
    raw[:, :2] = .3-(raw[:, 2:3]-.3)
    result, _ = teacher.extract_clip(raw, valid, THRESHOLDS, "joint")
    assert len(result["events"]) == 1
    assert len(result["events"][0]["groups"]) == 2


def test_empty_validity_and_nonfinite_validation():
    raw = np.zeros((20, 5))
    result, _ = teacher.extract_clip(raw, np.zeros(20, bool), THRESHOLDS, "invalid")
    assert result["summary"]["valid_frames"] == 0
    raw[8, 3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        teacher.extract_clip(raw, np.ones(20, bool), THRESHOLDS, "bad")


def test_arkit52_mapping_does_not_treat_eye_channels_as_brows():
    raw, valid = pulse()
    full = np.zeros((len(raw), 52))
    full[:, :5] = raw
    result, arrays = teacher.extract_clip(full, valid, THRESHOLDS, "eye_only")
    assert result["events"] == []
    assert np.max(arrays["raw5"]) == 0
    full[:, teacher.BROW52] = raw
    result, arrays = teacher.extract_clip(full, valid, THRESHOLDS, "brow52")
    assert len(result["events"]) == 1
    np.testing.assert_array_equal(arrays["raw5"], raw)
    assert teacher.fit_noise_thresholds([(full, valid)]) == teacher.fit_noise_thresholds([(raw, valid)])
    with pytest.raises(ValueError, match="exactly5"):
        teacher.extract_clip(full[:, :9], valid, THRESHOLDS, "ambiguous")


def test_double_peak_without_hold_is_rejected_and_reported():
    raw, valid = pulse()
    first = raw[:, 2].copy()-.3
    shifted = np.roll(first, 12)
    raw[:, 2:] = .3 + (first + shifted)[:, None]
    result, _ = teacher.extract_clip(raw, valid, THRESHOLDS, "composite")
    # Result cannot silently accept two back-to-back unsupported excursions.
    assert len(result["events"]) <= 1
    assert result["summary"]["candidate_count"] >= result["summary"]["complete_event_count"]
