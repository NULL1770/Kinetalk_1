import json
from pathlib import Path

import numpy as np
import pytest

from scripts import diagnose_brow_state_nuisance as d


def sample(n=12):
    return {"coeffs": np.zeros((n, 52)), "valid": np.ones(n, bool),
            "times": np.arange(n) / 25., "geometry": np.zeros((n, 4)),
            "head_pose": np.zeros((n, 3))}


def test_signed_same_clock_association_and_constant_null():
    a = sample()
    x = np.array([0, .1, .4, .2, .1, .6, .3, .2, .5, .1, .2, .4])
    a["coeffs"][:, d.GROUPS["raise"]] = x[:, None]
    a["coeffs"][:, 0] = x
    a["geometry"][:, 2] = 1 - x
    report, arrays = d.diagnose_clip(a, "correlated")
    correlations = report["groups"]["raise"]["difference_correlations"]
    assert correlations["blink_left"]["pearson"] == pytest.approx(1.)
    assert correlations["eye_aperture_right"]["pearson"] == pytest.approx(-1.)
    assert correlations["pose_yaw"] == {"pearson": None, "pairs": 11, "reason": "constant_difference"}
    assert np.array_equal(arrays["transition_times"], a["times"][1:])
    assert report["independent_ground_truth"] is False
    assert report["causal_confidence"] is False


def test_invalid_gap_never_contributes_difference_or_blink_dilation():
    a = sample(9)
    a["valid"][4] = False
    a["coeffs"][:4, d.GROUPS["raise"]] = .1
    a["coeffs"][5:, d.GROUPS["raise"]] = .9
    a["coeffs"][3, 0] = .8
    report, values = d.diagnose_clip(a, "gap")
    assert report["valid_runs"] == [[0, 4], [5, 9]]
    assert np.flatnonzero(values["blink_neighborhood"]).tolist() == [1, 2, 3]
    assert not values["raise_pair_valid"][3:5].any()
    assert np.isnan(values["raise_difference"][3:5]).all()
    assert report["groups"]["raise"]["blink_neighborhood_energy"]["all_observed_brow_energy"] == 0


def test_clock_gap_splits_observed_runs_without_retiming():
    a = sample(8)
    a["times"][4:] += .2
    a["coeffs"][3, 0] = .8
    report, values = d.diagnose_clip(a, "clock_gap")
    assert report["clock_gap_transitions"] == 1
    assert report["valid_runs"] == [[0, 4], [4, 8]]
    assert not values["adjacent_valid"][3]
    assert not values["blink_neighborhood"][4:].any()


def test_missing_blink_observation_is_not_negative_or_bridge():
    a = sample(8)
    a["coeffs"][2, 0] = .9
    a["coeffs"][3, 7] = np.nan
    a["coeffs"][:, d.GROUPS["raise"]] = np.arange(8)[:, None] / 10
    report, values = d.diagnose_clip(a, "missing_blink")
    assert np.flatnonzero(values["blink_neighborhood"]).tolist() == [0, 1, 2]
    energy = report["groups"]["raise"]["blink_neighborhood_energy"]
    assert energy["blink_observable_brow_pairs"] == 5
    assert energy["blink_observable_pair_fraction"] == pytest.approx(5 / 7)
    assert energy["near_blink_fraction"] == pytest.approx(2 / 5)


def test_blink_energy_uses_channel_squares_and_both_near_endpoints():
    a = sample(9)
    a["coeffs"][4, 0] = .8
    # Equal-and-opposite channels have zero group-mean motion but nonzero energy.
    a["coeffs"][3:, 43] = .2
    a["coeffs"][3:, 44] = -.2
    a["coeffs"][8:, 43] += .2
    a["coeffs"][8:, 44] -= .2
    report, values = d.diagnose_clip(a, "energy")
    assert np.flatnonzero(values["blink_neighborhood"]).tolist() == [2, 3, 4, 5, 6]
    energy = report["groups"]["raise"]["blink_neighborhood_energy"]
    assert energy["all_observed_brow_energy"] == pytest.approx(.16)
    assert energy["near_blink_brow_energy"] == pytest.approx(.08)
    assert energy["near_blink_fraction"] == pytest.approx(.5)
    assert report["groups"]["raise"]["difference_correlations"]["blink_left"]["pearson"] is None


def test_missing_optional_arrays_report_explicitly_and_leave_inputs_unchanged():
    a = sample(5)
    del a["geometry"], a["head_pose"]
    before = {key: value.copy() for key, value in a.items()}
    report, _ = d.diagnose_clip(a, "missing")
    assert report["availability"]["eye_aperture_left"]["missing_reason"] == "missing_array"
    assert report["availability"]["pose_roll"]["missing_reason"] == "missing_array"
    assert report["groups"]["down"]["difference_correlations"]["pose_roll"]["pearson"] is None
    for key in a:
        np.testing.assert_array_equal(a[key], before[key])


def test_shortest_pose_difference_avoids_wraparound_artifact():
    a = sample(5)
    a["head_pose"][:, 0] = [178, 179, -179, -177, -174]
    _, values = d.diagnose_clip(a, "pose_wrap")
    np.testing.assert_allclose(values["nuisance_differences"][:, 4], [1, 2, 2, 3])


def test_malformed_clock_or_optional_geometry_rejected():
    a = sample()
    a["times"][3] = a["times"][2]
    with pytest.raises(ValueError, match="strictly increasing"):
        d.diagnose_clip(a, "duplicate")
    a = sample()
    a["geometry"] = np.zeros((12, 3))
    with pytest.raises(ValueError, match="geometry must have shape"):
        d.diagnose_clip(a, "bad_geometry")


def locked_fixture(path: Path, count=32):
    path.mkdir()
    (path / "arrays").mkdir()
    rows = [{"clip_id": f"clip_{i:02d}"} for i in range(count)]
    d.write_json(path / "selection.json", {"schema": "tracking_reset_selection_v1", "clips": rows})
    d.write_json(path / "provenance.json", {"schema": "tracking_reset_order_diagnostic_v1",
                                            "selection_sha256": d.sha(path / "selection.json"),
                                            "fps": 25, "arms": ["fresh_forward"]})
    for row in rows:
        a = sample(6)
        a["coeffs"][:, 43:46] = np.array([0, .1, .3, .2, .1, 0])[:, None]
        np.savez_compressed(path / "arrays" / (row["clip_id"] + "_fresh_forward.npz"), **a)
    d.write_json(path / "manifest.json", {p.relative_to(path).as_posix(): {"sha256": d.sha(p), "bytes": p.stat().st_size}
                                         for p in path.rglob("*") if p.is_file()})


def test_locked32_run_writes_strict_json_and_plot_arrays(tmp_path):
    audit, output = tmp_path / "audit", tmp_path / "out"
    locked_fixture(audit)
    report = d.run(audit, output)
    assert report["clip_count"] == 32
    assert report["pooled_groups"]["raise"]["observed_pairs"] == 32 * 5
    assert report["pooled_groups"]["raise"]["blink_neighborhood_energy"]["near_blink_fraction"] == 0
    npz = np.load(output / "arrays" / "clip_00.npz", allow_pickle=False)
    assert npz["nuisance_differences"].shape == (5, 7)
    assert len(json.loads((output / "manifest.json").read_text(encoding="utf8"))) == 66
    with pytest.raises(ValueError, match="fresh directory"):
        d.run(audit, output)


def test_manifest_tamper_and_nonlocked_count_rejected(tmp_path):
    audit = tmp_path / "audit"
    locked_fixture(audit)
    with (audit / "arrays" / "clip_00_fresh_forward.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        d.run(audit, tmp_path / "out")
    smaller = tmp_path / "smaller"
    locked_fixture(smaller, count=31)
    with pytest.raises(ValueError, match="exactly32"):
        d.run(smaller, tmp_path / "out2")
