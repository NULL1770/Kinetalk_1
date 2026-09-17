import copy

import numpy as np
import pytest

from scripts.audit_centered_equal_budget_reports import (
    ARMS, EPOCHS, GROUPS, KINDS, MODES, POPULATIONS, SEEDS,
    audit_reports, compare_rows, mean_noise_rows, sentence_rows, validate_rng_prefix,
)


def metric(errors):
    target, count = [1., 9.], [10., 30.]
    rows = {f"s{i}": {"sse": error, "zero": target[i], "sst": target[i] * .5,
        "weighted_values": count[i], "clips": 1, "r2_against_zero": 1 - error / target[i]}
        for i, error in enumerate(errors)}
    return {"per_sentence": rows, "sse": sum(errors), "target_energy": sum(target),
        "sst": sum(target) * .5, "weighted_values": sum(count), "clips": 2, "sentences": 2,
        "native_mse": sum(errors) / sum(count), "r2_against_zero": 1 - sum(errors) / sum(target)}


def test_noise_mean_is_sufficient_sse_and_targets_are_strict():
    reports = {seed: metric(errors) for seed, errors in zip(SEEDS, ([1., 2.], [2., 5.], [3., 8.]))}
    names, rows = mean_noise_rows(reports)
    assert names == ["s0", "s1"]
    np.testing.assert_array_equal(rows[:, 0], [2., 5.])
    np.testing.assert_array_equal(rows[:, 1], [1., 9.])
    changed = copy.deepcopy(reports)
    changed["42"]["per_sentence"]["s0"]["zero"] += 1e-6
    with pytest.raises(ValueError, match="aggregate"):
        mean_noise_rows(changed)
    bad = metric([1., 2.]); bad["sse"] = 100
    with pytest.raises(ValueError, match="aggregate"):
        sentence_rows(bad)


def test_pair_uses_ratio_of_summed_sse_and_paired_sentence_bootstrap():
    a = sentence_rows(metric([.5, 4.5]))
    b = sentence_rows(metric([1., 9.]))
    result = compare_rows(a, b, samples=500)
    assert result["r2_delta_candidate_minus_baseline"] == .5
    assert result["native_mse_delta_candidate_minus_baseline"] == -.125
    np.testing.assert_allclose(result["r2_delta_ci95"], [.5, .5])
    bad = (a[0], a[1].copy()); bad[1][0, 3] += 1
    with pytest.raises(ValueError, match="different"):
        compare_rows(bad, b, samples=100)


def records():
    return {e: {"epoch": e, "step": e * 145, "samples_seen": 2315, "teacher_fraction": .5,
        "minibatch_sha256": "a" * 64, "noise_time_sha256": "b" * 64,
        "teacher_choice_draw_sha256": "c" * 64} for e in range(1, 11)}


def test_rng_only_complete_epoch_prefix_and_no_future_records_needed():
    rows = records()
    validate_rng_prefix(rows, copy.deepcopy(rows))
    changed = copy.deepcopy(rows); changed[10]["step"] = 1500
    with pytest.raises(ValueError, match="schedule"):
        validate_rng_prefix(rows, changed)
    changed = copy.deepcopy(rows); changed[8]["noise_time_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="epoch8"):
        validate_rng_prefix(rows, changed)


def arm_data(centered):
    from scripts.audit_projection_rollout_probe import ROLLOUT_SCHEMA, ROLLOUT_ARM, ROLLOUT_LOSS
    from scripts.train_projection_centered_rollout_probe import SCHEMA, ARM, LOSS
    recipe = {"schema": ROLLOUT_SCHEMA, "loss": ROLLOUT_LOSS,
        "args": {"arm": ROLLOUT_ARM, "output": "raw"},
        "source_sha256": {"/repo/scripts/shared.py": "same", "/repo/scripts/train_projection_rollout_probe.py": "entry"},
        "input_sha256": {"cache": "same"}, "data_scope": {"validation_clips": 405}}
    if centered:
        recipe.update(schema=SCHEMA, loss=LOSS, mean_anchoring_loss=False,
            loss_centering="Per-clip/per-channel mean of raw prediction-target error over observed frames only; no detach; raw generated output untouched")
        recipe["args"].update(arm=ARM, output="centered")
        recipe["source_sha256"].pop("/repo/scripts/train_projection_rollout_probe.py")
        recipe["source_sha256"]["/repo/scripts/train_projection_centered_rollout_probe.py"] = "new"
    reports = {}
    for epoch in EPOCHS:
        seeds = {}
        for seed in SEEDS:
            modes = {}
            for mode in MODES:
                errors = [.5, 4.5] if centered and mode == "full" else [1., 9.]
                value = {pop: {group: {**{kind: metric(errors) for kind in KINDS}, "velocity_mse_per_second": .1}
                    for group in GROUPS} for pop in POPULATIONS}
                value["frozen_teacher_emotion_accuracy"] = .5
                modes[mode] = value
            seeds[seed] = modes
        reports[epoch] = {"noise_reports": seeds}
    return {"recipe": recipe, "records": records(), "reports": reports, "sha256": {"report": "same"}}


def test_full_audit_preserves_report_only_boundary_and_all_raw_populations():
    centered, raw = arm_data(True), arm_data(False)
    result = audit_reports(centered, raw, samples=50)
    assert result["primary_equal_budget_epoch"] == 8
    assert result["supplemental_actual_checkpoint_epoch"] == 10
    assert result["original_epoch_reports_bound_to_checkpoint_or_curves"] is False
    assert result["claimed_final18epoch_completion"] is False
    assert result["curves_loaded"] is False
    for epoch in ("8", "10"):
        row = result["epochs"][epoch]["centered_minus_raw"]["full"]["nonneutral"]["upper_expression"]["raw_motion"]
        assert row["r2_delta_candidate_minus_baseline"] == .5
        assert set(result["epochs"][epoch]["full_minus_same_arm_zero"]) == set(ARMS)
    changed = arm_data(False)
    changed["reports"][10]["noise_reports"]["42"]["zero"]["neutral"]["mouth"]["raw_motion"] = metric([1.1, 9.])
    with pytest.raises(ValueError, match="Frozen zero"):
        audit_reports(centered, changed, samples=20)
