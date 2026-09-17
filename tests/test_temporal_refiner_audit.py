import copy

import numpy as np
import pytest
import torch

from kinetalk_b0.predictable_motion import weighted_clip_center
from scripts.audit_predictable_motion_predictions import clip_statistics
from scripts.audit_temporal_audio_refiner_probe import make_report, relative_error_protection


def sufficient_rows(errors):
    # Valid nondegenerate correlation and identical target/count columns; only
    # SSE varies. Used to isolate the relative-MSE protection's sign/threshold.
    return np.asarray([[error, 10., 5., 20., 2., 5., 10.] for error in errors])


def test_relative_error_protection_sign_threshold_and_sentence_cluster_ci():
    baseline = sufficient_rows([1., 2., 3., 4.])
    sentences, ids = ["s0", "s0", "s1", "s2"], list(range(4))
    for factor, expected_pass in ((.9, True), (1.005, True), (1.02, False)):
        candidate = baseline.copy()
        candidate[:, 0] *= factor
        result = relative_error_protection(candidate, baseline, sentences, ids, samples=100)
        assert result["relative_native_mse_increase"] == pytest.approx(factor - 1)
        assert result["relative_native_mse_increase_upper90"] == pytest.approx(factor - 1)
        assert result["mse_protection_pass"] is expected_pass
        assert result["mouth_correlation_protection_pass"]
    candidate = baseline.copy()
    candidate[:, 0] = [2., 1., 2., 7.]
    result = relative_error_protection(candidate, baseline, sentences, ids, samples=300)
    clustered = np.array([[3., 3.], [2., 3.], [7., 4.]])
    indices = np.random.default_rng(45).integers(3, size=(300, 3))
    sums = clustered[indices].sum(1)
    expected_upper = np.quantile(sums[:, 0] / sums[:, 1] - 1, .9)
    assert result["relative_native_mse_increase_upper90"] == pytest.approx(expected_upper)
    # A correlation decline over .005 fails independently of improving MSE.
    candidate = baseline.copy()
    candidate[:, 0] *= .9
    candidate[:, 4] *= .5
    result = relative_error_protection(candidate, baseline, sentences, ids, samples=100)
    assert result["mse_protection_pass"] and not result["mouth_correlation_protection_pass"]
    candidate[0, 1] += 1
    with pytest.raises(ValueError, match="Unpaired"):
        relative_error_protection(candidate, baseline, sentences, ids, samples=100)


def curves_fixture():
    generator = torch.Generator().manual_seed(19)
    n, bins, channels = 40, 6, [5, 14, 41]
    weight = torch.ones(n, bins, dtype=torch.float64)
    target = weighted_clip_center(torch.randn(n, bins, 3, generator=generator, dtype=torch.float64), weight)
    noise = weighted_clip_center(torch.randn(n, bins, 3, generator=generator, dtype=torch.float64), weight)
    ridge, pointwise, temporal = (.35 * target + .2 * noise,
                                 .5 * target + .1 * noise,
                                 .75 * target + .1 * noise)
    predictions = {"ridge": {"full": ridge, "reverse": ridge.flip(1),
                              "zero": torch.zeros_like(target), "metric_projection_oracle": target},
                   "pointwise": {"full": pointwise, "reverse": pointwise.flip(1)},
                   "temporal": {"full": temporal, "reverse": temporal.flip(1)}}
    groups = {"all_expression": channels, "eyes_expression": [5], "mouth": [14], "brows": [41]}
    statistics = {arm: {mode: {name: clip_statistics(pred, target, weight,
                    [channels.index(c) for c in ids]) for name, ids in groups.items()}
                  for mode, pred in modes.items()} for arm, modes in predictions.items()}
    return {"provenance": {"motion_channel_indices": channels}, "target": target, "weight": weight,
            "groups": groups, "predictions": predictions, "statistics": statistics,
            "emotion_id": torch.tensor([0, 1] * 20),
            "sentence_id": [f"sentence{i // 2}" for i in range(n)],
            "speaker_id": [f"speaker{i // 4}" for i in range(n)]}


def test_make_report_recomputes_paired_stats_and_uses_shared_true_zero_baseline():
    curves = curves_fixture()
    report = make_report(curves, samples=100)
    assert report["positive_brow_people"] == 10
    assert report["generation_entry_pass"]
    assert report["temporal_brow_advantage_vs_pointwise_ci_positive"]
    for arm in ("ridge", "pointwise", "temporal"):
        row = report["comparisons"][arm + "__vs__zero"]
        assert row["baseline"] == ("ridge", "zero")
        summary = row["populations"]["nonneutral"]["brows"]
        assert summary["baseline"]["r2_against_zero"] == 0
        assert summary["baseline"]["prediction_rms"] == 0
        assert summary["baseline"]["pooled_centered_correlation"] is None
        assert summary["r2_improvement"] == pytest.approx(summary["candidate"]["r2_against_zero"])
        assert summary["clips"] == 20 and summary["sentences"] == 20
    # Cached stats cannot silently refer to a different target or prediction.
    bad = copy.deepcopy(curves)
    bad["statistics"]["temporal"]["full"]["brows"][0, 1] += .01
    with pytest.raises(AssertionError):
        make_report(bad, samples=100)
    bad = copy.deepcopy(curves)
    bad["predictions"]["temporal"]["full"][0, 0, 2] += .1
    with pytest.raises(AssertionError):
        make_report(bad, samples=100)


def test_make_report_prespecified_protection_blocks_mouth_regression():
    curves = curves_fixture()
    # Improve brows/eyes but intentionally invert mouth correlation. Recompute
    # matching sufficient statistics so only protection logic rejects this arm.
    curves["predictions"]["temporal"]["full"][..., 1] *= -1
    pred = curves["predictions"]["temporal"]["full"]
    for name, channels in curves["groups"].items():
        local = [curves["provenance"]["motion_channel_indices"].index(c) for c in channels]
        curves["statistics"]["temporal"]["full"][name] = clip_statistics(
            pred, curves["target"], curves["weight"], local)
    report = make_report(curves, samples=100)
    assert report["generation_entry_checks"]["brow_improvement_ci_positive"]
    assert report["generation_entry_checks"]["brow_at_least_ten_people_improve"]
    assert not report["generation_entry_checks"]["eyes_mouth_mse_protection"]
    assert not report["generation_entry_checks"]["mouth_correlation_protection"]
    assert not report["generation_entry_pass"]
