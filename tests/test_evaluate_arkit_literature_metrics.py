import numpy as np
import pytest

from scripts.evaluate_arkit_literature_metrics import (
    BEAT_LIP12, BEAT_UPPER16, UPPER9, literature_coefficient_metrics, pending_external_metrics,
    region_coverage, validate_inputs, validate_region,
)


def inputs(length=6):
    target = np.zeros((length, 52))
    return target[None].copy(), target, np.ones_like(target, dtype=bool), np.arange(length) / 25


def test_validation_preserves_missing_values_and_original_gaps():
    x, y, mask, times = inputs()
    mask[2:4] = False
    x[:, 2:4] = np.nan
    y[2:4] = np.nan
    checked = validate_inputs(x, y, mask, times)
    assert np.isnan(checked[0][:, 2:4]).all()
    assert checked[4].all()
    times[3:] += .04
    assert validate_inputs(x, y, mask, times)[4].tolist() == [True, True, False, True, True]


def test_rejects_nonfinite_observed_values_and_nonboolean_mask():
    x, y, mask, times = inputs()
    x[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="Observed"):
        validate_inputs(x, y, mask, times)
    x[0, 1, 0] = 0
    with pytest.raises(ValueError, match="boolean"):
        validate_inputs(x, y, mask.astype(float), times)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_inputs(x, y, mask, times[::-1])
    with pytest.raises(ValueError, match="positive"):
        validate_inputs(x, y, mask, times, fps=0)


def test_regions_are_explicit_and_unique():
    assert validate_region([1, 2], "test") == (1, 2)
    for region in ([1, 1], [], [-1], [52], [1.2]):
        with pytest.raises(ValueError):
            validate_region(region, "test")


def test_missing_dependencies_are_pending_not_perfect_scores():
    result = pending_external_metrics()
    assert set(result) == {"av_offset", "av_confidence", "multimodality", "fd", "wind"}
    for row in result.values():
        assert row["status"] == "pending" and row["value"] is None
        assert row["reason"] and row["required_inputs"]


def test_partial_channel_coverage_is_disclosed():
    x, y, mask, times = inputs()
    mask[:, 0] = False
    mask[2] = False
    support = region_coverage(mask, range(52))
    assert support["valid_frames"] == 5
    assert support["complete_region_frames"] == 0
    assert support["observed_channel_count_min"] == 51
    assert support["partial_support"]


def test_region_norm_not_element_mae_and_no_ensemble_cancellation():
    x, y, mask, times = inputs()
    x[:, :, 0] = 3
    x[:, :, 1] = 4
    x = np.concatenate([x, -x])
    result = literature_coefficient_metrics(x, y, mask, times,
                                            lip_channels=(0, 1), upper_channels=(0, 1))
    assert result["metrics"]["arkit_mbe"]["value"] == 5
    assert result["metrics"]["arkit_lbe"]["value"] == 5
    assert result["metrics"]["arkit_lbe"]["per_sample"] == [5, 5]
    assert result["region_adaptation"]


def test_literal_official_facediffuser_beat_numpy_equivalence():
    rng = np.random.default_rng(614)
    x, y, mask, times = inputs(31)
    y[:] = rng.normal(size=y.shape)
    x = rng.normal(size=(4, *y.shape))
    upper = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 49, 50]
    mouth = [23, 25, 27, 28, 31, 37, 39, 40, 41, 42, 47, 48]
    actual = literature_coefficient_metrics(x, y, mask, times,
                                            lip_channels=mouth, upper_channels=upper)["metrics"]
    mbe, lbe, fdd = [], [], []
    for pred in x:
        mbe.append(np.linalg.norm(pred - y, axis=1).mean())
        lbe.append(np.linalg.norm(pred[:, mouth] - y[:, mouth], axis=1).mean())
        values = np.array([np.square(y[:, v]) for v in upper])
        values = np.transpose(values, (1, 0))
        gt_motion_std = np.mean(np.std(np.sum(values, axis=1), axis=0))
        values = np.array([np.square(pred[:, v]) for v in upper])
        values = np.transpose(values, (1, 0))
        pred_motion_std = np.mean(np.std(np.sum(values, axis=1), axis=0))
        fdd.append(gt_motion_std - pred_motion_std)
    assert actual["arkit_mbe"]["value"] == pytest.approx(np.mean(mbe), abs=1e-14)
    assert actual["arkit_lbe"]["value"] == pytest.approx(np.mean(lbe), abs=1e-14)
    assert actual["arkit_fdd_signed"]["value"] == pytest.approx(np.mean(fdd), abs=1e-14)
    assert actual["arkit_fdd_absolute"]["value"] == pytest.approx(np.mean(np.abs(fdd)), abs=1e-14)


def test_fdd_is_raw_squared_energy_not_velocity_and_is_reversal_invariant():
    x, y, mask, times = inputs()
    y[:, 0] = [0, 1, 4, 1, 2, 3]
    x = y[::-1][None].copy()
    result = literature_coefficient_metrics(x, y, mask, times,
                                            lip_channels=(0,), upper_channels=(0,))["metrics"]
    assert result["arkit_fdd_absolute"]["value"] == pytest.approx(0, abs=1e-14)
    assert result["arkit_lbe"]["value"] > 0
    x[:] = 0
    result = literature_coefficient_metrics(x, y, mask, times,
                                            upper_channels=(0,))["metrics"]
    assert result["arkit_fdd_signed"]["value"] == pytest.approx(np.std(y[:, 0] ** 2))
    assert result["arkit_fdd_signed"]["value"] != pytest.approx(np.std(np.diff(y[:, 0])))


def test_masked_norm_and_fdd_complete_support_do_not_use_fills_or_compact_gaps():
    x, y, mask, times = inputs()
    y[:, 0] = [1, 2, 100, 200, 5, 6]
    x[:, :, 0] = [0, 0, -100, -200, 0, 0]
    mask[2:4] = False
    x[:, 2:4] = np.nan
    y[2:4] = np.nan
    x[:, :, 1] = np.nan
    y[:, 1] = np.nan
    mask[:, 1] = False
    times[4:] += .2
    report = literature_coefficient_metrics(x, y, mask, times,
                                            lip_channels=(0, 1), upper_channels=(0,))
    actual = report["metrics"]
    assert actual["arkit_lbe"]["value"] == pytest.approx(np.mean([1, 2, 5, 6]))
    assert actual["arkit_fdd_signed"]["value"] == pytest.approx(np.std(np.square([1, 2, 5, 6])))
    assert report["clock_gaps"] == 1
    assert actual["arkit_lbe"]["coverage"]["partial_support"]
    pending = literature_coefficient_metrics(x, y, mask, times, upper_channels=(0, 1))["metrics"]
    assert pending["arkit_fdd_signed"]["status"] == "pending"


def test_empty_or_insufficient_support_produces_pending():
    x, y, mask, times = inputs()
    mask[:] = False
    metrics = literature_coefficient_metrics(x, y, mask, times)["metrics"]
    for name in ("arkit_mbe", "arkit_lbe", "arkit_fdd_signed", "arkit_fdd_absolute"):
        assert metrics[name]["status"] == "pending" and metrics[name]["value"] is None
    mask[0] = True
    metrics = literature_coefficient_metrics(x, y, mask, times)["metrics"]
    assert metrics["arkit_mbe"]["status"] == "computed"
    assert metrics["arkit_fdd_signed"]["status"] == "pending"


def test_default_beat_semantic_regions_exclude_brows_and_project_upper_is_custom():
    x, y, mask, times = inputs()
    result = literature_coefficient_metrics(x, y, mask, times)
    assert result["region_indices"] == {"lips": list(BEAT_LIP12), "upper": list(BEAT_UPPER16)}
    assert not result["region_adaptation"]
    assert not result["fdd_includes_eyebrows"]
    result = literature_coefficient_metrics(x, y, mask, times, upper_channels=UPPER9)
    assert result["region_adaptation"] and result["fdd_includes_eyebrows"]


def test_official_51_channel_order_mapping_matches_literal_beat_scores():
    # Names from pinned official utils/arkit2metahuman.py (not Apple ordering).
    from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
    names = ('browDownLeft browDownRight browInnerUp browOuterUpLeft browOuterUpRight '
             'cheekPuff cheekSquintLeft cheekSquintRight eyeBlinkLeft eyeBlinkRight '
             'eyeLookDownLeft eyeLookDownRight eyeLookInLeft eyeLookInRight eyeLookOutLeft '
             'eyeLookOutRight eyeLookUpLeft eyeLookUpRight eyeSquintLeft eyeSquintRight '
             'eyeWideLeft eyeWideRight jawForward jawLeft jawOpen jawRight mouthClose '
             'mouthDimpleLeft mouthDimpleRight mouthFrownLeft mouthFrownRight mouthFunnel '
             'mouthLeft mouthLowerDownLeft mouthLowerDownRight mouthPressLeft mouthPressRight '
             'mouthPucker mouthRight mouthRollLower mouthRollUpper mouthShrugLower '
             'mouthShrugUpper mouthSmileLeft mouthSmileRight mouthStretchLeft mouthStretchRight '
             'mouthUpperUpLeft mouthUpperUpRight noseSneerLeft noseSneerRight').split()
    order = [ARKIT_NAMES.index(n) for n in names]
    mouth = [23, 25, 27, 28, 31, 37, 39, 40, 41, 42, 47, 48]
    upper = list(range(8, 22)) + [49, 50]
    assert tuple(order[i] for i in mouth) == BEAT_LIP12
    assert set(order[i] for i in upper) == set(BEAT_UPPER16)
    rng = np.random.default_rng(19)
    gt, pred = rng.uniform(size=(2, 31, 51))
    target = np.zeros((31, 52)); sample = target.copy()
    target[:, order] = gt; sample[:, order] = pred
    mask = np.ones_like(target, dtype=bool); mask[:, 51] = False
    metric = literature_coefficient_metrics(sample, target, mask, np.arange(31)/25)['metrics']
    assert metric['arkit_mbe']['value'] == pytest.approx(np.linalg.norm(pred-gt, axis=1).mean())
    assert metric['arkit_lbe']['value'] == pytest.approx(np.linalg.norm(pred[:, mouth]-gt[:, mouth], axis=1).mean())
    expected = np.std(np.sum(gt[:, upper]**2, axis=1)) - np.std(np.sum(pred[:, upper]**2, axis=1))
    assert metric['arkit_fdd_signed']['value'] == pytest.approx(expected)
