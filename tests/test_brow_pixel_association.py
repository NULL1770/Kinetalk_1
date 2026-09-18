import numpy as np
import pytest

from scripts import analyze_brow_pixel_association as a


def arrays():
    valid = np.array([True, True, True, True, True])
    observed = np.ones((5, 2), bool)
    observed[0] = False
    movement = np.zeros((5, 2, 2))
    movement[0] = np.nan
    movement[1:, :, 1] = [[-.1, -.2], [-.2, -.3], [.1, .2], [.2, .3]]
    p = {"valid": valid, "times": np.arange(5) / 25., "corrected_observed": observed,
         "corrected_normalized": movement, "pair_valid": np.array([False, True, True, True, True])}
    s = {"valid": valid.copy()}
    n = {"valid": valid.copy(), "times": p["times"].copy(),
         "blink_observed_pairs": np.array([True, True, False, True]),
         "blink_neighborhood_pairs": np.array([False, True, False, False])}
    return p, s, n


def test_image_down_sign_and_coefficient_direction_are_explicit():
    dy = np.array([-.2, -.1, .1, .3])
    raise_delta = -dy
    down_delta = dy
    assert a.pearson(raise_delta, a.directional_pixel("raise", dy)) == pytest.approx(1)
    assert a.pearson(down_delta, a.directional_pixel("down", dy)) == pytest.approx(1)


def test_difference_ending_frame_is_same_clock_and_blink_missing_is_neither_arm():
    p, s, n = arrays()
    dy, support, arms = a.aligned_evidence(p, s, n)
    np.testing.assert_array_equal(dy[:, :2], p["corrected_normalized"][1:, :, 1])
    assert support.shape == (4, 3)
    assert arms["blink_near"].tolist() == [False, True, False, False]
    assert arms["nonblink"].tolist() == [True, False, False, True]
    assert not (arms["blink_near"] & arms["nonblink"]).any()


def test_shifted_clock_is_rejected_not_aligned_by_index():
    p, s, n = arrays()
    n["times"] += .04
    with pytest.raises(ValueError, match="clocks"):
        a.aligned_evidence(p, s, n)


def test_unilateral_missing_does_not_create_bilateral_zero_or_false_evidence():
    p, s, n = arrays()
    p["corrected_observed"][2, 1] = False
    p["corrected_normalized"][2, 1] = np.nan
    dy, support, _ = a.aligned_evidence(p, s, n)
    assert support[1].tolist() == [True, False, False]
    assert np.isnan(dy[1, 2])
    stats = a.stat(np.arange(4, dtype=float), dy[:, 2], support[:, 2])
    assert stats["pairs"] == 3


def test_missing_data_and_constant_signal_have_no_correlation():
    assert a.stat(np.ones(4), np.ones(4), np.ones(4, bool))["pearson_expected_sign"] is None
    assert a.stat(np.ones(4), np.full(4, np.nan), np.ones(4, bool))["pairs"] == 0


def test_gap_pairs_and_invalid_blink_masks_rejected():
    p, s, n = arrays()
    p["pair_valid"][0] = True
    with pytest.raises(ValueError, match="crosses"):
        a.aligned_evidence(p, s, n)
    p, s, n = arrays()
    n["blink_neighborhood_pairs"][2] = True
    with pytest.raises(ValueError, match="blink"):
        a.aligned_evidence(p, s, n)
