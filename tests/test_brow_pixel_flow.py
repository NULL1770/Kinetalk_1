import hashlib

import cv2
import numpy as np
import pytest

from scripts import audit_brow_pixel_flow as p


def landmarks():
    a = np.full((478, 3), .5, float)
    a[33, :2], a[263, :2] = [.25, .45], [.75, .45]
    for indices, xs in [(p.BROWS["right"], np.linspace(.22, .4, 5)),
                        (p.BROWS["left"], np.linspace(.6, .78, 5))]:
        a[indices, :2] = np.column_stack([xs, np.full(5, .30)])
    for index, point in zip(p.STABLE["nose_bridge"], [[.5, .38], [.5, .44], [.5, .50], [.5, .56]]):
        a[index, :2] = point
    for index, point in zip(p.STABLE["nose_sides"], [[.42, .6], [.58, .6], [.40, .56], [.60, .56]]):
        a[index, :2] = point
    for index, point in zip(p.STABLE["outer_temples"], [[.10, .35], [.90, .35], [.10, .55], [.90, .55]]):
        a[index, :2] = point
    for inds, x in [([133, 160, 159, 158, 144, 145, 153], .32), ([362, 385, 386, 387, 380, 374, 373], .68)]:
        a[inds, :2] = [x, .45]
    for index in [61, 291, 0, 17, 13, 14]:
        a[index, :2] = [.5, .8]
    return a


def texture():
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, (240, 320), np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .6)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def test_roi_initialization_excludes_eyelids_and_lips_from_stable_motion():
    lm = landmarks()
    brow_masks, stable_masks, distance = p.initial_regions(lm, (240, 320))
    assert distance == pytest.approx(160.)
    assert all(mask.any() for mask in brow_masks)
    combined = np.maximum.reduce(stable_masks)
    for index in [33, 133, 263, 362, 13, 14, 61, 291]:
        x, y = np.rint(lm[index, :2] * [320, 240]).astype(int)
        assert combined[y, x] == 0


def test_similarity_removes_translation_and_keeps_residual_brow_motion():
    old = np.array([[10, 10], [40, 10], [70, 10], [10, 40], [40, 40], [70, 40], [10, 70], [70, 70]], np.float32)
    matrix, report = p.fit_similarity(old, old + [3., -2.], 100)
    assert report["reason"] is None
    np.testing.assert_allclose(matrix, [[1, 0, 3], [0, 1, -2]], atol=1e-5)
    eyebrow = np.array([[20, 15], [25, 15], [30, 15], [35, 15]], float)
    observed = eyebrow + [3, -4]
    corrected = observed - (eyebrow @ matrix[:, :2].T + matrix[:, 2])
    np.testing.assert_allclose(corrected[:, 1], -2, atol=1e-5)


def test_lk_translation_is_corrected_using_only_image_features():
    first = texture()
    second = cv2.warpAffine(first, np.float32([[1, 0, 2], [0, 1, 1]]), (320, 240))
    tracker = p.PixelTracker(first, landmarks())
    result = tracker.step(second)
    assert result["missing_reason"] == [None, None]
    np.testing.assert_allclose(result["raw_px"][:, 1], 1, atol=.2)
    np.testing.assert_allclose(result["corrected_px"], 0, atol=.2)


def test_low_texture_outputs_missing_instead_of_zero_motion():
    first = np.full((240, 320, 3), 120, np.uint8)
    tracker = p.PixelTracker(first, landmarks())
    result = tracker.step(first)
    assert np.isnan(result["raw_px"]).all()
    assert np.isnan(result["corrected_px"]).all()
    assert result["missing_reason"] == ["insufficient_brow_tracks", "insufficient_brow_tracks"]


def test_insufficient_or_clustered_stable_features_reject_similarity():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, 2], [2, 2]], np.float32)
    assert p.fit_similarity(points[:4], points[:4] + 1, 100)[1]["reason"] == "insufficient_stable_tracks"
    assert p.fit_similarity(points, points + 1, 100)[1]["reason"] == "stable_inliers_not_spatially_distributed"


def test_frame_hash_is_exact_and_cannot_be_silently_retimed():
    first = texture()
    digest = hashlib.sha256(first.tobytes()).hexdigest()
    np.testing.assert_array_equal(p.checked_frame(first, digest), first)
    changed = first.copy()
    changed[0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="RGB SHA"):
        p.checked_frame(changed, digest)


def test_valid_gap_reinitializes_without_cross_gap_motion_or_later_landmarks():
    first = texture()
    frames = [first.copy() for _ in range(5)]
    lm = np.repeat(landmarks()[None], 5, axis=0)
    # Later landmarks are deliberately unusable: only run-start ROIs may be read.
    lm[1] = np.nan
    lm[4] = np.nan
    a = {"valid": np.array([True, True, False, True, True]), "times": np.arange(5) / 25.,
         "landmarks": lm, "rgb_sha256": np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in frames])}
    report, arrays = p.diagnose_frames(iter(frames), a, "gap")
    assert report["valid_pairs"] == 2
    assert [r["start"] for r in report["runs"]] == [0, 3]
    assert arrays["corrected_observed"][[1, 4]].all()
    assert np.isnan(arrays["corrected_px"][[0, 2, 3]]).all()
    assert arrays["missing_reason"][3, 0] == "run_initialization_no_difference"


def test_missing_initial_roi_stays_missing_until_next_valid_run():
    frames = [texture(), texture()]
    a = {"valid": np.ones(2, bool), "times": np.arange(2) / 25.,
         "landmarks": np.full((2, 478, 3), np.nan),
         "rgb_sha256": np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in frames])}
    report, arrays = p.diagnose_frames(iter(frames), a, "missing")
    assert not arrays["corrected_observed"].any()
    assert report["sides"]["left"]["observed_pairs"] == 0
    assert report["sides"]["left"]["corrected_vertical_rms_normalized"] is None


def test_decoded_frame_count_mismatch_rejected():
    first = texture()
    a = {"valid": np.ones(2, bool), "times": np.arange(2) / 25.,
         "landmarks": np.repeat(landmarks()[None], 2, axis=0),
         "rgb_sha256": np.array([hashlib.sha256(first.tobytes()).hexdigest()] * 2)}
    with pytest.raises(ValueError, match="fewer frames"):
        p.diagnose_frames(iter([first]), a, "short")
