"""Independent image-level checks for the diagnostic, not semantic validation."""
import cv2
import numpy as np
import pytest

from scripts import audit_brow_pixel_flow as flow


def _face():
    points = np.full((478, 3), .5, dtype=float)
    points[33, :2], points[263, :2] = (.25, .45), (.75, .45)
    for indices, xs in [(flow.BROWS["right"], np.linspace(.2, .4, 5)),
                        (flow.BROWS["left"], np.linspace(.6, .8, 5))]:
        points[indices, :2] = np.column_stack([xs, np.full(5, .28)])
    for indices, xy in [(flow.STABLE["nose_bridge"], [( .5, .40), (.5, .45), (.5, .50), (.5, .55)]),
                        (flow.STABLE["nose_sides"], [(.42, .58), (.58, .58), (.40, .62), (.60, .62)]),
                        (flow.STABLE["outer_temples"], [(.08, .40), (.92, .40), (.08, .60), (.92, .60)])]:
        points[indices, :2] = xy
    for indices, x in [([133, 160, 159, 158, 144, 145, 153], .33),
                       ([362, 385, 386, 387, 380, 374, 373], .67)]:
        points[indices, :2] = (x, .45)
    points[[61, 291, 0, 17, 13, 14], :2] = (.5, .8)
    return points


def _image(shape=(300, 400)):
    rng = np.random.default_rng(97612)
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .55)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


@pytest.mark.parametrize("degrees,scale", [(3., 1.025), (-3., .975)])
def test_global_rotation_and_scale_are_removed_from_pixel_flow(degrees, scale):
    first = _image()
    transform = cv2.getRotationMatrix2D((200, 150), degrees, scale)
    transform[:, 2] += [1., -1.]
    second = cv2.warpAffine(first, transform, (400, 300))
    result = flow.PixelTracker(first, _face()).step(second)
    assert result["missing_reason"] == [None, None]
    np.testing.assert_allclose(result["corrected_px"], 0., atol=.25)


def test_similarity_keeps_local_signed_deformation_under_rotation_and_zoom():
    old = np.array([(x, y) for x in (40, 100, 220, 300) for y in (80, 140, 190)], dtype=np.float32)
    transform = cv2.getRotationMatrix2D((180, 130), 9., 1.2)
    new = old @ transform[:, :2].T + transform[:, 2]
    estimated, diagnostics = flow.fit_similarity(old, new.astype(np.float32), 160)
    assert diagnostics["reason"] is None
    brow = np.array([[80., 60.], [85., 60.], [90., 60.], [95., 60.]])
    local = np.array([0., -3.])
    moved = (brow + local) @ transform[:, :2].T + transform[:, 2]
    corrected = moved - (brow @ estimated[:, :2].T + estimated[:, 2])
    np.testing.assert_allclose(corrected, np.tile(local @ transform[:, :2].T, (4, 1)), atol=2e-5)


def test_image_local_brow_displacement_remains_while_other_side_is_stationary():
    first = _image()
    yy, xx = np.mgrid[:300, :400].astype(np.float32)
    # Broad constant displacement over one brow; transition lies outside LK windows.
    influence = np.zeros((300, 400), np.float32)
    influence[45:120, 55:185] = 1.
    influence = cv2.GaussianBlur(influence, (11, 11), 2.)
    second = cv2.remap(first, xx, yy + 2 * influence, cv2.INTER_LINEAR)
    result = flow.PixelTracker(first, _face()).step(second)
    assert result["missing_reason"] == [None, None]
    np.testing.assert_allclose(result["corrected_px"][0], [0., -2.], atol=.35)
    np.testing.assert_allclose(result["corrected_px"][1], 0., atol=.15)


def test_total_feature_loss_is_missing_and_reappearance_does_not_reseed():
    first = _image()
    tracker = flow.PixelTracker(first, _face())
    absent = np.full_like(first, 127)
    lost = tracker.step(absent)
    assert np.isnan(lost["corrected_px"]).all()
    remaining = [len(points) for points in tracker.brows]
    recovered_image = tracker.step(first)
    assert all(len(points) <= before for points, before in zip(tracker.brows, remaining))
    assert np.isnan(recovered_image["corrected_px"]).all()


def test_one_brow_occlusion_marks_only_that_side_missing():
    first = _image()
    occluded = first.copy()
    occluded[20:130, 30:195] = 127
    tracker = flow.PixelTracker(first, _face())
    result = tracker.step(occluded)
    assert result["missing_reason"] == ["insufficient_brow_tracks", None]
    assert np.isnan(result["corrected_px"][0]).all()
    np.testing.assert_allclose(result["corrected_px"][1], 0., atol=.15)
    returned = tracker.step(first)
    assert returned["missing_reason"][0] == "insufficient_brow_tracks"
    assert np.isnan(returned["corrected_px"][0]).all()


def test_rounded_resize_returns_original_pixel_units():
    first = _image((751, 1001))
    second = cv2.warpAffine(first, np.float32([[1, 0, 4], [0, 1, -3]]), (1001, 751))
    result = flow.PixelTracker(first, _face()).step(second)
    assert result["missing_reason"] == [None, None]
    np.testing.assert_allclose(result["raw_px"], np.tile([4., -3.], (2, 1)), atol=.3)
    np.testing.assert_allclose(result["corrected_px"], 0., atol=.3)
