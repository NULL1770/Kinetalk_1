import numpy as np
import pytest

from scripts import extract_visual_semantics_pilot as s


def test_va_is_real_regression_head_and_classes_softmax_separately():
    output = np.array([[1000, 999, -1000, 0, 0, 0, 0, 0, -.3, .7]], np.float32)
    result = s.split_outputs(output)
    np.testing.assert_allclose(result["va"], [[-.3, .7]])
    np.testing.assert_allclose(result["q"].sum(1), 1)
    assert result["q"][0, 0] > result["q"][0, 1]
    np.testing.assert_array_equal(result["logits"], output[:, :8])
    assert s.CLASS_NAMES[0] == "angry" and s.CLASS_NAMES[5] == "neutral"


def test_raw_va_preserved_even_if_clipped_output_is_bounded():
    output = np.zeros((1, 10), np.float32)
    output[0, 8:] = [-1.1, 1.2]
    result = s.split_outputs(output)
    np.testing.assert_array_equal(result["va"], [[-1, 1]])
    np.testing.assert_array_equal(result["va_raw"], output[:, 8:])


def test_nonfinite_or_wrong_teacher_output_rejected():
    with pytest.raises(ValueError):
        s.split_outputs(np.zeros((3, 8)))
    output = np.zeros((2, 10))
    output[0, 8] = np.nan
    with pytest.raises(ValueError):
        s.split_outputs(output)


def test_rgb_preprocessing_preserves_channel_order():
    image = np.zeros((10, 10, 3), np.uint8)
    image[:, :, 0] = 255
    processed = s.preprocess(image)
    assert processed.shape == (3, 224, 224)
    np.testing.assert_allclose(processed[:, 0, 0], [(1 - .485) / .229, -.456 / .224, -.406 / .225], rtol=1e-6)


def test_scrfd_nms_removes_duplicate_and_preserves_separate_face():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [40, 40, 50, 50]], float)
    np.testing.assert_array_equal(s.nms(boxes, np.array([.9, .8, .7])), [0, 2])


def test_stride_times_missing_faces_and_q_are_not_interpolated(monkeypatch, tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"pinned-test")
    frames = [np.full((10, 10, 3), index, np.uint8) for index in range(12)]
    monkeypatch.setattr(s, "video_frames", lambda *args: iter(frames))
    class Generator:
        def __iter__(self):
            return iter(frames)
        def close(self):
            pass
    monkeypatch.setattr(s, "video_frames", lambda *args: Generator())
    class Teacher:
        def detect(self, rgb):
            if rgb[0, 0, 0] == 5:
                return np.empty((0, 4)), np.empty(0)
            return np.array([[0, 0, 10, 10]]), np.array([.8])
        def infer(self, crops):
            return s.split_outputs(np.zeros((len(crops), 10), np.float32))
    output, report = s.extract_clip({"video": str(video), "video_sha256": s.sha(video),
                                     "clip_id": "x", "needs_resample": True}, Teacher(), "none", stride=5)
    np.testing.assert_array_equal(output["frame_indices"], [0, 5, 10])
    np.testing.assert_allclose(output["times"], [0, .2, .4])
    assert output["valid"].tolist() == [True, False, True]
    assert np.isnan(output["q"][1]).all() and np.isnan(output["va"][1]).all()
    assert report["valid_samples"] == 2
