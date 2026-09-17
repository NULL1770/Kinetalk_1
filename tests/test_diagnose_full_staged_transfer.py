import pytest
import torch

from scripts.diagnose_full_staged_transfer import MODES, global_metrics, sequence_metrics, swap_condition


def test_swaps_only_requested_paths_and_binds_global_with_intensity():
    audio = {"global": torch.ones(2, 3), "intensity_value": torch.ones(2, 1) * 2,
             "local": torch.ones(2, 5, 3) * 3, "emotion_logits": torch.full((2, 8), float("nan"))}
    teacher = {"global": torch.ones(2, 3) * 4, "intensity_value": torch.ones(2, 1) * 5, "local": torch.ones(2, 5, 3) * 6}
    for mode in MODES:
        value = swap_condition(audio, teacher, mode)
        assert set(value) == {"global", "intensity_value", "local"}
        tg = mode in ("teacher_global_audio_local", "teacher_all")
        assert torch.equal(value["global"], teacher["global"] if tg else audio["global"])
        assert torch.equal(value["intensity_value"], teacher["intensity_value"] if tg else audio["intensity_value"])
        expected = torch.zeros_like(audio["local"]) if mode == "audio_global_zero_local" else teacher["local"] if mode in ("audio_global_teacher_local", "teacher_all") else audio["local"]
        assert torch.equal(value["local"], expected)
    with pytest.raises(ValueError): swap_condition(audio, teacher, "unknown")


def test_sequence_metrics_separates_mean_offset_from_temporal_error_and_masks_gaps():
    target = torch.tensor([[[1.], [2.], [float("nan")], [3.]], [[2.], [4.], [6.], [float("nan")]]])
    valid = torch.isfinite(target[..., 0])
    prediction = target + 5
    result = sequence_metrics(prediction, target, valid)
    assert result["raw_mse"] == pytest.approx(25)
    assert result["clip_mean_mse"] == pytest.approx(25)
    assert result["centered_mse"] == pytest.approx(0, abs=1e-14)
    assert result["centered_correlation"] == pytest.approx(1)
    assert result["centered_r2"] == pytest.approx(1)
    amplified = sequence_metrics(target * 2, target, valid)
    assert amplified["centered_mse"] > 0
    assert amplified["prediction_temporal_rms"] == pytest.approx(2 * amplified["target_temporal_rms"])


def test_global_metrics_cross_clip_is_separate_from_temporal_coordinates():
    value = torch.tensor([[1., 1.], [2., 1.], [3., 1.]])
    result = global_metrics(value + 2, value)
    assert result["raw_mse"] == 4
    assert result["nonconstant_dimensions"] == 1
    assert result["across_clip_pooled_centered_correlation"] == pytest.approx(1)
    assert result["across_clip_mean_nonconstant_dimension_correlation"] == pytest.approx(1)
