import pytest
import torch

from scripts.inspect_audio_text_fit import intensity_scores


def test_correct_clip_dc_cannot_masquerade_as_dynamic_fit():
    target = torch.tensor([[[9.], [11.], [9.], [11.]], [[99.], [101.], [99.], [101.]]])
    prediction = target.mean(1, keepdim=True).expand_as(target)
    scores = intensity_scores(prediction, target, torch.ones_like(target, dtype=torch.bool))
    assert scores["raw"]["correlation"] > .999
    assert scores["raw"]["r2_against_target_mean"] > .999
    assert scores["clip_centered"]["prediction_variance_ddof0"] == 0
    assert scores["clip_centered"]["mse"] == 1
    assert scores["clip_centered"]["r2_against_target_mean"] == 0
    assert scores["clip_centered"]["correlation"] is None


def test_centered_fit_removes_only_clip_dc_preserves_timing_and_ignores_invalid_poison():
    target = torch.tensor([[[1.], [3.], [1.], [3.], [float("nan")]],
                           [[5.], [8.], [5.], [8.], [float("nan")]]])
    valid = torch.tensor([[[True]] * 4 + [[False]], [[True]] * 4 + [[False]]])
    prediction = target + torch.tensor([10., -3.])[:, None, None]
    scores = intensity_scores(prediction, target, valid)
    assert scores["raw"]["mse"] == pytest.approx(54.5)
    assert scores["clip_centered"]["mse"] == 0
    assert scores["clip_centered"]["r2_against_target_mean"] == 1
    assert scores["clip_centered"]["correlation"] == pytest.approx(1.)
    assert scores["clip_centered"]["prediction_variance_ratio"] == 1
    reversed_values = prediction.clone(); reversed_values[:, :4] = reversed_values[:, :4].flip(1)
    wrong = intensity_scores(reversed_values, target, valid)
    assert wrong["clip_centered"]["correlation"] == pytest.approx(-1.)
    assert wrong["clip_centered"]["mse"] > 0
