from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.emotion_ray_metrics import field_metrics


def test_weighted_metrics_distinguish_correct_energy_from_correct_prediction():
    target = torch.tensor([[[-1., 2.], [1., -2.]], [[-2., 1.], [2., -1.]]])
    weight = torch.tensor([[1., 3.], [2., 2.]])
    exact = field_metrics(target, target, weight, ["a", "b"])
    assert exact["native_mse"] == 0
    assert exact["zero_mse"] == pytest.approx(2.5)
    assert exact["r2_against_zero"] == 1
    assert exact["pooled_centered_correlation"] == pytest.approx(1)
    assert exact["energy_ratio"] == 1
    assert exact["bootstrap_r2_ci95"] == [1., 1.]
    wrong = field_metrics(-target, target, weight, ["a", "b"])
    assert wrong["energy_ratio"] == 1
    assert wrong["r2_against_zero"] == -3
    assert wrong["pooled_centered_correlation"] == pytest.approx(-1)
    baseline = field_metrics(torch.zeros_like(target), target, weight, ["a", "b"])
    assert baseline["r2_against_zero"] == 0
    assert baseline["pooled_centered_correlation"] is None


def test_centering_is_per_clip_and_channel_and_padding_nan_is_ignored():
    target = torch.tensor([[[-1., 2.], [1., -2.], [0., 0.]],
                           [[-2., 1.], [2., -1.], [0., 0.]]])
    offsets = torch.tensor([[[10., -30.]], [[-40., 20.]]])
    prediction = target + offsets
    weight = torch.tensor([[1., 2., 0.], [3., 1., 0.]])
    reference = field_metrics(prediction, target, weight, ["a", "b"])
    assert reference["pooled_centered_correlation"] == pytest.approx(1)
    assert reference["r2_against_zero"] < 0
    prediction[:, 2] = float("nan")
    target[:, 2] = float("nan")
    actual = field_metrics(prediction, target, weight, ["a", "b"])
    assert actual == reference
    json.dumps(actual, allow_nan=False)


def test_sentence_bootstrap_samples_clusters_and_matches_direct_calculation():
    target = torch.tensor([[[-1.], [1.]], [[-2.], [2.]], [[-1.], [1.]]])
    prediction = target.clone()
    prediction[2] *= -1
    weight = torch.ones(3, 2)
    actual = field_metrics(prediction, target, weight, ["a", "a", "b"],
                           bootstrap_seed=12, bootstrap_samples=300)
    assert actual["sentences"] == 2
    assert actual["per_sentence"]["a"]["clips"] == 2
    assert actual["per_sentence"]["a"]["zero"] == 10
    assert actual["per_sentence"]["b"]["sse"] == 8
    sampled = np.random.default_rng(12).integers(2, size=(300, 2))
    totals = np.array([[0., 10.], [8., 2.]])[sampled].sum(1)
    expected = np.quantile(1 - totals[:, 0] / totals[:, 1], [.025, .975])
    assert actual["bootstrap_r2_ci95"] == pytest.approx(expected.tolist())
    assert actual["r2_against_zero"] == pytest.approx(1 - 8 / 12)


def test_undefined_targets_empty_weights_and_single_sentence_are_explicit():
    target = torch.zeros(2, 3, 1)
    prediction = torch.ones_like(target)
    weight = torch.ones(2, 3)
    actual = field_metrics(prediction, target, weight, ["a", "b"])
    assert actual["native_mse"] == 1
    assert actual["prediction_energy"] == 6
    for key in ("r2_against_zero", "pooled_centered_correlation", "energy_ratio",
                "bootstrap_r2_ci95"):
        assert actual[key] is None
    empty = field_metrics(prediction, target, weight * 0, ["a", "b"])
    assert empty["native_mse"] is None and empty["sentences"] == 0
    single = field_metrics(prediction, prediction, weight, ["a", "a"])
    assert single["r2_against_zero"] == 1
    assert single["bootstrap_r2_ci95"] is None
    json.dumps([actual, empty, single], allow_nan=False)


def test_invalid_observed_values_and_weights_are_rejected():
    target = torch.zeros(1, 2, 1)
    weight = torch.ones(1, 2)
    with pytest.raises(ValueError, match="sentence_ids"):
        field_metrics(target, target, weight, [])
    with pytest.raises(ValueError, match="nonnegative"):
        field_metrics(target, target, -weight, ["a"])
    with pytest.raises(ValueError, match="observed"):
        field_metrics(target + float("nan"), target, weight, ["a"])
