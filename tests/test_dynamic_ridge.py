import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_dynamic_ridge import (
    centered_features, intervene_features, nested_select_alpha, predict, ridge_path,
    sentence_folds, expression_only_field,
)
from scripts.probe_dynamic_predictability import center_controls


def test_sentence_folds_exclude_outer_and_preserve_groups():
    sentences = ["a", "a", "b", "b", "c", "d", "e", "heldout"]
    train = torch.arange(7)
    folds = sentence_folds(sentences, train, 45, 3)
    validation_ids = []
    for fit, validation in folds:
        assert set(fit.tolist()).isdisjoint(validation.tolist())
        assert set(sentences[i] for i in fit).isdisjoint(sentences[i] for i in validation)
        assert 7 not in fit and 7 not in validation
        validation_ids += validation.tolist()
    assert sorted(validation_ids) == train.tolist()


def test_ridge_path_matches_weighted_direct_solve_and_masks_padding():
    torch.manual_seed(21)
    frames = torch.randn(5, 11, 4)
    valid = torch.ones(5, 11, dtype=torch.bool)
    valid[-1, -3:] = False
    frames[-1, -3:] = float("nan")
    features, weight = centered_features(frames, valid, 3)
    target = features @ torch.tensor([[0.3], [-0.2], [0.5], [0.1]], dtype=torch.float64)
    train = torch.tensor([0, 1, 2, 3])
    coefficients, std = ridge_path(features, target, weight, train, [0.01, 1.0])
    x = (features[train] / std).reshape(-1, 4)
    y, w = target[train].reshape(-1, 1), weight[train].reshape(-1)
    gram, rhs = x.T @ (x * w[:, None]) / w.sum(), x.T @ (y * w[:, None]) / w.sum()
    for alpha, coefficient in zip([0.01, 1.0], coefficients):
        expected = torch.linalg.solve(gram + alpha * torch.eye(4, dtype=torch.float64), rhs)
        torch.testing.assert_close(coefficient, expected, rtol=1e-10, atol=1e-12)
    output = predict(features, weight, coefficients[0], std)
    assert output.dtype == torch.float64 and torch.isfinite(output).all()
    torch.testing.assert_close((output * weight[..., None]).sum(1), torch.zeros(5, 1, dtype=torch.float64), atol=1e-12, rtol=0)


def test_inner_selection_and_fit_are_invariant_to_outer_heldout_values():
    torch.manual_seed(2)
    weight = torch.ones(7, 8, dtype=torch.float64)
    features = center_controls(torch.randn(7, 8, 3, dtype=torch.float64), weight)
    target = features[..., :1] * 0.4
    sentences = list("abcdefg")
    train = torch.arange(6)
    first_alpha, first = nested_select_alpha(features, target, weight, sentences, train, [0.001, 1.0])
    first_coefficients, first_std = ridge_path(features, target, weight, train, [first_alpha])
    altered_x, altered_y = features.clone(), target.clone()
    altered_x[6] = float("nan")
    altered_y[6] = float("nan")
    second_alpha, second = nested_select_alpha(altered_x, altered_y, weight, sentences, train, [0.001, 1.0])
    second_coefficients, second_std = ridge_path(altered_x, altered_y, weight, train, [second_alpha])
    assert first_alpha == second_alpha and first == second
    torch.testing.assert_close(first_std, second_std, rtol=0, atol=0)
    torch.testing.assert_close(first_coefficients[0], second_coefficients[0], rtol=0, atol=0)


def test_interventions_handle_partial_bins_and_cannot_reintroduce_dc():
    weight = torch.tensor([[4., 4., 1.], [4., 2., 0.]], dtype=torch.float64)
    features = center_controls(torch.tensor([[[1.], [3.], [7.]], [[4.], [2.], [float("nan")]]], dtype=torch.float64), weight)
    for mode in ("zero", "reverse", "shuffle"):
        altered = intervene_features(features, weight, mode)
        assert torch.isfinite(altered).all()
        assert torch.equal(altered[weight == 0], torch.zeros_like(altered[weight == 0]))
        torch.testing.assert_close((altered * weight[..., None]).sum(1), torch.zeros(2, 1, dtype=torch.float64), atol=1e-12, rtol=0)


def test_expression_target_excludes_blink_gaze_and_mouth_nuisance():
    torch.manual_seed(18)
    residual = torch.randn(2, 13, 52)
    valid = torch.ones(2, 13, dtype=torch.bool)
    valid[1, -2:] = False
    mask = torch.ones(2, 52, dtype=torch.bool)
    expected, weight = expression_only_field(residual, valid, mask, 4)
    changed = residual.clone()
    keep = [5, 6, 12, 13, 41, 42, 43, 44, 45]
    nuisance = [i for i in range(52) if i not in keep]
    changed[:, :, nuisance] = 1e5
    changed[~valid] = float('nan')
    actual, _ = expression_only_field(changed, valid, mask, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    changed[0, 0, 43] += 3
    changed_field, _ = expression_only_field(changed, valid, mask, 4)
    assert not torch.equal(changed_field, expected)
