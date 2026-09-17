from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinetalk_b0.predictable_motion import (
    bin_centered_frames, decode_controls, fit_motion_path, fit_predictable_motion,
    motion_target, predict_controls, predict_motion, weighted_clip_center,
)


def _sample(seed=7):
    torch.manual_seed(seed)
    x = torch.randn(8, 11, 5, dtype=torch.float64)
    y = torch.randn(8, 11, 3, dtype=torch.float64)
    w = torch.randint(1, 5, (8, 11)).double()
    w[:, -2:] = 0
    return x, y, w, torch.arange(6)


def test_weighted_ridge_matches_direct_solve_and_rrr_is_regularized_optimum():
    x, y, w, ids = _sample()
    states = fit_motion_path(x, y, w, ids, [.03, 1.], [1, 3])
    xc, yc = weighted_clip_center(x[ids], w[ids]), weighted_clip_center(y[ids], w[ids])
    flat_w = w[ids].reshape(-1, 1)
    for alpha in (.03, 1.):
        direct = next(s for s in states if s["alpha"] == alpha and s["method"] == "ridge")
        design = (xc / direct["std"]).reshape(-1, x.shape[-1])
        target = yc.reshape(-1, y.shape[-1])
        gram = design.T @ (design * flat_w) / flat_w.sum()
        cross = design.T @ (target * flat_w) / flat_w.sum()
        a = gram + alpha * torch.eye(x.shape[-1], dtype=torch.float64)
        expected = torch.linalg.solve(a, cross)
        torch.testing.assert_close(direct["weights"], expected, atol=1e-12, rtol=1e-10)
        # Independent SVD of A^-1/2 H verifies the regularized RRR basis.
        ev, evec = torch.linalg.eigh(a)
        whitened_cross = (evec * ev.rsqrt()) @ evec.T @ cross
        _, _, vh = torch.linalg.svd(whitened_cross, full_matrices=False)
        u = vh[:1].T
        expected_rrr = expected @ u @ u.T
        reduced = next(s for s in states if s["alpha"] == alpha and s["method"] == "rrr" and s["rank"] == 1)
        torch.testing.assert_close(reduced["weights"] @ reduced["basis"].T, expected_rrr, atol=1e-12, rtol=1e-10)
        for method in ("rrr", "pca"):
            full = next(s for s in states if s["alpha"] == alpha and s["method"] == method and s["rank"] == 3)
            torch.testing.assert_close(predict_motion(x, w, full), predict_motion(x, w, direct), atol=1e-12, rtol=1e-10)
    for state in states:
        torch.testing.assert_close(state["basis"].T @ state["basis"], torch.eye(state["rank"], dtype=torch.float64), atol=1e-12, rtol=0)


def test_rrr_selects_predictable_direction_while_pca_selects_large_independent_motion():
    torch.manual_seed(9)
    w = torch.ones(30, 12, dtype=torch.float64)
    x = weighted_clip_center(torch.randn(30, 12, 3, dtype=torch.float64), w)
    nuisance = weighted_clip_center(torch.randn(30, 12, 1, dtype=torch.float64), w)
    # Make nuisance exactly uncorrelated with every training input direction.
    ids = torch.arange(24)
    xt = x[ids].reshape(-1, 3)
    coef = torch.linalg.lstsq(xt, nuisance[ids].reshape(-1, 1)).solution
    nuisance = nuisance - x @ coef
    y = torch.cat([x[..., :1], 50 * nuisance, torch.zeros_like(nuisance)], -1)
    states = fit_motion_path(x, y, w, ids, [.001], [1], methods=("rrr", "pca"))
    rrr, pca = states
    assert rrr["basis"][0, 0].abs() > .999999
    assert pca["basis"][1, 0].abs() > .999999
    pred_rrr, pred_pca = predict_motion(x[24:], w[24:], rrr), predict_motion(x[24:], w[24:], pca)
    assert ((pred_rrr[..., 0] - y[24:, :, 0]) ** 2).mean() < 1e-5
    assert ((pred_pca[..., 0] - y[24:, :, 0]) ** 2).mean() > .1
    # The teacher is real motion projection, not the audio predictor's labels.
    changed_y = y[24:].clone()
    changed_y[:, :, 0] += x[24:, :, 1]
    assert not torch.allclose(motion_target(changed_y, w[24:], rrr), motion_target(y[24:], w[24:], rrr))
    torch.testing.assert_close(predict_motion(x[24:], w[24:], rrr), pred_rrr)


def test_heldout_values_never_affect_fit_even_when_nonfinite():
    x, y, w, ids = _sample()
    first = fit_motion_path(x, y, w, ids, [.01, .3], [1, 2])
    x[6:] = float("nan")
    y[6:] = float("inf")
    w[6:] = float("nan")
    second = fit_motion_path(x, y, w, ids, [.01, .3], [1, 2])
    for a, b in zip(first, second):
        for key in ("std", "basis", "weights", "train_ids"):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_partial_bins_padding_and_weighted_centering_ignore_invalid_values():
    frames = torch.tensor([[[1.], [2.], [3.], [4.], [5.], [float("nan")]],
                           [[float("nan")]] * 6])
    valid = torch.tensor([[True, True, True, True, True, False], [False] * 6])
    values, w = bin_centered_frames(frames, valid, 2)
    torch.testing.assert_close(w, torch.tensor([[2., 2., 1.], [0., 0., 0.]], dtype=torch.float64))
    torch.testing.assert_close(values[0, :, 0], torch.tensor([-1.5, .5, 2.], dtype=torch.float64))
    assert not values[1].any()
    x, y, weight, ids = _sample()
    expected = fit_predictable_motion(x, y, weight, ids, alpha=.1, rank=2)
    x[weight == 0] = float("nan")
    y[weight == 0] = float("inf")
    actual = fit_predictable_motion(x, y, weight, ids, alpha=.1, rank=2)
    for key in ("std", "basis", "weights"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    pred = predict_controls(x, weight, actual)
    assert torch.isfinite(pred).all()
    assert not pred[weight == 0].any()
    torch.testing.assert_close((pred * weight[..., None]).sum(1), torch.zeros(8, 2, dtype=torch.float64), atol=1e-12, rtol=0)


def test_zero_inputs_targets_produce_finite_zero_predictions():
    x, y = torch.zeros(3, 4, 5), torch.zeros(3, 4, 2)
    w = torch.ones(3, 4)
    states = fit_motion_path(x, y, w, [0, 1], [.1], [1, 2])
    for state in states:
        pred = predict_motion(x, w, state)
        assert torch.isfinite(pred).all() and not pred.any()
        assert state["zero_variance_features"] == 5
        assert not motion_target(y, w, state).any()
        assert not predict_motion(x + float("nan"), w * 0, state).any()


def test_validation_rejects_bad_weights_values_dimensions_and_parameters():
    x, y, w, ids = _sample()
    with pytest.raises(ValueError, match="positive observed weight"):
        fit_predictable_motion(x, y, w * 0, ids, rank=1)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_predictable_motion(x, y, -w, ids, rank=1)
    bad_x = x.clone()
    bad_x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="observed"):
        fit_predictable_motion(bad_x, y, w, ids, rank=1)
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alphas"):
            fit_predictable_motion(x, y, w, ids, alpha=invalid, rank=1)
    for rank in (0, 4, 1.5, True, None):
        with pytest.raises(ValueError, match="ranks"):
            fit_predictable_motion(x, y, w, ids, rank=rank)
    with pytest.raises(ValueError, match="unique"):
        fit_predictable_motion(x, y, w, [0, 0], rank=1)
    state = fit_predictable_motion(x, y, w, ids, rank=1)
    with pytest.raises(ValueError, match="dimension"):
        predict_motion(x[..., :2], w, state)
    with pytest.raises(ValueError, match="dimension"):
        motion_target(y[..., :2], w, state)
    with pytest.raises(ValueError, match="controls"):
        decode_controls(torch.zeros(2, 3, 2), state)
