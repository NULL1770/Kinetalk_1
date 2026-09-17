import copy

import pytest
import torch

from kinetalk_b0.predictable_motion import fit_motion_path, weighted_clip_center
from kinetalk_b0.temporal_motion_refiner import fit_control_scale
from scripts.train_temporal_audio_refiner_probe import (
    buffer_hash, matched_models, native_loss, train_arm,
)


def sample():
    generator = torch.Generator().manual_seed(88)
    x = torch.randn(5, 7, 10, generator=generator, dtype=torch.float64)
    y = .3 * torch.randn(5, 7, 12, generator=generator, dtype=torch.float64)
    y += x @ (.1 * torch.randn(10, 12, generator=generator, dtype=torch.float64))
    w = torch.full((5, 7), 4., dtype=torch.float64)
    w[:, -1] = 0
    w[:, -2] = 2
    fit = torch.arange(3)
    state = fit_motion_path(x, y, w, fit, [1.], [8], methods=("rrr",))[0]
    return x, y, w, fit, state


def test_matched_models_have_equal_nonzero_hidden_functions_before_training():
    x, y, w, fit, state = sample()
    scale = fit_control_scale(y, w, fit, state)
    rng = torch.random.get_rng_state().clone()
    models, report = matched_models(state, scale, 46)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert report["parameter_counts"]["temporal"] - report["parameter_counts"]["pointwise"] == 256
    assert len(report["center_matched_zero_side_kernels"]) == 2
    a, b = models["pointwise"].state_dict(), models["temporal"].state_dict()
    for key in report["shared_state_keys"]:
        assert torch.equal(a[key], b[key]), key
    for key in report["center_matched_zero_side_kernels"]:
        assert torch.equal(a[key], b[key][..., 1:2])
        assert b[key][..., [0, 2]].count_nonzero() == 0
    # Capture the input to the zero-output layer: comparing final predictions
    # alone would pass even if the hidden initial functions differed.
    hidden = {}
    hooks = [model.refiner.output.register_forward_pre_hook(
        lambda module, args, name=name: hidden.__setitem__(name, args[0].detach().clone()))
        for name, model in models.items()]
    poisoned = x.clone()
    poisoned[w == 0] = float("nan")
    predictions = {name: model(poisoned, w) for name, model in models.items()}
    for hook in hooks:
        hook.remove()
    assert hidden["pointwise"].abs().sum() > 0
    torch.testing.assert_close(hidden["pointwise"], hidden["temporal"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(predictions["pointwise"], predictions["temporal"], rtol=0, atol=0)


def test_native_loss_original_motion_units_padding_and_gradient():
    prediction = torch.tensor([[[.2, -.1], [.4, .8], [float("nan"), float("nan")]]],
                              dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[.1, .3], [.2, .4], [float("nan"), float("nan")]]], dtype=torch.float64)
    weight = torch.tensor([[4., 2., 0.]], dtype=torch.float64)
    loss = native_loss(prediction, target, weight)
    clean_error = prediction.detach()[:, :2] - target[:, :2]
    expected = (clean_error.square() * weight[:, :2, None]).sum() / (6 * 2 * .25 ** 2)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    expected_grad = 2 * clean_error * weight[:, :2, None] / (6 * 2 * .25 ** 2)
    torch.testing.assert_close(prediction.grad[:, :2], expected_grad, rtol=0, atol=0)
    assert torch.equal(prediction.grad[:, -1], torch.zeros_like(prediction.grad[:, -1]))
    with pytest.raises(ValueError, match="empty training"):
        native_loss(prediction, target, torch.zeros_like(weight))
    target[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite observed"):
        native_loss(prediction, target, weight)


def test_eight_epoch_runner_matches_draws_and_ignores_heldout_poison(tmp_path):
    # Keep this exact-eight-epoch test cheap; it checks plumbing rather than
    # attempting to establish a generalization result on synthetic data.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y, w, fit, state = sample()
        scale = fit_control_scale(y, w, fit, state)
        base, _ = matched_models(state, scale, 46)
        initial = {name: copy.deepcopy(model) for name, model in base.items()}
        finished, histories = {}, {}
        for arm in ("pointwise", "temporal"):
            frozen = buffer_hash(base[arm])
            finished[arm], histories[arm] = train_arm(base[arm], x, y, w, fit, seed=46,
                device="cpu", output=tmp_path / arm, provenance={"fixture": True}, fold=0, arm=arm)
            assert buffer_hash(finished[arm]) == frozen
            assert [row["epoch"] for row in histories[arm]] == list(range(1, 9))
            assert [row["step"] for row in histories[arm]] == list(range(1, 9))
            saved = torch.load(tmp_path / arm / "last.pt", map_location="cpu", weights_only=False)
            assert saved["epoch"] == 8 and saved["step"] == 8
            assert torch.equal(saved["fit_ids"], fit)
            assert saved["frozen_buffer_sha256"] == frozen
            assert saved["checkpoint_selection"] == "fixed epoch8 only, no held-out selection"
            for key, value in finished[arm].state_dict().items():
                assert torch.equal(value, saved["model"][key]), key
        assert [r["batch_order_sha256"] for r in histories["pointwise"]] == [
            r["batch_order_sha256"] for r in histories["temporal"]]
        assert finished["temporal"].refiner.blocks[0].depthwise.weight[..., [0, 2]].abs().sum() > 0
        # Poison every held-out tensor and replay both arms. No evaluation call
        # occurs in train_arm, and fitted model values must remain bitwise equal.
        x[3:], y[3:], w[3:] = float("nan"), float("nan"), float("nan")
        for arm in ("pointwise", "temporal"):
            replay, rows = train_arm(initial[arm], x, y, w, fit, seed=46,
                device="cpu", output=tmp_path / (arm + "_poison"),
                provenance={"fixture": True}, fold=0, arm=arm)
            for key, value in replay.state_dict().items():
                assert torch.equal(value, finished[arm].state_dict()[key]), key
            assert [r["online_training_native_mse"] for r in rows] == [
                r["online_training_native_mse"] for r in histories[arm]]
    finally:
        torch.set_num_threads(prior_threads)
