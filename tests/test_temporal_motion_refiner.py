import copy

import pytest
import torch

from kinetalk_b0.predictable_motion import fit_motion_path, predict_motion, weighted_clip_center
from kinetalk_b0.temporal_motion_refiner import FrozenRidgeRefiner, fit_control_scale


def sample():
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(6, 11, 10, generator=generator, dtype=torch.float64)
    y = .2 * torch.randn(6, 11, 12, generator=generator, dtype=torch.float64)
    y += x @ (.1 * torch.randn(10, 12, generator=generator, dtype=torch.float64))
    w = torch.ones(6, 11, dtype=torch.float64) * 4
    w[:, -2:] = 0
    w[:, -3] = 2
    ids = torch.arange(4)
    state = fit_motion_path(x, y, w, ids, [1.], [8], methods=("rrr",))[0]
    return x, y, w, ids, state


@pytest.mark.parametrize("arm", ["pointwise", "temporal"])
def test_zero_init_reproduces_frozen_ridge_and_masks_padding_and_empty_clips(arm):
    x, y, w, ids, state = sample()
    scale = fit_control_scale(y, w, ids, state)
    model = FrozenRidgeRefiner(state, scale, arm, hidden=8)
    pred = model(x, w)
    torch.testing.assert_close(pred, predict_motion(x, w, state), rtol=0, atol=0)
    poison = x.clone()
    poison[w == 0] = float("nan")
    torch.testing.assert_close(model(poison, w), pred, rtol=0, atol=0)
    # Train output is no longer zero: biased intermediate padding must still
    # never propagate through temporal convolutions into valid bins.
    with torch.no_grad():
        model.refiner.output.weight.fill_(.07)
        model.refiner.output.bias.fill_(.8)
    torch.testing.assert_close(model(poison, w), model(x, w), rtol=0, atol=0)
    padded = torch.cat([poison, torch.full((6, 4, 10), float("nan"), dtype=x.dtype)], 1)
    wp = torch.cat([w, torch.zeros(6, 4, dtype=w.dtype)], 1)
    torch.testing.assert_close(model(padded, wp)[:, :11], model(poison, w), rtol=1e-6, atol=1e-8)
    empty = model(torch.full_like(x, float("nan")), torch.zeros_like(w))
    assert torch.equal(empty, torch.zeros_like(empty))
    bad = poison.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Observed audio"):
        model(bad, w)


def test_control_scale_fits_real_motion_fit_rows_only_with_no_rng_or_input_mutation():
    x, y, w, ids, state = sample()
    before_y, before_w = y.clone(), w.clone()
    rng = torch.random.get_rng_state().clone()
    scale = fit_control_scale(y, w, ids, state)
    real = weighted_clip_center(y[ids], w[ids]) @ state["basis"]
    expected = ((real.square() * w[ids, :, None]).sum((0, 1)) / w[ids].sum()).sqrt().clamp_min(1e-6)
    torch.testing.assert_close(scale, expected, rtol=0, atol=0)
    assert torch.equal(before_y, y) and torch.equal(before_w, w)
    y[4:], w[4:] = float("nan"), float("nan")
    torch.testing.assert_close(fit_control_scale(y, w, ids, state), scale, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    zero = fit_control_scale(torch.zeros_like(y), before_w, ids, state)
    torch.testing.assert_close(zero, torch.full_like(zero, 1e-6), rtol=0, atol=0)
    full = torch.full((6, 11, 52), float("nan"), dtype=before_y.dtype)
    full[..., 20:32] = before_y
    mapped = {**state, "motion_channel_indices": list(range(20, 32))}
    torch.testing.assert_close(fit_control_scale(full, before_w, ids, mapped), scale, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unique"):
        fit_control_scale(before_y, before_w, [0, 0], state)


def test_same_seed_shared_layers_match_and_only_temporal_kernels_add_parameters():
    _, y, w, ids, state = sample()
    scale = fit_control_scale(y, w, ids, state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        pointwise = FrozenRidgeRefiner(state, scale, "pointwise")
        torch.manual_seed(123)
        temporal = FrozenRidgeRefiner(state, scale, "temporal")
    for name, value in pointwise.state_dict().items():
        other = temporal.state_dict()[name]
        if value.shape == other.shape:
            assert torch.equal(value, other), name
    assert (sum(p.numel() for p in temporal.parameters())
            - sum(p.numel() for p in pointwise.parameters())) == 256
    assert pointwise.refiner.output.weight.count_nonzero() == 0
    assert pointwise.refiner.output.bias.count_nonzero() == 0
    assert pointwise.refiner.input.weight.count_nonzero() > 0
    assert all(not b.requires_grad for b in temporal.buffers())


@pytest.mark.parametrize("arm", ["pointwise", "temporal"])
def test_true_native_motion_loss_trains_final_then_upstream_but_keeps_buffers(arm):
    x, y, w, ids, state = sample()
    model = FrozenRidgeRefiner(state, fit_control_scale(y, w, ids, state), arm, hidden=8)
    before = {k: b.clone() for k, b in model.named_buffers()}
    initial_input = model.refiner.input.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.2)
    target = weighted_clip_center(y, w)
    loss_history = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, w)
        loss = ((prediction - target).square() * w[..., None]).sum() / (w.sum() * y.shape[-1])
        loss_history.append(float(loss.detach()))
        loss.backward()
        assert model.refiner.output.weight.grad.abs().sum() > 0
        if step == 0:
            assert model.refiner.input.weight.grad.count_nonzero() == 0
        else:
            assert model.refiner.input.weight.grad.abs().sum() > 0
            assert model.refiner.blocks[0].depthwise.weight.grad.abs().sum() > 0
        optimizer.step()
    assert loss_history[1] < loss_history[0]
    assert not torch.equal(initial_input, model.refiner.input.weight)
    for name, value in model.named_buffers():
        assert torch.equal(value, before[name]), name
        assert value.grad is None
    # Prediction takes only x/weight. Altering supervision changes the loss,
    # never the fixed input-only forward result.
    prediction = model(x, w).detach()
    alternate = weighted_clip_center(y + x[..., :1].square(), w)
    assert not torch.allclose((prediction - target).square().sum(),
                              (prediction - alternate).square().sum())
    torch.testing.assert_close(model(x, w), prediction, rtol=0, atol=0)


def test_temporal_mixes_neighbours_pointwise_does_not_despite_clip_centering():
    # Equal/opposite changes preserve the input clip mean. Differences between
    # two untouched output bins cancel the final common centering shift.
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(1, 15, 10, generator=generator, dtype=torch.float64)
    w = torch.ones(1, 15, dtype=torch.float64)
    _, y, fit_weight, ids, state = sample()
    scale = fit_control_scale(y, fit_weight, ids, state)
    changed = x.clone()
    delta = torch.linspace(-1., 2., 10)
    changed[:, 5] += delta
    changed[:, 10] -= delta
    differences = {}
    for arm in ("pointwise", "temporal"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            model = FrozenRidgeRefiner(state, scale, arm, hidden=8).double()
        with torch.no_grad():
            model.refiner.output.weight.copy_(torch.randn(8, 8, generator=generator, dtype=torch.float64))
        original, altered = model.controls(x, w), model.controls(changed, w)
        differences[arm] = ((altered[:, 4] - altered[:, 0])
                            - (original[:, 4] - original[:, 0])).abs().max()
    assert differences["pointwise"] < 1e-12
    assert differences["temporal"] > 1e-5


def test_rejects_scaled_or_nonorthogonal_basis_and_invalid_scale():
    _, y, w, ids, state = sample()
    scale = fit_control_scale(y, w, ids, state)
    bad = copy.deepcopy(state)
    bad["basis"][0] *= 2
    with pytest.raises(ValueError, match="orthonormal"):
        FrozenRidgeRefiner(bad, scale, "temporal")
    with pytest.raises(ValueError, match="Scaled/transformed"):
        FrozenRidgeRefiner({**state, "coordinate_system": "scaled"}, scale, "temporal")
    with pytest.raises(ValueError, match="positive finite"):
        FrozenRidgeRefiner(state, torch.zeros(8), "temporal")
