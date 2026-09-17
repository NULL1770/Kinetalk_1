import copy

import pytest
import torch

from scripts.train_direct_audio_dynamics import DirectAudioEncoder, feature_statistics
from scripts.train_output_motion_dynamics import (
    GROUPS, fit_output_scales, generated_output_losses, motion_statistics, training_loss,
)
from scripts.train_predictable_renderer import state_hash
from scripts.train_projection_schedule_ablation import draws
from tests.test_audio_conditioned_flow_probe import fixture


def output_batch():
    target = torch.zeros(2, 5, 52)
    target[:] = torch.tensor([0., .02, .04, .06, .08])[None, :, None]
    valid = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    cm = torch.ones(2, 52, dtype=torch.bool)
    cm[:, 0] = False
    return {"q": {"motion": target, "valid": valid, "channel_mask": cm}}


def test_fit_scales_use_observed_native_motion_and_ignore_padding_poison():
    batch = output_batch()
    expected = fit_output_scales(batch)
    torch.testing.assert_close(expected["displacement"][1:], torch.full((51,), .02))
    poisoned = copy.deepcopy(batch)
    mask = poisoned["q"]["valid"][..., None] & poisoned["q"]["channel_mask"][:, None]
    poisoned["q"]["motion"][~mask] = float("nan")
    actual = fit_output_scales(poisoned)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert expected["displacement"][0] == pytest.approx(.005)
    assert expected["std"][0] == pytest.approx(.02)
    prediction = poisoned["q"]["motion"].clone()
    prediction[mask] += .4
    losses = generated_output_losses(prediction, poisoned, expected)
    assert all(torch.isfinite(value) and value.abs() < 1e-10 for value in losses.values())


def test_constant_std_has_exact_value_and_finite_gradient():
    values = torch.full((2, 5, 52), .3, requires_grad=True)
    mask = torch.ones_like(values, dtype=torch.bool)
    _, std, _ = motion_statistics(values, mask)
    assert torch.equal(std, torch.zeros_like(std))
    (std - .2).square().mean().backward()
    assert torch.isfinite(values.grad).all()
    batch = output_batch()
    prediction = torch.zeros_like(batch["q"]["motion"], requires_grad=True)
    losses = generated_output_losses(prediction, batch, fit_output_scales(batch))
    sum(losses.values()).backward()
    assert torch.isfinite(prediction.grad).all()


def test_displacement_uses_adjacent_pairs_and_region_means_without_fps():
    batch = output_batch()
    pred = batch["q"]["motion"].clone()
    pred[..., GROUPS["brows"]] += torch.arange(5.)[None, :, None] * .005
    scales = {"displacement": torch.full((52,), .005), "std": torch.ones(52)}
    loss = generated_output_losses(pred, batch, scales)["displacement"]
    # Only the brow group's normalized displacement error is one; groups
    # receive equal weights despite five brow and 27 mouth channels.
    assert loss == pytest.approx(1 / 3, abs=2e-7)
    altered = copy.deepcopy(batch)
    altered["q"]["times"] = torch.arange(5.)[None].expand(2, -1) / 200
    torch.testing.assert_close(generated_output_losses(pred, altered, scales)["displacement"], loss, rtol=0, atol=0)


def test_audio_zero_start_equal_then_only_audio_encoder_gets_gradient_with_matched_draws():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        source, batch, _, weight = fixture()
        for p in source.parameters(): p.requires_grad_(False)
        source.renderer.requires_grad_(True); source.eval()
        features = torch.randn(2, 2, 13)
        original_encoder = DirectAudioEncoder(*feature_statistics(features, weight), output_dim=64, hidden=8).eval()
        scales = fit_output_scales(batch)
        results, model_states, noise_states = {}, {}, []
        for arm in ("audio", "zero"):
            model, encoder = copy.deepcopy(source), copy.deepcopy(original_encoder)
            generator = torch.Generator().manual_seed(46)
            noise, time, choice = draws(generator, 2, (8, 52))
            noise_states.append((noise, time, choice, generator.get_state()))
            total, losses = training_loss(model, encoder, batch, features, weight, scales, noise, time, arm)
            assert set(losses) == {"flow", "displacement", "std"}
            total.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if arm == "audio":
                assert encoder.output.weight.grad.abs().sum() > 0
            else:
                assert all(p.grad is None for p in encoder.parameters())
            results[arm] = {key: value.detach() for key, value in losses.items()}
            opt = torch.optim.Adam([*model.renderer.parameters(), *encoder.parameters()], lr=1e-5)
            opt.step()
            model_states[arm] = state_hash(encoder.state_dict())
        for key in results["audio"]:
            torch.testing.assert_close(results["audio"][key], results["zero"][key], rtol=0, atol=0)
        assert all(torch.equal(a, b) for a, b in zip(*noise_states))
        assert model_states["zero"] == state_hash(original_encoder.state_dict())
        assert model_states["audio"] != model_states["zero"]
    finally:
        torch.set_num_threads(prior)


def test_zero_condition_is_independent_of_nonzero_encoder_and_feature_order():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        system, batch, _, weight = fixture()
        features = torch.randn(2, 2, 13)
        encoder = DirectAudioEncoder(*feature_statistics(features, weight), output_dim=64, hidden=8).eval()
        with torch.no_grad(): encoder.output.weight.normal_()
        noise, time, _ = draws(torch.Generator().manual_seed(46), 2, (8, 52))
        scales = fit_output_scales(batch)
        a, _ = training_loss(system, encoder, batch, features, weight, scales, noise, time, "zero")
        b, _ = training_loss(system, encoder, batch, features.flip(1) * 30, weight, scales, noise, time, "zero")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.set_num_threads(prior)
