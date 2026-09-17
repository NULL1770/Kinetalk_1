import copy

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_formal_predictable_projection import frozen_hash
from scripts.train_predictable_renderer import configure_trainable, observed, state_hash
from scripts.train_projection_centered_rollout_probe import centered_rollout_error
from scripts.train_projection_scaled_centered_probe import (
    apply_channel_weights, channel_metric_report, fit_training_channel_metric,
    scaled_centered_rollout_error,
)


def training_cache():
    # Channels0/1/2 have RMS1/2/0, channel3 is unobserved. Padding is NaN.
    valid = torch.tensor([[True, True, False], [True, True, True]])
    residual = torch.tensor([[[-1., -2., 0., float("nan")], [1., 2., 0., float("nan")],
                             [float("nan")] * 4],
                            [[-1., -2., 0., float("nan")], [1., 2., 0., float("nan")], [0., 0., 0., float("nan")]]])
    base = torch.arange(3).float()[None, :, None].expand(2, 3, 4) * 3
    identity = torch.ones(2, 4) * .75
    target = residual + base + identity[:, None]
    train = {"q": {"motion": target, "valid": valid,
        "channel_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)},
        "base": {"b0": base}, "identity": {"baseline": identity}}
    class TrainOnly(dict):
        def __getitem__(self, key):
            if key != "train":
                raise AssertionError("Validation/test was accessed")
            return super().__getitem__(key)
    return {"splits": TrainOnly(train=train)}


def test_metric_fits_train_only_observed_centered_residual_with_fixed_median_floor():
    cache = training_cache()
    rng = torch.random.get_rng_state().clone()
    before = cache["splits"]["train"]["q"]["motion"].clone()
    state = fit_training_channel_metric(cache)
    scale = (4 / 5) ** .5
    torch.testing.assert_close(state["rms"], torch.tensor([scale, 2 * scale, 0., 0.], dtype=torch.float64))
    assert state["floor"] == pytest.approx(1.5 * scale)
    assert state["train_rms_weights"][3] == 0
    assert state["train_rms_weights"][0] == state["train_rms_weights"][2]
    assert state["train_rms_weights"][:3].mean() == pytest.approx(1.)
    torch.testing.assert_close(state["uniform_weights"], torch.ones(4))
    assert state["observed_counts"].tolist() == [5., 5., 5., 0.]
    torch.testing.assert_close(cache["splits"]["train"]["q"]["motion"], before, equal_nan=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    report = channel_metric_report(state)
    assert report["state_sha256"] == state_hash(state)
    assert report["validation_values_used"] is False
    changed = training_cache()
    changed["splits"]["train"]["q"]["motion"] += torch.tensor([[[100., -13., 27., 0.]]])
    assert state_hash(fit_training_channel_metric(changed)) == state_hash(state)


def test_metric_rejects_observed_nans_or_all_zero_variance():
    cache = training_cache()
    cache["splits"]["train"]["q"]["motion"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_training_channel_metric(cache)
    cache = training_cache()
    train = cache["splits"]["train"]
    train["q"]["motion"] = train["base"]["b0"] + train["identity"]["baseline"][:, None]
    with pytest.raises(ValueError, match="No positive"):
        fit_training_channel_metric(cache)


def system_and_batch():
    torch.manual_seed(7)
    system = NeutralAffectSystem({
        "data": {"content_dim": 8, "motion_dim": 52, "neutral_output_indices": [17, 18],
                 "emotion_classes": ["neutral", "angry", "happy", "sad"], "num_intensity_levels": 4,
                 "audio_emotion_dim": 5},
        "model": {"content_dim": 8, "emotion_dim": 64, "style_dim": 8, "hidden_dim": 8,
                  "heads": 2, "dropout": 0., "dit_dim": 8, "dit_depth": 1,
                  "residual_scale": .25, "affect_stride": 4, "affect_rank": 8,
                  "global_condition_dropout": 0., "style_condition_dropout": 0.}}).eval()
    valid = torch.ones(2, 8, dtype=torch.bool)
    content = torch.randn(2, 8, 8)
    with torch.no_grad():
        base = system.base(content, valid)
        identity = system.encode_identity(torch.randn(2, 2, 8, 52))
        affect = system.encode_audio(torch.randn(2, 8, 5), valid)
    q = {"content": content, "motion": torch.randn(2, 8, 52), "valid": valid,
         "channel_mask": torch.ones(2, 52, dtype=torch.bool)}
    configure_trainable(system, seed=46, projection_only=True)
    return system, {"q": q, "base": base, "identity": identity, "affect": affect}


def test_uniform_equals_old_loss_output_gradient_and_optimizer_update():
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        old, batch = system_and_batch()
        new = copy.deepcopy(old)
        controls, weight, noise = torch.randn(2, 2, 8), torch.full((2, 2), 4.), torch.randn(2, 8, 52)
        before = frozen_hash(old)
        old_error, old_prediction = centered_rollout_error(old, batch, controls, weight, noise)
        new_error, new_prediction = scaled_centered_rollout_error(new, batch, controls, weight, noise, torch.ones(52))
        assert torch.equal(old_error, new_error)
        assert torch.equal(old_prediction, new_prediction)
        old_error[observed(batch["q"])].mean().backward()
        new_error[observed(batch["q"])].mean().backward()
        assert torch.equal(old.local_projection.weight.grad, new.local_projection.weight.grad)
        for system in (old, new):
            torch.optim.Adam(system.local_projection.parameters(), lr=.0002).step()
        assert torch.equal(old.local_projection.weight, new.local_projection.weight)
        assert frozen_hash(old) == frozen_hash(new) == before
    finally:
        torch.set_num_threads(old_threads)


def test_scaled_real_twelve_step_rollout_changes_only512_parameters_and_keeps_raw_output(monkeypatch):
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        system, batch = system_and_batch()
        weights = fit_training_channel_metric({"splits": {"train": batch}})["train_rms_weights"]
        before = frozen_hash(system)
        original_generate = system.generate
        captured, calls = [], []
        def generate(*args, **kwargs):
            value = original_generate(*args, **kwargs)
            captured.append(value["motion"])
            return value
        monkeypatch.setattr(system, "generate", generate)
        monkeypatch.setattr(system, "base", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Recomputed B0")))
        hook = system.renderer.register_forward_hook(lambda *args: calls.append(1))
        error, prediction = scaled_centered_rollout_error(system, batch, torch.randn(2, 2, 8),
            torch.full((2, 2), 4.), torch.randn(2, 8, 52), weights)
        hook.remove()
        assert len(calls) == 12
        assert prediction is captured[0]
        prediction.retain_grad()
        error[observed(batch["q"])].mean().backward()
        torch.testing.assert_close(prediction.grad.sum(1), torch.zeros(2, 52), atol=3e-7, rtol=0)
        assert system.local_projection.weight.numel() == 512
        assert system.local_projection.weight.grad.abs().sum() > 0
        assert torch.isfinite(system.local_projection.weight.grad).all()
        assert all(p.grad is None for name, p in system.named_parameters() if name != "local_projection.weight")
        torch.optim.Adam(system.local_projection.parameters(), lr=.0002).step()
        assert system.local_projection.weight.count_nonzero() > 0
        assert frozen_hash(system) == before
        with pytest.raises(ValueError, match="nonnegative"):
            apply_channel_weights(error, -torch.ones(52))
    finally:
        torch.set_num_threads(old_threads)
