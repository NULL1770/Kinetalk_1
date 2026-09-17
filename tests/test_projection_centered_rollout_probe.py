import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_projection_centered_rollout_probe import center_observed_error, centered_rollout_error
from scripts.train_predictable_renderer import configure_trainable, observed
from scripts.train_formal_predictable_projection import frozen_hash


def test_centered_error_ignores_missing_values_and_is_shift_invariant_with_zero_mean_gradient():
    error = torch.tensor([[[1., 2., float("nan")], [3., float("nan"), float("nan")],
                           [7., 8., float("nan")], [float("nan"), float("nan"), float("nan")]]], requires_grad=True)
    mask = torch.tensor([[[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]]], dtype=torch.bool)
    centered = center_observed_error(error, mask)
    shifted = center_observed_error(error + torch.tensor([[[30., -14., 25.]]]), mask)
    torch.testing.assert_close(centered, shifted, atol=1e-6, rtol=1e-6)
    assert centered[~mask].count_nonzero() == 0
    torch.testing.assert_close(centered.sum(1), torch.zeros(1, 3), atol=1e-6, rtol=0)
    centered[mask].square().mean().backward()
    assert torch.isfinite(error.grad).all()
    assert error.grad[~mask].count_nonzero() == 0
    torch.testing.assert_close(error.grad.sum(1), torch.zeros(1, 3), atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="Nonfinite"):
        center_observed_error(error, torch.ones_like(mask))


def test_real_twelve_step_centered_rollout_preserves_raw_output_and_only512_projection_gradient(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
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
        batch = {"q": q, "base": base, "identity": identity, "affect": affect}
        configure_trainable(system, seed=46, projection_only=True)
        before = frozen_hash(system)
        monkeypatch.setattr(system, "base", lambda *a, **k: (_ for _ in ()).throw(AssertionError("B0 recomputed")))
        controls, weight, noise = torch.randn(2, 2, 8), torch.full((2, 2), 4.), torch.randn(2, 8, 52)
        captured, calls = [], []
        original_generate = system.generate
        def generate(*a, **kw):
            out = original_generate(*a, **kw)
            captured.append(out["motion"])
            return out
        monkeypatch.setattr(system, "generate", generate)
        hook = system.renderer.register_forward_hook(lambda *a: calls.append(1))
        error, prediction = centered_rollout_error(system, batch, controls, weight, noise)
        hook.remove()
        assert len(calls) == 12
        assert prediction is captured[0]
        prediction.retain_grad()
        torch.testing.assert_close(error, (center_observed_error(prediction - q["motion"], observed(q)) / .25).square())
        error[observed(q)].mean().backward()
        torch.testing.assert_close(prediction.grad.sum(1), torch.zeros(2, 52), atol=2e-7, rtol=0)
        assert system.local_projection.weight.numel() == 512
        assert system.local_projection.weight.grad.abs().sum() > 0
        assert torch.isfinite(system.local_projection.weight.grad).all()
        assert all(p.grad is None for name, p in system.named_parameters() if name != "local_projection.weight")
        torch.optim.Adam(system.local_projection.parameters(), lr=.0002).step()
        assert system.local_projection.weight.count_nonzero() > 0
        assert frozen_hash(system) == before
    finally:
        torch.set_num_threads(previous)
