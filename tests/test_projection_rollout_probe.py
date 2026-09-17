import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_projection_rollout_probe import rollout_error
from scripts.train_predictable_renderer import configure_trainable, projected_affect, state_hash
from scripts.train_formal_predictable_projection import frozen_hash


def test_real_twelve_step_rollout_gradient_updates_only_existing512_projection(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(4)
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
        batch = {"q": {"content": content, "motion": torch.randn(2, 8, 52), "valid": valid},
                 "base": base, "identity": identity, "affect": affect}
        configure_trainable(system, seed=46, projection_only=True)
        assert system.local_projection.weight.numel() == 512
        before = frozen_hash(system)
        monkeypatch.setattr(system, "base", lambda *a, **k: (_ for _ in ()).throw(AssertionError("B0 recomputed")))
        controls = torch.randn(2, 2, 8)
        weight, noise = torch.full((2, 2), 4.), torch.randn(2, 8, 52)
        calls = []
        hook = system.renderer.register_forward_hook(lambda *args: calls.append(1))
        error, prediction = rollout_error(system, batch, controls, weight, noise)
        hook.remove()
        assert len(calls) == 12
        torch.testing.assert_close(error, ((prediction - batch["q"]["motion"]) / .25).square())
        error.mean().backward()
        assert system.local_projection.weight.grad is not None
        assert torch.isfinite(system.local_projection.weight.grad).all()
        assert system.local_projection.weight.grad.abs().sum() > 0
        assert all(p.grad is None for name, p in system.named_parameters() if name != "local_projection.weight")
        optimizer = torch.optim.Adam(system.local_projection.parameters(), lr=.0002)
        optimizer.step()
        assert system.local_projection.weight.count_nonzero() > 0
        assert frozen_hash(system) == before
    finally:
        torch.set_num_threads(previous)
