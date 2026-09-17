import copy

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_audio_conditioned_flow_probe import (
    adaptation_flow_loss, configure_adaptation, frozen_except_adaptation_hash,
)
from scripts.train_predictable_renderer import cached_flow, observed, state_hash
from scripts.train_projection_schedule_ablation import draws


def fixture():
    torch.manual_seed(91)
    system = NeutralAffectSystem({
        "data": {"content_dim": 8, "motion_dim": 52, "neutral_output_indices": [17, 18],
                 "emotion_classes": ["neutral", "angry", "happy", "sad"], "num_intensity_levels": 4,
                 "audio_emotion_dim": 5},
        "model": {"content_dim": 8, "emotion_dim": 64, "style_dim": 8, "hidden_dim": 8,
                  "heads": 2, "dropout": .2, "dit_dim": 8, "dit_depth": 2,
                  "residual_scale": .25, "affect_stride": 4, "affect_rank": 8,
                  "global_condition_dropout": .2, "style_condition_dropout": .2}}).eval()
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[1, -2:] = False
    content = torch.randn(2, 8, 8)
    with torch.no_grad():
        base = system.base(content, valid)
        identity = system.encode_identity(torch.randn(2, 2, 8, 52))
        affect = system.encode_audio(torch.randn(2, 8, 5), valid)
    q = {"content": content, "motion": torch.randn(2, 8, 52), "valid": valid,
         "channel_mask": torch.ones(2, 52, dtype=torch.bool)}
    q["channel_mask"][:, 0] = False
    batch = {"q": q, "base": base, "identity": identity, "affect": affect}
    controls, weight = torch.randn(2, 2, 8), torch.tensor([[4., 4.], [4., 2.]])
    return system, batch, controls, weight


def test_adaptation_leaves_source_exact_and_both_arms_have_gradient_and_frozen_state(monkeypatch):
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        original, batch, controls, weight = fixture()
        original_state = state_hash(original.state_dict())
        projection = state_hash(original.local_projection.state_dict())
        noise, time, choice = draws(torch.Generator().manual_seed(46), 2, (8, 52))
        grads, trained = {}, {}
        for arm in ("audio_local", "zero_local"):
            model = copy.deepcopy(original)
            name, parameter = configure_adaptation(model, expected_dim=8)
            assert name == "renderer.blocks.1.cross_attention.out_proj.weight"
            assert state_hash(model.state_dict()) == original_state
            assert not any(m.training for m in model.modules())
            assert parameter.numel() == 64
            assert [n for n, p in model.named_parameters() if p.requires_grad] == [name]
            frozen = frozen_except_adaptation_hash(model, name)
            monkeypatch.setattr(model, "base", lambda *a, **k: (_ for _ in ()).throw(AssertionError("B0 recomputed")))
            monkeypatch.setattr(model.motion_teacher, "forward", lambda *a, **k: (_ for _ in ()).throw(AssertionError("teacher local read")))
            loss = adaptation_flow_loss(model, batch, controls, weight, noise, time, arm)
            reference = cached_flow(model, batch, controls, weight, noise, time, zero=arm == "zero_local", audio_gate=True)
            exact = (reference["prediction"] - reference["velocity_target"])[observed(batch["q"])].square().mean()
            torch.testing.assert_close(loss, exact, rtol=0, atol=0)
            loss.backward()
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            assert all(p.grad is None for n, p in model.named_parameters() if n != name)
            grads[arm] = parameter.grad.detach().clone()
            before = parameter.detach().clone()
            torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
            torch.optim.Adam([parameter], lr=1e-5).step()
            assert not torch.equal(before, parameter)
            assert frozen_except_adaptation_hash(model, name) == frozen
            assert state_hash(model.local_projection.state_dict()) == projection
            trained[arm] = parameter.detach().clone()
        assert not torch.equal(grads["audio_local"], grads["zero_local"])
        assert not torch.equal(trained["audio_local"], trained["zero_local"])
    finally:
        torch.set_num_threads(prior)


def test_zero_local_arm_does_not_depend_on_audio_controls_and_rng_draws_pair():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        model, batch, controls, weight = fixture()
        configure_adaptation(model)
        a, b = torch.Generator().manual_seed(46), torch.Generator().manual_seed(46)
        na, ta, ca = draws(a, 2, (8, 52)); nb, tb, cb = draws(b, 2, (8, 52))
        assert torch.equal(na, nb) and torch.equal(ta, tb) and torch.equal(ca, cb)
        first = adaptation_flow_loss(model, batch, controls, weight, na, ta, "zero_local")
        second = adaptation_flow_loss(model, batch, controls * 1000, weight, nb, tb, "zero_local")
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        with pytest.raises(ValueError, match="Unknown"):
            adaptation_flow_loss(model, batch, controls, weight, na, ta, "teacher")
        with pytest.raises(ValueError, match="dimension"):
            configure_adaptation(model, expected_dim=192)
    finally:
        torch.set_num_threads(prior)
