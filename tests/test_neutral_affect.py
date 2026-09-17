"""Contract checks; real-data training must establish learned affect usefulness."""

import pytest
import torch
from torch.nn import functional as F

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.train_neutral_affect_pilot import affect_residual


def test_identity_subtraction_keeps_unmeasured_channels_missing():
    batch = {"residual": torch.ones(1, 3, 2),
             "valid": torch.tensor([[True, True, False]]),
             "channel_mask": torch.tensor([[True, False]])}
    identity = {"baseline": torch.tensor([[.2, .5]])}
    result = affect_residual(batch, identity)
    torch.testing.assert_close(result[0, :2, 0], torch.full((2,), .8))
    assert result[0, :, 1].count_nonzero() == 0
    assert result[0, 2].count_nonzero() == 0


@pytest.fixture
def cfg():
    return {
        "data": {"content_dim": 8, "motion_dim": 6, "neutral_output_indices": [2, 3],
                 "emotion_classes": ["neutral", "happy", "sad"], "num_intensity_levels": 3,
                 "audio_emotion_dim": 5},
        "model": {"content_dim": 8, "emotion_dim": 8, "style_dim": 8, "hidden_dim": 8,
                  "heads": 2, "dropout": 0., "dit_dim": 8, "dit_depth": 1,
                  "residual_scale": .25, "affect_stride": 4, "affect_rank": 3,
                  "global_condition_dropout": 0., "style_condition_dropout": 0.}}


@pytest.fixture(autouse=True)
def seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    yield
    torch.set_num_threads(previous)


def test_affect_padding_does_not_change_timing_amplitude_or_global(cfg):
    system = NeutralAffectSystem(cfg).eval()
    sequence = torch.randn(1, 11, 6)
    mask = torch.ones(1, 11, dtype=torch.bool)
    expected = system.encode_motion(sequence, mask)
    padded = F.pad(sequence, (0, 0, 0, 14), value=float("nan"))
    padded_mask = F.pad(mask, (0, 14), value=False)
    actual = system.encode_motion(padded, padded_mask)
    for key in ("global", "emotion_logits", "intensity_logits"):
        torch.testing.assert_close(expected[key], actual[key], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(expected["controls"], actual["controls"][:, :3], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(expected["local"], actual["local"][:, :11], atol=2e-6, rtol=2e-6)
    assert actual["local"][:, 11:].count_nonzero() == 0
    assert actual["controls"][:, 3:].count_nonzero() == 0
    torch.testing.assert_close((expected["controls"] * expected["control_weight"].unsqueeze(-1)).sum(1),
                               torch.zeros(1, 3), atol=2e-6, rtol=0)


def test_reference_pooling_is_permutation_invariant_and_ignores_missing(cfg):
    system = NeutralAffectSystem(cfg)
    refs = torch.randn(2, 3, 9, 6, requires_grad=True)
    mask = torch.ones(2, 3, 9, dtype=torch.bool)
    mask[:, 2] = False
    with torch.no_grad():
        refs[:, 2] = float("nan")
    original = system.encode_identity(refs, mask)
    perm = torch.tensor([2, 0, 1])
    shuffled = system.encode_identity(refs[:, perm], mask[:, perm])
    torch.testing.assert_close(original["code"], shuffled["code"])
    assert original["baseline"].count_nonzero() == 0
    original["code"].square().sum().backward()
    assert torch.isfinite(refs.grad).all()
    assert refs.grad[:, :2].abs().sum() > 0
    assert refs.grad[:, 2].count_nonzero() == 0
    with pytest.raises(ValueError, match="at least one valid"):
        system.encode_identity(refs.detach(), torch.zeros_like(mask))


def test_flow_endpoint_reconstruction_teaches_dynamic_controls_and_freezes_b0(cfg):
    system = NeutralAffectSystem(cfg).train()
    mask = torch.ones(2, 12, dtype=torch.bool)
    motion = torch.randn(2, 12, 6) * .1
    content = torch.randn(2, 12, 8)
    identity = system.encode_identity(torch.randn(2, 2, 12, 6) * .03)
    base = system.base(content, mask)
    affect = system.encode_motion(motion - base["b0"] - identity["baseline"][:, None], mask)
    affect["controls"].retain_grad()
    output = system.flow(motion, content, mask, identity, affect, noise=torch.zeros_like(motion), time=torch.zeros(2))
    loss = F.mse_loss(output["prediction"], output["velocity_target"])
    loss.backward()
    assert affect["controls"].grad.abs().sum() > 0
    assert system.motion_teacher.control_head.weight.grad.abs().sum() > 0
    assert system.local_projection.weight.grad.abs().sum() > 0
    assert system.identity_bias.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in system.stage1.parameters())
    assert not system.stage1.training
    torch.testing.assert_close(output["motion"], base["b0"] + identity["baseline"][:, None] + output["predicted_residual"])
    torch.testing.assert_close(output["target_residual"], motion - base["b0"] - identity["baseline"][:, None])


def test_audio_student_receives_shared_temporal_coordinates_from_frozen_teacher(cfg):
    system = NeutralAffectSystem(cfg).train()
    freeze_module(system.motion_teacher)
    freeze_module(system.local_projection)
    mask = torch.ones(2, 16, dtype=torch.bool)
    with torch.no_grad():
        teacher = system.encode_motion(torch.randn(2, 16, 6), mask)
    audio = system.encode_audio(torch.randn(2, 16, 5), mask)
    loss = F.mse_loss(audio["controls"], teacher["controls"]) + F.mse_loss(audio["global"], teacher["global"])
    loss.backward()
    assert system.audio_encoder.control_head.weight.grad.abs().sum() > 0
    assert system.audio_encoder.global_head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in system.motion_teacher.parameters())
    torch.testing.assert_close(system.project_affect(teacher, mask)["local"], teacher["local"])


def test_fixed_noise_dynamic_intervention_changes_motion_not_b0_or_identity(cfg):
    system = NeutralAffectSystem(cfg).eval()
    content = torch.randn(1, 13, 8)
    mask = torch.ones(1, 13, dtype=torch.bool)
    identity = system.encode_identity(torch.randn(1, 3, 13, 6) * .02)
    affect = system.encode_audio(torch.randn(1, 13, 5), mask)
    noise = torch.randn(1, 13, 6)
    with torch.no_grad():
        first = system.generate(content, mask, identity, affect, noise, steps=2)
        repeated = system.generate(content, mask, identity, affect, noise, steps=2)
        zero_local = {**affect, "local": torch.zeros_like(affect["local"])}
        intervened = system.generate(content, mask, identity, zero_local, noise, steps=2)
    torch.testing.assert_close(first["motion"], repeated["motion"], rtol=0, atol=0)
    assert (first["motion"] - intervened["motion"]).abs().max() > 1e-6
    torch.testing.assert_close(first["b0"], intervened["b0"], rtol=0, atol=0)


def test_complete_generation_ignores_appended_padding(cfg):
    system = NeutralAffectSystem(cfg).eval()
    content = torch.randn(1, 11, 8)
    audio = torch.randn(1, 11, 5)
    mask = torch.ones(1, 11, dtype=torch.bool)
    refs = torch.randn(1, 2, 11, 6)
    identity = system.encode_identity(refs)
    noise = torch.randn(1, 11, 6)
    with torch.no_grad():
        expected = system.generate(content, mask, identity, system.encode_audio(audio, mask), noise, 2)
        longer_mask = F.pad(mask, (0, 9), value=False)
        actual = system.generate(F.pad(content, (0, 0, 0, 9), value=float("nan")), longer_mask, identity,
                                 system.encode_audio(F.pad(audio, (0, 0, 0, 9), value=float("nan")), longer_mask),
                                 F.pad(noise, (0, 0, 0, 9)), 2)
    for key in ("b0", "motion", "residual"):
        torch.testing.assert_close(expected[key], actual[key][:, :11], atol=2e-5, rtol=2e-5)
        assert actual[key][:, 11:].count_nonzero() == 0


def test_cached_frozen_base_matches_flow_and_generation(cfg):
    system = NeutralAffectSystem(cfg).eval()
    content = torch.randn(2, 12, 8)
    motion = torch.randn(2, 12, 6)
    mask = torch.ones(2, 12, dtype=torch.bool)
    identity = system.encode_identity(torch.randn(2, 2, 12, 6))
    affect = system.encode_audio(torch.randn(2, 12, 5), mask)
    noise = torch.randn_like(motion)
    time = torch.tensor([0., .5])
    base = system.base(content, mask)
    for method, args, kwargs in ((system.flow, (motion, content, mask, identity, affect), {"noise": noise, "time": time}),
                                 (system.generate, (content, mask, identity, affect), {"initial_noise": noise, "steps": 2})):
        expected = method(*args, **kwargs)
        actual = method(*args, **kwargs, base=base)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
