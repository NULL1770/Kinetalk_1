import pytest
import torch

from kinetalk_b0.models.semantic import (
    MotionSemanticReadout,
    MultiReferenceStyleEncoder,
    SemanticAudioEncoder,
    SemanticConditionProjector,
    SemanticGenerator,
)
from kinetalk_b0.models.encoders import LowRateAffectField
from kinetalk_b0.utils import freeze_module


@pytest.fixture
def cfg():
    return {
        "data": {
            "content_dim": 8, "motion_dim": 6, "neutral_output_indices": [2, 3],
            "emotion_classes": ["neutral", "happy", "sad"], "num_intensity_levels": 3,
            "audio_emotion_dim": 5,
        },
        "model": {
            "content_dim": 8, "emotion_dim": 8, "style_dim": 8,
            "hidden_dim": 8, "heads": 2, "dropout": 0.0,
            "dit_dim": 8, "dit_depth": 1, "residual_scale": 0.25,
            "intensity_condition_dim": 1,
            "global_condition_dropout": 0.0, "style_condition_dropout": 0.0,
        },
    }


@pytest.fixture(autouse=True)
def seed_and_cpu_threads():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    yield
    torch.set_num_threads(previous_threads)


def semantics():
    return {
        "emotion_id": torch.tensor([1, 2]),
        "intensity_id": torch.tensor([2, -1]),
        "va": torch.rand(2, 6, 2) * 2 - 1,
        "mask": torch.tensor([[True] * 6, [True] * 4 + [False] * 2]),
    }


def test_semantic_projector_is_deterministic_and_ignores_invalid_va(cfg):
    projector = SemanticConditionProjector(cfg).train()
    values = semantics()
    valid = values["mask"].clone()
    valid[:, 2] = False
    original = projector(**values, va_valid=valid)
    corrupted = values["va"].clone()
    corrupted[~valid] = float("nan")
    repeated = projector(**{**values, "va": corrupted}, va_valid=valid)
    for key in ("global", "local", "intensity_value"):
        torch.testing.assert_close(original[key], repeated[key], rtol=0, atol=0)
    assert original["local"][~values["mask"]].count_nonzero() == 0
    assert not any(key in original for key in ("hidden", "style", "residual"))


def test_low_rate_affect_field_is_temporal_and_mask_safe():
    field = LowRateAffectField(4, 6, stride=4).eval()
    hidden = torch.randn(2, 19, 4)
    mask = torch.ones(2, 19, dtype=torch.bool)
    mask[1, -5:] = False
    output = field(hidden, mask)
    assert output.shape == (2, 19, 6)
    assert torch.isfinite(output).all()
    assert output[1, -5:].abs().max() == 0
    # Interpolation from a coarse control grid yields a continuous field;
    # neighbouring changes stay bounded instead of framewise white noise.
    assert output[:, 1:].sub(output[:, :-1]).abs().mean() < 3.0


def test_missing_intensity_has_own_condition_and_is_not_level_zero(cfg):
    projector = SemanticConditionProjector(cfg).eval()
    values = semantics()
    values["intensity_id"] = torch.tensor([0, 0])
    known = projector(**values)
    unknown = projector(**values, intensity_valid=torch.tensor([False, False]))
    explicit = projector(**{**values, "intensity_id": torch.tensor([-1, -1])})
    assert unknown["intensity_value"].eq(-1).all()
    assert not unknown["intensity_valid"].any()
    assert known["intensity_value"].eq(0).all()
    assert not torch.allclose(known["global"], unknown["global"])
    torch.testing.assert_close(unknown["global"], explicit["global"])


def test_semantic_readouts_expose_no_free_latent_and_frozen_critic_keeps_input_gradient(cfg):
    critic = MotionSemanticReadout(cfg)
    freeze_module(critic)
    residual = torch.randn(2, 6, 6, requires_grad=True)
    output = critic(residual, semantics()["mask"])
    assert set(output) == {"emotion_logits", "intensity_logits", "va"}
    loss = output["emotion_logits"].square().mean() + output["va"].square().mean()
    loss.backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all() and residual.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in critic.parameters())
    audio_output = SemanticAudioEncoder(cfg)(torch.randn(2, 6, 5), semantics()["mask"])
    assert set(audio_output) == set(output)
    assert audio_output["va"].abs().max() <= 1


def test_reference_pooling_skips_invalid_references_and_keeps_gradients(cfg):
    encoder = MultiReferenceStyleEncoder(cfg).eval()
    residual = torch.randn(2, 3, 6, 6, requires_grad=True)
    mask = torch.ones(2, 3, 6, dtype=torch.bool)
    mask[:, 2] = False
    with torch.no_grad():
        residual[:, 2] = float("nan")
    output = encoder(residual, mask, return_per_reference=True)
    individual = encoder(residual[:, :2].reshape(4, 6, 6)).reshape(2, 2, -1)
    expected = torch.nn.functional.normalize(individual.mean(1), dim=-1)
    torch.testing.assert_close(output["style"], expected)
    assert output["per_reference"][:, 2].count_nonzero() == 0
    output["style"][..., 0].sum().backward()
    assert torch.isfinite(residual.grad).all()
    assert residual.grad[:, :2].abs().sum() > 0
    assert residual.grad[:, 2].count_nonzero() == 0
    with pytest.raises(ValueError, match="at least one valid"):
        encoder(residual.detach(), torch.zeros_like(mask))


def test_generator_frozen_b0_and_residual_coordinate_are_consistent(cfg):
    model = SemanticGenerator(cfg).train()
    assert model.renderer.training and not model.stage1.training
    assert all(not parameter.requires_grad for parameter in model.stage1.parameters())
    values = semantics()
    content = torch.randn(2, 6, 8)
    target = torch.rand(2, 6, 6)
    output = model.forward_flow(target, content, style=torch.randn(2, 8), **values)
    expected = (target - output["b0"]) * values["mask"].unsqueeze(-1)
    torch.testing.assert_close(output["target_residual"], expected)
    loss = ((output["prediction"] - output["velocity_target"]) ** 2)[values["mask"]].mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.stage1.parameters())
    assert model.renderer.style.weight.grad.abs().sum() > 0


def test_fixed_conditions_and_noise_allow_only_style_to_change_raw_motion(cfg):
    model = SemanticGenerator(cfg).eval()
    values = semantics()
    content = torch.randn(2, 6, 8)
    noise = torch.randn(2, 6, 6)
    style = torch.randn(2, 8, requires_grad=True)
    first = model.generate(content, style=style, initial_noise=noise, steps=2, **values)
    repeat = model.generate(content, style=style, initial_noise=noise, steps=2, **values)
    swapped = model.generate(content, style=-style, initial_noise=noise, steps=2, **values)
    torch.testing.assert_close(first["motion"], repeat["motion"], rtol=0, atol=0)
    torch.testing.assert_close(first["b0"], swapped["b0"], rtol=0, atol=0)
    torch.testing.assert_close(first["conditions"]["local"], swapped["conditions"]["local"], rtol=0, atol=0)
    assert (first["motion"] - swapped["motion"])[values["mask"]].abs().max() > 1e-5
    first["motion"][..., :2].square().mean().backward()
    assert style.grad is not None and torch.isfinite(style.grad).all()
    assert style.grad.abs().sum() > 0
