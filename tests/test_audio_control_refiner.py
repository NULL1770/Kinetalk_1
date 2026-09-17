"""Optional local capacity preserves the original global and padding contracts."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from kinetalk_b0.models.neutral_affect import LowRateAffectEncoder, NeutralAffectSystem


@pytest.fixture
def cfg():
    return {
        "data": {"content_dim": 8, "motion_dim": 6, "neutral_output_indices": [2, 3],
                 "emotion_classes": ["neutral", "happy", "sad"], "num_intensity_levels": 3,
                 "audio_emotion_dim": 5},
        "model": {"content_dim": 8, "emotion_dim": 8, "style_dim": 8, "hidden_dim": 8,
                  "heads": 2, "dropout": 0., "dit_dim": 8, "dit_depth": 1,
                  "residual_scale": .25, "affect_stride": 4, "affect_rank": 3}}


@pytest.fixture(autouse=True)
def seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(91)
    yield
    torch.set_num_threads(previous)


def with_refiner(cfg, enabled=True):
    result = deepcopy(cfg)
    result["model"]["audio_control_refiner"] = enabled
    return result


def load_original_with_only_new_refiner(original, refined):
    """Explicit upgrade: initialize only new keys, then use strict loading."""
    original_state = original.state_dict()
    refined_state = refined.state_dict()
    added = refined_state.keys() - original_state.keys()
    assert added
    assert all(key.startswith("audio_encoder.control_refiner.") for key in added)
    assert not original_state.keys() - refined_state.keys()
    merged = {**original_state, **{key: refined_state[key] for key in added}}
    refined.load_state_dict(merged, strict=True)
    return added


def test_default_disabled_has_identical_keys_initialization_and_outputs(cfg):
    torch.manual_seed(31)
    original = NeutralAffectSystem(cfg).eval()
    torch.manual_seed(31)
    disabled = NeutralAffectSystem(with_refiner(cfg, False)).eval()
    assert original.state_dict().keys() == disabled.state_dict().keys()
    assert original.audio_encoder.control_refiner is None
    for key, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, disabled.state_dict()[key], rtol=0, atol=0)
    audio = torch.randn(2, 19, 5)
    for key, tensor in original.encode_audio(audio).items():
        torch.testing.assert_close(tensor, disabled.encode_audio(audio)[key], rtol=0, atol=0)


def test_zero_initialization_and_strict_upgrade_preserve_audio_teacher_and_decode(cfg):
    original = NeutralAffectSystem(cfg).eval()
    refined = NeutralAffectSystem(with_refiner(cfg)).eval()
    # Loading an old checkpoint directly remains strict: no missing key is
    # silently accepted merely because the optional extension was enabled.
    with pytest.raises(RuntimeError, match="Missing key"):
        refined.load_state_dict(original.state_dict(), strict=True)
    load_original_with_only_new_refiner(original, refined)
    assert refined.motion_teacher.control_refiner is None
    assert refined.audio_encoder.control_refiner.output.weight.count_nonzero() == 0
    assert refined.audio_encoder.control_refiner.output.bias.count_nonzero() == 0
    mask = torch.ones(2, 19, dtype=torch.bool)
    mask[1, 13:] = False
    audio, motion = torch.randn(2, 19, 5), torch.randn(2, 19, 6)
    for method, sequence in (("encode_audio", audio), ("encode_motion", motion)):
        expected, actual = getattr(original, method)(sequence, mask), getattr(refined, method)(sequence, mask)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    content, references, noise = torch.randn(2, 19, 8), torch.randn(2, 2, 19, 6), torch.randn(2, 19, 6)
    with torch.no_grad():
        expected = original.generate(content, mask, original.encode_identity(references), original.encode_audio(audio, mask), noise, 2)
        actual = refined.generate(content, mask, refined.encode_identity(references), refined.encode_audio(audio, mask), noise, 2)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_local_training_keeps_global_logits_and_intensity_bitwise_unchanged(cfg):
    encoder = LowRateAffectEncoder(with_refiner(cfg), 5, motion=False).eval()
    encoder.requires_grad_(False)
    encoder.control_head.requires_grad_(True)
    encoder.control_refiner.requires_grad_(True)
    audio = torch.randn(2, 29, 5)
    mask = torch.ones(2, 29, dtype=torch.bool)
    mask[1, 23:] = False
    before = {key: value.detach().clone() for key, value in encoder(audio, mask).items()}
    frozen = {key: value.clone() for key, value in encoder.state_dict().items()
              if not key.startswith(("control_head.", "control_refiner."))}
    optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=.01)
    target = torch.randn_like(before["controls"])
    loss = F.mse_loss(encoder(audio, mask)["controls"], target)
    loss.backward()
    assert encoder.control_head.weight.grad.abs().sum() > 0
    assert encoder.control_refiner.output.weight.grad.abs().sum() > 0
    # Zero output weights intentionally block upstream refinement gradients
    # for the first step only. They are released after that output learns.
    assert encoder.control_refiner.input.weight.grad.count_nonzero() == 0
    optimizer.step()
    after = encoder(audio, mask)
    assert not torch.equal(before["controls"], after["controls"])
    for key in ("global", "emotion_logits", "intensity_logits", "intensity_value"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    for key, tensor in frozen.items():
        torch.testing.assert_close(tensor, encoder.state_dict()[key], rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(encoder(audio, mask)["controls"], target).backward()
    assert encoder.control_refiner.input.weight.grad.abs().sum() > 0
    assert encoder.control_refiner.temporal[0].conv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for name, p in encoder.named_parameters()
               if not name.startswith(("control_head.", "control_refiner.")))


def test_active_refiner_preserves_padding_missing_frames_and_control_centering(cfg):
    system = NeutralAffectSystem(with_refiner(cfg)).eval()
    with torch.no_grad():
        # Exercise the new path after learning, not only its zero-init bypass.
        system.audio_encoder.control_refiner.output.weight.normal_(std=.05)
        system.audio_encoder.control_refiner.output.bias.normal_(std=.02)
    audio = torch.randn(2, 37, 5)
    mask = torch.ones(2, 37, dtype=torch.bool)
    mask[0, 12:14] = False
    mask[1, 29:] = False
    audio[~mask] = float("nan")
    expected = system.encode_audio(audio, mask)
    padded_audio = F.pad(audio, (0, 0, 0, 31), value=float("nan"))
    padded_mask = F.pad(mask, (0, 31), value=False)
    actual = system.encode_audio(padded_audio, padded_mask)
    for key in ("global", "emotion_logits", "intensity_logits", "intensity_value"):
        torch.testing.assert_close(expected[key], actual[key], rtol=2e-6, atol=2e-6)
    for key in ("controls", "control_mask", "control_weight"):
        torch.testing.assert_close(expected[key], actual[key][:, :expected[key].shape[1]], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(expected["local"], actual["local"][:, :37], rtol=2e-6, atol=2e-6)
    assert actual["local"][~padded_mask].count_nonzero() == 0
    assert actual["controls"][~actual["control_mask"]].count_nonzero() == 0
    torch.testing.assert_close((actual["controls"] * actual["control_weight"].unsqueeze(-1)).sum(1),
                               torch.zeros(2, 3), rtol=0, atol=2e-6)
    padded_audio.requires_grad_(True)
    system.encode_audio(padded_audio, padded_mask)["local"].square().mean().backward()
    assert torch.isfinite(padded_audio.grad).all()
    assert padded_audio.grad[~padded_mask].count_nonzero() == 0


@pytest.mark.parametrize("invalid", [None, "true", 1, 0, []])
def test_audio_refiner_option_requires_explicit_boolean(cfg, invalid):
    with pytest.raises(ValueError, match="must be a boolean"):
        LowRateAffectEncoder(with_refiner(cfg, invalid), 5, motion=False)
