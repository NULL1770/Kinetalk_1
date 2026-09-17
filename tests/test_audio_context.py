"""Audio context changes preserve the teacher and checkpoint tensor contract."""

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
    torch.manual_seed(29)
    yield
    torch.set_num_threads(previous)


def configured(cfg, dilations):
    out = deepcopy(cfg)
    out["model"]["audio_dilations"] = dilations
    return out


def test_default_config_preserves_parameters_and_outputs_bitwise(cfg):
    torch.manual_seed(71)
    implicit = NeutralAffectSystem(cfg).eval()
    torch.manual_seed(71)
    explicit = NeutralAffectSystem(configured(cfg, [1, 2, 4])).eval()
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for key, tensor in implicit.state_dict().items():
        torch.testing.assert_close(tensor, explicit.state_dict()[key], rtol=0, atol=0)
    sequence = torch.randn(2, 25, 5)
    mask = torch.ones(2, 25, dtype=torch.bool)
    mask[1, 19:] = False
    expected, actual = implicit.encode_audio(sequence, mask), explicit.encode_audio(sequence, mask)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_audio_dilation_change_strictly_loads_old_weights_and_preserves_teacher(cfg):
    original = NeutralAffectSystem(cfg).eval()
    longer = NeutralAffectSystem(configured(cfg, [1, 4, 16])).eval()
    longer.load_state_dict(original.state_dict(), strict=True)
    assert longer.state_dict().keys() == original.state_dict().keys()
    assert sum(p.numel() for p in longer.parameters()) == sum(p.numel() for p in original.parameters())
    for key, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, longer.state_dict()[key], rtol=0, atol=0)
    assert [block.conv.dilation for block in original.audio_encoder.temporal] == [(1,), (2,), (4,)]
    assert [block.conv.dilation for block in longer.audio_encoder.temporal] == [(1,), (4,), (16,)]
    assert [block.conv.dilation for block in longer.motion_teacher.temporal] == [(1,), (2,), (4,)]
    assert 1 + 2 * sum(original.audio_encoder.dilations) == 15
    assert 1 + 2 * sum(longer.audio_encoder.dilations) == 43
    sequence = torch.randn(2, 49, 6)
    expected, actual = original.encode_motion(sequence), longer.encode_motion(sequence)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_longer_context_uses_distant_input_beyond_old_local_receptive_field(cfg):
    short = LowRateAffectEncoder(cfg, 5, motion=False).eval()
    longer = LowRateAffectEncoder(configured(cfg, [1, 4, 16]), 5, motion=False).eval()
    # Nonzero positive weights make the dependency reproducible, without
    # relying on an accidental random cancellation of the impulse response.
    with torch.no_grad():
        for parameter in short.parameters():
            parameter.fill_(.02 if parameter.ndim > 1 else 0.)
    longer.load_state_dict(short.state_dict(), strict=True)
    clean = torch.zeros(1, 65, 5)
    perturbed = clean.clone()
    perturbed[0, 48, 0] = 1.
    mask = torch.ones(1, 65, dtype=torch.bool)
    observed = {}
    for name, encoder in (("short", short), ("long", longer)):
        values = []
        handle = encoder.temporal[-1].register_forward_hook(lambda module, args, output: values.append(output.detach().clone()))
        encoder(clean, mask)
        encoder(perturbed, mask)
        handle.remove()
        # Inspect before global centering: centered controls always have a
        # whole-sequence DC dependency, which would be a misleading RF test.
        observed[name] = (values[1] - values[0])[0, :, 32]
    assert observed["short"].count_nonzero() == 0
    assert observed["long"].abs().sum() > 1e-7


def test_long_audio_context_preserves_padding_invariance(cfg):
    system = NeutralAffectSystem(configured(cfg, [1, 4, 16])).eval()
    sequence = torch.randn(2, 37, 5)
    mask = torch.ones(2, 37, dtype=torch.bool)
    mask[0, 12:14] = False
    mask[1, 29:] = False
    expected = system.encode_audio(sequence, mask)
    padded_sequence = F.pad(sequence, (0, 0, 0, 31), value=float("nan"))
    padded_mask = F.pad(mask, (0, 31), value=False)
    actual = system.encode_audio(padded_sequence, padded_mask)
    for key in ("global", "emotion_logits", "intensity_logits", "intensity_value"):
        torch.testing.assert_close(expected[key], actual[key], atol=2e-6, rtol=2e-6)
    for key in ("controls", "control_mask", "control_weight"):
        torch.testing.assert_close(expected[key], actual[key][:, :expected[key].shape[1]], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(expected["local"], actual["local"][:, :37], atol=2e-6, rtol=2e-6)
    assert actual["local"][~padded_mask].count_nonzero() == 0
    assert actual["controls"][~actual["control_mask"]].count_nonzero() == 0


@pytest.mark.parametrize("dilations", [[], [1], [1, 2], [1, 2, 4, 8], [1, 0, 4], [1, -2, 4],
                                      [1, 2., 4], [True, 2, 4], [1, "2", 4], "1,2,4", None, 4])
def test_invalid_audio_dilations_fail_instead_of_changing_depth(cfg, dilations):
    with pytest.raises(ValueError, match="exactly three positive integers"):
        LowRateAffectEncoder(configured(cfg, dilations), 5, motion=False)
