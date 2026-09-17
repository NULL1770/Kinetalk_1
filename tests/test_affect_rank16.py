"""Small shape contracts for the higher-capacity low-rate affect field.

Rank 16 is a fresh experiment arm.  It must be constructed and trained from
the same teacher input, rather than pretending that an 8D checkpoint is a
16D checkpoint by zero-padding its tensors.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem


@pytest.fixture
def cfg():
    return {
        "data": {
            "content_dim": 8,
            "motion_dim": 6,
            "neutral_output_indices": [2, 3],
            "emotion_classes": ["neutral", "happy", "sad"],
            "num_intensity_levels": 3,
            "audio_emotion_dim": 5,
        },
        "model": {
            "content_dim": 8,
            "emotion_dim": 8,
            "style_dim": 8,
            "hidden_dim": 8,
            "affect_hidden_dim": 8,
            "heads": 2,
            "dropout": 0.0,
            "dit_dim": 8,
            "dit_depth": 1,
            "residual_scale": 0.25,
            "affect_stride": 4,
            "affect_rank": 16,
            "global_condition_dropout": 0.0,
            "style_condition_dropout": 0.0,
        },
    }


def test_rank16_forward_and_local_projection_contract(cfg):
    torch.manual_seed(7)
    system = NeutralAffectSystem(cfg).train()
    assert system.motion_teacher.rank == 16
    assert system.audio_encoder.rank == 16
    assert tuple(system.local_projection.weight.shape) == (8, 16)
    assert system.renderer.local_emotion.in_features == 8

    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[1, -2:] = False
    motion = torch.randn(2, 13, 6)
    audio = torch.randn(2, 13, 5)
    motion_out = system.encode_motion(motion, mask)
    audio_out = system.encode_audio(audio, mask)
    for output in (motion_out, audio_out):
        assert output["controls"].shape == (2, 4, 16)
        assert output["local"].shape == (2, 13, 8)
        assert output["controls"][~output["control_mask"]].count_nonzero() == 0
        assert output["local"][~mask].count_nonzero() == 0

    # Exercise the complete local-to-flow path.  The renderer still receives
    # emotion_dim features, while every 16D control can receive its gradient.
    content = torch.randn(2, 13, 8)
    identity = system.encode_identity(torch.randn(2, 2, 13, 6))
    output = system.flow(motion, content, mask, identity, audio_out,
                         noise=torch.zeros_like(motion), time=torch.zeros(2))
    assert output["motion"].shape == motion.shape
    assert torch.isfinite(output["motion"]).all()
    loss = (output["prediction"] - output["velocity_target"]).square().mean()
    loss.backward()
    assert system.local_projection.weight.grad is not None
    assert system.local_projection.weight.grad.shape == (8, 16)
    assert (system.local_projection.weight.grad.abs().sum(0) > 0).all()
    assert (system.audio_encoder.control_head.weight.grad.abs().sum(1) > 0).all()
    assert all(parameter.grad is None for parameter in system.stage1.parameters())


def test_rank16_is_not_strictly_compatible_with_rank8_checkpoint(cfg):
    rank8_cfg = deepcopy(cfg)
    rank8_cfg["model"]["affect_rank"] = 8
    rank8 = NeutralAffectSystem(rank8_cfg)
    rank16 = NeutralAffectSystem(cfg)
    with pytest.raises(RuntimeError, match="size mismatch"):
        rank16.load_state_dict(rank8.state_dict(), strict=True)


def test_rank16_reference_config_is_explicit():
    root = Path(__file__).resolve().parents[1] / "configs"
    rank8 = yaml.safe_load((root / "neutral_affect_pilot.yaml").read_text(encoding="utf-8"))
    rank16 = yaml.safe_load((root / "neutral_affect_pilot_rank16.yaml").read_text(encoding="utf-8"))
    assert rank8["model"]["affect_rank"] == 8
    assert rank16["model"]["affect_rank"] == 16
    rank16["model"]["affect_rank"] = 8
    assert rank16 == rank8
