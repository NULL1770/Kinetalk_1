"""Feature input plumbing and matched initialization, not motion quality."""
from __future__ import annotations

import copy

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_neutral_affect_feature_probe import (
    build_feature_system, fit_feature_stats, prepare_feature_batch,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def teacher_checkpoint():
    cfg = {
        "data": {"content_dim": 768, "motion_dim": 6, "audio_dim": 83,
                 "neutral_output_indices": [2, 3], "emotion_classes": ["neutral", "happy", "sad"],
                 "num_intensity_levels": 3},
        "model": {"content_dim": 8, "emotion_dim": 8, "style_dim": 8, "hidden_dim": 8,
                  "heads": 2, "dropout": 0., "dit_dim": 8, "dit_depth": 1,
                  "residual_scale": .25, "affect_stride": 4, "affect_rank": 3,
                  "global_condition_dropout": 0., "style_condition_dropout": 0.},
    }
    return cfg, {"stage": "motion_teacher", "model": NeutralAffectSystem(cfg).state_dict(),
                 "provenance": {"config": copy.deepcopy(cfg)}}


@pytest.mark.parametrize("source", ["acoustic", "content"])
def test_transform_keeps_content_and_uses_only_training_valid_statistics(source):
    train = {"audio": torch.randn(2, 5, 3), "content": torch.randn(2, 5, 4),
             "valid": torch.ones(2, 5, dtype=torch.bool)}
    train["valid"][1, -2:] = False
    for key in ("audio", "content"):
        train[key][~train["valid"]] = float("nan")
    original = {key: value.clone() for key, value in train.items()}
    feature_key = "audio" if source != "content" else "content"
    values = train[feature_key][train["valid"]]
    stats = fit_feature_stats(train, source)
    assert stats["count"] == 8
    torch.testing.assert_close(stats["mean"], values.mean(0))
    torch.testing.assert_close(stats["std"], values.std(0).clamp_min(1e-3))
    transformed = prepare_feature_batch(train, source, stats)
    assert transformed is not train and transformed["content"] is train["content"]
    assert transformed["audio"] is not train["audio"]
    assert transformed["audio"][~train["valid"]].count_nonzero() == 0
    for key in train:
        torch.testing.assert_close(train[key], original[key], equal_nan=True)
    heldout = {"audio": torch.full((1, 5, 3), 100.), "content": torch.full((1, 5, 4), 200.),
               "valid": torch.ones(1, 5, dtype=torch.bool)}
    saved_mean, saved_std = stats["mean"].clone(), stats["std"].clone()
    result = prepare_feature_batch(heldout, source, stats)
    torch.testing.assert_close(result["audio"], (heldout[feature_key] - saved_mean) / saved_std)
    torch.testing.assert_close(stats["mean"], saved_mean)
    torch.testing.assert_close(stats["std"], saved_std)


def test_equal_shaped_audio_initialization_matches_acoustic_and_content(teacher_checkpoint):
    cfg, checkpoint = teacher_checkpoint
    acoustic, _ = build_feature_system(cfg, checkpoint, "acoustic", 44)
    content, _ = build_feature_system(cfg, checkpoint, "content", 44)
    for name, value in acoustic.state_dict().items():
        if name != "audio_encoder.input.weight":
            torch.testing.assert_close(value, content.state_dict()[name], rtol=0, atol=0)
        if not name.startswith("audio_encoder."):
            torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)
    for name in ("emotion_classifier", "intensity_classifier"):
        for key, value in getattr(acoustic.audio_encoder, name).state_dict().items():
            torch.testing.assert_close(value, getattr(acoustic.motion_teacher, name).state_dict()[key], rtol=0, atol=0)


def test_768_cached_acoustic_override_preserves_teacher_config_and_loads(teacher_checkpoint):
    cfg, checkpoint = teacher_checkpoint
    system, effective = build_feature_system(cfg, checkpoint, "acoustic", 44, input_dim=768)
    content, _ = build_feature_system(cfg, checkpoint, "content", 44)
    assert cfg["data"]["audio_dim"] == effective["data"]["audio_dim"] == 83
    assert "audio_emotion_dim" not in cfg["data"]
    assert effective["data"]["audio_emotion_dim"] == system.audio_encoder.input.in_features == 768
    for name, value in system.state_dict().items():
        torch.testing.assert_close(value, content.state_dict()[name], rtol=0, atol=0)
    query = {"audio": torch.randn(2, 12, 768), "content": torch.randn(2, 12, 768),
             "valid": torch.ones(2, 12, dtype=torch.bool)}
    batch = prepare_feature_batch(query, "acoustic", fit_feature_stats(query, "acoustic"))
    output = system.encode_audio(batch["audio"], batch["valid"])
    assert output["controls"].shape == (2, 3, 3)
    assert torch.isfinite(output["controls"]).all()
    reloaded = NeutralAffectSystem(effective)
    reloaded.load_state_dict(system.state_dict(), strict=True)
    with pytest.raises(ValueError, match="acoustic source only"):
        build_feature_system(cfg, checkpoint, "content", 44, input_dim=768)
