"""Tests cover isolation and target plumbing, not empirical dynamic quality."""
import pytest
import torch
import json
import sys
import yaml

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import SCHEMA

from scripts.probe_dynamic_predictability import (
    bin_frames, center_controls, fit_pca_target, fit_target_scale, fresh_student,
    sentence_split, smooth_controls, target_metrics,
    training_only_audio,
    main,
)
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import sha


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(45)
        yield
    torch.set_num_threads(old)


def test_sentence_groups_are_disjoint_reproducible_and_metadata_only():
    sentences = [f"sentence{i % 10}" for i in range(50)]
    train, heldout = sentence_split(sentences, 45)
    again = sentence_split(sentences, 45)
    assert torch.equal(train, again[0]) and torch.equal(heldout, again[1])
    assert len(train) == 40 and len(heldout) == 10
    assert not {sentences[i] for i in train} & {sentences[i] for i in heldout}
    with pytest.raises(ValueError):
        sentence_split(["a", "b"], 45)


def test_audio_statistics_are_refit_after_inverse_transform_on_train_only():
    raw = torch.randn(5, 12, 3)
    valid = torch.ones(5, 12, dtype=torch.bool)
    valid[0, -2:] = False
    old = {"mean": torch.tensor([5., 8., -2.]), "std": torch.tensor([2., 3., 7.])}
    cached = {"audio": (raw - old["mean"]) / old["std"], "valid": valid}
    cached["audio"][~valid] = float("nan")
    train = torch.arange(3)
    transformed, stats = training_only_audio(cached, old, train)
    actual = raw[train][valid[train]]
    torch.testing.assert_close(stats["mean"], actual.mean(0), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(stats["std"], actual.std(0), atol=2e-6, rtol=2e-6)
    modified = {**cached, "audio": cached["audio"].clone()}
    modified["audio"][3:] = 999
    changed, stats2 = training_only_audio(modified, old, train)
    assert torch.equal(stats["mean"], stats2["mean"]) and torch.equal(stats["std"], stats2["std"])
    torch.testing.assert_close(transformed["audio"][:3], changed["audio"][:3], rtol=0, atol=0)
    assert transformed["audio"][~valid].count_nonzero() == 0


def test_bins_and_smoothing_ignore_padding_and_preserve_partial_weights():
    frames = torch.randn(2, 11, 3)
    valid = torch.ones(2, 11, dtype=torch.bool)
    valid[0, 5] = False
    valid[1, 8:] = False
    frames[~valid] = float("nan")
    values, weight = bin_frames(frames, valid, 4)
    assert weight.tolist() == [[4., 3., 3.], [4., 4., 0.]]
    torch.testing.assert_close(values[0, 1], frames[0, [4, 6, 7]].mean(0))
    centered = center_controls(values, weight)
    smooth = smooth_controls(values, weight)
    for result in (centered, smooth):
        torch.testing.assert_close((result * weight[..., None]).sum(1), torch.zeros(2, 3), atol=1e-6, rtol=0)
        assert result[weight == 0].count_nonzero() == 0
    padded = torch.cat([values, torch.full((2, 4, 3), float("nan"))], 1)
    padded_weight = torch.cat([weight, torch.zeros(2, 4)], 1)
    torch.testing.assert_close(smooth_controls(padded, padded_weight)[:, :3], smooth)


def test_pca_and_scales_fit_only_train_and_missing_channels_are_not_measured():
    residual = torch.randn(6, 20, 7)
    valid = torch.ones(6, 20, dtype=torch.bool)
    channels = torch.ones(6, 7, dtype=torch.bool)
    channels[:, -1] = False
    residual[:, :, -1] = float("nan")
    train = torch.arange(4)
    target, weight, pca = fit_pca_target(residual, valid, channels, train, 3, 4)
    changed = residual.clone()
    changed[4:, :, :6] = torch.randn_like(changed[4:, :, :6]) * 1000
    target2, weight2, pca2 = fit_pca_target(changed, valid, channels, train, 3, 4)
    torch.testing.assert_close(pca["basis"], pca2["basis"], rtol=0, atol=0)
    torch.testing.assert_close(target[:4], target2[:4], rtol=0, atol=0)
    torch.testing.assert_close(fit_target_scale(target, weight, train), fit_target_scale(target2, weight2, train), rtol=0, atol=0)
    assert pca["channel_indices"].tolist() == list(range(6))
    channels[5, 0] = False
    with pytest.raises(ValueError, match="heldout clip lacks"):
        fit_pca_target(residual, valid, channels, train, 3, 4)


def test_metrics_use_zero_dynamic_baseline_and_ignore_padded_values():
    weight = torch.tensor([[4., 4., 1.], [4., 4., 0.]])
    target = center_controls(torch.randn(2, 3, 3), weight)
    metrics = target_metrics(target, target, weight, ["a", "b"], ["s1", "s2"], torch.ones(3))
    assert metrics["normalized_mse"] == 0 and metrics["r2_against_zero"] == 1
    assert metrics["temporal_correlation"] == pytest.approx(1.)
    zero = target_metrics(torch.zeros_like(target), target, weight, ["a", "b"], ["s1", "s2"], torch.ones(3))
    assert zero["r2_against_zero"] == pytest.approx(0.)
    assert zero["temporal_correlation"] is None


def test_fresh_students_have_matching_initial_states_and_only_dynamic_path_trains():
    cfg = {"data": {"emotion_classes": ["neutral", "happy"], "num_intensity_levels": 3},
           "model": {"hidden_dim": 8, "emotion_dim": 8, "affect_stride": 4, "affect_rank": 3}}
    first = fresh_student(cfg, 45, 5, "cpu")
    torch.manual_seed(999)
    second = fresh_student(cfg, 45, 5, "cpu")
    assert state_hash(first.state_dict()) == state_hash(second.state_dict())
    x = torch.randn(2, 20, 5)
    target = torch.randn(2, 5, 3)
    loss = (first(x)["controls"] - target).square().mean()
    loss.backward()
    assert first.input.weight.grad.abs().sum() > 0
    assert first.control_head.weight.grad.abs().sum() > 0
    assert first.global_head.weight.grad is None


def test_full_cli_four_arm_smoke_reloads_and_preserves_frozen_checkpoint(tmp_path, monkeypatch):
    cfg = {"device": "cpu", "data": {"content_dim": 8, "motion_dim": 6, "audio_dim": 5,
             "neutral_output_indices": [2, 3], "emotion_classes": ["neutral", "happy"], "num_intensity_levels": 3},
           "model": {"content_dim": 8, "emotion_dim": 8, "style_dim": 8, "hidden_dim": 8,
                     "heads": 2, "dropout": 0., "dit_dim": 8, "dit_depth": 1, "residual_scale": .25,
                     "affect_stride": 4, "affect_rank": 3}, "training": {"batch_size": 4, "audio_lr": .001}}
    def clip(speaker, sentence, suffix, emotion=0):
        return {"audio": torch.randn(16, 5), "content": torch.randn(16, 8), "motion": torch.randn(16, 6) * .1,
                "valid": torch.ones(16, dtype=torch.bool), "channel_mask": torch.ones(6, dtype=torch.bool),
                "times": torch.arange(16).double() / 25, "speaker_id": torch.tensor(speaker),
                "speaker": f"person{speaker}", "emotion_id": torch.tensor(emotion), "intensity_id": torch.tensor(1),
                "sentence_id": sentence, "clip_id": f"person{speaker}_{sentence}_{suffix}",
                "metadata": {"artifact_sha256": "0" * 64}}
    data = tmp_path / "data"
    data.mkdir()
    cache = {"schema": SCHEMA, "queries": [clip(i % 2, f"sentence{i // 2}", "query", i % 2) for i in range(12)],
             "identity_references": {speaker: [clip(speaker, f"ref{i}", "reference") for i in range(4)] for speaker in range(2)},
             "audio_stats": {"mean": torch.zeros(5), "std": torch.ones(5)}}
    torch.save(cache, data / "train.pt")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf8")
    checkpoint = tmp_path / "audio.pt"
    model = NeutralAffectSystem(cfg).state_dict()
    torch.save({"stage": "audio", "audio_source": "acoustic", "feature_stats": {}, "config": cfg,
                "model": model, "provenance": {"train_sha256": sha(data / "train.pt")}}, checkpoint)
    before = sha(checkpoint)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["probe", "--config", str(config), "--data", str(data), "--checkpoint", str(checkpoint),
                                      "--output", str(output), "--steps", "2", "--seed", "45"])
    main()
    assert sha(checkpoint) == before
    result = json.loads((output / "summary.json").read_text())
    assert all(result["checks"].values())
    assert len({result[name]["minibatch_sha256"] for name in result if name != "checks"}) == 1
    for name in result:
        if name == "checks":
            continue
        saved = torch.load(output / (name + ".pt"), weights_only=False)
        student = fresh_student(cfg, 45, 5, "cpu")
        student.load_state_dict(saved["model"], strict=True)
        assert result[name]["heldout"]["n"] > 0
        assert result[name]["train"]["normalized_target_energy"] == pytest.approx(1., abs=1e-4)
    curves = torch.load(output / "curves.pt", weights_only=False)
    assert len(curves["targets"]) == 4
