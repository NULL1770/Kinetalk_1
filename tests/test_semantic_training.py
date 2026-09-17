import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import train
from kinetalk_b0.models.semantic import (
    MotionSemanticReadout,
    MultiReferenceStyleEncoder,
    SemanticAudioEncoder,
    SemanticGenerator,
)
from kinetalk_b0.utils import freeze_module


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def config():
    return {
        "data": {"content_dim": 8, "motion_dim": 52, "neutral_output_indices": [14, 17, 18],
                 "emotion_classes": ["happy", "sad"], "num_intensity_levels": 3, "audio_emotion_dim": 5},
        "model": {"content_dim": 8, "emotion_dim": 8, "style_dim": 8, "hidden_dim": 8,
                  "heads": 2, "dropout": 0.0, "dit_dim": 8, "dit_depth": 1,
                  "residual_scale": 0.25, "timing_indices": [14, 17, 18]},
        "training": {"decode_steps": 1, "batch_size": 2, "cpu_threads": 1,
                     "critic_steps": 1, "generator_steps": 1, "audio_steps": 1},
        "validation": {"diagnostic_batches": 1},
        "device": "cpu",
    }


def branch(refs=False):
    shape = (2, 2, 5) if refs else (2, 5)
    result = {
        "motion": torch.rand(*shape, 52), "content": torch.rand(*shape, 8),
        "audio": torch.rand(*shape, 5), "valid": torch.ones(shape, dtype=torch.bool),
        "channel_mask": torch.ones(*shape[:-1], 52, dtype=torch.bool),
        "va": torch.rand(*shape, 2) * 2 - 1, "va_valid": torch.ones(shape, dtype=torch.bool),
        "va_confidence": torch.ones(shape), "times": torch.arange(5).double().expand(shape) * 0.04,
        "emotion_id": torch.tensor([0, 1]), "intensity_id": torch.tensor([1, 2]),
        "intensity_valid": torch.ones(2, dtype=torch.bool), "speaker_id": torch.tensor([0, 1]),
        "metadata": [{}, {}],
    }
    result["channel_mask"][..., 51] = False
    return result


def batch():
    value = {"query": branch(), "same_style_references": branch(True),
             "positive_style_references": branch(True), "donor_references": branch(True),
             "donor_anchor": branch(), "donor_intensity_matched": torch.ones(2, dtype=torch.bool)}
    value["donor_anchor"]["speaker_id"] = torch.tensor([1, 0])
    return value


class Tiny(Dataset):
    def __init__(self):
        self.epochs = []

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return index

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def test_iterator_restarts_and_masks_exclude_missing_channels():
    dataset = Tiny()
    iterator = train.infinite_batches(DataLoader(dataset, batch_size=2))
    assert [next(iterator)[0] for _ in range(3)] == [0, 1, 2]
    assert dataset.epochs == [0, 1, 2]
    q = branch()
    q["valid"][:, -1] = False
    observed = train.observed(q)
    assert not observed[..., 51].any() and not observed[:, -1].any()
    prediction = torch.ones_like(q["motion"])
    prediction[~observed] = float("nan")
    assert train.masked_flow(prediction, torch.zeros_like(prediction), observed).item() == 1


def test_generator_cross_decode_updates_style_path_but_frozen_critics_have_no_gradients():
    cfg = config()
    models = {"generator": SemanticGenerator(cfg), "readout": MotionSemanticReadout(cfg),
              "style_encoder": MultiReferenceStyleEncoder(cfg), "audio": SemanticAudioEncoder(cfg)}
    freeze_module(models["readout"])
    freeze_module(models["style_encoder"])
    models["generator"].train()
    loss, components = train.generator_step(models, batch(), cfg, "cpu")
    assert "cross" in components and torch.isfinite(loss)
    loss.backward()
    assert models["generator"].renderer.style.weight.grad.abs().sum() > 0
    assert all(p.grad is None for key in ("readout", "style_encoder") for p in models[key].parameters())
    assert all(p.grad is None for p in models["generator"].stage1.parameters())
    assert not models["generator"].stage1.training


def test_critic_and_audio_gates_cannot_turn_failed_metrics_into_acceptance():
    failed = train.critic_gate({"emotion_balanced_accuracy": 0.1, "va_ccc_mean": None,
                               "va_mae": 0.8, "style_retrieval": 1.0}, config())
    assert not failed["critics_accepted"] and len(failed["failures"]) == 3
    with pytest.raises(RuntimeError, match="nonexploratory"):
        train.require_audio_gate({"stage": "generator", "validation": {"visual_path_accepted": True}, "exploratory": True})
    train.require_audio_gate({"stage": "generator", "validation": {"visual_path_accepted": True}})


def test_main_preflight_loads_cpu_foundation_and_records_full_iteration(tmp_path, monkeypatch):
    cfg = config()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("seed: 2\n", encoding="utf8")
    checkpoint = tmp_path / "stage1.pt"
    torch.save({"model": SemanticGenerator(cfg).stage1.state_dict()}, checkpoint)
    cfg["paths"] = {"output_dir": str(tmp_path / "out"), "stage1_ckpt": str(checkpoint)}
    class Loader:
        def __init__(self, split):
            self.dataset = Tiny()
            self.dataset.items = [{"dataset": "mead", "speaker": f"{split}_{i}"} for i in range(2)]
            self.dataset.manifest = tmp_path / f"{split}.jsonl"
            self.dataset.manifest.write_text("{}\n", encoding="utf8")
        def __iter__(self):
            return iter([batch()])
    monkeypatch.setattr(train, "load_yaml", lambda _: cfg)
    monkeypatch.setattr(train, "loader", lambda _, split, training: Loader(split))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "--stage", "preflight"])
    train.main()
    audit = json.loads((tmp_path / "out" / "preflight.json").read_text())
    assert audit["stage1_strict_load"] and audit["complete_dataset_iteration"]
    assert audit["loaded_examples"] == {"train": 2, "val": 2}


def test_validation_and_diagnostic_execute_on_real_model_interfaces(tmp_path):
    cfg = config()
    models = {"generator": SemanticGenerator(cfg), "readout": MotionSemanticReadout(cfg),
              "style_encoder": MultiReferenceStyleEncoder(cfg), "audio": SemanticAudioEncoder(cfg)}
    data = [batch()]
    metrics = train.validate(models, data, "cpu", cfg)
    assert metrics["examples"] == 2 and metrics["style_speakers"] == 2
    assert metrics["va_ccc_mean"] is not None and metrics["va_mae"] >= 0
    report = train.evaluate(models, data, "cpu", cfg, tmp_path, {"critics_accepted": False})
    assert report["examples"] == 2 and not report["visual_path_accepted"]
    assert report["aggregate"]["repeat_raw_max_abs"] == 0
    assert (tmp_path / "generator_diagnostic.json").is_file()
