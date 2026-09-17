"""Integration guards for the shipped native-data v9 configuration."""
from pathlib import Path

import pytest
import torch
import yaml

from kinetalk_b0.models.semantic import SemanticAudioEncoder, SemanticGenerator
from scripts.diagnose_semantic import ARKIT_NAMES


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


@pytest.fixture(scope="module")
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)


def test_native_audio_configuration_matches_real_83_channel_input(config):
    assert config["data"]["audio_dim"] == config["semantic_data"]["audio_dim"] == 83
    encoder = SemanticAudioEncoder(config).eval()
    with torch.no_grad():
        outputs = encoder(torch.randn(2, 8, 83), torch.ones(2, 8, dtype=torch.bool))
    assert outputs["va"].shape == (2, 8, 2)
    assert torch.isfinite(outputs["va"]).all()


def test_primary_configuration_has_no_old_factor_training_path(config):
    assert "optim" not in config
    assert "audio_emotion_dim" not in config["data"]
    assert "aligned_dtw_root" not in config["data"]
    assert "require_neutral_partner" not in config["data"]
    assert set(config["paths"]) == {"stage1_ckpt", "output_dir"}
    assert set(config["loss"]) == {"cross", "mouth_timing"}
    assert Path(config["paths"]["stage1_ckpt"]).name == "stage1_neutral.pt"


def test_articulation_timing_indices_use_declared_arkit_channels(config):
    names = {ARKIT_NAMES[index] for index in config["model"]["timing_indices"]}
    assert names == {
        "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper", "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
    }
    assert not any(any(token in name.lower() for token in ("smile", "frown", "brow", "eye", "dimple")) for name in names)


def test_primary_generator_preserves_frozen_b0_v7_architecture(config):
    model = SemanticGenerator(config)
    model.train()
    assert model.architecture_version.item() == 9
    assert model.stage1.architecture_version.item() == 7
    assert not model.stage1.training
    assert not any(parameter.requires_grad for parameter in model.stage1.parameters())
