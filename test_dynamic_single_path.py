import torch

from kinetalk_b0.losses import stage3_loss, stage4_loss
from kinetalk_b0.models.dit import ResidualDiT
from kinetalk_b0.models.encoders import AudioEmotionDistributionEncoder
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model


def small_config():
    return {
        "data": {
            "content_dim": 16, "motion_dim": 8, "audio_emotion_dim": 9,
            "neutral_output_indices": [2, 3, 4, 5],
            "emotion_classes": ["neutral", "happy"], "num_intensity_levels": 3,
            "stage1_native_context": 3,
        },
        "model": {
            "content_dim": 8, "emotion_dim": 6, "style_dim": 5, "hidden_dim": 12,
            "heads": 3, "dropout": 0.0, "dit_dim": 12, "dit_depth": 1,
            "residual_scale": 0.25, "intensity_condition_dim": 1,
            "articulatory_indices": [2, 3], "style_condition_dropout": 0.0,
            "stage4_style_condition_initial_gain": 1.0,
            "stage4_style_condition_max_gain": 2.0,
            "stage4_style_adapter_hidden": 8,
            "stage4_style_adapter_max_gate": 1.0,
            "stage4_style_adapter_initial_gate": 0.1,
        },
    }


def test_stage3_audio_outputs_global_emotion_and_intensity_only():
    encoder = AudioEmotionDistributionEncoder(69, 8, 16, 2, 3, 4, 0.0)
    audio = encoder(torch.randn(2, 12, 69), torch.ones(2, 12, dtype=torch.bool))
    assert {"global", "local", "emotion_logits", "intensity_logits", "intensity_value", "hidden"} == set(audio)
    assert audio["intensity_value"].shape == (2, 1)


def test_stage3_loss_uses_global_emotion_and_intensity_only():
    batch = 2
    audio = {
        "global": torch.randn(batch, 8, requires_grad=True),
        "local": torch.randn(batch, 12, 8, requires_grad=True),
        "emotion_logits": torch.randn(batch, 3, requires_grad=True),
        "intensity_logits": torch.randn(batch, 2, requires_grad=True),
    }
    teacher = {"global": torch.randn(2, 8), "local": torch.randn(2, 12, 8)}
    losses = stage3_loss(audio, teacher, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(torch.isfinite(value.grad).all() for value in audio.values())


def test_residual_dit_has_no_dynamic_argument_or_module():
    model = ResidualDiT(3, 5, 8, 4, 16, 1, 2, 0.0, intensity_dim=1)
    assert not any("dynamic" in name for name, _ in model.named_parameters())
    output = model(
        torch.randn(2, 6, 3), torch.rand(2), torch.randn(2, 6, 5),
        torch.randn(2, 8), torch.randn(2, 1), torch.randn(2, 4),
        torch.ones(2, 6, dtype=torch.bool), condition_dropout=False,
    )
    assert output.shape == (2, 6, 3)


def test_stage4_loss_is_flow_only():
    flow_target = torch.zeros(2, 6, 3)
    flow_prediction = torch.zeros_like(flow_target)
    mask = torch.ones(2, 6, dtype=torch.bool)
    losses = stage4_loss(flow_prediction, flow_target, mask)
    assert set(losses) == {"total", "flow"}
    assert losses["total"].item() == 0.0


def test_stage1_audio_only_contract():
    cfg = small_config()
    model = Stage1Model(cfg)
    content = torch.randn(2, 10, 3, 16)
    target = torch.rand(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.bool)
    output = model(content, mask)
    assert {"b0_canonical", "b0", "h0", "lag_frames"}.issubset(output)
    assert "prototype_logits" not in output and "s_art" not in output
    total = output["b0"].square().mean()
    total.backward()
    assert torch.isfinite(total)


def test_stage4_cross_mode_uses_cross_speaker_style_reference():
    cfg = small_config()
    stage1 = Stage1Model(cfg)
    stage2 = Stage2Model(cfg)
    stage3 = Stage3Model(cfg, stage2)
    model = Stage4Model(cfg, stage1, stage2, stage3)
    batch, frames = {}, 6

    def branch(motion_value):
        return {
            "motion": torch.full((2, frames, 8), motion_value),
            "content": torch.randn(2, frames, 16),
            "audio_emotion": torch.randn(2, frames, 9),
            "mask": torch.ones(2, frames, dtype=torch.bool),
            "b0_gt": torch.zeros(2, frames, 8),
        }

    batch["query"] = branch(0.1)
    batch["style_reference"] = branch(0.8)
    batch["emotion_pair"] = branch(0.4)
    seen = []
    hook = model.factor_style.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0].detach().clone()))
    with torch.no_grad():
        reference_b0 = model.stage1(
            batch["style_reference"]["content"], batch["style_reference"]["mask"]
        )["b0"]
        model.conditions(batch, training_target=True, style_mode="cross")
    hook.remove()
    assert len(seen) == 1
    assert torch.allclose(seen[0], batch["style_reference"]["motion"] - reference_b0)


def test_stage4_only_unfreezes_style_specific_renderer_parameters():
    cfg = small_config()
    stage2 = Stage2Model(cfg)
    model = Stage4Model(cfg, Stage1Model(cfg), stage2, Stage3Model(cfg, stage2))
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "style_adapter_gate_logit" in trainable
    assert "residual_gate_logit" not in trainable
    adapter_trainable = {name.removeprefix("style_adapter.") for name in trainable if name.startswith("style_adapter.")}
    assert adapter_trainable
    assert all(not name.startswith("renderer.") for name in trainable)
