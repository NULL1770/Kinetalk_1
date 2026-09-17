"""Verify the objective change and gradient path, not real-data usefulness."""
from __future__ import annotations

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import affect_residual, distill, semantics
from scripts.train_neutral_affect_task_ablation import task_objective


@pytest.fixture
def tiny_case():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(11)
        cfg = {
            "data": {"content_dim": 8, "motion_dim": 6, "neutral_output_indices": [2, 3],
                     "emotion_classes": ["neutral", "happy", "sad"], "num_intensity_levels": 3,
                     "audio_emotion_dim": 5},
            "model": {"content_dim": 8, "emotion_dim": 8, "style_dim": 8, "hidden_dim": 8,
                      "heads": 2, "dropout": 0., "dit_dim": 8, "dit_depth": 1,
                      "residual_scale": .25, "affect_stride": 4, "affect_rank": 3,
                      "global_condition_dropout": 0., "style_condition_dropout": 0.},
        }
        system = NeutralAffectSystem(cfg)
        freeze_module(system)
        system.audio_encoder.requires_grad_(True)
        system.eval()
        query = {
            "motion": torch.randn(2, 12, 6) * .1, "content": torch.randn(2, 12, 8),
            "audio": torch.randn(2, 12, 5), "valid": torch.ones(2, 12, dtype=torch.bool),
            "channel_mask": torch.ones(2, 6, dtype=torch.bool),
            "emotion_id": torch.tensor([1, 2]), "intensity_id": torch.tensor([1, 2]),
            "intensity_valid": torch.tensor([True, False]),
        }
        query["valid"][1, -3:] = False
        query["channel_mask"][1, -1] = False
        with torch.no_grad():
            base = system.base(query["content"], query["valid"])
            identity = system.encode_identity(torch.randn(2, 4, 12, 6) * .02)
            query["residual"] = torch.where(
                query["valid"].unsqueeze(-1) & query["channel_mask"].unsqueeze(1),
                query["motion"] - base["b0"], 0)
            teacher = system.encode_motion(affect_residual(query, identity), query["valid"])
        scales = {"global": torch.linspace(.1, .3, 8), "controls": torch.tensor([.05, .1, .2])}
        try:
            yield system, query, base, identity, teacher, scales
        finally:
            torch.set_num_threads(previous_threads)


def test_latent_objective_matches_original_distillation(tiny_case):
    system, query, base, identity, teacher, scales = tiny_case
    audio = system.encode_audio(query["audio"], query["valid"])
    expected = distill(audio, teacher, scales) + .1 * semantics(audio, query)
    actual, _ = task_objective(system, query, base, identity, teacher, scales,
                               "latent", torch.Generator().manual_seed(43))
    torch.testing.assert_close(actual, expected)


def test_flow_task_alone_reaches_audio_controls_and_preserves_frozen_state(tiny_case):
    system, query, base, identity, teacher, scales = tiny_case
    frozen_state = lambda: {k: v for k, v in system.state_dict().items()
                            if not k.startswith("audio_encoder.")}
    before = state_hash(frozen_state())
    _, parts = task_objective(system, query, base, identity, teacher, scales,
                              "flow", torch.Generator().manual_seed(43))
    # Exclude global distillation and classifier CE: these cannot explain a
    # nonzero dynamic-control gradient in this check.
    assert torch.isfinite(parts["task"])
    parts["task"].backward()
    grad = system.audio_encoder.control_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    frozen_parameters = [p for name, p in system.named_parameters()
                         if not name.startswith("audio_encoder.")]
    assert all(not p.requires_grad and p.grad is None for p in frozen_parameters)
    torch.optim.AdamW(system.audio_encoder.parameters(), lr=.0005).step()
    assert state_hash(frozen_state()) == before
