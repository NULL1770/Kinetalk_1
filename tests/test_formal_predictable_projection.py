"""Formal-run recovery and checkpoint decisions, not a quality benchmark."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from scripts.train_formal_predictable_projection import (
    SCHEMA, canonical_hash, capture_rng, cpu_tree, frozen_hash,
    restore_checkpoint, restore_rng, save_checkpoint, selection_metrics,
    teacher_probability, write_curve_provenance, CURVE_PROVENANCE_SCHEMA,
)
from scripts.train_predictable_renderer import sha, state_hash


class TinySystem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.local_projection = torch.nn.Linear(2, 2, bias=False)
        self.base.requires_grad_(False)


def reports(gain=.02, neutral_ratio=1.01, corr_drop=.002, class_delta=0., velocity_ratio=1.01):
    result = {}
    for seed in (42, 123, 2026):
        item = {}
        for mode in ("full", "zero"):
            raw = {"native_mse": 1. * (neutral_ratio if mode == "full" else 1.),
                   "pooled_centered_correlation": .6 - (corr_drop if mode == "full" else 0.)}
            item[mode] = {
                "nonneutral": {"upper_expression": {"centered_residual": {"r2_against_zero": -.1 + (gain if mode == "full" else 0.)},
                    "velocity_mse_per_second": velocity_ratio if mode == "full" else 1.}},
                "neutral": {"mouth": {"raw_motion": dict(raw), "velocity_mse_per_second": velocity_ratio if mode == "full" else 1.},
                            "upper_expression": {"raw_motion": dict(raw), "velocity_mse_per_second": velocity_ratio if mode == "full" else 1.}},
                "all": {"mouth": {"raw_motion": dict(raw), "velocity_mse_per_second": velocity_ratio if mode == "full" else 1.}},
                "frozen_teacher_emotion_accuracy": .9 + (class_delta if mode == "full" else 0.),
            }
        result[str(seed)] = item
    return result


def test_teacher_schedule_full_five_epochs_then_audio_only():
    assert teacher_probability(0, 0, 7) == .5
    assert teacher_probability(4, 6, 7) == 0
    assert teacher_probability(5, 0, 7) == 0
    assert teacher_probability(39, 6, 7) == 0
    values = [teacher_probability(e, b, 7) for e in range(6) for b in range(7)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_negative_absolute_r2_can_be_eligible_but_neutral_drift_cannot():
    chosen = selection_metrics(reports(), .9)
    assert chosen["eligible"]
    assert chosen["upper_full_r2"] < 0
    assert not selection_metrics(reports(neutral_ratio=1.031), .9)["eligible"]
    assert not selection_metrics(reports(corr_drop=.011), .9)["eligible"]
    assert not selection_metrics(reports(class_delta=-.011), .9)["eligible"]
    assert not selection_metrics(reports(velocity_ratio=1.051), .9)["eligible"]
    assert not selection_metrics(reports(gain=0), .9)["eligible"]


def test_selection_uses_three_noise_means_and_rejects_undefined_metrics():
    sample = reports()
    sample["42"]["full"]["neutral"]["mouth"]["raw_motion"]["native_mse"] = 1.2
    assert not selection_metrics(sample, .9)["eligible"]
    with pytest.raises(ValueError, match="three"):
        selection_metrics({"42": sample["42"]}, .9)
    sample = reports()
    sample["42"]["full"]["all"]["mouth"]["raw_motion"]["pooled_centered_correlation"] = None
    with pytest.raises(ValueError, match="Undefined"):
        selection_metrics(sample, .9)


def test_rng_roundtrip_covers_python_numpy_cpu_and_training_generator():
    random.seed(4); np.random.seed(4); torch.manual_seed(4)
    generator = torch.Generator().manual_seed(5)
    state = capture_rng(generator)
    expected = random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator)
    restore_rng(state, generator)
    actual = random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def test_compact_resume_reproduces_next_optimizer_update_and_rejects_hash_change(tmp_path):
    torch.manual_seed(13)
    system, head = TinySystem(), torch.nn.Linear(2, 2, bias=False)
    head.requires_grad_(False)
    optimizer = torch.optim.Adam(system.local_projection.parameters(), lr=.01)
    generator = torch.Generator().manual_seed(23)
    recipe = {"input_sha256": {"cache": "exact-input"}, "config": {"epochs": 40}}
    def step():
        inputs = torch.randn(3, 2, generator=generator)
        loss = system.local_projection(inputs).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    step()
    payload = {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
               "frozen_state_sha256": frozen_hash(system), "head_sha256": state_hash(head.state_dict()),
               "local_projection": system.local_projection.state_dict(), "head": head.state_dict(),
               "optimizer": optimizer.state_dict(), "rng": capture_rng(generator), "completed_epochs": 1, "step": 1}
    save_checkpoint(tmp_path / "last.pt", payload)
    step()
    expected = cpu_tree(system.local_projection.state_dict())
    loaded = torch.load(tmp_path / "last.pt", weights_only=False)
    restore_checkpoint(loaded, recipe=recipe, system=system, head=head, optimizer=optimizer, generator=generator)
    step()
    for key, value in expected.items():
        torch.testing.assert_close(system.local_projection.state_dict()[key], value, rtol=0, atol=0)
    changed = copy.deepcopy(recipe)
    changed["input_sha256"]["cache"] = "another-input"
    with pytest.raises(ValueError, match="mismatch"):
        restore_checkpoint(loaded, recipe=changed, system=system, head=head, optimizer=optimizer, generator=generator)
    with torch.no_grad():
        system.base.weight.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        restore_checkpoint(loaded, recipe=recipe, system=system, head=head, optimizer=optimizer, generator=generator)


def test_cpu_tree_is_a_snapshot_not_mutable_state_alias():
    source = {"nested": [torch.tensor([1.])], "tuple": (torch.tensor([2.]),)}
    snapshot = cpu_tree(source)
    source["nested"][0].add_(5)
    assert snapshot["nested"][0].item() == 1
    assert isinstance(snapshot["tuple"], tuple)


def test_final_curves_sidecar_binds_adapter_recipe_cache_and_actual_bytes(tmp_path):
    curves = tmp_path / "selected_development_curves.pt"
    torch.save({"motion": torch.randn(1, 3, 2)}, curves)
    identity = write_curve_provenance(curves, recipe_sha256="recipe", selected_checkpoint_sha256="adapter", cache_sha256="cache")
    sidecar = tmp_path / "selected_development_curves.provenance.json"
    assert identity["sha256"] == sha(sidecar)
    assert json.loads(sidecar.read_text(encoding="utf8")) == {
        "schema": CURVE_PROVENANCE_SCHEMA, "curve_sha256": sha(curves),
        "recipe_sha256": "recipe", "selected_checkpoint_sha256": "adapter", "cache_sha256": "cache"}
    previous = json.loads(sidecar.read_text(encoding="utf8"))["curve_sha256"]
    torch.save({"motion": torch.zeros(1, 3, 2)}, curves)
    assert previous != sha(curves)
