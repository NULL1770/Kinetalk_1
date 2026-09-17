import copy
import gc
import json

import pytest
import torch
from torch import nn

from scripts.audit_direct_audio_dynamics import (
    MODES, NOISE_SEEDS, load_arm, load_historical_source_curves,
)
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import sha, state_hash
from tests.test_audio_conditioned_flow_audit import fixture_curves


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


def sidecar(run, stem, recipe):
    path = run / f"{stem}_curves.provenance.json"
    write_json(path, {"schema": "projection_schedule_curves_provenance_v1",
        "curve_sha256": sha(run / f"{stem}_curves.pt"),
        "checkpoint_sha256": sha(run / f"{stem}.pt"),
        "recipe_sha256": canonical_hash(recipe), "cache_sha256": recipe["input_sha256"]["cache"]})
    return {"path": str(path.resolve()), "sha256": sha(path)}


def test_loader_reads_actual_separate_epoch0_report(tmp_path):
    recipe = {"schema": "direct_audio_dynamics_v1", "arm": "flow", "seed": 46,
        "epochs": 8, "batch_size": 16, "optimizer": "Adam", "renderer_lr": 1e-5,
        "encoder_lr": 1e-4, "clip_grad_norm": 1., "decode_steps": 12,
        "eval_noise_seeds": list(NOISE_SEEDS), "eval_modes": list(MODES),
        "loss_weights": {"centered_upper": 1.}, "default_replaced": False,
        "test_loaded": False, "checkpoint_selection_performed": False,
        "input_sha256": {"cache": "test-cache"}, "protected_sha256": "protected"}
    state, scales = {"weight": torch.ones(1)}, {"dynamic": torch.ones(52)}
    recipe["motion_scales_sha256"] = state_hash(scales)
    digest = canonical_hash(recipe)
    write_json(tmp_path / "provenance.json", {"recipe": recipe, "recipe_sha256": digest})
    for epoch, stem in ((0, "epoch000"), (8, "final_epoch008")):
        payload = {"schema": recipe["schema"], "recipe": recipe, "recipe_sha256": digest,
            "completed_epochs": epoch, "step": epoch * 145, "protected_sha256": "protected",
            "renderer": state, "encoder": state, "renderer_sha256": state_hash(state),
            "encoder_sha256": state_hash(state), "motion_scales": scales}
        torch.save(payload, tmp_path / f"{stem}.pt")
    torch.save(payload, tmp_path / "last.pt")
    _, all_curves = fixture_curves()
    final = {**all_curves, "motion": {seed: {m: rows[m] for m in MODES}
             for seed, rows in all_curves["motion"].items()}}
    epoch0 = {**final, "noise_seeds": [42], "motion": {"42": final["motion"]["42"]}}
    torch.save(final, tmp_path / "final_epoch008_curves.pt")
    torch.save(epoch0, tmp_path / "epoch000_curves.pt")
    initial_report = {"42": {mode: {"sentinel": 17} for mode in MODES}}
    write_json(tmp_path / "epoch000_evaluation.json", initial_report)
    for epoch in range(1, 9):
        write_json(tmp_path / f"epoch{epoch:03d}.json", {"epoch": epoch})
    summary = {"schema": recipe["schema"], "arm": "flow", "recipe_sha256": digest,
        "completed_epochs": 8, "optimizer_steps": 1160, "protected_unchanged": True,
        "test_loaded": False, "default_replaced": False, "checkpoint_selection_performed": False,
        "final": {}, "curve_provenance": sidecar(tmp_path, "final_epoch008", recipe)}
    write_json(tmp_path / "summary.json", summary)
    write_json(tmp_path / "output_hashes.json", {p.name: sha(p) for p in tmp_path.iterdir()})
    loaded = load_arm(tmp_path, "flow")
    assert "epoch000" not in loaded["summary"]
    assert loaded["epoch0_reports"] == initial_report


def historical_fixture(tmp_path):
    system = nn.Module()
    system.renderer = nn.Linear(2, 2)
    system.local_projection = nn.Linear(2, 3, bias=False)
    head = nn.Linear(2, 2)
    loaded = {"system": system, "head": head}
    direct = {"input_sha256": {"cache": "source-cache"}, "source_adapter_sha256": "adapter",
              "initial_system_sha256": state_hash(system.state_dict())}
    recipe = {**direct, "schema": "audio_conditioned_flow_probe_v1",
        "initial_head_sha256": state_hash(head.state_dict()), "decode_steps": 12,
        "eval_noise_seeds": list(NOISE_SEEDS), "eval_modes": ["full", "zero", "reverse", "oracle"]}
    digest = canonical_hash(recipe)
    write_json(tmp_path / "provenance.json", {"recipe": recipe, "recipe_sha256": digest})
    name = "renderer.weight"
    checkpoint = {"schema": recipe["schema"], "recipe": recipe, "recipe_sha256": digest,
        "completed_epochs": 0, "step": 0, "trainable_name": name,
        "trainable": {name: system.renderer.weight},
        "frozen_state_sha256": state_hash({k: v for k, v in system.state_dict().items() if k != name}),
        "local_projection": system.local_projection.state_dict(),
        "head": head.state_dict(), "head_sha256": state_hash(head.state_dict())}
    torch.save(checkpoint, tmp_path / "epoch000.pt")
    reference, curves = fixture_curves()
    path = tmp_path / "epoch000_curves.pt"
    torch.save(curves, path)
    write_json(tmp_path / "epoch000_evaluation.json",
               {"curve_provenance": sidecar(tmp_path, "epoch000", recipe)})
    return path, loaded, direct, reference


def test_historical_baseline_is_bound_to_source_weights_and_all_seed_curves(tmp_path):
    path, loaded, recipe, reference = historical_fixture(tmp_path)
    curves, evidence = load_historical_source_curves(path, loaded, recipe, reference)
    assert evidence["source_system_and_cache_bound"]
    # Changing an unrelated seed's full output used to pass the seed42-zero
    # check. The curve/checkpoint sidecar now rejects that substitution.
    changed = copy.deepcopy(curves)
    changed["motion"]["997"]["full"] += .5
    del curves
    gc.collect()  # Windows cannot overwrite a file while its mmap is alive.
    torch.save(changed, path)
    with pytest.raises(ValueError, match="curve/checkpoint/cache"):
        load_historical_source_curves(path, loaded, recipe, reference)


def test_historical_baseline_rejects_another_cache_even_with_valid_files(tmp_path):
    path, loaded, recipe, reference = historical_fixture(tmp_path)
    different = copy.deepcopy(recipe)
    different["input_sha256"]["cache"] = "another-cache"
    with pytest.raises(ValueError, match="different source/cache"):
        load_historical_source_curves(path, loaded, different, reference)


def test_historical_baseline_rejects_a_mutated_source_weight(tmp_path):
    path, loaded, recipe, reference = historical_fixture(tmp_path)
    with torch.no_grad():
        loaded["system"].renderer.bias.add_(1.)
    with pytest.raises(ValueError, match="source system differs"):
        load_historical_source_curves(path, loaded, recipe, reference)
