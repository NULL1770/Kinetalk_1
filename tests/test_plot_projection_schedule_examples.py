"""Preselected examples remain metadata-only and coefficient provenance is bound."""
import copy
import json

import pytest
import torch

from scripts.plot_projection_schedule_examples import (
    CHANNELS, SPEAKERS, canonical_hash, centered_residual, load_final_curves,
    make_lock, plot_locked_examples, select_examples, sha, write_new_json,
)


def metadata():
    rows = [{"clip_id": f"{speaker}_clip{i:03}", "speaker": speaker, "speaker_id": sid,
             "emotion_id": (0, 1, 5, 6)[i % 4], "sentence": f"sentence{i % 17}",
             "dataset": "mead", "split": "train"}
            for sid, speaker in enumerate(SPEAKERS) for i in range(135)]
    lock = {"schema": "projection_schedule_internal_split_v1", "source_role": "formal_training_queries_only",
            "outer_development_loaded": False, "new_identity_development_loaded": False,
            "outer_development_target_values_accessed": False, "test_target_values_accessed": False,
            "heldout_speakers": SPEAKERS, "fit_speakers": ["mead_M001"], "fit_clip_ids": ["mead_M001_train"],
            "validation_clip_ids": [row["clip_id"] for row in rows]}
    return rows, lock


def test_hash_selection_is_metadata_only_and_order_invariant():
    rows, lock = metadata()
    result = select_examples(rows, lock)
    assert len(result) == 9
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(target=float("nan"), prediction=float("inf"), quality=-100)
    assert result == select_examples(changed, lock)
    changed.reverse()
    reverse_lock = {**lock, "validation_clip_ids": [row["clip_id"] for row in changed]}
    assert [p["clip_id"] for p in result] == [p["clip_id"] for p in select_examples(changed, reverse_lock)]
    with pytest.raises(ValueError, match="validation405"):
        select_examples(rows[:-1], lock)
    with pytest.raises(ValueError, match="exposure"):
        select_examples(rows, {**lock, "test_target_values_accessed": True})


def test_lock_does_not_deserialize_tensors_or_predictions(tmp_path, monkeypatch):
    rows, lock = metadata()
    data = tmp_path / "data_locked"
    data.mkdir()
    (data / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    write_new_json(data / "split_lock.json", lock)
    (data / "renderer_cache.pt").write_bytes(b"not even a serialized tensor: byte-hash only")
    def forbidden(*args, **kwargs):
        raise AssertionError("Selection opened tensors")
    monkeypatch.setattr(torch, "load", forbidden)
    value = make_lock(data)
    assert len(value["clips"]) == 9
    assert value["tensor_values_deserialized_during_selection"] is False
    assert value["source_sha256"]["cache"] == sha(data / "renderer_cache.pt")


def test_centering_removes_cached_baseline_and_keeps_invalid_gaps():
    valid = torch.tensor([[True, True, False, True]])
    baseline = torch.tensor([1., 4., 9., 16.])[None, :, None].expand(1, 4, 52)
    residual = torch.tensor([2., 3., float("nan"), 10.])[None, :, None].expand(1, 4, 52)
    identity = torch.ones(1, 52) * .7
    target = baseline + identity[:, None] + residual
    split = {"q": {"motion": target, "valid": valid, "channel_mask": torch.ones(1, 52, dtype=torch.bool)},
             "base": {"b0": baseline}, "identity": {"baseline": identity}}
    actual = centered_residual(target, split)
    assert torch.allclose(actual[0, valid[0], 0], torch.tensor([-3., -2., 5.]).double(), atol=2e-6)
    assert actual[0, 2].eq(0).all()


def create_run(run, recipe, curves):
    run.mkdir()
    digest = canonical_hash(recipe)
    write_new_json(run / "provenance.json", {"recipe": recipe, "recipe_sha256": digest})
    checkpoint_path, curve_path = run / "final_epoch018.pt", run / "final_epoch018_curves.pt"
    torch.save({"recipe": recipe, "recipe_sha256": digest, "completed_epochs": 18, "selection": "none"}, checkpoint_path)
    torch.save({"noise_seeds": [42, 123, 2026], "decode_steps": 12,
                "motion": {str(seed): curves for seed in (42, 123, 2026)}}, curve_path)
    sidecar = run / "final_epoch018_curves.provenance.json"
    write_new_json(sidecar, {"schema": "projection_schedule_curves_provenance_v1", "curve_sha256": sha(curve_path),
        "checkpoint_sha256": sha(checkpoint_path), "recipe_sha256": digest, "cache_sha256": recipe["input_sha256"]["cache"]})
    write_new_json(run / "summary.json", {"schema": recipe["schema"], "arm": recipe["args"]["arm"],
        "recipe_sha256": digest, "completed_epochs": 18, "outer280_loaded": False, "new_identity439_loaded": False,
        "test_loaded": False, "checkpoint_selection_performed": False, "frozen_unchanged": True,
        "head_unchanged": True, "curve_provenance": {"sha256": sha(sidecar)}})


def test_full_plot_uses_nine_locked_clips_and_rejects_changed_curves(tmp_path):
    rows, split_lock = metadata()
    data = tmp_path / "data_locked"
    data.mkdir()
    (data / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    write_new_json(data / "split_lock.json", split_lock)
    motion = torch.randn(405, 6, 52, generator=torch.Generator().manual_seed(51))
    valid = torch.ones(405, 6, dtype=torch.bool)
    valid[:, 2] = False
    split = {"q": {"motion": motion, "valid": valid, "channel_mask": torch.ones(405, 52, dtype=torch.bool),
        "times": torch.arange(6)[None].expand(405, -1).double() / 25,
        "speaker_id": torch.tensor([r["speaker_id"] for r in rows]),
        "emotion_id": torch.tensor([r["emotion_id"] for r in rows]),
        "speaker": [r["speaker"] for r in rows], "clip_id": [r["clip_id"] for r in rows],
        "sentence_id": [r["sentence"] for r in rows]},
        "base": {"b0": motion * .2}, "identity": {"baseline": torch.ones(405, 52) * .1}}
    cache_path = data / "renderer_cache.pt"
    torch.save({"schema": "predictable_renderer_cache_v1", "splits": {"validation": split},
        "provenance": {"internal_split_lock_sha256": canonical_hash(split_lock),
                       "manifest_hashes": {"validation": sha(data / "validation.jsonl")}}}, cache_path)
    lock = make_lock(data)
    lock_path = tmp_path / "lock.json"
    write_new_json(lock_path, lock)
    flow = {"schema": "projection_schedule_ablation_v1", "loss": "observed_flow_mse_only",
        "args": {"arm": "constant_teacher", "output": "flow", "epochs": 18, "seed": 46},
        "input_sha256": {"cache": sha(cache_path), "split_lock": sha(data / "split_lock.json")},
        "source_sha256": {"/repo/scripts/train_projection_schedule_ablation.py": "same"},
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False}
    rollout = copy.deepcopy(flow)
    rollout.update(schema="projection_rollout_probe_v1", loss="observed_12step_rollout_motion_mse_divided_by_residual_scale_squared_only",
                   teacher_probability=.5, decode_steps=12,
                   flow_time_draws="consumed and hashed identically to constant_teacher, unused in rollout loss")
    rollout["args"].update(arm="constant_teacher_rollout", output="rollout")
    rollout["source_sha256"]["/repo/scripts/train_projection_rollout_probe.py"] = "new"
    curves = {"zero": motion * .4, "full": motion * .6, "oracle": motion * .8}
    create_run(tmp_path / "flow", flow, curves)
    create_run(tmp_path / "rollout", rollout, curves)
    report = plot_locked_examples(lock_path, cache_path, tmp_path / "flow", tmp_path / "rollout", tmp_path / "plots")
    assert len(report["files"]) == 9
    assert (tmp_path / "plots" / "montage.png").exists()
    assert report["zero_local_equal_exactly"]
    assert set(report["channel_y_limits_shared_across_clips"]) == {c for c, _ in CHANNELS}
    with (tmp_path / "rollout" / "final_epoch018_curves.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="binding mismatch"):
        load_final_curves(tmp_path / "rollout", lock, rollout=True)
