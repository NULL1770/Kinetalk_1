import copy
import json

import numpy as np
import pytest
import torch
import yaml

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.full_staged_data import _references, load_training_inputs, sha
from scripts.prepare_expression_intensity_targets import prepare_targets
from scripts.prepare_label_guided_audio_cache import fit_feature_statistics
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import state_hash


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path):
    torch.manual_seed(14)
    native = tmp_path / "native"; native.mkdir()
    source = tmp_path / "source"; source.mkdir()
    config = {"data": {"content_dim": 768, "motion_dim": 52, "neutral_output_indices": [17, 18],
                       "emotion_classes": ["neutral", "angry", "happy", "sad"], "num_intensity_levels": 4,
                       "audio_emotion_dim": 768},
              "model": {"content_dim": 8, "emotion_dim": 64, "style_dim": 8, "hidden_dim": 8,
                        "heads": 2, "dropout": 0., "dit_dim": 8, "dit_depth": 1,
                        "residual_scale": .25, "affect_stride": 4, "affect_rank": 8}}
    system = NeutralAffectSystem(config).eval()
    paths = {k: tmp_path / (k + (".yaml" if k == "config" else ".json" if k in ("fit_ids", "validation_ids", "split_lock") else ".pt"))
             for k in ("checkpoint", "config", "cache", "bundle", "weights", "fit_ids", "validation_ids", "split_lock")}
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf8")
    torch.save({"model": system.state_dict(), "config": config}, paths["checkpoint"])
    torch.save({"unused": True}, paths["bundle"]); torch.save({"unused": True}, paths["weights"])
    native_rows, cache_splits, audio_splits = [], {}, {}
    for role, sids in (("train", [7, 17]), ("validation", [29])):
        n = len(sids); valid = torch.ones(n, 32, dtype=torch.bool); valid[:, -1] = False
        features = torch.randn(n, 32, 1540).masked_fill(~valid[..., None], 0.)
        times = torch.arange(32, dtype=torch.float64)[None].expand(n, -1) / 25
        q = {"motion": .3 + .02 * torch.randn(n, 32, 52), "content": features[..., :768].half(),
             "valid": valid, "times": times, "channel_mask": torch.ones(n, 52, dtype=torch.bool),
             "clip_id": [role + str(s) for s in sids], "speaker": ["person" + str(s) for s in sids],
             "speaker_id": torch.tensor(sids), "sentence_id": ["query" + str(s) for s in sids],
             "emotion_id": torch.ones(n, dtype=torch.long), "intensity_id": torch.ones(n, dtype=torch.long),
             "intensity_valid": torch.ones(n, dtype=torch.bool)}
        cache_splits[role] = {"q": q, "base": {"bad": torch.tensor(float("nan"))},
                              "identity": {"bad": torch.tensor(float("nan"))}, "affect": {"bad": torch.tensor(float("nan"))}}
        audio_splits[role] = {"features": features, "times": times.clone(), "valid": valid.clone(),
                              "clip_id": list(q["clip_id"]), "sentence_id": list(q["sentence_id"])}
        for sid in sids:
            for j in range(2):
                path = native / f"ref{sid}_{j}.npz"
                np.savez(path, motion=np.full((32, 52), .1 + j * .1, dtype=np.float32),
                         content=np.zeros((32, 768), dtype=np.float32), audio=np.zeros((32, 83), dtype=np.float32),
                         times=np.arange(32, dtype=np.float64) / 25, mask=np.ones(32, dtype=bool),
                         channel_mask=np.ones(52, dtype=bool),
                         provenance=np.asarray(json.dumps({"schema": "native_affect_style_v4.1", "clock_evidence": "embedded_video", "fps": 25})))
                native_rows.append({"clip_id": f"ref{sid}_{j}", "speaker": f"person{sid}", "speaker_id": sid,
                                    "sentence": f"ref_sentence{sid}_{j}", "split": "train", "emotion": 0,
                                    "artifact": path.name, "artifact_sha256": sha(path), "intensity": 0})
    fit_ids = cache_splits["train"]["q"]["clip_id"]; dev_ids = cache_splits["validation"]["q"]["clip_id"]
    lock = {"schema": "projection_schedule_internal_split_v1", "source_role": "formal_training_queries_only",
            "outer_development_loaded": False, "new_identity_development_loaded": False,
            "fit_clip_ids": fit_ids, "validation_clip_ids": dev_ids, "heldout_speakers": ["person29"]}
    paths["split_lock"].write_text(json.dumps(lock), encoding="utf8")
    for key, ids in (("fit_ids", fit_ids), ("validation_ids", dev_ids)):
        paths[key].write_text(json.dumps({"clip_ids": ids}), encoding="utf8")
    cache = {"schema": "predictable_renderer_cache_v1", "splits": cache_splits,
             "provenance": {**{k + "_sha256": sha(paths[k]) for k in ("checkpoint", "config", "bundle")},
                            "internal_split_lock_sha256": canonical_hash(lock)}}
    torch.save(cache, paths["cache"])
    enrollment = tmp_path / "enrollment.jsonl"
    enrollment.write_text("\n".join(map(json.dumps, native_rows)), encoding="utf8")
    target_output = tmp_path / "target"
    targets = prepare_targets(paths["cache"], enrollment, native, target_output)
    audio = {"schema": "label_guided_audio_cache_v1", "splits": audio_splits,
             "provenance": {"renderer_cache_sha256": sha(paths["cache"])},
             "feature_stats": fit_feature_statistics(audio_splits["train"]["features"], audio_splits["train"]["valid"], fit_ids)}
    audio_path = tmp_path / "audio.pt"; torch.save(audio, audio_path)
    recipe = {"schema": "projection_scaled_centered_probe_v1", "args": {**{k: str(v) for k, v in paths.items()}, "arm": "uniform", "epochs": 8},
              "input_sha256": {k: sha(p) for k, p in paths.items()}}
    (source / "provenance.json").write_text(json.dumps({"recipe": recipe, "recipe_sha256": canonical_hash(recipe)}), encoding="utf8")
    projection = {k: torch.full_like(v, .333) for k, v in system.local_projection.state_dict().items()}
    torch.save({"schema": recipe["schema"], "recipe": recipe, "recipe_sha256": canonical_hash(recipe), "completed_epochs": 8,
                "local_projection": projection,
                "frozen_state_sha256": state_hash({k: v for k, v in system.state_dict().items() if not k.startswith("local_projection.")})},
               source / "final_epoch008.pt")
    args = (source, audio_path, target_output / "targets.pt", enrollment, native)
    return args, paths, cache, audio, targets, native_rows


def test_load_reconstructs_inputs_without_stale_latents_and_maps_two_refs(tmp_path):
    args, paths, _, audio, _, _ = fixture(tmp_path)
    loaded = load_training_inputs(*args)
    assert loaded["fit_sids"] == [7, 17] and loaded["dev_sids"] == [29]
    assert set(loaded["refs"]) == {7, 17, 29}
    assert all(not p.requires_grad for p in loaded["system"].parameters())
    original = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)["model"]
    assert state_hash(loaded["system"].state_dict()) == state_hash(original)
    assert not torch.all(loaded["system"].local_projection.weight == .333)
    assert loaded["provenance"]["uniform_projection_loaded"] is False
    for role, q in loaded["splits"].items():
        assert not ({"base", "identity", "affect"} & set(q))
        assert q["content"].dtype == torch.float32
        torch.testing.assert_close(q["content"], audio["splits"][role]["features"][..., :768], rtol=0, atol=0)
        assert q["intensity_valid"].shape == (len(q["valid"]),)  # Class-level field is preserved.
        assert q["target_intensity_valid"].shape == (*q["valid"].shape, 1)
    for sid, group in loaded["ref_groups"].items():
        assert group["a"].tolist() == [0] and group["b"].tolist() == [1]
        assert loaded["refs"][sid]["content"].shape == (2, 32, 768)
    assert loaded["provenance"]["encoded_cache_groups_used"] == []
    assert loaded["provenance"]["test_loaded"] is False


@pytest.mark.parametrize("change", ["content", "clock", "clip", "statistics", "cache_binding"])
def test_rejects_audio_contract_changes(tmp_path, change):
    args, _, _, audio, _, _ = fixture(tmp_path)
    a = audio["splits"]["train"]
    if change == "content": a["features"][0, 0, 0] += 5
    if change == "clock": a["times"][0, 0] += .01
    if change == "clip": a["clip_id"].reverse()
    if change == "statistics": audio["feature_stats"]["fit_clip_ids"] = ["wrong"]
    if change == "cache_binding": audio["provenance"]["renderer_cache_sha256"] = "wrong"
    torch.save(audio, args[1])
    with pytest.raises(ValueError): load_training_inputs(*args)


def test_source_hash_and_target_fit_statistics_are_checked(tmp_path):
    args, paths, _, _, targets, _ = fixture(tmp_path)
    targets["scales"] *= 2; torch.save(targets, args[2])
    with pytest.raises(ValueError, match="Intensity supervision|scales"):
        load_training_inputs(*args)
    with paths["checkpoint"].open("ab") as handle: handle.write(b"changed")
    with pytest.raises(ValueError, match="input SHA256"):
        load_training_inputs(*args)


def test_original_target_source_sha_layout_accepted_but_conflicting_binding_rejected(tmp_path):
    args, _, _, _, targets, _ = fixture(tmp_path)
    del targets["provenance"]["renderer_cache_sha256"]
    torch.save(targets, args[2])
    loaded = load_training_inputs(*args)
    assert loaded["provenance"]["test_loaded"] is False
    targets["provenance"]["renderer_cache_sha256"] = "conflicting"
    alternate = tmp_path / "conflicting_targets.pt"
    torch.save(targets, alternate)
    with pytest.raises(ValueError, match="conflicting"):
        load_training_inputs(args[0], args[1], alternate, args[3], args[4])


@pytest.mark.parametrize("change", ["test", "emotion", "query_sentence", "outside", "duplicate", "hash"])
def test_enrollment_is_independent_native_train_only(tmp_path, change):
    args, _, cache, _, targets, rows = fixture(tmp_path)
    if change == "test": rows[0]["split"] = "test"
    if change == "emotion": rows[0]["emotion"] = 1
    if change == "query_sentence": rows[0]["sentence"] = "query7"
    if change == "outside": rows[0]["artifact"] = "../elsewhere.npz"
    if change == "duplicate": rows[1]["sentence"] = rows[0]["sentence"]
    if change == "hash": rows[0]["artifact_sha256"] = "bad"
    args[3].write_text("\n".join(map(json.dumps, rows)), encoding="utf8")
    with pytest.raises(ValueError): _references(args[3], args[4], cache, targets, window=32)


def test_extra_test_split_is_rejected_without_reading_tensors(tmp_path):
    args, paths, cache, _, _, _ = fixture(tmp_path)
    cache["splits"]["test"] = {"forbidden": True}
    torch.save(cache, paths["cache"])
    # The source hash catches even container changes before split tensor access.
    with pytest.raises(ValueError, match="input SHA256"):
        load_training_inputs(*args)
