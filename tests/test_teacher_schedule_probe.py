import copy

import pytest
import torch

from scripts.prepare_teacher_schedule_probe import (
    choose_identity_split, fit_training_only, take_cache_split, take_bundle,
)


def metadata():
    rows, references = [], []
    mapping = {f"mead_P{sid:03}": sid for sid in range(6)}
    for name, sid in mapping.items():
        for e in (0, 1, 5, 6):
            for sentence in range(4):
                rows.append({"dataset": "mead", "split": "train", "speaker": name, "speaker_id": sid,
                             "emotion_id": e, "emotion": e, "clip_id": f"{name}_{e}_{sentence}",
                             "sentence": f"script{sentence}", "artifact_sha256": "queryhash"})
        for reference in range(2):
            references.append({"dataset": "mead", "split": "train", "speaker": name, "speaker_id": sid,
                "emotion_id": 0, "emotion": 0, "clip_id": f"{name}_ref{reference}",
                "sentence": f"reference{reference}", "artifact_sha256": "referencehash"})
    selection = {"schema": "formal_predictable_metadata_lock_v1", "train_speaker_to_id": mapping,
                 "min_references": 2, "reference_counts": {"train": {speaker: 2 for speaker in mapping}}}
    return rows, references, selection


def synthetic_bundle(rows):
    torch.manual_seed(44)
    count = len(rows)
    return {"clip_id": [r["clip_id"] for r in rows], "sentence_id": [r["sentence"] for r in rows],
            "speaker_id": torch.tensor([r["speaker_id"] for r in rows]),
            "emotion_id": torch.tensor([r["emotion_id"] for r in rows]),
            "features": {"content": torch.randn(count, 5, 4), "middle": torch.randn(count, 5, 3),
                         "prosody": torch.randn(count, 5, 2)},
            "motion_bins": torch.randn(count, 5, 52), "weight": torch.ones(count, 5),
            "channel_mask": torch.ones(count, 52, dtype=torch.bool), "groups": {"untrusted_source_group": []}}


def test_identity_split_is_metadata_deterministic_disjoint_and_full_partition():
    rows, refs, selection = metadata()
    fit, heldout, report = choose_identity_split(rows, refs, selection)
    same_fit, same_heldout, same = choose_identity_split(rows, refs, selection)
    assert torch.equal(fit, same_fit) and torch.equal(heldout, same_heldout) and report == same
    assert all(report["checks"].values())
    assert len(report["roles"]["validation"]["speaker_to_id"]) == 3
    assert set(report["roles"]["train"]["speaker_to_id"]).isdisjoint(report["roles"]["validation"]["speaker_to_id"])
    assert report["shared_sentence_count"] == 4
    # A different input row order cannot change hash-selected identity names.
    _, _, flipped = choose_identity_split(list(reversed(rows)), list(reversed(refs)), selection)
    assert report["roles"]["validation"]["speaker_to_id"] == flipped["roles"]["validation"]["speaker_to_id"]


def test_heldout_nan_targets_features_weights_cannot_change_basis_alpha_scales_or_head():
    rows, refs, selection = metadata()
    fit, heldout, _ = choose_identity_split(rows, refs, selection)
    source = synthetic_bundle(rows)
    first = fit_training_only(source, fit, alphas=(.1, 1.), seed=45)
    changed = copy.deepcopy(source)
    changed["motion_bins"][heldout] = float("nan")
    changed["weight"][heldout] = float("nan")
    changed["channel_mask"][heldout] = False
    changed["groups"] = {"all_expression": [17]}
    for values in changed["features"].values():
        values[heldout] = float("nan")
    second = fit_training_only(changed, fit, alphas=(.1, 1.), seed=45)
    assert first[3] == second[3]  # Every inner score and selected alpha.
    for key in first[1]:
        for field in ("std", "basis", "weights", "train_ids"):
            torch.testing.assert_close(first[1][key][field], second[1][key][field], atol=0, rtol=0)
        assert first[1][key]["alpha"] == second[1][key]["alpha"]
    for method in first[2]:
        assert first[2][method]["sha256"] == second[2][method]["sha256"]
        for key, value in first[2][method]["state_dict"].items():
            torch.testing.assert_close(value, second[2][method]["state_dict"][key], atol=0, rtol=0)
    for fold in first[3]["folds"]:
        assert set(fold["fit_sentences"]).isdisjoint(fold["validation_sentences"])


def test_cache_values_are_copied_exactly_without_recomputing_base_or_global():
    ids = torch.tensor([4, 1])
    source = {group: {"value": torch.arange(30, dtype=torch.float32).reshape(5, 6), "clip_id": list("abcde")}
              for group in ("q", "base", "identity", "affect")}
    result = take_cache_split(source, ids)
    for group in source:
        assert torch.equal(result[group]["value"], source[group]["value"][ids])
        assert result[group]["clip_id"] == ["e", "b"]
        assert result[group]["value"].data_ptr() != source[group]["value"].data_ptr()


def test_identity_split_rejects_outer_source_and_reference_overlap():
    rows, refs, selection = metadata()
    rows[0]["split"] = "val"
    with pytest.raises(ValueError, match="native-train"):
        choose_identity_split(rows, refs, selection)
    rows, refs, selection = metadata()
    refs[0]["sentence"] = "script0"
    with pytest.raises(ValueError, match="globally sentence"):
        choose_identity_split(rows, refs, selection)


def test_output_lineage_contract_is_accepted_by_schedule_runner():
    from scripts.train_projection_schedule_ablation import validate_internal_split
    from scripts.train_formal_predictable_projection import canonical_hash
    rows, references, selection = metadata()
    fit, hold, report = choose_identity_split(rows, references, selection)
    source = synthetic_bundle(rows)
    fit_bundle, hold_bundle = take_bundle(source, fit), take_bundle(source, hold)
    queries = {}
    for role, ids in (("train", fit), ("validation", hold)):
        queries[role] = {"q": {"clip_id": [rows[int(i)]["clip_id"] for i in ids],
            "sentence_id": [rows[int(i)]["sentence"] for i in ids],
            "speaker": [rows[int(i)]["speaker"] for i in ids]}}
    lock = {"schema": "projection_schedule_internal_split_v1", "source_role": "formal_training_queries_only",
            "outer_development_loaded": False, "new_identity_development_loaded": False,
            "fit_clip_ids": queries["train"]["q"]["clip_id"], "validation_clip_ids": queries["validation"]["q"]["clip_id"],
            "heldout_speakers": sorted(report["roles"]["validation"]["speaker_to_id"])}
    prepared = {"bundles": {"internal": fit_bundle, "external_dev": hold_bundle},
                "provenance": {"internal_split_lock_sha256": canonical_hash(lock)}}
    scope = validate_internal_split({"splits": queries}, prepared, lock, lock["fit_clip_ids"], lock["validation_clip_ids"])
    assert scope["fit_clips"] + scope["validation_clips"] == len(rows)
    assert len(scope["heldout_speakers"]) == 3
