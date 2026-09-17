import json

import pytest

from scripts.lock_predictable_motion_data import (
    CELLS, ORIGINAL_SPEAKERS, historical_metadata, lock_selection, unique_native,
)


def fixtures(extra_speakers=2, fresh=20):
    speakers = list(ORIGINAL_SPEAKERS) + [f"mead_X{i:03}" for i in range(extra_speakers)]
    sentences = [f"reference{i}" for i in range(4)] + ["old_train", "old_dev", "old_audit"]
    sentences += [f"fresh{i}" for i in range(fresh)]
    rows = []
    for speaker in speakers:
        for sentence in sentences:
            for emotion, level in CELLS:
                rows.append({"dataset": "mead", "speaker": speaker, "sentence": sentence,
                             "emotion": emotion, "intensity": level, "valid_frames": 60,
                             "clip_id": f"{speaker}_{sentence}_{emotion}_{level}",
                             "artifact": f"{speaker}_{sentence}_{emotion}_{level}.npz", "split": "train"})
    refs = [r for r in rows if r["speaker"] in ORIGINAL_SPEAKERS and
            r["sentence"].startswith("reference") and r["emotion"] == 0]
    history = {"all": set(sentences[:7]), "train": {"old_train"},
               "validation": {"old_dev", "old_audit"}, "enrollment": set(sentences[:4]),
               "clips": {r["clip_id"] for r in rows if r["sentence"] in sentences[:7]},
               "references": refs, "sources": []}
    return rows, history


def test_global_isolation_and_original_identities_are_preserved():
    rows, history = fixtures()
    splits, report = lock_selection(rows, history, speakers=6, max_train=1200)
    assert list(report["speaker_to_id"])[:4] == list(ORIGINAL_SPEAKERS)
    assert report["selected_speakers"] == 6
    assert all(report["isolation_checks"].values())
    assert {r["sentence"] for r in splits["new_test"]}.isdisjoint(history["all"])
    assert {"old_dev", "old_audit"} <= {r["sentence"] for r in splits["validation"]}
    assert "old_train" in {r["sentence"] for r in splits["train"]}
    assert len(splits["enrollment"]) == 24
    assert len(splits["train"]) <= 1200
    assert all(len({r["sentence"] for r in splits["enrollment"] if r["speaker"] == s}) == 4
               for s in report["speaker_to_id"])


def test_no_fresh_test_never_falls_back_to_old_sentences():
    rows, history = fixtures(fresh=0)
    splits, report = lock_selection(rows, history, speakers=6)
    assert splits["new_test"] == []
    assert report["status"] == "no_unexposed_test_sentences"
    assert len(report["coverage"]["new_test"]["missing_cells"]) == 6 * len(CELLS)


def test_incomplete_cells_reported_without_changing_test_allocation():
    rows, history = fixtures()
    full, first = lock_selection(rows, history, speakers=6)
    missing = [r for r in rows if not (r["speaker"] == ORIGINAL_SPEAKERS[0] and
                                     r["emotion"] == 5 and r["sentence"].startswith("fresh"))]
    splits, report = lock_selection(missing, history, speakers=6)
    assert report["sentence_allocation"]["new_test"] == first["sentence_allocation"]["new_test"]
    assert len(report["coverage"]["new_test"]["missing_cells"]) == 2
    assert {r["sentence"] for r in splits["new_test"]}.isdisjoint(history["all"])


def test_stable_selection_ignores_input_order_and_reports_speaker_shortage():
    rows, history = fixtures(extra_speakers=1)
    first, a = lock_selection(rows, history, speakers=12, max_train=81)
    second, b = lock_selection(list(reversed(rows)), history, speakers=12, max_train=81)
    assert first == second
    assert a == b
    assert a["speaker_shortage"] == 7
    assert len(first["train"]) == 81


def test_native_duplicates_and_invalid_metadata():
    rows, _ = fixtures(extra_speakers=0, fresh=1)
    assert unique_native(rows + rows, 32) == rows
    conflict = dict(rows[0], sentence="conflicting_sentence")
    with pytest.raises(ValueError, match="Conflicting"):
        unique_native(rows + [conflict], 32)
    assert unique_native([dict(rows[0], valid_frames=31)], 32) == []


def test_history_reads_sidecars_without_opening_caches(tmp_path, monkeypatch):
    rows, _ = fixtures(extra_speakers=0, fresh=1)
    directory = tmp_path / "old"
    directory.mkdir()
    by_role = {"train": [r for r in rows if r["sentence"] == "old_train"],
               "heldout": [r for r in rows if r["sentence"] == "old_dev"],
               "enrollment": [r for r in rows if r["sentence"].startswith("reference") and r["emotion"] == 0]}
    for role, selected in by_role.items():
        (directory / f"{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    (directory / "train.pt").write_bytes(b"tensor values must never be read")
    (directory / "heldout.pt").write_bytes(b"tensor values must never be read")
    monkeypatch.setattr("scripts.lock_predictable_motion_data.cache_metadata",
                        lambda path: pytest.fail("Sidecars should avoid cache loading"))
    history = historical_metadata([directory])
    assert history["train"] == {"old_train"}
    assert history["validation"] == {"old_dev"}
    assert history["enrollment"] == {f"reference{i}" for i in range(4)}


def test_cache_fallback_reads_strings_without_tensor_values(tmp_path):
    import torch
    from scripts.lock_predictable_motion_data import cache_metadata
    path = tmp_path / "heldout.pt"
    query = {"clip_id": "q", "sentence_id": "query", "speaker": "mead_M003",
             "motion": torch.full((6, 52), float("nan")), "emotion_id": torch.tensor(5)}
    ref = {"clip_id": "r", "sentence_id": "ref", "speaker": "mead_M003", "motion": torch.randn(3, 52)}
    torch.save({"queries": [query], "identity_references": {0: [ref]}}, path)
    queries, refs = cache_metadata(path)
    assert queries == [{"clip_id": "q", "sentence_id": "query", "speaker": "mead_M003"}]
    assert refs == [{"clip_id": "r", "sentence_id": "ref", "speaker": "mead_M003"}]
    history = historical_metadata([path])
    assert history["all"] == {"query", "ref"}
    assert history["validation"] == {"query"}


def test_past_audit_summary_is_exposure_even_without_cache(tmp_path):
    summary = tmp_path / "selection.json"
    summary.write_text(json.dumps({"excluded_sentence_ids": ["old_train"], "selected_sentence_ids": ["old_audit"]}))
    history = historical_metadata([summary])
    assert history["all"] == {"old_train", "old_audit"}
    assert history["validation"] == {"old_audit"}
