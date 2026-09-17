import json
from pathlib import Path

import pytest

from scripts.lock_formal_predictable_data import (
    CELLS, collect_historical_development, main, select_formal, sha,
)


def make_fixture():
    sources = {"train": [], "val": [], "test": []}

    def row(split, speaker, sentence, emotion=0, intensity=0):
        clip = f"{speaker}_{sentence}_{emotion}_{intensity}"
        value = {"clip_id": clip, "dataset": "mead", "speaker": speaker, "sentence": sentence,
                 "emotion": emotion, "intensity": intensity, "split": split, "artifact": f"missing/{clip}.npz",
                 "artifact_sha256": "hash-" + clip, "valid_frames": 40, "stage1": False}
        sources[split].append(value)
        return value

    references = {}
    for split, speaker, count in (("train", "mead_A", 4), ("train", "mead_B", 2),
                                  ("train", "mead_C", 1), ("val", "mead_V", 2), ("test", "mead_T", 2)):
        references[speaker] = [row(split, speaker, f"ref{i}") for i in range(count)]
        for sentence in ("old_train", "old_dev", "old_extra_dev", "reserved0", "reserved1", "reserved2", "fresh"):
            for emotion, intensity in sorted(CELLS):
                row(split, speaker, sentence, emotion, intensity)
    old_queries = [r for r in sources["train"] if r["speaker"] == "mead_A"]

    def tagged(rows, role):
        return [dict(r, speaker_id=3, emotion_id=r["emotion"], intensity_id=r["intensity"], pilot_split=role) for r in rows]

    locked = {"train": tagged([r for r in old_queries if r["sentence"] == "old_train"], "train"),
              "validation": tagged([r for r in old_queries if r["sentence"] == "old_dev"], "validation"),
              "enrollment": tagged(references["mead_A"], "enrollment"),
              "new_test": tagged([r for r in old_queries if r["sentence"].startswith("reserved")], "new_test")}
    lock_info = {"speaker_to_id": {"mead_A": 3}, "sentence_allocation": {
        "enrollment": [f"ref{i}" for i in range(4)], "validation": ["old_dev"],
        "new_test": [f"reserved{i}" for i in range(3)]}}
    return sources, locked, lock_info


def test_formal_expansion_keeps_roles_and_native_emotion_ids():
    sources, locked, info = make_fixture()
    outputs, report = select_formal(sources, locked, info, extra_dev_sentences={"old_extra_dev"})
    assert report["speaker_to_id"]["mead_A"] == 3
    assert report["speaker_to_id"]["mead_B"] == 4
    assert report["speaker_to_id"]["mead_V"] == 5
    assert report["speaker_to_id"]["mead_T"] == 6
    assert "mead_C" not in report["speaker_to_id"]
    assert report["excluded_identities"][0]["speaker"] == "mead_C"
    assert report["reference_counts"]["train"] == {"mead_A": 4, "mead_B": 2}
    assert all(report["isolation_checks"].values())
    assert outputs["validation"] == locked["validation"]
    assert {r["sentence"] for r in outputs["train"]} == {"old_train", "fresh"}
    assert {r["emotion_id"] for r in outputs["train"]} == {0, 1, 5, 6}
    assert {r["split"] for r in outputs["train"]} == {"train"}
    assert {r["split"] for r in outputs["new_identity_validation"]} == {"val"}
    assert {r["split"] for r in outputs["sealed_test_metadata"]} == {"test"}
    # Shared scripts are legal only for the declared cross-identity roles.
    assert report["cross_identity_sentence_overlap"]["new_identity_validation"] == 2
    assert report["test_targets_read"] is False


def test_insufficient_new_identity_never_borrows_development_for_references():
    sources, locked, info = make_fixture()
    outputs, report = select_formal(sources, locked, info)
    assert not any(r["speaker"] == "mead_C" for r in outputs["train"] + outputs["enrollment"])
    assert not any(r["sentence"] == "old_dev" for r in outputs["enrollment"])
    assert report["excluded_identities"][0]["available_references"] == 1


def test_old_train_cannot_silently_become_development():
    sources, locked, info = make_fixture()
    with pytest.raises(ValueError, match="reassign a prior training"):
        select_formal(sources, locked, info, extra_dev_sentences={"old_train"})


def test_native_duplicate_or_identity_overlap_fails():
    sources, locked, info = make_fixture()
    sources["train"].append(dict(sources["train"][0], sentence="conflict"))
    with pytest.raises(ValueError, match="Duplicate/conflicting"):
        select_formal(sources, locked, info)
    sources, locked, info = make_fixture()
    sources["val"][0]["speaker"] = "mead_A"
    with pytest.raises(ValueError, match="identity split overlap"):
        select_formal(sources, locked, info)


def test_history_excluded_list_is_not_assumed_development(tmp_path):
    selected = tmp_path / "selection.json"
    selected.write_text(json.dumps({"selected_sentence_ids": ["old_dev"], "excluded_sentence_ids": ["old_train"]}))
    values, sources = collect_historical_development([selected])
    assert values == {"old_dev"}
    assert sources[0]["reason"] == "explicit_historical_development_selection"
    selected.write_text(json.dumps({"excluded_sentence_ids": ["old_train"]}))
    with pytest.raises(ValueError, match="explicit selected_sentence_ids"):
        collect_historical_development([selected])


def test_cli_locks_metadata_without_artifact_reads_and_preserves_validation_bytes(tmp_path, monkeypatch):
    sources, locked, info = make_fixture()
    native, previous, output = (tmp_path / name for name in ("native", "previous", "output"))
    native.mkdir()
    previous.mkdir()
    for name, rows in sources.items():
        (native / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    for role, rows in locked.items():
        (previous / f"{role}.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf8")
    info["manifest_sha256"] = {role: sha(previous / f"{role}.jsonl") for role in locked}
    (previous / "selection.json").write_text(json.dumps(info), encoding="utf8")
    monkeypatch.setattr("sys.argv", ["lock", "--native-root", str(native), "--locked-root", str(previous), "--output", str(output)])
    main()
    assert (output / "validation.jsonl").read_bytes() == (previous / "validation.jsonl").read_bytes()
    assert (output / "reserved_test_metadata.jsonl").read_bytes() == (previous / "new_test.jsonl").read_bytes()
    report = json.loads((output / "selection.json").read_text(encoding="utf8"))
    assert report["metadata_only"] is True and report["native_observations_read"] is False
    assert all(report["isolation_checks"].values())
    # Native .npz paths intentionally do not exist: successful locking proves
    # observations are not required/read by this path.
    assert not (native / "missing").exists()
    with pytest.raises(FileExistsError, match="fresh"):
        main()


def test_order_is_stable_and_missing_cells_reported():
    sources, locked, info = make_fixture()
    first, report = select_formal(sources, locked, info)
    flipped = {name: list(reversed(rows)) for name, rows in sources.items()}
    second, same = select_formal(flipped, locked, info)
    assert first == second and report == same
    sources["train"] = [r for r in sources["train"] if not (r["speaker"] == "mead_B" and r["emotion"] == 5)]
    _, missing = select_formal(sources, locked, info)
    assert len(missing["coverage"]["train"]["missing_cells"]) == 2
