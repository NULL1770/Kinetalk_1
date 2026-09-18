import json

import pytest

from scripts import extend_continuous_motion_selection as mod


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf8")


def row(cid, sentence, split="train"):
    parts = cid.split("_")
    return {"clip_id": cid, "sentence": sentence, "speaker": 0,
            "speaker_name": "mead_" + parts[1], "emotion": 0, "split": split}


def make_protocol(tmp):
    fit = [row("mead_M001_a", "s_a"), row("mead_M001_b", "s_b"),
           row("mead_M002_c", "s_c"), row("mead_M002_d", "s_d")]
    protocol = tmp / "protocol.json"
    write(protocol, {"split": {"fit": fit}})
    old = tmp / "old_selection.json"
    old_rows = [row("mead_M001_a", "s_a"), row("mead_M001_b", "s_b"),
                row("mead_M002_c", "s_c", "holdout")]
    write(old, {"schema": "visual_semantic_pilot_selection_v1", "clips": old_rows,
                "source_protocol_sha256": mod.sha(protocol), "holdout_sentences": ["s_c"]})
    return protocol, old


def test_select_preserves_old_and_adds_fit_only_train(tmp_path):
    protocol, old = make_protocol(tmp_path)
    combined = tmp_path / "combined.json"
    missing = tmp_path / "missing.json"
    result, added = mod.select(protocol, old, combined, missing)
    assert [r["clip_id"] for r in result["clips"]] == ["mead_M001_a", "mead_M001_b", "mead_M002_c", "mead_M002_d"]
    assert result["counts"] == {"train": 3, "holdout": 1}
    assert [r["clip_id"] for r in added["clips"]] == ["mead_M002_d"]
    assert added["clips"][0]["speaker_name"] == "mead_M002"
    with pytest.raises(FileExistsError):
        mod.select(protocol, old, combined, tmp_path / "other.json")


def test_select_rejects_old_membership_drift_and_overlap(tmp_path):
    protocol, old = make_protocol(tmp_path)
    payload = json.loads(old.read_text())
    payload["clips"][1]["sentence"] = "s_c"
    write(old, payload)
    with pytest.raises(ValueError):
        mod.select(protocol, old, tmp_path / "combined.json", tmp_path / "missing.json")


def test_select_excludes_all_declared_held_sentences_even_if_unrepresented(tmp_path):
    protocol, old = make_protocol(tmp_path)
    source = json.loads(protocol.read_text())
    source["split"]["fit"].append(row("mead_M003_e", "s_e"))
    write(protocol, source)
    selection = json.loads(old.read_text())
    selection["source_protocol_sha256"] = mod.sha(protocol)
    selection["holdout_sentences"].append("s_e")
    write(old, selection)
    combined, missing = mod.select(protocol, old, tmp_path / "combined.json", tmp_path / "missing.json")
    assert combined["unrepresented_held_sentences"] == ["s_e"]
    assert combined["omitted_held_sentence_clip_ids"] == ["mead_M003_e"]
    assert "s_e" in combined["holdout_sentences"]
    assert all(r["sentence"] not in ("s_c", "s_e") for r in combined["clips"] if r["split"] == "train")
    assert [r["clip_id"] for r in missing["clips"]] == ["mead_M002_d"]


def _manifest(root, provenance, names):
    for name, text in names.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf8")
    write(root / "provenance.json", provenance)
    manifest = {p.relative_to(root).as_posix(): {"sha256": mod.sha(p), "bytes": p.stat().st_size}
                for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    write(root / "manifest.json", manifest)


def _export(root, clips, selection_hash, speaker=0):
    clips = [dict(clip) for clip in clips]
    provenance = {"schema": mod.BASELINE_SCHEMA, "clips": clips, "selection_sha256": selection_hash,
        "source": {"cache": "cache", "audio": "audio", "targets": "targets"},
        "checkpoint_sha256": "checkpoint", "decode_steps": 8, "native_source": {"x": 1},
        "seed_policy": "seed42", "inference_frozen": True, "query_motion_inference": False,
        "reference_render_exported": True, "test_loaded": False, "dev405_indexed": False,
        "neutral_reference_bindings": [{"speaker_id": speaker, "clip_id": "neutral_" + str(speaker),
                                        "artifact_sha256": "r" * 64}], "new_training": False}
    names = {}
    for clip in clips:
        cid = clip["clip_id"]
        names.update({"arrays/" + cid + ".npz": "array:" + cid,
                      "video_npz/" + cid + ".npz": "video:" + cid,
                      "audio/" + cid + ".wav": "audio:" + cid})
        for folder, key in (("arrays", "arrays"), ("video_npz", "video_npz")):
            clip[key] = folder + "/" + cid + ".npz"
            clip[key + "_sha256"] = ""
    _manifest(root, provenance, names)
    manifest = json.loads((root / "manifest.json").read_text())
    for clip in clips:
        for key in ("arrays", "video_npz"):
            clip[key + "_sha256"] = manifest[clip[key]]["sha256"]
    _manifest(root, provenance, names)


@pytest.mark.parametrize("force_copy", [False, True])
def test_merge_verifies_sources_and_materializes_new_manifest(tmp_path, monkeypatch, force_copy):
    protocol, old_selection = make_protocol(tmp_path)
    combined = tmp_path / "combined.json"
    missing = tmp_path / "missing.json"
    combined_rows, missing_rows = mod.select(protocol, old_selection, combined, missing)
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    _export(old_root, json.loads(old_selection.read_text())["clips"], mod.sha(old_selection), speaker=0)
    _export(new_root, missing_rows["clips"], mod.sha(missing), speaker=1)
    source_hashes = {str(p): mod.sha(p) for root in (old_root, new_root) for p in root.rglob("*") if p.is_file()}
    if force_copy:
        def unavailable(*args):
            raise OSError("Cross-device hard links unavailable")
        monkeypatch.setattr(mod.os, "link", unavailable)
    out = tmp_path / "merged"
    result = mod.merge(old_root, new_root, combined, out)
    assert [r["clip_id"] for r in result["clips"]] == [r["clip_id"] for r in combined_rows["clips"]]
    assert result["selection_sha256"] == mod.sha(combined)
    assert len(result["neutral_reference_bindings"]) == 2
    assert sum(result["merge_materialization"].values()) == 12
    if force_copy:
        assert result["merge_materialization"] == {"hardlinked": 0, "copied": 12}
    exported_manifest, verified, members = mod._verified_export(out)
    assert set(members) == {r["clip_id"] for r in combined_rows["clips"]}
    assert verified["selection_sha256"] == mod.sha(combined)
    for name, record in exported_manifest.items():
        mod.checked(out / name, record)
    for path, digest in source_hashes.items():
        assert mod.sha(path) == digest
    with pytest.raises(FileExistsError):
        mod.merge(old_root, new_root, combined, out)


def test_merge_rejects_manifest_corruption_before_creating_output(tmp_path):
    protocol, old_selection = make_protocol(tmp_path)
    combined, missing = tmp_path / "combined.json", tmp_path / "missing.json"
    _, missing_rows = mod.select(protocol, old_selection, combined, missing)
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    _export(old_root, json.loads(old_selection.read_text())["clips"], mod.sha(old_selection))
    _export(new_root, missing_rows["clips"], mod.sha(missing))
    (new_root / "arrays/mead_M002_d.npz").write_text("changed", encoding="utf8")
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="hash/size"):
        mod.merge(old_root, new_root, combined, output)
    assert not output.exists()


def test_manifest_path_safety():
    with pytest.raises(ValueError):
        mod._safe_relative("../outside")
