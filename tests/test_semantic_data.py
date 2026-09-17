"""Behavior tests for independent references and the native visual-label clock."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetalk_b0.semantic_data import SemanticMotionDataset, interpolate_clock, semantic_collate


def make_manifest(tmp_path: Path, *, length: int = 15, window: int = 10) -> tuple[dict, list[dict]]:
    times = np.arange(length, dtype=np.float64) * 0.04
    rows = []
    for speaker in ("s1", "s2"):
        for emotion in ("happy", "sad", "angry"):
            for sentence in range(3):
                clip = f"{speaker}_{emotion}_{sentence}"
                path = tmp_path / f"{clip}.npz"
                motion = np.repeat(times[:, None], 4, axis=1).astype(np.float32)
                motion[:, 1] += 1 if speaker == "s1" else 2
                np.savez(path, motion=motion, content=np.repeat(times[:, None], 3, axis=1),
                         audio=np.repeat(times[:, None], 2, axis=1), times=times,
                         mask=np.ones(length, dtype=bool), channel_mask=[1, 1, 1, 0])
                va_path = tmp_path / f"{clip}_va.npz"
                np.savez(va_path, va=np.stack((times, -times), axis=-1), times=times,
                         valid=np.ones(length, dtype=bool), confidence=np.ones(length, dtype=np.float32),
                         source_modality=np.array("visual"))
                rows.append(dict(clip_id=clip, dataset="mead", speaker=speaker, sentence_id=str(sentence),
                                 emotion=emotion, intensity_id=1, split="train", motion_path=path.name,
                                 content_path=path.name, audio_path=path.name, va_path=va_path.name))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    cfg = {"data": {"emotion_classes": ["happy", "sad", "angry"], "motion_dim": 4,
                    "content_dim": 3, "audio_dim": 2},
           "semantic_data": {"manifest": str(manifest), "window": window, "reference_count": 2,
                             "motion_key": "motion", "audio_keys": ["audio"], "motion_valid_key": "mask",
                             "channel_mask_key": "channel_mask", "cache_size": 2}}
    return cfg, rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def rewrite_npz(path: Path, **updates) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    for name, value in updates.items():
        if value is None:
            values.pop(name, None)
        else:
            values[name] = value
    np.savez(path, **values)


def test_interpolation_respects_actual_clock_and_invalid_runs():
    times = np.array([0.0, 0.04, 0.08, 0.2])
    values = np.array([[0.0], [4.0], [8.0], [20.0]])
    targets = np.array([-0.01, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.21])
    aligned, valid = interpolate_clock(values, times, targets, valid=np.array([1, 1, 0, 1]), max_gap_s=0.1)
    assert valid.tolist() == [False, True, True, True, False, False, False, True, False]
    np.testing.assert_allclose(aligned[valid, 0], [0, 2, 4, 20])
    assert np.all(aligned[~valid] == 0)


def test_independent_references_donor_and_collate(tmp_path):
    cfg, rows = make_manifest(tmp_path)
    dataset = SemanticMotionDataset(cfg, random_crop=False, indices=[0, 1])
    first = dataset[0]
    query = first["query"]["metadata"]
    same = first["same_style_references"]["metadata"]
    positive = first["positive_style_references"]["metadata"]
    assert {r["clip_id"] for r in same}.isdisjoint(r["clip_id"] for r in positive)
    for ref in same + positive:
        assert ref["speaker"] == query["speaker"]
        assert ref["emotion"] != query["emotion"]
        assert ref["sentence_id"] != query["sentence_id"]
    anchor = first["donor_anchor"]["metadata"]
    assert anchor["speaker"] != query["speaker"]
    assert anchor["emotion"] == query["emotion"]
    for ref in first["donor_references"]["metadata"]:
        assert ref["speaker"] == anchor["speaker"]
        assert ref["sentence_id"] != anchor["sentence_id"]
        assert ref["emotion"] != anchor["emotion"]
    batch = semantic_collate([first, dataset[1]])
    assert batch["query"]["motion"].shape == (2, 10, 4)
    assert batch["same_style_references"]["motion"].shape == (2, 2, 10, 4)
    assert batch["same_style_references"]["channel_mask"].shape == (2, 2, 4)
    assert batch["query"]["channel_mask"].tolist() == [[True, True, True, False]] * 2
    assert batch["query"]["emotion_id"].tolist() == [0, 0]
    assert len(batch["same_style_references"]["metadata"]) == 2


def test_native_offsets_masks_and_padding(tmp_path):
    cfg, rows = make_manifest(tmp_path, length=8, window=12)
    rows[0]["source_audio_offset_s"] = 0.04
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    motion_mask = np.ones(8, dtype=bool)
    motion_mask[2] = False
    rewrite_npz(tmp_path / rows[0]["motion_path"], mask=motion_mask)
    va_valid = np.ones(8, dtype=bool)
    va_valid[4] = False
    rewrite_npz(tmp_path / rows[0]["va_path"], valid=va_valid)
    dataset = SemanticMotionDataset(cfg, random_crop=False, indices=[0])
    query = dataset[0]["query"]
    assert query["valid"].tolist() == [True, True, False, True, False, True, True, False, False, False, False, False]
    torch.testing.assert_close(query["content"][:7, 0], query["motion"][:7, 0] + 0.04)
    torch.testing.assert_close(query["va"][:4, 0], query["motion"][:4, 0])
    assert query["motion_valid"][4] and not query["va_valid"][4]
    assert query["times"].dtype == torch.float64
    assert not query["motion"][8:].any()


def test_va_missing_and_audio_provenance_are_rejected(tmp_path):
    cfg, rows = make_manifest(tmp_path)
    del rows[0]["va_path"]
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    with pytest.raises(ValueError, match="missing va_path"):
        SemanticMotionDataset(cfg)
    rows[0]["va_path"] = rows[0]["clip_id"] + "_va.npz"
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    rewrite_npz(tmp_path / rows[0]["va_path"], source_modality=np.array("audio"))
    dataset = SemanticMotionDataset(cfg, indices=[0])
    with pytest.raises(ValueError, match="visual provenance"):
        dataset[0]


@pytest.mark.parametrize("failure", ["nan", "clock", "dimension", "confidence", "no_va"])
def test_rejects_corrupt_assets(tmp_path, failure):
    cfg, rows = make_manifest(tmp_path)
    motion_path, va_path = tmp_path / rows[0]["motion_path"], tmp_path / rows[0]["va_path"]
    if failure == "nan":
        motion = np.zeros((15, 4))
        motion[0, 0] = np.nan
        rewrite_npz(motion_path, motion=motion)
    elif failure == "clock":
        rewrite_npz(va_path, times=np.zeros(15))
    elif failure == "dimension":
        rewrite_npz(motion_path, motion=np.zeros((15, 3)))
    elif failure == "confidence":
        rewrite_npz(va_path, confidence=np.full(15, 2.0))
    else:
        rewrite_npz(va_path, valid=np.zeros(15, dtype=bool))
    dataset = SemanticMotionDataset(cfg, indices=[0])
    with pytest.raises(ValueError):
        dataset[0]


def test_intensity_unknown_is_explicit_and_not_neutral(tmp_path):
    cfg, rows = make_manifest(tmp_path)
    del rows[0]["intensity_id"]
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    with pytest.raises(ValueError, match="explicit intensity validity"):
        SemanticMotionDataset(cfg, indices=[0])
    rows[0]["intensity_valid"] = False
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    dataset = SemanticMotionDataset(cfg, indices=[0])
    query = dataset[0]["query"]
    assert query["intensity_id"].item() == -1
    assert not query["intensity_valid"].item()


def test_reference_shortage_and_split_leakage_fail_loudly(tmp_path):
    cfg, rows = make_manifest(tmp_path)
    rows[0]["split"] = "val"
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    with pytest.raises(ValueError, match="multiple splits"):
        SemanticMotionDataset(cfg)
    rows[0]["split"] = "train"
    for row in rows:
        row["sentence_id"] = "same_sentence"
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    with pytest.raises(ValueError, match="different emotion AND sentence"):
        SemanticMotionDataset(cfg)


def test_deterministic_evaluation_and_bounded_cache(tmp_path):
    cfg, _ = make_manifest(tmp_path)
    dataset = SemanticMotionDataset(cfg, random_crop=False, subset_indices=[0])
    before = dataset[0]
    dataset.set_epoch(7)
    after = dataset[0]
    assert before["same_style_references"]["metadata"] == after["same_style_references"]["metadata"]
    torch.testing.assert_close(before["query"]["motion"], after["query"]["motion"])
    assert len(dataset._cache) <= 2


def test_motion_fps_fallback_is_explicit(tmp_path):
    cfg, rows = make_manifest(tmp_path)
    row = rows[0]
    new_path = tmp_path / "motion_without_clock.npz"
    np.savez(new_path, motion=np.zeros((15, 4)), mask=np.ones(15), channel_mask=np.ones(4))
    row["motion_path"] = new_path.name
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    dataset = SemanticMotionDataset(cfg, indices=[0])
    with pytest.raises(ValueError, match="timestamps missing"):
        dataset[0]
    row.update(motion_clock="native_fps", fps=25.0, motion_start_s=0.0)
    write_manifest(Path(cfg["semantic_data"]["manifest"]), rows)
    query = SemanticMotionDataset(cfg, indices=[0], random_crop=False)[0]["query"]
    torch.testing.assert_close(query["times"], torch.arange(2, 12, dtype=torch.float64) / 25)


def test_va_identity_conflict_is_rejected(tmp_path):
    cfg, rows = make_manifest(tmp_path)
    rewrite_npz(tmp_path / rows[0]["va_path"], clip_id=np.array("wrong_clip"))
    with pytest.raises(ValueError, match="clip_id mismatch"):
        SemanticMotionDataset(cfg, indices=[0])[0]


def test_exact_sample_after_roundoff_does_not_depend_on_invalid_neighbor():
    values = np.array([[1.0], [2.0], [3.0]])
    aligned, valid = interpolate_clock(values, np.array([0.0, 0.04, 0.08]), np.array([0.04 + 1e-10]),
                                       valid=np.array([0, 1, 0]))
    assert valid.item()
    assert aligned.item() == 2.0
