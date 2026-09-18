import copy
import json

import numpy as np
import pytest
import torch

from scripts import prepare_continuous_motion_dataset as dataset


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


def _fixture(root, channel_shape="constant"):
    root.mkdir()
    (root / "arrays").mkdir()
    (root / "video_npz").mkdir()
    rows, records = [], []
    for index, split in enumerate(("train", "holdout")):
        cid = "clip_" + split
        row = {"clip_id": cid, "split": split, "sentence": "sentence_" + split,
               "speaker": index, "speaker_name": "person_" + str(index), "emotion": index}
        rows.append(row)
        n = 17
        valid = np.ones(n, bool)
        valid[8] = False
        channels = np.ones((n, 52) if channel_shape == "time" else (52,), bool)
        if channel_shape == "time":
            channels[3, dataset.UPPER[0]] = False
        baseline = np.full((n, 52), .2 + index, np.float32)
        baseline[~valid] = 50.
        target = np.full((n, 52), .3 + index, np.float32)
        features = np.full((n, 1540), float(index), np.float32)
        features[~valid] = np.nan
        payload = {"features": features, "valid": valid, "channel_mask": channels,
                   "times": np.arange(n, dtype=np.float64) / 25,
                   "baseline52": baseline, "affect_global": np.array([.1, .2], np.float32),
                   "affect_intensity": np.array([.7], np.float32),
                   "identity_code": np.array([.4, .5], np.float32),
                   "clip_id": np.asarray(cid), "audio_relative_path": np.asarray("audio/" + cid + ".wav"),
                   "audio_sha256": np.asarray("a" * 64), "audio_offset_seconds": np.asarray(0.)}
        array_name, video_name = "arrays/" + cid + ".npz", "video_npz/" + cid + ".npz"
        np.savez_compressed(root / array_name, **payload)
        np.savez_compressed(root / video_name,
            mode_names=np.asarray(["native coefficient reference (not inference input)", "run12 frozen audio baseline"]),
            motions=np.stack([target, baseline]), **{k: payload[k] for k in
                ("times", "valid", "channel_mask", "clip_id", "audio_relative_path", "audio_sha256", "audio_offset_seconds")})
        records.append({**row, "arrays": array_name, "arrays_sha256": dataset.sha(root / array_name),
                        "video_npz": video_name, "video_npz_sha256": dataset.sha(root / video_name),
                        "native_metadata": {"historical_fit": True}})
    selection = root.parent / (root.name + "_selection.json")
    _write_json(selection, {"clips": rows})
    _write_json(root / "provenance.json", {"clips": records, "selection_sha256": dataset.sha(selection),
        "query_motion_inference": False, "inference_frozen": True, "test_loaded": False,
        "dev405_indexed": False})
    _manifest(root)
    return selection


def _manifest(root):
    _write_json(root / "manifest.json", {p.relative_to(root).as_posix():
        {"sha256": dataset.sha(p), "bytes": p.stat().st_size}
        for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"})


def _clip(cid="a", frames=10, value=.1, split="train"):
    b9 = torch.full((9,), .2)
    return {"clip_id": cid, "split": split, "sentence": "sentence_" + cid,
            "features": torch.full((frames, 1540), float(value)),
            "context": torch.full((14,), float(value)), "b9": b9,
            "motion9": torch.full((frames, 9), float(value)) + b9,
            "valid": torch.ones(frames, dtype=torch.bool),
            "motion_mask": torch.ones(frames, 9, dtype=torch.bool), "metadata": {}}


@pytest.mark.parametrize("mask,expected", [([], []), ([0, 0], []), ([1], [(0, 1)]),
    ([1, 1, 0, 1, 0, 1, 1], [(0, 2), (3, 4), (5, 7)])])
def test_contiguous_runs(mask, expected):
    assert dataset.contiguous_runs(mask) == expected


@pytest.mark.parametrize("mask", [[[1]], [0, .5], [float("nan")]])
def test_contiguous_runs_rejects_nonbinary_or_matrix(mask):
    with pytest.raises(ValueError):
        dataset.contiguous_runs(mask)


@pytest.mark.parametrize("shape", ["constant", "time"])
def test_build_expands_masks_binds_sources_and_has_no_target_mean(tmp_path, shape):
    root = tmp_path / "baseline"
    selection = _fixture(root, shape)
    result = dataset.build(selection, root, tmp_path / "dataset.pt")
    assert result["schema"] == dataset.SCHEMA
    assert len(result["clips"]) == 2
    clip = result["clips"][0]
    assert clip["channel_mask"].shape == (17, 52)
    assert clip["motion_mask"].shape == (17, 9)
    assert torch.allclose(clip["b9"], torch.full((9,), .2))
    assert torch.allclose(clip["global"], torch.tensor([.1, .2, .7]))
    assert torch.equal(clip["context"], torch.cat([clip["global"], clip["identity_code"], clip["b9"]]))
    assert dataset.contiguous_runs(clip["valid"]) == [(0, 8), (9, 17)]
    assert not clip["motion_mask"][8].any()
    assert not result["provenance"]["va_used"]
    assert not any("va" == k or "posterior" == k for k in clip)
    assert result["stats"]["train_clip_ids"] == ["clip_train"]
    assert all(v.dtype == torch.float32 for v in result["stats"].values() if torch.is_tensor(v))
    assert (tmp_path / "dataset.json").is_file()
    with pytest.raises(FileExistsError):
        dataset.build(selection, root, tmp_path / "dataset.pt")


def test_fit_statistics_are_train_only_clip_equal_and_raw_residual_rms():
    clips = [_clip("a", 5, 1.), _clip("b", 100, 3.), _clip("c", 200, 100., "holdout")]
    stats = dataset.fit_statistics(clips)
    assert torch.allclose(stats["audio_mean"], torch.full((1540,), 2.))
    assert torch.allclose(stats["audio_scale"], torch.ones(1540))
    assert torch.allclose(stats["residual_scale"], torch.full((9,), np.sqrt(5.)), atol=1e-6)
    clips[-1]["motion9"][:] = float("nan")
    clips[-1]["features"][:] = float("nan")
    changed = dataset.fit_statistics(clips)
    for key in stats:
        if torch.is_tensor(stats[key]):
            assert torch.equal(stats[key], changed[key])
    zero = dataset.fit_statistics([_clip(value=0.)])
    assert torch.equal(zero["residual_scale"], torch.full((9,), .02))


def test_audio_statistics_and_deployment_support_ignore_target_mask():
    clip = _clip(frames=12)
    clip["features"][6:] = 2.
    original = dataset.fit_statistics([clip])
    corrupted = copy.deepcopy(clip)
    corrupted["motion_mask"][6:] = False
    corrupted["motion9"][6:] = float("nan")
    modified = dataset.fit_statistics([corrupted])
    for key in ("audio_mean", "audio_scale", "context_mean", "context_scale"):
        assert torch.equal(original[key], modified[key])
    assert dataset.contiguous_runs(corrupted["valid"]) == [(0, 12)]
    segments, report = dataset.prepare_segments([corrupted], modified, "train")
    assert segments[0]["audio"].shape[0] == 6
    assert report["totals"]["unobserved_native_valid_frames"] == 6


def test_segments_do_not_cross_gaps_duplicate_or_hide_dropped_frames():
    clip = _clip(frames=29)
    clip["valid"][12] = False
    clip["motion_mask"][20] = False
    stats = dataset.fit_statistics([clip])
    segments, report = dataset.prepare_segments([clip], stats, "train", max_frames=10)
    assert [(s["metadata"]["start"], s["metadata"]["end"]) for s in segments] == [(0, 10), (13, 20), (21, 29)]
    assert report["totals"]["joint_observed_frames"] == 27
    assert report["totals"]["kept_frames"] == 25
    assert report["totals"]["dropped_short_frames"] == 2
    assert report["totals"]["dropped_short_segments"] == 1
    assert report["totals"]["segments"] == 3
    for segment in segments:
        assert torch.isfinite(segment["residual"]).all()
        assert segment["residual"].shape[0] == segment["audio"].shape[0]
    empty, holdout_report = dataset.prepare_segments([clip], stats, "holdout")
    assert empty == [] and holdout_report["totals"]["kept_frames"] == 0


def test_completely_unobserved_clip_is_retained_and_reported():
    visible, unobserved = _clip("visible"), _clip("missing")
    unobserved["motion_mask"][:] = False
    unobserved["motion9"][:] = float("nan")
    stats = dataset.fit_statistics([visible, unobserved])
    assert stats["residual_no_joint_observation_clip_ids"] == ["missing"]
    _, report = dataset.prepare_segments([visible, unobserved], stats, "train")
    assert report["excluded_clip_ids"] == ["missing"]
    assert report["totals"]["unobserved_native_valid_frames"] == 10


def test_manifest_corruption_is_rejected_before_loading(tmp_path):
    root = tmp_path / "baseline"
    selection = _fixture(root)
    with (root / "arrays/clip_train.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash/size"):
        dataset.build(selection, root, tmp_path / "dataset.pt")


def test_selection_hash_and_train_holdout_sentence_binding(tmp_path):
    root = tmp_path / "baseline"
    selection = _fixture(root)
    picked = json.loads(selection.read_text())
    picked["clips"][1]["sentence"] = picked["clips"][0]["sentence"]
    _write_json(selection, picked)
    with pytest.raises(ValueError, match="Selection hash"):
        dataset.build(selection, root, tmp_path / "dataset.pt")
    provenance = json.loads((root / "provenance.json").read_text())
    provenance["selection_sha256"] = dataset.sha(selection)
    _write_json(root / "provenance.json", provenance)
    _manifest(root)
    with pytest.raises(ValueError, match="sentences overlap"):
        dataset.build(selection, root, tmp_path / "dataset.pt")


def test_provenance_rejects_target_conditioned_baseline(tmp_path):
    root = tmp_path / "baseline"
    selection = _fixture(root)
    provenance = json.loads((root / "provenance.json").read_text())
    provenance["query_motion_inference"] = True
    _write_json(root / "provenance.json", provenance)
    _manifest(root)
    with pytest.raises(ValueError, match="acoustic-only"):
        dataset.build(selection, root, tmp_path / "dataset.pt")
