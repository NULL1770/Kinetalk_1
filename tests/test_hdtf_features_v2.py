import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kinetalk_v2.hdtf_dataset import HDTFSplit
from scripts.extract_hdtf_features_v2 import (Backend, SAMPLE_RATE, chunked_backend,
    extract_source, interpolate, model_conv_geometry, source_rows, sha256)


GEOMETRY = {"sample_rate": 16000, "hop_samples": 320,
            "receptive_samples": 400, "center_offset_samples": 199.5}


class ClockBackend:
    geometry = GEOMETRY

    def __init__(self):
        self.lengths = []

    def extract(self, wave):
        self.lengths.append(len(wave))
        n = max(2, (len(wave) - 400) // 320 + 1)
        times = (199.5 + np.arange(n) * 320) / 16000.
        return np.broadcast_to(times[:, None], (n, 768)).astype(np.float32), times


def write_wave(path, seconds=12):
    import wave
    samples = (np.sin(np.arange(round(seconds * 16000)) * .07) * 0.1 * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
        out.writeframes(samples.tobytes())


def source_artifacts(tmp_path, sid="one", speaker="alice", split="train", frames=600):
    motion_dir = tmp_path / "motion_raw"; motion_dir.mkdir(exist_ok=True)
    audio_dir = tmp_path / "audio16k"; audio_dir.mkdir(exist_ok=True)
    motion_path = motion_dir / f"{sid}.npz"
    audio_path = audio_dir / f"{sid}.wav"
    valid = np.ones(frames, bool); valid[8] = False
    channels = np.ones(52, bool); channels[-1] = False
    times = 10. + np.arange(frames) / 25.
    np.savez_compressed(motion_path, motion=np.ones((frames, 52), np.float32) * .2,
                        times=times, valid=valid, motion_valid=valid, channel_mask=channels,
                        source_start_seconds=np.float64(10.))
    write_wave(audio_path, frames / 25.)
    return {"source_id": sid, "source_group": sid, "speaker_group": speaker,
            "split": split, "motion_path": str(motion_path), "audio_path": str(audio_path),
            "processed_source_start_seconds": 10.}


def window(rec, start, end):
    return dict(rec, clip_id=f"{rec['source_id']}:{start}:{end}", start_frame=start, end_frame=end,
                start_seconds=10. + start / 25., end_seconds=10. + end / 25.,
                emotion_id=0, emotion_valid=True, intensity_id=0, intensity_valid=True)


def feature_artifact(tmp_path, rec):
    root = tmp_path / "audio_features"
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "receipts").mkdir(exist_ok=True)
    sid = rec["source_id"]
    with np.load(rec["motion_path"]) as m:
        n = len(m["times"]); times = m["times"]
    path = root / "features" / f"{sid}.npz"
    np.savez_compressed(path, audio_features=np.ones((n, 1540), np.float16),
                        times=times, feature_valid=np.ones(n, bool))
    receipt = dict(rec, status="ok", features_path=str(path), features_sha256=sha256(path),
                   motion_sha256=sha256(Path(rec["motion_path"])))
    (root / "receipts" / f"{sid}.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root


def write_manifest(tmp_path, rows):
    path = tmp_path / "windows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_chunk_alignment_and_bounded_context_with_source_offset(tmp_path):
    path = tmp_path / "wave.wav"; write_wave(path, 19.)
    times = np.arange(475) / 25.
    backend = ClockBackend()
    result, ext = chunked_backend(path, times, np.ones(475, bool), backend, 6., 19., 1.)
    assert result.shape == (475, 768)
    assert np.isfinite(result).all()
    assert max(backend.lengths) <= 8 * SAMPLE_RATE
    # Each core's local relative clock is aligned, with no artificial zero
    # jump at the chunk boundary. A mock time ramp is offset by context start.
    assert result[150, 0] == pytest.approx(1., abs=1e-5)
    assert result[300, 0] == pytest.approx(1., abs=1e-5)
    assert ext[0] and not ext[150]


def test_geometry_and_outside_clock_are_rejected():
    cfg = SimpleNamespace(conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2])
    geometry = model_conv_geometry(cfg)
    assert geometry["hop_samples"] == 320 and geometry["receptive_samples"] == 400
    f = np.ones((50, 768), np.float32); t = (199.5 + np.arange(50) * 320) / 16000
    with pytest.raises(ValueError, match="outside waveform"):
        interpolate(f, t, np.array([-0.1]), np.ones(1, bool), 1., GEOMETRY)


def test_hubert_global_waveform_normalization_reaches_the_native_processor():
    class Processor:
        do_normalize = True
        def __call__(self, wave, **kwargs):
            self.seen = np.array(wave)
            return SimpleNamespace(input_values=torch.from_numpy(self.seen)[None])
    class Model:
        def __call__(self, value, **kwargs):
            self.seen = value
            feat = torch.zeros(1, 4, 768)
            return SimpleNamespace(hidden_states=tuple(feat for _ in range(7)))
    processor, model = Processor(), Model()
    backend = Backend("mock_hubert", model, GEOMETRY, "cpu", processor=processor)
    backend.set_global_normalization(2., 4.)
    backend.extract(np.array([2., 4., 6.], dtype=np.float32))
    np.testing.assert_allclose(processor.seen, np.array([0., 1., 2.]), rtol=1e-6)
    assert not processor.do_normalize


def test_source_extraction_separates_audio_observation_from_face_quality_and_resume(tmp_path):
    rec = source_artifacts(tmp_path, frames=300)
    backend = ClockBackend(); out = tmp_path / "features_out"
    done = extract_source(rec, out, backend, backend, {"pin": "test"}, 6.)
    assert done["n_frames"] == 300 and done["feature_valid_ratio"] == 1.
    with np.load(done["features_path"]) as f:
        assert f["audio_features"].shape == (300, 1540)
        np.testing.assert_array_equal(f["times"], 10. + np.arange(300) / 25.)
        assert f["feature_valid"][8] and not f["motion_valid"][8]
    lengths = list(backend.lengths)
    cached = extract_source(rec, out, backend, backend, {"pin": "test"}, 6.)
    assert cached["status"] == "cached" and backend.lengths == lengths
    with pytest.raises(ValueError, match="train/validation"):
        extract_source(dict(rec, split="test"), out, backend, backend, {}, 6.)


def test_loader_masks_missing_labels_channels_invalid_frames_and_keeps_absolute_crop_clock(tmp_path):
    rec = source_artifacts(tmp_path)
    root = feature_artifact(tmp_path, rec)
    path = write_manifest(tmp_path, [window(rec, 0, 128), window(rec, 200, 320)])
    q = HDTFSplit(path, root, split="train", speaker_offset=64)
    batch = q.batch(torch.tensor([0, 1]))
    assert batch["audio_features"].shape == (2, 128, 1540)
    assert batch["times"][1, 0] == 18. and batch["times"][1, 119] == pytest.approx(22.76)
    assert not batch["valid"][0, 8] and not batch["valid"][1, 120]
    assert not batch["channel_mask"][:, -1].any()
    assert not batch["neutral_target_valid"].any() and not batch["anchor_valid"].any()
    assert (batch["emotion_id"] == -1).all() and not batch["emotion_valid"].any()
    assert (batch["intensity_id"] == -1).all() and not batch["intensity_valid"].any()
    assert q["speaker"] == ["HDTF:alice", "HDTF:alice"]
    assert int(q["speaker_id"][0]) >= 64


def test_no_test_targets_opened_and_metadata_leakage_detected_before_io(tmp_path):
    train = source_artifacts(tmp_path, sid="train", speaker="alice")
    test = dict(train, source_id="sealed", source_group="sealed", speaker_group="bob", split="test",
                motion_path="must_not_open.npz", audio_path="must_not_open.wav")
    root = feature_artifact(tmp_path, train)
    path = write_manifest(tmp_path, [window(train, 0, 128), window(test, 0, 128)])
    q = HDTFSplit(path, root, split="train")
    assert len(q) == 1
    with pytest.raises(ValueError, match="sealed"):
        HDTFSplit(path, root, split="test")
    path = write_manifest(tmp_path, [window(train, 0, 128), window(dict(test, speaker_group="alice"), 0, 128)])
    with pytest.raises(ValueError, match="crosses splits"):
        HDTFSplit(path, root, split="train")
    with pytest.raises(ValueError, match="multiple splits"):
        source_rows(path)


def test_reference_never_self_or_nearby_and_prefers_independent_source(tmp_path):
    one = source_artifacts(tmp_path, sid="one", speaker="alice")
    two = source_artifacts(tmp_path, sid="two", speaker="alice")
    root = feature_artifact(tmp_path, one); feature_artifact(tmp_path, two)
    rows = [window(one, 0, 128), window(one, 128, 256), window(one, 400, 528), window(two, 0, 128)]
    path = write_manifest(tmp_path, rows); q = HDTFSplit(path, root, split="train")
    assert q.reference_candidates(0) == [3]
    refs = q.single_references([0, 1], seed=12)
    assert refs["reference_valid"].all() and refs["clip_id"] == [rows[3]["clip_id"]] * 2
    path = write_manifest(tmp_path, rows[:3]); q = HDTFSplit(path, root, split="train")
    assert q.reference_candidates(0) == [2]
    assert 1 not in q.reference_candidates(0)  # adjacent, less than .5s buffer
    path = write_manifest(tmp_path, rows[:1]); q = HDTFSplit(path, root, split="train")
    refs = q.single_references([0])
    assert not refs["reference_valid"].any() and not refs["valid"].any()
    assert refs["clip_id"] == [None]


def test_source_hash_or_feature_clock_corruption_fails(tmp_path):
    rec = source_artifacts(tmp_path)
    root = feature_artifact(tmp_path, rec)
    path = write_manifest(tmp_path, [window(rec, 0, 128)])
    q = HDTFSplit(path, root, split="train")
    Path(rec["motion_path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash changed"):
        q.batch([0])


def test_explicit_sg5_policy_verifies_hash_schema_raw_lineage_and_keeps_raw_default(tmp_path):
    rec = source_artifacts(tmp_path)
    root = feature_artifact(tmp_path, rec)
    processed_path = tmp_path / "processed.npz"
    raw_hash = sha256(Path(rec["motion_path"]))
    with np.load(rec["motion_path"]) as m:
        arrays = {k: np.asarray(m[k]).copy() for k in m.files}
    arrays.update(motion=np.ones((600, 52), np.float32) * .8,
                  postprocess_schema=np.asarray("hdtf-motion-postprocess-v2.1"),
                  candidate_window=np.int64(5), raw_motion_sha256=np.asarray(raw_hash))
    np.savez_compressed(processed_path, **arrays)
    post = dict(rec, motion_processed_path=str(processed_path), motion_processed_sha256=sha256(processed_path),
                raw_motion_sha256=raw_hash, postprocess_schema="hdtf-motion-postprocess-v2.1", candidate_window=5)
    post_path = tmp_path / "post.jsonl"; post_path.write_text(json.dumps(post) + "\n", encoding="utf-8")
    path = write_manifest(tmp_path, [window(rec, 0, 128)])
    raw = HDTFSplit(path, root, split="train")
    sg5 = HDTFSplit(path, root, split="train", motion_policy="sg5", postprocess_manifest=post_path)
    assert float(raw.batch([0])["motion"][0, 0, 0]) == pytest.approx(.2)
    assert float(sg5.batch([0])["motion"][0, 0, 0]) == pytest.approx(.8)
    assert sg5.provenance["motion_target_policy"] == "sg5"
    bad = dict(post, raw_motion_sha256="wrong")
    post_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw lineage"):
        HDTFSplit(path, root, split="train", motion_policy="sg5", postprocess_manifest=post_path)
    with pytest.raises(ValueError, match="requires complete"):
        HDTFSplit(path, root, split="train", motion_policy="sg5")


def test_event_context_preserves_source_clock_and_crops_after_annotation(tmp_path):
    rec = source_artifacts(tmp_path)
    root = feature_artifact(tmp_path, rec)
    row = dict(window(rec, 200, 328), context_start_frame=168, context_end_frame=360)
    path = write_manifest(tmp_path, [row])
    q = HDTFSplit(path, root, split="train")
    context = q.context_batch([0])
    assert context["motion"].shape[1] == 192
    assert context["times"][0, 0] == pytest.approx(16.72)
    assert int(context["query_offset"][0]) == 32
    label = torch.arange(192)[None, :, None]
    crop = q.crop_context_tensor(label, context)
    assert crop.shape == (1, 128, 1)
    assert int(crop[0, 0, 0]) == 32 and int(crop[0, -1, 0]) == 159
    torch.testing.assert_close(q.crop_context_tensor(context["times"], context), q.batch([0])["times"])
