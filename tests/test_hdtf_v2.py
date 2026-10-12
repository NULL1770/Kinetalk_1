from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from kinetalk_v2.hdtf import (ExtractOptions, MULTIFACE, TRACK_JUMP, atomic_json,
                              channel_gather, head_pose, identity_group,
                              max_false_run, sha256, split_for_group, training_windows,
                              verify_cached)


def test_speaker_suffix_and_prefix_aliases_cannot_leak_between_splits():
    names = ["WDA_AmyKlobuchar0_000", "WDA_AmyKlobuchar1_001", "WRA_AmyKlobuchar_000"]
    groups = [identity_group(n) for n in names]
    assert len({g[0] for g in groups}) == 1
    assert len({split_for_group(g[0], g[1]) for g in groups}) == 1
    assert identity_group("WDA_WhipJimClyburn_000")[0] == identity_group("WDA_JamesClyburn1_000")[0]


def test_ambiguous_radio_identity_is_never_assumed_unseen_person():
    group, status, _ = identity_group("RD_Radio34_001")
    assert split_for_group(group, status) == "quarantine"


def test_rig_channel_mapping_is_name_based_with_tongue_mask():
    # Test deliberately permuted rig and MediaPipe order, not mirrored order.
    with np.load(Path(__file__).resolve().parents[1] / "final_experiment/rig/fixed_arkit_rig_20260923.npz") as rig:
        rig_names = list(rig["channels"])
    channels = list(reversed(rig_names))
    mp_names = [n for n in rig_names if n.lower() != "tongueout"] + ["_neutral"]
    categories = mp_names[9:] + mp_names[:9]
    idx, mask = channel_gather(channels, categories)
    for i, have in enumerate(mask):
        if have:
            assert categories[idx[i]].lower() == channels[i].lower()
        else:
            assert channels[i].lower() == "tongueout"
    with pytest.raises(ValueError):
        channel_gather(channels, ["_neutral"] * 52)


def test_maximum_invalid_run_including_boundaries():
    assert max_false_run(np.ones(8, bool)) == 0
    assert max_false_run(np.array([0, 0, 1, 0, 1, 0, 0, 0], bool)) == 3
    assert max_false_run(np.zeros(8, bool)) == 8


def test_qc_windows_never_cross_multiface_or_tracking_switch():
    opts = ExtractOptions("model", "rig", "out", window_frames=4, stride_frames=4, context_frames=2)
    record = dict(source_id="x", source_group="x", speaker_group="p", identity_status="named_filename",
                  split="train", motion_path="x.npz", audio_path="x.wav")
    q, f = np.ones(12, bool), np.zeros(12, np.uint16)
    f[5], f[9] = MULTIFACE, TRACK_JUMP
    result = training_windows(record, q, f, opts)
    assert [r["start_frame"] for r in result] == [0]
    assert result[0]["emotion_id"] == -1 and not result[0]["emotion_valid"]
    assert result[0]["context_end_frame"] == 6
    assert result[0]["event_annotations_require_source_context"]


def test_scaled_rotation_pose_and_atomic_unicode_record(tmp_path):
    angle = np.deg2rad(27.)
    mat = np.eye(4)
    mat[:3, :3] = 1.7 * np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    np.testing.assert_allclose(head_pose(mat), [27, 0, 0], atol=1e-5)
    path = tmp_path / "状态.json"
    atomic_json(path, {"状态": "完成"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"状态": "完成"}


def test_resume_rejects_corrupted_motion_and_modified_source(tmp_path):
    source, motion, audio, receipt, windows = [tmp_path / n for n in ("source.mp4", "motion.npz", "audio.wav", "receipt.json", "windows.jsonl")]
    source.write_bytes(b"source content")
    audio.write_bytes(b"audio content")
    windows.write_bytes(b"window content")
    np.savez(motion, motion=np.ones((3, 52), np.float32), times=np.arange(3) / 25)
    st = source.stat()
    data = {"signature": "v1", "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns,
            "source_sha256": sha256(source), "motion_path": str(motion), "audio_path": str(audio),
            "motion_sha256": sha256(motion), "audio_sha256": sha256(audio), "n_frames": 3,
            "windows_path": str(windows), "windows_sha256": sha256(windows)}
    atomic_json(receipt, data)
    assert verify_cached(receipt, source, "v1")["processing_status"] == "cached"
    assert verify_cached(receipt, source, "different") is None
    motion.write_bytes(b"partial")
    assert verify_cached(receipt, source, "v1") is None


def test_short_invalid_runs_are_masked_but_not_interpolated():
    opts = ExtractOptions("model", "rig", "out", window_frames=40, stride_frames=40, context_frames=2)
    record = dict(source_id="x", source_group="x", speaker_group="p", identity_status="named_filename",
                  split="train", motion_path="x.npz", audio_path="x.wav")
    q, f = np.ones(40, bool), np.zeros(40, np.uint16)
    q[10:12] = False
    assert len(training_windows(record, q, f, opts)) == 1
    q[12] = False
    assert training_windows(record, q, f, opts) == []
