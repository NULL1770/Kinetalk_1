"""Fixed renderer-capacity example export contracts."""
import numpy as np
import pytest
import torch

import scripts.export_renderer_capacity_examples as exporter
from scripts.export_renderer_capacity_examples import (
    DISPLAY_MODES, MODES, SEEDS, assemble_modes, bind_native_audio, native_arrays,
    validate_curve_protocol, video_picks,
)
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input


def test_video_selection_takes_first_locked_clip_per_speaker_only():
    clips = [
        {"speaker": speaker, "clip_id": f"{speaker}_{emotion}", "index": index}
        for index, (speaker, emotion) in enumerate((
            ("M023", "angry"), ("M023", "happy"), ("M023", "sad"),
            ("M024", "angry"), ("M024", "happy"), ("M024", "sad"),
            ("M030", "angry"), ("M030", "happy"), ("M030", "sad")))]
    assert [row["index"] for row in video_picks(clips)] == [0, 3, 6]
    with pytest.raises(ValueError, match="three locked speakers"):
        video_picks(clips[:6])


def sample_split(frames=4):
    gt = torch.full((2, frames, 52), 1.)
    b0 = torch.full_like(gt, .2)
    identity = torch.full((2, 52), .05)
    return {"q": {"motion": gt, "times": torch.arange(frames).double()[None].expand(2, -1) / 25,
                   "valid": torch.tensor([[True, True, False, True]] * 2),
                   "channel_mask": torch.ones(2, 52, dtype=torch.bool)},
            "base": {"b0": b0}, "identity": {"baseline": identity}}


def test_mode_mapping_and_native_npz_are_exact(tmp_path):
    split = sample_split()
    source = torch.full_like(split["q"]["motion"], .3)
    audio = torch.full_like(source, .4)
    oracle = torch.full_like(source, .5)
    zero = torch.full_like(source, .6)
    values = assemble_modes(split, 0, source, audio, oracle, zero)
    assert tuple(DISPLAY_MODES) == ("GT", "B0", "source", "audio", "oracle", "zero")
    assert torch.equal(values[0], split["q"]["motion"][0])
    assert torch.equal(values[1], split["base"]["b0"][0] + split["identity"]["baseline"][0])
    for index, expected in enumerate((source[0], audio[0], oracle[0], zero[0]), 2):
        assert torch.equal(values[index], expected)
    times, valid, channel_mask, motions = native_arrays(split["q"], 0, values)
    path = tmp_path / "example.npz"
    np.savez_compressed(path, channels=np.asarray(ARKIT_NAMES), times=times, valid=valid,
        channel_mask=channel_mask, mode_names=np.asarray(DISPLAY_MODES), motions=motions,
        clip_id=np.asarray("fixed"), noise_seed=np.asarray(42))
    channels, actual_times, actual_valid, modes, display, report = inspect_input(path, 25)
    assert channels == ARKIT_NAMES and modes == list(DISPLAY_MODES)
    assert np.array_equal(actual_times, times) and np.array_equal(actual_valid, valid)
    assert report["metadata"]["noise_seed"] == 42
    assert display.shape == motions.shape


def test_native_export_rejects_clock_and_valid_value_errors():
    split = sample_split()
    curves = [torch.zeros_like(split["q"]["motion"]) for _ in range(4)]
    values = assemble_modes(split, 0, *curves)
    split["q"]["times"][0, 2] = .09
    with pytest.raises(ValueError, match="native clock"):
        native_arrays(split["q"], 0, values)
    split = sample_split()
    values = assemble_modes(split, 0, *curves)
    values[2, 0, 4] = float("nan")
    with pytest.raises(ValueError, match="motion values"):
        native_arrays(split["q"], 0, values)
    split = sample_split()
    split["q"]["valid"] = split["q"]["valid"].long()
    with pytest.raises(ValueError, match="must be boolean"):
        native_arrays(split["q"], 0, values)


def test_curve_protocol_rejects_missing_seed_or_mode():
    row = {mode: torch.zeros(2, 4, 52) for mode in MODES}
    curves = {"noise_seeds": list(SEEDS), "decode_steps": 12,
              "motion": {str(seed): dict(row) for seed in SEEDS}}
    validate_curve_protocol(curves)
    del curves["motion"]["42"]["oracle"]
    with pytest.raises(ValueError, match="missing seed or condition"):
        validate_curve_protocol(curves)


def test_native_audio_binding_verifies_exact_cached_crop(tmp_path, monkeypatch):
    split = sample_split()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixed audio bytes")
    clip = {key: split["q"][key][0].clone() for key in ("motion", "times", "valid", "channel_mask")}
    clip["metadata"] = {"artifact": "native.npz", "artifact_sha256": "artifact-hash", "crop_start": 7,
        "provenance": {"audio_path": str(audio), "audio_sha256": exporter.sha(audio),
                       "audio_offset_s": .125, "fps": 25, "clock_evidence": "embedded_video"}}
    monkeypatch.setattr(exporter, "native_clip", lambda row, root: clip)
    pick = {"clip_id": "fixed", "index": 0}
    result = bind_native_audio(split["q"], pick, {"clip_id": "fixed"}, tmp_path)
    assert result["audio_offset_s"] == .125 and result["crop_start"] == 7
    split["q"]["times"][0, 1] += .001
    with pytest.raises(ValueError, match="native artifact: times"):
        bind_native_audio(split["q"], pick, {"clip_id": "fixed"}, tmp_path)
