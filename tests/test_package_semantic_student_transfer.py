import argparse
import copy
import json

import numpy as np
import pytest
import torch

from scripts import package_semantic_student_transfer as pack


def example():
    baseline = np.arange(8 * 52, dtype=np.float32).reshape(8, 52) / 500
    common = np.array([False, True, True, False, True, True, False, False])
    native = np.array([True, True, True, False, True, True, True, False])
    return {"baseline52": baseline, "valid": common, "native_valid": native,
            "times": np.arange(8) / 25., "target": np.ones((8, 9), np.float32),
            "samples": {arm: np.stack([np.full((8, 9), 2., np.float32),
                                       np.full((8, 9), .1 + i / 10, np.float32)])
                        for i, arm in enumerate(pack.ARMS)},
            "seeds": [123, 42], "metadata": {"split": "holdout", "emotion": 0}}


def test_all_eight_arms_preserve43_and_native_invalid_and_continue_in_teacher_gaps():
    case = example(); before = copy.deepcopy(case)
    values, common, native, _, draw = pack.compose_case(case)
    assert draw == 1
    for index, arm in enumerate(pack.ARMS):
        np.testing.assert_array_equal(values[arm][:, pack.OTHER], case["baseline52"][:, pack.OTHER])
        np.testing.assert_array_equal(values[arm][~native], case["baseline52"][~native])
        assert np.all(values[arm][np.ix_(native, pack.UPPER)] == np.float32(.1 + index / 10))
    np.testing.assert_array_equal(values["reference"][~common], case["baseline52"][~common])
    np.testing.assert_array_equal(before["baseline52"], case["baseline52"])
    np.testing.assert_array_equal(before["samples"]["va_new"], case["samples"]["va_new"])


def test_missing_seed_native_nan_and_leaking_visual_mask_fail_closed():
    case = example(); case["seeds"] = [1, 2]
    with pytest.raises(ValueError, match="seed42"):
        pack.compose_case(case)
    case = example(); case["samples"]["va_reverse"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pack.compose_case(case)
    case = example(); case["valid"][3] = True
    with pytest.raises(ValueError, match="nested"):
        pack.compose_case(case)


def test_raw_values_are_not_clamped_and_metadata_selection_is_fixed():
    case = example(); case["samples"]["posterior_new"][1] = 1.7
    values, _, native, _, _ = pack.compose_case(case)
    assert np.all(values["posterior_new"][np.ix_(native, pack.UPPER)] == np.float32(1.7))
    cases = {"z": example(), "a": example(), "train": example()}
    cases["train"]["metadata"]["split"] = "train"
    assert pack.select_examples(cases) == ["a"]


def bound_source(tmp_path, case):
    root = tmp_path / "baseline"; root.mkdir()
    (root / "audio").mkdir(); (root / "video_npz").mkdir()
    waveform = root / "audio" / "a.wav"; waveform.write_bytes(b"bound-waveform-placeholder")
    saved = root / "video_npz" / "a.npz"
    np.savez_compressed(saved, mode_names=np.asarray(["run12 frozen audio baseline"]),
        motions=case["baseline52"][None], valid=case["native_valid"], times=case["times"],
        audio_relative_path=np.asarray("audio/a.wav"), audio_sha256=np.asarray(pack.sha(waveform)),
        audio_offset_seconds=np.asarray(.1), channel_mask=np.ones(52, bool))
    record = {"clip_id": "a", "emotion": 0, "arrays": "arrays/a.npz", "arrays_sha256": "not-downloaded",
              "video_npz": "video_npz/a.npz", "video_npz_sha256": pack.sha(saved)}
    pack.write(root / "provenance.json", {"clips": [record]})
    return root, record


def test_downloaded_video_baseline_binding_verifies_arrays_and_audio(tmp_path):
    case = example(); source, record = bound_source(tmp_path, case)
    values, _, native, times, _ = pack.compose_case(case)
    waveform, _, _, _ = pack.bind_baseline(source, record, values, native, times)
    waveform.write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio"):
        pack.bind_baseline(source, record, values, native, times)


def test_package_contains_input_transfer_limits_static_reverse_and_six_ordered_modes(tmp_path, monkeypatch):
    case = example(); source, _ = bound_source(tmp_path, case)
    predictions = tmp_path / "predictions.pt"
    torch.save({"schema": pack.SOURCE_SCHEMA, "clips": {"a": case}}, predictions)
    args = argparse.Namespace(predictions=predictions, baseline_root=source, output=tmp_path / "review",
        preview_frames=96, tile_size=320, samples=8, render=False)
    monkeypatch.setattr(pack, "plot_curves", lambda path, *unused: path.write_bytes(b"plot-placeholder"))
    records = pack.package(args)
    assert records[0]["protected43_exact"] and records[0]["native_invalid_exact"]
    report = json.loads((args.output / "report.json").read_text(encoding="utf8"))
    assert report["distribution_shift_diagnostic"] is True
    assert report["matched_receiver_retraining"] is False
    assert report["static_and_reverse_in_full_curves"] is True
    with np.load(args.output / "video_npz" / "a.npz") as saved:
        assert saved["mode_names"].tolist() == list(pack.MODES)
        np.testing.assert_array_equal(saved["motions"][3][:, pack.UPPER][case["native_valid"]],
                                      case["samples"]["va_new"][1, case["native_valid"]])
    with np.load(args.output / "curves" / "a.npz") as saved:
        assert set(pack.ARMS) <= set(saved.files)
    assert "分布偏移" in (args.output / "index.html").read_text(encoding="utf8")


def test_incorrect_source_schema_is_not_silently_packaged(tmp_path):
    path = tmp_path / "prediction.pt"; torch.save({"schema": "old"}, path)
    args = argparse.Namespace(predictions=path, output=tmp_path / "out", preview_frames=96, tile_size=320, samples=8)
    with pytest.raises(ValueError, match="semantic_student_transfer_v1"):
        pack.package(args)


def test_optional_constant_arms_are_preserved_separately_from_static():
    case = example()
    for arm in pack.OPTIONAL_ARMS:
        case["samples"][arm] = np.full((2, 8, 9), .93, np.float32)
    values, _, native, _, _ = pack.compose_case(case)
    for arm in pack.OPTIONAL_ARMS:
        assert np.all(values[arm][np.ix_(native, pack.UPPER)] == np.float32(.93))
        assert not np.array_equal(values[arm], values[arm.replace("constant", "static")])
        np.testing.assert_array_equal(values[arm][:, pack.OTHER], case["baseline52"][:, pack.OTHER])
