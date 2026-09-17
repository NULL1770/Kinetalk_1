import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.prepare_label_guided_audio_cache import (
    combine_frame_features, fit_feature_statistics, index_sidecars,
    load_authorized_native, read_manifest, validate_cache,
)
from scripts.extract_emotion2vec_pilot import sha
from scripts import prepare_label_guided_audio_cache as module


def native_fixture(tmp_path):
    root = tmp_path / "native"; root.mkdir()
    path = root / "clip.npz"
    content = (np.arange(6 * 768, dtype=np.float32).reshape(6, 768) / 250 + 4)
    times = np.arange(6, dtype=np.float64) / 25
    mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    provenance = {"schema": "native_affect_style_v4.1", "clock_evidence": "embedded_video", "fps": 25,
                  "audio_path": "/authorized.wav", "audio_sha256": "abc", "audio_offset_s": .12}
    # There is deliberately no motion array: this loader must never require it.
    np.savez(path, content=content, times=times, mask=mask, provenance=json.dumps(provenance))
    row = {"clip_id": "fit", "sentence": "s1", "speaker": "p1", "emotion": 1, "split": "train",
           "artifact": "clip.npz", "artifact_sha256": sha(path)}
    target_times = torch.arange(5, dtype=torch.float64) / 25 + .12
    valid = torch.tensor([True, True, True, False, False])
    cached = torch.zeros(5, 768, dtype=torch.float16)
    cached[:3] = torch.from_numpy(content[3:]).half()
    return root, row, target_times, valid, cached, content


def test_native_load_preserves_exact_crop_native_fp32_and_clock_without_motion(tmp_path):
    root, row, times, valid, cached, original = native_fixture(tmp_path)
    clip, content, evidence = load_authorized_native(row, root, times, valid, cached)
    torch.testing.assert_close(content[:3], torch.from_numpy(original[3:]), rtol=0, atol=0)
    assert content.dtype == torch.float32 and torch.equal(clip["times"], times)
    assert torch.equal(clip["valid"], valid) and not content[3:].any()
    assert evidence["crop_start"] == 3 and evidence["crop_in_range_frames"] == 3


@pytest.mark.parametrize("corruption", ["hash", "clock", "mask", "content", "escape", "test"])
def test_native_load_rejects_wrong_binding_and_unauthorized_inputs(tmp_path, corruption):
    root, row, times, valid, cached, _ = native_fixture(tmp_path)
    if corruption == "hash": row["artifact_sha256"] = "changed"
    if corruption == "clock": times += .001
    if corruption == "mask": valid[3] = True
    if corruption == "content": cached[0, 0] += 1
    if corruption == "escape": row["artifact"] = "../outside.npz"
    if corruption == "test": row["split"] = "test"
    with pytest.raises((ValueError, AssertionError)):
        load_authorized_native(row, root, times, valid, cached)


def audio_fixture():
    clip = {"clip_id": "fit", "sentence_id": "s1", "valid": torch.tensor([True, True, False]),
            "times": torch.arange(3, dtype=torch.float64) / 25,
            "metadata": {"provenance": {"audio_sha256": "wave", "audio_offset_s": .12}}}
    record = {"clip_id": "fit", "sentence_id": "s1", "valid": clip["valid"].clone(),
              "times": clip["times"].clone(), "middle": torch.full((3, 768), 3., dtype=torch.float16),
              "prosody": torch.tensor([[4., -7., .9, 1.], [5., -6., .8, 1.], [float("nan")] * 4]),
              "record": {"wave_sha256": "wave", "audio_offset_s": .12}}
    content = torch.full((3, 768), 6.); content[-1] = float("nan")
    return clip, record, content


def test_combination_keeps_clip_dc_absolute_energy_and_masks_invalid_nan():
    clip, record, content = audio_fixture()
    value = combine_frame_features(content, clip, record)
    assert value.shape == (3, 1540) and torch.isfinite(value).all()
    assert value[:2, :768].mean() == 6 and value[:2, 768:1536].mean() == 3
    torch.testing.assert_close(value[:2, 1537], torch.tensor([-7., -6.]))
    assert not value[-1].any()


@pytest.mark.parametrize("field", ["middle", "times", "wave", "offset", "sentence"])
def test_combination_rejects_bins_or_mismatched_provenance(field):
    clip, record, content = audio_fixture()
    if field == "middle": record["middle"] = torch.zeros(24, 768)
    if field == "times": record["times"] += .01
    if field == "wave": record["record"]["wave_sha256"] = "other"
    if field == "offset": record["record"]["audio_offset_s"] = 0.
    if field == "sentence": record["sentence_id"] = "other"
    with pytest.raises(ValueError): combine_frame_features(content, clip, record)


def test_fit_statistics_use_valid_train_frames_population_variance_and_keep_between_clip_dc():
    x = torch.tensor([[[2., 7.], [4., 7.], [float("nan"), float("nan")]],
                      [[12., 7.], [14., 7.], [16., 7.]]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    before = x.clone()
    stats = fit_feature_statistics(x, mask, ["a", "b"])
    selected = x[mask]
    torch.testing.assert_close(stats["mean"], selected.mean(0))
    torch.testing.assert_close(stats["std"], selected.std(0, correction=0).clamp_min(.001))
    assert stats["count"] == 5 and stats["ddof"] == 0 and stats["fit_clip_ids"] == ["a", "b"]
    torch.testing.assert_close(x, before, equal_nan=True)
    assert stats["mean"][0] > 9  # Centering each clip would erase this value.


def renderer_fixture():
    def split(cid, sentence):
        return {"q": {"valid": torch.ones(1, 3, dtype=torch.bool),
                      "times": torch.arange(3, dtype=torch.float64)[None] / 25,
                      "content": torch.zeros(1, 3, 768), "clip_id": [cid], "sentence_id": [sentence],
                      "speaker": ["p1"], "emotion_id": torch.tensor([1])}}
    cache = {"schema": "predictable_renderer_cache_v1", "splits": {
        "train": split("fit", "s1"), "validation": split("dev", "s2")}}
    rows = {cid: {"sentence": sentence, "speaker": "p1", "emotion": 1}
            for cid, sentence in [("fit", "s1"), ("dev", "s2")]}
    return cache, rows


def test_cache_allowlist_rejects_test_extra_unlisted_duplicates_and_manifest_nontrain(tmp_path):
    cache, rows = renderer_fixture()
    assert validate_cache(cache, rows) == {"fit", "dev"}
    for mutate in (lambda c: c["splits"].update(test=c["splits"]["train"]),
                   lambda c: c["splits"]["validation"]["q"].update(clip_id=["not_allowed"]),
                   lambda c: c["splits"]["validation"]["q"].update(clip_id=["fit"])):
        altered = copy.deepcopy(cache); mutate(altered)
        with pytest.raises(ValueError): validate_cache(altered, rows)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"split": "test", "clip_id": "test", "artifact_sha256": "abc"}))
    with pytest.raises(ValueError, match="TRAIN"): read_manifest(manifest)


def test_sidecar_requires_matching_frozen_implementation_and_fullframe_record(tmp_path):
    clip, record, _ = audio_fixture()
    model = {key: key + "-hash" for key in ("model_id", "model_sha256", "config_sha256",
             "model_source_sha256", "audio_encoder_source_sha256", "frontend_source_sha256")}
    geometry = {"hop_samples": 320}
    saved = {"provenance": {"schema": "predictable_audio_v1", "layers": [2, 4, 6],
                           "geometry": geometry, "model": model}, "clips": [record]}
    path = tmp_path / "audio.pt"; torch.save(saved, path)
    index, sources = index_sidecars([path], model, geometry)
    assert index[clip["clip_id"]][1] == str(path.resolve()) and sources[str(path.resolve())] == sha(path)
    altered = dict(model, model_source_sha256="other-code")
    with pytest.raises(ValueError): index_sidecars([path], altered, geometry)


def test_audio_cache_cli_exact_roles_and_stats_never_fit_development(tmp_path, monkeypatch):
    root = tmp_path / "native"; root.mkdir()
    rows, splits = [], {}
    for role, cid, sentence, level in (("train", "fit", "s1", 2.), ("validation", "dev", "s2", 100.)):
        times = np.arange(4, dtype=np.float64) / 25
        valid = np.array([True, True, True, False])
        content = np.full((4, 768), level, dtype=np.float32)
        provenance = {"schema": "native_affect_style_v4.1", "clock_evidence": "embedded_video", "fps": 25,
                      "audio_path": "/authorized.wav", "audio_sha256": "wave", "audio_offset_s": 0.}
        path = root / (cid + ".npz")
        np.savez(path, content=content, times=times, mask=valid, provenance=json.dumps(provenance))
        rows.append({"clip_id": cid, "sentence": sentence, "speaker": "p1", "emotion": 1, "split": "train",
                     "artifact": path.name, "artifact_sha256": sha(path)})
        splits[role] = {"q": {"clip_id": [cid], "sentence_id": [sentence], "speaker": ["p1"],
            "emotion_id": torch.tensor([1]), "content": torch.from_numpy(content)[None].half(),
            "times": torch.from_numpy(times)[None], "valid": torch.from_numpy(valid)[None]}}
    rows.append({**rows[0], "clip_id": "unused", "artifact": "DO_NOT_OPEN.npz"})
    manifest = tmp_path / "train.jsonl"; manifest.write_text("\n".join(json.dumps(row) for row in rows))
    renderer = tmp_path / "renderer.pt"
    torch.save({"schema": "predictable_renderer_cache_v1", "splits": splits}, renderer)
    monkeypatch.setattr(module, "load_extractor", lambda *a: (SimpleNamespace(blocks=list(range(8))), {}, {}))
    opened = []
    def fake_extract(clip, *args):
        opened.append(clip["clip_id"])
        level = 2. if clip["clip_id"] == "fit" else 100.
        return {"clip_id": clip["clip_id"], "sentence_id": clip["sentence_id"], "valid": clip["valid"],
                "times": clip["times"], "middle": torch.full((4, 768), level).half(),
                "prosody": torch.full((4, 4), level), "record": {"wave_sha256": "wave", "audio_offset_s": 0.}}
    monkeypatch.setattr(module, "extract", fake_extract)
    output = tmp_path / "audio.pt"
    monkeypatch.setattr(module.sys, "argv", ["prepare_label_guided_audio_cache.py", "--renderer-cache", str(renderer),
        "--native-train-manifest", str(manifest), "--native-root", str(root), "--model-dir", str(tmp_path),
        "--output", str(output), "--device", "cpu"])
    module.main()
    saved = torch.load(output, weights_only=False)
    assert opened == ["fit", "dev"] and saved["schema"] == module.SCHEMA
    assert saved["splits"]["train"]["features"].shape == (1, 4, 1540)
    assert (saved["splits"]["validation"]["features"][0, :3] == 100).all()
    assert (saved["feature_stats"]["mean"] == 2).all() and saved["feature_stats"]["count"] == 3
    assert saved["feature_stats"]["fit_clip_ids"] == ["fit"]
    assert saved["provenance"]["renderer_cache_sha256"] == sha(renderer)
    assert saved["provenance"]["native_motion_arrays_read"] is False
    summary = json.loads(output.with_suffix(".json").read_text(encoding="utf8"))
    assert summary["cache_sha256"] == sha(output)
