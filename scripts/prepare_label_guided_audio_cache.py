"""Uncentered frame audio on the exact existing fit/development motion clock.

Only manifest-authorized current-cache clips are opened. Native motion arrays
are never read. The old frozen B0/global cache is bound by hash and untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_emotion2vec_pilot import load_extractor, sha
from scripts.extract_predictable_audio import extract
from scripts.train_formal_predictable_projection import save_checkpoint, save_json
from scripts.train_predictable_renderer import state_hash

SCHEMA = "label_guided_audio_cache_v1"
LAYERS = (2, 4, 6)
FEATURE_SLICES = {"content": [0, 768], "middle": [768, 1536], "prosody": [1536, 1540]}


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    result = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Only native TRAIN metadata is accepted")
        cid = str(row["clip_id"])
        if cid in result: raise ValueError("Duplicate native clip: " + cid)
        if not row.get("artifact_sha256"): raise ValueError("Native artifact SHA256 is required")
        result[cid] = row
    return result


def validate_cache(cache, rows):
    if cache.get("schema") != "predictable_renderer_cache_v1" or set(cache.get("splits", {})) != {"train", "validation"}:
        raise ValueError("Require the existing train/validation renderer cache; no test split")
    seen = set()
    for role in ("train", "validation"):
        q = cache["splits"][role]["q"]
        ids, valid, times = list(map(str, q["clip_id"])), q["valid"], q["times"]
        if (valid.dtype != torch.bool or valid.ndim != 2 or times.shape != valid.shape
                or len(ids) != len(valid) or not valid.any(1).all()):
            raise ValueError("Invalid cache clock/validity for " + role)
        if not torch.isfinite(times).all() or not (times[:, 1:] > times[:, :-1]).all():
            raise ValueError("Cache timestamps must be finite and increasing")
        if q["content"].shape != (*valid.shape, 768): raise ValueError("Expected native frame content768")
        for index, cid in enumerate(ids):
            if cid in seen or cid not in rows: raise ValueError("Duplicate/unavailable authorized clip: " + cid)
            row = rows[cid]
            if str(q["sentence_id"][index]) != str(row["sentence"]):
                raise ValueError("Cache/manifest sentence differs: " + cid)
            if "speaker" in q and str(q["speaker"][index]) != str(row["speaker"]):
                raise ValueError("Cache/manifest speaker differs: " + cid)
            if "emotion_id" in q and int(q["emotion_id"][index]) != int(row["emotion"]):
                raise ValueError("Cache/manifest emotion differs: " + cid)
            seen.add(cid)
    return seen


def load_authorized_native(row, native_root, times, valid, cached_content):
    """Read content/clock/provenance only, never source['motion']."""
    if row.get("split") != "train": raise ValueError("Only authorized native TRAIN clips may be opened")
    root = Path(native_root).resolve()
    path = (root / str(row["artifact"])).resolve()
    if not path.is_relative_to(root): raise ValueError("Native artifact escapes declared root")
    if sha(path) != row["artifact_sha256"]: raise ValueError("Native artifact changed: " + str(path))
    with np.load(path, allow_pickle=False) as source:
        required = ("content", "times", "mask", "provenance")
        if any(key not in source for key in required): raise ValueError("Incomplete native audio/clock artifact")
        content = np.asarray(source["content"])
        native_times = np.asarray(source["times"], dtype=np.float64)
        native_mask = np.asarray(source["mask"])
        provenance = json.loads(str(source["provenance"].item()))
    if (not str(provenance.get("schema", "")).startswith("native_affect_style_v4")
            or provenance.get("clock_evidence") != "embedded_video"):
        raise ValueError("Native clock provenance differs")
    if (native_times.ndim != 1 or len(native_times) < 2
            or content.shape != (len(native_times), 768) or native_mask.shape != native_times.shape
            or not np.isin(native_mask, [0, 1]).all() or not np.isfinite(content).all()
            or not np.isfinite(native_times).all() or np.any(np.diff(native_times) <= 0)):
        raise ValueError("Invalid native content/clock/mask")
    fps = float(provenance["fps"])
    target_times = times.detach().cpu().double().numpy()
    expected_valid = valid.detach().cpu().bool().numpy()
    if (target_times.ndim != 1 or len(target_times) < 2 or expected_valid.shape != target_times.shape
            or cached_content.shape != (len(target_times), 768) or not expected_valid.any()
            or not np.isfinite(target_times).all() or fps != 25.
            or not np.allclose(np.diff(native_times), 1 / fps, rtol=0, atol=1e-5)
            or not np.allclose(np.diff(target_times), 1 / fps, rtol=0, atol=1e-5)):
        raise ValueError("Expected exact existing 25Hz clock")
    first = int(np.argmin(np.abs(native_times - target_times[0])))
    if abs(native_times[first] - target_times[0]) > 1e-7:
        raise ValueError("Cache start is absent from native timestamps")
    length = len(target_times)
    count = min(length, len(native_times) - first)
    if not np.allclose(native_times[first:first + count], target_times[:count], rtol=0, atol=1e-7):
        raise ValueError("Cache/native crop timestamps differ")
    mask = np.zeros(length, dtype=bool); mask[:count] = native_mask[first:first + count].astype(bool)
    if not np.array_equal(mask, expected_valid): raise ValueError("Cache/native valid masks differ")
    frames = np.zeros((length, 768), dtype=np.float32)
    frames[:count] = content[first:first + count]
    frames = torch.from_numpy(frames)
    # Historical cache may store content in FP16. Verify its actual storage
    # contract, then preserve the native FP32 values in the new feature cache.
    torch.testing.assert_close(frames.to(cached_content.dtype)[valid], cached_content.cpu()[valid], rtol=0, atol=0)
    for key in ("audio_path", "audio_sha256", "audio_offset_s"):
        if key not in provenance: raise ValueError("Native waveform binding missing: " + key)
    if not np.isfinite(float(provenance["audio_offset_s"])): raise ValueError("Nonfinite audio offset")
    clip = {"clip_id": str(row["clip_id"]), "sentence_id": str(row["sentence"]),
        "valid": valid.cpu().clone(), "times": times.cpu().clone(),
        "metadata": {"provenance": provenance}}
    return clip, frames, {"artifact": str(path), "artifact_sha256": row["artifact_sha256"],
        "crop_start": first, "native_frames": len(native_times), "crop_in_range_frames": count,
        "wave_path": provenance["audio_path"], "wave_sha256": provenance["audio_sha256"],
        "audio_offset_s": float(provenance["audio_offset_s"])}


def index_sidecars(paths, model_info, geometry):
    """Only actual full-frame predictable_audio_v1 records can be reused."""
    index, sources = {}, {}
    for path in paths:
        path = Path(path).resolve()
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        provenance = saved.get("provenance", {})
        if (provenance.get("schema") != "predictable_audio_v1"
                or provenance.get("layers") != list(LAYERS)
                or provenance.get("geometry") != geometry
                or any(provenance.get("model", {}).get(key) != model_info[key]
                       for key in ("model_id", "model_sha256", "config_sha256", "model_source_sha256",
                                   "audio_encoder_source_sha256", "frontend_source_sha256"))):
            raise ValueError("Sidecar is not the matching uncentered native frame extraction: " + str(path))
        sources[str(path)] = sha(path)
        for record in saved.get("clips", []):
            cid = str(record["clip_id"])
            if cid in index: raise ValueError("Duplicate sidecar clip: " + cid)
            index[cid] = (record, str(path))
    return index, sources


def combine_frame_features(content, clip, record):
    valid, times = clip["valid"], clip["times"]
    if (record.get("clip_id") != clip["clip_id"] or record.get("sentence_id") != clip["sentence_id"]
            or not torch.equal(record["valid"].cpu(), valid)
            or not torch.equal(record["times"].cpu(), times)):
        raise ValueError("Frame feature record identity/clock differs")
    native_prov, evidence = clip["metadata"]["provenance"], record["record"]
    if (evidence.get("wave_sha256") != native_prov["audio_sha256"]
            or float(evidence.get("audio_offset_s", float("nan"))) != float(native_prov["audio_offset_s"])):
        raise ValueError("Frame sidecar waveform binding differs")
    middle, prosody = record["middle"].cpu().float(), record["prosody"].cpu().float()
    if content.shape != (len(valid), 768) or middle.shape != content.shape or prosody.shape != (len(valid), 4):
        raise ValueError("Require frame content768/middle768/prosody4, never pooled bins")
    features = torch.cat((content.float(), middle, prosody), -1)
    if not torch.isfinite(features[valid]).all(): raise ValueError("Nonfinite valid frame audio features")
    return torch.where(valid[:, None], features, 0.)


def fit_feature_statistics(features, valid, fit_clip_ids):
    """Population mean/std on train observed frames only; no clip centering."""
    if features.ndim != 3 or valid.shape != features.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("Frame features and Boolean validity must agree")
    if len(fit_clip_ids) != len(features) or len(set(fit_clip_ids)) != len(fit_clip_ids):
        raise ValueError("Unique fitting clip IDs required")
    count = int(valid.sum())
    if count < 2: raise ValueError("At least two fit frames required")
    total = torch.zeros(features.shape[-1], dtype=torch.float64)
    square = torch.zeros_like(total)
    for frames, mask in zip(features, valid):
        x = frames[mask].double()
        if not torch.isfinite(x).all(): raise ValueError("Nonfinite fitting audio")
        total += x.sum(0); square += x.square().sum(0)
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(0.)
    return {"mean": mean.float(), "std": variance.sqrt().clamp_min(1e-3).float(),
        "count": count, "ddof": 0, "std_floor": 1e-3,
        "fit_clip_ids": list(fit_clip_ids), "source": "train_valid_native_frames_only",
        "application": "(raw_features - fit_mean)/fit_std then mask; never clip-center"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("renderer-cache", "native-train-manifest", "native-root", "model-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--frame-sidecar", type=Path, action="append", default=[])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Fresh audio cache and sidecar output paths required")
    torch.set_num_threads(4)
    rows = read_manifest(args.native_train_manifest)
    cache_sha = sha(args.renderer_cache)
    manifest_sha = sha(args.native_train_manifest)
    cache = torch.load(args.renderer_cache, map_location="cpu", weights_only=False, mmap=True)
    allowed = validate_cache(cache, rows)
    model, geometry, model_info = load_extractor(args.model_dir, args.device)
    if any(number >= len(model.blocks) for number in LAYERS): raise ValueError("Missing audited intermediate layers")
    sidecars, sidecar_sources = index_sidecars(args.frame_sidecar, model_info, geometry)
    splits, records, reused, extracted = {}, [], 0, 0
    for role in ("train", "validation"):
        q = cache["splits"][role]["q"]
        ids = list(map(str, q["clip_id"]))
        features = torch.empty((len(ids), q["valid"].shape[1], 1540), dtype=torch.float32)
        for index, cid in enumerate(ids):
            if cid not in allowed: raise ValueError("Unapproved extraction clip")
            clip, content, evidence = load_authorized_native(rows[cid], args.native_root,
                q["times"][index], q["valid"][index], q["content"][index])
            if cid in sidecars:
                record, sidecar_path = sidecars[cid]
                # Reusing a frame sidecar also checks its original wave bytes.
                if sha(Path(evidence["wave_path"])) != evidence["wave_sha256"]:
                    raise ValueError("Sidecar's original waveform changed")
                reused += 1; origin = {"mode": "reused_frame_sidecar", "path": sidecar_path}
            else:
                record = extract(clip, model, geometry, list(LAYERS), args.device)
                extracted += 1; origin = {"mode": "fresh_pinned_wave_extraction"}
            features[index] = combine_frame_features(content, clip, record)
            records.append({"split": role, "clip_id": cid, **evidence, **origin,
                "extraction_record": record["record"]})
            if (index + 1) % 50 == 0 or index + 1 == len(ids):
                print(json.dumps({"stage": "frame_audio", "split": role, "done": index + 1,
                    "total": len(ids), "reused": reused, "extracted": extracted}), flush=True)
        splits[role] = {"features": features, "valid": q["valid"].cpu().clone(),
            "times": q["times"].cpu().clone(), "clip_id": ids,
            "sentence_id": list(map(str, q["sentence_id"]))}
    stats = fit_feature_statistics(splits["train"]["features"], splits["train"]["valid"], splits["train"]["clip_id"])
    if sha(args.renderer_cache) != cache_sha or sha(args.native_train_manifest) != manifest_sha:
        raise RuntimeError("Immutable source inputs changed during extraction")
    sources = [Path(__file__), Path(__file__).with_name("extract_predictable_audio.py"),
               Path(__file__).with_name("extract_emotion2vec_pilot.py")]
    provenance = {"schema": SCHEMA, "renderer_cache": str(args.renderer_cache.resolve()),
        "renderer_cache_sha256": cache_sha, "native_train_manifest": str(args.native_train_manifest.resolve()),
        "native_train_manifest_sha256": manifest_sha, "native_root": str(args.native_root.resolve()),
        "model": model_info, "geometry": geometry, "layers": list(LAYERS), "feature_slices": FEATURE_SLICES,
        "source_sha256": {str(path.resolve()): sha(path) for path in sources}, "sidecar_sha256": sidecar_sources,
        "storage": "raw uncentered per-frame content FP32 + middle FP16 extraction promoted to FP32 + prosody FP32",
        "middle_definition": "equal mean of per-frame LayerNorm blocks2/4/6, no temporal centering or pooling",
        "prosody": ["log_f0_unvoiced_zero", "log_rms", "periodicity", "voiced"],
        "raw_wave_normalization": "Official emotion2vec waveform LayerNorm is retained; absolute log energy is measured separately before waveform normalization",
        "statistics_fit_role": "train only", "fit_stat_tensor_sha256": state_hash({k: stats[k] for k in ("mean", "std")}),
        "reused_clips": reused, "fresh_extracted_clips": extracted,
        "native_motion_arrays_read": False, "test_loaded": False, "default_replaced": False,
        "contract": "Existing fixed native96-frame roles/clock preserved. No new split, no target labels in audio features, no query clip centering."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, {"schema": SCHEMA, "splits": splits, "feature_stats": stats,
                                 "records": records, "provenance": provenance})
    save_json(args.output.with_suffix(".json"), {"schema": SCHEMA, "cache_sha256": sha(args.output),
        "shapes": {role: list(value["features"].shape) for role, value in splits.items()},
        "train_valid_frames": stats["count"], "feature_stats": {k: v for k, v in stats.items() if k not in ("mean", "std")},
        "provenance": provenance})
    print(json.dumps({"complete": True, "output": str(args.output.resolve()), "sha256": sha(args.output),
        "reused": reused, "extracted": extracted, "train_valid_frames": stats["count"]}), flush=True)


if __name__ == "__main__":
    main()
