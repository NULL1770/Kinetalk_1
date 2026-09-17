"""Extend an existing audio cache on its authorized native TRAIN clocks only.

Only extra valid-frame middle/prosody features are persisted. Native motion and
audio arrays are never opened, and the original normalization is never refit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_emotion2vec_pilot import load_extractor, sha
from scripts.extract_predictable_audio import extract
from scripts.prepare_label_guided_audio_cache import read_manifest

SCHEMA = "full_native_audio_delta_v1"
LAYERS = [2, 4, 6]
MIDDLE_MAX_ATOL = 2e-3
MIDDLE_RMS_ATOL = 1e-4
PROSODY_MAX_ATOL = 1e-6
MODEL_KEYS = ("model_id", "model_sha256", "config_sha256", "model_source_sha256",
              "audio_encoder_source_sha256", "frontend_source_sha256")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tensor_sha(value):
    """Exact dtype, shape and storage-byte binding, also used by the data store."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def scientific_model(model):
    if not isinstance(model, dict) or any(not model.get(k) for k in MODEL_KEYS):
        raise ValueError("Missing pinned model identity or implementation hashes")
    return {k: v for k, v in model.items() if k != "model_dir"}


def contained_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its declared root")
    return path


def native_artifact_path(root, artifact):
    """Old manifests may use absolute artifacts, still confined to native_root."""
    root = Path(root).resolve()
    path = (root / str(artifact)).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Native artifact escapes its declared root")
    return path


def validate_audio(audio, rows):
    """Return original-order authorized rows without opening native artifacts."""
    if (audio.get("schema") != "label_guided_audio_cache_v1"
            or set(audio.get("splits", {})) != {"train", "validation"}):
        raise ValueError("Require old label-guided train/validation audio cache")
    provenance = audio.get("provenance", {})
    scientific_model(provenance.get("model"))
    if provenance.get("layers") != LAYERS or not isinstance(provenance.get("geometry"), dict):
        raise ValueError("Old cache layers or geometry are missing/different")
    records = {}
    for record in audio.get("records", []):
        cid = record.get("clip_id")
        if not isinstance(cid, str) or cid in records:
            raise ValueError("Duplicate/malformed old extraction record")
        records[cid] = record
    allowed, seen = [], set()
    for role in ("train", "validation"):
        q = audio["splits"][role]
        features, valid, times = (q[k] for k in ("features", "valid", "times"))
        ids, sentences = q["clip_id"], q["sentence_id"]
        if (not all(torch.is_tensor(x) for x in (features, valid, times))
                or features.dtype != torch.float32 or features.ndim != 3
                or features.shape[1:] != (96, 1540)
                or valid.dtype != torch.bool or valid.shape != features.shape[:2]
                or not times.is_floating_point() or times.shape != valid.shape
                or len(ids) != len(valid) or len(sentences) != len(valid)
                or not valid.any(1).all() or not torch.isfinite(times).all()
                or not torch.isfinite(features[valid]).all()
                or not torch.allclose(times[:, 1:].double() - times[:, :-1].double(),
                                      torch.full_like(times[:, 1:].double(), .04), atol=1e-5, rtol=0)):
            raise ValueError("Invalid old audio storage/clock: " + role)
        for index, cid in enumerate(ids):
            if (not isinstance(cid, str) or cid in seen or cid not in rows
                    or cid not in records or rows[cid].get("split") != "train"
                    or str(rows[cid]["sentence"]) != str(sentences[index])
                    or records[cid].get("split") != role):
                raise ValueError("Unauthorized/duplicate/mismatched old clip: " + str(cid))
            seen.add(cid)
            allowed.append({"clip_id": cid, "role": role, "index": index,
                            "sentence_id": str(sentences[index]), "record": records[cid]})
    if set(records) != seen:
        raise ValueError("Old extraction records do not exactly cover the old allowlist")
    return allowed


def validate_native_arrays(content, native_times, native_mask, provenance,
                           center_features, center_times, center_valid):
    """Validate native arrays and exact FP32 center content; return crop origin."""
    if (not isinstance(provenance, dict)
            or not str(provenance.get("schema", "")).startswith("native_affect_style_v4")
            or provenance.get("clock_evidence") != "embedded_video"
            or float(provenance.get("fps", 0)) != 25.):
        raise ValueError("Native clock provenance differs")
    # Native artifacts historically store the 768-d content in FP16.  The
    # immutable cache contract compares its values after conversion to the
    # old cache dtype; extraction itself never consumes this array (the
    # waveform is re-encoded by the pinned extractor).  Accept FP16/FP32 and
    # perform all finite/shape checks in FP32 without rewriting the artifact.
    if (content.dtype not in (np.float16, np.float32) or native_times.ndim != 1 or len(native_times) < 2
            or content.shape != (len(native_times), 768)
            or native_mask.shape != native_times.shape or not np.isin(native_mask, [0, 1]).all()
            or not np.isfinite(content).all() or not np.isfinite(native_times).all()
            or not np.allclose(np.diff(native_times), .04, rtol=0, atol=1e-5)):
        raise ValueError("Invalid native FP32 content/clock/mask")
    target = center_times.detach().cpu().double().numpy()
    valid = center_valid.detach().cpu().numpy()
    if (center_features.dtype != torch.float32 or center_features.shape != (len(target), 1540)
            or center_valid.dtype != torch.bool or valid.shape != target.shape
            or target.ndim != 1 or not valid.any() or not np.isfinite(target).all()
            or not np.allclose(np.diff(target), .04, rtol=0, atol=1e-5)):
        raise ValueError("Invalid old center clock or features")
    start = int(np.argmin(np.abs(native_times - target[0])))
    count = min(len(target), len(native_times) - start)
    if not np.allclose(native_times[start:start + count], target[:count], atol=1e-7, rtol=0):
        raise ValueError("Old center timestamp is absent/different in native clock")
    expected_valid = np.zeros(len(target), dtype=bool)
    expected_valid[:count] = native_mask[start:start + count].astype(bool)
    if not np.array_equal(expected_valid, valid):
        raise ValueError("Old/native center validity differs")
    observed = np.flatnonzero(valid)
    actual = torch.from_numpy(content[start + observed].astype(np.float32, copy=False))
    expected = center_features.detach().cpu()[center_valid.cpu(), :768]
    if not torch.equal(actual, expected):
        raise ValueError("Native FP32 center content differs from old audio")
    return start


def overlap_statistics(actual, expected):
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError("Empty/mismatched overlap")
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Nonfinite overlap")
    delta = actual - expected
    return {"count": delta.numel(), "max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()),
            "changed_fraction": float((delta != 0).double().mean()),
            "exact": bool(torch.equal(actual, expected))}


def validate_overlap(middle, prosody, center_features, center_valid, center_start):
    """Check post-FP16 middle against old FP32, with immutable dual tolerances."""
    ids = torch.where(center_valid.cpu())[0] + center_start
    old = center_features.detach().cpu()[center_valid.cpu()]
    result = {"middle": overlap_statistics(middle.cpu()[ids].float(), old[:, 768:1536]),
              "prosody": overlap_statistics(prosody.cpu()[ids].float(), old[:, 1536:1540])}
    if (result["middle"]["max_abs"] > MIDDLE_MAX_ATOL
            or result["middle"]["rms"] > MIDDLE_RMS_ATOL
            or result["prosody"]["max_abs"] > PROSODY_MAX_ATOL):
        raise ValueError("Locked center overlap tolerance failed: " + json.dumps(result))
    return result


def _load_native(row, root, selected, q):
    path = native_artifact_path(root, row["artifact"])
    if sha(path) != row["artifact_sha256"]:
        raise ValueError("Native artifact hash changed: " + selected["clip_id"])
    with np.load(path, allow_pickle=False) as data:
        # Deliberately do not access any native motion/audio payload.
        content = np.asarray(data["content"])
        times = np.asarray(data["times"], dtype=np.float64)
        mask = np.asarray(data["mask"])
        provenance = json.loads(str(data["provenance"].item()))
    index = selected["index"]
    start = validate_native_arrays(content, times, mask, provenance,
                                   q["features"][index], q["times"][index], q["valid"][index])
    for key in ("audio_path", "audio_sha256", "audio_offset_s"):
        if key not in provenance:
            raise ValueError("Missing native waveform binding: " + key)
    wave = Path(provenance["audio_path"])
    if not wave.is_absolute() or not np.isfinite(float(provenance["audio_offset_s"])):
        raise ValueError("Waveform path must be absolute with finite clock offset")
    old = selected["record"]
    expected = {"artifact_sha256": row["artifact_sha256"], "crop_start": start,
                "native_frames": len(times), "crop_in_range_frames": min(96, len(times) - start),
                "wave_sha256": provenance["audio_sha256"],
                "audio_offset_s": float(provenance["audio_offset_s"])}
    if any(old.get(k) != v for k, v in expected.items()):
        raise ValueError("Native metadata differs from authorized old extraction record")
    if Path(old.get("wave_path", "")).resolve() != wave.resolve():
        raise ValueError("Native waveform location differs from authorized old record")
    if sha(wave) != provenance["audio_sha256"]:
        raise ValueError("Authorized waveform changed")
    clip = {"clip_id": selected["clip_id"], "sentence_id": selected["sentence_id"],
            "valid": torch.from_numpy(mask.astype(bool)), "times": torch.from_numpy(times),
            "metadata": {"provenance": provenance}}
    return clip, content, start


def _binding(source, row, selected, q, clip):
    index = selected["index"]
    return {"old_audio_sha256": source["audio_sha256"],
            "native_manifest_sha256": source["manifest_sha256"],
            "native_artifact_sha256": row["artifact_sha256"],
            "wave_sha256": clip["metadata"]["provenance"]["audio_sha256"],
            "model_sha256": canonical_sha(scientific_model(source["model"])),
            "geometry_sha256": canonical_sha(source["geometry"]), "layers": LAYERS,
            "center_features_sha256": tensor_sha(q["features"][index]),
            "center_valid_sha256": tensor_sha(q["valid"][index]),
            "center_times_sha256": tensor_sha(q["times"][index])}


def _extra_indices(valid, start, length):
    ids = torch.where(valid.cpu())[0]
    return ids[(ids < start) | (ids >= start + length)]


def validate_delta(payload, selected, binding, valid, start):
    """Pure validation of a loaded delta against source-derived metadata."""
    expected = {"schema": SCHEMA, "clip_id": selected["clip_id"], "role": selected["role"],
                "sentence_id": selected["sentence_id"], "native_frames": len(valid),
                "center_start": start, "center_length": 96, "binding": binding}
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("Delta metadata/source binding differs")
    ids = payload.get("native_indices")
    if (not torch.is_tensor(ids) or ids.dtype != torch.int64
            or not torch.equal(ids.cpu(), _extra_indices(valid, start, 96))):
        raise ValueError("Delta indices are not exactly the extra valid native frames")
    for key, dim, dtype in (("middle", 768, torch.float16), ("prosody", 4, torch.float32)):
        tensor = payload.get(key)
        if (not torch.is_tensor(tensor) or tensor.dtype != dtype or tensor.shape != (len(ids), dim)
                or not torch.isfinite(tensor).all()):
            raise ValueError("Invalid delta feature storage: " + key)
    checks = payload.get("overlap_check", {})
    observed_center = int(valid[start:min(start + 96, len(valid))].sum())
    for key in ("content", "middle", "prosody"):
        stat = checks.get(key, {})
        expected_count = observed_center * (4 if key == "prosody" else 768)
        if (not isinstance(stat.get("count"), int) or stat["count"] != expected_count or expected_count <= 0
                or not isinstance(stat.get("exact"), bool)
                or any(not isinstance(stat.get(k), (float, int)) or not np.isfinite(stat[k])
                       or stat[k] < 0 for k in ("max_abs", "rms", "changed_fraction"))
                or stat["changed_fraction"] > 1 or stat["rms"] > stat["max_abs"] + 1e-15
                or (stat["max_abs"] == 0) != (stat["rms"] == 0)
                or (stat["max_abs"] == 0) != (stat["changed_fraction"] == 0)
                or stat["exact"] != (stat["max_abs"] == 0 and stat["changed_fraction"] == 0)):
            raise ValueError("Malformed overlap audit: " + key)
    if (not checks["content"]["exact"] or checks["middle"]["max_abs"] > MIDDLE_MAX_ATOL
            or checks["middle"]["rms"] > MIDDLE_RMS_ATOL
            or checks["prosody"]["max_abs"] > PROSODY_MAX_ATOL):
        raise ValueError("Delta overlap audit exceeds locked tolerances")
    return payload


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def _entry(payload, filename, digest):
    return {"file": filename, "sha256": digest, "status": "complete",
            **{k: payload[k] for k in ("role", "sentence_id", "native_frames", "center_start", "center_length")},
            "extra_valid_frames": len(payload["native_indices"]),
            "native_artifact_sha256": payload["binding"]["native_artifact_sha256"],
            "wave_sha256": payload["binding"]["wave_sha256"]}


def prepare(args):
    """Prepare/resume a hash-bound selected allowlist; never normalize features."""
    audio_path, manifest_path = Path(args.audio).resolve(), Path(args.native_manifest).resolve()
    root, output = Path(args.native_root).resolve(), Path(args.output).resolve()
    if output in (root, audio_path, manifest_path) or audio_path.is_relative_to(output):
        raise ValueError("Output must be separate from immutable input files/root")
    max_clips = getattr(args, "max_clips", None)
    clip_ids_path = getattr(args, "clip_ids", None)
    if max_clips is not None and (not isinstance(max_clips, int) or max_clips < 1):
        raise ValueError("max_clips must be a positive smoke allowlist limit")
    if max_clips is not None and clip_ids_path is not None:
        raise ValueError("Use exactly one smoke selection: clip_ids or max_clips")
    audio_hash, manifest_hash = sha(audio_path), sha(manifest_path)
    rows = read_manifest(manifest_path)
    audio = torch.load(audio_path, map_location="cpu", weights_only=False, mmap=True)
    allowed = validate_audio(audio, rows)
    clip_ids_hash = None
    if clip_ids_path is not None:
        clip_ids_path = Path(clip_ids_path).resolve()
        clip_ids_hash = sha(clip_ids_path)
        ids = json.loads(clip_ids_path.read_text(encoding="utf-8-sig"))
        if (not isinstance(ids, list) or not ids or not all(isinstance(cid, str) for cid in ids)
                or len(ids) != len(set(ids))
                or not set(ids).issubset(s["clip_id"] for s in allowed)):
            raise ValueError("clip_ids must be unique IDs exclusively in the old audio allowlist")
        ids = set(ids)
        allowed = [s for s in allowed if s["clip_id"] in ids]
    elif max_clips is not None:
        allowed = allowed[:max_clips]
    elif {role: sum(s["role"] == role for s in allowed) for role in ("train", "validation")} != {"train": 2315, "validation": 405}:
        raise ValueError("Formal extraction requires exactly the old 2315 train/405 validation clips")
    if not allowed:
        raise ValueError("Empty authorized allowlist")
    provenance = audio["provenance"]
    if provenance.get("native_train_manifest_sha256") != manifest_hash:
        raise ValueError("Native manifest differs from old cache provenance")
    # The upstream model hashes do not bind the alignment/prosody wrapper.
    # Verify the exact two historical extraction implementations as well.
    old_scripts = provenance.get("source_sha256", {})
    for name in ("extract_predictable_audio.py", "extract_emotion2vec_pilot.py"):
        matches = [value for key, value in old_scripts.items()
                   if str(key).replace("\\", "/").split("/")[-1] == name]
        if len(matches) != 1 or matches[0] != sha(Path(__file__).with_name(name)):
            raise ValueError("Historical extraction wrapper source differs: " + name)
    model, geometry, model_info = load_extractor(Path(args.model_dir), args.device)
    if (scientific_model(model_info) != scientific_model(provenance["model"])
            or geometry != provenance["geometry"] or max(LAYERS) >= len(model.blocks)):
        raise ValueError("Model, implementation, versions or geometry differ from old extraction")
    source = {"audio_sha256": audio_hash, "manifest_sha256": manifest_hash,
              "model": provenance["model"], "geometry": geometry, "layers": LAYERS}
    config = {"middle_max_atol": MIDDLE_MAX_ATOL, "middle_rms_atol": MIDDLE_RMS_ATOL,
              "prosody_max_atol": PROSODY_MAX_ATOL, "rtol": 0,
              "storage": {"middle": "float16", "prosody": "float32"},
              "normalization_recomputed": False, "max_clips": max_clips,
              "clip_ids_sha256": clip_ids_hash,
              "smoke": max_clips is not None or clip_ids_path is not None}
    selected_roles = {role: [s["clip_id"] for s in allowed if s["role"] == role]
                      for role in ("train", "validation")}
    identity = {"schema": SCHEMA, "source": source, "config": config, "selected_roles": selected_roles}
    manifest_file, complete_file = output / "manifest.json", output / "complete.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (any(manifest.get(k) != v for k, v in identity.items())
                or manifest.get("status") not in ("incomplete", "complete")
                or not isinstance(manifest.get("clips"), dict)
                or not set(manifest["clips"]).issubset(s["clip_id"] for s in allowed)):
            raise ValueError("Resume identity/config/selection/status differs")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Nonempty output without a bound manifest")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {**identity, "clips": {}, "status": "incomplete"}
        _atomic_json(manifest_file, manifest)
    declared = {contained_path(output, entry["file"]) for entry in manifest["clips"].values()}
    files = {p.resolve() for p in output.rglob("*") if p.is_file()}
    extras = files - declared - {manifest_file, complete_file}
    if extras:
        raise ValueError("Orphan/partial output files cannot masquerade as completed: " + str(sorted(map(str, extras))))
    if not declared.issubset(files):
        raise ValueError("Manifest references missing delta files")
    if manifest["status"] == "complete" and len(manifest["clips"]) != len(allowed):
        raise ValueError("Partial manifest masquerades as complete")
    if complete_file.exists():
        complete = json.loads(complete_file.read_text(encoding="utf-8"))
        expected = {"schema": SCHEMA, "status": "complete", "manifest_sha256": sha(manifest_file),
                    "selected_count": len(allowed), "completed_count": len(allowed)}
        if complete != expected or manifest["status"] != "complete":
            raise ValueError("Completion marker is stale/mismatched/partial")
    (output / "clips").mkdir(exist_ok=True)
    for position, selected in enumerate(allowed):
        cid, role, index = selected["clip_id"], selected["role"], selected["index"]
        q, row = audio["splits"][role], rows[cid]
        clip, content, start = _load_native(row, root, selected, q)
        binding = _binding(source, row, selected, q, clip)
        filename = "clips/" + hashlib.sha256(cid.encode()).hexdigest() + ".pt"
        path = contained_path(output, filename)
        if cid in manifest["clips"]:
            entry = manifest["clips"][cid]
            if entry.get("file") != filename or sha(path) != entry.get("sha256"):
                raise ValueError("Existing delta file location/hash differs")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_delta(payload, selected, binding, clip["valid"], start)
            if entry != _entry(payload, filename, sha(path)):
                raise ValueError("Existing manifest record differs from its validated payload")
            mode = "validated_resume"
        else:
            if manifest["status"] != "incomplete" or complete_file.exists() or path.exists():
                raise ValueError("Refusing mutation of complete or orphan output")
            result = extract(clip, model, geometry, LAYERS, args.device)
            if (result.get("clip_id") != cid or result.get("sentence_id") != selected["sentence_id"]
                    or not torch.equal(result["valid"].cpu(), clip["valid"])
                    or not torch.equal(result["times"].cpu(), clip["times"])
                    or result["record"].get("wave_sha256") != binding["wave_sha256"]
                    or float(result["record"].get("audio_offset_s", float("nan")))
                    != float(clip["metadata"]["provenance"]["audio_offset_s"])):
                raise ValueError("Full extractor identity/clock/wave binding differs")
            middle, prosody = result["middle"].cpu(), result["prosody"].cpu()
            if (middle.dtype != torch.float16 or middle.shape != (len(clip["valid"]), 768)
                    or prosody.dtype != torch.float32 or prosody.shape != (len(clip["valid"]), 4)):
                raise ValueError("Full extractor did not return audited frame storage")
            check = validate_overlap(middle, prosody, q["features"][index], q["valid"][index], start)
            count = int(q["valid"][index].sum()) * 768
            check["content"] = {"count": count, "max_abs": 0., "rms": 0., "changed_fraction": 0., "exact": True}
            ids = _extra_indices(clip["valid"], start, 96)
            payload = {"schema": SCHEMA, "clip_id": cid, "role": role,
                       "sentence_id": selected["sentence_id"], "native_frames": len(clip["valid"]),
                       "center_start": start, "center_length": 96, "native_indices": ids,
                       "middle": middle[ids].clone(), "prosody": prosody[ids].clone(),
                       "binding": binding, "overlap_check": check}
            validate_delta(payload, selected, binding, clip["valid"], start)
            if (sha(native_artifact_path(root, row["artifact"])) != binding["native_artifact_sha256"]
                    or sha(clip["metadata"]["provenance"]["audio_path"]) != binding["wave_sha256"]):
                raise RuntimeError("Native artifact or waveform changed during extraction")
            _atomic_checkpoint(path, payload)
            manifest["clips"][cid] = _entry(payload, filename, sha(path))
            _atomic_json(manifest_file, manifest)
            mode = "extracted"
        print(json.dumps({"stage": "full_native_audio_delta", "done": position + 1,
                          "total": len(allowed), "clip_id": cid, "mode": mode,
                          "extra_valid_frames": len(payload["native_indices"])}), flush=True)
    if sha(audio_path) != audio_hash or sha(manifest_path) != manifest_hash:
        raise RuntimeError("Immutable source inputs changed during extraction")
    if clip_ids_path is not None and sha(clip_ids_path) != clip_ids_hash:
        raise RuntimeError("Smoke allowlist changed during extraction")
    if manifest["status"] != "complete":
        manifest["status"] = "complete"
        _atomic_json(manifest_file, manifest)
    complete = {"schema": SCHEMA, "status": "complete", "manifest_sha256": sha(manifest_file),
                "selected_count": len(allowed), "completed_count": len(allowed)}
    if not complete_file.exists():
        _atomic_json(complete_file, complete)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio", "native-manifest", "native-root", "model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--clip-ids", type=Path, help="Smoke only: JSON list of authorized old clip IDs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args)


if __name__ == "__main__":
    main()
