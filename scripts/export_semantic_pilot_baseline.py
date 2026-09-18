"""Export pinned run12 acoustic inference on metadata-selected historical fit clips.

Per-clip arrays contain baseline52, native features, frozen global/local affect,
and identity from independent neutral references. Query motion is only used by
the native loader for data binding and, optionally, a separate render reference.
No query motion reaches ``infer_acoustic_baseline``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.export_clocked_fullface_examples import load_identities, load, read, write
from scripts.full_native_context_data import NativeContextStore
from scripts.full_staged_data import sha, _renderer_cache_binding
from scripts.joint_prior_audio_context import load_frozen_audio, assert_source_binding
from scripts.train_full_staged import base_forward
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input

SCHEMA = "semantic_pilot_frozen_baseline_v1"


def select_fit_queries(selection, query):
    """Validate selection against fit metadata before any selected raw clips load."""
    picks = selection.get("clips", [])
    if not isinstance(picks, list) or not picks:
        raise ValueError("Nonempty fixed metadata selection required")
    mapping = {cid: index for index, cid in enumerate(query["clip_id"])}
    if len(mapping) != len(query["clip_id"]):
        raise ValueError("Duplicate fit query IDs")
    seen, indices = set(), []
    for row in picks:
        cid = row.get("clip_id")
        if (not isinstance(cid, str) or not cid or cid in (".", "..")
                or any(character in cid for character in "/\\:") or cid in seen):
            raise ValueError("Unique safe selected clip IDs required")
        if cid not in mapping:
            raise ValueError("Selected clip outside historical fit membership")
        index = mapping[cid]
        if (row.get("sentence") != query["sentence_id"][index]
                or row.get("speaker_name") != query["speaker"][index]
                or row.get("speaker") != int(query["speaker_id"][index])
                or row.get("emotion") != int(query["emotion_id"][index])):
            raise ValueError("Selection and fit query metadata differ")
        seen.add(cid)
        indices.append(index)
    return picks, indices


@torch.no_grad()
def infer_acoustic_baseline(system, audio, content, features, valid, identity,
                            clip_id, steps, device):
    """Only acoustic inputs and independent identity; no query-motion argument."""
    n = len(valid)
    if (valid.dtype != torch.bool or valid.ndim != 1 or not valid.any()
            or content.shape != (n, 768) or features.shape != (n, 1540)
            or not torch.isfinite(content[valid]).all()
            or not torch.isfinite(features[valid]).all()
            or not torch.equal(content[valid], features[valid, :768])):
        raise ValueError("Matching finite acoustic arrays and nonempty native mask required")
    length = ((n + 15) // 16) * 16
    x = torch.zeros(1, length, 768, device=device)
    f = torch.zeros(1, length, 1540, device=device)
    mask = torch.zeros(1, length, dtype=torch.bool, device=device)
    x[0, :n] = torch.where(valid[:, None], content, 0.).to(device)
    f[0, :n] = torch.where(valid[:, None], features, 0.).to(device)
    mask[0, :n] = valid.to(device)
    base = base_forward(system, x, mask)
    affect = audio(f, mask)
    seed = int.from_bytes(hashlib.sha256(("clocked_fullface:42:" + clip_id).encode()).digest()[:8], "little") % (2**63 - 1)
    noise = torch.randn(1, length, 52, generator=torch.Generator().manual_seed(seed)).to(device)
    output = system.generate(x, mask, identity, affect, initial_noise=noise, steps=steps, base=base)["motion"]
    result = {
        "baseline52": output[0, :n].cpu(),
        "affect_global": affect["global"][0].cpu(),
        "affect_local": affect["local"][0, :n].cpu(),
        "affect_intensity": affect["intensity_value"][0].cpu(),
        "emotion_logits": affect["emotion_logits"][0].cpu(),
        "intensity_logits": affect["intensity_logits"][0].cpu(),
        "identity_code": identity["code"][0].detach().cpu(),
        "identity_baseline": identity["baseline"][0].detach().cpu(),
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite frozen baseline output")
    return result, {"seed": 42, "clip_seed": seed, "query_motion_inference": False}


def subset_query(query, indices):
    count = len(query["clip_id"])
    return {key: value[indices] if torch.is_tensor(value) else [value[i] for i in indices]
            for key, value in query.items()
            if (torch.is_tensor(value) and value.ndim and len(value) == count)
            or (isinstance(value, (list, tuple)) and len(value) == count)}


def run(args):
    if args.output.exists():
        raise FileExistsError("Fresh export directory required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    # This loader pins checkpoint bytes and completion/recipe sidecars.
    audio = load_frozen_audio(args.audio_checkpoint, args.device)
    checkpoint = load(args.audio_checkpoint)
    system = NeutralAffectSystem(checkpoint["config"])
    system.load_state_dict(checkpoint["system"], strict=True)
    system.to(args.device).eval().requires_grad_(False)
    recipe = read(args.audio_checkpoint.parent.parent / "provenance.json")["recipe"]
    steps = int(recipe["args"]["decode_steps"])
    archive, targets, cache = load(args.audio), load(args.targets), load(args.cache)
    lineage = {"cache": sha(args.cache), "audio": sha(args.audio), "targets": sha(args.targets)}
    assert_source_binding(audio, lineage)
    if (_renderer_cache_binding(archive) != lineage["cache"]
            or _renderer_cache_binding(targets) != lineage["cache"]):
        raise ValueError("Frozen source and audio/anchor cache binding differ")
    # Access only the original fit query role; no validation/test selection.
    query = dict(cache["splits"]["train"]["q"])
    raw_audio = archive["splits"]["train"]
    raw_targets = targets["splits"]["train"]
    if query["clip_id"] != raw_audio["clip_id"] or query["clip_id"] != raw_targets["clip_id"]:
        raise ValueError("Fit query/audio/anchor membership differs")
    for name in ("valid", "times"):
        if not torch.equal(query[name], raw_audio[name]):
            raise ValueError("Fit acoustic clock or mask differs")
    query.update(content=raw_audio["features"][..., :768], audio_features=raw_audio["features"],
                 anchors=raw_targets["anchors"], anchor_valid=raw_targets["anchor_valid"])
    selection = json.loads(args.selection.read_text(encoding="utf-8-sig"))
    picks, indices = select_fit_queries(selection, query)
    speakers = {query["speaker"][i]: int(query["speaker_id"][i]) for i in indices}
    identities, references = load_identities(system, targets, args.enrollment,
        args.native_root.resolve(), query, speakers, args.device)
    store = NativeContextStore(subset_query(query, indices), args.native_root,
        args.native_manifest, args.delta_dir, role="train")
    args.output.mkdir(parents=True)
    for name in ("arrays", "video_npz", "audio"):
        (args.output / name).mkdir()
    records, jobs = [], []
    for index, pick in enumerate(picks):
        cid = pick["clip_id"]
        raw = store.clip(index)
        predicted, inference = infer_acoustic_baseline(system, audio,
            raw["content"], raw["audio_features"], raw["valid"],
            identities[int(raw["speaker_id"])], cid, steps, args.device)
        native_provenance = raw["metadata"]["provenance"]
        waveform = (args.waveform_root / (cid + ".wav") if args.waveform_root
                    else Path(native_provenance["audio_path"]))
        if sha(waveform) != native_provenance["audio_sha256"]:
            raise ValueError("Native waveform hash differs")
        offset = float(native_provenance["audio_offset_s"])
        if not np.isfinite(offset):
            raise ValueError("Native audio offset must be finite")
        copied = Path("audio") / (cid + waveform.suffix)
        shutil.copy2(waveform, args.output / copied)
        payload = {key: value.numpy() for key, value in predicted.items()}
        payload.update(features=raw["audio_features"].numpy(), valid=raw["valid"].numpy(),
            times=raw["times"].numpy(), channel_mask=raw["channel_mask"].numpy(),
            clip_id=np.asarray(cid), audio_relative_path=np.asarray(copied.as_posix()),
            audio_sha256=np.asarray(native_provenance["audio_sha256"]),
            audio_offset_seconds=np.asarray(offset))
        relative = Path("arrays") / (cid + ".npz")
        np.savez_compressed(args.output / relative, **payload)
        modes, curves = ["run12 frozen audio baseline"], [predicted["baseline52"].numpy()]
        if args.include_reference:
            modes.insert(0, "native coefficient reference (not inference input)")
            curves.insert(0, raw["motion"].numpy())
        video = Path("video_npz") / (cid + ".npz")
        np.savez_compressed(args.output / video, channels=np.asarray(ARKIT_NAMES),
            mode_names=np.asarray(modes), motions=np.stack(curves), clip_id=np.asarray(cid),
            noise_seed=np.asarray(42), times=payload["times"], valid=payload["valid"],
            channel_mask=payload["channel_mask"], audio_relative_path=payload["audio_relative_path"],
            audio_sha256=payload["audio_sha256"], audio_offset_seconds=payload["audio_offset_seconds"])
        record = {key: pick[key] for key in ("clip_id", "sentence", "speaker", "speaker_name", "emotion")}
        record.update(arrays=relative.as_posix(), arrays_sha256=sha(args.output / relative),
            video_npz=video.as_posix(), video_npz_sha256=sha(args.output / video),
            native_frames=len(raw["valid"]), native_metadata=raw["metadata"], inference=inference,
            display_report=inspect_input(args.output / video, 25)[-1])
        records.append(record)
        jobs.append({"input": video.as_posix(), "output": cid, "fps": 25, "columns": len(modes),
            "tile_size": 360, "samples": 16, "max_frames": 0,
            "audio": copied.as_posix(), "audio_sha256": native_provenance["audio_sha256"],
            "audio_offset_seconds": offset, "expected_video": cid + "/comparison.mp4"})
        print("FROZEN_BASELINE_EXPORTED", cid, len(raw["valid"]), flush=True)
    write(args.output / "render_jobs.json", {"schema": SCHEMA, "jobs": jobs, "rendered": False,
        "driver": "scripts/render_dynamic_rig_comparison.py", "paths_relative_to": "directory containing this JSON"})
    write(args.output / "provenance.json", {"schema": SCHEMA, "clips": records,
        "selection_sha256": sha(args.selection), "source": lineage,
        "checkpoint_sha256": sha(args.audio_checkpoint), "decode_steps": steps,
        "neutral_reference_bindings": references, "native_source": store.provenance,
        "query_motion_inference": False, "query_motion_loaded_for_native_binding": True,
        "reference_render_exported": args.include_reference, "inference_frozen": True,
        "new_training": False, "test_loaded": False, "dev405_indexed": False,
        "default_replaced": False, "identity_mouth_emotion_certified": False,
        "scope": "Metadata-selected previously exposed historical fit clips, not held-out evaluation",
        "seed_policy": "Same clip-keyed42 noise as clocked fullface exporter; no outcome selection",
        "script_sha256": sha(__file__)})
    write(args.output / "manifest.json", {path.relative_to(args.output).as_posix():
        {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(args.output.rglob("*")) if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "audio-checkpoint", "audio", "targets", "cache", "enrollment",
                 "native-root", "native-manifest", "delta-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path,
        help="Optional relocated directory containing clip_id.wav; original waveform SHA still required")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
