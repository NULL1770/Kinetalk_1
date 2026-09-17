"""Build compact frozen renderer inputs from locked train/development manifests.

Official final emotion2vec features are re-extracted on the native clock, then
passed through BOTH saved run09 normalization stages. Centered probe bins
cannot recover utterance emotion and are never used as global affect inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import native_clip
from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.predictable_motion import bin_centered_frames
from kinetalk_b0.utils import freeze_module
from scripts.extract_emotion2vec_pilot import align_features, expected_length, load_extractor, sha
from scripts.prepare_predictable_motion_bundle import read_rows, row_with_ids
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_feature_probe import prepare_feature_batch
from scripts.train_neutral_affect_pilot import device_batch, observed


QUERY_KEYS = ("motion", "content", "valid", "motion_valid", "times", "channel_mask", "speaker_id",
              "emotion_id", "intensity_id", "intensity_valid", "clip_id", "sentence_id", "speaker")
AFFECT_KEYS = ("global", "intensity_value", "local", "controls", "control_mask", "control_weight",
               "emotion_logits", "intensity_logits")


def cpu(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.float() if value.is_floating_point() and value.dtype != torch.float64 else value
    return value


def validate_stats(stats, dimension, name):
    if not isinstance(stats, dict):
        raise ValueError(f"Missing {name}; raw final features require both saved transforms")
    for key in ("mean", "std"):
        if key not in stats:
            raise ValueError(f"Missing {name}.{key}")
        value = torch.as_tensor(stats[key])
        if value.shape != (dimension,) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {name}.{key}")
    if not (torch.as_tensor(stats["std"]) > 0).all():
        raise ValueError(f"Nonpositive {name}.std")


def normalize_official_final(query, raw_final, checkpoint):
    """Raw final -> original cache normalization -> saved probe normalization."""
    if checkpoint.get("audio_source") != "acoustic":
        raise ValueError("This cache builder requires the frozen acoustic-source emotion2vec checkpoint")
    if raw_final.ndim != 3 or raw_final.shape[-1] != 768 or raw_final.shape[:2] != query["valid"].shape:
        raise ValueError("Raw final features must match native [B,T,768]")
    valid = query["valid"]
    if not torch.isfinite(raw_final[valid]).all():
        raise ValueError("Nonfinite observed official final features")
    stats = checkpoint.get("audio_stats")
    validate_stats(stats, raw_final.shape[-1], "audio_stats")
    mean, std = [torch.as_tensor(stats[k], device=raw_final.device, dtype=raw_final.dtype) for k in ("mean", "std")]
    cached_audio = torch.where(valid[..., None], (raw_final - mean) / std, 0)
    return prepare_feature_batch({**query, "audio": cached_audio}, "acoustic", checkpoint["feature_stats"])


def validate_manifest_order(rows, bundle):
    ids = [str(row["clip_id"]) for row in rows]
    if ids != list(map(str, bundle["clip_id"])) or len(set(ids)) != len(ids):
        raise ValueError("Manifest clip order must exactly equal diagnostic bundle order without duplicates")
    if [str(row["sentence"]) for row in rows] != list(map(str, bundle["sentence_id"])):
        raise ValueError("Manifest sentences differ from diagnostic bundle")
    for row_key, bundle_key in (("speaker_id", "speaker_id"), ("emotion_id", "emotion_id")):
        values = torch.tensor([int(row_with_ids(row)[row_key]) for row in rows])
        if not torch.equal(values, bundle[bundle_key].cpu().long()):
            raise ValueError(f"Manifest {row_key} differs from diagnostic bundle")


def validate_enrollment(references, all_query_rows, min_references=4):
    if min_references < 2:
        raise ValueError("Require at least two independent neutral references")
    query_sentences = {str(row["sentence"]) for row in all_query_rows}
    query_ids = {str(row["clip_id"]) for row in all_query_rows}
    by_speaker = defaultdict(list)
    for row in references:
        row = row_with_ids(row)
        if row["emotion_id"] != 0 or row["sentence"] in query_sentences or row["clip_id"] in query_ids:
            raise ValueError("Neutral enrollment must be globally sentence/clip disjoint from queries")
        by_speaker[int(row["speaker_id"])].append(row)
    for sid in {int(row["speaker_id"]) for row in all_query_rows}:
        refs = by_speaker[sid]
        if len(refs) < min_references or len({r["sentence"] for r in refs}) != len(refs) or len({r["clip_id"] for r in refs}) != len(refs):
            raise ValueError(f"Identity {sid} requires >={min_references} distinct neutral enrollment sentences")
        name = {str(row["speaker"]) for row in all_query_rows if int(row["speaker_id"]) == sid}
        if len(name) != 1 or {str(row["speaker"]) for row in refs} != name:
            raise ValueError(f"Speaker name/ID mismatch for {sid}")
    return dict(by_speaker)


def assert_residual_bins(query, base, identity, diagnostic, offset, stride):
    """Same observed residual and native bin targets as the locked probe."""
    residual = torch.where(observed(query), query["motion"] - base["b0"] - identity["baseline"][:, None], 0)
    residual[:, :, list(NUISANCE_CHANNELS_52)] = 0
    values, weight = bin_centered_frames(residual, query["valid"], stride)
    target = diagnostic["motion_bins"][offset:offset + len(values)].float()
    expected_weight = diagnostic["weight"][offset:offset + len(values)].float()
    if not torch.equal(weight.float(), expected_weight):
        raise ValueError("Rebuilt native bin weights differ from diagnostic bundle")
    if not torch.allclose(values.float(), target, atol=2e-6, rtol=2e-5):
        error = float((values.float() - target).abs().max())
        raise ValueError(f"Rebuilt motion residual bins differ from diagnostic bundle: max_abs={error}")


@torch.inference_mode()
def extract_final_native(clip, model, geometry, device):
    """Same official forward/alignment as original extraction, with no FP16 cast."""
    import soundfile as sf
    provenance = clip["metadata"]["provenance"]
    path = Path(provenance["audio_path"])
    if sha(path) != provenance["audio_sha256"]:
        raise ValueError(f"Audited waveform changed: {path}")
    wave, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError("Official final extraction requires finite 16 kHz mono audio")
    waveform = torch.from_numpy(wave).to(device)
    output = model.extract_features(F.layer_norm(waveform, waveform.shape)[None], padding_mask=None,
                                    mask=False, remove_extra_tokens=True)
    final = output["x"][0].float().cpu().numpy()
    count = expected_length(len(wave), geometry)
    if final.shape != (count, 768) or not np.isfinite(final).all():
        raise ValueError("Official final feature shape/clock mismatch")
    times = (geometry["center_offset_samples"] + np.arange(count) * geometry["hop_samples"]) / 16000
    targets = clip["times"].numpy() + float(provenance["audio_offset_s"])
    aligned, _, extended = align_features(final, times, targets, clip["valid"].numpy(), len(wave), geometry)
    return torch.from_numpy(aligned), {"wave_sha256": provenance["audio_sha256"],
        "audio_offset_s": provenance["audio_offset_s"], "boundary_extended_frames": int(extended.sum())}


def append_batch(storage, mapping, keys, content_storage=None):
    for key in keys:
        value = cpu(mapping[key])
        if key == "content" and content_storage == "float16":
            value = value.half()
            if not torch.isfinite(value).all():
                raise ValueError("Content overflows float16 storage; use --content-storage float32")
        storage[key].append(value)


def finish_batches(storage):
    return {key: torch.cat(values) if torch.is_tensor(values[0]) else [item for value in values for item in value]
            for key, values in storage.items()}


@torch.no_grad()
def build_split(rows, native_root, diagnostic, identities, system, extractor, geometry, checkpoint,
                device, batch_size, content_storage, split_name):
    validate_manifest_order(rows, diagnostic)
    storage = {key: defaultdict(list) for key in ("q", "base", "identity", "affect")}
    audio_records = []
    for offset in range(0, len(rows), batch_size):
        part = [native_clip(row_with_ids(row), native_root) for row in rows[offset:offset + batch_size]]
        query = device_batch(part, device)
        base = system.base(query["content"], query["valid"])
        ids = query["speaker_id"].detach().cpu().tolist()
        identity = {key: torch.cat([identities[int(sid)][key] for sid in ids]) for key in ("code", "baseline")}
        assert_residual_bins(query, base, identity, diagnostic, offset, system.motion_teacher.stride)
        raw, records = zip(*(extract_final_native(clip, extractor, geometry, device) for clip in part))
        raw = torch.stack(raw).to(device)
        audio_query = normalize_official_final(query, raw, checkpoint)
        affect = system.encode_audio(audio_query["audio"], query["valid"])
        # Preserve original model conditions and base in float32. Content is an
        # optional storage-only half representation; always pass cached base.
        append_batch(storage["q"], query, QUERY_KEYS, content_storage)
        append_batch(storage["base"], base, ("b0", "h0"))
        append_batch(storage["identity"], identity, ("code", "baseline"))
        append_batch(storage["affect"], affect, AFFECT_KEYS)
        audio_records.extend(records)
        print(json.dumps({"split": split_name, "done": offset + len(part), "total": len(rows)}), flush=True)
    output = {key: finish_batches(values) for key, values in storage.items()}
    if not torch.equal(output["q"]["channel_mask"], diagnostic["channel_mask"]):
        raise ValueError("Rebuilt query observation channels differ from frozen diagnostic bundle")
    extra = (-output["q"]["valid"].shape[1]) % system.motion_teacher.stride
    bin_weight = F.pad(output["q"]["valid"].float(), (0, extra)).reshape(len(rows), -1, system.motion_teacher.stride).sum(-1)
    if not torch.equal(bin_weight, diagnostic["weight"].float()):
        raise ValueError("Rebuilt query native valid-frame bins differ from diagnostic bundle")
    output["audio_records"] = audio_records
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest-dir", "native-root", "checkpoint", "config", "model-dir", "bundle", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Match the motion-bundle batch shape to reproduce frozen GPU base numerics")
    parser.add_argument("--content-storage", choices=("float16", "float32"), default="float16")
    parser.add_argument("--min-references", type=int, default=4,
                        help="Predeclared enrollment minimum; formal cross-identity protocol permits two")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh cache directory; previous caches are never replaced")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    torch.set_num_threads(4)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if any(cfg.get(key) != checkpoint.get("config", {}).get(key) for key in ("data", "model")):
        raise ValueError("Frozen model config/checkpoint mismatch")
    validate_stats(checkpoint.get("audio_stats"), 768, "audio_stats")
    validate_stats(checkpoint.get("feature_stats"), 768, "feature_stats")
    if checkpoint.get("audio_source") != "acoustic":
        raise ValueError("Require run09 frozen emotion2vec acoustic checkpoint")
    source = torch.load(args.bundle, map_location="cpu", weights_only=False, mmap=True)
    manifest_rows = {name: read_rows(args.manifest_dir / f"{name}.jsonl") for name in ("train", "validation", "enrollment")}
    for name, bundle_key in (("train", "internal"), ("validation", "external_dev")):
        if not manifest_rows[name]:
            raise ValueError(f"Empty {name} manifest")
        validate_manifest_order(manifest_rows[name], source["bundles"][bundle_key])
    if {r["sentence"] for r in manifest_rows["train"]} & {r["sentence"] for r in manifest_rows["validation"]}:
        raise ValueError("Training and validation sentence overlap")
    refs = validate_enrollment(manifest_rows["enrollment"], manifest_rows["train"] + manifest_rows["validation"], args.min_references)
    manifest_hashes = {name: sha(args.manifest_dir / f"{name}.jsonl") for name in manifest_rows}
    if source.get("provenance", {}).get("manifest_hashes") != manifest_hashes:
        raise ValueError("Locked manifests changed since diagnostic bundle creation")
    if source.get("provenance", {}).get("checkpoint_sha256") != sha(args.checkpoint):
        raise ValueError("Diagnostic bundle was built with different frozen weights")
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    before = state_hash(system.state_dict())
    extractor, geometry, model_info = load_extractor(args.model_dir, args.device)
    if checkpoint["audio_stats"].get("feature_type") != model_info["model_id"]:
        raise ValueError("Saved audio normalization is not from this official extractor")
    identities = {}
    with torch.no_grad():
        for sid, rows in sorted(refs.items()):
            clips = [native_clip(row_with_ids(row), args.native_root) for row in rows]
            query = device_batch(clips, args.device)
            base = system.base(query["content"], query["valid"])
            residual = torch.where(observed(query), query["motion"] - base["b0"], 0)
            encoded = system.encode_identity(residual[None], query["valid"][None])
            identities[sid] = {key: encoded[key] for key in ("code", "baseline")}
    splits = {}
    for name, bundle_key in (("train", "internal"), ("validation", "external_dev")):
        splits[name] = build_split(manifest_rows[name], args.native_root, source["bundles"][bundle_key], identities,
            system, extractor, geometry, checkpoint, args.device, args.batch_size, args.content_storage, name)
    after = state_hash(system.state_dict())
    if after != before or any(parameter.grad is not None for parameter in system.parameters()):
        raise RuntimeError("Frozen model weights changed or accumulated gradients")
    root = Path(__file__).resolve().parents[1]
    provenance = {"schema": "predictable_renderer_cache_v1", "checkpoint_sha256": sha(args.checkpoint),
        "config_sha256": sha(args.config), "bundle_sha256": sha(args.bundle), "manifest_hashes": manifest_hashes,
        "frozen_before": before, "frozen_after": after, "frozen_unchanged": before == after,
        "model": model_info, "geometry": geometry, "new_test_loaded": False,
        "normalization": "Official raw aligned float32 final -> checkpoint.audio_stats -> checkpoint.feature_stats, no newly fitted statistics",
        "audio_stats_sha256": state_hash({k: checkpoint["audio_stats"][k] for k in ("mean", "std")}),
        "feature_stats_sha256": state_hash({k: checkpoint["feature_stats"][k] for k in ("mean", "std")}),
        "content_storage": args.content_storage, "cached_base_required": True,
        "batch_size": args.batch_size,
        "min_references": args.min_references,
        "batch_note": "Frozen base may differ numerically with GPU batch shape; residual agreement is asserted without relaxing tolerance.",
        "content_note": "Storage-only half precision when requested; frozen b0/h0 always computed from original float32 content. Pass cached base for all comparisons.",
        "global_labels_used": False, "metadata_labels_role": "emotion/intensity labels stored for evaluation only",
        "affect_local": "Frozen original audio local condition, preserved for pretrained comparison",
        "source_sha256": {str(path.relative_to(root)): sha(path) for path in
            (Path(__file__).resolve(), root / "scripts/extract_emotion2vec_pilot.py",
             root / "scripts/train_neutral_affect_feature_probe.py", root / "kinetalk_b0/neutral_data.py")},
        "scope": "Locked training and development only. No test reads, model updates, new audio normalization, or centered-bin reconstruction of global affect."}
    output = {"schema": provenance["schema"], "splits": splits,
              "identities": {sid: {key: cpu(value) for key, value in identity.items()} for sid, identity in identities.items()},
              "provenance": provenance}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "renderer_cache.pt"
    torch.save(output, path)
    provenance["output_bytes"] = path.stat().st_size
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"complete": True, "train": len(splits["train"]["q"]["clip_id"]),
                      "validation": len(splits["validation"]["q"]["clip_id"]), "output_bytes": path.stat().st_size,
                      "output": str(path), "frozen_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
