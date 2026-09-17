"""Prepare only the metadata-locked native-val cross-identity development set.

This is new-identity/shared-script development, not novel-sentence evaluation.
No native-test manifests/targets or formal training targets are opened. Frozen
normalization, B0, identity and global paths are reused without fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import native_clip
from kinetalk_b0.utils import freeze_module
from scripts.extract_emotion2vec_pilot import load_extractor, sha
from scripts.prepare_predictable_motion_bundle import build_split as build_motion_split, read_rows, row_with_ids
from scripts.prepare_predictable_renderer_cache import build_split as build_renderer_split, validate_enrollment, validate_stats
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import device_batch, observed


def validate_cross_identity_protocol(rows, references, selection):
    if selection.get("schema") != "formal_predictable_metadata_lock_v1":
        raise ValueError("Require a formal metadata lock")
    if not rows or not references:
        raise ValueError("Cross-identity development queries and references must be nonempty")
    mapping = {str(name): int(sid) for name, sid in selection["new_identity_validation_speaker_to_id"].items()}
    if set(mapping) & (set(selection["train_speaker_to_id"]) | set(selection["sealed_test_speaker_to_id"])):
        raise ValueError("Cross-identity development overlaps training or sealed-test identities")
    if set(mapping.values()) & (set(selection["train_speaker_to_id"].values()) | set(selection["sealed_test_speaker_to_id"].values())):
        raise ValueError("Cross-identity speaker IDs overlap another role")
    enrollment_sentences = set(selection["sentence_roles"]["enrollment"])
    reserved_sentences = set(selection["sentence_roles"]["reserved_prior_test"])
    for role, values in (("new_identity_validation", rows), ("new_identity_enrollment", references)):
        if len({r["clip_id"] for r in values}) != len(values):
            raise ValueError(f"Duplicate {role} clips")
        if {r["speaker"] for r in values} != set(mapping):
            raise ValueError(f"{role} does not contain exactly the locked native-val identities")
        for row in values:
            if row.get("dataset") != "mead" or row.get("split") != "val" or row.get("pilot_split") != role:
                raise ValueError("Only locked native-val development recordings are allowed")
            if int(row["speaker_id"]) != mapping[row["speaker"]] or int(row.get("valid_frames", 0)) < 32:
                raise ValueError("Locked speaker ID or minimum valid-frame contract differs")
            if int(row["emotion_id"]) != int(row["emotion"]) or int(row["emotion_id"]) not in (0, 1, 5, 6):
                raise ValueError("Native frozen-model emotion IDs must remain unchanged")
            if row["sentence"] in reserved_sentences:
                raise ValueError("Reserved prior-test sentence cannot enter development")
            if role == "new_identity_validation" and row["sentence"] in enrollment_sentences:
                raise ValueError("Enrollment sentence cannot be a development query")
            if role == "new_identity_enrollment" and row["sentence"] not in enrollment_sentences:
                raise ValueError("Reference not drawn from locked enrollment sentences")
    grouped = validate_enrollment(references, rows, min_references=int(selection["min_references"]))
    expected = selection["reference_counts"]["val"]
    if {speaker: sum(r["speaker"] == speaker for r in references) for speaker in mapping} != expected:
        raise ValueError("Neutral reference counts differ from metadata lock")
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-dir", "native-root", "checkpoint", "config", "model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--content-storage", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh cross-identity development output directory")
    selection_path = args.manifest_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf8"))
    roles = ("new_identity_validation", "new_identity_enrollment")
    hashes = {role: sha(args.manifest_dir / f"{role}.jsonl") for role in roles}
    if any(selection["manifest_sha256"].get(role) != digest for role, digest in hashes.items()):
        raise ValueError("Cross-identity manifests changed since locking")
    rows, references = [read_rows(args.manifest_dir / f"{role}.jsonl") for role in roles]
    grouped = validate_cross_identity_protocol(rows, references, selection)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if any(cfg.get(key) != checkpoint.get("config", {}).get(key) for key in ("data", "model")):
        raise ValueError("Frozen checkpoint/config mismatch")
    if checkpoint.get("audio_source") != "acoustic":
        raise ValueError("Require official final emotion2vec acoustic-source checkpoint")
    validate_stats(checkpoint.get("audio_stats"), 768, "audio_stats")
    validate_stats(checkpoint.get("feature_stats"), 768, "feature_stats")
    torch.set_num_threads(4)
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    before = state_hash(system.state_dict())
    extractor, geometry, model_info = load_extractor(args.model_dir, args.device)
    if checkpoint["audio_stats"].get("feature_type") != model_info["model_id"]:
        raise ValueError("Saved normalization is from a different feature extractor")
    layers = [2, 4, 6]
    if any(layer >= len(extractor.blocks) for layer in layers):
        raise ValueError("Extractor does not support the fixed formal intermediate layers")
    identities = {}
    with torch.no_grad():
        for sid, refs in sorted(grouped.items()):
            clips = [native_clip(row_with_ids(row), args.native_root) for row in refs]
            query = device_batch(clips, args.device)
            base = system.base(query["content"], query["valid"])
            residual = torch.where(observed(query), query["motion"] - base["b0"], 0)
            identities[sid] = system.encode_identity(residual[None], query["valid"][None])
    # Both audited builders use batch32 for identical frozen-base GPU grouping.
    bundle = build_motion_split(rows, args.native_root, identities, system, extractor, geometry, args.device, layers)
    split = build_renderer_split(rows, args.native_root, bundle, identities, system, extractor, geometry,
        checkpoint, args.device, 32, args.content_storage, "cross_identity_development")
    after = state_hash(system.state_dict())
    if after != before or any(parameter.grad is not None for parameter in system.parameters()):
        raise RuntimeError("Frozen system changed while preparing development")
    paths = [Path(__file__).resolve(), Path(__file__).with_name("prepare_predictable_motion_bundle.py"),
             Path(__file__).with_name("prepare_predictable_renderer_cache.py"),
             Path(__file__).with_name("extract_predictable_audio.py"), Path(__file__).with_name("extract_emotion2vec_pilot.py")]
    provenance = {"schema": "cross_identity_development_v1", "role": "native_val_new_identity_development",
        "selection_sha256": sha(selection_path), "manifest_hashes": hashes,
        "formal_training_manifest_sha256": selection["manifest_sha256"]["train"],
        "original_development_manifest_sha256": selection["manifest_sha256"]["validation"],
        "checkpoint_sha256": sha(args.checkpoint), "config_sha256": sha(args.config),
        "frozen_before": before, "frozen_after": after, "frozen_unchanged": before == after,
        "speaker_to_id": selection["new_identity_validation_speaker_to_id"],
        "reference_counts": selection["reference_counts"]["val"], "min_references": selection["min_references"],
        "clips": len(rows), "sentences": len({r["sentence"] for r in rows}),
        "model": model_info, "geometry": geometry, "layers": layers, "batch_size": 32,
        "content_storage": args.content_storage, "cached_base_required": True,
        "normalization": "Frozen original checkpoint audio_stats followed by feature_stats; no new fit",
        "audio_stats_sha256": state_hash({key: checkpoint["audio_stats"][key] for key in ("mean", "std")}),
        "feature_stats_sha256": state_hash({key: checkpoint["feature_stats"][key] for key in ("mean", "std")}),
        "training_targets_read": False, "test_manifests_loaded": False, "test_targets_read": False,
        "basis_or_scale_fitted": False, "source_sha256": {str(path.resolve()): sha(path) for path in paths},
        "scope": "Native-val cross-identity development with shared scripts. Neutral references are independent from queries. B0/pretrained exposure is unknown; this is not whole-system unseen or novel-sentence testing."}
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "cross_identity_development.pt"
    torch.save({"bundle": bundle, "split": split, "provenance": provenance}, path)
    (args.output / "provenance.json").write_text(json.dumps({**provenance, "output_bytes": path.stat().st_size}, indent=2), encoding="utf8")
    print(json.dumps({"complete": True, "clips": len(rows), "output": str(path), "bytes": path.stat().st_size,
                      "frozen_unchanged": True, "test_targets_read": False}), flush=True)


if __name__ == "__main__":
    main()
