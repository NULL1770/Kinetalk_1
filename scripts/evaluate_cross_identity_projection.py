"""Evaluate one already-selected formal adapter on native-val new identities.

The saved fixed audio head is restored verbatim, including training scales and
basis. No fitting, rank/gate search or checkpoint selection happens here. This
is shared-script cross-identity development; native test remains sealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.prepare_predictable_renderer_cache import assert_residual_bins
from scripts.train_formal_predictable_projection import (
    SCHEMA, canonical_hash, evaluate_compact, frozen_hash, save_json, write_curve_provenance,
)
from scripts.train_predictable_renderer import PredictableAudioHead, sha, state_hash


class RestoredFixedAudioHead(PredictableAudioHead):
    """Use saved buffers exactly; deliberately never call the fitting constructor."""
    def __init__(self, saved):
        nn.Module.__init__(self)
        required = {"basis", "channels", "feature_std", "target_scale", "linear.weight"}
        if set(saved) != required:
            raise ValueError("Saved fixed-head state has missing or unexpected entries")
        if any(not torch.is_tensor(value) or not torch.isfinite(value).all() for value in saved.values()):
            raise ValueError("Saved head tensors must be finite")
        basis, channels, std, scale, weight = [saved[key] for key in
            ("basis", "channels", "feature_std", "target_scale", "linear.weight")]
        if basis.ndim != 2 or basis.shape[1] != 8 or channels.ndim != 1 or len(channels) != basis.shape[0]:
            raise ValueError("Formal head requires the saved shared rank-eight basis")
        if channels.dtype != torch.long or len(channels.unique()) != len(channels) or channels.min() < 0 or channels.max() >= 52:
            raise ValueError("Invalid saved motion channel indices")
        if std.ndim != 1 or scale.shape != (8,) or weight.shape != (8, len(std)) or not (std > 0).all() or not (scale > 0).all():
            raise ValueError("Invalid saved head scales or coefficient shape")
        for key in ("basis", "channels", "feature_std", "target_scale"):
            self.register_buffer(key, saved[key].detach().cpu().clone())
        self.linear = nn.Linear(len(std), 8, bias=False)
        self.load_state_dict(saved, strict=True)
        freeze_module(self)
        self.eval()


def validate_selected_run(provenance, summary, selected, selected_hash):
    recipe = provenance["recipe"]
    digest = canonical_hash(recipe)
    if provenance.get("recipe_sha256") != digest or selected.get("schema") != SCHEMA:
        raise ValueError("Formal recipe or selected checkpoint schema mismatch")
    if selected.get("recipe") != recipe or selected.get("recipe_sha256") != digest or summary.get("recipe_sha256") != digest:
        raise ValueError("Selected checkpoint is not from the supplied formal recipe")
    if summary.get("has_eligible_checkpoint") is not True or not selected.get("latest_selection", {}).get("eligible"):
        raise ValueError("Cross-identity candidate evaluation requires the preselected eligible best checkpoint")
    if summary.get("selected_checkpoint_sha256") != selected_hash:
        raise ValueError("Selected adapter differs from the completed formal run selection")
    if selected.get("completed_epochs") != summary.get("best_epoch") or selected.get("best_epoch") != summary.get("best_epoch"):
        raise ValueError("Selected adapter does not match the formal best epoch")
    if selected.get("head_sha256") != provenance.get("head_sha256") or state_hash(selected["head"]) != selected["head_sha256"]:
        raise ValueError("Saved frozen audio head hash mismatch")
    if selected.get("frozen_state_sha256") != provenance.get("frozen_state_sha256"):
        raise ValueError("Frozen backbone provenance mismatch")
    if recipe.get("audio_activity_gate") != "1-softmax(frozen_audio_logits)[neutral]" or recipe.get("trainable") != ["local_projection.weight"]:
        raise ValueError("Unsupported formal gate or trainable parameter contract")
    if recipe.get("noise_seeds") != [42, 123, 2026] or recipe.get("new_test_loaded") is not False:
        raise ValueError("Formal evaluation requires the fixed three noises and sealed test")
    return recipe


def validate_cross_payload(payload, formal_bundle, recipe, checkpoint_hash, config_hash):
    provenance = payload.get("provenance", {})
    if provenance.get("schema") != "cross_identity_development_v1" or provenance.get("role") != "native_val_new_identity_development":
        raise ValueError("Require prepared native-val cross-identity development data")
    for key in ("test_manifests_loaded", "test_targets_read", "basis_or_scale_fitted", "training_targets_read"):
        if provenance.get(key) is not False:
            raise ValueError(f"Invalid preparation isolation flag: {key}")
    if provenance.get("checkpoint_sha256") != checkpoint_hash or provenance.get("config_sha256") != config_hash:
        raise ValueError("Cross-identity cache uses different frozen checkpoint/config")
    if recipe["input_sha256"]["checkpoint"] != checkpoint_hash or recipe["input_sha256"]["config"] != config_hash:
        raise ValueError("Formal run frozen checkpoint/config differs")
    formal_provenance = formal_bundle["provenance"]
    if provenance["formal_training_manifest_sha256"] != formal_provenance["manifest_hashes"]["train"]:
        raise ValueError("Cross-identity cache and formal run use different metadata locks")
    query, bundle = payload["split"]["q"], payload["bundle"]
    train = formal_bundle["bundles"]["internal"]
    if query["clip_id"] != bundle["clip_id"] or query["sentence_id"] != bundle["sentence_id"]:
        raise ValueError("Cross-identity renderer/bundle order mismatch")
    if len(set(query["clip_id"])) != len(query["clip_id"]) or set(query["clip_id"]) & set(train["clip_id"]):
        raise ValueError("Duplicate or training-overlapping development recordings")
    if set(query["speaker_id"].tolist()) & set(train["speaker_id"].tolist()):
        raise ValueError("New-identity development IDs overlap formal training")
    if not torch.equal(query["channel_mask"], bundle["channel_mask"]) or bundle["groups"] != train["groups"]:
        raise ValueError("Cross-identity observation layout differs from formal training")
    return provenance


def verify_recipe_sources(recipe):
    root = Path(__file__).resolve().parents[1]
    checked = {}
    for source_path, digest in recipe["source_sha256"].items():
        normalized = source_path.replace("\\", "/")
        relative = None
        for directory in ("kinetalk_b0/", "scripts/"):
            if directory in normalized:
                relative = normalized[normalized.index(directory):]
                break
        if relative is None:
            raise ValueError(f"Cannot locate recorded formal source: {source_path}")
        path = root / relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Current implementation differs from formal recipe: {relative}")
        checked[relative] = digest
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cross-data", "run-dir", "checkpoint", "config", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, help="Relocated formal best checkpoint; verified against final selection hash")
    parser.add_argument("--formal-bundle", type=Path, help="Relocated bundle used for identity/provenance verification only; no target fitting")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    provenance_path, summary_path = args.run_dir / "provenance.json", args.run_dir / "summary.json"
    formal_provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    formal_summary = json.loads(summary_path.read_text(encoding="utf8"))
    selected_path = args.selected_checkpoint or args.run_dir / "best.pt"
    selected_hash = sha(selected_path)
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    recipe = validate_selected_run(formal_provenance, formal_summary, selected, selected_hash)
    checked_sources = verify_recipe_sources(recipe)
    bundle_path = args.formal_bundle or Path(recipe["args"]["bundle"])
    if sha(bundle_path) != recipe["input_sha256"]["bundle"]:
        raise ValueError("Formal training bundle hash differs from saved recipe")
    formal_bundle = torch.load(bundle_path, map_location="cpu", weights_only=False, mmap=True)
    payload = torch.load(args.cross_data, map_location="cpu", weights_only=False, mmap=True)
    ck_hash, cfg_hash = sha(args.checkpoint), sha(args.config)
    cross_provenance = validate_cross_payload(payload, formal_bundle, recipe, ck_hash, cfg_hash)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if any(checkpoint.get("config", {}).get(key) != cfg.get(key) for key in ("data", "model")):
        raise ValueError("Frozen model config mismatch")
    if cfg["data"]["emotion_classes"][0] != "neutral":
        raise ValueError("Saved shared amplitude gate requires neutral class zero")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = bool(recipe["tf32_matmul"])
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    if frozen_hash(system) != selected["frozen_state_sha256"] or state_hash(system.state_dict()) != cross_provenance["frozen_before"]:
        raise ValueError("Frozen model state differs from prepared data or formal training")
    head = RestoredFixedAudioHead(selected["head"]).to(args.device).eval()
    if state_hash(head.state_dict()) != selected["head_sha256"]:
        raise ValueError("Restored fixed head differs from training; no scale refitting is permitted")
    bundle, split = payload["bundle"], payload["split"]
    if head.linear.in_features != sum(bundle["features"][key].shape[-1] for key in ("content", "middle", "prosody")):
        raise ValueError("Development audio features differ from the fixed saved head")
    assert_residual_bins(split["q"], split["base"], split["identity"], bundle, 0, int(cfg["model"].get("affect_stride", 4)))
    del checkpoint, formal_bundle
    args.output.mkdir(parents=True, exist_ok=False)
    base_hash, head_hash = frozen_hash(system), state_hash(head.state_dict())
    seeds = tuple(recipe["noise_seeds"])
    evaluation = {"device": args.device, "seeds": seeds,
                  "batch_size": int(recipe["args"]["eval_batch_size"]),
                  "decode_steps": int(recipe["args"]["decode_steps"])}
    report_provenance = {"schema": "cross_identity_projection_evaluation_v1", "recipe_sha256": canonical_hash(recipe),
        "selected_checkpoint_sha256": selected_hash, "selected_epoch": selected["completed_epochs"],
        "cross_data_sha256": sha(args.cross_data), "formal_provenance_sha256": sha(provenance_path),
        "formal_summary_sha256": sha(summary_path), "checkpoint_sha256": ck_hash, "config_sha256": cfg_hash,
        "formal_bundle_sha256": recipe["input_sha256"]["bundle"], "frozen_state_sha256": base_hash,
        "head_sha256": head_hash, "source_sha256": checked_sources, "evaluation_source_sha256": sha(__file__),
        "noise_seeds": list(seeds), "decode_steps": evaluation["decode_steps"],
        "audio_activity_gate": recipe["audio_activity_gate"], "new_test_loaded": False,
        "training_targets_read": False, "rank_or_scale_or_gate_refitted": False,
        "cross_preparation_provenance": cross_provenance,
        "scope": "Post-selection native-val cross-identity development on shared scripts. This is not novel-sentence testing. B0/pretrained exposure remains unknown; original and adapted outputs use identical cached base/global/identity and noise."}
    save_json(args.output / "provenance.json", report_provenance)
    original_curves = args.output / "original_curves.pt"
    original = evaluate_compact(system, None, split, bundle, modes=("full",), original=True,
                               curves_path=original_curves, **evaluation)
    system.local_projection.load_state_dict(selected["local_projection"], strict=True)
    freeze_module(system)
    projection_hash = state_hash(system.local_projection.state_dict())
    selected_curves = args.output / "selected_curves.pt"
    adapted = evaluate_compact(system, head, split, bundle, modes=("full", "zero", "reverse", "oracle"),
                              curves_path=selected_curves, **evaluation)
    if frozen_hash(system) != base_hash or state_hash(head.state_dict()) != head_hash or state_hash(system.local_projection.state_dict()) != projection_hash:
        raise RuntimeError("Model/head state changed during read-only evaluation")
    curve_sidecars = {role: write_curve_provenance(path, recipe_sha256=canonical_hash(recipe),
        selected_checkpoint_sha256=selected_hash if role == "selected" else ck_hash,
        cache_sha256=report_provenance["cross_data_sha256"]) for role, path in
        (("original", original_curves), ("selected", selected_curves))}
    save_json(args.output / "summary.json", {"schema": report_provenance["schema"],
        "before": original, "after": adapted, "provenance": report_provenance,
        "local_projection_sha256": projection_hash, "curves": curve_sidecars,
        "frozen_unchanged": True, "head_unchanged": True, "default_replaced": False})
    print(json.dumps({"complete": True, "clips": len(split["q"]["clip_id"]),
                      "identities": len(set(split["q"]["speaker_id"].tolist())),
                      "output": str(args.output), "test_loaded": False}), flush=True)


if __name__ == "__main__":
    main()
