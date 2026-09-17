"""Post-hoc fixed-noise diagnosis of an already trained neutral-affect pilot.

Teacher and mixed conditions use observed target motion: they are oracle
diagnostics, not audio-only generation results. No fitting takes place here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset, stack_clips
from scripts.neutral_affect_metrics import motion_metrics, plot_motion_curves


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def _same_value(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(_same_value(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same_value(a, b) for a, b in zip(left, right)))
    return left == right


def _require_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError(f"{label} hash differs from its declared provenance")


def _cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _verify_locked_selection(selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf8"))
    if (selection.get("schema") != "neutral_affect_locked_sentence_audit_v1"
            or selection.get("status") not in ("complete_locked_audit", "locked_incomplete_cell_audit")):
        raise ValueError("Audit selection is not a completed locked audit")
    checks = selection.get("isolation_checks", {})
    required_checks = ("selected_sentence_disjoint_from_all_development_train_enrollment",
                       "selected_clip_disjoint_from_all_development_train_enrollment", "copied_train_hash_unchanged",
                       "source_audio_statistics_reused")
    if any(checks.get(key) is not True for key in required_checks):
        raise ValueError("Locked audit lacks verified isolation checks")
    if selection["status"] == "locked_incomplete_cell_audit":
        if selection.get("allow_incomplete_cells") is not True or not selection.get("missing_cells"):
            raise ValueError("Incomplete audit needs explicit authorization and missing-cell disclosure")
        minima = selection.get("incomplete_minimum_checks", {})
        if any(minima.get(key) is not True for key in ("at_least_24_clips", "all_four_enrolled_speakers", "all_four_emotions")):
            raise ValueError("Incomplete audit minimum coverage was not met")
    folder = selection_path.parent
    sources, outputs = selection.get("source_sha256", {}), selection.get("output_sha256", {})
    if sources.get("train_cache") != outputs.get("train_cache"):
        raise ValueError("Locked audit training cache does not preserve original bytes")
    _require_hash(folder / "train.pt", outputs.get("train_cache"), "Locked raw train")
    _require_hash(folder / "heldout.pt", outputs.get("audit_cache"), "Locked raw audit")
    _require_hash(folder / "selected_manifest.jsonl", outputs.get("selected_manifest"), "Locked selection manifest")
    _require_hash(folder / "available.jsonl", outputs.get("available_manifest"), "Locked availability manifest")
    raw_train, raw_audit = _cache(folder / "train.pt"), _cache(folder / "heldout.pt")
    if raw_audit.get("audit_role") != "locked_sentence_audit" or raw_audit.get("selection_source_sha256") != sources:
        raise ValueError("Audit cache is not bound to the locked selection source chain")
    if raw_audit.get("audit_coverage") != selection.get("coverage") or raw_audit.get("missing_cells") != selection.get("missing_cells"):
        raise ValueError("Audit cache coverage differs from locked selection")
    if raw_audit.get("eval_scope") != selection.get("eval_scope"):
        raise ValueError("Audit scope differs from locked selection")
    queries = raw_audit["queries"]
    if [query["clip_id"] for query in queries] != selection.get("selected_clip_ids"):
        raise ValueError("Audit query ordering differs from locked clip selection")
    if len(queries) != len({query["clip_id"] for query in queries}) or len(queries) != selection.get("query_count"):
        raise ValueError("Audit queries are duplicated or have an unexpected count")
    sentences = {query["sentence_id"] for query in queries}
    if sorted(sentences) != selection.get("selected_sentence_ids"):
        raise ValueError("Audit sentences differ from locked selection")
    if sentences & set(selection.get("excluded_sentence_ids", [])):
        raise ValueError("Audit reused a development/train/enrollment sentence")
    if {query["clip_id"] for query in queries} & set(selection.get("excluded_clip_ids", [])):
        raise ValueError("Audit reused a development/train/enrollment clip")
    original_observations = raw_train["queries"] + [ref for refs in raw_train["identity_references"].values() for ref in refs]
    if sentences & {query["sentence_id"] for query in original_observations}:
        raise ValueError("Audit overlaps actual training or enrollment sentences")
    if not _same_value(raw_train["identity_references"], raw_audit["identity_references"]):
        raise ValueError("Audit changed neutral enrollment references")
    if not _same_value(raw_train["audio_stats"], raw_audit["audio_stats"]):
        raise ValueError("Audit changed source audio statistics")
    return selection, raw_train, raw_audit


def _feature_recipe(provenance):
    if provenance.get("schema") != "emotion2vec_native_pilot_v1":
        raise ValueError("Only the audited emotion2vec extraction chain is supported")
    recipe = {key: value for key, value in provenance.items() if key not in
              ("source_data", "source_cache_sha256", "normalization", "reuse_feature_data")}
    if recipe.get("query_audio_dim") != 768 or recipe.get("reference_audio_dim") != 83:
        raise ValueError("Unexpected emotion2vec query/reference dimensions")
    for key in ("script_sha256", "geometry", "alignment", "boundary_policy", "audio_stats_source", "model"):
        if key not in recipe:
            raise ValueError(f"Extraction recipe missing {key}")
    recipe["model"] = {key: value for key, value in recipe["model"].items() if key != "model_dir"}
    return recipe


def _verify_query_conversion(converted, raw):
    if len(converted["queries"]) != len(raw["queries"]):
        raise ValueError("Feature conversion changed query count")
    for after, before in zip(converted["queries"], raw["queries"]):
        for key, value in before.items():
            if key in ("audio", "metadata"):
                continue
            if key not in after or not _same_value(after[key], value):
                raise ValueError(f"Feature conversion changed non-audio field {key}")
        metadata = {key: value for key, value in after.get("metadata", {}).items() if key != "audio_feature"}
        if not _same_value(metadata, before.get("metadata", {})) or "audio_feature" not in after.get("metadata", {}):
            raise ValueError("Feature conversion changed original metadata or lacks an extraction record")
        audio, valid = after["audio"], before["valid"]
        if audio.shape != (len(valid), 768) or not torch.isfinite(audio).all() or audio[~valid].count_nonzero():
            raise ValueError("Invalid emotion2vec frame features or padded values")
        covered, extended = after.get("audio_center_covered"), after.get("audio_boundary_extended")
        if (not torch.is_tensor(covered) or not torch.is_tensor(extended) or covered.shape != valid.shape
                or extended.shape != valid.shape or not torch.equal(covered | extended, valid) or (covered & extended).any()):
            raise ValueError("Feature clock coverage masks do not preserve native validity")
    if not _same_value(converted["identity_references"], raw["identity_references"]):
        raise ValueError("Feature conversion changed identity references")
    if not _same_value(converted.get("original_audio_stats"), raw["audio_stats"]):
        raise ValueError("Feature conversion lost original audio statistics")


def validate_evaluation_data(checkpoint, data, audit_selection=None, feature_source_data=None):
    """Bind new audit data to a completed metadata lock and its training origin."""
    provenance = checkpoint.get("provenance", {})
    if audit_selection is None:
        if feature_source_data is not None:
            raise ValueError("--feature-source-data is only valid with --audit-selection")
        for split in ("train", "heldout"):
            expected = provenance.get(split + "_sha256")
            if expected is not None:
                _require_hash(data / (split + ".pt"), expected, split)
        return None
    selection, raw_train, raw_audit = _verify_locked_selection(audit_selection)
    sources, outputs = selection["source_sha256"], selection["output_sha256"]
    info = {"selection": str(audit_selection.resolve()), "selection_sha256": sha(audit_selection),
            "coverage": selection["coverage"], "missing_cells": selection["missing_cells"],
            "eval_scope": selection["eval_scope"], "source_sha256": sources, "raw_output_sha256": outputs}
    if feature_source_data is None:
        if (provenance.get("train_sha256") != sources["train_cache"]
                or provenance.get("heldout_sha256") != sources["development_cache"]):
            raise ValueError("Audit raw sources differ from checkpoint caches; extracted features require --feature-source-data")
        _require_hash(data / "train.pt", sources["train_cache"], "Audit train")
        _require_hash(data / "heldout.pt", outputs["audit_cache"], "Audit heldout")
        info["feature_conversion"] = None
        return info
    if "feature_stats" not in checkpoint or checkpoint.get("audio_source") != "acoustic":
        raise ValueError("Feature audit chain requires an acoustic feature-probe checkpoint")
    original = {}
    for split in ("train", "heldout"):
        path = feature_source_data / (split + ".pt")
        _require_hash(path, provenance.get(split + "_sha256"), "Original extracted " + split)
        original[split] = _cache(path)
    # Conversion embeds split hashes, so re-extraction changes train.pt bytes.
    # Require the original extracted train cache to be copied unchanged.
    _require_hash(data / "train.pt", provenance.get("train_sha256"), "Audit extracted train")
    converted = _cache(data / "heldout.pt")
    previous_feature = original["train"].get("audio_feature_provenance", {})
    if previous_feature != original["heldout"].get("audio_feature_provenance"):
        raise ValueError("Original feature caches have inconsistent extraction provenance")
    if previous_feature.get("source_cache_sha256") != {"train": sources["train_cache"], "heldout": sources["development_cache"]}:
        raise ValueError("Checkpoint feature source chain differs from locked audit origins")
    current_feature = converted.get("audio_feature_provenance", {})
    if current_feature.get("source_cache_sha256") != {"train": outputs["train_cache"], "heldout": outputs["audit_cache"]}:
        raise ValueError("New feature extraction is not bound to locked raw audit caches")
    if _feature_recipe(previous_feature) != _feature_recipe(current_feature):
        raise ValueError("Audit feature extractor weights, implementation, clock or normalization recipe changed")
    conversion_selection = json.loads((data / "selection.json").read_text(encoding="utf8"))
    _require_hash(data / "heldout.pt", conversion_selection.get("splits", {}).get("heldout", {}).get("cache_sha256"), "Converted audit")
    if conversion_selection.get("audio_feature_provenance") != current_feature:
        raise ValueError("Converted selection and cache extraction provenance differ")
    for key, value in selection.items():
        if not _same_value(conversion_selection.get(key), value):
            raise ValueError(f"Feature conversion changed locked selection field {key}")
    disk_feature = json.loads((data / "emotion2vec_provenance.json").read_text(encoding="utf8"))
    if disk_feature != current_feature:
        raise ValueError("Feature provenance file differs from the audit cache")
    for key in ("audio_stats",):
        if not _same_value(original["train"].get(key), original["heldout"].get(key)) or not _same_value(converted.get(key), original["train"].get(key)):
            raise ValueError("Audit extraction changed training-fitted feature normalization")
    if not _same_value(checkpoint.get("audio_stats"), original["train"].get("audio_stats")):
        raise ValueError("Checkpoint normalization differs from original feature cache")
    _verify_query_conversion(original["train"], raw_train)
    _verify_query_conversion(converted, raw_audit)
    for key in ("audit_role", "selection_source_sha256", "audit_coverage", "missing_cells", "eval_scope"):
        if not _same_value(converted.get(key), raw_audit.get(key)):
            raise ValueError(f"Feature conversion lost locked audit field {key}")
    info["feature_conversion"] = {"original_data": str(feature_source_data.resolve()),
                                  "original_train_sha256": provenance["train_sha256"],
                                  "original_development_sha256": provenance["heldout_sha256"],
                                  "audit_sha256": sha(data / "heldout.pt"), "extraction_provenance": current_feature,
                                  "feature_manifest_sha256": sha(data / "feature_manifest.jsonl")}
    return info


def batch(clips, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in stack_clips(clips).items()}


def observed(query):
    return query["valid"].unsqueeze(-1) & query["channel_mask"].unsqueeze(1)


def pick(identity, ids):
    return {key: value[ids] for key, value in identity.items()}


def interventions(teacher, audio, mask):
    """Mean preserves actual valid-frame local DC; reverse preserves its set."""
    result = {}
    for source, affect in (("teacher", teacher), ("audio", audio)):
        local = affect["local"]
        clean = torch.where(mask.unsqueeze(-1), local, 0)
        mean = clean.sum(1, keepdim=True) / mask.sum(1).clamp_min(1)[:, None, None]
        reversed_local = clean.clone()
        for index in range(len(local)):
            reversed_local[index, mask[index]] = clean[index, mask[index]].flip(0)
        variants = {"full": local, "zero": torch.zeros_like(local),
                    "mean": mean.expand_as(local) * mask.unsqueeze(-1), "reverse": reversed_local}
        for intervention, value in variants.items():
            result[source + "_" + intervention] = {**affect, "local": value}
    result["teacher_global_audio_local"] = {**teacher, "local": audio["local"]}
    result["audio_global_teacher_local"] = {**audio, "local": teacher["local"]}
    result["audio_global_teacher_intensity_audio_local"] = {**audio, "intensity_value": teacher["intensity_value"]}
    return result


def per_clip_mse(prediction, target, mask):
    error = torch.where(mask, (prediction - target).square(), 0)
    count = mask.sum((1, 2))
    values = error.sum((1, 2)) / count.clamp_min(1)
    return [float(value) if int(number) else None for value, number in zip(values.cpu(), count.cpu())]


def affect_diagnostics(teacher, audio, query):
    result = {}
    valid = teacher["control_mask"]
    for source, affect in (("teacher", teacher), ("audio", audio)):
        intensity_valid = query["intensity_valid"] & (query["intensity_id"] >= 0)
        result[source + "_emotion_accuracy"] = float((affect["emotion_logits"].argmax(-1) == query["emotion_id"]).float().mean())
        result[source + "_intensity_accuracy"] = float((affect["intensity_logits"].argmax(-1)[intensity_valid]
                                                        == query["intensity_id"][intensity_valid]).float().mean()) if intensity_valid.any() else None
        result[source + "_control_std"] = float(affect["controls"][valid].std())
        result[source + "_local_mean_abs"] = float((affect["local"].sum(1)
                                                    / query["valid"].sum(1)[:, None]).abs().mean())
    result["audio_teacher_control_mse"] = float((audio["controls"] - teacher["controls"])[valid].square().mean())
    result["zero_teacher_control_mse"] = float(teacher["controls"][valid].square().mean())
    result["audio_teacher_global_mse"] = float((audio["global"] - teacher["global"]).square().mean())
    result["audio_teacher_intensity_value_mae"] = float((audio["intensity_value"] - teacher["intensity_value"]).abs().mean())
    return result


@torch.no_grad()
def evaluate_seed(system, query, base, identity, teacher, audio, *, seed, steps, save_curves=False):
    rng = torch.Generator(device=query["motion"].device).manual_seed(seed)
    noise = torch.randn(query["motion"].shape, device=query["motion"].device, generator=rng)
    mask = observed(query)
    result = {"seed": seed, "conditions": {}}
    curves = {}
    full = {}
    for name, affect in interventions(teacher, audio, query["valid"]).items():
        generated = system.generate(query["content"], query["valid"], identity, affect,
                                    initial_noise=noise, steps=steps, base=base)["motion"]
        metrics = motion_metrics(generated, query["motion"], base["b0"], query["valid"],
                                 query["channel_mask"], query["times"])
        if name in ("teacher_full", "audio_full"):
            full[name] = generated
        compared = "teacher_full" if name.startswith("teacher_") and name != "teacher_global_audio_local" else "audio_full"
        if compared in full:
            metrics["change_from_" + compared + "_mse"] = float((generated - full[compared])[mask].square().mean())
        metrics["per_clip_mse"] = per_clip_mse(generated, query["motion"], mask)
        for region, ids in (("upper", list(range(14))), ("brows", list(range(41, 46))),
                            ("mouth", list(range(14, 41))), ("jaw17", [17])):
            metrics[region + "_per_clip_mse"] = per_clip_mse(generated[..., ids], query["motion"][..., ids], mask[..., ids])
        result["conditions"][name] = metrics
        if save_curves:
            curves[name] = generated.cpu()
    if save_curves:
        curves.update({key: query[key].cpu() if torch.is_tensor(query[key]) else query[key]
                       for key in ("valid", "times", "channel_mask", "clip_id", "speaker_id", "emotion_id")})
        curves.update({"target": query["motion"].cpu(), "b0": base["b0"].cpu(), "noise": noise.cpu(),
                       "teacher_controls": teacher["controls"].cpu(), "audio_controls": audio["controls"].cpu(),
                       "teacher_local": teacher["local"].cpu(), "audio_local": audio["local"].cpu()})
    return result, curves


def aggregate_seeds(results):
    """Average metrics over decoding seeds, never count seeds as new clips."""
    output = {}
    for condition in results[0]["conditions"]:
        records = [item["conditions"][condition] for item in results]
        averaged, spread = {}, {}
        for key, first in records[0].items():
            if isinstance(first, list):
                averaged[key] = [None if any(record[key][i] is None for record in records)
                                 else float(np.mean([record[key][i] for record in records])) for i in range(len(first))]
            else:
                values = [record[key] for record in records if record[key] is not None]
                averaged[key] = float(np.mean(values)) if values else None
                spread[key] = float(np.std(values)) if values else None
        output[condition] = {"mean": averaged, "seed_std": spread, "seed_count": len(records)}
    return output


def paired_bootstrap(conditions, speaker_ids, *, repetitions=4000, seed=913):
    """Resample seed-averaged paired clip errors, with a cluster sensitivity CI."""
    rng = np.random.default_rng(seed)
    result = {}
    comparisons = [(source + "_full", source + "_" + kind) for source in ("teacher", "audio")
                   for kind in ("mean", "zero", "reverse")]
    comparisons += [(name, "audio_full") for name in
                    ("teacher_global_audio_local", "audio_global_teacher_local", "audio_global_teacher_intensity_audio_local")]
    speakers = np.asarray(speaker_ids)
    for preferred, alternative in comparisons:
        comparison = {}
        for region in ("", "upper_", "brows_", "mouth_", "jaw17_"):
            key = region + "per_clip_mse"
            a, b = conditions[preferred]["mean"][key], conditions[alternative]["mean"][key]
            valid = np.array([x is not None and y is not None for x, y in zip(a, b)])
            if not valid.any():
                comparison[region + "mse"] = None
                continue
            delta = np.asarray([y - x for x, y in zip(a, b) if x is not None and y is not None])
            clip_draws = delta[rng.integers(0, len(delta), size=(repetitions, len(delta)))].mean(1)
            ids = speakers[valid]
            unique = np.unique(ids)
            grouped = [delta[ids == speaker] for speaker in unique]
            sampled_groups = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
            cluster_sums = np.asarray([group.sum() for group in grouped])
            cluster_counts = np.asarray([len(group) for group in grouped])
            clustered = cluster_sums[sampled_groups].sum(1) / cluster_counts[sampled_groups].sum(1)
            comparison[region + "mse"] = {
                "mean_improvement": float(delta.mean()), "median_improvement": float(np.median(delta)),
                "fraction_clips_improved": float((delta > 0).mean()), "clips": len(delta), "speakers": len(unique),
                "clip_bootstrap_ci95": np.quantile(clip_draws, [.025, .975]).tolist(),
                "speaker_cluster_bootstrap_ci95": np.quantile(clustered, [.025, .975]).tolist()}
        result[preferred + "_vs_" + alternative] = comparison
    return {"definition": "Positive = alternative MSE minus preferred MSE; seeds averaged within each clip before resampling.",
            "replicates": repetitions, "seed": seed,
            "limitations": "Exploratory descriptive intervals only. Clips share sentences, emotions and enrolled speakers; clip bootstrap ignores these dependencies. Speaker-cluster sensitivity has only four enrolled speakers and cannot establish population generalization or significance. Multiple comparisons are uncorrected.",
            "comparisons": result}


def training_identity_report(checkpoint):
    path = checkpoint.parent / "identity.json"
    if not path.exists():
        return {"training_view_fit": None, "note": "Original training-view identity report unavailable; no independent evaluation inferred."}
    original = json.loads(path.read_text(encoding="utf8"))
    fit = dict(original)
    if "independent_neutral_retrieval" in fit:
        fit["retrieval_on_training_reference_views"] = fit.pop("independent_neutral_retrieval")
    return {"training_view_fit": fit, "source": str(path), "sha256": sha(path),
            "note": "These enrollment views trained the identity encoder; this is fit, not independent identity validation."}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--splits", nargs="+", choices=("train", "heldout"), default=["train", "heldout"])
    parser.add_argument("--audit-selection", type=Path, help="Original raw locked audit selection.json")
    parser.add_argument("--feature-source-data", type=Path, help="Checkpoint-bound original emotion2vec data directory")
    args = parser.parse_args()
    if args.steps < 1 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Positive decode steps and unique seeds required")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Use a fresh evaluation output directory")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    device = torch.device(cfg.get("device", "cuda"))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    system = NeutralAffectSystem(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_provenance = checkpoint.get("provenance", {})
    if training_provenance.get("config") is not None and training_provenance["config"] != cfg:
        raise ValueError("Evaluation configuration differs from checkpoint provenance")
    audit_info = validate_evaluation_data(checkpoint, args.data, args.audit_selection, args.feature_source_data)
    system.load_state_dict(checkpoint["model"], strict=True)
    system.eval()
    train, heldout = [NeutralAffectDataset(args.data / (split + ".pt")) for split in ("train", "heldout")]
    if {clip["sentence_id"] for clip in train.queries} & {clip["sentence_id"] for clip in heldout.queries}:
        raise ValueError("Train and heldout sentence overlap")
    refs = train.identity_references
    speakers = sorted(refs)
    if speakers != list(range(len(speakers))) or len({len(refs[s]) for s in speakers}) != 1:
        raise ValueError("Expected contiguous enrollment speaker IDs and equal reference counts")
    for speaker, references in heldout.identity_references.items():
        if [clip["clip_id"] for clip in references] != [clip["clip_id"] for clip in refs[speaker]]:
            raise ValueError("Heldout enrollment differs from training enrollment")
    enrollment = batch([clip for speaker in speakers for clip in refs[speaker]], device)
    ref_base = system.base(enrollment["content"], enrollment["valid"])
    residual = torch.where(observed(enrollment), enrollment["motion"] - ref_base["b0"], 0)
    identities = system.encode_identity(residual.reshape(len(speakers), len(refs[speakers[0]]), *residual.shape[1:]),
                                        enrollment["valid"].reshape(len(speakers), len(refs[speakers[0]]), -1))
    tracked = [Path(__file__), Path("scripts/neutral_affect_metrics.py"), Path("kinetalk_b0/models/neutral_affect.py"),
               Path("kinetalk_b0/models/dit.py"), Path("kinetalk_b0/models/encoders.py"), Path("kinetalk_b0/neutral_data.py")]
    root = Path(__file__).resolve().parents[1]
    provenance = {"schema": "neutral_affect_pilot_posthoc_v1", "config": cfg, "checkpoint": str(args.checkpoint),
                  "checkpoint_sha256": sha(args.checkpoint), "config_sha256": sha(args.config),
                  "cache_sha256": {split: sha(args.data / (split + ".pt")) for split in ("train", "heldout")},
                  "source_sha256": {str(path): sha(path if path.is_absolute() else root / path) for path in tracked},
                  "torch": torch.__version__, "device": str(device), "steps": args.steps, "seeds": args.seeds, "splits": args.splits,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "checkpoint_training_provenance": checkpoint.get("provenance"), "locked_audit": audit_info,
                  "scope": "No fitting or checkpoint selection. Heldout sentences of four enrolled people; pretrained B0 may have seen them. Teacher and mixed conditions use target motion and are diagnostic oracles. Audio full/zero/mean/reverse use audio plus enrolled neutral identity only.",
                  "corrections": ["Teacher input re-masked after identity subtraction", "Mean uses actual masked framewise local mean, not zero", "Brows 41:46 reported separately from eyes 0:14", "Identity enrollment retrieval labelled training-view fit"]}
    write_json(args.output / "provenance.json", provenance)
    report = {"provenance": provenance, "identity": training_identity_report(args.checkpoint), "splits": {}}
    for split, dataset in (("train", train), ("heldout", heldout)):
        if split not in args.splits:
            continue
        query = batch(dataset.queries, device)
        if "feature_stats" in checkpoint:
            from scripts.train_neutral_affect_feature_probe import prepare_feature_batch
            query = prepare_feature_batch(query, checkpoint["audio_source"], checkpoint["feature_stats"])
        base = system.base(query["content"], query["valid"])
        identity = pick(identities, query["speaker_id"])
        centered = torch.where(observed(query), query["motion"] - base["b0"] - identity["baseline"][:, None], 0)
        teacher, audio = system.encode_motion(centered, query["valid"]), system.encode_audio(query["audio"], query["valid"])
        entry = {"clips": query["clip_id"], "speaker_ids": query["speaker_id"].cpu().tolist(),
                 "emotion_ids": query["emotion_id"].cpu().tolist(), "sentences": query["sentence_id"],
                 "affect_diagnostics": affect_diagnostics(teacher, audio, query),
                 "b0": motion_metrics(base["b0"], query["motion"], base["b0"], query["valid"], query["channel_mask"], query["times"]),
                 "seeds": []}
        for seed in args.seeds:
            seed_result, curves = evaluate_seed(system, query, base, identity, teacher, audio, seed=seed, steps=args.steps,
                                                save_curves=(split == "heldout" and seed == 42))
            entry["seeds"].append(seed_result)
            write_json(args.output / f"{split}_seed{seed}.json", seed_result)
            print(json.dumps({"split": split, "seed": seed,
                              "audio_mse": seed_result["conditions"]["audio_full"]["masked_mse"],
                              "teacher_mse": seed_result["conditions"]["teacher_full"]["masked_mse"]}), flush=True)
            if curves:
                torch.save(curves, args.output / "heldout_seed42_curves.pt")
                nonneutral = (query["emotion_id"] != 0).nonzero(as_tuple=True)[0]
                indices = list(dict.fromkeys([0] + ([int(nonneutral[0])] if len(nonneutral) else [])))
                for source in ("teacher", "audio"):
                    for index in indices:
                        plot_motion_curves(curves[source + "_full"], curves["target"], curves["b0"], curves["valid"], curves["times"],
                                           args.output / f"heldout_seed42_{source}_index{index}.png", sample_index=index,
                                           title=f"Heldout {source} | fixed index {index}: {query['clip_id'][index]}", channel_mask=curves["channel_mask"])
        entry["aggregate"] = aggregate_seeds(entry["seeds"])
        entry["paired_improvement"] = paired_bootstrap(entry["aggregate"], entry["speaker_ids"])
        report["splits"][split] = entry
        write_json(args.output / (split + "_summary.json"), entry)
    write_json(args.output / "report.json", report)
    print(str((args.output / "report.json").resolve()), flush=True)


if __name__ == "__main__":
    main()
