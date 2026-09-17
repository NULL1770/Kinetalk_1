"""Refit schedule-probe inputs using only formal-training identities.

Three identities are held out by a fixed metadata hash. Their query motion,
features and weights do not enter alpha selection, basis, scales or head
initialization. Original formal development/test targets are not accessed.
Frozen cached B0/identity/global tensors are sliced without recomputation.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.predictable_motion import fit_motion_path
from scripts.prepare_predictable_renderer_cache import validate_enrollment, validate_manifest_order
from scripts.probe_predictable_motion import DEFAULT_ALPHAS, nested_select
from scripts.train_predictable_renderer import PredictableAudioHead, audio_features, sha, state_hash
from scripts.train_formal_predictable_projection import canonical_hash


SCHEMA = "teacher_schedule_inner_identity_v1"


def speaker_rank(speaker, seed):
    return hashlib.sha256(f"teacher_schedule_identity:{seed}:{speaker}".encode("utf8")).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def choose_identity_split(rows, references, selection, *, seed=20260921, heldout_count=3):
    if selection.get("schema") != "formal_predictable_metadata_lock_v1":
        raise ValueError("Require formal metadata lock")
    mapping = {str(k): int(v) for k, v in selection["train_speaker_to_id"].items()}
    if heldout_count != 3 or len(mapping) <= heldout_count:
        raise ValueError("Schedule diagnosis requires exactly three held-out formal-training identities")
    if not rows or len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("Formal training clips must be nonempty and unique")
    if {r["speaker"] for r in rows} != set(mapping):
        raise ValueError("Formal training manifest identity coverage differs from lock")
    for row in rows + references:
        if row.get("dataset") != "mead" or row.get("split") != "train":
            raise ValueError("Only native-train MEAD rows can enter schedule diagnosis")
        if row["speaker"] not in mapping or int(row["speaker_id"]) != mapping[row["speaker"]]:
            raise ValueError("Formal speaker mapping changed")
    enrollment = validate_enrollment(references, rows, min_references=int(selection["min_references"]))
    expected_references = selection["reference_counts"]["train"]
    if dict(Counter(r["speaker"] for r in references)) != expected_references:
        raise ValueError("Formal independent neutral enrollment differs from metadata lock")
    eligible = []
    for speaker, sid in mapping.items():
        emotions = {int(r["emotion_id"]) for r in rows if r["speaker"] == speaker}
        if emotions == {0, 1, 5, 6} and len(enrollment[sid]) >= int(selection["min_references"]):
            eligible.append(speaker)
    if len(eligible) < heldout_count:
        raise ValueError("Not enough identities with all four emotions and independent neutral references")
    ordered = sorted(eligible, key=lambda speaker: (speaker_rank(speaker, seed), speaker))
    heldout = set(ordered[:heldout_count])
    fit_indices = torch.tensor([i for i, r in enumerate(rows) if r["speaker"] not in heldout], dtype=torch.long)
    hold_indices = torch.tensor([i for i, r in enumerate(rows) if r["speaker"] in heldout], dtype=torch.long)
    if not len(fit_indices) or not len(hold_indices):
        raise ValueError("Empty inner identity role")
    roles = {}
    for role, indices in (("train", fit_indices), ("validation", hold_indices)):
        chosen_rows = [rows[int(i)] for i in indices]
        speakers = sorted({r["speaker"] for r in chosen_rows})
        roles[role] = {"source_indices": indices.tolist(), "clips": len(chosen_rows),
            "clip_ids": [r["clip_id"] for r in chosen_rows], "sentence_ids": sorted({r["sentence"] for r in chosen_rows}),
            "speaker_to_id": {s: mapping[s] for s in speakers},
            "emotion_counts": dict(Counter(str(r["emotion_id"]) for r in chosen_rows)),
            "references": [{k: row[k] for k in ("clip_id", "sentence", "speaker", "speaker_id", "emotion_id", "artifact_sha256")}
                           for row in references if row["speaker"] in speakers]}
    report = {"schema": SCHEMA, "seed": seed, "selection_uses_metadata_only": True,
        "speaker_selection": "lowest sha256('teacher_schedule_identity:{seed}:{speaker}') among formal-train identities with four emotions and independent neutral references",
        "eligible_speaker_order": [{"speaker": s, "hash": speaker_rank(s, seed)} for s in ordered],
        "roles": roles,
        "checks": {"identities_disjoint": not bool(set(roles["train"]["speaker_to_id"]) & set(roles["validation"]["speaker_to_id"])),
                   "clips_disjoint": not bool(set(roles["train"]["clip_ids"]) & set(roles["validation"]["clip_ids"])),
                   "all_formal_training_clips_partitioned": sorted(fit_indices.tolist() + hold_indices.tolist()) == list(range(len(rows))),
                   "heldout_has_exactly_three_identities": len(heldout) == 3,
                   "neutral_references_query_sentence_disjoint": not bool({r["sentence"] for r in references} & {r["sentence"] for r in rows})},
        "shared_sentence_count": len(set(roles["train"]["sentence_ids"]) & set(roles["validation"]["sentence_ids"])),
        "scope": "Training-internal held-out identities; shared scripts permitted. All source clips were formal training candidates. Frozen B0/global/identity and prior full-data experiments may have seen these identities; this is not an untouched whole-system test."}
    if not all(report["checks"].values()):
        raise AssertionError("Inner identity split isolation failed")
    return fit_indices, hold_indices, report


def take_bundle(source, indices, groups=None):
    count = len(source["clip_id"])
    result = {}
    for key, value in source.items():
        if key == "groups":
            result[key] = copy.deepcopy(groups if groups is not None else value)
        elif key == "features":
            result[key] = {name: tensor.index_select(0, indices).clone() for name, tensor in value.items()}
        elif torch.is_tensor(value):
            if value.shape[0] != count:
                raise ValueError(f"Unexpected bundle tensor batch: {key}")
            result[key] = value.index_select(0, indices).clone()
        elif isinstance(value, (list, tuple)) and len(value) == count:
            result[key] = [copy.deepcopy(value[int(i)]) for i in indices]
        else:
            raise ValueError(f"Unexpected bundle field: {key}")
    return result


def take_cache_split(source, indices):
    result = {}
    for group in ("q", "base", "identity", "affect"):
        result[group] = {}
        for key, value in source[group].items():
            if torch.is_tensor(value):
                result[group][key] = value.index_select(0, indices).clone()
            elif isinstance(value, (list, tuple)):
                result[group][key] = [copy.deepcopy(value[int(i)]) for i in indices]
            else:
                raise ValueError(f"Unexpected cache field: {group}.{key}")
    if "audio_records" in source:
        result["audio_records"] = [copy.deepcopy(source["audio_records"][int(i)]) for i in indices]
    return result


def train_groups(channel_mask):
    common = channel_mask.bool().all(0)
    common[list(NUISANCE_CHANNELS_52)] = False
    return {"all_expression": common.nonzero(as_tuple=True)[0].tolist(),
            "upper_expression": [c for c in (5, 6, 12, 13, 41, 42, 43, 44, 45) if common[c]],
            "mouth": [c for c in range(14, 41) if common[c]], "jaw17": [17] if common[17] else []}


def fit_training_only(source_bundle, fit_indices, *, alphas=DEFAULT_ALPHAS, seed=45):
    """Slice before accessing observations; held-out NaNs cannot affect fits."""
    # The only source tensor indexing is along the explicitly selected fit IDs.
    train = take_bundle(source_bundle, fit_indices)
    groups = train_groups(train["channel_mask"])
    train["groups"] = groups
    features = audio_features(train).double()
    weight = train["weight"]
    channels = groups["all_expression"]
    if len(channels) < 8:
        raise ValueError("Need at least eight training-observed expression channels")
    target = train["motion_bins"][..., channels]
    ids = torch.arange(len(features))
    selection = nested_select(features, target, weight, train["sentence_id"], ids,
        alphas=tuple(alphas), ranks=(8,), folds=3, seed=seed)
    chosen = selection["selected_per_method_rank"]
    states = fit_motion_path(features, target, weight, ids,
        sorted({value["alpha"] for value in chosen.values()}), (8,))
    states = {f"{state['method']}_rank{state['rank']}": {**state, "motion_channel_indices": channels}
              for state in states if chosen[f"{state['method']}_rank{state['rank']}"]["alpha"] == state["alpha"]}
    heads = {}
    for name in ("rrr_rank8", "pca_rank8"):
        head = PredictableAudioHead(states[name], train["motion_bins"], train["weight"]).eval()
        heads[name] = {"state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                       "sha256": state_hash(head.state_dict())}
    return train, states, heads, selection


def assert_cache_bundle_order(cache_train, bundle_train):
    query = cache_train["q"]
    if query["clip_id"] != bundle_train["clip_id"] or query["sentence_id"] != bundle_train["sentence_id"]:
        raise ValueError("Source formal-training cache/bundle order mismatch")
    for key in ("speaker_id", "emotion_id", "channel_mask"):
        if not torch.equal(query[key], bundle_train[key]):
            raise ValueError(f"Source formal-training cache/bundle {key} mismatch")


def tensor_group_hashes(split):
    return {group: state_hash({key: value for key, value in split[group].items() if torch.is_tensor(value)})
            for group in ("q", "base", "identity", "affect")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "cache", "manifest-dir", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921, help="Prespecified metadata identity hash seed")
    parser.add_argument("--cv-seed", type=int, default=45)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh schedule-probe preparation directory")
    torch.set_num_threads(args.threads)
    selection_path = args.manifest_dir / "selection.json"
    formal_selection = json.loads(selection_path.read_text(encoding="utf8"))
    metadata_hashes = {role: sha(args.manifest_dir / f"{role}.jsonl") for role in ("train", "enrollment")}
    if any(formal_selection["manifest_sha256"].get(role) != digest for role, digest in metadata_hashes.items()):
        raise ValueError("Formal training/enrollment manifest changed")
    rows, references = [read_rows(args.manifest_dir / f"{role}.jsonl") for role in ("train", "enrollment")]
    fit_ids, hold_ids, selection = choose_identity_split(rows, references, formal_selection, seed=args.seed)
    hashes = {"bundle": sha(args.bundle), "cache": sha(args.cache), "selection": sha(selection_path), **metadata_hashes}
    split_lock = {"schema": "projection_schedule_internal_split_v1", "source_role": "formal_training_queries_only",
        "outer_development_loaded": False, "new_identity_development_loaded": False,
        "outer_development_target_values_accessed": False, "test_target_values_accessed": False,
        "fit_clip_ids": selection["roles"]["train"]["clip_ids"],
        "validation_clip_ids": selection["roles"]["validation"]["clip_ids"],
        "heldout_speakers": sorted(selection["roles"]["validation"]["speaker_to_id"]),
        "fit_speakers": sorted(selection["roles"]["train"]["speaker_to_id"]),
        "source_sha256": hashes,
        "source_paths": {"bundle": str(args.bundle.resolve()), "cache": str(args.cache.resolve()),
                         "selection": str(selection_path.resolve()),
                         "train": str((args.manifest_dir / "train.jsonl").resolve()),
                         "enrollment": str((args.manifest_dir / "enrollment.jsonl").resolve())},
        "identity_selection": selection,
        "source_read_note": "Files are byte-hashed; serialized containers are mmap-loaded. Only formal-training tensor values are accessed; original formal-validation values are not indexed, fitted or copied."}
    # mmap avoids touching any original formal-validation tensor storage. File
    # byte hashing and serialized container metadata are read for provenance.
    source_bundle = torch.load(args.bundle, map_location="cpu", weights_only=False, mmap=True)
    source_cache = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)
    if source_cache.get("schema") != "predictable_renderer_cache_v1":
        raise ValueError("Wrong source renderer cache")
    if source_cache["provenance"].get("bundle_sha256") != hashes["bundle"]:
        raise ValueError("Source cache is paired with a different formal bundle")
    for data in (source_bundle, source_cache):
        if any(data["provenance"]["manifest_hashes"].get(role) != digest for role, digest in metadata_hashes.items()):
            raise ValueError("Source cache/bundle formal-training manifest provenance differs")
    full_train = source_bundle["bundles"]["internal"]
    full_cache = source_cache["splits"]["train"]
    validate_manifest_order(rows, full_train)
    assert_cache_bundle_order(full_cache, full_train)
    # Fit is complete before held-out query targets/features are copied.
    train, states, heads, inner_selection = fit_training_only(full_train, fit_ids, seed=args.cv_seed)
    heldout = take_bundle(full_train, hold_ids, groups=train["groups"])
    cache_splits = {"train": take_cache_split(full_cache, fit_ids), "validation": take_cache_split(full_cache, hold_ids)}
    for role, data in (("train", train), ("validation", heldout)):
        assert_cache_bundle_order(cache_splits[role], data)
    expected_source = {"checkpoint_sha256", "config_sha256", "frozen_before", "frozen_after", "frozen_unchanged",
                       "audio_stats_sha256", "feature_stats_sha256", "model", "geometry", "content_storage", "batch_size", "cached_base_required"}
    cache_provenance = {key: copy.deepcopy(value) for key, value in source_cache["provenance"].items() if key in expected_source}
    provenance = {"schema": SCHEMA, "input_sha256": hashes, "identity_selection": selection,
        "internal_split_lock_sha256": canonical_hash(split_lock),
        "fit_source_indices": fit_ids.tolist(), "heldout_source_indices": hold_ids.tolist(),
        "source_roles_accessed": ["source_bundle.bundles.internal", "source_cache.splits.train"],
        "original_formal_validation_target_values_accessed": False, "new_identity439_loaded": False,
        "test_manifests_loaded": False, "test_targets_loaded": False,
        "source_read_note": "Complete files byte-hashed and container metadata loaded with mmap; only formal training tensor values indexed. No source formal validation tensor is accessed or copied.",
        "refitted_components": ["feature_std", "motion_basis", "ridge_coefficients", "head_target_scale", "head_initialization"],
        "fit_policy": "Only nineteen training identities; three sentence folds select alpha; rank fixed8; every fold refits all fitted components. Held-out identity targets are copied only after fits finish.",
        "cv_seed": args.cv_seed, "alpha_grid": list(DEFAULT_ALPHAS), "fixed_rank": 8,
        "source_frozen_state_sha256": source_cache["provenance"]["frozen_before"],
        "cached_values_policy": "Slice original source cached base, neutral identity and audio global/local without recomputation or normalization refitting",
        "copied_tensor_group_sha256": {role: tensor_group_hashes(data) for role, data in cache_splits.items()},
        "head_sha256": {name: value["sha256"] for name, value in heads.items()},
        "source_sha256": {str(path.resolve()): sha(path) for path in
            (Path(__file__), Path(__file__).with_name("probe_predictable_motion.py"),
             Path(__file__).with_name("train_predictable_renderer.py"),
             Path(__file__).parents[1] / "kinetalk_b0/predictable_motion.py")},
        "scope": selection["scope"]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "split_lock.json").write_text(json.dumps(split_lock, indent=2), encoding="utf8")
    for filename, key in (("fit_ids.json", "fit_clip_ids"), ("validation_ids.json", "validation_clip_ids")):
        (args.output / filename).write_text(json.dumps({"clip_ids": split_lock[key]}, indent=2), encoding="utf8")
    for role, indices in (("train", fit_ids), ("validation", hold_ids)):
        chosen = [rows[int(i)] for i in indices]
        (args.output / f"{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen), encoding="utf8")
    (args.output / "enrollment.jsonl").write_text("".join(json.dumps(r) + "\n" for r in references), encoding="utf8")
    manifest_hashes = {role: sha(args.output / f"{role}.jsonl") for role in ("train", "validation", "enrollment")}
    provenance["manifest_hashes"] = manifest_hashes
    prepared_bundle = {"bundles": {"internal": train, "external_dev": heldout},
        "train_ids": torch.arange(len(train["clip_id"])), "heldout_ids": torch.empty(0, dtype=torch.long), "provenance": provenance}
    bundle_path = args.output / "diagnostic_bundle.pt"
    torch.save(prepared_bundle, bundle_path)
    bundle_hash = sha(bundle_path)
    prepared_cache = {"schema": "predictable_renderer_cache_v1", "splits": cache_splits,
        "identities": source_cache.get("identities", {}),
        "provenance": {**cache_provenance, **provenance, "bundle_sha256": bundle_hash}}
    torch.save(prepared_cache, args.output / "renderer_cache.pt")
    weights_provenance = {**provenance, "bundle_sha256": bundle_hash}
    torch.save({"states": states, "selection": inner_selection, "provenance": weights_provenance}, args.output / "content_temporal_weights.pt")
    torch.save({"heads": heads, "provenance": weights_provenance}, args.output / "fixed_heads.pt")
    result = {**provenance, "inner_selection": inner_selection,
        "output_sha256": {name: sha(args.output / name) for name in
            ("diagnostic_bundle.pt", "renderer_cache.pt", "content_temporal_weights.pt", "fixed_heads.pt")}}
    (args.output / "selection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"complete": True, "roles": {role: {k: data[k] for k in ("clips", "speaker_to_id")}
         for role, data in selection["roles"].items()}, "selected_alphas": {k: state["alpha"] for k, state in states.items()},
         "output": str(args.output), "original_formal_validation_target_values_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
