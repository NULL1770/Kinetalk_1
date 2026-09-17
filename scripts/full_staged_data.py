"""Validated raw inputs for staged retraining; never reuse encoded model caches.

The historical cache fixes query membership and native timestamps. Its old B0,
identity and affect tensors are deliberately discarded because those modules
will change during this experiment. Neutral references remain independent.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch
import yaml

from kinetalk_b0.label_guided_intensity import fit_intensity_scales, regional_intensity
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import native_clip, stack_clips
from kinetalk_b0.utils import freeze_module
from scripts.prepare_expression_intensity_targets import _query_metadata
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import state_hash
from scripts.train_projection_schedule_ablation import read_allowlist

SCHEMA = "full_staged_raw_inputs_v1"
INPUT_NAMES = ("cache", "bundle", "weights", "checkpoint", "config", "fit_ids", "validation_ids", "split_lock")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _require_equal(actual, expected, message):
    if not torch.equal(actual, expected):
        raise ValueError(message)


def _renderer_cache_binding(payload):
    """Accept the two recorded v1 provenance layouts, rejecting conflicts."""
    provenance = payload.get("provenance", {})
    candidates = [provenance.get("renderer_cache_sha256"),
                  provenance.get("source_sha256", {}).get("cache")]
    candidates = [value for value in candidates if value is not None]
    if not candidates or any(value != candidates[0] for value in candidates):
        raise ValueError("Missing or conflicting renderer cache provenance binding")
    return candidates[0]


def _validate_split_lock(cache, lock, fit_ids, dev_ids):
    if (lock.get("schema") != "projection_schedule_internal_split_v1"
            or lock.get("source_role") != "formal_training_queries_only"
            or lock.get("outer_development_loaded") is not False
            or lock.get("new_identity_development_loaded") is not False):
        raise ValueError("Require explicit internal formal-training-only split lock")
    for role, ids, key in (("train", fit_ids, "fit_clip_ids"),
                           ("validation", dev_ids, "validation_clip_ids")):
        if list(cache["splits"][role]["q"]["clip_id"]) != ids or lock.get(key) != ids:
            raise ValueError("Cache, allowlist and split lock clip order differ")
    if set(fit_ids) & set(dev_ids):
        raise ValueError("Fit/development clips overlap")
    fit = cache["splits"]["train"]["q"]
    dev = cache["splits"]["validation"]["q"]
    fit_sids, dev_sids = set(map(int, fit["speaker_id"])), set(map(int, dev["speaker_id"]))
    if fit_sids & dev_sids or set(fit["speaker"]) & set(dev["speaker"]):
        raise ValueError("Fit/development identities overlap")
    if sorted(set(dev["speaker"])) != sorted(lock.get("heldout_speakers", [])):
        raise ValueError("Development identities differ from split lock")
    if cache["provenance"].get("internal_split_lock_sha256") != canonical_hash(lock):
        raise ValueError("Cache split-lock hash differs")
    return sorted(fit_sids), sorted(dev_sids)


def _references(enrollment_path, native_root, cache, targets, *, window):
    mapping, query_clips, query_sentences = _query_metadata(cache)
    root = Path(native_root).resolve()
    rows = [json.loads(line) for line in Path(enrollment_path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty enrollment manifest")
    groups, seen, sentences = defaultdict(list), set(), defaultdict(set)
    # Check scope and path containment before opening any reference artifact.
    for row in rows:
        speaker, sid = row.get("speaker"), int(row.get("speaker_id", -1))
        if (row.get("split") != "train" or row.get("emotion_id", row.get("emotion")) != 0
                or row.get("emotion", 0) not in (0, "neutral")):
            raise ValueError("Enrollment must be native-TRAIN neutral references")
        if speaker not in mapping or mapping[speaker] != sid:
            raise ValueError("Enrollment identity is outside query mapping")
        cid, sentence = row.get("clip_id"), row.get("sentence")
        if not isinstance(cid, str) or not cid or not isinstance(sentence, str) or not sentence:
            raise ValueError("Reference clip and sentence metadata required")
        if cid in query_clips or sentence in query_sentences:
            raise ValueError("Reference/query clips and sentences must be independent")
        if cid in seen or sentence in sentences[sid]:
            raise ValueError("Repeated reference clip or per-person sentence")
        path = (root / row["artifact"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Reference artifact leaves native root")
        if not isinstance(row.get("artifact_sha256"), str) or sha(path) != row["artifact_sha256"]:
            raise ValueError("Reference artifact SHA256 differs")
        seen.add(cid); sentences[sid].add(sentence); groups[sid].append(row)
    if set(groups) != set(mapping.values()) or any(len(part) < 2 for part in groups.values()):
        raise ValueError("Every query identity requires at least two independent neutral references")
    refs, reference_groups, bindings = {}, {}, []
    for sid, part in sorted(groups.items()):
        speaker = part[0]["speaker"]
        saved = targets.get("by_speaker", {}).get(speaker, {})
        if int(saved.get("speaker_id", -1)) != sid or list(saved.get("reference_clip_ids", [])) != [r["clip_id"] for r in part]:
            raise ValueError("Target anchors use different neutral references")
        clips = []
        for row in part:
            clip = native_clip({**row, "emotion_id": 0, "intensity_id": int(row.get("intensity_id", row.get("intensity", 0)))},
                               root, window=window, min_valid_frames=min(32, window))
            clips.append(clip)
            bindings.append({"clip_id": row["clip_id"], "speaker_id": sid, "artifact": str((root / row["artifact"]).resolve()),
                             "artifact_sha256": row["artifact_sha256"]})
        refs[sid] = stack_clips(clips)
        middle = len(part) // 2
        reference_groups[sid] = {"a": torch.arange(middle), "b": torch.arange(middle, len(part))}
    return refs, reference_groups, bindings


def load_training_inputs(source_run, audio_path, targets_path, enrollment_path, native_root, *, device="cpu"):
    """Return CPU data and an explicitly frozen warm-start system.

    ``splits[role]`` is the query dictionary with raw ``audio_features`` (1540D),
    FP32 ``content`` (768D), anchors/masks and diagnostic intensity supervision.
    ``refs[speaker_id]`` is a native batched reference dictionary; ``ref_groups``
    supplies disjoint A/B row indices, including the two-reference case.
    No old ``base``, ``identity`` or ``affect`` group is returned. Callers must
    explicitly unfreeze the appropriate stage and only train fit references.
    """
    source_run, audio_path, targets_path, enrollment_path = map(Path, (source_run, audio_path, targets_path, enrollment_path))
    provenance_path = source_run / "provenance.json"
    source_provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    recipe = source_provenance["recipe"]
    if source_provenance.get("recipe_sha256") != canonical_hash(recipe):
        raise ValueError("Source recipe hash differs")
    if recipe.get("schema") != "projection_scaled_centered_probe_v1" or recipe["args"].get("arm") != "uniform":
        raise ValueError("Require audited uniform source initialization")
    paths = {name: Path(recipe["args"][name]) for name in INPUT_NAMES}
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes != recipe.get("input_sha256"):
        raise ValueError("Source input SHA256 differs")
    adapter_path = source_run / "final_epoch008.pt"
    adapter = _load(adapter_path)
    if (adapter.get("schema") != recipe["schema"] or adapter.get("recipe") != recipe
            or adapter.get("recipe_sha256") != canonical_hash(recipe)
            or adapter.get("completed_epochs") != recipe["args"].get("epochs")):
        raise ValueError("Source adapter is incomplete or bound to a different recipe")
    config = yaml.safe_load(paths["config"].read_text(encoding="utf8"))
    checkpoint = _load(paths["checkpoint"])
    if any(checkpoint.get("config", checkpoint.get("provenance", {}).get("config", {})).get(k) != config[k]
           for k in ("data", "model")):
        raise ValueError("Original checkpoint model/data config differs")
    system = NeutralAffectSystem(config)
    system.load_state_dict(checkpoint["model"], strict=True)
    frozen = state_hash({k: v for k, v in system.state_dict().items() if not k.startswith("local_projection.")})
    if frozen != adapter.get("frozen_state_sha256"):
        raise ValueError("Original frozen model does not match source adapter")
    # The uniform adapter is an RRR-control interface, not the learned local
    # coordinate system of the original motion teacher. Keep the complete
    # original checkpoint, including its own matching local projection.
    initial_system_hash = state_hash(system.state_dict())
    freeze_module(system); system.eval(); system.to(device)
    del checkpoint, adapter
    cache = _load(paths["cache"])
    _query_metadata(cache)
    for name in ("checkpoint", "config", "bundle"):
        if cache["provenance"].get(name + "_sha256") != hashes[name]:
            raise ValueError("Historical cache provenance differs: " + name)
    lock = json.loads(paths["split_lock"].read_text(encoding="utf8"))
    fit_ids, dev_ids = read_allowlist(paths["fit_ids"]), read_allowlist(paths["validation_ids"])
    fit_sids, dev_sids = _validate_split_lock(cache, lock, fit_ids, dev_ids)
    audio, targets = _load(audio_path), _load(targets_path)
    for obj, schema in ((audio, "label_guided_audio_cache_v1"), (targets, "expression_intensity_targets_v1")):
        if (obj.get("schema") != schema or set(obj.get("splits", {})) != {"train", "validation"}
                or _renderer_cache_binding(obj) != hashes["cache"]):
            raise ValueError("Derived audio/target cache binding differs")
    enrollment_hash = sha(enrollment_path)
    if targets["provenance"].get("source_sha256", {}).get("enrollment") != enrollment_hash:
        raise ValueError("Target enrollment SHA256 differs")
    stats = audio["feature_stats"]
    if stats.get("fit_clip_ids") != fit_ids or stats.get("source") != "train_valid_native_frames_only":
        raise ValueError("Audio statistics fit IDs/scope differ")
    for key in ("mean", "std"):
        if stats[key].shape != (1540,) or not torch.isfinite(stats[key]).all():
            raise ValueError("Invalid audio feature statistics")
    if (stats["std"] <= 0).any():
        raise ValueError("Audio feature standard deviation must be positive")
    scales = targets["scales"]
    if scales.shape != (52,) or not torch.isfinite(scales).all() or (scales < .02).any():
        raise ValueError("Invalid target scales")
    splits = {}
    for role in ("train", "validation"):
        query, a, target = cache["splits"][role]["q"], audio["splits"][role], targets["splits"][role]
        ids, valid, times = list(query["clip_id"]), query["valid"], query["times"]
        if a["clip_id"] != ids or target["clip_id"] != ids:
            raise ValueError("Derived cache clip order differs")
        if a.get("sentence_id") != list(query["sentence_id"]):
            raise ValueError("Audio/query sentence order differs")
        _require_equal(a["valid"], valid, "Audio validity differs")
        _require_equal(a["times"], times, "Audio frame clock differs")
        if (times.shape != valid.shape or not torch.isfinite(times).all()
                or not ((times[:, 1:] - times[:, :-1]) > 0).all()):
            raise ValueError("Invalid query clock")
        features = a["features"]
        if features.shape != (*valid.shape, 1540) or features.dtype != torch.float32 or not torch.isfinite(features[valid]).all():
            raise ValueError("Expected finite FP32 native1540 frame audio")
        content = features[..., :768]
        _require_equal(content.to(query["content"].dtype)[valid], query["content"][valid], "Native FP32/cached content storage differs")
        if target["anchors"].shape != (len(ids), 52) or target["anchor_valid"].shape != (len(ids), 52) or target["anchor_valid"].dtype != torch.bool:
            raise ValueError("Invalid target anchors/masks")
        if list(target.get("speaker", [])) != list(query["speaker"]):
            raise ValueError("Target/query speaker order differs")
        _require_equal(target["speaker_id"], query["speaker_id"], "Target/query speaker IDs differ")
        for i, speaker in enumerate(query["speaker"]):
            person = targets.get("by_speaker", {}).get(speaker, {})
            if int(person.get("speaker_id", -1)) != int(query["speaker_id"][i]):
                raise ValueError("Target anchor speaker mapping differs")
            _require_equal(target["anchor_valid"][i], person["anchor_valid"], "Target anchor availability differs")
            keep = target["anchor_valid"][i]
            _require_equal(target["anchors"][i][keep], person["anchor"][keep], "Target anchor differs from bound speaker reference")
        observed = valid[..., None] & query["channel_mask"][:, None] & target["anchor_valid"][:, None]
        intensity, intensity_valid = regional_intensity(query["motion"], observed, target["anchors"], scales)
        _require_equal(intensity_valid, target["valid"], "Intensity supervision validity differs")
        if not torch.allclose(intensity, target["intensity"], rtol=1e-6, atol=1e-7):
            raise ValueError("Intensity supervision differs from bound raw targets")
        splits[role] = {**query, "content": content, "audio_features": features,
                        "anchors": target["anchors"], "anchor_valid": target["anchor_valid"],
                        "target_intensity": target["intensity"], "target_intensity_valid": target["valid"]}
    train = splits["train"]
    observed = train["valid"][..., None] & train["channel_mask"][:, None] & train["anchor_valid"][:, None]
    expected_scales = fit_intensity_scales(train["motion"], observed, train["anchors"], floor=float(targets["provenance"].get("scale_floor", .02)))
    if not torch.allclose(scales, expected_scales, rtol=1e-6, atol=1e-7):
        raise ValueError("Target scales are not fitted solely on current training queries")
    windows = {q["valid"].shape[1] for q in splits.values()}
    if len(windows) != 1:
        raise ValueError("Staged input splits must use one native frame window")
    refs, ref_groups, bindings = _references(enrollment_path, native_root, cache, targets, window=windows.pop())
    provenance = {"schema": SCHEMA, "source_recipe_sha256": canonical_hash(recipe),
                  "source_paths": {k: str(v.resolve()) for k, v in paths.items()},
                  "input_sha256": {**hashes, "source_provenance": sha(provenance_path), "source_adapter": sha(adapter_path),
                                   "audio": sha(audio_path), "targets": sha(targets_path), "enrollment": enrollment_hash},
                  "reference_bindings": bindings, "fit_sids": fit_sids, "dev_sids": dev_sids,
                  "fit_clips": len(fit_ids), "development_clips": len(dev_ids),
                  "encoded_cache_groups_used": [], "raw_content_source": "Verified audio.features[...,0:768] native FP32",
                  "warmstart": "Complete original checkpoint model including its learned-control local projection; uniform RRR adapter is verified for data lineage only and is not loaded",
                  "initial_system_sha256": initial_system_hash,
                  "uniform_projection_loaded": False,
                  "global_audio_note": "1540D features contain middle emotion2vec layers, not the historical final-layer global input",
                  "reference_policy": "Fit identity references may train; development references are enrollment-only and must never optimize parameters",
                  "scope": "Internal held-out identities, shared scripts allowed; inherited model may have seen these identities previously",
                  "test_loaded": False, "outer_development_loaded": False, "default_replaced": False}
    return {"system": system, "config": config, "splits": splits, "refs": refs, "ref_groups": ref_groups,
            "fit_sids": fit_sids, "dev_sids": dev_sids, "feature_stats": stats, "target_scales": scales,
            "provenance": provenance}
