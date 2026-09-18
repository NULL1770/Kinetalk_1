"""Metadata-only fit expansion and verified frozen-baseline export merge."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_sparse_brow_teacher import checked, sha

SELECTION_SCHEMA = "continuous_motion_fit_selection_v1"
BASELINE_SCHEMA = "semantic_pilot_frozen_baseline_v1"
META_KEYS = ("clip_id", "sentence", "speaker", "emotion")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _safe_id(cid):
    if not isinstance(cid, str) or not cid or cid in (".", "..") or any(c in cid for c in "/\\:"):
        raise ValueError("Safe nonempty clip ID required")
    return cid


def _members(rows):
    result = {}
    if not isinstance(rows, list):
        raise ValueError("Clip metadata must be a list")
    for row in rows:
        cid = _safe_id(row.get("clip_id"))
        if cid in result or any(k not in row for k in META_KEYS):
            raise ValueError("Duplicate clip or incomplete membership metadata")
        result[cid] = row
    return result


def _new_paths(*paths):
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Distinct output paths required")
    for p in resolved:
        if p.exists():
            raise FileExistsError(p)


def select(prior_protocol, old_selection, output_selection, missing_selection):
    """Keep old diagnostic members and expand training only outside held sentences."""
    prior_protocol, old_selection = Path(prior_protocol), Path(old_selection)
    output_selection, missing_selection = Path(output_selection), Path(missing_selection)
    _new_paths(output_selection, missing_selection)
    protocol, old = _read(prior_protocol), _read(old_selection)
    fit = _members(protocol["split"]["fit"])
    old_members = _members(old["clips"])
    if not fit or not old_members:
        raise ValueError("Nonempty fit and old selection required")
    declared_source = old.get("source_protocol_sha256")
    if declared_source is not None and declared_source != sha(prior_protocol):
        raise ValueError("Old selection source protocol hash differs")
    held = set(old["holdout_sentences"])
    if not held or not held <= {row["sentence"] for row in fit.values()}:
        raise ValueError("Declared old holdout sentences must exist in fit")
    for cid, row in old_members.items():
        if cid not in fit or any(row[k] != fit[cid][k] for k in META_KEYS):
            raise ValueError("Old selection differs from historical fit metadata: " + cid)
        if row.get("split") not in ("train", "holdout"):
            raise ValueError("Only old train/holdout membership accepted")
        if (row["sentence"] in held) != (row["split"] == "holdout"):
            raise ValueError("Old split and held sentence membership disagree")
    represented_held = {r["sentence"] for r in old_members.values() if r["split"] == "holdout"}
    if not represented_held <= held:
        raise ValueError("Old holdout rows include undeclared held sentences")
    added = []
    for cid in sorted(fit):
        row = fit[cid]
        if cid in old_members or row["sentence"] in held:
            continue
        parts = cid.split("_")
        if len(parts) < 2 or parts[0] != "mead" or not parts[1]:
            raise ValueError("Cannot derive MEAD speaker metadata: " + cid)
        added.append({**row, "speaker_name": "mead_" + parts[1], "split": "train"})
    combined = copy.deepcopy(old["clips"]) + added
    counts = {role: sum(row["split"] == role for row in combined) for role in ("train", "holdout")}
    omitted = sorted(cid for cid, row in fit.items() if row["sentence"] in held and cid not in old_members)
    shared = {"schema": SELECTION_SCHEMA, "source_protocol_sha256": sha(prior_protocol),
        "old_selection_sha256": sha(old_selection), "source_pool": "historical prior protocol split.fit only",
        "source_fit_count": len(fit), "holdout_sentences": sorted(held),
        "unrepresented_held_sentences": sorted(held - represented_held),
        "historical_upstream_exposure": True, "sealed_test_loaded": False, "dev405_loaded": False,
        "raw_video_loaded": False, "motion_outcomes_loaded": False, "code_sha256": sha(__file__),
        "rule": "Preserve old rows and held sentences; add every remaining fit member outside held sentences, sorted by clip ID",
        "omitted_held_sentence_clip_ids": omitted, "old_clip_count": len(old_members), "added_clip_count": len(added)}
    result = {**shared, "clips": combined, "counts": counts}
    missing = {**shared, "clips": added, "counts": {"train": len(added), "holdout": 0},
               "rule": "Only metadata-selected new train members absent from old selection"}
    for path in (output_selection, missing_selection):
        path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_selection, result)
    missing["combined_selection_sha256"] = sha(output_selection)
    _write(missing_selection, missing)
    return result, missing


def _safe_relative(name):
    rel = PurePosixPath(name)
    if (not isinstance(name, str) or "\\" in name or ":" in name or rel.is_absolute()
            or ".." in rel.parts or not rel.parts or rel.as_posix() != name):
        raise ValueError("Unsafe manifest path")
    return rel


def _verified_export(root):
    manifest = _read(root / "manifest.json")
    for name, record in manifest.items():
        rel = _safe_relative(name)
        path = root.joinpath(*rel.parts)
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("Manifest artifact leaves frozen source")
        checked(path, record)
    if "provenance.json" not in manifest:
        raise ValueError("Hash-bound provenance required")
    provenance = _read(root / "provenance.json")
    if (provenance.get("schema") != BASELINE_SCHEMA
            or provenance.get("query_motion_inference") is not False
            or provenance.get("inference_frozen") is not True
            or provenance.get("reference_render_exported") is not True
            or provenance.get("test_loaded") is not False
            or provenance.get("dev405_indexed") is not False):
        raise ValueError("Expected frozen acoustic-only historical-fit export with references")
    members = _members(provenance["clips"])
    for cid, row in members.items():
        for key, folder in (("arrays", "arrays"), ("video_npz", "video_npz")):
            name = folder + "/" + cid + ".npz"
            if (row.get(key) != name or name not in manifest
                    or row.get(key + "_sha256") != manifest[name]["sha256"]):
                raise ValueError("Clip record and manifest binding differ")
    return manifest, provenance, members


def _reference_union(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() & b.keys():
            if a[key] != b[key]:
                raise ValueError("Neutral reference binding differs: " + str(key))
        return {**a, **b}
    if not isinstance(a, list) or not isinstance(b, list):
        raise ValueError("Unsupported neutral reference bindings")
    groups = []
    for refs in (a, b):
        grouped, seen = {}, set()
        for record in refs:
            key = (record["speaker_id"], record["clip_id"])
            if key in seen:
                raise ValueError("Duplicate neutral reference binding")
            seen.add(key)
            grouped.setdefault(record["speaker_id"], []).append(record)
        groups.append({speaker: sorted(records, key=lambda row: row["clip_id"])
                       for speaker, records in grouped.items()})
    for speaker in groups[0].keys() & groups[1].keys():
        if groups[0][speaker] != groups[1][speaker]:
            raise ValueError("Neutral reference set differs for shared speaker")
    merged = {**groups[0], **groups[1]}
    return [row for speaker in sorted(merged, key=str) for row in merged[speaker]]


def merge(old_baseline, missing_baseline, combined_selection, output):
    """Hard-link (or copy) checked exports into a new, re-bound baseline export."""
    old_baseline, missing_baseline = Path(old_baseline), Path(missing_baseline)
    combined_selection, output = Path(combined_selection), Path(output)
    _new_paths(output)
    if old_baseline.resolve() == missing_baseline.resolve():
        raise ValueError("Distinct old and missing baseline roots required")
    old_manifest, old, old_members = _verified_export(old_baseline)
    new_manifest, new, new_members = _verified_export(missing_baseline)
    required_shared = ("schema", "source", "checkpoint_sha256", "decode_steps", "native_source",
                       "seed_policy", "inference_frozen", "query_motion_inference", "reference_render_exported")
    for key in required_shared:
        if key not in old or key not in new or old[key] != new[key]:
            raise ValueError("Frozen export source/configuration differs: " + key)
    references = _reference_union(old["neutral_reference_bindings"], new["neutral_reference_bindings"])
    selection = _read(combined_selection)
    selected = _members(selection["clips"])
    if old_members.keys() & new_members.keys() or set(selected) != set(old_members) | set(new_members):
        raise ValueError("Old/missing export membership is not an exact disjoint selection cover")
    if selection.get("old_selection_sha256") != old.get("selection_sha256"):
        raise ValueError("Combined selection is not bound to old baseline selection")
    held = set(selection["holdout_sentences"])
    merged_members = {**old_members, **new_members}
    for cid, row in selected.items():
        source = merged_members[cid]
        if any(row[k] != source[k] for k in META_KEYS) or row.get("speaker_name") != source.get("speaker_name"):
            raise ValueError("Combined selection and exported metadata differ")
        if row.get("split") not in ("train", "holdout") or ((row["sentence"] in held) != (row["split"] == "holdout")):
            raise ValueError("Combined roles violate held sentence isolation")
        if cid in new_members and row["split"] != "train":
            raise ValueError("New export includes diagnostic holdout members")
    artifacts = {}
    for root, manifest in ((old_baseline, old_manifest), (missing_baseline, new_manifest)):
        for name, record in manifest.items():
            if PurePosixPath(name).parts[0] not in ("arrays", "video_npz", "audio"):
                continue
            if name in artifacts and artifacts[name][1] != record:
                raise ValueError("Colliding frozen artifact differs")
            artifacts[name] = (root / name, record)
    output.mkdir(parents=True)
    modes = {"hardlinked": 0, "copied": 0}
    for name in sorted(artifacts):
        source_path, record = artifacts[name]
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, target)
            modes["hardlinked"] += 1
        except OSError:
            shutil.copy2(source_path, target)
            modes["copied"] += 1
        checked(target, record)
    provenance = copy.deepcopy(old)
    provenance.update(clips=[copy.deepcopy(merged_members[row["clip_id"]]) for row in selection["clips"]],
        selection_sha256=sha(combined_selection), neutral_reference_bindings=references,
        merge_code_sha256=sha(__file__), merge_sources=[
            {"manifest_sha256": sha(root / "manifest.json"), "provenance_sha256": sha(root / "provenance.json"),
             "selection_sha256": p["selection_sha256"], "clip_count": len(p["clips"])}
            for root, p in ((old_baseline, old), (missing_baseline, new))],
        merge_materialization=modes, new_training=False)
    _write(output / "provenance.json", provenance)
    _write(output / "selection.json", selection)
    jobs = []
    for root, manifest in ((old_baseline, old_manifest), (missing_baseline, new_manifest)):
        if "render_jobs.json" in manifest:
            jobs.extend(_read(root / "render_jobs.json")["jobs"])
    if jobs:
        _write(output / "render_jobs.json", {"schema": BASELINE_SCHEMA, "jobs": jobs, "rendered": False,
            "driver": "scripts/render_dynamic_rig_comparison.py", "paths_relative_to": "directory containing this JSON"})
    manifest = {p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(output.rglob("*")) if p.is_file()}
    _write(output / "manifest.json", manifest)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection_parser = commands.add_parser("select")
    for name in ("prior-protocol", "old-selection", "output-selection", "missing-selection"):
        selection_parser.add_argument("--" + name, type=Path, required=True)
    merge_parser = commands.add_parser("merge")
    for name in ("old-baseline", "missing-baseline", "combined-selection", "output"):
        merge_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        combined, missing = select(args.prior_protocol, args.old_selection, args.output_selection, args.missing_selection)
        print(json.dumps({"combined": combined["counts"], "missing": missing["counts"]}))
    else:
        result = merge(args.old_baseline, args.missing_baseline, args.combined_selection, args.output)
        print(json.dumps({"clips": len(result["clips"]), "materialization": result["merge_materialization"]}))


if __name__ == "__main__":
    main()
