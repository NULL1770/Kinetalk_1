"""Lock formal MEAD roles from metadata only; never load native observations.

The original 280 development recordings are preserved. Native val is a
separate, new-identity development role with shared scripts permitted. Native
test is metadata-sealed only. Neither role supplies training observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


EMOTIONS = {0: "neutral", 1: "angry", 5: "happy", 6: "sad"}
CELLS = {(0, 0), (1, 1), (1, 3), (5, 1), (5, 3), (6, 1), (6, 3)}
IDENTITY_KEYS = ("clip_id", "dataset", "speaker", "sentence", "emotion", "intensity", "split", "artifact", "artifact_sha256")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stable(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode("utf8")).hexdigest()


def eligible(row, minimum_frames=32):
    return (row["dataset"] == "mead" and int(row.get("valid_frames", 0)) >= minimum_frames
            and (int(row["emotion"]), int(row["intensity"])) in CELLS)


def sentences(rows):
    return {str(row["sentence"]) for row in rows}


def check_native(sources):
    """Reject conflicting/duplicate native IDs, source and identity overlap."""
    clips, speakers = {}, {}
    for split, rows in sources.items():
        local_speakers = set()
        for row in rows:
            for key in IDENTITY_KEYS[:-1]:
                if key not in row or row[key] == "":
                    raise ValueError(f"Missing native {key}")
            if row["split"] != split:
                raise ValueError(f"Native source split mismatch: {row['clip_id']}")
            if row["clip_id"] in clips:
                raise ValueError(f"Duplicate/conflicting native clip: {row['clip_id']}")
            clips[row["clip_id"]] = row
            local_speakers.add(row["speaker"])
        speakers[split] = local_speakers
    for a in speakers:
        for b in speakers:
            if a < b and speakers[a] & speakers[b]:
                raise ValueError(f"Native identity split overlap: {a}/{b}")
    return clips


def check_selected_native(selected, by_id):
    for row in selected:
        native = by_id.get(row["clip_id"])
        if native is None or any(row.get(key) != native.get(key) for key in IDENTITY_KEYS):
            raise ValueError(f"Historical locked recording changed: {row['clip_id']}")
        if not eligible(native):
            raise ValueError(f"Historical locked recording is no longer eligible: {row['clip_id']}")
        if int(row.get("emotion_id", row["emotion"])) != int(native["emotion"]):
            raise ValueError("Preserve native eight-class emotion IDs; never remap happy/sad to 2/3")


def collect_historical_development(paths):
    result, sources = set(), []
    for path in paths:
        path = Path(path)
        if path.suffix == ".jsonl":
            values = sentences(read_rows(path))
        elif path.suffix == ".json":
            data = read_json(path)
            if "selected_sentence_ids" not in data:
                raise ValueError(f"Historical development JSON lacks explicit selected_sentence_ids: {path}")
            values = set(map(str, data["selected_sentence_ids"]))
        else:
            raise ValueError("Historical development inputs must be explicit jsonl or selected-sentence JSON metadata")
        result.update(values)
        sources.append({"path": str(path.resolve()), "sha256": sha(path), "sentences": sorted(values),
                        "reason": "explicit_historical_development_selection"})
    return result, sources


def coverage(rows, speakers=None):
    speaker_set = set(speakers if speakers is not None else (r["speaker"] for r in rows))
    present = {(r["speaker"], int(r["emotion"]), int(r["intensity"])) for r in rows}
    expected = {(s, emotion, intensity) for s in speaker_set for emotion, intensity in CELLS}
    return {"clips": len(rows), "sentences": len(sentences(rows)), "speakers": sorted({r["speaker"] for r in rows}),
            "speaker_count": len({r["speaker"] for r in rows}),
            "emotion_counts": dict(sorted(Counter(EMOTIONS[int(r["emotion"])] for r in rows).items())),
            "nonneutral_sentences": len(sentences([r for r in rows if int(r["emotion"]) != 0])),
            "by_emotion_sentences": {name: len(sentences([r for r in rows if int(r["emotion"]) == e]))
                                     for e, name in EMOTIONS.items()},
            "populated_cells": len(present), "expected_cells": len(expected),
            "missing_cells": [{"speaker": s, "emotion": EMOTIONS[e], "intensity": i}
                              for s, e, i in sorted(expected - present)]}


def select_formal(sources, locked, lock_info, *, extra_dev_sentences=(), seed=20260920,
                  min_references=2, max_references=4):
    if not 2 <= min_references <= max_references <= 4:
        raise ValueError("Require 2 <= min_references <= max_references <= 4")
    native_by_id = check_native(sources)
    for role in ("train", "validation", "enrollment", "new_test"):
        check_selected_native(locked[role], native_by_id)
    previous_mapping = {str(name): int(sid) for name, sid in lock_info["speaker_to_id"].items()}
    if len(set(previous_mapping.values())) != len(previous_mapping) or min(previous_mapping.values()) < 0:
        raise ValueError("Historical speaker mapping is invalid")
    for row in locked["train"] + locked["validation"] + locked["enrollment"]:
        if previous_mapping.get(row["speaker"]) != int(row["speaker_id"]):
            raise ValueError("Historical speaker ID changed")
    preferred_refs = sentences(locked["enrollment"])
    allocations = lock_info["sentence_allocation"]
    enrollment_sentences = preferred_refs | set(map(str, allocations["enrollment"]))
    reserved = sentences(locked["new_test"]) | set(map(str, allocations["new_test"]))
    dev_sentences = sentences(locked["validation"]) | set(map(str, allocations["validation"])) | set(map(str, extra_dev_sentences))
    if (reserved & enrollment_sentences) or (dev_sentences & enrollment_sentences) or (dev_sentences & reserved):
        raise ValueError("Historical enrollment/development/reserved sentence roles conflict")
    train_candidates = [r for r in sources["train"] if eligible(r)]
    old_refs = defaultdict(list)
    for row in locked["enrollment"]:
        if int(row["emotion"]) != 0:
            raise ValueError("Historical enrollment must be neutral")
        old_refs[row["speaker"]].append(row)
    reference_roles, included, excluded = {}, {}, []
    for split in ("train", "val", "test"):
        candidates = [r for r in sources[split] if eligible(r)]
        speakers = sorted({r["speaker"] for r in candidates})
        reference_roles[split] = []
        included[split] = []
        for speaker in speakers:
            neutral = [r for r in candidates if r["speaker"] == speaker and int(r["emotion"]) == 0
                       and r["sentence"] in enrollment_sentences]
            grouped = defaultdict(list)
            for row in neutral:
                grouped[row["sentence"]].append(row)
            if split == "train" and speaker in old_refs:
                selected = old_refs[speaker]
                if len(selected) > max_references or len(sentences(selected)) != len(selected):
                    raise ValueError("Cannot preserve historical enrollment under requested reference contract")
                if {r["clip_id"] for r in selected} - {r["clip_id"] for r in neutral}:
                    raise ValueError("Historical neutral references missing from native metadata")
            else:
                selected = [min(values, key=lambda r: (stable(r["clip_id"], seed), r["clip_id"]))
                            for _, values in sorted(grouped.items(), key=lambda pair: (stable(pair[0], seed), pair[0]))][:max_references]
            if len(selected) < min_references:
                if speaker in previous_mapping:
                    raise ValueError(f"Previously enrolled identity lacks required references: {speaker}")
                excluded.append({"speaker": speaker, "source_split": split, "available_references": len(selected),
                                 "minimum": min_references, "reason": "insufficient_neutral_references_on_locked_enrollment_sentences",
                                 "eligible_query_clips": len([r for r in candidates if r["speaker"] == speaker])})
                continue
            included[split].append(speaker)
            reference_roles[split].extend(selected)
    if set(previous_mapping) - set(included["train"]):
        raise ValueError("All previously enrolled training identities must remain represented")
    mapping = dict(previous_mapping)
    next_id = max(mapping.values()) + 1
    for split in ("train", "val", "test"):
        for speaker in included[split]:
            if speaker not in mapping:
                mapping[speaker] = next_id
                next_id += 1

    def tagged(rows, role):
        return [dict(row, speaker_id=mapping[row["speaker"]], emotion_id=int(row["emotion"]),
                     intensity_id=int(row["intensity"]), pilot_split=role) for row in rows]

    train = [r for r in train_candidates if r["speaker"] in included["train"]
             and r["sentence"] not in dev_sentences | reserved | enrollment_sentences]
    train.sort(key=lambda r: (mapping[r["speaker"]], int(r["emotion"]), int(r["intensity"]), r["sentence"], r["clip_id"]))
    if {r["clip_id"] for r in locked["train"]} - {r["clip_id"] for r in train}:
        raise ValueError("Formal training would reassign a prior training clip; inspect exposure-role conflicts")
    # New identity development/test deliberately share scripts with training.
    # They are source-identity isolation protocols, not novel-sentence tests.
    outputs = {"train": tagged(train, "formal_train"), "validation": locked["validation"],
               "enrollment": tagged(reference_roles["train"], "formal_enrollment"),
               "reserved_test_metadata": locked["new_test"]}
    for split, query_role, reference_role in (
        ("val", "new_identity_validation", "new_identity_enrollment"),
        ("test", "sealed_test_metadata", "sealed_test_enrollment_metadata")):
        queries = [r for r in sources[split] if eligible(r) and r["speaker"] in included[split]
                   and r["sentence"] not in enrollment_sentences | reserved]
        queries.sort(key=lambda r: (mapping[r["speaker"]], int(r["emotion"]), int(r["intensity"]), r["sentence"], r["clip_id"]))
        outputs[query_role] = tagged(queries, query_role)
        outputs[reference_role] = tagged(reference_roles[split], reference_role)
    query_roles = ("train", "validation", "new_identity_validation", "sealed_test_metadata")
    query_sentences = set.union(*(sentences(outputs[role]) for role in query_roles))
    checks = {
        "train_excludes_all_historical_development_sentences": not bool(sentences(train) & dev_sentences),
        "train_excludes_reserved_sentences": not bool(sentences(train) & reserved),
        "train_validation_sentence_disjoint": not bool(sentences(train) & sentences(outputs["validation"])),
        "all_query_roles_exclude_enrollment_sentences": not bool(query_sentences & enrollment_sentences),
        "all_query_roles_exclude_reserved_sentences": not bool(query_sentences & reserved),
        "original_training_is_subset": {r["clip_id"] for r in locked["train"]} <= {r["clip_id"] for r in train},
        "original_validation_unchanged": outputs["validation"] == locked["validation"],
        "original_speaker_ids_unchanged": all(mapping[k] == v for k, v in previous_mapping.items()),
        "train_has_native_train_only": all(r["split"] == "train" for r in train),
        "new_identity_validation_disjoint_from_training_identities": not bool(set(included["train"]) & set(included["val"])),
        "sealed_test_identities_disjoint_from_train_and_development": not bool(set(included["test"]) & (set(included["train"]) | set(included["val"]))),
    }
    if not all(checks.values()):
        raise AssertionError(f"Formal metadata isolation failed: {checks}")
    coverage_roles = {name: coverage(rows) for name, rows in outputs.items() if "enrollment" not in name}
    ref_counts = {split: dict(sorted(Counter(r["speaker"] for r in rows).items())) for split, rows in reference_roles.items()}
    report = {"schema": "formal_predictable_metadata_lock_v1", "metadata_only": True, "seed": seed,
        "native_observations_read": False, "test_targets_read": False, "status": "locked_before_materialization",
        "speaker_to_id": mapping, "original_speaker_to_id": previous_mapping,
        "train_speaker_to_id": {s: mapping[s] for s in included["train"]},
        "new_identity_validation_speaker_to_id": {s: mapping[s] for s in included["val"]},
        "sealed_test_speaker_to_id": {s: mapping[s] for s in included["test"]},
        "emotion_id_contract": {str(k): name for k, name in EMOTIONS.items()}, "min_valid_frames": 32,
        "min_references": min_references, "max_references": max_references, "reference_counts": ref_counts,
        "excluded_identities": excluded, "coverage": coverage_roles,
        "source_eligible_train_clips": len(train_candidates),
        "training_exclusion_counts": {
            "historical_development_sentence": sum(r["sentence"] in dev_sentences for r in train_candidates),
            "reserved_sentence": sum(r["sentence"] in reserved for r in train_candidates),
            "enrollment_sentence": sum(r["sentence"] in enrollment_sentences for r in train_candidates),
            "insufficient_enrollment_identity": sum(r["speaker"] not in included["train"] for r in train_candidates)},
        "sentence_roles": {"historical_development": sorted(dev_sentences), "enrollment": sorted(enrollment_sentences),
                           "reserved_prior_test": sorted(reserved)}, "isolation_checks": checks,
        "cross_identity_sentence_overlap": {role: len(sentences(outputs[role]) & sentences(train))
                                            for role in ("new_identity_validation", "sealed_test_metadata")},
        "training_expansion": {"old_clips": len(locked["train"]), "new_total": len(train),
                               "additional_clips": len(train) - len(locked["train"])},
        "selection_policy": "All eligible native-train MEAD recordings except historical development, enrollment and reserved sentences; no training budget cap. Existing neutral references preserved, newcomers use at most four and at least two distinct locked enrollment sentences; insufficient identities excluded without relaxing roles.",
        "scope": "Formal training metadata lock only. Original validation remains development. Native val is separate cross-identity development with shared scripts permitted. Native test is metadata-sealed, not loaded or evaluated. B0/global external exposure remains uncertified; never claim whole-system unseen identities or fresh sentences."}
    return outputs, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True, help="Metadata copies or native directory; only train/val/test.jsonl are opened")
    parser.add_argument("--locked-root", type=Path, required=True, help="Existing run31 locked manifests")
    parser.add_argument("--historical-development", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--min-references", type=int, default=2)
    parser.add_argument("--max-references", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh formal lock directory; prior selections are never overwritten")
    sources = {name: read_rows(args.native_root / f"{name}.jsonl") for name in ("train", "val", "test")}
    locked = {role: read_rows(args.locked_root / f"{role}.jsonl") for role in ("train", "validation", "enrollment", "new_test")}
    lock_info = read_json(args.locked_root / "selection.json")
    for role, expected in lock_info.get("manifest_sha256", {}).items():
        if role in locked and sha(args.locked_root / f"{role}.jsonl") != expected:
            raise ValueError(f"Historical locked {role} manifest hash changed")
    extra_dev, extra_sources = collect_historical_development(args.historical_development)
    outputs, report = select_formal(sources, locked, lock_info, extra_dev_sentences=extra_dev,
        seed=args.seed, min_references=args.min_references, max_references=args.max_references)
    paths = [args.native_root / f"{name}.jsonl" for name in sources] + [args.locked_root / "selection.json"]
    paths += [args.locked_root / f"{role}.jsonl" for role in locked]
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256={str(path.resolve()): sha(path) for path in paths},
                  additional_development_sources=extra_sources, preparation_script_sha256=sha(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    for role, rows in outputs.items():
        destination = args.output / f"{role}.jsonl"
        if role == "validation":
            shutil.copyfile(args.locked_root / "validation.jsonl", destination)
        elif role == "reserved_test_metadata":
            shutil.copyfile(args.locked_root / "new_test.jsonl", destination)
        else:
            destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
    report["manifest_sha256"] = {role: sha(args.output / f"{role}.jsonl") for role in outputs}
    report["isolation_checks"]["original_validation_byte_identical"] = report["manifest_sha256"]["validation"] == sha(args.locked_root / "validation.jsonl")
    (args.output / "selection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({"status": report["status"], "metadata_only": True,
        "roles": {role: {k: counts[k] for k in ("clips", "sentences", "speaker_count")} for role, counts in report["coverage"].items()},
        "excluded_identities": report["excluded_identities"], "all_isolation_checks": all(report["isolation_checks"].values()),
        "output": str(args.output)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
