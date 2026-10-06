"""Audit the sealed-test *metadata* without opening test artifacts.

This command is deliberately a metadata-only preflight.  It parses the
manifest JSON, computes its file hash and canonical manifest hash, and checks
the identity-disjoint test protocol.  It never opens an ``artifact`` path and
never loads coefficients, audio, targets, checkpoints, or model features.

The current MEAD/ARKit52 manifest is approved for train/validation preparation
and keeps ``sealed_test_targets_loaded=false``.  The resulting audit therefore
must not be interpreted as a test score or as evidence that a sealed evaluator
has consumed the test set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


EMOTION_NAMES = {
    0: "neutral",
    1: "angry",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sad",
    7: "surprise",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def _rows(manifest: dict[str, Any], role: str, kind: str) -> list[dict[str, Any]]:
    value = manifest.get("roles", {}).get(role, {}).get(kind)
    if not isinstance(value, list):
        raise ValueError(f"roles.{role}.{kind} must be a list")
    if any(not isinstance(row, dict) for row in value):
        raise ValueError(f"roles.{role}.{kind} contains a non-object row")
    return value


def _identity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row.get("speaker"))].append(row)
    return {
        speaker: {
            "rows": len(part),
            "clips": len({row.get("clip_id") for row in part}),
            "sentences": len({row.get("sentence") for row in part}),
            "emotions": sorted({int(row.get("emotion")) for row in part}),
            "emotion_counts": dict(sorted(Counter(int(row.get("emotion")) for row in part).items())),
        }
        for speaker, part in sorted(by_speaker.items())
    }


def audit(path: Path) -> dict[str, Any]:
    # Reading this one JSON file is the entire data access of this command.
    # In particular, no row['artifact'] path is opened or stat'ed.
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be an object")

    roles = manifest.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"train", "val", "test"}:
        raise ValueError("Manifest must expose train, val and test roles")
    test_query = _rows(manifest, "test", "query")
    test_enrollment = _rows(manifest, "test", "enrollment")

    # Every test row is checked from metadata only.  Requiring the fields here
    # prevents a later evaluator from silently inferring identity/sentence
    # exclusions from a filename.
    required = ("clip_id", "dataset", "speaker", "sentence", "emotion",
                "intensity", "source_split", "frames", "valid_frames",
                "artifact", "artifact_sha256", "manifest_role", "track")
    for kind, rows in (("query", test_query), ("enrollment", test_enrollment)):
        for index, row in enumerate(rows):
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"test.{kind}[{index}] missing metadata: {missing}")
            if row["dataset"] != "mead" or row["source_split"] != "test":
                raise ValueError(f"test.{kind}[{index}] has non-test MEAD provenance")
            if row["manifest_role"] != kind:
                raise ValueError(f"test.{kind}[{index}] manifest_role mismatch")
            if not isinstance(row["artifact_sha256"], str) or len(row["artifact_sha256"]) != 64:
                raise ValueError(f"test.{kind}[{index}] artifact hash is not SHA256 metadata")
            if int(row["frames"]) < int(row["valid_frames"]) or int(row["valid_frames"]) < 32:
                raise ValueError(f"test.{kind}[{index}] invalid frame counts")

    query_ids = {row["clip_id"] for row in test_query}
    enrollment_ids = {row["clip_id"] for row in test_enrollment}
    if len(query_ids) != len(test_query) or len(enrollment_ids) != len(test_enrollment):
        raise ValueError("Duplicate test clip_id metadata")
    if query_ids & enrollment_ids:
        raise ValueError("Test query/enrollment clip overlap")

    query_speakers = {row["speaker"] for row in test_query}
    enrollment_speakers = {row["speaker"] for row in test_enrollment}
    if query_speakers != enrollment_speakers:
        raise ValueError("Test query/enrollment identity sets differ")
    if any(int(row["emotion"]) != 0 for row in test_enrollment):
        raise ValueError("Test enrollment must be neutral-only")

    # Reference independence is per identity.  Shared sentence IDs across
    # different speakers are allowed by MEAD; same-speaker query/reference
    # sentences are forbidden.
    per_speaker_query = defaultdict(set)
    per_speaker_ref = defaultdict(set)
    for row in test_query:
        per_speaker_query[row["speaker"]].add(row["sentence"])
    for row in test_enrollment:
        per_speaker_ref[row["speaker"]].add(row["sentence"])
    if any(per_speaker_query[speaker] & per_speaker_ref[speaker] for speaker in query_speakers):
        raise ValueError("Test query/reference sentence overlap within an identity")
    if any(len(per_speaker_ref[speaker]) < 2 for speaker in query_speakers):
        raise ValueError("Every test identity needs at least two neutral references")
    if any(len(per_speaker_ref[speaker]) != len([r for r in test_enrollment if r["speaker"] == speaker])
           for speaker in query_speakers):
        raise ValueError("Duplicate neutral reference sentence for a test identity")

    # The roles must remain identity-disjoint.  This is metadata-only and is
    # the relevant guarantee for an identity-disjoint sealed test.
    role_speakers = {
        role: {row["speaker"] for kind in ("query", "enrollment")
               for row in _rows(manifest, role, kind)}
        for role in ("train", "val", "test")
    }
    cross_role_overlap = {
        f"{left}_vs_{right}": sorted(role_speakers[left] & role_speakers[right])
        for left in role_speakers
        for right in role_speakers
        if left < right
    }
    if any(cross_role_overlap.values()):
        raise ValueError(f"Cross-role speaker overlap: {cross_role_overlap}")

    emotions = sorted({int(row["emotion"]) for row in test_query})
    expected_emotions = list(range(8))
    stored_manifest_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    canonical_manifest_hash = canonical_hash(unsigned)

    checks = {
        "manifest_schema_present": bool(manifest.get("schema")),
        "status_approved_train_val_only": manifest.get("status") == "approved_train_val_only",
        "sealed_test_targets_loaded_false": manifest.get("sealed_test_targets_loaded") is False,
        "test_role_declared": "test" in roles,
        "test_query_count_declared": len(test_query) == int(manifest.get("test_query_count", len(test_query))),
        "test_enrollment_count_declared": len(test_enrollment) == int(manifest.get("test_enrollment_count", len(test_enrollment))),
        "test_identity_count_3": len(query_speakers) == 3,
        "test_emotions_are_declared_subset": set(emotions).issubset(set(expected_emotions)),
        "test_is_eight_emotion": set(emotions) == set(expected_emotions) if manifest.get("test_is_eight_emotion_benchmark") else True,
        "test_query_enrollment_disjoint": not (query_ids & enrollment_ids),
        "test_neutral_enrollment": all(int(row["emotion"]) == 0 for row in test_enrollment),
        "test_per_identity_reference_independence": True,
        "cross_role_identity_disjoint": not any(cross_role_overlap.values()),
        "canonical_manifest_hash_matches": stored_manifest_hash == canonical_manifest_hash,
    }
    result = {
        "schema": "sealed_test_manifest_metadata_audit_v1",
        "scope": "metadata_only; no test coefficient/audio/target/checkpoint reads",
        "manifest": str(path.resolve()),
        "manifest_file_sha256": sha256_file(path),
        "manifest_sha256_stored": stored_manifest_hash,
        "manifest_sha256_canonical": canonical_manifest_hash,
        "manifest_schema": manifest.get("schema"),
        "manifest_status": manifest.get("status"),
        "sealed_test_targets_loaded": manifest.get("sealed_test_targets_loaded"),
        "test_policy": manifest.get("test_policy"),
        "test_role_canonical_sha256": manifest.get("test_role_canonical_sha256"),
        "roles": {
            role: {
                "query_count": len(_rows(manifest, role, "query")),
                "enrollment_count": len(_rows(manifest, role, "enrollment")),
                "speakers": sorted(role_speakers[role]),
            }
            for role in ("train", "val", "test")
        },
        "test": {
            "query_count": len(test_query),
            "enrollment_count": len(test_enrollment),
            "identity_count": len(query_speakers),
            "identities": sorted(query_speakers),
            "query_emotion_ids": emotions,
            "query_emotion_names": [EMOTION_NAMES[e] for e in emotions],
            "expected_eight_emotion_ids": expected_emotions,
            "missing_emotion_ids_from_test": sorted(set(expected_emotions) - set(emotions)),
            "query_identity_summary": _identity_summary(test_query),
            "enrollment_identity_summary": _identity_summary(test_enrollment),
            "reference_sentence_overlap_within_identity": {
                speaker: sorted(per_speaker_query[speaker] & per_speaker_ref[speaker])
                for speaker in sorted(query_speakers)
            },
        },
        "cross_role_speaker_overlap": cross_role_overlap,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "formal_evaluator_blocked_until": [
            "A sealed data-preparation/evaluation entry point accepts the test role without fitting statistics or selecting checkpoints.",
            "The final test run records predictions before any target arrays are opened for scoring.",
            "If an eight-emotion test table is required, the manifest must declare test_is_eight_emotion_benchmark=true and contain all eight IDs.",
        ],
    }
    if not result["all_checks_passed"]:
        raise ValueError(json.dumps(result["checks"], ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(args.output.resolve()), "all_checks_passed": True,
                      "test_query_count": result["test"]["query_count"],
                      "test_emotion_ids": result["test"]["query_emotion_ids"],
                      "metadata_only": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
