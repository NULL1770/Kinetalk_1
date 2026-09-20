"""Read-only preflight for metadata-only KineTalk paper candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def check_manifest(path: Path, metadata_root: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf8"))
    errors: list[str] = []
    roles = manifest.get("roles", {})
    fail(errors, set(roles) == {"train", "val", "test"}, "roles must be train/val/test")
    fail(errors, manifest.get("sealed_test_targets_loaded") is False, "sealed test target flag is not false")
    fail(errors, manifest.get("query_enrollment_disjoint") is True, "query/enrollment flag is not true")
    fail(errors, manifest.get("candidate_status") == "metadata_locked_candidate_requires_review_before_training",
         "candidate status changed or is missing")

    source_hashes = {}
    for role in ("train", "val", "test"):
        source = metadata_root / f"{role}.jsonl"
        source_hashes[role] = sha(source)
    fail(errors, source_hashes == manifest.get("source_metadata_sha256"), "source metadata SHA256 mismatch")

    unsigned = dict(manifest)
    recorded_manifest_hash = unsigned.pop("manifest_sha256", None)
    canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    fail(errors, hashlib.sha256(canonical).hexdigest() == recorded_manifest_hash, "manifest SHA256 mismatch")

    summaries = {}
    all_query_sentences = {}
    for role in ("train", "val", "test"):
        rd = roles.get(role, {})
        query = rd.get("query", [])
        enrollment = rd.get("enrollment", [])
        q_ids = [r.get("clip_id") for r in query]
        e_ids = [r.get("clip_id") for r in enrollment]
        q_pairs = {(r.get("speaker"), r.get("sentence")) for r in query}
        e_pairs = {(r.get("speaker"), r.get("sentence")) for r in enrollment}
        fail(errors, len(q_ids) == len(set(q_ids)), f"{role}: duplicate query clip ID")
        fail(errors, len(e_ids) == len(set(e_ids)), f"{role}: duplicate enrollment clip ID")
        fail(errors, not (set(q_ids) & set(e_ids)), f"{role}: query/enrollment clip overlap")
        fail(errors, not (q_pairs & e_pairs), f"{role}: query/enrollment speaker-sentence overlap")
        fail(errors, all(r.get("source_split") == role for r in query + enrollment), f"{role}: source split mismatch")
        fail(errors, all(isinstance(r.get("artifact_sha256"), str) and len(r["artifact_sha256"]) == 64 for r in query + enrollment),
             f"{role}: invalid artifact hash")
        fail(errors, len(enrollment) >= 2 * len({r.get("speaker") for r in query}), f"{role}: fewer than two enrollment clips per speaker")
        summaries[role] = {
            "query_clips": len(query),
            "enrollment_clips": len(enrollment),
            "query_speakers": len({r.get("speaker") for r in query}),
            "query_sentences": len({r.get("sentence") for r in query}),
            "query_valid_frames": sum(int(r.get("valid_frames", 0)) for r in query),
            "enrollment_valid_frames": sum(int(r.get("valid_frames", 0)) for r in enrollment),
            "query_emotions": dict(sorted(Counter(str(r.get("emotion")) for r in query).items())),
        }
        all_query_sentences[role] = {r.get("sentence") for r in query}

    if manifest.get("track") in {"joint_disjoint", "joint_balanced_disjoint"}:
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            fail(errors, not (all_query_sentences[left] & all_query_sentences[right]),
                 f"joint track query sentence overlap: {left}/{right}")
    assignment = manifest.get("sentence_assignment")
    if isinstance(assignment, dict):
        for role, sentences in all_query_sentences.items():
            fail(errors, all(assignment.get(s) == role for s in sentences), f"{role}: assignment mismatch")

    return {
        "manifest": str(path),
        "track": manifest.get("track"),
        "manifest_sha256": recorded_manifest_hash,
        "source_metadata_sha256": source_hashes,
        "summaries": summaries,
        "query_sentence_counts": {role: len(v) for role, v in all_query_sentences.items()},
        "sealed_test_targets_loaded": manifest.get("sealed_test_targets_loaded"),
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, default=Path("artifacts/formal_readiness/native_metadata"))
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_takeover_20260920/paper_manifest_preflight_20260920.json"))
    args = parser.parse_args()
    results = [check_manifest(path, args.metadata_root) for path in args.manifest]
    report = {"schema": "kinetalk_paper_manifest_preflight_v1", "metadata_only": True,
              "sealed_test_targets_loaded": False, "results": results,
              "passed": all(item["passed"] for item in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "tracks": {item["track"]: item["passed"] for item in results}}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
