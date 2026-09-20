"""Create a metadata-only, support-balanced joint-disjoint MEAD candidate.

The first joint candidate used a global hash over sentence IDs.  That is
reproducible, but it can assign sparse or role-specific sentences to test and
leave an unstable query count.  This candidate keeps the source speaker
split, assigns only sentences with adequate non-neutral support in all three
roles to validation/test, and records the deterministic selection rule.

No motion/audio arrays are opened and no sealed-test target is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from lock_paper_manifests_20260920 import (
    DATASET,
    EMOTIONS,
    SCHEMA,
    choose_enrollment,
    four_class,
    metadata_row,
    read_rows,
    sha,
    stable_hex,
    summarize,
)


def nonneutral_support(rows: dict[str, list[dict]]) -> dict[str, Counter]:
    return {
        role: Counter(
            r["sentence"]
            for r in rows[role]
            if four_class(r) and int(r["emotion"]) != 0
        )
        for role in ("train", "val", "test")
    }


def assign_sentences(rows: dict[str, list[dict]], *, val_sentences: int = 12,
                     test_sentences: int = 8, min_support: int = 12) -> tuple[dict[str, str], dict]:
    support = nonneutral_support(rows)
    universe = sorted(set().union(*(set(c) for c in support.values())))
    common = [
        sentence for sentence in universe
        if all(support[role][sentence] >= min_support for role in ("train", "val", "test"))
    ]
    common.sort(key=lambda s: (stable_hex("kinetalk_balanced_joint_v1:" + s), s))
    needed = val_sentences + test_sentences
    if len(common) < needed:
        raise ValueError(f"only {len(common)} robust common sentences; need {needed}")
    assignment = {}
    test_set = set(common[:test_sentences])
    val_set = set(common[test_sentences:test_sentences + val_sentences])
    for sentence in universe:
        assignment[sentence] = "test" if sentence in test_set else "val" if sentence in val_set else "train"
    details = {
        "selection": "stable hash among sentences with >= min_support non-neutral clips in every source role",
        "min_support_non_neutral_per_role": min_support,
        "robust_common_sentence_count": len(common),
        "val_sentence_count": val_sentences,
        "test_sentence_count": test_sentences,
        "robust_common_order": common,
        "support": {role: dict(sorted(counter.items())) for role, counter in support.items()},
    }
    return assignment, details


def build(rows: dict[str, list[dict]], *, val_sentences: int = 12,
          test_sentences: int = 8, min_support: int = 12) -> dict:
    track = "joint_balanced_disjoint"
    assignment, details = assign_sentences(
        rows, val_sentences=val_sentences, test_sentences=test_sentences,
        min_support=min_support,
    )
    filtered = {role: [r for r in rows[role] if four_class(r)] for role in rows}
    # Neutral-only sentences have no non-neutral support and therefore never
    # enter the held-out sentence sets; keep them in the training role so that
    # neutral enrollment remains available without leaking held-out sentences.
    all_sentences = {r["sentence"] for role in filtered for r in filtered[role]}
    assignment.update({sentence: "train" for sentence in all_sentences if sentence not in assignment})
    out = {
        "schema": SCHEMA,
        "track": track,
        "dataset": DATASET,
        "filter": "emotion in {neutral, angry, happy, sad}; neutral any intensity; nonneutral intensity in {1,3}",
        "sentence_policy": "speaker-disjoint source roles; query sentences are globally disjoint; sparse/role-specific sentences forced to train",
        "sentence_assignment": assignment,
        "selection_rule": details,
        "roles": {},
        "valid": True,
    }
    for role in ("train", "val", "test"):
        query_pool = [r for r in filtered[role] if assignment[r["sentence"]] == role]
        if not query_pool:
            raise ValueError(f"No query rows in {track}/{role}")
        # Use train-assigned sentence IDs for neutral enrollment in every
        # role.  This prevents enrollment selection from consuming a whole
        # held-out query sentence for a three-speaker validation/test split.
        restrict = {s for s, assigned in assignment.items() if assigned == "train"}
        enrollment = choose_enrollment(
            filtered[role], query_pool, role, track, restrict_sentences=restrict,
        )
        enroll_ids = {r["clip_id"] for r in enrollment}
        enroll_pairs = {(r["speaker"], r["sentence"]) for r in enrollment}
        query = [
            r for r in query_pool
            if r["clip_id"] not in enroll_ids
            and (r["speaker"], r["sentence"]) not in enroll_pairs
        ]
        if not query:
            raise ValueError(f"Enrollment consumed all query rows in {track}/{role}")
        out["roles"][role] = {
            "query": [metadata_row(r, role, track, True) for r in query],
            "enrollment": [metadata_row(r, role, track, False) for r in enrollment],
            "summary": summarize(query, enrollment),
        }
    q_sentences = {
        role: {r["sentence"] for r in out["roles"][role]["query"]}
        for role in out["roles"]
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if q_sentences[left] & q_sentences[right]:
            raise AssertionError(f"query sentence overlap: {left}/{right}")
    out["query_sentence_counts"] = {role: len(values) for role, values in q_sentences.items()}
    out["query_enrollment_disjoint"] = True
    out["sealed_test_targets_loaded"] = False
    out["candidate_status"] = "metadata_locked_candidate_requires_review_before_training"
    return out


def finalize(manifest: dict, source_hashes: dict[str, str]) -> dict:
    manifest = dict(manifest)
    manifest["source_metadata_sha256"] = source_hashes
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, default=Path("artifacts/formal_readiness/native_metadata"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_takeover_20260920/paper_manifests_20260920/joint_balanced_disjoint.json"))
    parser.add_argument("--val-sentences", type=int, default=12)
    parser.add_argument("--test-sentences", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=12)
    args = parser.parse_args()
    rows, hashes = read_rows(args.metadata_root)
    manifest = finalize(build(rows, val_sentences=args.val_sentences,
                               test_sentences=args.test_sentences,
                               min_support=args.min_support), hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"track": manifest["track"], "manifest": str(args.output),
                      "manifest_sha256": manifest["manifest_sha256"],
                      "summaries": {role: manifest["roles"][role]["summary"] for role in ("train", "val", "test")}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
