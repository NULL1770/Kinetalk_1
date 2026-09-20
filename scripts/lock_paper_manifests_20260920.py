"""Lock metadata-only MEAD paper manifests for KineTalk.

This script never opens motion/audio arrays. It consumes the checked-in native
metadata JSONL copies, applies the predeclared four-class filter, and writes two
tracks:

* identity_disjoint: existing speaker-disjoint train/val/test protocol;
* joint_disjoint: speaker and sentence disjoint, with sentence assignment made
  from a stable hash over the complete filtered sentence universe.

Neutral enrollment clips are removed from query membership. They are conditioning
references, not query targets. The output is a protocol candidate and must be
reviewed before any formal training or sealed-test evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA = "kinetalk_paper_manifest_candidate_v1"
DATASET = "mead"
EMOTIONS = {0, 1, 5, 6}  # neutral, angry, happy, sad in the native metadata


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def read_rows(root: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    rows, hashes = {}, {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[split] = sha(path)
        part = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
        part = [r for r in part if r.get("dataset") == DATASET]
        rows[split] = part
    return rows, hashes


def four_class(row: dict) -> bool:
    emotion = int(row["emotion"])
    intensity = int(row["intensity"])
    return emotion in EMOTIONS and (emotion == 0 or intensity in {1, 3})


def metadata_row(row: dict, role: str, track: str, query: bool) -> dict:
    return {
        "clip_id": row["clip_id"], "dataset": row["dataset"],
        "speaker": row["speaker"], "sentence": row["sentence"],
        "emotion": int(row["emotion"]), "intensity": int(row["intensity"]),
        "source_split": role, "frames": int(row["frames"]),
        "valid_frames": int(row["valid_frames"]), "stage1": bool(row.get("stage1", False)),
        "artifact": row["artifact"], "artifact_sha256": row["artifact_sha256"],
        "manifest_role": "query" if query else "enrollment",
        "track": track,
    }


def choose_enrollment(rows: list[dict], query_rows: list[dict], role: str, track: str,
                      count: int = 2, restrict_sentences: set[str] | None = None) -> list[dict]:
    speakers = sorted({r["speaker"] for r in query_rows})
    selected: list[dict] = []
    for speaker in speakers:
        # ``rows`` may be the same source pool as ``query_rows``. Enrollment
        # is selected first and then removed from queries; excluding every
        # query-row ID here would make the candidate set empty.
        candidates = [r for r in rows if r["speaker"] == speaker and int(r["emotion"]) == 0]
        if restrict_sentences is not None:
            restricted = [r for r in candidates if r["sentence"] in restrict_sentences]
            if len(restricted) >= count:
                candidates = restricted
        candidates.sort(key=lambda r: (stable_hex(f"{track}:enrollment:{speaker}:{r['clip_id']}"), r["clip_id"]))
        picked: list[dict] = []
        sentences: set[str] = set()
        for row in candidates:
            if row["sentence"] in sentences:
                continue
            picked.append(row); sentences.add(row["sentence"])
            if len(picked) >= count:
                break
        if len(picked) < count:
            raise ValueError(f"{track}/{role}: speaker {speaker} has only {len(picked)} independent neutral enrollment clips")
        selected.extend(picked)
    return selected


def summarize(query: list[dict], enrollment: list[dict]) -> dict:
    def one(rows: list[dict]) -> dict:
        return {
            "clips": len(rows), "speakers": sorted({r["speaker"] for r in rows}),
            "speaker_count": len({r["speaker"] for r in rows}),
            "sentences": sorted({r["sentence"] for r in rows}),
            "sentence_count": len({r["sentence"] for r in rows}),
            "valid_frames": sum(int(r["valid_frames"]) for r in rows),
            "hours_at_25fps": sum(int(r["valid_frames"]) for r in rows) / 25 / 3600,
            "emotion_counts": dict(sorted(Counter(str(r["emotion"]) for r in rows).items())),
            "intensity_counts": dict(sorted(Counter(str(r["intensity"]) for r in rows).items())),
        }
    return {"query": one(query), "enrollment": one(enrollment)}


def build_identity(rows: dict[str, list[dict]]) -> dict:
    track = "identity_disjoint"
    filtered = {role: [r for r in rows[role] if four_class(r)] for role in rows}
    out = {"schema": SCHEMA, "track": track,
           "dataset": DATASET, "filter": "emotion in {neutral, angry, happy, sad}; neutral any intensity; nonneutral intensity in {1,3}",
           "sentence_policy": "sentences may overlap across speaker-disjoint splits; this is an identity-generalization track",
           "roles": {}, "valid": True}
    for role in ("train", "val", "test"):
        pool = filtered[role]
        enrollment = choose_enrollment(pool, pool, role, track)
        enroll_ids = {r["clip_id"] for r in enrollment}
        enroll_sentences = {(r["speaker"], r["sentence"]) for r in enrollment}
        query = [r for r in pool if r["clip_id"] not in enroll_ids
                 and (r["speaker"], r["sentence"]) not in enroll_sentences]
        out["roles"][role] = {
            "query": [metadata_row(r, role, track, True) for r in query],
            "enrollment": [metadata_row(r, role, track, False) for r in enrollment],
            "summary": summarize(query, enrollment),
        }
    return out


def sentence_assignment(rows: dict[str, list[dict]]) -> dict[str, str]:
    universe = sorted({r["sentence"] for role in rows for r in rows[role] if four_class(r)})
    assignment = {}
    for sentence in universe:
        value = int(stable_hex("kinetalk_joint_sentence_v1:" + sentence)[:8], 16) % 10
        assignment[sentence] = "train" if value < 7 else "val" if value < 9 else "test"
    return assignment


def build_joint(rows: dict[str, list[dict]]) -> dict:
    track = "joint_disjoint"
    assignment = sentence_assignment(rows)
    out = {"schema": SCHEMA, "track": track, "dataset": DATASET,
           "filter": "emotion in {neutral, angry, happy, sad}; neutral any intensity; nonneutral intensity in {1,3}",
           "sentence_policy": "stable global sentence hash: 70% train, 20% val, 10% test; speaker and query sentence are disjoint",
           "sentence_assignment": assignment, "roles": {}, "valid": True}
    filtered = {role: [r for r in rows[role] if four_class(r)] for role in rows}
    for role in ("train", "val", "test"):
        # Preserve the original speaker split and apply the global sentence role.
        query_pool = [r for r in filtered[role] if assignment[r["sentence"]] == role]
        if not query_pool:
            raise ValueError(f"No query rows in {track}/{role}")
        # For validation/test identity enrollment, use only that speaker's source
        # split; enrollment is never used to optimize parameters. For train we
        # restrict to train-assigned sentences to keep the fit fully sentence-safe.
        restrict = set(assignment) if role != "train" else {s for s, r in assignment.items() if r == "train"}
        enrollment = choose_enrollment(filtered[role], query_pool, role, track, restrict_sentences=restrict)
        enroll_ids = {r["clip_id"] for r in enrollment}
        enroll_sentences = {(r["speaker"], r["sentence"]) for r in enrollment}
        query = [r for r in query_pool if r["clip_id"] not in enroll_ids
                 and (r["speaker"], r["sentence"]) not in enroll_sentences]
        if not query:
            raise ValueError(f"Enrollment consumed all query rows in {track}/{role}")
        out["roles"][role] = {
            "query": [metadata_row(r, role, track, True) for r in query],
            "enrollment": [metadata_row(r, role, track, False) for r in enrollment],
            "summary": summarize(query, enrollment),
        }
    # Exact disjointness checks are part of the written artifact contract.
    q_sentences = {role: {r["sentence"] for r in out["roles"][role]["query"]} for role in out["roles"]}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if q_sentences[a] & q_sentences[b]:
            raise AssertionError(f"Query sentence overlap: {a}/{b}")
    return out


def finalize(manifest: dict, source_hashes: dict[str, str]) -> dict:
    manifest = dict(manifest)
    manifest["source_metadata_sha256"] = source_hashes
    manifest["query_enrollment_disjoint"] = True
    manifest["sealed_test_targets_loaded"] = False
    manifest["candidate_status"] = "metadata_locked_candidate_requires_review_before_training"
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, default=Path("artifacts/formal_readiness/native_metadata"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_takeover_20260920/paper_manifests_20260920"))
    args = parser.parse_args()
    rows, hashes = read_rows(args.metadata_root)
    args.output.mkdir(parents=True, exist_ok=True)
    for builder, name in ((build_identity, "identity_disjoint"), (build_joint, "joint_disjoint")):
        manifest = finalize(builder(rows), hashes)
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"track": name, "manifest": str(path), "manifest_sha256": manifest["manifest_sha256"],
                          "summaries": {role: manifest["roles"][role]["summary"] for role in ("train", "val", "test")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
