"""Build the actual eight-emotion MEAD sealed-test metadata role.

This command reads only the native ``test.jsonl`` metadata file.  It never
opens an NPZ artifact.  The existing train/validation roles are copied
verbatim, while the old four-emotion test role is replaced by all eligible
MEAD test clips for the same held-out identities.  Two neutral enrollment
clips per identity are selected deterministically and every clip sharing an
enrollment sentence for that identity is excluded from test queries.

The resulting manifest is still ``sealed_test_targets_loaded=false``.  A
separate final evaluator is responsible for opening test artifacts after the
checkpoint is frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_emotion2vec_pilot import sha
from scripts.prepare_paper_full_data import validate_manifest
from scripts.train_formal_predictable_projection import canonical_hash, save_json

SCHEMA = "paper_mead_eight_emotion_sealed_manifest_v2"
EMOTION_NAMES = ("neutral", "angry", "contempt", "disgust", "fear", "happy", "sad", "surprise")
REQUIRED = ("clip_id", "dataset", "speaker", "sentence", "emotion", "intensity",
            "artifact", "artifact_sha256", "frames", "valid_frames")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _verify(row: dict) -> None:
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        raise ValueError(f"Missing native metadata fields for {row.get('clip_id')}: {missing}")
    if row.get("dataset") != "mead" or row.get("split", row.get("source_split")) != "test":
        raise ValueError(f"Non-MEAD-test row: {row.get('clip_id')}")
    if row.get("source_split", "test") != "test":
        raise ValueError(f"source_split mismatch: {row.get('clip_id')}")
    if type(row["emotion"]) is not int or row["emotion"] not in range(8):
        raise ValueError(f"Invalid emotion: {row.get('clip_id')}")
    expected = {0} if row["emotion"] == 0 else {1, 2, 3}
    if type(row["intensity"]) is not int or row["intensity"] not in expected:
        raise ValueError(f"Invalid MEAD intensity: {row.get('clip_id')}")
    if int(row["frames"]) < int(row["valid_frames"]) or int(row["valid_frames"]) < 32:
        raise ValueError(f"Invalid frame coverage: {row.get('clip_id')}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(row["artifact_sha256"])):
        raise ValueError(f"Invalid artifact hash: {row.get('clip_id')}")
    path = PurePosixPath(str(row["artifact"]).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError(f"Unsafe artifact path: {row.get('clip_id')}")


def _decorate(row: dict, role: str, kind: str) -> dict:
    out = copy.deepcopy(row)
    out["source_split"] = "test"
    out["manifest_role"] = kind
    out["track"] = "identity_disjoint"
    return out


def _summary(rows: list[dict]) -> dict:
    return {
        "clips": len(rows),
        "speakers": sorted({r["speaker"] for r in rows}),
        "emotions": sorted({int(r["emotion"]) for r in rows}),
        "emotion_counts": dict(sorted(Counter(int(r["emotion"]) for r in rows).items())),
        "intensity_counts": dict(sorted(Counter(int(r["intensity"]) for r in rows).items())),
        "sentences": len({(r["speaker"], r["sentence"]) for r in rows}),
        "valid_frames": sum(int(r["valid_frames"]) for r in rows),
    }


def build(base: dict, native_rows: list[dict], *, native_sha: str, base_sha: str) -> tuple[dict, dict]:
    validate_manifest(base)
    # The native test metadata file also contains CREMA-D rows.  They are
    # outside this MEAD benchmark and must be excluded before validation.
    native_rows = [row for row in native_rows if row.get("dataset") == "mead"]
    for row in native_rows:
        _verify(row)
    old_test_people = {r["speaker"] for r in base["roles"]["test"]["query"]}
    if not old_test_people:
        raise ValueError("Base test role has no held-out identities")
    rows = [r for r in native_rows if r["speaker"] in old_test_people]
    by_person: dict[str, list[dict]] = {p: [] for p in sorted(old_test_people)}
    for row in rows:
        by_person[row["speaker"]].append(row)

    enrollment: list[dict] = []
    excluded: list[dict] = []
    query: list[dict] = []
    # Stable ordering is independent of filesystem ordering.
    for person in sorted(old_test_people):
        part = sorted(by_person[person], key=lambda r: (r["sentence"], r["clip_id"]))
        neutral = [r for r in part if int(r["emotion"]) == 0]
        if len(neutral) < 2:
            raise ValueError(f"Not enough neutral references for {person}")
        refs = neutral[:2]
        ref_sentences = {r["sentence"] for r in refs}
        enrollment.extend(_decorate(r, "test", "enrollment") for r in refs)
        for row in part:
            if row["sentence"] in ref_sentences:
                excluded.append({"clip_id": row["clip_id"], "speaker": person,
                                 "reasons": ["test_enrollment_sentence_reserved"],
                                 "native_metadata": copy.deepcopy(row)})
            else:
                query.append(_decorate(row, "test", "query"))
        if {int(r["emotion"]) for r in part if r["sentence"] not in ref_sentences} != set(range(8)):
            raise ValueError(f"Eight-emotion coverage lost after enrollment reservation: {person}")

    query_ids = {r["clip_id"] for r in query}
    ref_ids = {r["clip_id"] for r in enrollment}
    if query_ids & ref_ids:
        raise ValueError("Test query/enrollment clip overlap")
    if any((r["speaker"], r["sentence"]) in {(x["speaker"], x["sentence"]) for x in enrollment} for r in query):
        raise ValueError("Test query/reference sentence overlap")
    if sorted({int(r["emotion"]) for r in query}) != list(range(8)):
        raise ValueError("Final test query is not eight-emotion")

    out = copy.deepcopy(base)
    out["schema"] = SCHEMA
    out["status"] = "approved_train_val_only"
    out["track"] = "fixed_identity_disjoint_mead_eight_emotion_sealed_test"
    out["roles"]["test"] = {
        "query": sorted(query, key=lambda r: (r["speaker"], r["emotion"], r["intensity"], r["sentence"], r["clip_id"])),
        "enrollment": sorted(enrollment, key=lambda r: (r["speaker"], r["sentence"], r["clip_id"])),
        "summary": {"query": _summary(query), "enrollment": _summary(enrollment)},
    }
    out["test_policy"] = "all eight MEAD emotions from native test metadata; two neutral enrollment clips per held-out identity; enrollment sentences excluded from same-identity queries"
    out["test_is_eight_emotion_benchmark"] = True
    out["test_query_count"] = len(query)
    out["test_enrollment_count"] = len(enrollment)
    out["test_metadata_source_sha256"] = native_sha
    out["base_manifest_sha256"] = base["manifest_sha256"]
    out["base_manifest_file_sha256"] = base_sha
    out["test_role_canonical_sha256"] = canonical_hash(out["roles"]["test"])
    out["test_query_enrollment_disjoint"] = True
    out["sealed_test_targets_loaded"] = False
    out["sealed_test_rebuild"] = {
        "metadata_only": True,
        "native_arrays_read": False,
        "artifact_hashes_recomputed": False,
        "source_file": "test.jsonl",
        "heldout_identities": sorted(old_test_people),
        "enrollment_selection": "first two neutral rows by (sentence, clip_id)",
        "query_filter": "same-identity enrollment sentences removed; valid_frames>=32",
    }
    out["exclusions"] = copy.deepcopy(base.get("exclusions", [])) + excluded
    out["builder_sha256"] = sha(Path(__file__))
    out.pop("manifest_sha256", None)
    out["manifest_sha256"] = canonical_hash(out)
    validate_manifest(out)
    summary = {
        "schema": SCHEMA,
        "manifest_sha256": out["manifest_sha256"],
        "native_test_metadata_sha256": native_sha,
        "base_manifest_sha256": base["manifest_sha256"],
        "test_is_eight_emotion_benchmark": True,
        "test_metadata_file_opened": True,
        "native_arrays_read": False,
        "test": out["roles"]["test"]["summary"],
        "heldout_identities": sorted(old_test_people),
        "excluded_rows": len(excluded),
    }
    return out, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh output directory required")
    base = json.loads(args.base_manifest.read_text(encoding="utf-8-sig"))
    native_path = args.native_root / "test.jsonl"
    rows = _read_jsonl(native_path)
    manifest, summary = build(base, rows, native_sha=sha(native_path), base_sha=sha(args.base_manifest))
    args.output.mkdir(parents=True)
    save_json(args.output / "manifest.json", manifest)
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
