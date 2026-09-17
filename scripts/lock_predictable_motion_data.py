"""Lock expanded MEAD manifests using metadata only, never tensor observations.

Historical sentence exposure is global across identities and emotions. A new
test sentence must be absent from every supplied historical cache/manifest.
This does not establish absence from B0 or external encoder pretraining.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


EMOTIONS = {0: "neutral", 1: "angry", 5: "happy", 6: "sad"}
CELLS = ((0, 0), (1, 1), (1, 3), (5, 1), (5, 3), (6, 1), (6, 3))
ORIGINAL_SPEAKERS = ("mead_M003", "mead_M005", "mead_M007", "mead_M009")
MANIFEST_NAMES = {"train.jsonl", "expanded_train.jsonl", "heldout.jsonl", "validation.jsonl",
                  "enrollment.jsonl", "selected_manifest.jsonl", "new_test.jsonl", "audit.jsonl"}
CACHE_NAMES = {"train.pt", "heldout.pt", "validation.pt", "new_test.pt", "audit.pt"}


def rank(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode("utf8")).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def source_role(path):
    name = Path(path).stem.lower()
    if "enroll" in name:
        return "enrollment"
    if name in {"train", "expanded_train"}:
        return "train"
    return "validation"  # Past test/audit results are development exposure now.


def cache_metadata(path):
    """FakeTensorMode loads only tensor metadata, never tensor storage values."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        cache = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or "queries" not in cache:
        raise ValueError(f"Not a recognized historical query cache: {path}")
    # Deliberately do not access emotion/speaker tensors or any motion/audio field.
    def metadata(clip):
        return {key: clip[key] for key in ("clip_id", "sentence_id", "sentence", "speaker")
                if key in clip}
    queries = [metadata(clip) for clip in cache["queries"]]
    references = [metadata(clip) for refs in cache.get("identity_references", {}).values()
                  for clip in refs]
    return queries, references


def historical_metadata(paths):
    """Return conservative exposure sets, retaining explicit enrollment records."""
    result = {name: set() for name in ("all", "train", "validation", "enrollment", "clips")}
    result.update(references=[], sources=[])
    visited = set()

    def add(rows, role):
        for row in rows:
            sentence = str(row.get("sentence", row.get("sentence_id", "")))
            if not sentence:
                raise ValueError("Historical selected clip lacks sentence identity; cannot certify isolation")
            result["all"].add(sentence)
            result[role].add(sentence)
            if row.get("clip_id"):
                result["clips"].add(str(row["clip_id"]))
            if role == "enrollment":
                result["references"].append(row)

    for given in paths:
        given = Path(given)
        if not given.exists():
            raise FileNotFoundError(given)
        files = sorted(p for p in given.rglob("*") if p.is_file() and
                       (p.name in MANIFEST_NAMES | CACHE_NAMES or p.name == "selection.json")) if given.is_dir() else [given]
        if not files:
            raise ValueError(f"No historical selected metadata found in {given}")
        for path in files:
            path = path.resolve()
            if path in visited:
                continue
            visited.add(path)
            if path.suffix == ".pt":
                sidecar = path.with_suffix(".jsonl")
                # A sidecar also needs enrollment metadata; otherwise inspect the
                # cache metadata to capture its independent neutral references.
                if sidecar.is_file() and (path.parent / "enrollment.jsonl").is_file():
                    for source in (sidecar, path.parent / "enrollment.jsonl"):
                        add(read_jsonl(source), source_role(source))
                        if source.resolve() not in visited:
                            result["sources"].append({"path": str(source.resolve()), "sha256": sha(source), "read_mode": "jsonl"})
                            visited.add(source.resolve())
                    continue
                query, refs = cache_metadata(path)
                add(query, source_role(path))
                add(refs, "enrollment")
                mode = "FakeTensorMode_metadata_only"
            elif path.suffix == ".jsonl":
                add(read_jsonl(path), source_role(path))
                mode = "jsonl"
            elif path.name == "selection.json":
                summary = json.loads(path.read_text(encoding="utf-8-sig"))
                # Audit selection summaries may be the only remaining evidence
                # of previously selected (or globally excluded) sentences.
                result["all"].update(map(str, summary.get("excluded_sentence_ids", [])))
                selected = set(map(str, summary.get("selected_sentence_ids", [])))
                result["all"].update(selected)
                result["validation"].update(selected)
                mode = "selection_summary"
            else:
                raise ValueError(f"Unsupported historical metadata file: {path}")
            result["sources"].append({"path": str(path), "sha256": sha(path), "read_mode": mode})
    if not result["all"]:
        raise ValueError("No historical sentences found; refuse to label candidates new")
    return result


def unique_native(rows, min_frames):
    found = {}
    for row in rows:
        if row.get("dataset") != "mead" or int(row.get("valid_frames", 0)) < min_frames:
            continue
        if (int(row["emotion"]), int(row["intensity"])) not in CELLS:
            continue
        if not row.get("sentence") or not row.get("clip_id") or not row.get("speaker"):
            raise ValueError("Usable native row has missing identity metadata")
        key = str(row["clip_id"])
        if key in found and any(found[key].get(k) != row.get(k) for k in
                                ("sentence", "speaker", "emotion", "intensity", "artifact")):
            raise ValueError(f"Conflicting native metadata: {key}")
        found[key] = row
    return list(found.values())


def cell(row):
    return str(row["speaker"]), int(row["emotion"]), int(row["intensity"])


def balanced_limit(rows, limit, seed):
    """Round robin across cells, retaining deterministic metadata-only order."""
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: (rank(r["clip_id"], seed), r["clip_id"])):
        groups[cell(row)].append(row)
    result = []
    index = 0
    while len(result) < limit:
        added = False
        for key in sorted(groups):
            if index < len(groups[key]) and len(result) < limit:
                result.append(groups[key][index])
                added = True
        if not added:
            break
        index += 1
    return result


def lock_selection(rows, history, *, speakers=12, max_train=1200, seed=20260919,
                   min_frames=32, references=4, test_fraction=0.15, validation_fraction=0.15,
                   max_validation=280, max_test=280, required_speakers=ORIGINAL_SPEAKERS):
    if speakers < len(required_speakers) or references < 4 or min_frames < 32:
        raise ValueError("Require original identities, >=4 neutral references and >=32 valid frames")
    if not 0 < test_fraction < 0.5 or not 0 < validation_fraction < 0.5:
        raise ValueError("Split fractions must lie in (0, .5)")
    if min(max_train, max_validation, max_test) < 1:
        raise ValueError("Clip budgets must be positive")
    rows = unique_native(rows, min_frames)
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker"])].append(row)
    old_query = history["train"] | history["validation"]
    eligible = []
    for speaker, values in by_speaker.items():
        neutral = {str(r["sentence"]) for r in values if int(r["emotion"]) == 0
                   and (r["sentence"] in history["enrollment"] or r["sentence"] not in old_query)}
        if len(neutral) >= references:
            eligible.append(speaker)
    missing_required = sorted(set(required_speakers) - set(eligible))
    if missing_required:
        raise ValueError(f"Original identities lack isolated neutral enrollment: {missing_required}")
    extra = sorted(set(eligible) - set(required_speakers), key=lambda s: (-len({cell(r) for r in by_speaker[s]}),
                            -len(by_speaker[s]), rank(s, seed), s))
    chosen = list(required_speakers) + extra[:speakers - len(required_speakers)]
    speaker_to_id = {name: index for index, name in enumerate(chosen)}
    rows = [r for r in rows if r["speaker"] in speaker_to_id]
    old_ref_ids = {str(r["clip_id"]) for r in history["references"] if r.get("clip_id")}
    enrollment = []
    enrollment_sentences = set(history["enrollment"])
    for speaker in chosen:
        neutral = [r for r in by_speaker[speaker] if int(r["emotion"]) == 0 and
                   (r["sentence"] in history["enrollment"] or r["sentence"] not in old_query)]
        # Preserve existing references where possible; for new people reuse the
        # same enrollment sentences before reserving additional global sentences.
        neutral.sort(key=lambda r: (r["clip_id"] not in old_ref_ids,
                     r["sentence"] not in enrollment_sentences, rank(r["sentence"], seed), rank(r["clip_id"], seed)))
        selected, seen = [], set()
        for row in neutral:
            if row["sentence"] in seen:
                continue
            selected.append(row)
            seen.add(row["sentence"])
            if len(selected) == references:
                break
        enrollment.extend(selected)
        enrollment_sentences.update(seen)
    candidates = [r for r in rows if r["sentence"] not in enrollment_sentences]
    fresh = {r["sentence"] for r in candidates if r["sentence"] not in history["all"]
             and r["clip_id"] not in history["clips"]}
    # Globally lock a fixed fraction of eligible fresh sentences. No requirement
    # for a complete cell grid may cause reuse of previously seen sentences.
    ordered_fresh = sorted(fresh, key=lambda s: (rank(s, seed + 1), s))
    test_count = max(1, round(len(ordered_fresh) * test_fraction)) if fresh else 0
    test_sentences = set(ordered_fresh[:test_count])
    dev_sentences = history["validation"] - enrollment_sentences
    remaining = sorted({r["sentence"] for r in candidates} - test_sentences - dev_sentences,
                       key=lambda s: (rank(s, seed + 2), s))
    # Preserve old training allocation; reserve validation from previously unused
    # sentences only. Existing dev sentences remain dev for every identity.
    eligible_dev = [s for s in remaining if s not in history["all"]]
    dev_count = round(len(remaining) * validation_fraction)
    dev_sentences.update(eligible_dev[:max(0, dev_count - len(dev_sentences))])
    split_rows = {
        "new_test": [r for r in candidates if r["sentence"] in test_sentences],
        "validation": [r for r in candidates if r["sentence"] in dev_sentences],
        "train": [r for r in candidates if r["sentence"] not in test_sentences | dev_sentences],
        "enrollment": enrollment,
    }
    budgets = {"train": max_train, "validation": max_validation, "new_test": max_test}
    available = {}
    expected = {(s, e, level) for s in chosen for e, level in CELLS}
    for split, values in split_rows.items():
        available[split] = len(values)
        if split in budgets:
            split_rows[split] = balanced_limit(values, budgets[split], seed + 3)
        split_rows[split] = [dict(r, speaker_id=speaker_to_id[r["speaker"]],
                                emotion_id=int(r["emotion"]), intensity_id=int(r["intensity"]),
                                pilot_split=split) for r in split_rows[split]]
    sentence_sets = {split: {r["sentence"] for r in values} for split, values in split_rows.items()}
    checks = {f"{a}_{b}_sentence_disjoint": not bool(sentence_sets[a] & sentence_sets[b])
              for a in split_rows for b in split_rows if a < b}
    checks.update(new_test_excludes_all_historical_sentences=not bool(sentence_sets["new_test"] & history["all"]),
                  new_test_excludes_all_historical_clips=not bool({r["clip_id"] for r in split_rows["new_test"]} & history["clips"]),
                  old_development_never_training=not bool(sentence_sets["train"] & history["validation"]),
                  historical_enrollment_never_query=not bool(set.union(*(sentence_sets[s] for s in budgets)) & history["enrollment"]))
    if not all(checks.values()):
        raise AssertionError(f"Isolation failure: {checks}")
    coverage = {}
    for split in budgets:
        counts = Counter(cell(row) for row in split_rows[split])
        coverage[split] = {"clips": len(split_rows[split]), "available_before_clip_budget": available[split],
                          "sentences": len(sentence_sets[split]), "speakers": len({r["speaker"] for r in split_rows[split]}),
                          "populated_cells": len(counts), "expected_cells": len(expected),
                          "missing_cells": [{"speaker": s, "emotion": EMOTIONS[e], "intensity": level}
                                            for s, e, level in sorted(expected - set(counts))],
                          "emotion_counts": dict(Counter(EMOTIONS[int(r["emotion"])] for r in split_rows[split]))}
    summary = {"schema": "predictable_motion_metadata_lock_v1", "seed": seed,
               "status": "locked_metadata_only" if split_rows["new_test"] else "no_unexposed_test_sentences",
               "speaker_to_id": speaker_to_id, "requested_speakers": speakers,
               "eligible_speakers": len(eligible), "selected_speakers": len(chosen),
               "speaker_shortage": max(0, speakers - len(chosen)), "min_valid_frames": min_frames,
               "references_per_speaker": references, "max_train": max_train,
               "test_fraction": test_fraction, "validation_fraction": validation_fraction,
               "fresh_sentence_candidates": len(fresh), "coverage": coverage, "isolation_checks": checks,
               "sentence_allocation": {"enrollment": sorted(enrollment_sentences), "validation": sorted(dev_sentences),
                                       "new_test": sorted(test_sentences)},
               "historical_sentence_ids": sorted(history["all"]),
               "scope": "New sentences relative only to supplied pilot/development/audit history. B0 and pretrained encoders may have seen native clips; not a claim of an untouched whole-system test. No observations or predictions were inspected. Missing cell coverage is reported without relaxing exclusions.",
               "selection_rule": "Original four identities retained; extras ranked by metadata cell coverage and clip count. Four neutral references per identity reserve sentence IDs globally. Historical dev/audit sentences remain development. A deterministic fraction of unexposed sentences is locked as new_test, with no seen-sentence fallback. Balanced per-cell clip budget applied after sentence allocation."}
    return split_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--source", nargs="+", default=["train.jsonl"], help="Native manifests; defaults to original training identities only")
    parser.add_argument("--previous-data", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=1200)
    parser.add_argument("--speakers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--max-validation", type=int, default=280)
    parser.add_argument("--max-test", type=int, default=280)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; locked selections are never overwritten")
    native_paths = [args.native_root / name for name in args.source]
    rows = [row for path in native_paths for row in read_jsonl(path)]
    history = historical_metadata(args.previous_data)
    selections, summary = lock_selection(rows, history, speakers=args.speakers, max_train=args.max_train,
        seed=args.seed, max_validation=args.max_validation, max_test=args.max_test,
        test_fraction=args.test_fraction, validation_fraction=args.validation_fraction)
    summary.update(created_utc=datetime.now(timezone.utc).isoformat(), historical_sources=history["sources"],
                   native_sources=[{"path": str(path.resolve()), "sha256": sha(path)} for path in native_paths],
                   preparation_script_sha256=sha(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    for split, selected in selections.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf8")
    summary["manifest_sha256"] = {split: sha(args.output / f"{split}.jsonl") for split in selections}
    (args.output / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"status": summary["status"], "speakers": summary["selected_speakers"],
                      "fresh_sentences": summary["fresh_sentence_candidates"],
                      "clips": {s: len(rows) for s, rows in selections.items()},
                      "missing_cells": {s: len(c["missing_cells"]) for s, c in summary["coverage"].items()},
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
