"""Lock new sentence-disjoint audit clips without inspecting predictions.

Selection uses native-manifest metadata only. Sentences previously used by any
enrolled person's training, development, or enrollment clips are excluded
globally. If any required cell is exhausted, fail without relaxing isolation.
An explicit incomplete-cells option can keep populated cells while recording
missing cells; it never relaxes sentence exclusion or substitutes old clips.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import EMOTIONS, SCHEMA, NeutralAffectDataset, native_clip


SEED = 20260917
MIN_VALID_FRAMES = 32
EMOTION_LEVELS = ((0, 0), (1, 1), (1, 3), (5, 1), (5, 3), (6, 1), (6, 3))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(value):
    return hashlib.sha256(f"{SEED}:{value}".encode("utf8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf8")


def exclusions(train, development):
    """Return global exclusions, including every person's neutral references."""
    reference_sources = [ref for data in (train, development)
                         for references in data.identity_references.values() for ref in references]
    clips = list(train.queries) + list(development.queries) + reference_sources
    return {str(clip["sentence_id"]) for clip in clips}, {str(clip["clip_id"]) for clip in clips}


def select_audit(rows, speaker_to_id, excluded_sentences, excluded_clips, per_cell=1):
    """Metadata-only deterministic selection; return availability even on failure."""
    if type(per_cell) is not int or per_cell < 1:
        raise ValueError("per_cell must be a positive integer")
    expected = [(speaker, emotion, level) for speaker in sorted(speaker_to_id)
                for emotion, level in EMOTION_LEVELS]
    before, eligible = defaultdict(list), defaultdict(list)
    expected_set = set(expected)
    seen_clips = {}
    for row in rows:
        if row.get("dataset") != "mead" or row.get("speaker") not in speaker_to_id:
            continue
        cell = (str(row["speaker"]), int(row["emotion"]), int(row["intensity"]))
        if cell not in expected_set or int(row["valid_frames"]) < MIN_VALID_FRAMES:
            continue
        if not str(row.get("sentence", "")):
            raise ValueError(f"Missing sentence identity: {row.get('clip_id')}")
        clip_id = str(row["clip_id"])
        previous = seen_clips.get(clip_id)
        signature = (cell, str(row["sentence"]), str(row["artifact"]))
        if previous is not None:
            if previous != signature:
                raise ValueError(f"Conflicting manifest records for {clip_id}")
            continue
        seen_clips[clip_id] = signature
        before[cell].append(row)
        if str(row["sentence"]) in excluded_sentences or clip_id in excluded_clips:
            continue
        eligible[cell].append(row)
    available, selected, shortages = [], [], []
    eligible_manifest = []
    for cell in expected:
        grouped = defaultdict(list)
        for row in eligible[cell]:
            grouped[str(row["sentence"])].append(row)
        sentences = sorted(grouped, key=lambda sentence: (rank(sentence), sentence))
        speaker, emotion, intensity = cell
        cell_name = f"{speaker}/{EMOTIONS[emotion]}/level{intensity}"
        available.append({"cell": cell_name, "speaker": speaker, "emotion_id": emotion,
                          "intensity_id": intensity, "valid_source_clips": len(before[cell]),
                          "available_clips": len(eligible[cell]), "available_distinct_sentences": len(sentences),
                          "available_sentence_ids": sentences, "requested": per_cell})
        if len(sentences) < per_cell:
            shortages.append({"cell": cell_name, "requested": per_cell, "available_distinct_sentences": len(sentences)})
        for sentence in sentences:
            for row in sorted(grouped[sentence], key=lambda item: (rank(str(item["clip_id"])), str(item["clip_id"]))):
                eligible_manifest.append(dict(row, audit_cell=cell_name))
        for sentence in sentences[:per_cell]:
            row = min(grouped[sentence], key=lambda item: (rank(str(item["clip_id"])), str(item["clip_id"])))
            selected.append(dict(row, speaker_id=speaker_to_id[speaker], emotion_id=emotion,
                                 intensity_id=intensity, pilot_split="locked_sentence_audit"))
    return selected, available, shortages, eligible_manifest


def enrollment_identity(train, development):
    refs = train.identity_references
    if len(refs) != 4 or sorted(refs) != list(range(4)):
        raise ValueError("Audit requires the four existing contiguous enrolled speaker IDs")
    if set(development.identity_references) != set(refs):
        raise ValueError("Development and training enrolled speakers differ")
    speaker_to_id = {}
    for speaker_id, references in refs.items():
        names = {str(clip["speaker"]) for clip in references}
        if len(names) != 1:
            raise ValueError("Enrollment speaker name is ambiguous")
        speaker = next(iter(names))
        if not speaker.startswith("mead_") or speaker in speaker_to_id:
            raise ValueError("Audit requires four distinct MEAD enrollment identities")
        speaker_to_id[speaker] = speaker_id
        if [clip["clip_id"] for clip in references] != [clip["clip_id"] for clip in development.identity_references[speaker_id]]:
            raise ValueError("Development enrollment differs from training enrollment")
        for first, second in zip(references, development.identity_references[speaker_id]):
            for key in ("motion", "content", "audio", "valid", "channel_mask"):
                if not torch.equal(first[key], second[key]):
                    raise ValueError(f"Development reference {key} differs from training enrollment")
    return speaker_to_id


def incomplete_coverage(selected, available, per_cell):
    """Retain full eligible cells only and check a predeclared minimum scope."""
    complete = {(cell["speaker"], cell["emotion_id"], cell["intensity_id"])
                for cell in available if cell["available_distinct_sentences"] >= per_cell}
    selected = [row for row in selected if (row["speaker"], row["emotion_id"], row["intensity_id"]) in complete]
    speakers = {row["speaker"] for row in selected}
    emotions = {int(row["emotion_id"]) for row in selected}
    checks = {"at_least_24_clips": len(selected) >= 24, "all_four_enrolled_speakers": len(speakers) == 4,
              "all_four_emotions": emotions == {0, 1, 5, 6}}
    return selected, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True, help="Existing data224 with train.pt and development heldout.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--allow-incomplete-cells", action="store_true",
                        help="Explicitly retain populated cells, recording missing cells; requires >=24 clips, all four speakers and all four emotions")
    args = parser.parse_args()
    if args.per_cell < 1:
        raise ValueError("--per-cell must be positive")
    source = args.source_data.resolve()
    output = args.output.resolve()
    if output == source or output == args.native_root.resolve():
        raise ValueError("Audit output must be a separate fresh directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh audit output directory; existing selections must not be overwritten")
    train_path, old_heldout_path = source / "train.pt", source / "heldout.pt"
    train, development = NeutralAffectDataset(train_path), NeutralAffectDataset(old_heldout_path)
    speakers = enrollment_identity(train, development)
    audio_stats = train.cache["audio_stats"]
    for key in ("mean", "std"):
        value = audio_stats[key]
        if value.shape != (83,) or not torch.isfinite(value).all():
            raise ValueError("Source audio normalization must contain finite 83D mean/std")
        if not torch.equal(value, development.cache["audio_stats"][key]):
            raise ValueError("Source development audio normalization differs from training")
    if not (audio_stats["std"] > 0).all():
        raise ValueError("Source audio standard deviations must be positive")
    window = int(train.cache["window"])
    if window != int(development.cache["window"]) or window < MIN_VALID_FRAMES:
        raise ValueError("Source train/development crop windows differ or are too short")
    native_manifest = args.native_root / "train.jsonl"
    rows = [json.loads(line) for line in native_manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    excluded_sentences, excluded_clips = exclusions(train, development)
    selected, available, shortages, eligible = select_audit(rows, speakers, excluded_sentences, excluded_clips, args.per_cell)
    incomplete_checks = None
    if shortages and args.allow_incomplete_cells:
        selected, incomplete_checks = incomplete_coverage(selected, available, args.per_cell)
    coverage_failure = bool(shortages) and (not args.allow_incomplete_cells or not all(incomplete_checks.values()))
    populated_cells = len({(row["speaker"], row["emotion_id"], row["intensity_id"]) for row in selected})
    missing_cell_names = [item["cell"] for item in shortages]
    scope = "New locked audit sentences of the same four enrolled people, not unseen identity. Earlier B0 pretraining may have included these clips. Use for a predeclared comparison; after inspection this set must not be called untouched in later optimization."
    if shortages and args.allow_incomplete_cells:
        scope += f" Explicit incomplete-cell audit: {populated_cells}/{len(available)} populated cells; missing cells: {', '.join(missing_cell_names)}. This is not a complete balanced audit."
    summary = {"schema": "neutral_affect_locked_sentence_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "insufficient_metadata_candidates" if coverage_failure else "selection_locked_before_materialization",
               "selection_seed": SEED, "per_cell": args.per_cell, "window": window, "min_valid_frames": MIN_VALID_FRAMES,
               "speaker_to_id": speakers, "available": available, "shortages": shortages, "missing_cells": shortages,
               "allow_incomplete_cells": args.allow_incomplete_cells, "incomplete_minimum_checks": incomplete_checks,
               "coverage": {"populated_cells": populated_cells, "expected_cells": len(available),
                            "selected_clips": len(selected), "complete_balanced_cell_coverage": not bool(shortages)},
               "excluded_sentence_ids": sorted(excluded_sentences), "excluded_clip_ids": sorted(excluded_clips),
               "selected_clip_ids": [row["clip_id"] for row in selected],
               "selected_sentence_ids": sorted({row["sentence"] for row in selected}),
               "source_sha256": {"train_cache": sha(train_path), "development_cache": sha(old_heldout_path),
                                 "native_manifest": sha(native_manifest), "preparation_script": sha(__file__),
                                 "native_reader": sha(Path(__file__).resolve().parents[1] / "kinetalk_b0/neutral_data.py")},
               "selection_rule": "Exclude globally all train/development/enrollment sentence IDs; select distinct sentence hashes using seed20260917 per person/emotion/level cell. Duplicate clips never fill a sentence shortage. No predictions inspected; no automatic fallback after materialization failure.",
               "eval_scope": scope,
               "dependencies": "Copied original train.pt for evaluator compatibility only; no audit fitting and no audio-statistics recomputation.",
               "isolation_checks": {"selected_sentence_disjoint_from_all_development_train_enrollment": not bool({row["sentence"] for row in selected} & excluded_sentences),
                                    "selected_clip_disjoint_from_all_development_train_enrollment": not bool({row["clip_id"] for row in selected} & excluded_clips)}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selection.json", summary)
    write_jsonl(output / "available.jsonl", eligible)
    write_jsonl(output / "selected_manifest.jsonl", selected)
    if coverage_failure:
        raise ValueError("No complete locked audit set: " + "; ".join(
            f"{item['cell']} needs {item['requested']} distinct sentences, has {item['available_distinct_sentences']}"
            for item in shortages) + (f"; incomplete minimum checks={incomplete_checks}" if incomplete_checks is not None else "")
            + ". See selection.json; no exclusions were relaxed and no caches were written.")
    if not all(summary["isolation_checks"].values()):
        raise AssertionError("Audit isolation failed")
    # Read only the metadata-selected artifacts; an invalid one fails rather
    # than silently replacing a precommitted evaluation sample.
    try:
        queries = [native_clip(row, args.native_root, window=window, min_valid_frames=MIN_VALID_FRAMES) for row in selected]
        for clip in queries:
            clip["audio"] = (clip["audio"] - audio_stats["mean"]) / audio_stats["std"]
            clip["audio"][~clip["valid"]] = 0
        cache = {"schema": SCHEMA, "split": "heldout", "audit_role": "locked_sentence_audit", "queries": queries,
                 "identity_references": train.identity_references, "speaker_to_id": train.cache.get("speaker_to_id", speakers),
                 "audio_stats": audio_stats, "window": window, "seed": SEED,
                 "selection_source_sha256": summary["source_sha256"], "eval_scope": summary["eval_scope"],
                 "audit_coverage": summary["coverage"], "missing_cells": shortages}
        torch.save(cache, output / "heldout.pt")
        NeutralAffectDataset(output / "heldout.pt")
        shutil.copyfile(train_path, output / "train.pt")
        if sha(output / "train.pt") != summary["source_sha256"]["train_cache"]:
            raise RuntimeError("Copied training cache hash differs from original")
        write_jsonl(output / "heldout.jsonl", selected)
        summary.update(status="locked_incomplete_cell_audit" if shortages else "complete_locked_audit", query_count=len(queries),
                       valid_frames=sum(int(clip["valid"].sum()) for clip in queries),
                       output_sha256={"train_cache": sha(output / "train.pt"), "audit_cache": sha(output / "heldout.pt"),
                                      "selected_manifest": sha(output / "selected_manifest.jsonl"), "available_manifest": sha(output / "available.jsonl")})
        summary["isolation_checks"]["copied_train_hash_unchanged"] = True
        summary["isolation_checks"]["source_audio_statistics_reused"] = True
        write_json(output / "selection.json", summary)
    except Exception as error:
        summary.update(status="materialization_failed_no_reselection", error=f"{type(error).__name__}: {error}")
        write_json(output / "selection.json", summary)
        raise
    print(json.dumps({"status": summary["status"], "queries": len(queries), "output": str(output),
                      "audit_sha256": summary["output_sha256"]["audit_cache"]}), flush=True)


if __name__ == "__main__":
    main()
