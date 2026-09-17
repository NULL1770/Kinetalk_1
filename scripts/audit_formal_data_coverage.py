"""Metadata-only formal-data inventory; never loads arrays or chooses new tests.

MEAD sentence hashes and CREMA-D sentence codes are distinct metadata schemes.
Cross-dataset lexical novelty cannot be inferred from unequal identifiers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


CELLS = {(0, 0), (1, 1), (1, 3), (5, 1), (5, 3), (6, 1), (6, 3)}
EMOTIONS = {0: "neutral", 1: "angry", 2: "contempt", 3: "disgust",
            4: "fear", 5: "happy", 6: "sad", 7: "surprise"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentence(row):
    return str(row["dataset"]), str(row["sentence"])


def summary(rows):
    result = {"clips": len(rows), "sentences": len({sentence(r) for r in rows}),
              "speakers": sorted({r["speaker"] for r in rows}),
              "stage1_eligible_clips": sum(bool(r.get("stage1")) for r in rows),
              "emotion_counts": dict(sorted(Counter(EMOTIONS[int(r["emotion"])] for r in rows).items())),
              "emotion_intensity_counts": dict(sorted(Counter(f'{EMOTIONS[int(r["emotion"])]}/L{r["intensity"]}'
                                                               for r in rows).items()))}
    result["speaker_count"] = len(result["speakers"])
    result["by_emotion"] = {
        EMOTIONS[e]: {"clips": len(values), "sentences": len({sentence(r) for r in values}),
                       "speakers": len({r["speaker"] for r in values})}
        for e in sorted({int(r["emotion"]) for r in rows})
        if (values := [r for r in rows if int(r["emotion"]) == e])
    }
    return result


def overlap(a, b):
    return {"clips": len({r["clip_id"] for r in a} & {r["clip_id"] for r in b}),
            "speakers": len({r["speaker"] for r in a} & {r["speaker"] for r in b}),
            "sentences": len({sentence(r) for r in a} & {sentence(r) for r in b})}


def audit(native_root, locked_root, previous_root, stage1_diagnostic):
    sources = {}
    def rows(path):
        sources[str(path)] = sha(path)
        return read_jsonl(path)
    def obj(path):
        sources[str(path)] = sha(path)
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    native = {name: rows(native_root / f"{name}.jsonl") for name in ("train", "val", "test")}
    seen = {}
    for split, values in native.items():
        for row in values:
            for key in ("clip_id", "dataset", "speaker", "sentence", "emotion", "intensity", "artifact"):
                if key not in row or row[key] == "":
                    raise ValueError(f"Missing {key} in {split} metadata")
            if row["split"] != split:
                raise ValueError(f"Source split mismatch: {row['clip_id']}")
            if row["clip_id"] in seen:
                raise ValueError(f"Duplicate clip across native manifests: {row['clip_id']}")
            seen[row["clip_id"]] = row
    lock = obj(locked_root / "selection.json")
    selected = {name: rows(locked_root / f"{name}.jsonl")
                for name in ("train", "validation", "enrollment", "new_test")}
    old_audit = obj(previous_root / "audit_available26" / "selection.json")
    original = obj(previous_root / "selection.json")
    expanded = obj(previous_root / "selection224.json")
    run01 = obj(previous_root / "run01" / "provenance.json")
    run09 = obj(previous_root / "run09_emotion2vec_probe" / "provenance.json")
    historical_sentences = set(lock["historical_sentence_ids"])
    used_rows = selected["train"] + selected["validation"] + selected["enrollment"]
    used_sentences = historical_sentences | {r["sentence"] for r in used_rows}
    reserved_sentences = {r["sentence"] for r in selected["new_test"]}
    blocked_mead = used_sentences | reserved_sentences
    historical_clips = set(old_audit.get("excluded_clip_ids", [])) | set(old_audit.get("selected_clip_ids", []))
    known_used_clips = historical_clips | {r["clip_id"] for r in used_rows}
    used_speakers = {r["speaker"] for r in used_rows}
    reserved_clips = {r["clip_id"] for r in selected["new_test"]}
    reference_sentences = {r["sentence"] for r in selected["enrollment"]}
    stage1 = obj(stage1_diagnostic)
    stage1_read_clips = set(stage1.get("selected_clip_ids", []))

    coverage = {}
    fresh_rows = []
    for split, values in native.items():
        part = {}
        for dataset in sorted({r["dataset"] for r in values}):
            all_rows = [r for r in values if r["dataset"] == dataset]
            valid = [r for r in all_rows if int(r["valid_frames"]) >= 32]
            current = [r for r in valid if (int(r["emotion"]), int(r["intensity"])) in CELLS]
            four_any_level = [r for r in valid if int(r["emotion"]) in {0, 1, 5, 6}]
            unseen_clip = [r for r in current if r["clip_id"] not in known_used_clips | reserved_clips]
            fresh = [r for r in current if dataset != "mead" or r["sentence"] not in blocked_mead]
            fresh_rows.extend(fresh)
            same_script_new_clip = [r for r in unseen_clip if dataset == "mead" and r["sentence"] in used_sentences]
            part[dataset] = {"all": summary(all_rows), "valid_min32": summary(valid),
                "four_emotions_any_intensity": summary(four_any_level), "current_four_emotions_levels": summary(current),
                "current_cells_known_pilot_new_recording": summary(unseen_clip),
                "current_cells_new_recording_known_sentence": summary(same_script_new_clip),
                "current_cells_new_speaker": summary([r for r in unseen_clip if r["speaker"] not in used_speakers]),
                "current_cells_excluding_original_reference_and_reserved_sentences": summary(
                    [r for r in current if dataset != "mead" or r["sentence"] not in reference_sentences | reserved_sentences]),
                "current_cells_sentence_not_in_mead_history_or_reserved": summary(fresh),
                "fresh_sentence_ids": sorted({r["sentence"] for r in fresh}),
                "fresh_sentence_by_emotion": {
                    EMOTIONS[e]: sorted({r["sentence"] for r in fresh if int(r["emotion"]) == e})
                    for e in (0, 1, 5, 6)},
                "current_cells_stage1_diagnostic_seen_clips": len({r["clip_id"] for r in current} & stage1_read_clips),
                "neutral_distinct_sentences_per_speaker": {
                    speaker: len({r["sentence"] for r in valid if r["speaker"] == speaker and int(r["emotion"]) == 0})
                    for speaker in sorted({r["speaker"] for r in current})},
                "original_enrollment_sentence_coverage_per_speaker": {
                    speaker: len({r["sentence"] for r in valid if r["speaker"] == speaker
                                  and int(r["emotion"]) == 0 and r["sentence"] in reference_sentences})
                    for speaker in sorted({r["speaker"] for r in current})} if dataset == "mead" else {}}
            if dataset != "mead":
                part[dataset]["novelty_limit"] = (
                    "No CREMA-D in the audited dynamic pilot exposure; sentence codes are not comparable to MEAD text hashes. "
                    "This is a potential separate-domain evaluation, not certified new lexical sentences or unseen B0 data. "
                    "Current CELLS omits intensity -1 and level2; any-level counts are supplied without remapping labels.")
        coverage[split] = part
    fresh_mead = [r for r in fresh_rows if r["dataset"] == "mead"]
    fresh_by_cell = []
    for sid, emotion, intensity in sorted({(r["sentence"], int(r["emotion"]), int(r["intensity"])) for r in fresh_mead}):
        matches = [r for r in fresh_mead if (r["sentence"], int(r["emotion"]), int(r["intensity"])) == (sid, emotion, intensity)]
        fresh_by_cell.append({"sentence": sid, "emotion": EMOTIONS[emotion], "intensity": intensity,
                              "clips": len(matches), "speakers": sorted({r["speaker"] for r in matches}),
                              "source_splits": sorted({r["split"] for r in matches})})
    pairs = {f"{a}_{b}": {dataset: overlap([r for r in native[a] if r["dataset"] == dataset],
                                                [r for r in native[b] if r["dataset"] == dataset])
                          for dataset in ("mead", "crema_d")}
             for a, b in itertools.combinations(native, 2)}
    result = {"schema": "formal_metadata_coverage_v1", "metadata_only": True, "source_sha256": sources,
        "min_valid_frames": 32, "current_cells": sorted(CELLS), "coverage": coverage,
        "native_split_overlap": pairs, "exposure": {
            "historical_pre_run31_sentences": len(historical_sentences),
            "known_used_sentences_through_run33": len(used_sentences),
            "reserved_new_test_sentences": len(reserved_sentences), "reserved_new_test_clips": len(reserved_clips),
            "known_used_clip_count": len(known_used_clips), "pilot_speakers": sorted(used_speakers),
            "historical_exclusions": old_audit.get("excluded_sentence_ids", []),
            "blocked_mead_sentence_ids": sorted(blocked_mead)},
        "fresh_mead_current_cells": summary(fresh_mead), "fresh_mead_sentence_cells": fresh_by_cell,
        "component_exposure": {
            "audio_global_run09": {"fit_queries": expanded["expansion"]["expanded_query_count"],
                "speaker_to_id": original["speaker_to_id"], "teacher_checkpoint": run09["teacher_checkpoint"],
                "teacher_sha256": run09["teacher_sha256"], "train_cache_sha256": run09["train_sha256"],
                "scope": "Audio/global fit on data224; development and audit viewed subsequently. Full input tensor caches were not loaded by this audit."},
            "teacher_identity_renderer_run01": {"fit_queries": original["splits"]["train"]["clips"],
                "reference_clips": sum(map(len, original["references"].values())),
                "speaker_to_id": original["speaker_to_id"],
                "scope": "Teacher/renderer originally fit on56 queries and identity references; run31–33 adapter fits use1200 of12 identities."},
            "B0": {"checkpoint_path": run01["stage1_path"], "checkpoint_sha256": run01["stage1_sha256"],
                "stage1_training_manifest_certified": False,
                "available_train_diagnostic_clip_count": len(stage1_read_clips),
                "stage1_diagnostic_checkpoint_identity_certified": False,
                "note": "stage1=true is eligibility, not proof of fitting. Existing fidelity report says train_set_diagnostic_not_held_out but lacks an exact checkpoint hash; its clip overlap is only exposure-risk evidence. Need original checkpoint training manifest/config hash before whole-system unseen claims."}},
        "decision": {"balanced_fresh_mead_four_emotions_available": {0, 1, 5, 6}.issubset({int(r["emotion"]) for r in fresh_mead}),
            "does_not_lock_or_materialize_test": True,
            "scope": "Metadata novelty relative to explicit pilot history only; no feature, audio, video, motion, or prediction read. Native identity partitions can supply independent recordings with shared sentence scripts. No cross-dataset normalization or intensity remapping."}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=Path("artifacts/formal_readiness/native_metadata"))
    parser.add_argument("--locked-root", type=Path, default=Path("artifacts/predictable_motion_run31/locked_manifests"))
    parser.add_argument("--previous-root", type=Path, default=Path("artifacts/neutral_affect_pilot_20260916"))
    parser.add_argument("--stage1-diagnostic", type=Path, default=Path("artifacts/stage1_fidelity.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/formal_readiness/metadata_coverage.json"))
    args = parser.parse_args()
    result = audit(args.native_root, args.locked_root, args.previous_root, args.stage1_diagnostic)
    result["audit_script_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fresh_mead": result["fresh_mead_current_cells"], "decision": result["decision"],
                      "output": str(args.output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
