"""Read-only native TRAIN metadata/current-cache coverage audit.

Never loads a native NPZ, waveform, video, or motion/content/audio tensor.
The current cache is memory mapped; only TRAIN validity and metadata are read.
Explicit excluded JSONL inputs are metadata only; no directory is traversed.
This script writes one report, never selects data or changes a split lock.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


EMOTIONS = {0: "neutral", 1: "angry", 2: "contempt", 3: "disgust", 4: "fear",
            5: "happy", 6: "sad", 7: "surprise"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path):
    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Only explicit JSONL metadata inputs accepted")
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def dataset(row):
    return str(row["dataset"])


def speaker(row):
    return dataset(row), str(row["speaker"])


def sentence(row):
    return dataset(row), str(row["sentence"])


def summarize(rows, fps):
    frames = sum(int(r.get("frames", 0)) for r in rows)
    valid_frames = sum(int(r.get("valid_frames", 0)) for r in rows)
    return {"clips": len(rows), "speakers": len({speaker(r) for r in rows}),
        "sentences": len({sentence(r) for r in rows}),
        "frames": frames, "valid_frames": valid_frames,
        "nominal_hours_from_metadata_frames": frames / fps / 3600,
        "valid_observation_hours": valid_frames / fps / 3600,
        "emotion_counts": {EMOTIONS[e]: sum(int(r["emotion"]) == e for r in rows) for e in EMOTIONS},
        "emotion_intensity_counts": dict(sorted(Counter(
            f"{EMOTIONS[int(r['emotion'])]}/L{int(r['intensity'])}" for r in rows).items())),
        "by_emotion": {EMOTIONS[e]: {
            "clips": len(part), "speakers": len({speaker(r) for r in part}),
            "sentences": len({sentence(r) for r in part}),
            "valid_frames": sum(int(r["valid_frames"]) for r in part),
            "intensity_counts": dict(sorted(Counter(str(int(r["intensity"])) for r in part).items()))}
            for e in EMOTIONS if (part := [r for r in rows if int(r["emotion"]) == e])}}


def coverage(rows, fit_ids, development_ids, observed_frames, crop_frames, *, fps=25., min_valid=32,
             excluded_rows=(), known_development_rows=()):
    """Pure metadata/count calculation; receives no motion observations."""
    by_id = {}
    for row in rows:
        for key in ("clip_id", "dataset", "speaker", "sentence", "emotion", "intensity", "split", "frames", "valid_frames"):
            if key not in row: raise ValueError("Missing metadata field: " + key)
        if row["split"] != "train": raise ValueError("Native input must contain TRAIN metadata only")
        cid = str(row["clip_id"])
        if cid in by_id: raise ValueError("Duplicate native clip: " + cid)
        if int(row["emotion"]) not in EMOTIONS: raise ValueError("Unknown native emotion ID")
        if not 0 <= int(row["valid_frames"]) <= int(row["frames"]): raise ValueError("Invalid metadata frame counts")
        by_id[cid] = row
    if len(set(fit_ids)) != len(fit_ids) or set(fit_ids) - by_id.keys():
        raise ValueError("Fit clip IDs must be unique and present in native train metadata")
    if set(fit_ids) & set(development_ids): raise ValueError("Current train/development clips overlap")
    fit = [by_id[cid] for cid in fit_ids]
    if len(observed_frames) != len(fit): raise ValueError("Fit frame counts must match fit order")
    for row, count in zip(fit, observed_frames):
        if not 0 <= count <= min(crop_frames, int(row["valid_frames"])):
            raise ValueError("Current crop mask count inconsistent with metadata")
    dev = [by_id[cid] for cid in development_ids if cid in by_id] + list(known_development_rows)
    excluded_rows = list(excluded_rows)
    excluded_ids = set(development_ids) | {str(r["clip_id"]) for r in excluded_rows}
    excluded_sentences = {sentence(r) for r in excluded_rows + dev}
    dev_speakers = {speaker(r) for r in dev}
    fit_speakers, fit_sentences = {speaker(r) for r in fit}, {sentence(r) for r in fit}
    eligible = [r for r in rows if int(r["valid_frames"]) >= min_valid]
    not_fit = [r for r in eligible if str(r["clip_id"]) not in set(fit_ids)]
    conservative = [r for r in not_fit if str(r["clip_id"]) not in excluded_ids
                    and sentence(r) not in excluded_sentences and speaker(r) not in dev_speakers]
    datasets = {}
    for name in sorted({dataset(r) for r in rows}):
        part = lambda values: [r for r in values if dataset(r) == name]
        datasets[name] = {
            "all_native_train": summarize(part(rows), fps),
            "minimum32_valid_train": summarize(part(eligible), fps),
            "current_fit": summarize(part(fit), fps),
            "not_in_current_fit_not_yet_exclusion_filtered": summarize(part(not_fit), fps),
            "after_supplied_metadata_exclusions_only": summarize(part(conservative), fps),
            "same_fit_speaker_and_sentence_new_recordings": summarize(part([r for r in conservative
                if speaker(r) in fit_speakers and sentence(r) in fit_sentences]), fps),
            "potential_new_speakers_relative_to_current_fit": summarize(part([r for r in conservative
                if speaker(r) not in fit_speakers]), fps),
            "potential_new_sentence_ids_relative_to_current_fit": summarize(part([r for r in conservative
                if sentence(r) not in fit_sentences]), fps),
            "neutral_distinct_sentences_per_speaker": dict(sorted((s[1], len({sentence(r) for r in eligible
                if speaker(r) == s and int(r["emotion"]) == 0})) for s in {speaker(r) for r in part(eligible)})),
        }
    full_fit_valid = sum(int(r["valid_frames"]) for r in fit)
    actual = int(sum(observed_frames))
    fit_emotions = {int(r["emotion"]) for r in fit}
    return {"by_dataset": datasets, "current_fit": {
        **summarize(fit, fps), "actual_cached_valid_frames": actual,
        "actual_cached_valid_hours": actual / fps / 3600, "fixed_crop_frames": crop_frames,
        "fixed_crop_nominal_seconds": crop_frames / fps,
        "full_native_fit_valid_frames": full_fit_valid,
        "valid_frames_outside_current_fixed_crops": full_fit_valid - actual,
        "unique_exposure_fraction_of_full_fit_valid_frames": actual / full_fit_valid if full_fit_valid else None,
        "clips_longer_than_crop": sum(int(r["frames"]) > crop_frames for r in fit),
        "by_emotion_actual_cached_valid_frames": {EMOTIONS[e]: int(sum(count for row, count in zip(fit, observed_frames)
            if int(row["emotion"]) == e)) for e in EMOTIONS}},
        "omitted_emotions": {EMOTIONS[e]: summarize([r for r in eligible if int(r["emotion"]) == e], fps)
            for e in EMOTIONS if e not in fit_emotions},
        "exclusion_scope": {"current_development_clips": len(set(development_ids)),
            "explicit_excluded_metadata_rows": len(excluded_rows),
            "excluded_dataset_sentence_ids": len(excluded_sentences), "development_speakers": len(dev_speakers),
            "candidates_are_not_locked_or_approved": True,
            "note": "Only supplied metadata exposures were excluded. No historical or B0 pretraining novelty certification. Sentence IDs across datasets are not lexical comparability evidence."}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-train", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--exclude-metadata", type=Path, action="append", default=[],
                   help="Explicit enrollment/reserved/previous-development JSONL; metadata only")
    p.add_argument("--development-metadata", type=Path, action="append", default=[],
                   help="Explicit previous-development JSONL, also excludes their speaker IDs conservatively")
    p.add_argument("--fps", type=float, default=25., help="Declared native fps; metadata may not contain a clock")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Fresh report output required")
    if args.fps <= 0: raise ValueError("Positive declared native fps required")
    import torch
    inputs = [args.native_train, args.cache, *args.exclude_metadata, *args.development_metadata]
    hashes = {str(path.resolve()): sha(path) for path in inputs}
    rows = read_metadata(args.native_train)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)
    if cache.get("schema") != "predictable_renderer_cache_v1" or set(cache.get("splits", {})) != {"train", "validation"}:
        raise ValueError("Only current renderer train/validation cache accepted; no test cache")
    q = cache["splits"]["train"]["q"]
    fit_ids = list(map(str, q["clip_id"]))
    dev_ids = list(map(str, cache["splits"]["validation"]["q"]["clip_id"]))
    valid = q["valid"]
    if valid.dtype != torch.bool or valid.ndim != 2 or valid.shape[0] != len(fit_ids):
        raise ValueError("Invalid train validity mask")
    counts = valid.sum(1).tolist()
    excluded = [r for path in args.exclude_metadata for r in read_metadata(path)]
    previous_dev = [r for path in args.development_metadata for r in read_metadata(path)]
    result = coverage(rows, fit_ids, dev_ids, counts, valid.shape[1], fps=args.fps,
                      excluded_rows=excluded, known_development_rows=previous_dev)
    report = {"schema": "dynamic_training_coverage_v1", "source_sha256": hashes,
        "script_sha256": sha(__file__), "declared_native_fps": args.fps,
        "reads": {"native_train_metadata": True, "current_cache_train_valid_mask": True,
            "current_cache_train_validation_clip_metadata": True,
            "native_npz_or_motion_audio_video": False, "current_cache_motion_targets": False,
            "sealed_test_targets": False, "split_locks_modified": False},
        "duration_note": "Metadata frames/fps is nominal support, not independent media-duration certification. Valid hours count observations; repeated epochs do not increase unique data.",
        **result}
    # Verify input bytes remain unchanged; output is the only write.
    if any(sha(path) != hashes[str(path.resolve())] for path in inputs):
        raise RuntimeError("An input changed during read-only audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(args.output.resolve()), "current_fit": report["current_fit"],
        "datasets": {name: {key: value["clips"] for key, value in stats.items()
             if isinstance(value, dict) and "clips" in value} for name, stats in report["by_dataset"].items()}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
