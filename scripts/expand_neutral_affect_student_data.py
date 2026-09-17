"""Expand only student training queries while preserving the original protocol.

Enrollment, heldout observations, native clock, crop policy and audio scaling
stay fixed. This prepares a controlled training-data ablation, not a new split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import EMOTIONS, NeutralAffectDataset, native_clip
from scripts.prepare_neutral_affect_pilot import choose, read_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_match_cache(rows, clips, name):
    by_id = {row["clip_id"]: row for row in rows}
    if set(by_id) != {clip["clip_id"] for clip in clips} or len(by_id) != len(clips):
        raise ValueError(f"Recomputed {name} clip selection differs from original cache")
    for clip in clips:
        row = by_id[clip["clip_id"]]
        if str(row["sentence"]) != clip["sentence_id"] or str(row["speaker"]) != clip["speaker"]:
            raise ValueError(f"{name} sentence/speaker metadata changed")
        for key in ("speaker_id", "emotion_id", "intensity_id"):
            if int(row[key]) != int(clip[key]):
                raise ValueError(f"{name} {key} changed")
        if row.get("artifact_sha256") != clip["metadata"]["artifact_sha256"]:
            raise ValueError(f"{name} artifact hash changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-cell", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.train_per_cell <= 8:
        raise ValueError("This bounded ablation permits one through eight training clips per cell")
    if args.output.resolve() == args.source_data.resolve():
        raise ValueError("Expanded data needs a different output directory")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty output directory; original or prior results are never overwritten")
    source_selection = args.source_data / "selection.json"
    original = json.loads(source_selection.read_text(encoding="utf8"))
    train = NeutralAffectDataset(args.source_data / "train.pt")
    heldout = NeutralAffectDataset(args.source_data / "heldout.pt")
    for split in ("train", "heldout"):
        expected = original.get("splits", {}).get(split, {}).get("cache_sha256")
        if expected and sha(args.source_data / (split + ".pt")) != expected:
            raise ValueError(f"Original {split} cache changed since selection was recorded")
    speaker_to_id = {name: int(value) for name, value in original["speaker_to_id"].items()}
    if speaker_to_id != train.cache["speaker_to_id"]:
        raise ValueError("Speaker mapping differs between selection and cache")
    speakers = [name for name, value in sorted(speaker_to_id.items(), key=lambda item: item[1])]
    if len(speakers) != 4 or set(train.identity_references) != set(speaker_to_id.values()):
        raise ValueError("This ablation requires exactly the original four enrolled speakers")
    reference_counts = {len(refs) for refs in train.identity_references.values()}
    if reference_counts != {4}:
        raise ValueError("This ablation requires the original four references per person")
    source_manifest = args.native_root / "train.jsonl"
    manifest_hash = sha(source_manifest)
    if not any(source["sha256"] == manifest_hash for source in original["sources"]):
        raise ValueError("Native source manifest differs from the original pilot source")
    selection_args = argparse.Namespace(seed=int(original["seed"]), references=4,
        min_valid_frames=32, emotions=["neutral", "angry", "happy", "sad"], levels=[1, 3],
        train_per_cell=args.train_per_cell, heldout_per_cell=1)
    rows, references = choose(read_source(source_manifest), speakers, selection_args, speaker_to_id)
    rows_match_cache(rows["heldout"], heldout.queries, "heldout")
    for speaker, refs in references.items():
        rows_match_cache(refs, train.identity_references[speaker], "enrollment")
        rows_match_cache(refs, heldout.identity_references[speaker], "heldout enrollment")
    expanded_rows = rows["train"]
    original_by_id = {clip["clip_id"]: clip for clip in train.queries}
    expanded_ids = {row["clip_id"] for row in expanded_rows}
    if len(expanded_ids) != len(expanded_rows) or not set(original_by_id) <= expanded_ids:
        raise ValueError("Expanded training clips must uniquely contain every original training clip")
    rows_match_cache([row for row in expanded_rows if row["clip_id"] in original_by_id], train.queries, "original training subset")
    if len(expanded_rows) != 28 * args.train_per_cell or len(expanded_rows) > 224:
        raise ValueError("Expanded size differs from the fixed 28 cells or exceeds the 224-clip budget")
    original_sentences = {clip["sentence_id"] for clip in heldout.queries}
    enrollment = [clip for refs in train.identity_references.values() for clip in refs]
    forbidden_sentences = original_sentences | {clip["sentence_id"] for clip in enrollment}
    forbidden_clips = {clip["clip_id"] for clip in heldout.queries + enrollment}
    if any(row["sentence"] in forbidden_sentences or row["clip_id"] in forbidden_clips for row in expanded_rows):
        raise ValueError("Expanded training leaks a heldout or enrollment clip/sentence")
    stats = train.cache["audio_stats"]
    mean, std = stats["mean"], stats["std"]
    if mean.shape != (83,) or std.shape != (83,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Original audio normalization statistics are invalid")
    clips = []
    for row in expanded_rows:
        if row["clip_id"] in original_by_id:
            # Preserve the original experiment's observations byte for byte.
            clip = original_by_id[row["clip_id"]]
        else:
            clip = native_clip(row, args.native_root, window=int(train.cache["window"]), min_valid_frames=32)
            clip["audio"] = (clip["audio"] - mean) / std
            clip["audio"][~clip["valid"]] = 0
        if not torch.isfinite(clip["audio"]).all():
            raise ValueError(f"Nonfinite standardized audio: {clip['clip_id']}")
        clips.append(clip)
    expansion = {"purpose": "56-versus-224 student-training-data ablation with fixed teacher, enrollment, heldout and audio preprocessing",
        "source_data": str(args.source_data.resolve()), "source_selection_sha256": sha(source_selection),
        "source_train_sha256": sha(args.source_data / "train.pt"),
        "source_heldout_sha256": sha(args.source_data / "heldout.pt"),
        "source_manifest": str(source_manifest), "source_manifest_sha256": manifest_hash,
        "original_query_count": len(train.queries), "expanded_query_count": len(clips),
        "new_query_count": len(clips) - len(train.queries), "train_per_cell": args.train_per_cell,
        "seed": selection_args.seed, "max_queries": 224,
        "audio_stats_policy": "Reuse original train cache mean/std/count exactly; never refit on expanded or heldout clips",
        "checks": {"original_train_is_subset": True, "heldout_selection_unchanged": True,
                   "enrollment_selection_unchanged": True, "query_enrollment_sentence_disjoint": True,
                   "train_heldout_sentence_disjoint": True}}
    cache = {**train.cache, "queries": clips, "identity_references": train.identity_references,
             "audio_stats": stats, "expansion": expansion}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output / "train.pt")
    shutil.copy2(args.source_data / "heldout.pt", args.output / "heldout.pt")
    if sha(args.output / "heldout.pt") != expansion["source_heldout_sha256"]:
        raise IOError("Copied heldout cache hash differs from source")
    for name in ("heldout.jsonl", "enrollment.jsonl"):
        source = args.source_data / name
        if source.is_file():
            shutil.copy2(source, args.output / name)
    manifest_text = "".join(json.dumps(row) + "\n" for row in expanded_rows)
    (args.output / "expanded_train.jsonl").write_text(manifest_text, encoding="utf8")
    (args.output / "train.jsonl").write_text(manifest_text, encoding="utf8")
    summary = {**original, "expansion": expansion, "splits": {**original["splits"],
        "train": {"clips": len(clips), "speakers": len(speakers),
                  "valid_frames": sum(int(clip["valid"].sum()) for clip in clips),
                  "emotion_counts": dict(Counter(EMOTIONS[int(clip["emotion_id"])] for clip in clips)),
                  "cache_sha256": sha(args.output / "train.pt")}}}
    summary["expansion"]["expanded_manifest_sha256"] = sha(args.output / "expanded_train.jsonl")
    (args.output / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    NeutralAffectDataset(args.output / "train.pt")
    NeutralAffectDataset(args.output / "heldout.pt")
    print(json.dumps({"train": len(clips), "original_train": len(train.queries),
                      "added": len(clips) - len(train.queries), "heldout": len(heldout.queries),
                      "audio_stats_unchanged": True, "heldout_hash_unchanged": True,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
