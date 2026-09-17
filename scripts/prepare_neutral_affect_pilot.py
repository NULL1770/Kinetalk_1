"""Prepare a reproducible MEAD feasibility set, without VA or DTW targets.

Example (on the machine containing the native artifacts)::

    python scripts/prepare_neutral_affect_pilot.py --native-root /path/native \
        --output /path/pilot_data

Only reads native assets. train/heldout split sentences, not random windows.
Independent neutral enrollment references are shared across query splits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import EMOTIONS, SCHEMA, NeutralAffectDataset, native_clip


def rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def read_source(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def choose(rows: list[dict], speakers: list[str], args, speaker_to_id: dict[str, int],
           *, unseen: bool = False) -> tuple[dict[str, list[dict]], dict[int, list[dict]]]:
    relevant = [r for r in rows if r["dataset"] == "mead" and r["speaker"] in speakers
                and int(r["valid_frames"]) >= args.min_valid_frames]
    if set(speakers) != {r["speaker"] for r in relevant}:
        raise ValueError("Some requested MEAD speakers have no usable source records")
    neutral = {s: [r for r in relevant if r["speaker"] == s and int(r["emotion"]) == 0]
               for s in speakers}
    common = set.intersection(*({r["sentence"] for r in neutral[s]} for s in speakers))
    reference_sentences = sorted(common, key=lambda x: rank(x, args.seed))[:args.references]
    if len(reference_sentences) != args.references:
        raise ValueError("Not enough common independent neutral enrollment sentences")
    refs = {}
    for speaker in speakers:
        by_sentence = defaultdict(list)
        for row in neutral[speaker]:
            by_sentence[row["sentence"]].append(row)
        refs[speaker_to_id[speaker]] = [dict(sorted(by_sentence[s], key=lambda r: r["clip_id"])[0],
            speaker_id=speaker_to_id[speaker], emotion_id=0, intensity_id=0,
            pilot_split="unseen_enrollment" if unseen else "enrollment") for s in reference_sentences]
    candidates = [r for r in relevant if r["sentence"] not in reference_sentences
                  and EMOTIONS[int(r["emotion"])] in args.emotions
                  and (int(r["emotion"]) == 0 or int(r["intensity"]) in args.levels)]
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["speaker"], int(row["emotion"]), int(row["intensity"]))].append(row)
    expected_cells = {(s, EMOTIONS.index(e), level) for s in speakers for e in args.emotions
                      for level in ([0] if e == "neutral" else args.levels)}
    if expected_cells != set(groups):
        raise ValueError(f"Missing cells: {expected_cells - set(groups)}")
    # Stable sentence allocation across people and emotions. Never split one
    # sentence across train and heldout even when its native clip ID differs.
    allocation = {s: ("heldout" if int(rank(s, args.seed + 1), 16) % 3 == 0 else "train")
                  for s in {r["sentence"] for r in candidates}}
    result = {"unseen": []} if unseen else {"train": [], "heldout": []}
    for cell, cell_rows in sorted(groups.items()):
        splits = [("unseen", args.heldout_per_cell)] if unseen else [
            ("train", args.train_per_cell), ("heldout", args.heldout_per_cell)]
        for split, number in splits:
            seen = set()
            selected = []
            for row in sorted(cell_rows, key=lambda r: rank(r["clip_id"], args.seed)):
                if not unseen and allocation[row["sentence"]] != split:
                    continue
                if row["sentence"] in seen:
                    continue
                seen.add(row["sentence"])
                selected.append(dict(row, speaker_id=speaker_to_id[cell[0]], emotion_id=cell[1],
                                     intensity_id=cell[2], pilot_split=split))
                if len(selected) == number:
                    break
            if len(selected) != number:
                raise ValueError(f"Need {number} independent {split} sentences for {cell}; found {len(selected)}")
            result[split].extend(selected)
    return result, refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--train-source", type=Path)
    parser.add_argument("--val-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speakers", nargs="+", default=["mead_M003", "mead_M005", "mead_M007", "mead_M009"])
    parser.add_argument("--unseen-speakers", nargs="*", default=[])
    parser.add_argument("--emotions", nargs="+", choices=EMOTIONS, default=["neutral", "angry", "happy", "sad"])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--train-per-cell", type=int, default=2)
    parser.add_argument("--heldout-per-cell", type=int, default=1)
    parser.add_argument("--references", type=int, default=4)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--min-valid-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if len(args.speakers) < 4 or args.references < 4:
        raise ValueError("Feasibility experiment needs at least four speakers and four references each")
    if min(args.train_per_cell, args.heldout_per_cell) < 1:
        raise ValueError("Both training and heldout cells must be populated")
    if set(args.speakers) & set(args.unseen_speakers):
        raise ValueError("Unseen speakers overlap training speakers")
    train_source = args.train_source or args.native_root / "train.jsonl"
    val_source = args.val_source or args.native_root / "val.jsonl"
    sources = [train_source] + ([val_source] if args.unseen_speakers else [])
    speaker_to_id = {speaker: i for i, speaker in enumerate(sorted(args.speakers + args.unseen_speakers))}
    selections, references = choose(read_source(train_source), args.speakers, args, speaker_to_id)
    if args.unseen_speakers:
        unseen, extra_refs = choose(read_source(val_source), args.unseen_speakers, args, speaker_to_id, unseen=True)
        selections.update(unseen)
        references.update(extra_refs)
    if {r["sentence"] for r in selections["train"]} & {r["sentence"] for r in selections["heldout"]}:
        raise AssertionError("Training and heldout sentences overlap")
    all_queries = [row for rows in selections.values() for row in rows]
    if {r["clip_id"] for r in all_queries} & {r["clip_id"] for refs in references.values() for r in refs}:
        raise AssertionError("Query and enrollment clips overlap")
    kwargs = {"window": args.window, "min_valid_frames": args.min_valid_frames}
    materialized_refs = {speaker: [native_clip(row, args.native_root, **kwargs) for row in rows]
                         for speaker, rows in references.items()}
    materialized = {split: [native_clip(row, args.native_root, **kwargs) for row in rows]
                    for split, rows in selections.items()}
    # Fit preprocessing on training observations and training enrollment only.
    training_refs = [clip for speaker in args.speakers for clip in materialized_refs[speaker_to_id[speaker]]]
    audio = torch.cat([clip["audio"][clip["valid"]] for clip in materialized["train"] + training_refs])
    mean, std = audio.mean(0), audio.std(0).clamp_min(1e-3)
    audio_stats = {"mean": mean, "std": std, "count": len(audio),
                   "source": "train_query_and_train_neutral_enrollment_valid_frames"}
    for clip in [clip for clips in materialized.values() for clip in clips] + [
            clip for clips in materialized_refs.values() for clip in clips]:
        clip["audio"] = (clip["audio"] - mean) / std
        clip["audio"][~clip["valid"]] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": SCHEMA, "seed": args.seed, "window": args.window,
               "scope": "MEAD small-data feasibility; heldout sentences of enrolled speakers; no official test access",
               "sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
               "speaker_to_id": speaker_to_id, "audio_preprocessing": audio_stats["source"],
               "references": {str(s): [r["clip_id"] for r in rows] for s, rows in references.items()}, "splits": {}}
    for split, clips in materialized.items():
        speaker_ids = {int(clip["speaker_id"]) for clip in clips}
        selected_refs = {s: materialized_refs[s] for s in speaker_ids}
        cache = {"schema": SCHEMA, "split": split, "queries": clips,
                 "identity_references": selected_refs, "speaker_to_id": speaker_to_id,
                 "audio_stats": audio_stats, "window": args.window, "seed": args.seed}
        path = args.output / f"{split}.pt"
        torch.save(cache, path)
        NeutralAffectDataset(path)  # Validate strict reference contract on disk.
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selections[split]), encoding="utf8")
        summary["splits"][split] = {"clips": len(clips), "speakers": len(speaker_ids),
            "valid_frames": sum(int(c["valid"].sum()) for c in clips),
            "emotion_counts": dict(Counter(EMOTIONS[int(c["emotion_id"])] for c in clips)),
            "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "enrollment.jsonl").write_text("".join(json.dumps(row) + "\n" for rows in references.values() for row in rows), encoding="utf8")
    (args.output / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
