"""Summarize existing independent ASR checks against annotation content."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ratio(a: list[str], b: list[str]) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", type=Path, nargs="+", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pair-threshold", type=float, default=.75)
    args = ap.parse_args()
    manifest = {r["clip_id"]: r for r in read_jsonl(args.manifest)}
    rows = []
    for path in args.asr:
        for row in read_jsonl(path):
            row = dict(row)
            row["asr_file"] = str(path)
            expected = manifest.get(row.get("clip_id"), {}).get("text_normalized", "")
            row["annotation_similarity"] = ratio(tokens(expected), tokens(row.get("text", "")))
            row["expected_text"] = expected
            rows.append(row)
    by_clip = {row["clip_id"]: row for row in rows}
    groups = defaultdict(list)
    for row in rows:
        m = manifest.get(row["clip_id"], {})
        groups[(m.get("speaker"), m.get("content_id"))].append(row)
    # The old numeric groups are reconstructed from the manifest to measure
    # how dangerous the prior index based pairing was in the audited subset.
    old_groups = defaultdict(list)
    for row in rows:
        m = manifest.get(row["clip_id"], {})
        old_groups[(m.get("speaker"), m.get("legacy_sentence_id"))].append(row)
    exact_groups = []
    for key, group in groups.items():
        neutral = [r for r in group if manifest[r["clip_id"]].get("emotion") == "neutral"]
        if not neutral:
            continue
        ref = neutral[0]
        for row in group:
            if row["clip_id"] == ref["clip_id"]:
                continue
            exact_groups.append({
                "speaker": key[0], "content_id": key[1],
                "source_clip_id": row["clip_id"], "reference_clip_id": ref["clip_id"],
                "emotion": manifest[row["clip_id"]].get("emotion"),
                "source_to_reference_asr_similarity": ratio(tokens(row.get("text", "")), tokens(ref.get("text", ""))),
                "source_annotation_similarity": row["annotation_similarity"],
                "reference_annotation_similarity": ref["annotation_similarity"],
                "both_asr_agree": ratio(tokens(row.get("text", "")), tokens(ref.get("text", ""))) >= args.pair_threshold,
            })
    old_pairs = []
    for key, group in old_groups.items():
        neutral = [r for r in group if manifest[r["clip_id"]].get("emotion") == "neutral"]
        if not neutral:
            continue
        ref = neutral[0]
        for row in group:
            if manifest[row["clip_id"]].get("emotion") == "neutral":
                continue
            old_pairs.append({
                "speaker": key[0], "legacy_sentence_id": key[1],
                "source_clip_id": row["clip_id"], "reference_clip_id": ref["clip_id"],
                "emotion": manifest[row["clip_id"]].get("emotion"),
                "asr_similarity": ratio(tokens(row.get("text", "")), tokens(ref.get("text", ""))),
                "same_annotation_content_id": manifest[row["clip_id"]].get("content_id") == manifest[ref["clip_id"]].get("content_id"),
            })
    by_emotion = defaultdict(list)
    for row in exact_groups:
        by_emotion[row["emotion"]].append(row)
    confidence = [float(row.get("avg_logprob", [float("nan")])[0])
                  for row in rows if row.get("avg_logprob")]
    summary = {
        "asr_files": [str(p) for p in args.asr], "clips": len(rows),
        "unique_clips": len(by_clip), "duplicate_clip_rows": len(rows) - len(by_clip),
        "annotation_similarity": {
            "median": sorted(r["annotation_similarity"] for r in rows)[len(rows) // 2] if rows else None,
            "below_0.75": sum(r["annotation_similarity"] < .75 for r in rows),
            "below_0.5": sum(r["annotation_similarity"] < .5 for r in rows),
        },
        "asr_avg_logprob": {
            "median": sorted(confidence)[len(confidence) // 2] if confidence else None,
            "below_-0.5": sum(x < -.5 for x in confidence),
        },
        "exact_content_groups": len(exact_groups),
        "exact_content_pairs_below_threshold": sum(not r["both_asr_agree"] for r in exact_groups),
        "exact_content_pairs_below_threshold_fraction": sum(not r["both_asr_agree"] for r in exact_groups) / max(len(exact_groups), 1),
        "exact_by_emotion": {
            emotion: {
                "pairs": len(group),
                "below_threshold": sum(not r["both_asr_agree"] for r in group),
                "similarity_median": sorted(r["source_to_reference_asr_similarity"] for r in group)[len(group) // 2],
            } for emotion, group in sorted(by_emotion.items())},
        "old_number_pairs_in_audited_subset": len(old_pairs),
        "old_number_same_text_pairs": sum(r["same_annotation_content_id"] for r in old_pairs),
        "old_number_asr_similarity_below_threshold": sum(r["asr_similarity"] < args.pair_threshold for r in old_pairs),
        "old_number_examples": old_pairs[:50],
        "exact_content_examples": [r for r in exact_groups if not r["both_asr_agree"]][:50],
        "threshold": args.pair_threshold,
        "interpretation": (
            "This is an independent ASR screen, not human transcription ground truth. "
            "A high ASR disagreement is sufficient to quarantine a framewise teacher, "
            "but a high agreement does not prove the pair is correct."
        ),
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
