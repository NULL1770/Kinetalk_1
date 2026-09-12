"""Independent ASR content audit for the DTW consensus teacher manifest.

This script audits each unique clip once, then compares the source and neutral
reference transcripts.  It is a quarantine screen: ASR disagreement can remove
a pair from framewise teaching, while ASR agreement does not prove correctness.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, tokens(a), tokens(b), autojunk=False).ratio()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audio_path(row: dict, media_root: Path) -> Path:
    path = media_root / row["audio_rel"]
    if path.exists():
        return path
    candidate = Path(row.get("audio", ""))
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"audio missing for {row['clip_id']}: {path}")


def transcribe(model, row: dict, media_root: Path) -> dict:
    segments, info = model.transcribe(
        str(audio_path(row, media_root)), language="en", beam_size=5,
        condition_on_previous_text=False, word_timestamps=False,
        vad_filter=False)
    segments = list(segments)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    logprobs = [float(segment.avg_logprob) for segment in segments
                if segment.avg_logprob is not None]
    return {
        "clip_id": row["clip_id"], "speaker": row.get("speaker"),
        "emotion": row.get("emotion"), "intensity": row.get("intensity"),
        "content_id": row.get("content_id"), "expected_text": row.get("text_normalized", row.get("text", "")),
        "asr_text": text, "asr_tokens": tokens(text),
        "annotation_similarity": similarity(row.get("text_normalized", row.get("text", "")), text),
        "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else None,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }


def classify(pair: dict, source: dict, reference: dict, threshold: float,
             uncertain_logprob: float) -> dict:
    pair_sim = SequenceMatcher(None, source["asr_tokens"], reference["asr_tokens"], autojunk=False).ratio()
    source_ok = source["annotation_similarity"] >= threshold
    ref_ok = reference["annotation_similarity"] >= threshold
    if pair_sim < threshold:
        status = "quarantine_asr_disagreement"
    elif not source_ok or not ref_ok or any(
            value is not None and value < uncertain_logprob
            for value in (source["avg_logprob"], reference["avg_logprob"])):
        status = "review_asr_uncertain"
    else:
        status = "asr_consistent"
    return {
        "source_clip_id": pair["source_clip_id"],
        "reference_clip_id": pair["reference_clip_id"],
        "emotion": source.get("emotion"), "content_id": pair.get("content_id"),
        "expected_text": source.get("expected_text", ""),
        "source_asr_text": source["asr_text"], "reference_asr_text": reference["asr_text"],
        "source_annotation_similarity": source["annotation_similarity"],
        "reference_annotation_similarity": reference["annotation_similarity"],
        "source_reference_similarity": pair_sim,
        "source_avg_logprob": source["avg_logprob"],
        "reference_avg_logprob": reference["avg_logprob"],
        "status": status,
        "teacher_artifact": pair.get("teacher_artifact"),
        "audit_artifact_b": pair.get("audit_artifact_b"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-manifest", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--media-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--clip-output", type=Path, required=True)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--pair-threshold", type=float, default=.75)
    ap.add_argument("--uncertain-logprob", type=float, default=-.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    pairs = read_jsonl(args.teacher_manifest)
    if args.limit:
        pairs = pairs[:args.limit]
    manifest = {row["clip_id"]: row for row in read_jsonl(args.manifest)}
    clips = {}
    for pair in pairs:
        for key in ("source_clip_id", "reference_clip_id"):
            clips[pair[key]] = manifest[pair[key]]

    cached = {}
    if args.clip_output.exists():
        for row in read_jsonl(args.clip_output):
            cached[row["clip_id"]] = row
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type, cpu_threads=6)
    args.clip_output.parent.mkdir(parents=True, exist_ok=True)
    pending = [row for clip_id, row in clips.items() if clip_id not in cached]
    if pending:
        with args.clip_output.open("a", encoding="utf-8") as out:
            for index, row in enumerate(pending, 1):
                result = transcribe(model, row, args.media_root)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                cached[row["clip_id"]] = result
                if index % 100 == 0:
                    print(f"clips {index}/{len(pending)}", flush=True)

    results = [classify(pair, cached[pair["source_clip_id"]],
                         cached[pair["reference_clip_id"]],
                         args.pair_threshold, args.uncertain_logprob)
               for pair in pairs]
    by_emotion = defaultdict(list)
    for row in results:
        by_emotion[str(row.get("emotion"))].append(row)
    summary = {
        "pairs": len(results), "unique_clips": len(clips),
        "status": dict(Counter(row["status"] for row in results)),
        "by_emotion": {emotion: dict(Counter(row["status"] for row in group))
                        for emotion, group in sorted(by_emotion.items())},
        "pair_similarity": {
            "median": sorted(row["source_reference_similarity"] for row in results)[len(results) // 2] if results else None,
            "below_threshold": sum(row["source_reference_similarity"] < args.pair_threshold for row in results),
        },
        "source_annotation_similarity_below_threshold": sum(
            row["source_annotation_similarity"] < args.pair_threshold for row in results),
        "reference_annotation_similarity_below_threshold": sum(
            row["reference_annotation_similarity"] < args.pair_threshold for row in results),
        "threshold": args.pair_threshold,
        "uncertain_logprob": args.uncertain_logprob,
        "interpretation": (
            "ASR disagreement is a quarantine signal for framewise teaching. "
            "ASR consistency is only supporting evidence; it does not replace human verification "
            "or phoneme-level forced alignment. Identity pairs are expected to be self-consistent."),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("pairs", "unique_clips", "status", "by_emotion", "pair_similarity")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
