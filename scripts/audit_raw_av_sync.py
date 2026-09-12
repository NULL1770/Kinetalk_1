"""Audit raw-WAV versus motion activity on the consensus DTW release.

This is independent of the saved DTW audio features.  It compares the source
WAV envelope with (1) the original source motion and (2) the DTW neutral
teacher sampled on the source motion clock.
"""
from __future__ import annotations

import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np


MOUTH = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
         28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 51]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16: {path}")
        sr, n = w.getframerate(), w.getnframes()
        audio = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return audio, sr


def audio_activity(audio: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    # 25 ms RMS on a 10 ms native clock, then sample at motion times.
    win, hop = max(1, round(.025 * sr)), max(1, round(.010 * sr))
    if len(audio) < win:
        return np.zeros(len(times), dtype=np.float32)
    starts = np.arange(0, len(audio) - win + 1, hop)
    sq = audio * audio
    cs = np.concatenate(([0.0], np.cumsum(sq, dtype=np.float64)))
    rms = np.sqrt(np.maximum((cs[starts + win] - cs[starts]) / win, 1e-12))
    native_times = (starts + (win - 1) / 2) / sr
    sampled = np.interp(times, native_times, rms, left=rms[0], right=rms[-1])
    # The derivative removes stationary loudness and tracks speech activity.
    sampled = np.maximum(sampled, 1e-8)
    return np.diff(np.log(sampled), prepend=np.log(sampled[0])).astype(np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    scale = np.linalg.norm(x)
    return (x / scale).astype(np.float32) if scale > 1e-8 else np.zeros_like(x, dtype=np.float32)


def best_lag_corr(audio: np.ndarray, motion: np.ndarray, max_lag: int) -> tuple[float, int]:
    n = min(len(audio), len(motion))
    audio, motion = audio[:n], motion[:n]
    best = (-1.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, m = audio[-lag:], motion[:n + lag]
        elif lag > 0:
            a, m = audio[:n - lag], motion[lag:]
        else:
            a, m = audio, motion
        if len(a) < 4:
            continue
        corr = float(np.dot(normalize(a), normalize(m)))
        if corr > best[0]:
            best = (corr, lag)
    return best


def motion_activity(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mouth = motion[:, MOUTH].astype(np.float32)
    mouth_velocity = np.linalg.norm(np.diff(mouth, axis=0, prepend=mouth[:1]), axis=1)
    jaw_velocity = np.abs(np.diff(motion[:, 17], prepend=motion[:1, 17]))
    return mouth_velocity, jaw_velocity


def bs_path(record: dict, bs_root: Path) -> Path:
    candidate = bs_root / record["bs_rel"]
    if candidate.exists():
        return candidate
    for key in ("npz_final", "bs"):
        candidate = Path(record.get(key, ""))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"BS missing for {record['clip_id']}")


def one(row: dict, records: dict[str, dict], media_root: Path, bs_root: Path,
        max_lag: int) -> dict:
    source = records[row["source_clip_id"]]
    wav, sr = read_wav(media_root / source["audio_rel"])
    source_motion = np.load(bs_path(source, bs_root), allow_pickle=False)["coeffs"].astype(np.float32)
    with np.load(row["teacher_artifact"], allow_pickle=False) as z:
        teacher = z["neutral_teacher_on_source"].astype(np.float32)
        valid = z["native_teacher_mask"].astype(bool)
        source_times = z["source_motion_times"].astype(np.float64)
    n = min(len(source_motion), len(teacher), len(source_times))
    source_motion, teacher, valid, source_times = source_motion[:n], teacher[:n], valid[:n], source_times[:n]
    activity = audio_activity(wav, sr, source_times)
    source_mouth, source_jaw = motion_activity(source_motion)
    teacher_mouth, teacher_jaw = motion_activity(teacher)
    if valid.any():
        # Keep only contiguous valid samples for an honest teacher comparison;
        # masks can otherwise join unrelated intervals with a false velocity.
        idx = np.flatnonzero(valid)
        segments = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        seg = max((s for s in segments if len(s) >= 4), key=len, default=np.array([], dtype=int))
    else:
        seg = np.array([], dtype=int)
    if len(seg) >= 4:
        ta, tl = best_lag_corr(activity[seg], teacher_mouth[seg], max_lag)
        tj, tjl = best_lag_corr(activity[seg], teacher_jaw[seg], max_lag)
    else:
        ta = tl = tj = tjl = None
    sa, sl = best_lag_corr(activity, source_mouth, max_lag)
    sj, sjl = best_lag_corr(activity, source_jaw, max_lag)
    return {"source_clip_id": source["clip_id"], "reference_clip_id": row["reference_clip_id"],
            "emotion": source.get("emotion"), "teacher_frames": int(valid.sum()),
            "longest_valid_segment": int(len(seg)), "source_mouth_corr": sa,
            "source_mouth_lag_frames": sl, "source_jaw_corr": sj,
            "source_jaw_lag_frames": sjl, "teacher_mouth_corr": ta,
            "teacher_mouth_lag_frames": tl, "teacher_jaw_corr": tj,
            "teacher_jaw_lag_frames": tjl}


def finite_percentile(rows: list[dict], key: str) -> dict[str, float] | None:
    values = np.asarray([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], dtype=float)
    if not len(values):
        return None
    return {str(p): round(float(np.percentile(values, p)), 6) for p in (0, 5, 25, 50, 75, 95, 100)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-manifest", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--media-root", type=Path, required=True)
    ap.add_argument("--bs-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-lag", type=int, default=8)
    args = ap.parse_args()
    rows = read_jsonl(args.teacher_manifest)
    if args.limit:
        rows = rows[:args.limit]
    records = {r["clip_id"]: r for r in read_jsonl(args.manifest)}
    results = []
    errors = []
    for i, row in enumerate(rows, 1):
        try:
            results.append(one(row, records, args.media_root, args.bs_root, args.max_lag))
        except Exception as exc:
            errors.append({"source_clip_id": row.get("source_clip_id"), "error": f"{type(exc).__name__}: {exc}"})
        if i % 250 == 0:
            print(f"processed {i}/{len(rows)}", flush=True)
    by_emotion = defaultdict(list)
    for row in results:
        by_emotion[str(row.get("emotion"))].append(row)
    summary = {
        "rows": len(rows), "processed": len(results), "errors": len(errors),
        "error_examples": errors[:20], "max_lag_frames": args.max_lag,
        "all": {key: finite_percentile(results, key) for key in (
            "source_mouth_corr", "source_jaw_corr", "teacher_mouth_corr", "teacher_jaw_corr",
            "source_mouth_lag_frames", "source_jaw_lag_frames",
            "teacher_mouth_lag_frames", "teacher_jaw_lag_frames", "teacher_frames",
            "longest_valid_segment")},
        "by_emotion": {emotion: {
            "rows": len(group),
            **{key: finite_percentile(group, key) for key in (
                "source_mouth_corr", "teacher_mouth_corr", "source_jaw_corr", "teacher_jaw_corr",
                "source_mouth_lag_frames", "teacher_mouth_lag_frames")}}
            for emotion, group in sorted(by_emotion.items())},
        "interpretation": (
            "Correlations use raw WAV RMS-log derivative against mouth/jaw velocity. "
            "Teacher metrics use the longest contiguous native_teacher_mask segment. "
            "Lag is the motion-frame shift giving maximum correlation within the configured window; "
            "this is an independent sync diagnostic, not a phoneme alignment proof."),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rows", "processed", "errors", "all", "by_emotion")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
