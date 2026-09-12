"""Independent raw-WAV/audio versus motion timing audit, revision 2.

The first audit correlated a signed RMS derivative with non-negative motion
speed.  This revision reports physically interpretable pairings separately:
positive audio envelope with jaw opening, and positive audio flux with motion
speed.  It reports both zero-lag correlation and the best lag in a small
window, while comparing original source motion with the DTW neutral teacher.
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
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return audio, w.getframerate()


def median_filter(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    pad = width // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.asarray([np.median(padded[i:i + width]) for i in range(len(x))], dtype=np.float32)


def audio_features(audio: np.ndarray, sr: int, times: np.ndarray) -> dict[str, np.ndarray]:
    # Native 10 ms RMS, then interpolate onto the 25 fps motion clock.
    win, hop = max(1, round(.025 * sr)), max(1, round(.010 * sr))
    if len(audio) < win:
        z = np.zeros(len(times), dtype=np.float32)
        return {"envelope": z, "flux": z.copy()}
    starts = np.arange(0, len(audio) - win + 1, hop)
    sq = audio * audio
    cs = np.concatenate(([0.0], np.cumsum(sq, dtype=np.float64)))
    rms = np.sqrt(np.maximum((cs[starts + win] - cs[starts]) / win, 1e-12)).astype(np.float32)
    native_times = (starts + (win - 1) / 2) / sr
    env = np.interp(times, native_times, rms, left=rms[0], right=rms[-1]).astype(np.float32)
    log_env = np.log(np.maximum(env, 1e-7))
    flux = np.abs(np.diff(log_env, prepend=log_env[:1])).astype(np.float32)
    return {"envelope": env, "flux": flux}


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    norm = np.linalg.norm(x)
    return (x / norm).astype(np.float32) if norm > 1e-8 else np.zeros_like(x, dtype=np.float32)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 6:
        return float("nan")
    aa, bb = normalize(a), normalize(b)
    value = float(np.dot(aa, bb))
    return value if np.isfinite(value) else float("nan")


def lagged_corr(audio: np.ndarray, motion: np.ndarray, max_lag: int) -> tuple[float, int]:
    n = min(len(audio), len(motion))
    audio, motion = audio[:n], motion[:n]
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, m = audio[-lag:], motion[:n + lag]
        elif lag > 0:
            a, m = audio[:n - lag], motion[lag:]
        else:
            a, m = audio, motion
        candidates.append((corr(a, m), lag))
    candidates = [(v, lag) for v, lag in candidates if np.isfinite(v)]
    return max(candidates, default=(float("nan"), 0))


def motion_features(motion: np.ndarray) -> dict[str, np.ndarray]:
    mouth = motion[:, MOUTH].astype(np.float32)
    jaw_open = motion[:, 17].astype(np.float32)
    speed = np.linalg.norm(np.diff(mouth, axis=0, prepend=mouth[:1]), axis=1)
    jaw_speed = np.abs(np.diff(jaw_open, prepend=jaw_open[:1]))
    # A short median filter suppresses single-frame tracking spikes without
    # changing the 25 fps clock.
    return {"jaw_open": median_filter(jaw_open, 3),
            "mouth_speed": median_filter(speed, 3),
            "jaw_speed": median_filter(jaw_speed, 3)}


def bs_path(record: dict, bs_root: Path) -> Path:
    candidate = bs_root / record["bs_rel"]
    if candidate.exists():
        return candidate
    for key in ("npz_final", "bs"):
        candidate = Path(record.get(key, ""))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(record["clip_id"])


def restricted(values: dict[str, np.ndarray], valid: np.ndarray | None) -> dict[str, np.ndarray]:
    if valid is None:
        return values
    return {key: value[valid] for key, value in values.items()}


def one(row: dict, records: dict[str, dict], media_root: Path, bs_root: Path,
        max_lag: int) -> dict:
    source = records[row["source_clip_id"]]
    source_motion = np.load(bs_path(source, bs_root), allow_pickle=False)["coeffs"].astype(np.float32)
    with np.load(row["teacher_artifact"], allow_pickle=False) as z:
        teacher = z["neutral_teacher_on_source"].astype(np.float32)
        valid_mask = z["native_teacher_mask"].astype(bool)
        source_times = z["source_motion_times"].astype(np.float64)
    n = min(len(source_motion), len(teacher), len(source_times))
    source_motion, teacher, valid_mask, source_times = (source_motion[:n], teacher[:n],
                                                        valid_mask[:n], source_times[:n])
    audio, sr = read_wav(media_root / source["audio_rel"])
    audio_f = audio_features(audio, sr, source_times)
    source_f, teacher_f = motion_features(source_motion), motion_features(teacher)
    valid = np.flatnonzero(valid_mask)
    segments = np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1) if len(valid) else []
    segment = max((s for s in segments if len(s) >= 6), key=len, default=np.array([], dtype=int))

    result = {"source_clip_id": source["clip_id"], "reference_clip_id": row["reference_clip_id"],
              "emotion": source.get("emotion"), "identity": source["clip_id"] == row["reference_clip_id"],
              "teacher_frames": int(valid_mask.sum()), "longest_valid_segment": int(len(segment))}
    for name, audio_key, motion_key in (
            ("env_jaw", "envelope", "jaw_open"),
            ("flux_mouth", "flux", "mouth_speed"),
            ("flux_jaw", "flux", "jaw_speed")):
        a, s = audio_f[audio_key], source_f[motion_key]
        result[f"source_{name}_zero"] = corr(a, s)
        result[f"source_{name}_best"], result[f"source_{name}_lag"] = lagged_corr(a, s, max_lag)
        if len(segment):
            result[f"teacher_{name}_zero"] = corr(audio_f[audio_key][segment], teacher_f[motion_key][segment])
            result[f"teacher_{name}_best"], result[f"teacher_{name}_lag"] = lagged_corr(
                audio_f[audio_key][segment], teacher_f[motion_key][segment], max_lag)
        else:
            for prefix in ("teacher_",):
                result[f"{prefix}{name}_zero"] = float("nan")
                result[f"{prefix}{name}_best"] = float("nan")
                result[f"{prefix}{name}_lag"] = 0
    return result


def percentiles(rows: list[dict], key: str) -> dict[str, float] | None:
    values = np.asarray([row[key] for row in rows if np.isfinite(row.get(key, np.nan))], dtype=float)
    if not len(values):
        return None
    return {str(p): round(float(np.percentile(values, p)), 6) for p in (0, 5, 25, 50, 75, 95, 100)}


def summarize(rows: list[dict], prefix: str) -> dict:
    keys = ["env_jaw_zero", "env_jaw_best", "flux_mouth_zero", "flux_mouth_best",
            "flux_jaw_zero", "flux_jaw_best", "env_jaw_lag", "flux_mouth_lag", "flux_jaw_lag"]
    return {key: percentiles(rows, f"{prefix}_{key}") for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-manifest", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--media-root", type=Path, required=True)
    ap.add_argument("--bs-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-lag", type=int, default=4,
                    help="Best-lag window in 25 fps motion frames; default +/-160 ms.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = read_jsonl(args.teacher_manifest)
    if args.limit:
        rows = rows[:args.limit]
    records = {row["clip_id"]: row for row in read_jsonl(args.manifest)}
    results, errors = [], []
    for i, row in enumerate(rows, 1):
        try:
            results.append(one(row, records, args.media_root, args.bs_root, args.max_lag))
        except Exception as exc:
            errors.append({"source_clip_id": row.get("source_clip_id"),
                           "error": f"{type(exc).__name__}: {exc}"})
        if i % 250 == 0:
            print(f"processed {i}/{len(rows)}", flush=True)
    cross = [row for row in results if not row["identity"]]
    identity = [row for row in results if row["identity"]]
    by_emotion = defaultdict(list)
    for row in cross:
        by_emotion[str(row.get("emotion"))].append(row)
    summary = {
        "rows": len(rows), "processed": len(results), "errors": len(errors),
        "error_examples": errors[:20], "max_lag_frames": args.max_lag,
        "all": summarize(results, "source"),
        "cross": {"source": summarize(cross, "source"),
                  "teacher": summarize(cross, "teacher")},
        "identity": summarize(identity, "source"),
        "by_emotion": {emotion: {"source": summarize(group, "source"),
                                  "teacher": summarize(group, "teacher")}
                        for emotion, group in sorted(by_emotion.items())},
        "interpretation": (
            "Zero metrics use the actual source/teacher time axes. Best metrics search only +/- "
            f"{args.max_lag} motion frames. env_jaw pairs positive RMS envelope with jawOpen; "
            "flux_mouth and flux_jaw pair positive log-RMS change with mouth/jaw speed. "
            "Identity rows are the no-DTW baseline; a teacher-specific degradation is evidence "
            "against the path, while low identity values indicate the metric is weak for this data."),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("rows", "processed", "errors", "all", "cross", "identity")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
