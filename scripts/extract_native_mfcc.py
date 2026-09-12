"""Extract deterministic native-clock MFCC features for DTW auditing.

The feature clock is 16 kHz audio with a 25 ms window and 10 ms hop.  No
resampling to the motion frame count is performed.  This is an alignment
audit feature, not a replacement for an English HuBERT/WavLM checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np


def audio_to_float(path: Path, ffmpeg: str, sr: int = 16000) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (sr, 1, 2):
                raise ValueError("WAV is not 16 kHz mono PCM16")
            y = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    except (wave.Error, EOFError, ValueError):
        p = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        y = np.frombuffer(p.stdout, dtype="<f4").copy()
    if len(y) < 400:
        raise ValueError(f"audio too short: {path}")
    if not np.isfinite(y).all() or float(np.sqrt(np.mean(y * y))) <= 1e-6:
        raise ValueError(f"audio is silent/nonfinite: {path}")
    return y


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mel)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1:m + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, bank.shape[1])):
            bank[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, min(right, bank.shape[1])):
            bank[m - 1, k] = (right - k) / float(right - center)
    return bank


def dct_type_2(x: np.ndarray, n_out: int) -> np.ndarray:
    n = x.shape[1]
    k = np.arange(n_out, dtype=np.float32)[:, None]
    i = np.arange(n, dtype=np.float32)[None, :]
    basis = np.cos(np.pi / n * (i + 0.5) * k).astype(np.float32)
    basis[0] *= np.sqrt(1.0 / n)
    basis[1:] *= np.sqrt(2.0 / n)
    return x @ basis.T


def delta(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[1] - x[0]
    out[-1] = x[-1] - x[-2]
    if len(x) > 2:
        out[1:-1] = 0.5 * (x[2:] - x[:-2])
    return out


def extract(y: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    win, hop, n_fft = 400, 160, 512
    if len(y) < win:
        raise ValueError("audio shorter than one analysis window")
    n = 1 + (len(y) - win) // hop
    frames = np.stack([y[i * hop:i * hop + win] for i in range(n)], axis=0)
    frames = frames * np.hanning(win).astype(np.float32)[None, :]
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    bank = mel_filterbank(sr, n_fft, 26, 20.0, min(7600.0, sr / 2.0 - 200.0))
    logmel = np.log(np.maximum(power @ bank.T, 1e-10)).astype(np.float32)
    cep = dct_type_2(logmel, 13)
    feat = np.concatenate([cep, delta(cep), delta(delta(cep))], axis=1).astype(np.float32)
    # Per-clip CMVN improves cross-emotion matching while preserving the
    # native frame clock.  The unnormalized feature is never used by DTW.
    feat = (feat - feat.mean(0, keepdims=True)) / np.maximum(feat.std(0, keepdims=True), 1e-4)
    times = (np.arange(n, dtype=np.float64) * hop + win / 2.0) / sr
    return feat, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--groups", type=int, default=0,
                    help="Select complete speaker/content groups containing neutral clips.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("dataset") == "mead" and r.get("training_eligible", True)]
    if args.groups:
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row.get("speaker", ""), row.get("sentence_id", ""))].append(row)
        candidates = [group for _, group in sorted(grouped.items())
                      if any(r.get("emotion") == "neutral" for r in group)]
        rows = [row for group in candidates[:args.groups] for row in group]
    if args.limit:
        rows = rows[:args.limit]
    recipe = {"schema": 1, "feature": "mfcc39_cmvn", "sample_rate": 16000,
              "window_samples": 400, "hop_samples": 160, "n_fft": 512,
              "n_mels": 26, "n_mfcc": 13, "center": False}
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "recipe.json").write_text(json.dumps({**recipe, "signature": signature}, indent=2), encoding="utf-8")
    (args.out_root / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    done = 0
    for row in rows:
        out = args.out_root / row["dataset"] / f"{row['clip_id']}.npz"
        if out.exists() and not args.overwrite:
            with np.load(out, allow_pickle=False) as z:
                if str(z["signature"]) == signature:
                    done += 1
                    continue
            raise ValueError(f"incompatible cached feature: {out}")
        audio = Path(row["audio"])
        if not audio.is_file() and row.get("audio_rel"):
            audio = args.manifest.parent / row["audio_rel"]
        if not audio.is_file():
            raise FileNotFoundError(f"audio missing for {row['clip_id']}: {audio}")
        y = audio_to_float(audio, args.ffmpeg)
        feat, times = extract(y)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, content=feat.astype(np.float32), validation=feat.astype(np.float32),
                            times=times, signature=signature, audio_samples=len(y),
                            sample_rate=16000, hop_samples=160, receptive_samples=400)
        done += 1
        if done % 250 == 0:
            print(f"processed {done}/{len(rows)}", flush=True)
    print(json.dumps({"processed": done, "total": len(rows), "feature": recipe["feature"],
                      "out_root": str(args.out_root)}))


if __name__ == "__main__":
    main()
