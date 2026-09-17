"""Read-only fixed train-source brow audit; never treats tracking as visual GT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 20260922
EMOTIONS = {0: "neutral", 1: "angry", 5: "happy", 6: "sad"}
BROWS = {41: "browDownL", 42: "browDownR", 43: "browInnerUp",
         44: "browOuterUpL", 45: "browOuterUpR"}
EXCLUDED_SPEAKERS = {"mead_M024", "mead_M023", "mead_M030"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def select_metadata(fit_rows, old_rows):
    fit = {row["clip_id"]: row for row in fit_rows}
    if len(fit) != 2315 or len({r["speaker"] for r in fit_rows}) != 19:
        raise ValueError("This audit requires the locked 2315-clip/19-person fit manifest")
    candidates = [fit[r["clip_id"]] for r in old_rows if r["clip_id"] in fit]
    if len(candidates) != 14:
        raise ValueError("Expected 14 old metadata candidates in this fit split")
    chosen = []
    for emotion in EMOTIONS:
        subset = [r for r in candidates if int(r["emotion_id"]) == emotion]
        subset.sort(key=lambda r: hashlib.sha256(f"{SEED}:{r['clip_id']}".encode()).hexdigest())
        chosen.extend(subset[:2])
    if len(chosen) != 8 or len({r["clip_id"] for r in chosen}) != 8:
        raise ValueError("Expected exactly eight clips")
    if any(r["speaker"] in EXCLUDED_SPEAKERS for r in chosen):
        raise ValueError("Internal development person entered source audit")
    return chosen


def native_statistics(motion, times, valid, channel_mask):
    paired = valid[:-1] & valid[1:]
    tripled = valid[:-2] & valid[1:-1] & valid[2:]
    result = {}
    for channel, name in BROWS.items():
        if not channel_mask[channel]:
            raise ValueError(f"Unobserved brow channel: {name}")
        values = motion[:, channel]
        observed = values[valid]
        first = np.diff(values)[paired]
        second = np.diff(values, n=2)[tripled]
        result[name] = {
            "mean": float(observed.mean()), "std": float(observed.std()),
            "min": float(observed.min()), "max": float(observed.max()),
            "q01": float(np.quantile(observed, .01)), "q99": float(np.quantile(observed, .99)),
            "fraction_le_0_01": float(np.mean(observed <= .01)),
            "fraction_ge_0_99": float(np.mean(observed >= .99)),
            "adjacent_exact_repeat_fraction_atol_1e_6": float(np.mean(np.abs(first) <= 1e-6)),
            "adjacent_difference_rms": float(np.sqrt(np.mean(first ** 2))),
            "second_difference_rms": float(np.sqrt(np.mean(second ** 2))),
            "max_adjacent_velocity_per_s": float(np.max(np.abs(first / np.diff(times)[paired]))),
        }
    return result


def media_frames(video, targets, ffprobe):
    args = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)]
    metadata = json.loads(subprocess.run(args, capture_output=True, check=True).stdout)
    v = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    a = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    args = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
            "frame=best_effort_timestamp_time", "-of", "json", str(video)]
    entries = json.loads(subprocess.run(args, capture_output=True, check=True).stdout)["frames"]
    pts = np.asarray([float(f["best_effort_timestamp_time"]) for f in entries], dtype=np.float64)
    pts -= float(v.get("start_time", 0))
    if len(pts) < 2 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("Invalid decoded source frame timestamps")
    chosen = [int(np.argmin(abs(pts - t))) for t in targets]
    cap = cv2.VideoCapture(str(video))
    frames, index = {}, 0
    while index <= max(chosen):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Failed to decode source frame {index}: {video}")
        if index in chosen:
            h, w = frame.shape[:2]
            # Fixed central face crop, identical rule for every selected frame.
            crop = frame[int(h * .08):int(h * .88), int(w * .26):int(w * .74)]
            frames[index] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        index += 1
    cap.release()
    return [frames[i] for i in chosen], {
        "ffprobe_video": v, "ffprobe_audio": a,
        "source_frame_indices": chosen, "source_relative_pts_s": pts[chosen].tolist(),
        "target_to_source_time_error_s": (pts[chosen] - targets).tolist(),
        "decoded_source_frame_count": len(pts),
        "video_duration_s": float(v.get("duration", metadata["format"]["duration"])),
        "audio_video_stream_start_offset_s": float(a.get("start_time", 0)) - float(v.get("start_time", 0)),
    }


def audio_audit(video, wave_path, provenance, ffmpeg):
    if sha(wave_path) != provenance["audio_sha256"]:
        raise ValueError("Waveform differs from the native artifact provenance")
    with wave.open(str(wave_path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected 16 kHz mono PCM16")
        used = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64) / 32768
    command = [ffmpeg, "-nostdin", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
               "-f", "f32le", "pipe:1"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    embedded = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    n = min(len(used), len(embedded))
    return {
        "native_wave_sha256": sha(wave_path), "native_samples": len(used), "embedded_samples": len(embedded),
        "zero_lag_waveform_correlation": float(np.corrcoef(used[:n], embedded[:n])[0, 1]),
        "sample_duration_difference_s": (len(used) - len(embedded)) / 16000,
        "native_audio_offset_s": float(provenance["audio_offset_s"]),
        "alignment_fitted": False,
    }


def raw_processing_audit(raw_path, motion, valid, provenance):
    if sha(raw_path) != provenance["raw_sha256"]:
        raise ValueError("Raw coefficient source differs from native provenance")
    with np.load(raw_path, allow_pickle=False) as source:
        raw, raw_valid = source["coeffs"].astype(np.float64), source["valid"].astype(bool)
    if raw.shape != motion.shape or raw_valid.shape != valid.shape:
        raise ValueError("Raw/native dimensions disagree")
    observed = valid & raw_valid
    x, y = raw[observed][:, list(BROWS)], motion[observed][:, list(BROWS)]
    if not np.isfinite(x).all():
        raise ValueError("Observed raw coefficients are nonfinite")
    x, y = x - x.mean(0), y - y.mean(0)
    sx, sy = float(np.square(x).sum()), float(np.square(y).sum())
    return {
        "raw_path": str(raw_path), "raw_sha256": sha(raw_path),
        "common_valid_frames": int(observed.sum()), "raw_valid_frames": int(raw_valid.sum()),
        "native_over_raw_centered_brow_energy": sy / max(sx, 1e-12),
        "raw_native_centered_brow_correlation": float(np.sum(x * y) / max(np.sqrt(sx * sy), 1e-12)),
        "note": "Compares processing on five native brow channels, not independent visual label accuracy; per-channel time means removed on the same valid samples.",
    }


def plot_page(rows, path, centered_limit):
    fig = plt.figure(figsize=(18, 5.2 * len(rows)))
    grid = fig.add_gridspec(4 * len(rows), 8, height_ratios=[1.7, 1, 1, .08] * len(rows), hspace=.52, wspace=.04)
    for row_num, row in enumerate(rows):
        for col, frame in enumerate(row["frames"]):
            ax = fig.add_subplot(grid[4 * row_num, col]); ax.imshow(frame); ax.axis("off")
            ax.set_title(f"{row['sample_times'][col]:.2f}s", fontsize=8)
        ax = fig.add_subplot(grid[4 * row_num + 1, :])
        for channel, name in BROWS.items():
            ax.plot(row["times"], np.where(row["valid"], row["motion"][:, channel], np.nan), label=name, lw=1)
        for t in row["sample_times"]:
            ax.axvline(t, color="gray", lw=.6, alpha=.4)
        ax.set_title(row["clip_id"] + " | native brow coefficients (same raw scale)", fontsize=10)
        ax.set_ylim(-.03, 1.03); ax.grid(alpha=.2); ax.legend(ncol=5, fontsize=8, loc="upper right")
        bx = fig.add_subplot(grid[4 * row_num + 2, :])
        for channel, name in BROWS.items():
            values = row["motion"][:, channel]
            bx.plot(row["times"], np.where(row["valid"], values - values[row["valid"]].mean(), np.nan), label=name, lw=1)
        for t in row["sample_times"]:
            bx.axvline(t, color="gray", lw=.6, alpha=.4)
        bx.set_title("Observed temporal mean removed for inspection only; native values are unchanged", fontsize=9)
        bx.set_ylim(-centered_limit, centered_limit); bx.grid(alpha=.2); bx.set_xlabel("Native source time / seconds", fontsize=8)
    fig.suptitle("Train-only fixed source inspection | 8 uniform-time valid frames per clip | no fitted lag or gain", fontsize=12)
    fig.savefig(path, dpi=115, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", type=Path, default=Path("artifacts/teacher_schedule_v1/data_locked/train.jsonl"))
    parser.add_argument("--old-root", type=Path, default=Path("artifacts/formal_readiness/supervision_samples"))
    parser.add_argument("--video-root", type=Path, default=Path("E:/mead"))
    parser.add_argument("--raw-root", type=Path, default=Path("D:/kinetalk_data/processed/coeffs_raw/mead"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/teacher_schedule_v1/temporal_refiner_v1/supervision"))
    parser.add_argument("--ffmpeg", default="ffmpeg"); parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh audit output directory")
    args.output.mkdir(parents=True)
    old_selection = args.old_root / "selection.json"
    chosen = select_metadata(read_rows(args.fit_manifest), json.loads(old_selection.read_text(encoding="utf-8-sig"))["clips"])
    selection = {
        "schema": "train_brow_source_audit_v1", "seed": SEED,
        "selection_rule": "Old 16 metadata intersect locked fit IDs (14); per emotion 0,1,5,6 take two lowest sha256('20260922:'+clip_id)",
        "time_sampling": "Eight uniformly spaced requested times from first to last valid native timestamp, snapped to nearest valid native sample; then nearest decoded source PTS",
        "candidate_pool_previously_inspected": True, "new_blind_sample": False,
        "metadata_saved_before_native_target_read": True,
        "fit_manifest_sha256": sha(args.fit_manifest), "old_selection_sha256": sha(old_selection),
        "script_sha256": sha(__file__), "clips": chosen,
        "scope": "Only current fit clips; no internal-development, outer-development or final-test targets read",
    }
    save(args.output / "selection.json", selection)
    audits, plot_rows = [], []
    for row in chosen:
        native = args.old_root / "native" / Path(row["artifact"]).name
        if sha(native) != row["artifact_sha256"]:
            raise ValueError(f"Native artifact is not the locked fit artifact: {row['clip_id']}")
        with np.load(native, allow_pickle=False) as source:
            motion, times = source["motion"].astype(np.float64), source["times"].astype(np.float64)
            valid, channel_mask = source["mask"].astype(bool), source["channel_mask"].astype(bool)
            provenance = json.loads(str(source["provenance"].item()))
        if motion.shape != (len(times), 52) or len(valid) != len(times) or channel_mask.shape != (52,):
            raise ValueError("Unexpected native array dimensions")
        if not np.isfinite(motion).all() or not np.isfinite(times).all() or not valid.any():
            raise ValueError("Nonfinite native array or no valid observations")
        if np.any(np.diff(times) <= 0) or not np.allclose(np.diff(times), 1 / float(provenance["fps"]), atol=1e-8):
            raise ValueError("Unexpected native sample clock")
        if provenance["clock_evidence"] != "embedded_video":
            raise ValueError("Native provenance lacks embedded-video clock evidence")
        observed = np.flatnonzero(valid)
        requested = np.linspace(times[observed[0]], times[observed[-1]], 8)
        sample_ids = np.array([observed[np.argmin(abs(times[observed] - t))] for t in requested])
        if len(set(sample_ids.tolist())) != 8:
            raise ValueError("Insufficient distinct valid timestamps")
        _, speaker, emotion, level, number = row["clip_id"].split("_")
        video = args.video_root / speaker / "video/front" / emotion / f"level_{level[1:]}" / f"{number}.mp4"
        frames, clock = media_frames(video, times[sample_ids], args.ffprobe)
        audio = audio_audit(video, args.old_root / "wave" / f"{row['clip_id']}.wav", provenance, args.ffmpeg)
        raw_processing = raw_processing_audit(args.raw_root / f"{row['clip_id']}.npz", motion, valid, provenance)
        if abs(clock["audio_video_stream_start_offset_s"] - audio["native_audio_offset_s"]) > 1 / 16000:
            raise ValueError("Native/source declared audio clock differs")
        stats = native_statistics(motion, times, valid, channel_mask)
        audits.append({
            "clip_id": row["clip_id"], "speaker": row["speaker"], "emotion_id": int(row["emotion_id"]),
            "native_path": str(native.resolve()), "native_sha256": sha(native), "provenance": provenance,
            "video_path": str(video), "video_sha256": sha(video),
            "native_frames": len(times), "valid_frames": int(valid.sum()), "valid_fraction": float(valid.mean()),
            "native_duration_s": float(times[-1] + np.median(np.diff(times))),
            "whole_motion_range": [float(motion.min()), float(motion.max())],
            "observed_outside_zero_one_count": int(np.count_nonzero((motion[valid][:, channel_mask] < 0) | (motion[valid][:, channel_mask] > 1))),
            "sample_native_indices": sample_ids.tolist(), "sample_native_times_s": times[sample_ids].tolist(),
            "requested_uniform_native_times_s": requested.tolist(),
            "brow_statistics": stats, "video_clock": clock, "audio_audit": audio,
            "raw_processing_audit": raw_processing,
        })
        plot_rows.append(dict(clip_id=row["clip_id"], frames=frames, times=times, valid=valid,
                              motion=motion, sample_times=times[sample_ids]))
    max_centered = max(float(np.max(np.abs(r["motion"][r["valid"]][:, list(BROWS)]
                                          - r["motion"][r["valid"]][:, list(BROWS)].mean(0)))) for r in plot_rows)
    centered_limit = max(.1, float(np.ceil((max_centered + .01) * 20) / 20))
    for start in (0, 4):
        plot_page(plot_rows[start:start + 4], args.output / f"montage_{start // 4 + 1}.png", centered_limit)
    report = {
        "schema": "train_brow_source_audit_v1", "selection_sha256": sha(args.output / "selection.json"),
        "scope": "Sparse original source-frame/native-coefficient inspection. No generated faces, no GT correction, no independent perceptual or label-accuracy score.",
        "statistics_note": "Exact-repeat, saturation and first/second differences are descriptive, not automatic tracking failure or jitter classifications.",
        "centered_plot_shared_ylim": [-centered_limit, centered_limit],
        "clips": audits,
        "summary": {
            "clips": len(audits), "distinct_fit_people": len({r["speaker"] for r in audits}),
            "max_abs_native_video_duration_delta_s": max(abs(r["native_duration_s"] - r["video_clock"]["video_duration_s"]) for r in audits),
            "max_abs_sample_pts_error_s": max(abs(x) for r in audits for x in r["video_clock"]["target_to_source_time_error_s"]),
            "minimum_zero_lag_audio_correlation": min(r["audio_audit"]["zero_lag_waveform_correlation"] for r in audits),
            "all_audio_sample_counts_identical": all(r["audio_audit"]["native_samples"] == r["audio_audit"]["embedded_samples"] for r in audits),
            "all_coefficients_finite_within_zero_one": all(r["observed_outside_zero_one_count"] == 0 for r in audits),
            "native_over_raw_centered_brow_energy_range": [min(r["raw_processing_audit"]["native_over_raw_centered_brow_energy"] for r in audits), max(r["raw_processing_audit"]["native_over_raw_centered_brow_energy"] for r in audits)],
            "raw_native_centered_brow_correlation_range": [min(r["raw_processing_audit"]["raw_native_centered_brow_correlation"] for r in audits), max(r["raw_processing_audit"]["raw_native_centered_brow_correlation"] for r in audits)],
        },
    }
    save(args.output / "audit.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
