"""HDTF source-level ARKit extraction and conservative training-window QC.

MediaPipe outputs are pseudo labels in the existing MEAD extractor's semantics,
not newly fitted vertex truth. Raw values and source clocks are retained; neither
interpolation nor temporal smoothing is applied. TongueOut is unobserved.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SCHEMA = "hdtf-source-arkit-v2.1"
NO_FACE, MULTIFACE, POSE, SMALL_FACE, TRACK_JUMP, MISSING_POSE, BEFORE_VIDEO = 1, 2, 4, 8, 16, 32, 64
ALIASES = {"whipjimclyburn": "jamesclyburn"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def identity_group(stem: str) -> tuple[str, str, str]:
    """Merge recording suffix aliases before split; unresolved Radio IDs quarantine.

    Named IDs remain filename-based hypotheses, explicitly not face-verified IDs.
    Prefix differences cannot put the same named person into separate splits.
    """
    match = re.fullmatch(r"([^_]+)_(.+)_([0-9]+)", stem)
    if not match:
        return f"unresolved:{stem.lower()}", "quarantine", "unparsed_filename"
    prefix, speaker, _ = match.groups()
    if prefix.upper() == "RD" or speaker.lower().startswith("radio"):
        return f"unresolved-radio:{speaker.lower()}", "quarantine", "radio_identity_unverified"
    name = re.sub(r"[0-9]+$", "", speaker)
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    name = ALIASES.get(name, name)
    if not name or prefix.upper() not in {"WDA", "WRA"}:
        return f"unresolved:{stem.lower()}", "quarantine", "unknown_identity_namespace"
    return name, "named_filename", "suffix_merged_not_face_verified"


def split_for_group(group: str, identity_status: str, seed: int = 47) -> str:
    if identity_status == "quarantine":
        return "quarantine"
    u = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if u < .8 else ("val" if u < .9 else "test")


def probe_video(path: Path, ffprobe: str = "ffprobe") -> dict:
    proc = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format",
                           "-of", "json", str(path)], capture_output=True,
                          encoding="utf-8", errors="replace", timeout=120)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-800:])
    info = json.loads(proc.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise ValueError("no video stream")
    numerator, denominator = v.get("avg_frame_rate", "0/1").split("/")
    fps = float(numerator) / max(float(denominator), 1)
    return {"duration": float(v.get("duration", info["format"].get("duration", 0))),
            "width": int(v["width"]), "height": int(v["height"]), "source_fps": fps,
            "video_start": float(v.get("start_time", 0)),
            "audio_start": float(a.get("start_time", 0)) if a else None,
            "format_start": float(info["format"].get("start_time", 0)),
            "has_audio": a is not None, "audio_sample_rate": int(a.get("sample_rate", 0)) if a else 0}


def inventory_one(path: Path, seed: int, ffprobe: str = "ffprobe") -> dict:
    group, identity_status, identity_note = identity_group(path.stem)
    st = path.stat()
    out = {"source_id": path.stem, "video": str(path.resolve()), "dataset": "HDTF",
           "source_group": path.stem, "speaker_group": group,
           "identity_status": identity_status, "identity_note": identity_note,
           "split": split_for_group(group, identity_status, seed), "split_seed": seed,
           "emotion_id": -1, "emotion_valid": False, "intensity_id": -1,
           "intensity_valid": False, "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns}
    try:
        out.update(probe_video(path, ffprobe))
        out["inventory_status"] = "ok" if out["has_audio"] else "no_audio"
    except Exception as exc:
        out.update(inventory_status="probe_error", error=str(exc)[:600])
    return out


def channel_gather(rig_channels: list[str], mp_channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lower = {name.lower(): i for i, name in enumerate(mp_channels)}
    if len(lower) != len(mp_channels):
        raise ValueError("duplicate MediaPipe channel names")
    if len(rig_channels) != 52 or len({s.lower() for s in rig_channels}) != 52:
        raise ValueError("rig must define exactly 52 unique named channels")
    idx = np.array([lower.get(name.lower(), -1) for name in rig_channels], dtype=np.int64)
    missing = {rig_channels[i].lower() for i in np.flatnonzero(idx < 0)}
    if missing != {"tongueout"}:
        raise ValueError(f"unexpected unavailable rig channels: {sorted(missing)}")
    return idx, idx >= 0


def head_pose(mat: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(mat, dtype=np.float64)[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    p = np.arcsin(np.clip(-r[1, 2], -1, 1))
    if abs(np.sin(p)) < .99999:
        y, z = np.arctan2(r[0, 2], r[2, 2]), np.arctan2(r[1, 0], r[1, 1])
    else:
        y, z = np.arctan2(-r[2, 0], r[0, 0]), 0.
    return np.degrees([y, p, z]).astype(np.float32)


@dataclass(frozen=True)
class ExtractOptions:
    model: str
    rig: str
    output: str
    fps: int = 25
    sample_rate: int = 16000
    max_side: int = 640
    min_face_px: float = 64.
    max_yaw: float = 40.
    max_pitch: float = 35.
    max_roll: float = 35.
    min_detection: float = .5
    min_presence: float = .5
    min_tracking: float = .5
    start_seconds: float = 0.
    max_seconds: float = 0.
    window_frames: int = 128
    stride_frames: int = 128
    context_frames: int = 32
    min_window_quality: float = .95
    max_gap_frames: int = 2
    ffmpeg: str = "ffmpeg"
    timeout_seconds: int = 21600


def options_signature(opts: ExtractOptions) -> tuple[str, dict]:
    payload = {"schema": SCHEMA, "options": asdict(opts), "model_sha256": sha256(Path(opts.model)),
               "rig_sha256": sha256(Path(opts.rig)), "implementation_sha256": sha256(Path(__file__)),
               "versions": {name: importlib.metadata.version(name) for name in ("mediapipe", "numpy", "opencv-python")}}
    # Location is not a mathematical preprocessing parameter.
    payload["options"].pop("output")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def iter_rgb(source: Path, opts: ExtractOptions, duration: float) -> Iterator[np.ndarray]:
    width, height = 0, 0
    # Probe has already succeeded, but independently derive fixed pipe dimensions.
    import cv2
    cap = cv2.VideoCapture(str(source))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    factor = min(1., opts.max_side / max(width, height))
    width, height = max(2, int(width * factor) // 2 * 2), max(2, int(height * factor) // 2 * 2)
    cmd = [opts.ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(source),
           "-ss", str(opts.start_seconds), "-t", str(duration), "-an", "-sn", "-dn",
           "-vf", f"fps={opts.fps}:start_time=0,scale={width}:{height}",
           "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        nbytes = width * height * 3
        start = time.monotonic()
        completed = False
        try:
            while True:
                if time.monotonic() - start > opts.timeout_seconds:
                    raise TimeoutError("source extraction hard timeout")
                raw = proc.stdout.read(nbytes)
                if not raw:
                    break
                if len(raw) != nbytes:
                    raise RuntimeError("truncated ffmpeg raw frame")
                yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            completed = True
        finally:
            proc.stdout.close()
            if not completed and proc.poll() is None:
                proc.kill()
            code = proc.wait(timeout=30)
            err.seek(0)
            message = err.read().decode("utf-8", errors="replace")
            if completed and code:
                raise RuntimeError(f"ffmpeg video failed: {message[-800:]}")


def extract_wav(source: Path, out: Path, opts: ExtractOptions, duration: float) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp.wav")
    cmd = [opts.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(source),
           "-ss", str(opts.start_seconds), "-t", str(duration), "-vn", "-ac", "1", "-ar", str(opts.sample_rate),
           "-af", f"aresample={opts.sample_rate}:async=1:first_pts=0", "-c:a", "pcm_s16le", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, timeout=opts.timeout_seconds)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg audio failed: {proc.stderr.decode(errors='replace')[-800:]}")
    with wave.open(str(tmp), "rb") as f:
        if f.getframerate() != opts.sample_rate or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError("incorrect waveform contract")
        samples = f.getnframes()
    os.replace(tmp, out)
    return {"audio_samples": samples, "audio_seconds": samples / opts.sample_rate,
            "audio_sha256": sha256(out)}


def max_false_run(mask: np.ndarray) -> int:
    padded = np.r_[True, np.asarray(mask, bool), True]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max(initial=0))


def training_windows(record: dict, quality: np.ndarray, flags: np.ndarray, opts: ExtractOptions) -> list[dict]:
    out = []
    n = len(quality)
    for start in range(0, max(n - opts.window_frames + 1, 0), opts.stride_frames):
        end = start + opts.window_frames
        q, f = quality[start:end], flags[start:end]
        if float(q.mean()) < opts.min_window_quality or max_false_run(q) > opts.max_gap_frames:
            continue
        if np.any(f & (MULTIFACE | TRACK_JUMP)):
            continue
        out.append({"clip_id": f"{record['source_id']}:{start}:{end}", "source_id": record["source_id"],
                    "source_group": record["source_group"], "speaker_group": record["speaker_group"],
                    "identity_status": record["identity_status"], "split": record["split"], "dataset": "HDTF",
                    "motion_path": record["motion_path"], "audio_path": record["audio_path"],
                    "start_frame": start, "end_frame": end, "start_seconds": opts.start_seconds + start / opts.fps,
                    "audio_offset_seconds": start / opts.fps,
                    "end_seconds": opts.start_seconds + end / opts.fps, "quality_ratio": float(q.mean()),
                    "context_start_frame": max(0, start - opts.context_frames),
                    "context_end_frame": min(n, end + opts.context_frames),
                    "left_source_boundary": start == 0, "right_source_boundary": end == n,
                    "processed_source_start_seconds": opts.start_seconds,
                    "partial_source_processing": opts.start_seconds > 0 or opts.max_seconds > 0,
                    "event_annotations_require_source_context": True,
                    "emotion_id": -1, "emotion_valid": False, "intensity_id": -1, "intensity_valid": False})
    return out


def verify_cached(done_path: Path, source: Path, signature: str) -> dict | None:
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        st = source.stat()
        if done["signature"] != signature or done["source_size"] != st.st_size or done["source_mtime_ns"] != st.st_mtime_ns:
            return None
        if sha256(source) != done["source_sha256"]:
            return None
        motion, audio = Path(done["motion_path"]), Path(done["audio_path"])
        if not motion.exists() or not audio.exists() or sha256(motion) != done["motion_sha256"] or sha256(audio) != done["audio_sha256"]:
            return None
        windows = Path(done["windows_path"])
        if not windows.exists() or sha256(windows) != done["windows_sha256"]:
            return None
        with np.load(motion, allow_pickle=False) as data:
            if data["motion"].shape != (done["n_frames"], 52) or not np.all(np.diff(data["times"]) > 0):
                return None
        return dict(done, processing_status="cached")
    except (OSError, ValueError, KeyError):
        return None


def process_source(rec: dict, opts: ExtractOptions, signature: str, provenance: dict) -> dict:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    started = time.monotonic()
    source = Path(rec["video"])
    root = Path(opts.output)
    done_path = root / "sources" / f"{rec['source_id']}.json"
    cached = verify_cached(done_path, source, signature)
    if cached:
        return cached
    motion_path = root / "motion_raw" / f"{rec['source_id']}.npz"
    audio_path = root / "audio16k" / f"{rec['source_id']}.wav"
    duration = rec["duration"] - opts.start_seconds
    if opts.max_seconds:
        duration = min(duration, opts.max_seconds)
    if duration <= 0 or not rec["has_audio"]:
        raise ValueError("source window lacks audio/video duration")
    with np.load(opts.rig, allow_pickle=False) as rig:
        channels = [str(s) for s in rig["channels"]]
    # MediaPipe's Windows C++ file loader cannot reliably open non-ASCII paths;
    # Python's Unicode-aware read preserves the exact hashed task bytes.
    lm_opts = vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_buffer=Path(opts.model).read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_faces=2,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                min_face_detection_confidence=opts.min_detection,
                min_face_presence_confidence=opts.min_presence, min_tracking_confidence=opts.min_tracking)
    coeffs, poses, boxes, face_counts, flags = [], [], [], [], []
    gather, channel_mask = None, np.array([s.lower() != "tongueout" for s in channels])
    previous_box = None
    with vision.FaceLandmarker.create_from_options(lm_opts) as landmarker:
        for idx, rgb in enumerate(iter_rgb(source, opts, duration)):
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(idx * 1000 / opts.fps))
            count = len(result.face_landmarks)
            f = MULTIFACE if count > 1 else 0
            # fps:start_time=0 can duplicate the first frame to cover a .04s
            # video preroll. Keep the shared audio clock, mask synthetic preroll.
            if opts.start_seconds + idx / opts.fps < rec.get("video_start", 0) - 1e-6:
                f |= BEFORE_VIDEO
            motion, pose, box = np.zeros(52, np.float32), np.full(3, np.nan, np.float32), np.full(4, np.nan, np.float32)
            if not count or not result.face_blendshapes:
                f |= NO_FACE
                previous_box = None
            else:
                bbs = []
                for landmarks in result.face_landmarks:
                    xy = np.array([[p.x, p.y] for p in landmarks], dtype=np.float32)
                    bbs.append(np.r_[xy.min(0), xy.max(0)])
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bbs]
                main = int(np.argmax(areas))
                box = bbs[main]
                cats = result.face_blendshapes[main]
                idxmap, mask = channel_gather(channels, [c.category_name for c in cats])
                if gather is None:
                    gather, channel_mask = idxmap, mask
                elif not np.array_equal(gather, idxmap):
                    raise ValueError("MediaPipe channel order changed within source")
                scores = np.array([c.score for c in cats], np.float32)
                motion[channel_mask] = scores[gather[channel_mask]]
                if len(result.facial_transformation_matrixes) > main:
                    pose = head_pose(np.asarray(result.facial_transformation_matrixes[main]))
                    if abs(pose[0]) > opts.max_yaw or abs(pose[1]) > opts.max_pitch or abs(pose[2]) > opts.max_roll:
                        f |= POSE
                else:
                    f |= MISSING_POSE
                if min((box[2] - box[0]) * rgb.shape[1], (box[3] - box[1]) * rgb.shape[0]) < opts.min_face_px:
                    f |= SMALL_FACE
                if previous_box is not None:
                    prev_size = np.maximum(previous_box[2:] - previous_box[:2], .001)
                    size = np.maximum(box[2:] - box[:2], .001)
                    movement = np.linalg.norm((box[:2] + box[2:] - previous_box[:2] - previous_box[2:]) / 2)
                    area_ratio = float(np.prod(size) / np.prod(prev_size))
                    if movement > .35 * np.linalg.norm(prev_size) or not (1 / 2.2 <= area_ratio <= 2.2):
                        f |= TRACK_JUMP
                previous_box = box.copy() if count == 1 else None
            coeffs.append(motion)
            poses.append(pose)
            boxes.append(box)
            face_counts.append(count)
            flags.append(f)
            if idx % 250 == 0:
                atomic_json(root / "active" / f"{rec['source_id']}.json",
                            {"source_id": rec["source_id"], "pid": os.getpid(), "frames": idx + 1,
                             "elapsed_seconds": time.monotonic() - started, "updated_unix": time.time()})
    if not coeffs or gather is None:
        raise ValueError("no frames or no detectable face in source")
    motion = np.stack(coeffs)
    quality_flags = np.asarray(flags, np.uint16)
    observed = (quality_flags & (NO_FACE | BEFORE_VIDEO)) == 0
    quality = quality_flags == 0
    times = opts.start_seconds + np.arange(len(motion), dtype=np.float64) / opts.fps
    segment_ids = np.cumsum(np.r_[False, ~quality[:-1] | ~quality[1:]]).astype(np.int32)
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = motion_path.with_name(f"{motion_path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, motion=motion, coeffs=motion, times=times, valid=observed,
                            motion_valid=quality, quality_flags=quality_flags, channel_mask=channel_mask,
                            head_pose=np.stack(poses), face_bbox=np.stack(boxes), face_count=np.asarray(face_counts, np.uint8),
                            segment_id=segment_ids, channels=np.asarray(channels), fps=np.float64(opts.fps),
                            source_start_seconds=np.float64(opts.start_seconds), emotion_id=np.int64(-1),
                            emotion_valid=np.bool_(False), schema=np.asarray(SCHEMA))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, motion_path)
    audio_info = extract_wav(source, audio_path, opts, duration)
    clock_error = abs(audio_info["audio_seconds"] - len(motion) / opts.fps)
    clock_ok = clock_error <= 2 / opts.fps
    done = dict(rec, **audio_info, motion_path=str(motion_path.resolve()), audio_path=str(audio_path.resolve()),
                schema=SCHEMA, signature=signature, provenance=provenance,
                source_sha256=sha256(source), motion_sha256=sha256(motion_path), n_frames=len(motion),
                processed_start_seconds=opts.start_seconds, processed_seconds=len(motion) / opts.fps,
                detection_ratio=float(observed.mean()), quality_ratio=float(quality.mean()),
                multiface_ratio=float(((quality_flags & MULTIFACE) > 0).mean()),
                max_invalid_run=max_false_run(quality), audio_motion_duration_error=clock_error,
                clock_status="duration_consistent_sync_unverified" if clock_ok else "duration_mismatch_quarantine",
                processing_status="ok", elapsed_seconds=time.monotonic() - started,
                frames_per_second=len(motion) / max(time.monotonic() - started, .001),
                quality_flag_counts={str(bit): int(((quality_flags & bit) > 0).sum()) for bit in (1, 2, 4, 8, 16, 32, 64)})
    if not clock_ok:
        done["split"] = "quarantine"
    wins = training_windows(done, quality, quality_flags, opts)
    windows_path = root / "windows_by_source" / f"{rec['source_id']}.jsonl"
    atomic_jsonl(windows_path, wins)
    done["windows_path"] = str(windows_path.resolve())
    done["windows_sha256"] = sha256(windows_path)
    done["n_quality_windows"] = len(wins)
    done["n_training_eligible_windows"] = len(wins) if done["split"] != "quarantine" else 0
    atomic_json(done_path, done)  # Completion receipt is written last.
    return done
