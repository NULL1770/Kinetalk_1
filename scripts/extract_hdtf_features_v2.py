"""Extract source-clock HDTF audio features for the v2 training windows.

The extractor deliberately ignores ``audios.zip``.  It reads the audited 16-kHz
waveform emitted by :mod:`kinetalk_v2.hdtf`, runs local HuBERT layer 6 and the
middle emotion2vec layers (2/4/6), and appends four native-clock prosody
channels.  Features are calculated in bounded chunks so a long interview does
not become one Transformer attention sequence.  Unknown HDTF emotion labels
remain masked; this script never creates an emotion target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = "hdtf-audio-features-v2.1"
SAMPLE_RATE = 16000
TARGET_FPS = 25
AUDIO_DIM = 1540


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def model_conv_geometry(model: Any) -> dict[str, Any]:
    """Derive unpadded convolution timing from a transformers model config."""
    cfg = getattr(model, "config", model)
    kernels = getattr(cfg, "conv_kernel", None) or getattr(cfg, "conv_kernel_sizes", None)
    strides = getattr(cfg, "conv_stride", None) or getattr(cfg, "conv_strides", None)
    if kernels is None or strides is None:
        kernels, strides = [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]
    kernels, strides = [int(x) for x in kernels], [int(x) for x in strides]
    if len(kernels) != len(strides) or not kernels:
        raise ValueError("HuBERT convolution geometry is malformed")
    hop, receptive = 1, 1
    for kernel, stride in zip(kernels, strides):
        receptive += (kernel - 1) * hop
        hop *= stride
    return {"sample_rate": SAMPLE_RATE, "hop_samples": hop,
            "receptive_samples": receptive,
            "center_offset_samples": (receptive - 1) / 2.,
            "frame_rate": SAMPLE_RATE / hop,
            "kernels": kernels, "strides": strides}


def expected_conv_length(samples: int, geometry: dict[str, Any]) -> int:
    length = int(samples)
    for kernel, stride in zip(geometry["kernels"], geometry["strides"]):
        length = (length - int(kernel)) // int(stride) + 1
    return max(length, 0)


def _feature_times(n: int, geometry: dict[str, Any]) -> np.ndarray:
    return (float(geometry["center_offset_samples"]) + np.arange(n) * int(geometry["hop_samples"])) / SAMPLE_RATE


def _audio_info(path: Path) -> tuple[int, int]:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return int(info.samplerate), int(info.frames)
    except ImportError:
        import wave as wave_module
        with wave_module.open(str(path), "rb") as stream:
            return int(stream.getframerate()), int(stream.getnframes())


def _read_wave(path: Path, start: int, count: int) -> np.ndarray:
    try:
        import soundfile as sf
        with sf.SoundFile(str(path), "r") as stream:
            if stream.samplerate != SAMPLE_RATE or stream.channels != 1:
                raise ValueError(f"Expected audited 16-kHz mono WAV: {path}")
            start = max(0, min(int(start), len(stream)))
            stream.seek(start)
            wave = stream.read(max(0, int(count)), dtype="float32", always_2d=False)
        wave = np.asarray(wave, dtype=np.float32)
    except ImportError:
        import wave as wave_module
        with wave_module.open(str(path), "rb") as stream:
            if stream.getframerate() != SAMPLE_RATE or stream.getnchannels() != 1 or stream.getsampwidth() != 2:
                raise ValueError(f"Expected audited 16-kHz PCM WAV: {path}")
            start = max(0, min(int(start), stream.getnframes())); stream.setpos(start)
            raw = stream.readframes(max(0, int(count)))
        wave = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.
    if wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError(f"Non-finite waveform: {path}")
    return wave


def prosody(wave: np.ndarray, targets: np.ndarray, valid: np.ndarray,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Native-clock [log-f0, log-rms, periodicity, voicing]."""
    wave = np.asarray(wave, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    result = np.zeros((len(targets), 4), dtype=np.float32)
    length = 640
    lo, hi = int(sample_rate / 500), int(sample_rate / 60)
    if not len(wave):
        return result
    for i, center_s in enumerate(targets):
        if not valid[i]:
            continue
        center = int(round(center_s * sample_rate))
        indices = np.arange(center - length // 2, center + length // 2)
        seg = wave[np.clip(indices, 0, len(wave) - 1)]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        tapered = (seg - seg.mean()) * np.hanning(length)
        acf = np.correlate(tapered, tapered, mode="full")[length - 1:]
        if acf[0] > 1e-12:
            win = np.hanning(length)
            acf = acf / np.maximum(np.correlate(win, win, mode="full")[length - 1:], 1e-10)
            acf = acf / max(float(acf[0]), 1e-12)
            peaks = [j for j in range(lo, min(hi, len(acf) - 2) + 1)
                     if acf[j] > acf[j - 1] and acf[j] >= acf[j + 1]]
            best = max((float(acf[j]) for j in peaks), default=0.)
            plausible = [j for j in peaks if acf[j] >= max(.65, best * .95)]
            lag = plausible[0] if plausible else (max(peaks, key=lambda j: acf[j]) if peaks else lo)
            confidence = float(np.clip(acf[lag], 0., 1.)) if peaks else 0.
        else:
            lag, confidence = lo, 0.
        voiced = rms > 1e-4 and confidence >= .65
        result[i] = [math.log(sample_rate / lag) if voiced else 0.,
                      math.log(max(rms, 1e-7)), confidence, float(voiced)]
    return result


@dataclass
class Backend:
    name: str
    model: Any
    geometry: dict[str, Any]
    device: str
    layer: int = 6
    emotion: bool = False
    processor: Any = None
    global_mean: float | None = None
    global_std: float | None = None

    def set_global_normalization(self, mean: float, variance: float) -> None:
        # Native emotion2vec F.layer_norm uses eps1e-5; HuBERT's numpy
        # feature-extractor normalization uses eps1e-7.
        epsilon = 1e-5 if self.emotion else 1e-7
        self.global_mean = float(mean)
        self.global_std = float(math.sqrt(max(float(variance), 0.) + epsilon))
        if self.processor is not None and hasattr(self.processor, "do_normalize"):
            # The audited MEAD extractor normalizes each complete recording.
            # Chunks are pre-normalized with those source statistics so the
            # processor must not normalize each bounded context independently.
            self.processor.do_normalize = False

    @property
    def dim(self) -> int:
        return 768

    def extract(self, wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return [frames,768] and seconds relative to the chunk start."""
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            raw = np.asarray(wave, dtype=np.float32)
            if self.global_mean is not None and self.global_std is not None:
                raw = (raw - self.global_mean) / self.global_std
            x = torch.from_numpy(raw).to(self.device)
            if self.emotion:
                captures: dict[int, torch.Tensor] = {}
                hooks = []
                for number in (2, 4, 6):
                    block = self.model.blocks[number - 1]
                    hooks.append(block.register_forward_hook(
                        lambda _m, _i, output, key=number:
                        captures.__setitem__(key, output[0].detach() if isinstance(output, tuple) else output.detach())))
                try:
                    normalized = x if self.global_mean is not None else F.layer_norm(x, x.shape)
                    out = self.model.extract_features(normalized[None], mask=False,
                                                      remove_extra_tokens=True)
                finally:
                    for hook in hooks:
                        hook.remove()
                extra = int(self.model.modality_encoders["AUDIO"].modality_cfg.num_extra_tokens)
                n = len(out['x'][0])
                middle = []
                for number in (2, 4, 6):
                    if captures[number].shape != (1, n + extra, 768):
                        raise ValueError(f"Unexpected emotion2vec intermediate layout {captures[number].shape}")
                    value = captures[number][:, extra:extra + n]
                    middle.append(F.layer_norm(value, (768,)))
                feat = torch.stack(middle).mean(0)[0]
                if feat.shape[-1] != 768:
                    raise ValueError(f"emotion2vec output width {feat.shape}")
                arr = feat.float().cpu().numpy()
                # ``out`` is intentionally unused; hooks are the defined middle-layer path.
            else:
                # Exact native MEAD path: Wav2Vec2FeatureExtractor performs
                # waveform normalization before HuBERT. Input raw x directly
                # would silently change the content feature distribution.
                inputs = self.processor(raw, sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt", padding=False)
                attention = getattr(inputs, "attention_mask", None)
                out = self.model(inputs.input_values.to(self.device),
                                 attention_mask=attention.to(self.device) if attention is not None else None,
                                 output_hidden_states=True)
                hidden = out.hidden_states
                if hidden is None or len(hidden) <= self.layer:
                    raise ValueError("HuBERT did not return the requested hidden state")
                arr = hidden[self.layer][0].float().cpu().numpy()
                if arr.ndim != 2 or arr.shape[-1] != 768:
                    raise ValueError(f"HuBERT output shape {arr.shape}")
        return arr, _feature_times(len(arr), self.geometry)


def load_hubert(model_dir: Path, device: str, layer: int = 6) -> tuple[Backend, dict]:
    import torch
    import transformers
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    model = HubertModel.from_pretrained(str(model_dir), local_files_only=True,
                                       use_safetensors=True).to(device)
    model.eval().requires_grad_(False)
    geometry = model_conv_geometry(model)
    if getattr(model.config, "hidden_size", 768) != 768:
        raise ValueError("HDTF HuBERT contract requires 768 dimensions")
    assets = sorted(p for p in model_dir.iterdir() if p.is_file() and
                    (p.suffix in {".safetensors", ".json"} or p.name == "pytorch_model.bin"))
    if not any(p.suffix == ".safetensors" for p in assets):
        raise ValueError("Native HuBERT must use the audited local safetensors asset")
    info = {"type": "hubert", "model_dir": str(model_dir.resolve()),
            "assets_sha256": {p.name: sha256(p) for p in assets},
            "layer": layer, "geometry": geometry,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "waveform_normalization": "source-global mean/population-variance eps1e-7; local processor do_normalize=False after explicit normalization",
            "native_mead_recipe": "D:/实验室项目/新实验/scripts/extract_native_hubert.py layer6"}
    return Backend("hubert_layer6", model, geometry, device, layer=layer, processor=processor), info


def load_emotion2vec(model_dir: Path, device: str) -> tuple[Backend, dict]:
    from scripts.extract_emotion2vec_pilot import load_extractor
    model, geometry, info = load_extractor(model_dir, device)
    return Backend("emotion2vec_middle246", model, geometry, device, emotion=True), info


def interpolate(features: np.ndarray, feature_times: np.ndarray, targets: np.ndarray,
                valid: np.ndarray, duration: float, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a chunk and mark justified convolution boundary extension."""
    if features.ndim != 2 or features.shape[1] != 768 or len(features) < 2:
        raise ValueError(f"Feature shape is {features.shape}, expected [N,768]")
    if len(feature_times) != len(features) or not np.all(np.diff(feature_times) > 0):
        raise ValueError("Feature clock is not strictly increasing")
    targets, valid = np.asarray(targets, float), np.asarray(valid, bool)
    out = np.zeros((len(targets), 768), dtype=np.float32)
    ext = np.zeros(len(targets), dtype=bool)
    if not len(targets):
        return out, ext
    if np.any(valid & ((targets < -1e-7) | (targets > duration + 1e-7))):
        raise ValueError("Audio target falls outside waveform clock")
    max_ext = ((float(geometry["receptive_samples"]) - 1) / 2. + int(geometry["hop_samples"])) / SAMPLE_RATE
    use = valid & (targets >= feature_times[0] - max_ext) & (targets <= feature_times[-1] + max_ext)
    if np.any(valid & ~use):
        raise ValueError("Target lies beyond justified feature boundary")
    clipped = np.clip(targets, feature_times[0], feature_times[-1])
    right = np.searchsorted(feature_times, clipped, side="right").clip(1, len(features) - 1)
    left = right - 1
    alpha = (clipped - feature_times[left]) / np.maximum(feature_times[right] - feature_times[left], 1e-12)
    out[:] = features[left] * (1 - alpha[:, None]) + features[right] * alpha[:, None]
    out[~valid] = 0.
    ext = use & ((targets < feature_times[0]) | (targets > feature_times[-1]))
    return out, ext


def chunked_prosody(path: Path, target_rel: np.ndarray, valid: np.ndarray, duration: float,
                    block_seconds: float = 60.) -> np.ndarray:
    """Compute four prosody channels without loading a multi-hour waveform."""
    out = np.zeros((len(target_rel), 4), dtype=np.float32)
    for start in np.arange(0., max(duration, 1e-9), block_seconds):
        end = min(duration, float(start + block_seconds))
        sel = valid & (target_rel >= start - 1e-9) & (target_rel < end - 1e-9)
        if not np.any(sel):
            continue
        sample_start = max(0, int(round((start - .04) * SAMPLE_RATE)))
        sample_end = min(int(round(duration * SAMPLE_RATE)), int(round((end + .04) * SAMPLE_RATE)))
        wave = _read_wave(path, sample_start, sample_end - sample_start)
        out[sel] = prosody(wave, target_rel[sel] - sample_start / SAMPLE_RATE,
                            np.ones(int(sel.sum()), dtype=bool))
    return out


def chunked_backend(path: Path, target_rel: np.ndarray, valid: np.ndarray, backend: Backend,
                    chunk_seconds: float, duration: float, context_seconds: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """Run one frozen backend per bounded chunk and return target-aligned features."""
    if chunk_seconds < .5:
        raise ValueError("chunk_seconds must be at least 0.5")
    result = np.zeros((len(target_rel), 768), dtype=np.float32)
    extended = np.zeros(len(target_rel), dtype=bool)
    valid = np.asarray(valid, bool)
    if not len(target_rel):
        return result, extended
    # A fixed core is used so every target is assigned once; the model sees no
    # unbounded context and source timing remains unchanged.
    starts = np.arange(0., max(duration, 1e-9), chunk_seconds)
    for start in starts:
        end = min(duration, float(start + chunk_seconds))
        sel = valid & (target_rel >= start - 1e-9) & (target_rel < end - 1e-9)
        if not np.any(sel):
            continue
        if context_seconds < 0.:
            raise ValueError("context_seconds must be non-negative")
        pad = max(int(round(context_seconds * SAMPLE_RATE)),
                  int(round(float(backend.geometry["receptive_samples"]) / 2.)))
        sample_start = max(0, int(round(start * SAMPLE_RATE)) - pad)
        sample_end = min(int(round(duration * SAMPLE_RATE)), int(round(end * SAMPLE_RATE)) + pad)
        wave = _read_wave(path, sample_start, sample_end - sample_start)
        # Backend times are relative to sample_start; target is relative to that
        # same sample clock, not to the source's absolute timestamp.
        local_targets = target_rel[sel] - sample_start / SAMPLE_RATE
        arr, times = backend.extract(wave)
        aligned, ext = interpolate(arr, times, local_targets, np.ones(len(local_targets), bool),
                                   len(wave) / SAMPLE_RATE, backend.geometry)
        result[sel] = aligned
        extended[sel] = ext
    result[~valid] = 0.
    return result, extended


def source_rows(manifest: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    assignments: dict[str, str] = {}
    speaker_splits: dict[str, set[str]] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("source_id", ""))
            split = row.get("split")
            group = str(row.get("speaker_group", f"unknown:{sid}"))
            if sid in assignments and assignments[sid] != split:
                raise ValueError(f"source appears in multiple splits: {sid}")
            assignments[sid] = split
            speaker_splits.setdefault(group, set()).add(split)
            if row.get("split") not in {"train", "val"}:
                continue
            if not sid:
                continue
            # A source is accepted only when every row agrees on paths and the
            # source-level clock. This catches accidental partial/absolute path
            # mixing before a costly model call.
            candidate = {"source_id": sid, "split": row["split"],
                         "speaker_group": row.get("speaker_group", f"unknown:{sid}"),
                         "source_group": row.get("source_group", sid),
                         "motion_path": row.get("motion_path"), "audio_path": row.get("audio_path"),
                         "processed_source_start_seconds": float(row.get("processed_source_start_seconds",
                             float(row.get("start_seconds", 0.)) - int(row.get("start_frame", 0)) / TARGET_FPS)),
                         "windows": []}
            if sid not in rows:
                rows[sid] = candidate
            old = rows[sid]
            for key in ("split", "motion_path", "audio_path", "processed_source_start_seconds"):
                if old[key] != candidate[key]:
                    raise ValueError(f"inconsistent source manifest {sid}: {key}")
            old["windows"].append(row)
    for group, splits in speaker_splits.items():
        if len(splits) > 1 and splits & {"train", "val"}:
            raise ValueError(f"speaker group appears in multiple splits: {group}")
    return rows


def extract_source(rec: dict, output_root: Path, hubert: Backend, emotion: Backend,
                   provenance: dict, chunk_seconds: float, force: bool = False,
                   context_seconds: float = 1.) -> dict:
    source_id = rec["source_id"]
    if rec.get("split") not in {"train", "val"}:
        raise ValueError("feature extraction is train/validation only")
    motion_path, audio_path = Path(rec["motion_path"]), Path(rec["audio_path"])
    if not motion_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError(f"missing HDTF source artifacts for {source_id}")
    out_path = output_root / "features" / f"{source_id}.npz"
    receipt_path = output_root / "receipts" / f"{source_id}.json"
    audio_hash = sha256(audio_path)
    motion_hash = sha256(motion_path)
    signature = hashlib.sha256(json.dumps({"schema": SCHEMA, "source_id": source_id,
        "audio_sha256": audio_hash, "motion_sha256": motion_hash,
        "split": rec["split"], "speaker_group": rec["speaker_group"],
        "source_start_seconds": rec["processed_source_start_seconds"],
        "provenance": provenance, "chunk_seconds": chunk_seconds,
        "context_seconds": context_seconds}, sort_keys=True).encode()).hexdigest()
    if not force and receipt_path.is_file() and out_path.is_file():
        try:
            done = json.loads(receipt_path.read_text(encoding="utf-8"))
            if done.get("signature") == signature and sha256(out_path) == done.get("features_sha256"):
                return dict(done, status="cached")
        except (OSError, ValueError, KeyError):
            pass
    with np.load(motion_path, allow_pickle=False) as m:
        if "times" not in m or "motion" not in m:
            raise ValueError(f"invalid source motion artifact {motion_path}")
        times = np.asarray(m["times"], dtype=np.float64)
        motion = np.asarray(m["motion"])
        motion_valid = np.asarray(m["motion_valid"], bool) if "motion_valid" in m else np.ones(len(times), bool)
        channel_mask = np.asarray(m["channel_mask"], bool) if "channel_mask" in m else np.ones((len(times), 52), bool)
        if motion.shape != (len(times), 52) or not np.all(np.diff(times) > 0):
            raise ValueError(f"invalid source motion clock {motion_path}")
        source_start = float(m.get("source_start_seconds", np.asarray(times[0])).item())
        if not np.isclose(source_start, rec["processed_source_start_seconds"], atol=1e-7):
            raise ValueError("manifest and source NPZ clocks disagree")
    sample_rate, audio_n = _audio_info(audio_path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError("source audio must be 16-kHz mono")
    duration = audio_n / SAMPLE_RATE
    # Fit no statistics from motion, labels or the test split. This is only
    # the deterministic full-source waveform normalization used by the native
    # MEAD HuBERT recipe, computed by a streaming pass over this WAV.
    sum_x = sum_x2 = 0.0; count_x = 0
    for block_start in range(0, audio_n, SAMPLE_RATE * 60):
        block = _read_wave(audio_path, block_start, min(SAMPLE_RATE * 60, audio_n - block_start)).astype(np.float64)
        sum_x += float(block.sum()); sum_x2 += float(np.square(block).sum()); count_x += len(block)
    global_mean = sum_x / max(count_x, 1)
    global_variance = max(sum_x2 / max(count_x, 1) - global_mean * global_mean, 0.)
    for backend in (hubert, emotion):
        if hasattr(backend, "set_global_normalization"):
            backend.set_global_normalization(global_mean, global_variance)
    target_rel = times - source_start
    audio_valid = np.isfinite(target_rel) & (target_rel >= 0.) & (target_rel < duration - 1e-8)
    # Features stay available across a failed face frame. Motion quality is a
    # separate supervision mask; audio observations are not fabricated zeros.
    valid = audio_valid
    h, h_ext = chunked_backend(audio_path, target_rel, valid, hubert, chunk_seconds, duration, context_seconds)
    e, e_ext = chunked_backend(audio_path, target_rel, valid, emotion, chunk_seconds, duration, context_seconds)
    p = chunked_prosody(audio_path, target_rel, valid, duration)
    features = np.concatenate([h, e, p], axis=1)
    if features.shape != (len(times), AUDIO_DIM) or not np.isfinite(features).all():
        raise ValueError(f"invalid assembled feature shape {features.shape}")
    feature_valid = valid & ~np.any(~np.isfinite(features), axis=1)
    features[~feature_valid] = 0.
    arrays = {"audio_features": features.astype(np.float16), "times": times,
              "feature_valid": feature_valid, "motion_valid": motion_valid,
              "channel_mask": channel_mask, "source_start_seconds": np.float64(source_start),
              "audio_seconds": np.float64(duration), "hubert_boundary_extended": h_ext,
              "emotion_boundary_extended": e_ext, "schema": np.asarray(SCHEMA)}
    atomic_npz(out_path, arrays)
    done = {"schema": SCHEMA, "status": "ok", "source_id": source_id,
            "split": rec["split"], "speaker_group": rec["speaker_group"],
            "motion_path": str(motion_path.resolve()), "audio_path": str(audio_path.resolve()),
            "motion_sha256": motion_hash, "audio_sha256": audio_hash,
            "features_path": str(out_path.resolve()), "features_sha256": sha256(out_path),
            "signature": signature, "n_frames": len(times), "feature_valid_ratio": float(feature_valid.mean()),
            "boundary_extended_ratio": float((h_ext | e_ext).mean()),
            "source_start_seconds": source_start, "audio_seconds": duration,
            "waveform_normalization": {"mean": global_mean, "population_variance": global_variance,
                                       "hubert_epsilon": 1e-7, "emotion2vec_epsilon": 1e-5,
                                       "scope": "one source recording; no dataset-fitted statistics"},
            "feature_layout": {"hubert_layer6": [0, 768], "emotion2vec_middle246": [768, 1536], "prosody": [1536, 1540]},
            "provenance": provenance, "updated_unix": time.time()}
    atomic_json(receipt_path, done)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True,
                    help="HDTF_processed_v2 root containing training_windows.jsonl")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--hubert-model", type=Path, required=True)
    ap.add_argument("--emotion2vec-model", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-seconds", type=float, default=6.)
    ap.add_argument("--context-seconds", type=float, default=1.)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source-ids", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    manifest = args.manifest or (args.input_root / "training_windows.jsonl")
    if not manifest.is_file():
        ap.error(f"manifest does not exist: {manifest}")
    rows = source_rows(manifest)
    if args.source_ids:
        requested = {x for x in args.source_ids.split(",") if x}
        rows = {sid: row for sid, row in rows.items() if sid in requested}
    selected = sorted(rows.values(), key=lambda r: r["source_id"])
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        ap.error("no train/val HDTF sources in manifest")
    if args.dry_run:
        print(json.dumps({"schema": SCHEMA, "sources": len(selected),
                          "splits": {s: sum(r["split"] == s for r in selected) for s in ("train", "val")},
                          "windows": sum(len(r["windows"]) for r in selected)}, ensure_ascii=False))
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    hubert, hubert_info = load_hubert(args.hubert_model, args.device)
    emotion, emotion_info = load_emotion2vec(args.emotion2vec_model, args.device)
    provenance = {"schema": SCHEMA, "hubert": hubert_info, "emotion2vec": emotion_info,
                  "sample_rate": SAMPLE_RATE, "target_fps": TARGET_FPS,
                  "chunk_seconds": args.chunk_seconds,
                  "context_seconds": args.context_seconds,
                  "context_policy": "6-second fixed core + 1-second left/right context by default; crop aligned cores only; offline bounded attention",
                  "normalization_context": "source-global waveform mean/std computed streaming; reused in every bounded model context, matching native full-recording normalization",
                  "clock": "source motion times minus source_start_seconds; ffmpeg WAV starts at zero",
                  "audio_zip_used": False, "emotion_supervision": "unknown_hdtf_masked",
                  "script_sha256": sha256(Path(__file__).resolve())}
    atomic_json(args.output_root / "run_binding.json", {"schema": SCHEMA, "provenance": provenance,
                "manifest": str(manifest.resolve()), "source_count": len(selected), "started_unix": time.time()})
    done = []
    for i, rec in enumerate(selected, 1):
        try:
            result = extract_source(rec, args.output_root, hubert, emotion, provenance, args.chunk_seconds,
                                    args.force, args.context_seconds)
        except Exception as exc:
            result = {"source_id": rec["source_id"], "split": rec["split"], "status": "error",
                      "error": f"{type(exc).__name__}: {exc}"[:1000]}
        done.append(result)
        atomic_json(args.output_root / "status.json", {"schema": SCHEMA, "completed": i,
                    "requested": len(selected), "status_counts": {s: sum(r.get("status") == s for r in done) for s in ("ok", "cached", "error")},
                    "last": result, "updated_unix": time.time()})
        print(json.dumps({"done": i, "total": len(selected), "source_id": rec["source_id"],
                          "status": result.get("status"), "valid_ratio": result.get("feature_valid_ratio"),
                          "error": result.get("error")}, ensure_ascii=False), flush=True)
    with (args.output_root / "feature_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in done:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return 1 if any(r.get("status") == "error" for r in done) else 0


if __name__ == "__main__":
    raise SystemExit(main())
