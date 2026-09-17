"""Replace query audio with frozen emotion2vec frame features, on native times.

Use an isolated environment with funasr==1.4.15, modelscope==1.40.0 and the
existing PyTorch build. The official emotion2vec_plus_base model is 768-D at
50 Hz; this script verifies the actual convolution geometry rather than
assuming feature zero coincides with waveform zero. Identity references,
content, motion, timestamps and validity remain unchanged. ModelScope 1.40.0
also requires modelscope-hub>=0.3.0 for downloads. For locked audit extraction,
pass --reuse-feature-data ORIGINAL_EMOTION2VEC_DATA to copy the converted
training cache byte for byte, extract only audit heldout, and reuse the training
normalization without fitting any new statistics. --normalization-cache can
also reuse statistics when the training feature cache need not be preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import NeutralAffectDataset


MODEL_ID = "iic/emotion2vec_plus_base"
MODEL_SHA256 = "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd"
CONFIG_SHA256 = "50674ec187406b4f1208925594ceaf1f28f16a22333b5f2ac5b3d0cbffdd670e"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conv_geometry(model):
    """Read actual unpadded waveform convolutions; derive sample center times."""
    encoder = model.modality_encoders["AUDIO"].local_encoder
    layers = []
    hop, receptive, origin = 1, 1, 0.0
    for module in encoder.modules():
        if not isinstance(module, nn.Conv1d):
            continue
        kernel, stride, dilation, padding = (int(getattr(module, name)[0]) for name in
                                             ("kernel_size", "stride", "dilation", "padding"))
        if padding != 0 or dilation != 1:
            raise ValueError("Unexpected emotion2vec waveform convolution geometry")
        origin -= padding * hop
        receptive += (kernel - 1) * dilation * hop
        hop *= stride
        layers.append({"kernel": kernel, "stride": stride, "dilation": dilation, "padding": padding})
    if len(layers) != 7 or hop != 320 or receptive != 400:
        raise ValueError(f"Unexpected emotion2vec frontend: {layers}")
    center = origin + (receptive - 1) / 2
    return {"sample_rate": 16000, "hop_samples": hop, "receptive_samples": receptive,
            "center_offset_samples": center, "center_offset_s": center / 16000,
            "frame_rate": 16000 / hop, "convolutions": layers,
            "context": "Full-sentence Transformer; timestamps index input receptive-field centers, not localized causal support"}


def expected_length(samples, geometry):
    length = samples
    for layer in geometry["convolutions"]:
        length = (length + 2 * layer["padding"] - layer["dilation"] * (layer["kernel"] - 1) - 1) // layer["stride"] + 1
    return length


def align_features(features, feature_times, targets, valid, samples, geometry):
    """Interpolate centers; bounded nearest endpoints cover waveform boundaries.

    The no-padding frontend has no center at t=0. Endpoint extension is explicit
    and separately marked, while source observations and masks remain intact.
    No target beyond the actual recorded waveform is accepted.
    """
    if features.ndim != 2 or features.shape[1] != 768 or len(features) < 2:
        raise ValueError("Expected actual [frames>=2,768] frame embeddings")
    feature_times = np.asarray(feature_times, dtype=np.float64)
    if feature_times.shape != (len(features),) or not np.isfinite(feature_times).all():
        raise ValueError("Feature times must be finite and match the frame embeddings")
    if not np.isfinite(features).all() or np.any(np.diff(feature_times) <= 0):
        raise ValueError("Nonfinite features or invalid feature clock")
    targets = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if targets.ndim != 1 or targets.shape != valid.shape or not np.isfinite(targets).all():
        raise ValueError("Native targets and validity must share a finite one-dimensional clock")
    duration = samples / geometry["sample_rate"]
    if np.any(valid & ((targets < -1e-8) | (targets > duration + 1e-8))):
        raise ValueError("Valid native targets fall outside the actual waveform clock")
    max_extension = (geometry["receptive_samples"] - 1) / (2 * geometry["sample_rate"]) + geometry["hop_samples"] / geometry["sample_rate"]
    if np.any(valid & ((targets < feature_times[0] - max_extension - 1e-8) |
                       (targets > feature_times[-1] + max_extension + 1e-8))):
        raise ValueError("Native target needs more than the justified frontend boundary extension")
    centered = valid & (targets >= feature_times[0]) & (targets <= feature_times[-1])
    clipped = np.clip(targets, feature_times[0], feature_times[-1])
    right = np.searchsorted(feature_times, clipped, side="right").clip(1, len(feature_times) - 1)
    left = right - 1
    alpha = (clipped - feature_times[left]) / (feature_times[right] - feature_times[left])
    aligned = features[left] * (1 - alpha[:, None]) + features[right] * alpha[:, None]
    aligned[~valid] = 0
    return aligned.astype(np.float32), centered, valid & ~centered


def audio_statistics(train_queries, heldout_queries, model_info, geometry, normalization_cache=None):
    """Fit train-only statistics, or reuse previously fitted tensors unchanged.

    Reuse checks the model identity, frame geometry and actual normalization
    fitting clips. Current extraction observations never modify reused stats.
    """
    if normalization_cache is None:
        observed_audio = torch.cat([q["audio"][q["valid"]] for q in train_queries])
        if len(observed_audio) < 2 or observed_audio.shape[1] != 768 or not torch.isfinite(observed_audio).all():
            raise ValueError("Need at least two finite 768D training frames for normalization")
        stats = {"mean": observed_audio.mean(0), "std": observed_audio.std(0).clamp_min(1e-3),
                 "count": len(observed_audio), "source": "selected_training_query_valid_frames_only",
                 "feature_type": MODEL_ID,
                 "fit_clip_ids": [q["clip_id"] for q in train_queries],
                 "fit_sentence_ids": sorted({q["sentence_id"] for q in train_queries})}
        return stats, {"mode": "fit_train_only", "fit_query_count": len(train_queries)}
    path = Path(normalization_cache)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("split") != "train":
        raise ValueError("Normalization cache must be a training cache, never heldout/audit")
    provenance = saved.get("audio_feature_provenance", {})
    previous_model = provenance.get("model", {})
    for key in ("model_id", "model_sha256", "config_sha256", "model_source_sha256", "audio_encoder_source_sha256", "frontend_source_sha256"):
        if key not in model_info or previous_model.get(key) != model_info[key]:
            raise ValueError(f"Normalization cache has a different emotion2vec extractor: {key}")
    if provenance.get("geometry") != geometry:
        raise ValueError("Normalization cache uses different frame geometry")
    stats = saved.get("audio_stats", {})
    if stats.get("source") != "selected_training_query_valid_frames_only" or stats.get("feature_type") != MODEL_ID:
        raise ValueError("Normalization was not fitted exclusively on emotion2vec training query frames")
    for key in ("mean", "std"):
        value = stats.get(key)
        if not torch.is_tensor(value) or value.shape != (768,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid stored 768D normalization {key}")
    if not (stats["std"] > 0).all() or int(stats.get("count", 0)) < 2:
        raise ValueError("Stored normalization requires positive std and at least two training frames")
    # Older extractor caches precede explicit fit IDs. Their documented fit
    # rule is exactly their own train queries; subsequent reuse preserves IDs.
    fitting_clips = set(stats.get("fit_clip_ids", [q["clip_id"] for q in saved["queries"]]))
    fitting_sentences = set(stats.get("fit_sentence_ids", [q["sentence_id"] for q in saved["queries"]]))
    if fitting_clips & {q["clip_id"] for q in heldout_queries} or fitting_sentences & {q["sentence_id"] for q in heldout_queries}:
        raise ValueError("Heldout/audit overlaps the normalization fitting clips or sentences")
    # Shallow-copy metadata only; mean/std/count are the original values, with
    # no recomputation, float conversion, or re-centering on audit observations.
    stats = dict(stats)
    return stats, {"mode": "reuse_frozen_train_statistics", "source_cache": str(path.resolve()),
                   "source_cache_sha256": sha(path), "fit_query_count": len(fitting_clips),
                   "current_extraction_used_to_fit": False}


def reused_training_cache(feature_data, source_data, heldout_queries, model_info, geometry):
    """Verify the raw->feature chain before byte-copying an existing train cache."""
    feature_data, source_data = Path(feature_data), Path(source_data)
    cache_path = feature_data / "train.pt"
    converted = torch.load(cache_path, map_location="cpu", weights_only=False)
    previous = converted.get("audio_feature_provenance", {})
    raw_train_hash = sha(source_data / "train.pt")
    if previous.get("source_cache_sha256", {}).get("train") != raw_train_hash:
        raise ValueError("Audit raw training cache differs from the original emotion2vec source training cache")
    stats, normalization = audio_statistics([], heldout_queries, model_info, geometry, cache_path)
    original_raw = torch.load(source_data / "train.pt", map_location="cpu", weights_only=False)
    if [q["clip_id"] for q in converted["queries"]] != [q["clip_id"] for q in original_raw["queries"]]:
        raise ValueError("Converted training query order differs from its raw training source")
    for raw, features in zip(original_raw["queries"], converted["queries"]):
        if features["audio"].shape != (raw["audio"].shape[0], 768):
            raise ValueError("Reused training cache does not contain 768D frame audio")
        for key in ("motion", "content", "times", "valid", "motion_valid", "channel_mask", "speaker_id", "emotion_id", "intensity_id"):
            if not torch.equal(raw[key], features[key]):
                raise ValueError(f"Converted training cache changed raw {key}")
    return converted, stats, normalization, {"source_directory": str(feature_data.resolve()),
        "train_cache_sha256": sha(cache_path), "source_raw_train_sha256": raw_train_hash,
        "heldout_only_extraction": True, "train_cache_byte_copy": True}


def verify_loaded_weights(model, model_path):
    """Do not accept AutoModel's permissive missing-weight initialization."""
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    verified = 0
    for name, tensor in model.state_dict().items():
        candidates = [prefix + name for prefix in ("", "d2v_model.", "module.", "module.d2v_model.")]
        key = next((candidate for candidate in candidates if candidate in state), None)
        if key is None or tensor.shape != state[key].shape:
            raise ValueError(f"Pretrained tensor missing or wrong shape: {name}")
        if not torch.equal(tensor.detach().cpu(), state[key].to(dtype=tensor.dtype)):
            raise ValueError(f"Pretrained tensor did not load exactly: {name}")
        verified += 1
    return verified


def load_extractor(model_dir, device):
    from funasr import AutoModel
    if sha(model_dir / "model.pt") != MODEL_SHA256 or sha(model_dir / "config.yaml") != CONFIG_SHA256:
        raise ValueError("Model files differ from the audited official emotion2vec_plus_base release")
    wrapper = AutoModel(model=str(model_dir), device=device, disable_update=True, disable_pbar=True)
    model = wrapper.model
    model.eval().requires_grad_(False)
    if int(model.cfg.embed_dim) != 768 or not bool(model.cfg.normalize):
        raise ValueError("Unexpected emotion2vec model configuration")
    # FunASR constructs a pretraining reconstruction decoder absent from the
    # official inference checkpoint. extract_features(features_only=True)
    # never calls it. Remove it so accidental use fails, then verify EVERY
    # remaining tensor exactly against the pinned checkpoint.
    model.modality_encoders["AUDIO"].decoder = None
    verified = verify_loaded_weights(model, model_dir / "model.pt")
    geometry = conv_geometry(model)
    return model, geometry, {"model_id": MODEL_ID, "model_dir": str(model_dir.resolve()),
        "model_sha256": MODEL_SHA256, "config_sha256": CONFIG_SHA256,
        "verified_state_tensors": verified, "parameter_count": sum(p.numel() for p in model.parameters()),
        "removed_unused_module": "modality_encoders.AUDIO.decoder (pretraining reconstruction only)",
        "funasr_version": importlib.metadata.version("funasr"), "torch_version": torch.__version__,
        "model_source_sha256": sha(inspect.getfile(type(model))),
        "audio_encoder_source_sha256": sha(inspect.getfile(type(model.modality_encoders["AUDIO"]))),
        "frontend_source_sha256": sha(inspect.getfile(type(model.modality_encoders["AUDIO"].local_encoder)))}


@torch.inference_mode()
def extract_full_clip(clip, model, geometry, cache_dir, model_info, device):
    import soundfile as sf
    source = clip["metadata"]["provenance"]
    for key in ("audio_path", "audio_sha256", "audio_offset_s"):
        if key not in source:
            raise ValueError(f"Missing audited {key}: {clip['clip_id']}")
    wave_path = Path(source["audio_path"])
    wave_hash = sha(wave_path)
    if wave_hash != source["audio_sha256"]:
        raise ValueError(f"Waveform changed since native artifact: {wave_path}")
    offset = float(source["audio_offset_s"])
    if not np.isfinite(offset):
        raise ValueError("Nonfinite source audio offset")
    info = sf.info(str(wave_path))
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError(f"Expected audited 16k mono waveform, refusing implicit resampling: {wave_path}")
    # Pin full-feature reuse to actual waveform, weights and implementation.
    cache_signature = {"wave_sha256": wave_hash, "model_sha256": model_info["model_sha256"],
        "source_sha256": model_info["model_source_sha256"], "frontend_sha256": model_info["frontend_source_sha256"],
        "funasr_version": model_info["funasr_version"], "torch_version": model_info["torch_version"],
        "normalization": "layer_norm_entire_recorded_waveform", "remove_extra_tokens": True}
    cache_key = hashlib.sha256(json.dumps(cache_signature, sort_keys=True).encode()).hexdigest()
    cache_file = cache_dir / (cache_key + ".npz")
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as saved:
            if json.loads(str(saved["signature"].item())) != cache_signature:
                raise ValueError("Feature cache signature mismatch")
            features = saved["features"].copy()
    else:
        wave, sr = sf.read(str(wave_path), dtype="float32", always_2d=False)
        if sr != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError("Invalid original full waveform")
        sequence = torch.from_numpy(wave).to(device)
        sequence = F.layer_norm(sequence, sequence.shape).unsqueeze(0)
        output = model.extract_features(sequence, padding_mask=None, mask=False, remove_extra_tokens=True)
        if output["x"].ndim != 3 or output["x"].shape[0] != 1:
            raise ValueError("Extractor returned an utterance vector or unexpected output")
        features = output["x"][0].float().cpu().numpy()
        np.savez_compressed(cache_file, features=features, signature=json.dumps(cache_signature, sort_keys=True))
    expected = expected_length(info.frames, geometry)
    if features.shape != (expected, 768) or not np.isfinite(features).all():
        raise ValueError(f"Frame count fails audited convolution clock: got {features.shape}, expected {(expected, 768)}")
    if float(features.std(axis=0).mean()) <= 1e-8:
        raise ValueError("Temporally constant features; refusing an utterance embedding broadcast")
    feature_times = (geometry["center_offset_samples"] + np.arange(expected) * geometry["hop_samples"]) / geometry["sample_rate"]
    targets = clip["times"].numpy() + offset
    aligned, covered, extended = align_features(features, feature_times, targets, clip["valid"].numpy(), info.frames, geometry)
    record = {"clip_id": clip["clip_id"], "audio_path": str(wave_path), "audio_sha256": wave_hash,
        "wave_samples": info.frames, "frame_features": expected, "source_audio_offset_s": offset,
        "feature_cache": str(cache_file), "feature_cache_sha256": sha(cache_file),
        "native_valid_frames": int(clip["valid"].sum()), "center_interpolated_frames": int(covered.sum()),
        "boundary_extended_frames": int(extended.sum()), "full_waveform_temporal_std": float(features.std(axis=0).mean())}
    return aligned, covered, extended, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Official ModelScope snapshot directory")
    parser.add_argument("--feature-cache", type=Path, help="Optional shared full-wave feature cache")
    parser.add_argument("--normalization-cache", type=Path,
                        help="Previously extracted emotion2vec train.pt; reuse its mean/std exactly for a locked audit")
    parser.add_argument("--reuse-feature-data", type=Path,
                        help="Copy this emotion2vec training cache byte for byte and extract only new heldout with its frozen statistics")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.resolve() == args.source_data.resolve():
        raise ValueError("Extraction needs a new output directory")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    if args.normalization_cache and args.reuse_feature_data:
        raise ValueError("Use --reuse-feature-data or --normalization-cache, not both")
    datasets = {split: NeutralAffectDataset(args.source_data / (split + ".pt")) for split in ("train", "heldout")}
    if {q["clip_id"] for q in datasets["train"].queries} & {q["clip_id"] for q in datasets["heldout"].queries}:
        raise ValueError("Training and heldout clips overlap")
    if {q["sentence_id"] for q in datasets["train"].queries} & {q["sentence_id"] for q in datasets["heldout"].queries}:
        raise ValueError("Training and heldout sentences overlap")
    torch.set_num_threads(4)
    model, geometry, model_info = load_extractor(args.model_dir, args.device)
    reused_train, reused_stats, normalization, reuse_info = None, None, None, None
    if args.reuse_feature_data:
        reused_train, reused_stats, normalization, reuse_info = reused_training_cache(
            args.reuse_feature_data, args.source_data, datasets["heldout"].queries, model_info, geometry)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_cache = args.feature_cache or args.output / "full_wave_features"
    feature_cache.mkdir(parents=True, exist_ok=True)
    extracted, records = {}, []
    for split, dataset in datasets.items():
        if split == "train" and reused_train is not None:
            # Never decode, renormalize, or reserialize the original training
            # features. The file itself is copied after audit extraction.
            extracted[split] = reused_train["queries"]
            continue
        extracted[split] = []
        for index, original in enumerate(dataset.queries):
            aligned, covered, extended, record = extract_full_clip(original, model, geometry, feature_cache, model_info, args.device)
            query = {**original, "audio": torch.from_numpy(aligned),
                     "audio_center_covered": torch.from_numpy(covered), "audio_boundary_extended": torch.from_numpy(extended),
                     "metadata": {**original["metadata"], "audio_feature": {**record, "model_id": MODEL_ID}}}
            extracted[split].append(query)
            records.append({**record, "split": split})
            if index % 20 == 0 or index == len(dataset) - 1:
                print(json.dumps({"split": split, "done": index + 1, "total": len(dataset)}), flush=True)
    if reused_stats is None:
        stats, normalization = audio_statistics(extracted["train"], extracted["heldout"], model_info,
                                                geometry, args.normalization_cache)
    else:
        stats = reused_stats
    mean, std = stats["mean"], stats["std"]
    provenance = {"schema": "emotion2vec_native_pilot_v1", "model": model_info, "geometry": geometry,
        "source_data": str(args.source_data.resolve()),
        "source_cache_sha256": {s: sha(args.source_data / (s + ".pt")) for s in datasets},
        "script_sha256": sha(__file__), "query_audio_dim": 768,
        "reference_audio_dim": 83, "identity_references": "Preserved from source; their audio is not used by identity encoding",
        "alignment": "Full-wave features at verified convolution centers, linear interpolation to native video time plus source_audio_offset_s",
        "boundary_policy": "Nearest center endpoint extension within real waveform; maximum (receptive-1)/2+hop samples; per-frame masks recorded; motion valid mask unchanged",
        "audio_stats_source": stats["source"], "normalization": normalization,
        "reuse_feature_data": reuse_info,
        "limitations": ["Offline full-sentence context; not causal streaming features.",
                        "Pretrained speech affect embeddings are inputs, not frame-level visual emotion ground truth.",
                        "External pretraining/fine-tuning data overlap with academic SER corpora is not ruled out."]}
    for split, dataset in datasets.items():
        if split == "train" and reused_train is not None:
            shutil.copy2(args.reuse_feature_data / "train.pt", args.output / "train.pt")
            if sha(args.output / "train.pt") != reuse_info["train_cache_sha256"]:
                raise IOError("Byte-copy of original converted training cache failed")
            continue
        for query in extracted[split]:
            query["audio"] = (query["audio"] - mean) / std
            query["audio"][~query["valid"]] = 0
        cache = {**dataset.cache, "queries": extracted[split], "audio_stats": stats,
                 "original_audio_stats": dataset.cache["audio_stats"], "audio_feature_provenance": provenance,
                 "audio_dim": 768}
        torch.save(cache, args.output / (split + ".pt"))
        NeutralAffectDataset(args.output / (split + ".pt"))
    for name in ("train.jsonl", "heldout.jsonl", "enrollment.jsonl", "expanded_train.jsonl", "audit_manifest.jsonl", "audit_selection.json"):
        source = args.source_data / name
        if source.is_file():
            shutil.copy2(source, args.output / name)
    selection_path = args.source_data / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf8")) if selection_path.is_file() else {}
    selection["audio_feature_provenance"] = provenance
    selection["splits"] = {**selection.get("splits", {}), **{split: {
        **selection.get("splits", {}).get(split, {}), "clips": len(dataset),
        "cache_sha256": sha(args.output / (split + ".pt"))} for split, dataset in datasets.items()}}
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf8")
    (args.output / "feature_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf8")
    (args.output / "emotion2vec_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")
    print(json.dumps({"output": str(args.output), "query_audio_dim": 768, "train_frames_for_stats": int(stats["count"]),
                      "normalization_mode": normalization["mode"],
                      "boundary_extended_frames": sum(r["boundary_extended_frames"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
