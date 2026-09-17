"""Frozen automatic speech text tokens for the existing fit/development clips.

Whisper sees only each authorized native waveform, never a reference transcript
or emotion label. A frozen local Hugging Face text encoder returns its token
sequence (not sentence pooling). Tokens have no invented frame timestamps.
All models must already exist locally; this script never downloads weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_emotion2vec_pilot import sha
from scripts.prepare_label_guided_audio_cache import load_authorized_native, read_manifest, validate_cache
from scripts.train_formal_predictable_projection import save_checkpoint, save_json

SCHEMA = "audio_text_semantics_v1"
SAMPLE_RATE = 16000


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_directory_evidence(directory):
    """Bind local checkpoint, tokenizer and configs, including sharded weights."""
    root = Path(directory).resolve()
    if not root.is_dir(): raise ValueError("Existing local model directory required: " + str(root))
    ignored = {".git", ".cache", "__pycache__"}
    files = {path.relative_to(root).as_posix(): sha(path)
             for path in sorted(root.rglob("*"))
             if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts)}
    if "config.json" not in files or not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Require a local Hugging Face config and model weights: " + str(root))
    return {"directory": str(root), "files_sha256": files, "fingerprint": canonical_hash(files)}


def ensure_loaded_checkpoint(loading_info, role):
    # Unexpected downstream-head keys may be intentionally discarded by
    # AutoModel; missing/mismatched base weights would be random initialization.
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            raise ValueError(f"{role} did not load a complete frozen checkpoint: {key}={loading_info[key]}")


def load_models(asr_directory, text_directory, device, asr_backend="transformers", asr_compute_type="float16"):
    # Defense in depth: local_files_only is also passed on every from_pretrained.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModel, AutoTokenizer

    asr_evidence = model_directory_evidence(asr_directory)
    text_evidence = model_directory_evidence(text_directory)
    if asr_backend == "transformers":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        asr_processor = WhisperProcessor.from_pretrained(str(asr_directory), local_files_only=True)
        asr_model, asr_info = WhisperForConditionalGeneration.from_pretrained(
            str(asr_directory), local_files_only=True, output_loading_info=True)
        ensure_loaded_checkpoint(asr_info, "Whisper")
        if getattr(asr_model.config, "model_type", None) != "whisper": raise ValueError("Expected Whisper checkpoint")
        if int(asr_processor.feature_extractor.sampling_rate) != SAMPLE_RATE:
            raise ValueError("Whisper processor must use native 16 kHz")
        asr_model.eval().requires_grad_(False).to(device)
    elif asr_backend == "faster-whisper":
        from faster_whisper import WhisperModel
        target = torch.device(device)
        if target.type not in ("cuda", "cpu"): raise ValueError("faster-whisper requires cpu or cuda")
        asr_processor = None
        asr_model = WhisperModel(str(Path(asr_directory).resolve()), device=target.type,
            device_index=target.index or 0, compute_type=asr_compute_type, cpu_threads=4, local_files_only=True)
        asr_info = {"backend": "faster-whisper", "compute_type": asr_compute_type,
                    "version": importlib.metadata.version("faster-whisper"),
                    "ctranslate2_version": importlib.metadata.version("ctranslate2")}
    else:
        raise ValueError("Unsupported ASR backend")
    tokenizer = AutoTokenizer.from_pretrained(str(text_directory), local_files_only=True, trust_remote_code=False)
    text_model, text_info = AutoModel.from_pretrained(str(text_directory), local_files_only=True,
                                                   trust_remote_code=False, output_loading_info=True)
    ensure_loaded_checkpoint(text_info, "text encoder")
    if getattr(text_model.config, "is_encoder_decoder", False) or getattr(text_model.config, "is_decoder", False):
        raise ValueError("Use an encoder-only text checkpoint exposing last_hidden_state, e.g. BERT/RoBERTa")
    if tokenizer.pad_token_id is None: raise ValueError("Text tokenizer must define an existing padding token")
    text_model.eval().requires_grad_(False).to(device)
    evidence = {"asr": asr_evidence, "text": text_evidence,
                "asr_loading_info": asr_info, "text_loading_info": text_info,
                "transformers_version": importlib.metadata.version("transformers"),
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "soundfile_version": importlib.metadata.version("soundfile"),
                "implementation_sha256": {name: sha(inspect.getfile(type(obj))) for name, obj in
                    (("asr_model", asr_model), ("asr_processor", asr_processor),
                     ("text_model", text_model), ("text_tokenizer", tokenizer)) if obj is not None},
                "text_class": type(text_model).__name__, "tokenizer_class": type(tokenizer).__name__,
                "asr_class": type(asr_model).__name__, "asr_backend": asr_backend,
                "frozen": True, "offline_local_only": True}
    return asr_processor, asr_model, tokenizer, text_model, evidence


def read_authorized_wave(clip):
    import soundfile as sf
    provenance = clip["metadata"]["provenance"]
    path = Path(provenance["audio_path"])
    if sha(path) != provenance["audio_sha256"]: raise ValueError("Native waveform SHA256 changed")
    offset = float(provenance["audio_offset_s"])
    if not math.isfinite(offset): raise ValueError("Nonfinite native audio offset")
    wave, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE or wave.ndim != 1 or len(wave) == 0 or not np.isfinite(wave).all():
        raise ValueError("Expected finite, nonempty, native 16 kHz mono waveform; no implicit resampling")
    targets = clip["times"].double().numpy()[clip["valid"].numpy()] + offset
    if not len(targets) or np.any(targets < -1e-8) or np.any(targets > len(wave) / rate + 1e-8):
        raise ValueError("Valid motion crop falls outside native waveform")
    return wave


def wave_chunk_ranges(samples, chunk_seconds=25.):
    if samples < 1 or not math.isfinite(chunk_seconds) or not 1. <= chunk_seconds <= 30.:
        raise ValueError("Need nonempty waveform and chunk_seconds within [1,30]")
    width = int(round(chunk_seconds * SAMPLE_RATE))
    return [(start, min(start + width, samples)) for start in range(0, samples, width)]


@torch.inference_mode()
def transcribe_wave(wave, processor, model, device, *, chunk_seconds=25., beam_size=5,
                    max_new_tokens=440):
    """Cover the complete recorded waveform with deterministic English ASR."""
    if processor is None:
        return transcribe_faster_whisper(wave, model, chunk_seconds=chunk_seconds,
                                        beam_size=beam_size, max_new_tokens=max_new_tokens)
    if beam_size < 1 or not 1 <= max_new_tokens <= int(model.config.max_target_positions) - 4:
        raise ValueError("Invalid Whisper decoding limits")
    if getattr(model.config, "model_type", None) != "whisper": raise ValueError("Expected Whisper model")
    generation = {"do_sample": False, "num_beams": beam_size, "max_new_tokens": max_new_tokens,
                  "return_timestamps": False}
    if getattr(model.generation_config, "is_multilingual", True):
        generation.update(language="en", task="transcribe")
    pieces, chunks = [], []
    for start, end in wave_chunk_ranges(len(wave), chunk_seconds):
        inputs = processor(wave[start:end], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           return_attention_mask=True)
        inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
        generated = model.generate(**inputs, **generation)
        if not torch.is_tensor(generated) or generated.ndim != 2 or len(generated) != 1:
            raise ValueError("Unexpected Whisper generated sequence")
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != 1 or not isinstance(decoded[0], str): raise ValueError("Unexpected ASR transcript")
        text = " ".join(decoded[0].split())
        pieces.append(text)
        ids = generated[0].detach().cpu().tolist()
        eos = getattr(model.generation_config, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        # Failure to end with EOS may be decoding-length saturation; keep and
        # report it explicitly rather than pretending it is complete speech.
        chunks.append({"start_sample": start, "end_sample": end, "transcript": text,
                       "generated_token_ids": ids, "ended_with_eos": bool(ids and ids[-1] in eos_ids),
                       "possibly_decode_truncated": bool(ids and ids[-1] not in eos_ids)})
    return " ".join(piece for piece in pieces if piece), {"sample_rate": SAMPLE_RATE,
        "wave_samples": len(wave), "chunks": chunks, "chunk_seconds": chunk_seconds,
        "language": "en", "task": "transcribe", "beam_size": beam_size,
        "max_new_tokens_per_chunk": max_new_tokens, "do_sample": False,
        "full_waveform_covered": True, "previous_chunk_text_conditioning": False,
        "reference_transcript_used": False, "token_timestamps_available": False}


def transcribe_faster_whisper(wave, model, *, chunk_seconds=25., beam_size=5, max_new_tokens=440):
    """The CTranslate2 backend uses the same full-wave, no-prompt contract."""
    if beam_size < 1 or not 1 <= max_new_tokens <= 440: raise ValueError("Invalid faster-whisper decoding limits")
    pieces, chunks = [], []
    for start, end in wave_chunk_ranges(len(wave), chunk_seconds):
        segments, info = model.transcribe(wave[start:end], language="en", task="transcribe",
            beam_size=beam_size, temperature=0., condition_on_previous_text=False,
            word_timestamps=False, without_timestamps=True, vad_filter=False,
            max_new_tokens=max_new_tokens, initial_prompt=None, prefix=None)
        segments = list(segments)
        text = " ".join(" ".join(segment.text.split()) for segment in segments).strip()
        pieces.append(text)
        segment_records = [{"text": segment.text, "generated_token_ids": list(map(int, segment.tokens)),
            "avg_logprob": float(segment.avg_logprob), "no_speech_prob": float(segment.no_speech_prob),
            "compression_ratio": float(segment.compression_ratio)} for segment in segments]
        chunks.append({"start_sample": start, "end_sample": end, "transcript": text,
                       "segments": segment_records, "language": str(info.language),
                       "language_probability": float(info.language_probability),
                       "possibly_decode_truncated": any(len(s["generated_token_ids"]) >= max_new_tokens
                                                        for s in segment_records)})
    return " ".join(piece for piece in pieces if piece), {"backend": "faster-whisper",
        "sample_rate": SAMPLE_RATE, "wave_samples": len(wave), "chunks": chunks, "chunk_seconds": chunk_seconds,
        "language": "en", "task": "transcribe", "beam_size": beam_size, "temperature": 0.,
        "max_new_tokens_per_chunk": max_new_tokens, "full_waveform_covered": True,
        "previous_chunk_text_conditioning": False, "reference_transcript_used": False,
        "token_timestamps_available": False, "vad_filter": False}


def text_dimensions(tokenizer, model, max_tokens):
    if max_tokens < 2: raise ValueError("At least two text positions required")
    limits = [getattr(tokenizer, "model_max_length", None), getattr(model.config, "max_position_embeddings", None)]
    finite_limits = [int(limit) for limit in limits if isinstance(limit, (int, float)) and 0 < limit < 1000000]
    if finite_limits and max_tokens > min(finite_limits):
        raise ValueError("max_tokens exceeds local text model/tokenizer position limit")
    dimension = getattr(model.config, "hidden_size", None) or getattr(model.config, "dim", None)
    if not isinstance(dimension, int) or dimension < 1: raise ValueError("Cannot establish text hidden dimension")
    return dimension


@torch.inference_mode()
def encode_transcript(transcript, tokenizer, model, max_tokens, device):
    if not isinstance(transcript, str): raise ValueError("Transcript must be an ASR string")
    dimension = text_dimensions(tokenizer, model, max_tokens)
    text = " ".join(transcript.split())
    if not text:
        return torch.zeros(max_tokens, dimension), torch.zeros(max_tokens, dtype=torch.bool), {
            "token_ids": [], "token_strings": [], "untruncated_token_count": 0, "kept_token_count": 0,
            "truncated": False, "empty_transcript": True, "max_tokens": max_tokens}
    complete = tokenizer(text, add_special_tokens=True, truncation=False, return_attention_mask=False)
    all_ids = complete["input_ids"]
    if not isinstance(all_ids, list) or not all(isinstance(item, int) for item in all_ids):
        raise ValueError("Expected one unbatched token ID sequence")
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_tokens,
                        padding="max_length", return_tensors="pt", return_attention_mask=True)
    if (encoded["input_ids"].shape != (1, max_tokens)
            or encoded["attention_mask"].shape != (1, max_tokens)
            or not torch.isin(encoded["attention_mask"], torch.tensor([0, 1])).all()):
        raise ValueError("Text tokenizer did not preserve requested padded shape/mask")
    inputs = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    result = model(**inputs)
    hidden = result.last_hidden_state
    if hidden.shape != (1, max_tokens, dimension): raise ValueError("Text model must expose all token hidden states")
    valid = encoded["attention_mask"][0].cpu().bool()
    if not valid.any() or not torch.isfinite(hidden[0].cpu()[valid]).all():
        raise ValueError("Nonempty ASR must produce finite valid semantic tokens")
    tokens = torch.where(valid[:, None], hidden[0].detach().cpu().float(), 0.)
    ids = encoded["input_ids"][0].cpu()[valid].tolist()
    return tokens, valid, {"token_ids": ids, "token_strings": tokenizer.convert_ids_to_tokens(ids),
        "untruncated_token_count": len(all_ids), "kept_token_count": len(ids),
        "truncated": len(all_ids) > max_tokens, "empty_transcript": False, "max_tokens": max_tokens,
        "special_tokens_included": True, "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("renderer-cache", "native-train-manifest", "native-root", "asr-model-dir", "text-model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--asr-backend", choices=("transformers", "faster-whisper"), default="transformers")
    parser.add_argument("--asr-compute-type", default="float16")
    parser.add_argument("--chunk-seconds", type=float, default=25.)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-asr-new-tokens", type=int, default=440)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists(): raise FileExistsError("Fresh output paths required")
    torch.set_num_threads(4)
    rows = read_manifest(args.native_train_manifest)
    cache_sha, manifest_sha = sha(args.renderer_cache), sha(args.native_train_manifest)
    cache = torch.load(args.renderer_cache, map_location="cpu", weights_only=False, mmap=True)
    allowed = validate_cache(cache, rows)
    processor, asr_model, tokenizer, text_model, model_info = load_models(args.asr_model_dir, args.text_model_dir,
        args.device, args.asr_backend, args.asr_compute_type)
    dimension = text_dimensions(tokenizer, text_model, args.max_tokens)
    splits, records = {}, []
    for role in ("train", "validation"):
        q = cache["splits"][role]["q"]
        ids = list(map(str, q["clip_id"]))
        tokens = torch.empty(len(ids), args.max_tokens, dimension, dtype=torch.float32)
        masks = torch.empty(len(ids), args.max_tokens, dtype=torch.bool)
        for index, cid in enumerate(ids):
            if cid not in allowed: raise ValueError("Unapproved speech extraction clip")
            clip, _, evidence = load_authorized_native(rows[cid], args.native_root,
                q["times"][index], q["valid"][index], q["content"][index])
            wave = read_authorized_wave(clip)
            transcript, asr_record = transcribe_wave(wave, processor, asr_model, args.device,
                chunk_seconds=args.chunk_seconds, beam_size=args.beam_size, max_new_tokens=args.max_asr_new_tokens)
            tokens[index], masks[index], token_record = encode_transcript(transcript, tokenizer, text_model,
                                                                          args.max_tokens, args.device)
            records.append({"split": role, "clip_id": cid, "sentence_id": clip["sentence_id"], **evidence,
                "origin": "automatic_frozen_whisper_native_waveform_only", "transcript": transcript,
                "transcription": asr_record, "tokenization": token_record})
            if (index + 1) % 25 == 0 or index + 1 == len(ids):
                print(json.dumps({"stage": "audio_text", "split": role, "done": index + 1, "total": len(ids),
                    "empty_transcripts": sum(r["tokenization"]["empty_transcript"] for r in records),
                    "truncated_token_sequences": sum(r["tokenization"]["truncated"] for r in records)}), flush=True)
        splits[role] = {"tokens": tokens, "token_valid": masks, "clip_id": ids,
                        "sentence_id": list(map(str, q["sentence_id"]))}
    if sha(args.renderer_cache) != cache_sha or sha(args.native_train_manifest) != manifest_sha:
        raise RuntimeError("Immutable source inputs changed during extraction")
    for role, directory in (("asr", args.asr_model_dir), ("text", args.text_model_dir)):
        if model_directory_evidence(directory)["fingerprint"] != model_info[role]["fingerprint"]:
            raise RuntimeError("Frozen local model files changed during extraction")
    sources = [Path(__file__), Path(__file__).with_name("prepare_label_guided_audio_cache.py")]
    provenance = {"schema": SCHEMA, "renderer_cache": str(args.renderer_cache.resolve()),
        "renderer_cache_sha256": cache_sha, "native_train_manifest": str(args.native_train_manifest.resolve()),
        "native_train_manifest_sha256": manifest_sha, "native_root": str(args.native_root.resolve()),
        "models": model_info, "source_sha256": {str(path.resolve()): sha(path) for path in sources},
        "text_hidden_dim": dimension, "max_tokens": args.max_tokens, "storage": "FP32 token states; padding exactly zero",
        "fit_statistics": None, "global_sentence_pooling": False, "emotion_labels_as_input": False,
        "ground_truth_transcripts_used": False, "native_motion_arrays_read": False, "test_loaded": False,
        "token_timestamps_available": False, "default_replaced": False,
        "context": "Full recorded native wave (may extend beyond the 96-frame motion crop); ordered text tokens lack frame alignment",
        "empty_transcript_contract": "all token_valid=False and tokens=0; consumer must handle fully masked text safely",
        "controls": "same cache supports zero text or clip-mismatched text; preserve token_valid with each sequence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, {"schema": SCHEMA, "splits": splits, "records": records, "provenance": provenance})
    output_sha = sha(args.output)
    summary = {"schema": SCHEMA, "cache_sha256": output_sha,
        "shapes": {role: list(value["tokens"].shape) for role, value in splits.items()},
        "empty_transcripts": sum(r["tokenization"]["empty_transcript"] for r in records),
        "truncated_token_sequences": sum(r["tokenization"]["truncated"] for r in records),
        "possibly_decode_truncated_chunks": sum(c["possibly_decode_truncated"] for r in records for c in r["transcription"]["chunks"]),
        "provenance": provenance}
    save_json(args.output.with_suffix(".json"), summary)
    print(json.dumps({"complete": True, "output": str(args.output.resolve()), "sha256": output_sha,
                      "empty_transcripts": summary["empty_transcripts"],
                      "truncated_token_sequences": summary["truncated_token_sequences"]}), flush=True)


if __name__ == "__main__":
    main()
