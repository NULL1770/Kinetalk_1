import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import prepare_audio_text_semantics as module
from scripts.extract_emotion2vec_pilot import sha


class FakeTokenizer:
    model_max_length = 12
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, **kwargs):
        ids = [101] + [int(word) for word in text.split()] + [102]
        if not kwargs.get("truncation"):
            return {"input_ids": ids}
        length = kwargs["max_length"]
        ids = ids if len(ids) <= length else ids[:length - 1] + [102]
        valid = [1] * len(ids) + [0] * (length - len(ids))
        return {"input_ids": torch.tensor([ids + [0] * (length - len(ids))]),
                "attention_mask": torch.tensor([valid])}

    def convert_ids_to_tokens(self, ids):
        return [str(item) for item in ids]


class FakeTextModel:
    config = SimpleNamespace(hidden_size=3, max_position_embeddings=12)

    def __call__(self, input_ids, attention_mask):
        assert not torch.is_grad_enabled()
        hidden = input_ids[..., None].float().expand(-1, -1, 3).clone()
        hidden[~attention_mask.bool()] = float("nan")
        return SimpleNamespace(last_hidden_state=hidden)


class FakeAsrProcessor:
    def __init__(self):
        self.chunks = []

    def __call__(self, wave, **kwargs):
        assert kwargs == {"sampling_rate": 16000, "return_tensors": "pt", "return_attention_mask": True}
        self.chunks.append(wave.copy())
        return {"input_features": torch.tensor([[float(wave[0])]]), "attention_mask": torch.ones(1, 1)}

    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return [f" chunk{int(tokens[0, 0])}   words "]


class FakeAsrModel:
    config = SimpleNamespace(model_type="whisper", max_target_positions=448)

    def __init__(self, multilingual=True):
        self.generation_config = SimpleNamespace(is_multilingual=multilingual, eos_token_id=9)
        self.calls = []

    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls.append(kwargs)
        return torch.tensor([[len(self.calls), 9]])


def test_text_preserves_order_and_special_tokens_reports_truncation_and_zeros_padding():
    tokenizer, model = FakeTokenizer(), FakeTextModel()
    tokens, valid, record = module.encode_transcript("1 2", tokenizer, model, 6, "cpu")
    assert tokens.shape == (6, 3) and valid.tolist() == [True] * 4 + [False] * 2
    torch.testing.assert_close(tokens[:4, 0], torch.tensor([101., 1., 2., 102.]))
    assert not tokens[4:].any() and torch.isfinite(tokens).all() and not record["truncated"]
    tokens, valid, record = module.encode_transcript("1 2 3 4 5 6", tokenizer, model, 5, "cpu")
    assert record["truncated"] and record["untruncated_token_count"] == 8 and record["kept_token_count"] == 5
    assert record["token_ids"] == [101, 1, 2, 3, 102] and valid.all()
    reordered, _, _ = module.encode_transcript("3 2 1", tokenizer, model, 5, "cpu")
    assert not torch.equal(reordered, tokens)


def test_empty_asr_is_all_masked_without_a_spurious_special_token_embedding():
    class MustNotRun(FakeTextModel):
        def __call__(self, **kwargs): raise AssertionError("Empty speech must not call text encoder")
    tokens, valid, record = module.encode_transcript(" \n ", FakeTokenizer(), MustNotRun(), 6, "cpu")
    assert not tokens.any() and not valid.any() and record["empty_transcript"]
    assert record["token_ids"] == [] and record["untruncated_token_count"] == 0
    with pytest.raises(ValueError, match="position limit"):
        module.encode_transcript("1", FakeTokenizer(), FakeTextModel(), 13, "cpu")


def test_asr_reads_all_wave_samples_once_with_deterministic_english_and_no_reference_text():
    wave = np.arange(36005, dtype=np.float32)
    processor, model = FakeAsrProcessor(), FakeAsrModel()
    transcript, record = module.transcribe_wave(wave, processor, model, "cpu", chunk_seconds=1.)
    np.testing.assert_array_equal(np.concatenate(processor.chunks), wave)
    assert transcript == "chunk1 words chunk2 words chunk3 words"
    assert [(c["start_sample"], c["end_sample"]) for c in record["chunks"]] == [(0, 16000), (16000, 32000), (32000, 36005)]
    assert not record["reference_transcript_used"] and not record["token_timestamps_available"]
    for call in model.calls:
        assert call["language"] == "en" and call["task"] == "transcribe" and call["do_sample"] is False
        assert "prompt_ids" not in call and "text" not in call and "input_ids" not in call
    assert not any(c["possibly_decode_truncated"] for c in record["chunks"])
    en_model = FakeAsrModel(multilingual=False)
    module.transcribe_wave(wave[:10], FakeAsrProcessor(), en_model, "cpu")
    assert "language" not in en_model.calls[0] and "task" not in en_model.calls[0]


def test_model_evidence_binds_all_local_files_and_rejects_missing_random_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="weights"): module.model_directory_evidence(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"frozen-checkpoint")
    (tmp_path / "tokenizer.json").write_text("{}")
    before = module.model_directory_evidence(tmp_path)
    assert before["files_sha256"]["model.safetensors"] == sha(tmp_path / "model.safetensors")
    (tmp_path / "tokenizer.json").write_text('{"changed":true}')
    assert module.model_directory_evidence(tmp_path)["fingerprint"] != before["fingerprint"]
    module.ensure_loaded_checkpoint({"unexpected_keys": ["discarded_task_head"]}, "text")
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        with pytest.raises(ValueError, match="complete frozen checkpoint"):
            module.ensure_loaded_checkpoint({field: ["bad"]}, "text")


def test_native_wave_binding_rejects_hash_format_and_crop_outside_audio(tmp_path, monkeypatch):
    sf = SimpleNamespace()
    monkeypatch.setitem(module.sys.modules, "soundfile", sf)
    path = tmp_path / "wave.wav"; path.write_bytes(b"test-binding")
    clip = {"times": torch.tensor([0., .04, .08], dtype=torch.float64),
            "valid": torch.ones(3, dtype=torch.bool),
            "metadata": {"provenance": {"audio_path": str(path), "audio_sha256": sha(path), "audio_offset_s": .02}}}
    wave = np.ones(2000, dtype=np.float32)
    monkeypatch.setattr(sf, "read", lambda *args, **kwargs: (wave, 16000), raising=False)
    np.testing.assert_array_equal(module.read_authorized_wave(clip), wave)
    bad = copy.deepcopy(clip); bad["metadata"]["provenance"]["audio_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA256"): module.read_authorized_wave(bad)
    bad = copy.deepcopy(clip); bad["metadata"]["provenance"]["audio_offset_s"] = .2
    with pytest.raises(ValueError, match="outside"): module.read_authorized_wave(bad)
    monkeypatch.setattr(sf, "read", lambda *args, **kwargs: (wave, 22050))
    with pytest.raises(ValueError, match="16 kHz mono"): module.read_authorized_wave(clip)


def test_cli_only_opens_cache_allowlisted_train_native_and_saves_full_schema(tmp_path, monkeypatch):
    root = tmp_path / "native"; root.mkdir()
    rows, splits = [], {}
    for role, cid, sentence in (("train", "fit", "s1"), ("validation", "dev", "s2")):
        wave = root / (cid + ".wav"); wave.write_bytes(b"test-wave")
        times = np.arange(4, dtype=np.float64) / 25
        valid = np.array([True, True, True, False])
        content = np.full((4, 768), 3., dtype=np.float32)
        provenance = {"schema": "native_affect_style_v4.1", "clock_evidence": "embedded_video", "fps": 25,
                      "audio_path": str(wave), "audio_sha256": sha(wave), "audio_offset_s": 0.}
        artifact = root / (cid + ".npz")
        np.savez(artifact, content=content, times=times, mask=valid, provenance=json.dumps(provenance))
        rows.append({"clip_id": cid, "sentence": sentence, "speaker": "person", "emotion": 1, "split": "train",
                     "artifact": artifact.name, "artifact_sha256": sha(artifact)})
        splits[role] = {"q": {"clip_id": [cid], "sentence_id": [sentence], "speaker": ["person"],
                     "emotion_id": torch.tensor([1]), "content": torch.from_numpy(content)[None].half(),
                     "times": torch.from_numpy(times)[None], "valid": torch.from_numpy(valid)[None]}}
    # The metadata can contain unused train clips. Its nonexistent artifact
    # would fail immediately if extraction widened past current cache roles.
    rows.append({**rows[0], "clip_id": "unused", "artifact": "DO_NOT_OPEN.npz"})
    manifest = tmp_path / "train.jsonl"; manifest.write_text("\n".join(json.dumps(r) for r in rows))
    renderer = tmp_path / "renderer.pt"
    torch.save({"schema": "predictable_renderer_cache_v1", "splits": splits}, renderer)
    model_info = {"asr": {"fingerprint": "frozen"}, "text": {"fingerprint": "frozen"}}
    monkeypatch.setattr(module, "load_models", lambda *args: (None, None, FakeTokenizer(), FakeTextModel(), model_info))
    monkeypatch.setattr(module, "model_directory_evidence", lambda *args: {"fingerprint": "frozen"})
    opened = []
    def read_wave(clip):
        opened.append(clip["clip_id"])
        return np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(module, "read_authorized_wave", read_wave)
    monkeypatch.setattr(module, "transcribe_wave", lambda *a, **k: ("1 2", {"chunks": [{"possibly_decode_truncated": False}]}))
    output = tmp_path / "text.pt"
    monkeypatch.setattr(module.sys, "argv", ["prepare_audio_text_semantics.py", "--renderer-cache", str(renderer),
        "--native-train-manifest", str(manifest), "--native-root", str(root), "--asr-model-dir", str(tmp_path),
        "--text-model-dir", str(tmp_path), "--output", str(output), "--max-tokens", "6", "--device", "cpu"])
    module.main()
    saved = torch.load(output, weights_only=False)
    assert opened == ["fit", "dev"] and saved["schema"] == module.SCHEMA
    assert saved["splits"]["train"]["tokens"].shape == (1, 6, 3)
    assert saved["splits"]["validation"]["clip_id"] == ["dev"]
    assert saved["provenance"]["renderer_cache_sha256"] == sha(renderer)
    assert saved["provenance"]["native_motion_arrays_read"] is False
    assert saved["provenance"]["emotion_labels_as_input"] is False
    assert not any("times" in split for split in saved["splits"].values())
    summary = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["cache_sha256"] == sha(output) and summary["empty_transcripts"] == 0


def test_faster_whisper_backend_fullwave_and_no_label_or_gt_transcript_prompt():
    class FakeFaster:
        def __init__(self): self.calls = []
        def transcribe(self, wave, **kwargs):
            self.calls.append((wave.copy(), kwargs))
            segment = SimpleNamespace(text=" spoken words ", tokens=[1, 2], avg_logprob=-.1,
                                      no_speech_prob=.001, compression_ratio=1.)
            return iter([segment]), SimpleNamespace(language="en", language_probability=1.)
    wave = np.arange(24000, dtype=np.float32)
    model = FakeFaster()
    text, record = module.transcribe_wave(wave, None, model, "cpu", chunk_seconds=1.)
    assert text == "spoken words spoken words" and record["backend"] == "faster-whisper"
    np.testing.assert_array_equal(np.concatenate([call[0] for call in model.calls]), wave)
    for _, call in model.calls:
        assert call["initial_prompt"] is None and call["prefix"] is None
        assert call["language"] == "en" and call["temperature"] == 0.
        assert call["condition_on_previous_text"] is False and call["vad_filter"] is False
        assert call["word_timestamps"] is False and call["without_timestamps"] is True
