from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import channel_mask, first_array
from .protocol import load_split, statistics, legacy_path


class CanonicalStage1Dataset(Dataset):
    """One-branch Stage1 dataset: canonical content -> canonical neutral motion.

    Pair artifacts provide source content on ``canonical_times``.  The neutral
    target is materialized separately on that exact clock; source emotional
    motion is never used as a Stage1 target.
    """

    def __init__(self, cfg: dict[str, Any], split: str = "train", random_crop: bool = True):
        data = cfg["data"]
        rows, manifest = load_split(data, split, stage1=True)
        self.items = [dict(r) for r in rows if r.get("stage1_target_artifact") or r.get("teacher_artifact")]
        if len(self.items) != len(rows):
            raise ValueError(f"Stage1 manifest contains rows without teacher artifacts: {manifest}")
        self.protocol_stats = statistics(self.items)
        self.split = split
        if not self.items:
            raise ValueError(f"No Stage1 pair rows in {manifest}")
        self.aligned_root = Path(data["aligned_dtw_root"])
        self.target_root = Path(data.get("stage1_target_root", "")) if data.get("stage1_target_root") else None
        self.window = int(data.get("window", 96))
        self.motion_dim = int(data.get("motion_dim", 52))
        self.content_dim = int(data.get("content_dim", 768))
        self.neutral_indices = list(data.get("neutral_output_indices", []))
        self.random_crop = random_crop
        self.native_time = bool(data.get("stage1_native_time", False))
        self.native_feature_root = Path(data.get("stage1_native_feature_root", "")) if data.get("stage1_native_feature_root") else None
        self.fps = float(data.get("fps", 25.0))
        self.native_context = int(data.get("stage1_native_context", 9))
        self.cfg_pair_weight = float(data.get("stage1_cross_emotion_weight", 0.35))
        self.emotion_classes = [str(x).lower() for x in data.get("emotion_classes", ["neutral", "angry", "contempt", "disgust", "fear", "happy", "sad", "surprise"])]
        self.emotion_to_id = {name: index for index, name in enumerate(self.emotion_classes)}

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _speaker(record: dict[str, Any], key: str = "source_clip_id") -> str:
        if record.get("speaker"):
            return str(record["speaker"])
        clip = str(record.get(key, record.get("clip_id", "unknown")))
        parts = clip.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else clip

    @staticmethod
    def _sentence(record: dict[str, Any], key: str = "source_clip_id") -> str:
        return str(record.get("sentence_id", record.get("content_id", record.get(key, "unknown"))))

    @staticmethod
    def _fit_dim(array: np.ndarray, dim: int) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        out = np.zeros((len(array), dim), dtype=np.float32)
        out[:, : min(dim, array.shape[-1])] = array[:, :dim]
        return out

    @staticmethod
    def _pad_crop(array: np.ndarray, start: int, window: int, dim: int) -> tuple[np.ndarray, int]:
        out = np.zeros((window, dim), dtype=np.float32)
        if start >= len(array):
            return out, 0
        valid = min(window, len(array) - start)
        out[:valid, : min(dim, array.shape[-1])] = array[start:start + valid, :dim]
        return out, valid

    def _artifact(self, row: dict[str, Any]) -> Path:
        raw = Path(str(row.get("teacher_artifact", "")))
        path = raw if raw.is_file() else self.aligned_root / "pairs" / raw.name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage1 pair artifact: {path}")
        return path

    def _target(self, row: dict[str, Any], length: int) -> tuple[np.ndarray, np.ndarray]:
        raw = Path(str(row.get("stage1_target_artifact", "")))
        if not raw.name:
            raw = Path(f"{row.get('source_clip_id')}__to__{row.get('reference_clip_id')}.npz")
        path = raw if raw.is_file() else (self.target_root / raw.name if self.target_root and raw.name else None)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing canonical neutral target for {row.get('source_clip_id')}")
        with np.load(path, allow_pickle=False) as z:
            target = self._fit_dim(z["canonical_neutral_target"], self.motion_dim)
            mask = np.asarray(z["canonical_neutral_mask"], dtype=np.float32).astype(bool)
        n = min(length, len(target), len(mask))
        out = np.zeros((length, self.motion_dim), dtype=np.float32)
        valid = np.zeros(length, dtype=bool)
        out[:n] = target[:n]
        valid[:n] = mask[:n]
        return out, valid

    @staticmethod
    def _native_feature_path(root: Path, row: dict[str, Any]) -> Path:
        clip_id = str(row.get("source_clip_id", row.get("clip_id", "")))
        dataset = str(row.get("dataset", "mead"))
        candidates = [root / dataset / f"{clip_id}.npz", root / f"{clip_id}.npz"]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"Missing native Stage1 content for {clip_id}; tried: {candidates}")

    def _native_item(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.native_feature_root is None:
            raise ValueError("stage1_native_feature_root is required when stage1_native_time is enabled")
        feature_path = self._native_feature_path(self.native_feature_root, row)
        with np.load(self._artifact(row), allow_pickle=False) as pair:
            source_motion_times = np.asarray(pair["source_motion_times"], dtype=np.float64)
            native_teacher = np.asarray(pair["neutral_teacher_on_source"], dtype=np.float32)
            native_mask = np.asarray(pair["native_teacher_mask"], dtype=bool)
            geometry = np.asarray(pair["native_geometry_mask"], dtype=bool) if "native_geometry_mask" in pair.files else native_mask.copy()
        with np.load(feature_path, allow_pickle=False) as feature:
            content = np.asarray(feature["content"], dtype=np.float32)
            feature_times = np.asarray(feature["times"], dtype=np.float64)
        if content.ndim != 2 or len(content) != len(feature_times) or len(content) < 2:
            raise ValueError(f"Invalid native feature shape for {row.get('source_clip_id')}")
        if native_teacher.shape[0] != len(source_motion_times) or native_mask.shape[0] != len(source_motion_times):
            raise ValueError(f"Native teacher/source motion length mismatch for {row.get('source_clip_id')}")
        motion_times = np.asarray(source_motion_times, dtype=np.float64)
        if len(motion_times) < 2 or np.any(~np.isfinite(motion_times)) or np.any(np.diff(motion_times) <= 0):
            raise ValueError(f"Invalid source motion clock for {row.get('source_clip_id')}")
        source_audio_offset = float(row.get("source_audio_offset_s", 0.0))
        if not np.isfinite(source_audio_offset):
            raise ValueError(f"Invalid source audio offset for {row.get('source_clip_id')}")
        # Preserve a short native-clock context around every motion frame. A
        # learned aggregator can then handle small audio/motion clock offsets
        # and short phonetic events instead of receiving one interpolated token.
        radius = self.native_context // 2
        centers = np.searchsorted(feature_times, motion_times + source_audio_offset)
        centers = np.clip(centers, 0, len(feature_times) - 1)
        context = np.zeros((len(motion_times), self.native_context, self.content_dim), dtype=np.float32)
        for t, center in enumerate(centers):
            indices = np.arange(int(center) - radius, int(center) + radius + 1)
            valid_indices = np.clip(indices, 0, len(content) - 1)
            context[t] = self._fit_dim(content[valid_indices], self.content_dim)
        target = self._fit_dim(native_teacher, self.motion_dim)
        valid = native_mask & (motion_times + source_audio_offset >= feature_times[0]) & (motion_times + source_audio_offset <= feature_times[-1])
        quality = valid & geometry
        return context, target, valid, quality

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.items[index]
        if self.native_time:
            content, target, target_mask, quality = self._native_item(row)
            length = len(content)
            if length < 1 or not target_mask.any():
                raise ValueError(f"Empty native neutral supervision: {row.get('source_clip_id')}")
            start = random.randint(0, max(length - self.window, 0)) if self.random_crop else max((length - self.window) // 2, 0)
            content_crop = np.zeros((self.window, self.native_context, self.content_dim), dtype=np.float32)
            valid = min(self.window, length - start)
            if valid > 0:
                content_crop[:valid] = content[start:start + valid]
            target_crop, _ = self._pad_crop(target, start, self.window, self.motion_dim)
            mask_crop, _ = self._pad_crop(target_mask[:, None].astype(np.float32), start, self.window, 1)
            quality_crop, _ = self._pad_crop(quality[:, None].astype(np.float32), start, self.window, 1)
            mask = (mask_crop[:, 0] > 0.5) & (torch.arange(self.window).numpy() < valid)
            # Articulation calibration must be trained from the neutral target
            # that the canonical branch is expected to reproduce.  The old
            # `canonical_motion` field is source emotional motion on the
            # canonical clock; using it here leaks expression and timing error
            # into the Stage1 style controls.
            reference_start = start
            reference_crop, reference_valid = self._pad_crop(target, reference_start, self.window, self.motion_dim)
            reference_mask_crop, _ = self._pad_crop(target_mask[:, None].astype(np.float32), reference_start, self.window, 1)
            reference_mask_crop = (reference_mask_crop[:, 0] > 0.5) & (torch.arange(self.window).numpy() < reference_valid)
            return {
                "content": torch.from_numpy(content_crop),
                "target": torch.from_numpy(target_crop),
                "style_reference_motion": torch.from_numpy(reference_crop),
                "style_reference_mask": torch.from_numpy(reference_mask_crop),
                "mask": torch.from_numpy(mask),
                "quality": torch.from_numpy(quality_crop[:, 0].astype(np.float32)),
                "pair_weight": torch.tensor(1.0 if str(row.get("emotion", "neutral")).lower() == "neutral" else float(self.cfg_pair_weight), dtype=torch.float32),
                "pair_type": "identity" if str(row.get("emotion", "neutral")).lower() == "neutral" else "cross_emotion",
                "clip_id": str(row.get("source_clip_id", row.get("clip_id", "unknown"))),
                "reference_clip_id": str(row.get("reference_clip_id", "unknown")),
                "speaker": self._speaker(row, "reference_clip_id"),
                "sentence_id": self._sentence(row, "reference_clip_id"),
                "emotion": str(row.get("emotion", "neutral")).lower(),
                "emotion_id": self.emotion_to_id.get(str(row.get("emotion", "neutral")).lower(), 0),
            }
        with np.load(self._artifact(row), allow_pickle=False) as z:
            content = self._fit_dim(z["canonical_content"], self.content_dim)
            length = len(content)
        target, target_mask = self._target(row, length)
        if length < 1 or not target_mask.any():
            raise ValueError(f"Empty canonical neutral supervision: {row.get('source_clip_id')}")
        start = random.randint(0, max(length - self.window, 0)) if self.random_crop else max((length - self.window) // 2, 0)
        content_crop, valid = self._pad_crop(content, start, self.window, self.content_dim)
        target_crop, _ = self._pad_crop(target, start, self.window, self.motion_dim)
        mask_crop, _ = self._pad_crop(target_mask[:, None].astype(np.float32), start, self.window, 1)
        mask = (mask_crop[:, 0] > 0.5) & (torch.arange(self.window).numpy() < valid)
        # The calibration reference is derived from the neutral target.  The
        # emotional `canonical_motion` field is never used as Stage1 input.
        reference_start = start
        reference_crop, reference_valid = self._pad_crop(target, reference_start, self.window, self.motion_dim)
        reference_mask_crop, _ = self._pad_crop(target_mask[:, None].astype(np.float32), reference_start, self.window, 1)
        reference_mask_crop = (reference_mask_crop[:, 0] > 0.5) & (torch.arange(self.window).numpy() < reference_valid)
        return {
            "content": torch.from_numpy(content_crop),
            "target": torch.from_numpy(target_crop),
            "style_reference_motion": torch.from_numpy(reference_crop),
            "style_reference_mask": torch.from_numpy(reference_mask_crop),
            "mask": torch.from_numpy(mask),
            "quality": torch.ones(self.window, dtype=torch.float32),
            "pair_weight": torch.tensor(1.0),
            "pair_type": "legacy",
            "clip_id": str(row.get("source_clip_id", row.get("clip_id", "unknown"))),
            "reference_clip_id": str(row.get("reference_clip_id", "unknown")),
            "speaker": self._speaker(row, "reference_clip_id"),
            "sentence_id": self._sentence(row, "reference_clip_id"),
            "emotion": str(row.get("emotion", "neutral")).lower(),
            "emotion_id": self.emotion_to_id.get(str(row.get("emotion", "neutral")).lower(), 0),
        }


def collate_stage1(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty Stage1 batch")
    return {
        "content": torch.stack([x["content"] for x in items]),
        "target": torch.stack([x["target"] for x in items]),
        "style_reference_motion": torch.stack([x["style_reference_motion"] for x in items]),
        "style_reference_mask": torch.stack([x["style_reference_mask"] for x in items]),
        "mask": torch.stack([x["mask"] for x in items]),
        "quality": torch.stack([x["quality"] for x in items]),
        "pair_weight": torch.stack([x["pair_weight"] for x in items]),
        "pair_type": [x["pair_type"] for x in items],
        "clip_id": [x["clip_id"] for x in items],
        "reference_clip_id": [x["reference_clip_id"] for x in items],
        "speaker": [x["speaker"] for x in items],
        "sentence_id": [x["sentence_id"] for x in items],
        "emotion": [x["emotion"] for x in items],
        "emotion_id": torch.tensor([x["emotion_id"] for x in items], dtype=torch.long),
    }


class B0ResidualDataset(Dataset):
    """MEAD records on the release DTW timeline with optional safe sidecars.

    The canonical Stage-2 target is deliberately constructed from neutral
    *ground truth* rather than a current Stage-1 prediction:

        b0_gt = K0 * neutral_motion_aligned
        residual_gt = emotion_motion_aligned - b0_gt

    `b0_pred` only appears in Stage 4, where deployment conditions are used.
    """

    def __init__(self, cfg: dict[str, Any], split: str = "train", random_crop: bool = True):
        data_cfg = cfg["data"]
        self.cfg_data = data_cfg
        split_records, manifest_path = load_split(data_cfg, split)
        self.split = split
        self.aligned_root = Path(data_cfg["aligned_dtw_root"])
        self.legacy_aligned_root = Path(data_cfg.get("legacy_aligned_root", self.aligned_root.parent / "aligned_dtw_v2"))
        for record in split_records:
            if not record.get("teacher_artifact"):
                raise ValueError(f"Missing teacher artifact: {record['clip_id']}")
            source_id = str(record.get("source_clip_id", record.get("clip_id", "")))
            reference_id = str(record.get("reference_clip_id", ""))
            for kind in ("audio", "affect", "bs"):
                for clip_id in (source_id, reference_id):
                    if legacy_path(self.legacy_aligned_root, kind, clip_id, record.get("dataset", "mead")) is None:
                        raise FileNotFoundError(f"Missing {kind}: {clip_id}; rebuild protocol manifests to record exclusions")
        self.items = [dict(record) for record in split_records]
        if not self.items:
            raise ValueError(f"No accepted {split} records after quality filtering: {manifest_path}")

        self.motion_dim = int(data_cfg.get("motion_dim", 52))
        self.window = int(data_cfg.get("window", 96))
        self.content_dim = int(data_cfg.get("content_dim", 768))
        self.audio_dim = int(data_cfg.get("audio_dim", 2))
        self.audio_teacher_dim = int(data_cfg.get("audio_teacher_dim", 64))
        self.affect_dim = int(data_cfg.get("affect_dim", 3))
        self.audio_emotion_dim = int(data_cfg.get("audio_emotion_dim", self.audio_dim + self.audio_teacher_dim + self.affect_dim))
        self.random_crop = random_crop
        # Residual supervision is undefined without an exact same-speaker,
        # same-sentence neutral partner.  Never turn a missing partner into a
        # zero residual, even when a legacy config opted out of this check.
        self.require_neutral_partner = True
        self.emotion_classes = [str(item).lower() for item in data_cfg.get("emotion_classes", [])]
        if not self.emotion_classes:
            raise ValueError("data.emotion_classes cannot be empty")
        self.emotion_to_id = {name: index for index, name in enumerate(self.emotion_classes)}
        self.intensity_levels = int(data_cfg.get("num_intensity_levels", 4))
        self.neutral_indices = list(data_cfg.get("neutral_output_indices", []))
        self._channel_mask = channel_mask(self.motion_dim, self.neutral_indices)
        self.supervision_root = Path(data_cfg.get("supervision_root", "")) if data_cfg.get("supervision_root") else None
        self.supervision_dim = int(data_cfg.get("supervision_dim", 5))

        self.by_group: dict[tuple[str, str], list[int]] = {}
        self.by_speaker: dict[str, list[int]] = {}
        for index, item in enumerate(self.items):
            speaker, sentence = self._group_key(item)
            self.by_group.setdefault((speaker, sentence), []).append(index)
            self.by_speaker.setdefault(speaker, []).append(index)
        self.neutral_by_group: dict[tuple[str, str], int] = {}
        for group, indices in self.by_group.items():
            candidates = [index for index in indices if self._emotion_name(self.items[index]) == "neutral"]
            if candidates:
                self.neutral_by_group[group] = min(
                    candidates,
                    key=lambda index: self.items[index]["intensity_id"],
                )
        if self.require_neutral_partner:
            kept = [item for item in self.items if self._group_key(item) in self.neutral_by_group]
            if len(kept) != len(self.items):
                raise ValueError("Manifest includes records without a neutral partner; rebuild protocol manifests")
            for item in self.items:
                neutral = self.items[self.neutral_by_group[self._group_key(item)]]
                if neutral["clip_id"] != item["reference_clip_id"]:
                    raise ValueError(f"Ambiguous neutral clock for {item['clip_id']}; expected {item['reference_clip_id']}")
        self.protocol_stats = statistics(self.items)

    def _rebuild_indices(self) -> None:
        self.by_group = {}
        self.by_speaker = {}
        for index, item in enumerate(self.items):
            group = self._group_key(item)
            self.by_group.setdefault(group, []).append(index)
            self.by_speaker.setdefault(group[0], []).append(index)
        self.neutral_by_group = {}
        for group, indices in self.by_group.items():
            candidates = [index for index in indices if self._emotion_name(self.items[index]) == "neutral"]
            if candidates:
                self.neutral_by_group[group] = min(
                    candidates,
                    key=lambda index: self.items[index]["intensity_id"],
                )

    @staticmethod
    def _group_key(record: dict[str, Any]) -> tuple[str, str]:
        speaker = str(record.get("speaker", "")).strip()
        sentence = str(record.get("sentence_id", record.get("content_id", ""))).strip()
        clip = str(record.get("source_clip_id", record.get("clip_id", "unknown")))
        parts = clip.split("_")
        if not speaker and len(parts) >= 2:
            speaker = "_".join(parts[:2])
        if not sentence:
            # MEAD clip ids end in _L{int}_{sentence}; this is only a
            # fallback for legacy rows without pair metadata.
            sentence = parts[-1] if parts else clip
        return speaker or "unknown", sentence or "unknown"

    @staticmethod
    def _dataset_name(record: dict[str, Any]) -> str:
        return str(record.get("dataset", "mead")).lower()

    def _emotion_name(self, record: dict[str, Any]) -> str:
        aliases = {"disgusted": "disgust", "fearful": "fear", "surprised": "surprise"}
        raw = str(record["emotion"]).lower()
        return aliases.get(raw, raw)

    def __len__(self) -> int:
        return len(self.items)

    def _feature_path(self, kind: str, record: dict[str, Any]) -> Path:
        clip_id = str(record["clip_id"])
        dataset = self._dataset_name(record)
        root = self.aligned_root / kind
        candidates = [root / dataset / f"{clip_id}.npz", root / str(record.get("dataset", dataset)) / f"{clip_id}.npz", root / f"{clip_id}.npz"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Missing aligned {kind} feature for {clip_id}; tried: {candidates}")

    @staticmethod
    def _fit_dim(array: np.ndarray, dim: int) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        result = np.zeros((len(array), dim), dtype=np.float32)
        result[:, : min(dim, array.shape[-1])] = array[:, :dim]
        return result

    @staticmethod
    def _resample_time(array: np.ndarray, length: int) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if len(array) == length:
            return array
        if len(array) == 0 or length == 0:
            return np.zeros((length, array.shape[-1] if array.ndim == 2 else 0), dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        old = np.linspace(0.0, 1.0, len(array), dtype=np.float32)
        new = np.linspace(0.0, 1.0, length, dtype=np.float32)
        return np.stack([np.interp(new, old, array[:, dim]) for dim in range(array.shape[-1])], axis=-1).astype(np.float32)

    @staticmethod
    def _pad_crop(array: np.ndarray, start: int, window: int, dim: int) -> tuple[np.ndarray, int]:
        result = np.zeros((window, dim), dtype=np.float32)
        if start >= len(array):
            return result, 0
        valid = min(window, len(array) - start)
        result[:valid, : min(dim, array.shape[-1])] = array[start : start + valid, :dim]
        return result, valid

    def _load_full(self, record: dict[str, Any], *, reference: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Release v3 pair rows are self-contained: canonical HuBERT content
        # and canonical neutral motion live in the pair artifact.  This path
        # is used for Stage1 and never falls back to v2 aligned arrays.
        if record.get("teacher_artifact"):
            artifact = Path(str(record["teacher_artifact"]))
            if not artifact.is_file():
                candidate = self.aligned_root / "pairs" / artifact.name
                artifact = candidate
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing v3 teacher artifact for {record.get('source_clip_id')}: {artifact}")
            with np.load(artifact, allow_pickle=False) as archive:
                if "canonical_motion" not in archive.files or "canonical_content" not in archive.files:
                    raise ValueError(f"Incomplete v3 pair artifact: {artifact}")
                canonical_length = len(archive["canonical_content"])
                motion = self._fit_dim(archive["canonical_motion"], self.motion_dim)
                content = self._fit_dim(archive["canonical_content"], self.content_dim)
            clip_id = str(record.get("reference_clip_id" if reference else "source_clip_id", record.get("clip_id", "")))
            dataset = self._dataset_name(record)
            # v2 features are already on a motion-rate aligned clock and are
            # retained as the audio/affect teacher while v3 supplies the
            # higher quality HuBERT canonical content and motion pair.
            def legacy(kind: str, key: str, dim: int) -> np.ndarray:
                candidates = [
                    self.legacy_aligned_root / kind / dataset / f"{clip_id}.npz",
                    self.legacy_aligned_root / kind / f"{clip_id}.npz",
                ]
                if kind == "bs":
                    candidates.extend([
                        self.legacy_aligned_root.parent / "coeffs_final" / dataset / f"{clip_id}.npz",
                        self.legacy_aligned_root.parent / "coeffs_final" / f"{clip_id}.npz",
                    ])
                path = next((candidate for candidate in candidates if candidate.is_file()), None)
                if path is None:
                    raise FileNotFoundError(f"Missing legacy aligned {kind} feature for {clip_id}: {candidates}")
                with np.load(path, allow_pickle=False) as archive:
                    if key not in archive.files:
                        raise KeyError(f"Missing {key} in {path}")
                    return self._fit_dim(self._resample_time(archive[key], len(motion)), dim)
            if reference:
                motion = legacy("bs", "coeffs", self.motion_dim)
                motion = self._resample_time(motion, canonical_length)
            else:
                motion = self._resample_time(motion, canonical_length)
            audio = legacy("audio", "feat", self.audio_dim)
            teacher = legacy("audio", "emotion", self.audio_teacher_dim)
            affect = legacy("affect", "affect", self.affect_dim)
            emotion_audio = self._fit_dim(np.concatenate([audio, teacher, affect], axis=-1), self.audio_emotion_dim)
            return motion, content, audio, emotion_audio
        clip_id = str(record["clip_id"])
        motion_path = self.aligned_root / "bs" / f"{clip_id}.npz"
        if not motion_path.is_file():
            raise FileNotFoundError(f"Missing aligned BS target: {motion_path}")
        with np.load(motion_path, allow_pickle=False) as archive:
            motion = self._fit_dim(first_array(archive, "coeffs"), self.motion_dim)
        with np.load(self._feature_path("audio", record), allow_pickle=False) as archive:
            audio = self._fit_dim(first_array(archive, "feat"), self.audio_dim)
            teacher = self._fit_dim(first_array(archive, "emotion") if "emotion" in archive.files else np.zeros((len(audio), 0), np.float32), self.audio_teacher_dim)
        with np.load(self._feature_path("content", record), allow_pickle=False) as archive:
            content = self._fit_dim(first_array(archive, "content"), self.content_dim)
        with np.load(self._feature_path("affect", record), allow_pickle=False) as archive:
            affect = self._fit_dim(first_array(archive, "affect"), self.affect_dim)
        length = min(len(motion), len(audio), len(teacher), len(content), len(affect))
        if length < 1:
            raise ValueError(f"Empty aligned record: {clip_id}")
        emotion_audio = self._fit_dim(np.concatenate([audio[:length], teacher[:length], affect[:length]], axis=-1), self.audio_emotion_dim)
        return motion[:length], content[:length], audio[:length], emotion_audio

    def _load_supervision(self, record: dict[str, Any], length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load optional event/boundary sidecars without fabricating labels."""
        zeros = np.zeros((length, self.supervision_dim), np.float32)
        boundary = np.zeros((length, 3), np.float32)
        viseme = np.full(length, -1, np.int64)
        event_mask = np.zeros(length, np.float32)
        boundary_mask = np.zeros(length, np.float32)
        viseme_mask = np.zeros(length, np.float32)
        if self.supervision_root is None:
            return zeros, event_mask, boundary, boundary_mask, viseme, viseme_mask
        path = self.supervision_root / f"{record['clip_id']}.npz"
        if not path.is_file():
            return zeros, event_mask, boundary, boundary_mask, viseme, viseme_mask
        with np.load(path, allow_pickle=False) as archive:
            event = self._fit_dim(first_array(archive, "mouth_event"), self.supervision_dim)
            boundary = self._fit_dim(first_array(archive, "boundary"), 3)
            viseme = np.asarray(archive["viseme_id"], dtype=np.int64) if "viseme_id" in archive.files else viseme
            event_mask = np.asarray(archive["mouth_event_mask"], dtype=np.float32) if "mouth_event_mask" in archive.files else event_mask
            boundary_mask = np.asarray(archive["boundary_mask"], dtype=np.float32) if "boundary_mask" in archive.files else boundary_mask
            viseme_mask = np.asarray(archive["viseme_mask"], dtype=np.float32) if "viseme_mask" in archive.files else viseme_mask
        n = min(length, len(event), len(boundary), len(viseme), len(event_mask), len(boundary_mask), len(viseme_mask))
        out = np.zeros((length, self.supervision_dim), np.float32); out[:n] = event[:n]
        b = np.zeros((length, 3), np.float32); b[:n] = boundary[:n]
        v = np.full(length, -1, np.int64); v[:n] = viseme[:n]
        return out, event_mask[:length], b, boundary_mask[:length], v, viseme_mask[:length]

    def _read(self, record: dict[str, Any], start: int | None = None, *, reference: bool = False) -> dict[str, Any]:
        motion, content, audio, emotion_audio = self._load_full(record, reference=reference)
        length = min(len(motion), len(content), len(audio), len(emotion_audio))
        if start is None:
            start = random.randint(0, max(length - self.window, 0)) if self.random_crop else max((length - self.window) // 2, 0)
        start = max(0, min(int(start), max(length - 1, 0)))
        motion_crop, valid = self._pad_crop(motion, start, self.window, self.motion_dim)
        content_crop, _ = self._pad_crop(content, start, self.window, self.content_dim)
        audio_crop, _ = self._pad_crop(audio, start, self.window, self.audio_dim)
        emotion_audio_crop, _ = self._pad_crop(emotion_audio, start, self.window, self.audio_emotion_dim)
        event, event_mask, boundary, boundary_mask, viseme, viseme_mask = self._load_supervision(record, length)
        event_crop, _ = self._pad_crop(event, start, self.window, self.supervision_dim)
        event_mask_crop, _ = self._pad_crop(event_mask[:, None], start, self.window, 1)
        boundary_crop, _ = self._pad_crop(boundary, start, self.window, 3)
        boundary_mask_crop, _ = self._pad_crop(boundary_mask[:, None], start, self.window, 1)
        viseme_crop, _ = self._pad_crop(viseme[:, None].astype(np.float32), start, self.window, 1)
        viseme_mask_crop, _ = self._pad_crop(viseme_mask[:, None], start, self.window, 1)
        emotion = self._emotion_name(record)
        return {
            "motion": torch.from_numpy(motion_crop),
            "content": torch.from_numpy(content_crop),
            "audio": torch.from_numpy(audio_crop),
            "audio_emotion": torch.from_numpy(emotion_audio_crop),
            "mouth_event": torch.from_numpy(event_crop),
            "mouth_event_mask": torch.from_numpy(event_mask_crop[:, 0] > 0.5),
            "boundary": torch.from_numpy(boundary_crop),
            "boundary_mask": torch.from_numpy(boundary_mask_crop[:, 0] > 0.5),
            "viseme_id": torch.from_numpy(viseme_crop[:, 0].round().astype(np.int64)),
            "viseme_mask": torch.from_numpy(viseme_mask_crop[:, 0] > 0.5),
            "mask": torch.arange(self.window) < valid,
            "speaker": self._group_key(record)[0],
            "sentence_id": self._group_key(record)[1],
            "clip_id": str(record["clip_id"]),
            "emotion_id": self.emotion_to_id[emotion],
            "intensity_id": record["intensity_id"],
            "crop_start": start,
            "source_length": length,
        }

    def _neutral_index(self, index: int) -> tuple[int, bool]:
        neutral = self.neutral_by_group.get(self._group_key(self.items[index]))
        return (neutral, True) if neutral is not None else (index, False)

    def _emotion_partner_index(self, index: int) -> tuple[int, bool]:
        record = self.items[index]
        candidates = [
            item_index
            for item_index in self.by_group[self._group_key(record)]
            if self._emotion_name(self.items[item_index]) != self._emotion_name(record)
        ]
        return (random.choice(candidates), True) if candidates else (index, False)

    def _style_reference_index(self, index: int) -> tuple[int, bool]:
        record = self.items[index]
        speaker, sentence = self._group_key(record)
        preferred = [
            item_index
            for item_index in self.by_speaker[speaker]
            if self._group_key(self.items[item_index])[1] != sentence
            and self._emotion_name(self.items[item_index]) != self._emotion_name(record)
        ]
        candidates = preferred or [
            item_index for item_index in self.by_speaker[speaker] if self._group_key(self.items[item_index])[1] != sentence
        ]
        if bool(self.cfg_data.get("cross_speaker_style", False)):
            candidates = [i for sp, ids in self.by_speaker.items() if sp != speaker for i in ids]
        return (random.choice(candidates), True) if candidates else (index, False)

    def _style_positive_index(self, index: int) -> tuple[int, bool]:
        """Same-speaker/different-sentence positive for triplet + Stage-3 style.

        Independent of ``cross_speaker_style``: that toggle governs the
        ``style_reference`` branch used for cross-speaker intervention/cycle
        experiments, but the triplet positive and the Stage-3 render-closure
        teacher style both need a speaker-consistent reference so their
        frame-level reconstruction targets stay valid.
        """
        record = self.items[index]
        speaker, sentence = self._group_key(record)
        preferred = [
            item_index
            for item_index in self.by_speaker[speaker]
            if self._group_key(self.items[item_index])[1] != sentence
            and self._emotion_name(self.items[item_index]) != self._emotion_name(record)
        ]
        candidates = preferred or [
            item_index
            for item_index in self.by_speaker[speaker]
            if self._group_key(self.items[item_index])[1] != sentence
        ]
        return (random.choice(candidates), True) if candidates else (index, False)

    def _style_negative_index(self, index: int) -> tuple[int, bool]:
        """A different-speaker sample used as an explicit triplet negative.

        Sampling is speaker-first then item-random so a single large speaker
        cannot dominate the negative pool.
        """
        speaker, _ = self._group_key(self.items[index])
        other_speakers = [sp for sp in self.by_speaker if sp != speaker]
        if not other_speakers:
            return index, False
        chosen = random.choice(other_speakers)
        return random.choice(self.by_speaker[chosen]), True

    def _attach_targets(self, sample: dict[str, Any], neutral: dict[str, Any], neutral_valid: bool) -> None:
        mask = self._channel_mask.to(dtype=sample["motion"].dtype)
        b0_gt = neutral["motion"] * mask.unsqueeze(0)
        valid = sample["mask"] & neutral["mask"] if neutral_valid else torch.zeros_like(sample["mask"])
        sample["neutral_motion"] = neutral["motion"]
        sample["neutral_mask"] = neutral["mask"]
        sample["b0_gt"] = b0_gt
        sample["residual_gt"] = sample["motion"] - b0_gt
        sample["residual_mask"] = valid
        sample["neutral_valid"] = bool(neutral_valid)

    def __getitem__(self, index: int) -> dict[str, Any]:
        query = self._read(self.items[index])
        neutral_index, neutral_valid = self._neutral_index(index)
        # Every array in a DTW group is aligned to the same neutral timeline.
        neutral = self._read(self.items[neutral_index], start=query["crop_start"])
        self._attach_targets(query, neutral, neutral_valid)

        emotion_index, emotion_valid = self._emotion_partner_index(index)
        emotion_partner = self._read(self.items[emotion_index], start=query["crop_start"])
        emotion_neutral_index, emotion_neutral_valid = self._neutral_index(emotion_index)
        emotion_neutral = self._read(self.items[emotion_neutral_index], start=query["crop_start"])
        self._attach_targets(emotion_partner, emotion_neutral, emotion_neutral_valid)

        style_index, style_valid = self._style_reference_index(index)
        style_reference = self._read(self.items[style_index], reference=True)
        style_neutral_index, style_neutral_valid = self._neutral_index(style_index)
        style_neutral = self._read(self.items[style_neutral_index], start=style_reference["crop_start"])
        self._attach_targets(style_reference, style_neutral, style_neutral_valid)

        positive_index, positive_valid = self._style_positive_index(index)
        style_positive = self._read(self.items[positive_index], reference=True)
        positive_neutral_index, positive_neutral_valid = self._neutral_index(positive_index)
        positive_neutral = self._read(self.items[positive_neutral_index], start=style_positive["crop_start"])
        self._attach_targets(style_positive, positive_neutral, positive_neutral_valid)

        negative_index, negative_valid = self._style_negative_index(index)
        style_negative = self._read(self.items[negative_index], reference=True)
        negative_neutral_index, negative_neutral_valid = self._neutral_index(negative_index)
        negative_neutral = self._read(self.items[negative_neutral_index], start=style_negative["crop_start"])
        self._attach_targets(style_negative, negative_neutral, negative_neutral_valid)

        return {
            "query": query,
            "neutral": neutral,
            "emotion_pair": emotion_partner,
            "style_reference": style_reference,
            "style_positive": style_positive,
            "style_negative": style_negative,
            "relations": {
                "neutral": neutral_valid,
                "emotion": emotion_valid and emotion_neutral_valid,
                "style": style_valid and style_neutral_valid,
                "style_positive": positive_valid and positive_neutral_valid,
                "style_negative": negative_valid and negative_neutral_valid,
            },
        }


def collate_b0_residual(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")

    tensor_keys = {
        "motion", "content", "audio", "audio_emotion", "mouth_event", "mouth_event_mask", "boundary", "boundary_mask", "viseme_id", "viseme_mask", "mask", "neutral_motion", "neutral_mask",
        "b0_gt", "residual_gt", "residual_mask",
    }
    integer_keys = {"emotion_id", "intensity_id"}
    result: dict[str, Any] = {}
    for branch in ("query", "neutral", "emotion_pair", "style_reference", "style_positive", "style_negative"):
        result[branch] = {}
        for key in tensor_keys:
            if key in items[0][branch]:
                result[branch][key] = torch.stack([item[branch][key] for item in items])
        for key in integer_keys:
            if key in items[0][branch]:
                result[branch][key] = torch.tensor([item[branch][key] for item in items], dtype=torch.long)
        for key in ("speaker", "sentence_id", "clip_id", "crop_start", "source_length", "neutral_valid"):
            if key in items[0][branch]:
                result[branch][key] = [item[branch][key] for item in items]
    relation_keys = ("neutral", "emotion", "style", "style_positive", "style_negative")
    result["relations"] = {
        key: torch.tensor([bool(item["relations"].get(key, False)) for item in items], dtype=torch.bool)
        for key in relation_keys
    }
    return result
