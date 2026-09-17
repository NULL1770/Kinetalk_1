"""Native-clock data contract for the isolated semantic/style experiment.

The legacy pair/neutral data loader intentionally remains unchanged.  This
loader requires real visual VA sidecars; it never derives VA from BS magnitude
or silently substitutes audio affect for visual labels.  Manifest paths are
relative to ``root`` (the manifest directory by default).

Required JSONL/CSV fields: clip_id, dataset, speaker, sentence_id, emotion,
split, motion_path, content_path, audio_path, va_path.  Intensity must be an
explicit nonnegative intensity_id, or intensity_valid=false (unknown is -1).
VA provenance must be declared as va_source=visual_teacher/visual_annotation
or in the sidecar as source_modality='visual'.  Every asset needs ``times``;
only motion permits an explicit motion_clock='native_fps', fps and
motion_start_s fallback.  Audio sample time = motion time +
source_audio_offset_s.  VA time is the source video/motion clock.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


EMOTIONS = ("neutral", "angry", "contempt", "disgust", "fear", "happy", "sad", "surprise")


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1, "0", "1", "true", "false", "True", "False"):
        return str(value).lower() in ("1", "true")
    raise ValueError(f"{name} must be an explicit boolean, got {value!r}")


def _integer(value: Any, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be a finite integer, got {value!r}")
    return int(number)


def _clock(times: np.ndarray, length: int, source: str) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (length,) or length < 2:
        raise ValueError(f"{source}: times must have shape ({length},), with at least two samples")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{source}: timestamps must be finite and strictly increasing")
    return times


def _matrix(value: np.ndarray, dim: int | None, source: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or (dim is not None and value.shape[1] != dim):
        raise ValueError(f"{source}: expected [T,{dim if dim is not None else 'D'}], got {value.shape}")
    if value.shape[1] == 0 or not np.isfinite(value).all():
        raise ValueError(f"{source}: empty feature dimension or nonfinite data")
    return value


def _verify_identity(archive: Any, row: Mapping[str, Any], source: str) -> None:
    """Reject explicit asset identity conflicts rather than trusting filenames."""
    metadata: dict[str, Any] = {}
    if "metadata_json" in archive:
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{source}: invalid metadata_json") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}: metadata_json must describe an object")
    for key in ("clip_id", "dataset", "speaker"):
        actual = str(archive[key].item()) if key in archive else metadata.get(key)
        if actual is not None and str(actual) != str(row[key]):
            raise ValueError(f"{source}: asset/manifest {key} mismatch ({actual!r} != {row[key]!r})")


def interpolate_clock(
    values: np.ndarray,
    times: np.ndarray,
    targets: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    max_gap_s: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate by timestamps without extrapolating or bridging invalid gaps.

    A target exactly on a valid observed sample remains valid even when the
    next/previous sample is invalid.  Returned zeroes at invalid targets are
    placeholders and must always be consumed with the returned mask.
    """
    values = np.asarray(values)
    times = _clock(times, len(values), "interpolation")
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 1 or not np.isfinite(targets).all():
        raise ValueError("interpolation: target timestamps must be finite and one-dimensional")
    if max_gap_s <= 0 or not np.isfinite(max_gap_s):
        raise ValueError("max_gap_s must be finite and positive")
    if valid is None:
        valid = np.ones(len(times), dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != times.shape:
        raise ValueError("interpolation: validity mask and clock lengths disagree")
    right = np.searchsorted(times, targets, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    exact_right = np.isclose(targets, times[right], rtol=0, atol=1e-8)
    exact_left = np.isclose(targets, times[left], rtol=0, atol=1e-8)
    right = np.where(exact_left & ~exact_right, left, right)
    exact = exact_right | exact_left
    left = np.where(exact, right, left)
    gap = times[right] - times[left]
    covered = (targets >= times[0] - 1e-8) & (targets <= times[-1] + 1e-8)
    covered &= valid[left] & valid[right] & (exact | ((gap > 0) & (gap <= max_gap_s + 1e-8)))
    alpha = np.divide(targets - times[left], gap, out=np.zeros_like(targets), where=gap > 0)
    shape = (len(targets),) + (1,) * (values.ndim - 1)
    out = values[left] * (1 - alpha.reshape(shape)) + values[right] * alpha.reshape(shape)
    out[~covered] = 0
    return out.astype(np.float32), covered


def semantic_collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors while preserving variable metadata as per-item objects."""
    if not items:
        raise ValueError("Cannot collate an empty batch")

    def visit(values: Sequence[Any], key: str = "") -> Any:
        if key == "metadata":
            return list(values)
        if isinstance(values[0], dict):
            return {name: visit([value[name] for value in values], name) for name in values[0]}
        return default_collate(values)

    return visit(items)


class SemanticMotionDataset(Dataset):
    """Query plus independent same-person and genuinely cross-person references.

    Returns ``query``, ``same_style_references``, ``positive_style_references``,
    ``donor_references`` and ``donor_anchor``.  Query tensors use [T,D]; the
    reference groups use [K,T,D]. ``valid`` requires motion, audio/content and
    trusted visual VA coverage.  ``motion_valid`` and ``va_valid`` retain their
    separate meanings.  Both same-person reference sets exclude query's clip,
    sentence and emotion, and are disjoint.  Donor identity is selected from a
    different person of the same dataset and emotion, preferably the same
    known intensity.  Its reference set excludes the donor anchor's sentence
    and emotion.  No neutral teacher or paired neutral record is used.

    Call set_epoch for fresh deterministic training crops; evaluation ignores
    epoch. ``subset_indices`` selects query rows only, keeping the reference
    pool within the explicitly requested split.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        split: str = "train",
        random_crop: bool = True,
        *,
        subset_indices: Sequence[int] | None = None,
        indices: Sequence[int] | None = None,
    ):
        data = {**cfg.get("data", {}), **cfg.get("semantic_data", cfg if "data" not in cfg else {})}
        self.cfg = data
        if split not in ("train", "val", "test"):
            raise ValueError("Use an explicit train/val/test split")
        self.split = split
        self.random_crop = bool(random_crop)
        self.epoch = 0
        self.seed = int(data.get("seed", 1234))
        self.window = int(data.get("window", 96))
        self.reference_count = int(data.get("reference_count", 2))
        self.content_context = int(data.get("content_context", 1))
        self.cache_size = int(data.get("cache_size", 16))
        self.max_va_gap_s = float(data.get("max_va_gap_s", 0.12))
        self.max_feature_gap_s = float(data.get("max_feature_gap_s", 0.12))
        self.va_confidence_min = float(data.get("va_confidence_min", 0.0))
        self.min_valid_frames = int(data.get("min_valid_frames", 2))
        self.motion_dim = int(data.get("motion_dim", 52))
        self.content_dim = int(data.get("content_dim", 768))
        self.audio_dim = int(data["audio_dim"]) if data.get("audio_dim") is not None else None
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        if self.window < 2 or self.reference_count < 1 or self.content_context < 1 or self.content_context % 2 == 0:
            raise ValueError("window>=2, reference_count>=1 and an odd positive content_context are required")
        if not 1 <= self.min_valid_frames <= self.window or self.cache_size < 0:
            raise ValueError("Invalid min_valid_frames or cache_size")
        if not 0 <= self.va_confidence_min <= 1:
            raise ValueError("va_confidence_min must be in [0,1]")
        self.emotion_classes = list(data.get("emotion_classes", EMOTIONS))
        manifest = data.get(f"{split}_manifest", data.get("manifest"))
        if not manifest:
            raise ValueError("semantic_data.manifest (or explicit split_manifest) is required")
        self.manifest = Path(manifest).expanduser().resolve()
        self.root = Path(data.get("root", self.manifest.parent)).expanduser().resolve()
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            if self.manifest.suffix.lower() == ".csv":
                records = list(csv.DictReader(handle))
            else:
                records = [json.loads(line) for line in handle if line.strip()]
        required = ("clip_id", "dataset", "speaker", "sentence_id", "emotion", "split", "motion_path", "content_path", "audio_path", "va_path")
        seen: set[tuple[str, str]] = set()
        speaker_splits: dict[tuple[str, str], str] = {}
        for row in records:
            for name in required:
                if row.get(name) is None or str(row[name]).strip() == "":
                    raise ValueError(f"Manifest missing {name}: {row.get('clip_id', '<unknown>')}")
            key = (str(row["dataset"]), str(row["clip_id"]))
            if row["split"] not in ("train", "val", "test"):
                raise ValueError(f"Unknown manifest split {row['split']!r}: {key}")
            if key in seen:
                raise ValueError(f"Duplicate clip in manifest: {key}")
            seen.add(key)
            speaker = (str(row["dataset"]), str(row["speaker"]))
            if speaker in speaker_splits and speaker_splits[speaker] != row["split"]:
                raise ValueError(f"Speaker appears in multiple splits: {speaker}")
            speaker_splits[speaker] = str(row["split"])
        self.items = [dict(row) for row in records if row["split"] == split]
        if not self.items:
            raise ValueError(f"No {split!r} clips in {self.manifest}")
        # Stable IDs across separate train/val manifests; never enumerate only
        # the datasets that happen to occur in the current evaluation subset.
        datasets = list(data.get("dataset_classes", ("mead", "cremad")))
        self.dataset_to_id = {name: i for i, name in enumerate(datasets)}
        speakers = sorted({(str(row["dataset"]), str(row["speaker"])) for row in self.items})
        self.speaker_to_id = {name: i for i, name in enumerate(speakers)}
        self._identity_indices: dict[tuple[str, str], list[int]] = {}
        for index, row in enumerate(self.items):
            if row["emotion"] not in self.emotion_classes:
                raise ValueError(f"Unknown emotion {row['emotion']!r} in {row['clip_id']}")
            if row["dataset"] not in self.dataset_to_id:
                raise ValueError(f"Unknown dataset {row['dataset']!r}")
            supplied = row.get("intensity_id") not in (None, "")
            if not supplied and "intensity_valid" not in row:
                raise ValueError(f"Missing explicit intensity validity: {row['clip_id']}")
            row["intensity_valid"] = _boolean(row["intensity_valid"], "intensity_valid") if "intensity_valid" in row else supplied
            if row["intensity_valid"] and not supplied:
                raise ValueError(f"Known intensity without intensity_id: {row['clip_id']}")
            row["intensity_id"] = _integer(row["intensity_id"], "intensity_id") if row["intensity_valid"] else -1
            levels = data.get("intensity_levels", {}).get(row["dataset"])
            if row["intensity_valid"] and (row["intensity_id"] < 0 or (levels is not None and row["intensity_id"] >= int(levels))):
                raise ValueError(f"Out-of-range intensity_id: {row['clip_id']}")
            row["source_audio_offset_s"] = float(row.get("source_audio_offset_s", 0.0))
            if not np.isfinite(row["source_audio_offset_s"]):
                raise ValueError(f"Nonfinite audio/video offset: {row['clip_id']}")
            for field in ("motion_path", "content_path", "audio_path", "va_path"):
                path = Path(str(row[field]))
                row[field] = str(path if path.is_absolute() else self.root / path)
                if not Path(row[field]).is_file():
                    raise FileNotFoundError(f"{row['clip_id']}: missing {field}: {row[field]}")
            identity = (str(row["dataset"]), str(row["speaker"]))
            self._identity_indices.setdefault(identity, []).append(index)
        if indices is not None and subset_indices is not None:
            raise ValueError("Specify indices or subset_indices, not both")
        subset_indices = indices if indices is not None else subset_indices
        self.indices = list(range(len(self.items))) if subset_indices is None else [int(i) for i in subset_indices]
        if not self.indices or len(set(self.indices)) != len(self.indices) or any(i < 0 or i >= len(self.items) for i in self.indices):
            raise ValueError("subset_indices must contain distinct valid query indices")
        self._reference_counts = [len(self._references_for(i)) for i in range(len(self.items))]
        # Shared donor groups avoid a quadratic table of per-clip donor lists.
        self._donor_groups: dict[tuple[Any, ...], list[int]] = {}
        for index, row in enumerate(self.items):
            if self._reference_counts[index] >= self.reference_count:
                key = (row["dataset"], row["emotion"])
                self._donor_groups.setdefault(key, []).append(index)
                if row["intensity_valid"]:
                    self._donor_groups.setdefault((*key, row["intensity_id"]), []).append(index)
        validated_identities: set[tuple[Any, ...]] = set()
        for index in self.indices:
            row = self.items[index]
            if self._reference_counts[index] < 2 * self.reference_count:
                raise ValueError(f"{row['clip_id']}: need {2*self.reference_count} distinct same-person references with different emotion AND sentence; found {self._reference_counts[index]}")
            key = (row["dataset"], row["speaker"], row["emotion"], row["intensity_id"])
            if key not in validated_identities and not self._donors_for(index):
                raise ValueError(f"{row['clip_id']}: no different-speaker, same-dataset/emotion donor with independent references")
            validated_identities.add(key)

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _references_for(self, index: int) -> list[int]:
        row = self.items[index]
        return [i for i in self._identity_indices[(str(row["dataset"]), str(row["speaker"]))]
                if i != index and self.items[i]["sentence_id"] != row["sentence_id"] and self.items[i]["emotion"] != row["emotion"]]

    def _neutral_references_for(self, index: int) -> list[int]:
        """Return same-speaker neutral clips for the identity anchor.

        Neutral clips are sampled across sentences so the identity code cannot
        memorize one utterance's residual articulation.  The caller still
        validates that enough ordinary references exist for the legacy donor
        protocol; neutral references are an additional, optional view.
        """
        row = self.items[index]
        candidates = [
            i for i in self._identity_indices[(str(row["dataset"]), str(row["speaker"]))]
            if i != index and self.items[i]["emotion"] == "neutral"
            and self.items[i]["sentence_id"] != row["sentence_id"]
        ]
        return candidates

    def _donors_for(self, index: int) -> list[int]:
        row = self.items[index]
        key = (row["dataset"], row["emotion"])
        exact = [i for i in self._donor_groups.get((*key, row["intensity_id"]), [])
                 if self.items[i]["speaker"] != row["speaker"]] if row["intensity_valid"] else []
        if exact or self.cfg.get("require_intensity_matched_donor", False):
            return exact
        return [i for i in self._donor_groups.get(key, []) if self.items[i]["speaker"] != row["speaker"]]

    def _rng(self, index: int) -> np.random.Generator:
        payload = f"{self.seed}:{self.epoch if self.random_crop else 0}:{index}".encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))

    def _load_arrays(self, index: int) -> dict[str, np.ndarray]:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        row = self.items[index]
        clip = str(row["clip_id"])
        with np.load(row["motion_path"], allow_pickle=False) as archive:
            _verify_identity(archive, row, f"{clip} motion")
            motion = _matrix(archive[self.cfg.get("motion_key", "coeffs")], self.motion_dim, f"{clip} motion")
            if self.cfg.get("motion_times_key", "times") in archive:
                motion_times = archive[self.cfg.get("motion_times_key", "times")]
            elif row.get("motion_clock") == "native_fps" and "fps" in row and "motion_start_s" in row:
                fps, start = float(row["fps"]), float(row["motion_start_s"])
                if not np.isfinite(fps) or fps <= 0 or not np.isfinite(start):
                    raise ValueError(f"{clip}: invalid declared native motion FPS/start")
                motion_times = start + np.arange(len(motion), dtype=np.float64) / fps
            else:
                raise ValueError(f"{clip}: motion timestamps missing; declare native_fps, fps and motion_start_s explicitly")
            times = _clock(motion_times, len(motion), f"{clip} motion")
            motion_mask_key = self.cfg.get("motion_valid_key")
            observed_motion = np.asarray(archive[motion_mask_key]) if motion_mask_key else np.ones(len(motion), dtype=bool)
            if observed_motion.shape != (len(motion),) or not np.isin(observed_motion, [0, 1]).all():
                raise ValueError(f"{clip}: invalid motion observation mask")
            observed_motion = observed_motion.astype(bool)
            channel_mask_key = self.cfg.get("channel_mask_key")
            channel_mask = np.asarray(archive[channel_mask_key]) if channel_mask_key else np.ones(self.motion_dim, dtype=bool)
            if channel_mask.shape != (self.motion_dim,) or not np.isin(channel_mask, [0, 1]).all():
                raise ValueError(f"{clip}: invalid motion channel mask")
            channel_mask = channel_mask.astype(bool)
        audio_targets = times + row["source_audio_offset_s"]
        with np.load(row["content_path"], allow_pickle=False) as archive:
            _verify_identity(archive, row, f"{clip} content")
            source_content = _matrix(archive[self.cfg.get("content_key", "content")], self.content_dim, f"{clip} content")
            content_times = _clock(archive[self.cfg.get("content_times_key", "times")], len(source_content), f"{clip} content")
        content, content_valid = interpolate_clock(source_content, content_times, audio_targets, max_gap_s=self.max_feature_gap_s)
        centers = np.searchsorted(content_times, audio_targets).clip(0, len(content_times) - 1)
        neighborhood = np.arange(self.content_context) - self.content_context // 2
        context_indices = (centers[:, None] + neighborhood[None]).clip(0, len(source_content) - 1)
        content_context = source_content[context_indices].copy()
        content_context[~content_valid] = 0
        with np.load(row["audio_path"], allow_pickle=False) as archive:
            _verify_identity(archive, row, f"{clip} audio")
            keys = self.cfg.get("audio_keys", ["feat"])
            if not isinstance(keys, (list, tuple)) or not keys:
                raise ValueError("audio_keys must explicitly name one or more arrays")
            parts = [_matrix(archive[key], None, f"{clip} audio.{key}") for key in keys]
            if len({len(part) for part in parts}) != 1:
                raise ValueError(f"{clip}: explicitly concatenated audio fields have different clocks/lengths")
            source_audio = _matrix(np.concatenate(parts, axis=-1), self.audio_dim, f"{clip} audio")
            audio_times = _clock(archive[self.cfg.get("audio_times_key", "times")], len(source_audio), f"{clip} audio")
        audio, audio_valid = interpolate_clock(source_audio, audio_times, audio_targets, max_gap_s=self.max_feature_gap_s)
        with np.load(row["va_path"], allow_pickle=False) as archive:
            _verify_identity(archive, row, f"{clip} VA")
            source = str(archive["source_modality"].item()) if "source_modality" in archive else None
            if source not in (None, "visual") or (source != "visual" and row.get("va_source") not in ("visual_teacher", "visual_annotation")):
                raise ValueError(f"{clip}: VA must declare visual provenance; audio affect is not a visual VA label")
            va = _matrix(archive["va"], 2, f"{clip} VA")
            va_times = _clock(archive["times"], len(va), f"{clip} VA")
            source_valid = np.asarray(archive["valid"])
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
            if source_valid.shape != (len(va),) or confidence.shape != (len(va),):
                raise ValueError(f"{clip}: VA valid/confidence shape must match its own clock")
            if not np.isin(source_valid, [0, 1]).all() or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
                raise ValueError(f"{clip}: invalid VA validity/confidence")
            if np.any(np.abs(va) > 1.0001):
                raise ValueError(f"{clip}: expected declared VA range [-1,1]")
            source_valid = source_valid.astype(bool) & (confidence > self.va_confidence_min)
        va_aligned, va_valid = interpolate_clock(va, va_times, times, valid=source_valid, max_gap_s=self.max_va_gap_s)
        va_confidence, _ = interpolate_clock(confidence, va_times, times, valid=source_valid, max_gap_s=self.max_va_gap_s)
        motion_valid = observed_motion & content_valid & audio_valid
        valid = motion_valid & va_valid
        if valid.sum() < self.min_valid_frames:
            raise ValueError(f"{clip}: insufficient synchronized motion/audio/visual-VA frames ({valid.sum()})")
        arrays = dict(motion=motion, content=content, content_context=content_context, audio=audio, va=va_aligned,
                      va_confidence=va_confidence, valid=valid, motion_valid=motion_valid, va_valid=va_valid, times=times,
                      channel_mask=channel_mask)
        if self.cache_size:
            self._cache[index] = arrays
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def _crop(self, index: int, rng: np.random.Generator) -> dict[str, Any]:
        arrays = self._load_arrays(index)
        row = self.items[index]
        length = len(arrays["times"])
        last = max(0, length - self.window)
        starts = np.arange(last + 1)
        prefix = np.concatenate(([0], np.cumsum(arrays["valid"], dtype=np.int64)))
        counts = prefix[np.minimum(starts + self.window, length)] - prefix[starts]
        candidates = starts[counts >= self.min_valid_frames]
        if len(candidates) == 0:
            raise ValueError(f"{row['clip_id']}: no window has {self.min_valid_frames} valid synchronized frames")
        start = int(rng.choice(candidates)) if self.random_crop else int(candidates[np.argmin(np.abs(candidates - last / 2))])
        count = min(self.window, length - start)
        result: dict[str, Any] = {}
        for name, value in arrays.items():
            if name == "channel_mask":
                result[name] = torch.from_numpy(value.copy())
                continue
            padded = np.zeros((self.window, *value.shape[1:]), dtype=value.dtype)
            padded[:count] = value[start:start + count]
            result[name] = torch.from_numpy(padded)
        result["emotion_id"] = torch.tensor(self.emotion_classes.index(row["emotion"]), dtype=torch.long)
        result["intensity_id"] = torch.tensor(row["intensity_id"], dtype=torch.long)
        result["intensity_valid"] = torch.tensor(row["intensity_valid"], dtype=torch.bool)
        result["dataset_id"] = torch.tensor(self.dataset_to_id[row["dataset"]], dtype=torch.long)
        result["speaker_id"] = torch.tensor(self.speaker_to_id[(str(row["dataset"]), str(row["speaker"]))], dtype=torch.long)
        result["metadata"] = {key: row[key] for key in ("clip_id", "dataset", "speaker", "sentence_id", "emotion", "split", "source_audio_offset_s")}
        result["metadata"].update(crop_start=start, valid_frames=int(result["valid"].sum()))
        return result

    def _reference_group(self, indices: Sequence[int], rng: np.random.Generator) -> dict[str, Any]:
        return semantic_collate([self._crop(index, rng) for index in indices])

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        rng = self._rng(index)
        selected = rng.choice(self._references_for(index), size=2 * self.reference_count, replace=False).tolist()
        neutral_pool = self._neutral_references_for(index)
        neutral_count = min(self.reference_count, len(neutral_pool))
        neutral_refs = (
            rng.choice(neutral_pool, size=neutral_count, replace=False).tolist()
            if neutral_count else []
        )
        donor = int(rng.choice(self._donors_for(index)))
        donor_refs = rng.choice(self._references_for(donor), size=self.reference_count, replace=False).tolist()
        row, donor_row = self.items[index], self.items[donor]
        matched_intensity = row["intensity_valid"] and donor_row["intensity_valid"] and row["intensity_id"] == donor_row["intensity_id"]
        result = {
            "query": self._crop(index, rng),
            "same_style_references": self._reference_group(selected[:self.reference_count], rng),
            "positive_style_references": self._reference_group(selected[self.reference_count:], rng),
            "donor_references": self._reference_group(donor_refs, rng),
            "donor_anchor": self._crop(donor, rng),
            "donor_intensity_matched": torch.tensor(matched_intensity, dtype=torch.bool),
        }
        # Fall back to the ordinary same-speaker references for manifests
        # without neutral coverage; this keeps the batch contract stable while
        # allowing neutral-heavy manifests to provide the identity anchor.
        result["neutral_style_references"] = (
            self._reference_group(neutral_refs, rng)
            if neutral_refs else result["same_style_references"]
        )
        return result
