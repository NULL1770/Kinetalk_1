"""Small, auditable neutral-anchor datasets on the native motion clock.

This contract deliberately has no VA or paired-neutral target. Native artifacts
already contain synchronized audio/content; their original timestamps and masks
are retained. A cache stores queries separately from fixed per-person references.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA = "neutral_affect_pilot_v1"
EMOTIONS = ("neutral", "angry", "contempt", "disgust", "fear", "happy", "sad", "surprise")


def native_clip(row: Mapping[str, Any], root: str | Path, *, window: int = 96,
                min_valid_frames: int = 32) -> dict[str, Any]:
    """Read one independently recorded clip, with a deterministic valid crop."""
    path = Path(root) / str(row["artifact"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if row.get("artifact_sha256") and digest != row["artifact_sha256"]:
        raise ValueError(f"Artifact hash changed: {path}")
    with np.load(path, allow_pickle=False) as source:
        required = ("motion", "content", "audio", "times", "mask", "channel_mask", "provenance")
        if any(key not in source for key in required):
            raise ValueError(f"Not a complete native artifact: {path}")
        values = {key: np.asarray(source[key]).copy() for key in required[:-1]}
        provenance = json.loads(str(source["provenance"].item()))
    if not str(provenance.get("schema", "")).startswith("native_affect_style_v4"):
        raise ValueError(f"Unsupported artifact schema: {path}")
    if provenance.get("clock_evidence") != "embedded_video":
        raise ValueError(f"Native video clock evidence missing: {path}")
    length = len(values["times"])
    for key, dim in (("motion", 52), ("content", 768), ("audio", 83)):
        if values[key].shape != (length, dim) or not np.isfinite(values[key]).all():
            raise ValueError(f"Invalid {key} array: {path}")
    times = values["times"].astype(np.float64)
    if length < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Nonmonotonic native timestamps: {path}")
    if not np.allclose(np.diff(times), 1.0 / float(provenance["fps"]), atol=1e-5):
        raise ValueError(f"Native clock is not the recorded FPS: {path}")
    for key, shape in (("mask", (length,)), ("channel_mask", (52,))):
        if values[key].shape != shape or not np.isin(values[key], [0, 1]).all():
            raise ValueError(f"Invalid {key}: {path}")
    if window < min_valid_frames or min_valid_frames < 2:
        raise ValueError("window >= min_valid_frames >= 2 is required")
    valid = values["mask"].astype(bool)
    starts = np.arange(max(1, length - window + 1))
    prefix = np.r_[0, np.cumsum(valid)]
    counts = prefix[np.minimum(starts + window, length)] - prefix[starts]
    candidates = starts[counts >= min_valid_frames]
    if not len(candidates):
        raise ValueError(f"No crop has {min_valid_frames} valid frames: {path}")
    start = int(candidates[np.argmin(np.abs(candidates - max(0, length - window) / 2))])
    count = min(window, length - start)
    out: dict[str, Any] = {}
    for key in ("motion", "content", "audio"):
        cropped = np.zeros((window, values[key].shape[-1]), dtype=np.float32)
        cropped[:count] = values[key][start:start + count]
        out[key] = torch.from_numpy(cropped)
    cropped_valid = np.zeros(window, dtype=bool)
    cropped_valid[:count] = valid[start:start + count]
    out["valid"] = torch.from_numpy(cropped_valid)
    out["motion_valid"] = out["valid"].clone()
    # Padded times continue the clock but padded observations always stay invalid.
    padded_times = times[start] + np.arange(window, dtype=np.float64) / float(provenance["fps"])
    padded_times[:count] = times[start:start + count]
    out["times"] = torch.from_numpy(padded_times)
    out["channel_mask"] = torch.from_numpy(values["channel_mask"].astype(bool))
    for key in ("speaker_id", "emotion_id", "intensity_id"):
        out[key] = torch.tensor(int(row[key]), dtype=torch.long)
    out["intensity_valid"] = torch.tensor(True)
    out["dataset_id"] = torch.tensor(0, dtype=torch.long)
    out["clip_id"] = str(row["clip_id"])
    out["sentence_id"] = str(row["sentence"])
    out["speaker"] = str(row["speaker"])
    out["metadata"] = {"artifact": str(path), "artifact_sha256": digest,
                       "source_split": str(row["split"]), "pilot_split": row.get("pilot_split"),
                       "crop_start": start, "observed_frames": count, "provenance": provenance}
    return out


def stack_clips(clips: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack tensors and preserve metadata without recursively batching it."""
    if not clips:
        raise ValueError("Cannot stack empty clips")
    return {key: torch.stack([clip[key] for clip in clips]) if torch.is_tensor(value)
            else [clip[key] for clip in clips] for key, value in clips[0].items()}


class NeutralAffectDataset(Dataset):
    """Read a materialized cache; references are never substituted from emotions."""
    def __init__(self, path: str | Path):
        self.cache = torch.load(Path(path), map_location="cpu", weights_only=False)
        if self.cache.get("schema") != SCHEMA:
            raise ValueError("Unsupported neutral-affect cache")
        self.queries = self.cache["queries"]
        self.identity_references = self.cache["identity_references"]
        if not self.queries:
            raise ValueError("Empty query cache")
        for query in self.queries:
            refs = self.identity_references[int(query["speaker_id"])]
            if len(refs) < 4 or len({r["clip_id"] for r in refs}) != len(refs):
                raise ValueError("At least four distinct neutral references are required")
            if any(int(r["emotion_id"]) != 0 or int(r["speaker_id"]) != int(query["speaker_id"])
                   or r["clip_id"] == query["clip_id"] or r["sentence_id"] == query["sentence_id"] for r in refs):
                raise ValueError("Reference identity/neutrality/independence violation")

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        query = self.queries[index]
        refs = self.identity_references[int(query["speaker_id"])]
        return {"query": query, "neutral_identity_references": stack_clips(refs),
                "identity_anchor_references": stack_clips(refs[:len(refs) // 2]),
                "identity_positive_references": stack_clips(refs[len(refs) // 2:])}


def neutral_collate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {key: stack_clips([item[key] for item in items]) for key in items[0]}
