"""Lazy train/validation HDTF windows and independent single-reference motion.

This loader preserves source clocks and missing-label masks.  HDTF has no
verified neutral targets and therefore supplies neither neutral anchors nor
emotion/intensity supervision.  Source and identity grouping are checked from
manifest metadata before any target artifact is opened.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

SCHEMA = "hdtf-window-loader-v2.1"
FRAME_KEYS = ("audio_features", "content", "motion", "times", "valid", "motion_valid")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _split_checks(rows: list[dict]) -> None:
    source = defaultdict(set)
    speaker = defaultdict(set)
    clip_ids = set()
    for row in rows:
        for key in ("source_id", "source_group", "speaker_group", "split", "clip_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"HDTF manifest requires {key}")
        if row["clip_id"] in clip_ids:
            raise ValueError("duplicate HDTF window ID")
        clip_ids.add(row["clip_id"])
        canonical_split = "val" if row["split"] == "validation" else row["split"]
        source[row["source_group"]].add(canonical_split)
        speaker[row["speaker_group"]].add(canonical_split)
        if int(row["start_frame"]) < 0 or int(row["end_frame"]) <= int(row["start_frame"]):
            raise ValueError("invalid HDTF frame interval")
    for namespace, groups in (("source", source), ("speaker", speaker)):
        for group, splits in groups.items():
            if len(splits) > 1 and splits & {"train", "val", "validation"}:
                raise ValueError(f"HDTF {namespace} group crosses splits: {group}")


class HDTFSplit(dict):
    """PackedSplit-like metadata with source artifacts loaded only on demand.

    ``batch(ids, device, keys=...)`` returns float32 audio/motion and float64
    source times. ``single_references`` selects one disjoint same-speaker
    reference per query. A missing reference has ``reference_valid=False``;
    it is never replaced by the query itself. ``speaker_offset`` namespaces
    numeric IDs relative to a MEAD dataset combined by the caller.
    """

    def __init__(self, manifest: str | Path, feature_root: str | Path, *, split: str,
                 speaker_offset: int = 100000, source_root: str | Path | None = None,
                 cache_sources: int = 2, verify_hashes: bool = True,
                 require_features: bool = True, reference_buffer_seconds: float = .5,
                 allow_missing_features: bool = False, motion_policy: str = "raw",
                 postprocess_manifest: str | Path | None = None):
        if split not in {"train", "val", "validation"}:
            raise ValueError("HDTFSplit is train/validation only; test/quarantine targets are sealed")
        super().__init__()
        self.manifest, self.feature_root = Path(manifest), Path(feature_root)
        self.source_root = Path(source_root) if source_root is not None else None
        self.role = "val" if split == "validation" else split
        if motion_policy not in {"raw", "sg5"}:
            raise ValueError("motion_policy must explicitly be raw or sg5")
        self.motion_policy = motion_policy
        postprocess_rows = {}
        if motion_policy == "sg5" and postprocess_manifest is not None:
            for rec in _rows(Path(postprocess_manifest)):
                if rec.get("split") in {self.role, "validation" if self.role == "val" else self.role}:
                    postprocess_rows[rec["source_id"]] = rec
        self.cache_sources, self.verify_hashes = max(1, int(cache_sources)), bool(verify_hashes)
        self.reference_buffer_seconds = float(reference_buffer_seconds)
        if self.reference_buffer_seconds < 0:
            raise ValueError("reference buffer must be nonnegative")
        all_rows = _rows(self.manifest)
        _split_checks(all_rows)
        self.rows = []
        self._receipts: dict[str, dict] = {}
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._extras: dict[str, list[torch.Tensor]] = {}
        self.unavailable_sources: list[str] = []
        for row in all_rows:
            if row["split"] not in {self.role, "validation" if self.role == "val" else self.role}:
                continue
            sid = row["source_id"]
            receipt = self.feature_root / "receipts" / f"{sid}.json"
            features = self.feature_root / "features" / f"{sid}.npz"
            if not receipt.is_file() or not features.is_file():
                self.unavailable_sources.append(sid)
                if allow_missing_features:
                    continue
                raise FileNotFoundError(f"HDTF feature receipt missing for {sid}")
            if sid not in self._receipts:
                rec = json.loads(receipt.read_text(encoding="utf-8"))
                if rec.get("status") not in {"ok", "cached"} or rec.get("split") not in {self.role, "validation" if self.role == "val" else self.role}:
                    raise ValueError("feature receipt split/status does not match window")
                if rec.get("source_id") != sid or rec.get("speaker_group") != row["speaker_group"]:
                    raise ValueError("feature receipt source/identity does not match window")
                self._receipts[sid] = rec
            copied = dict(row, features_path=str(features))
            if motion_policy == "sg5":
                post = postprocess_rows.get(sid, row)
                required = ("motion_processed_path", "motion_processed_sha256", "raw_motion_sha256",
                            "postprocess_schema", "candidate_window")
                if any(k not in post for k in required):
                    raise ValueError(f"SG5 policy requires complete postprocess receipt for {sid}")
                if post["postprocess_schema"] != "hdtf-motion-postprocess-v2.1" or int(post["candidate_window"]) != 5:
                    raise ValueError(f"SG5 receipt has a different schema/window for {sid}")
                if post["raw_motion_sha256"] != self._receipts[sid]["motion_sha256"]:
                    raise ValueError(f"SG5 receipt raw lineage differs from audio-feature source for {sid}")
                if post.get("speaker_group", row["speaker_group"]) != row["speaker_group"]:
                    raise ValueError("SG5 receipt speaker does not match window")
                copied.update({key: post[key] for key in required})
            # Emotion is unknown even if an unverified downstream manifest
            # accidentally wrote 'neutral'. This prevents all-HDTF neutral CE.
            copied.update(emotion_id=-1, emotion_valid=False, intensity_id=-1, intensity_valid=False)
            self.rows.append(copied)
        # A shared manifest mapping keeps numeric identities stable across
        # independently opened train and validation splits.
        people = sorted({r["speaker_group"] for r in all_rows if r["split"] in {"train", "val", "validation"}})
        self.speaker_ids = {name: int(speaker_offset) + i for i, name in enumerate(people)}
        self["_lengths"] = torch.tensor([int(r["end_frame"]) - int(r["start_frame"]) for r in self.rows], dtype=torch.long)
        self["speaker_id"] = torch.tensor([self.speaker_ids[r["speaker_group"]] for r in self.rows], dtype=torch.long)
        self["emotion_id"] = torch.full((len(self.rows),), -1, dtype=torch.long)
        self["emotion_valid"] = torch.zeros(len(self.rows), dtype=torch.bool)
        self["intensity_id"] = torch.full((len(self.rows),), -1, dtype=torch.long)
        self["intensity_valid"] = torch.zeros(len(self.rows), dtype=torch.bool)
        self["dataset_id"] = torch.ones(len(self.rows), dtype=torch.long)
        self["anchors"] = torch.zeros(len(self.rows), 52)
        self["anchor_valid"] = torch.zeros(len(self.rows), 52, dtype=torch.bool)
        for key in ("clip_id", "source_id", "source_group", "speaker_group"):
            self[key] = [r[key] for r in self.rows]
        self["speaker"] = ["HDTF:" + r["speaker_group"] for r in self.rows]
        self["sentence_id"] = ["HDTF-source:" + r["source_group"] for r in self.rows]
        self["metadata"] = [{"dataset": "HDTF", "source_group": r["source_group"],
                             "identity_status": r.get("identity_status", "unverified"),
                             "emotion_supervision": "missing", "neutral_target_valid": False,
                             "motion_target_policy": self.motion_policy,
                             "event_context_start_frame": r.get("context_start_frame"),
                             "event_context_end_frame": r.get("context_end_frame"),
                             "partial_source_processing": r.get("partial_source_processing", True),
                             "left_source_boundary": r.get("left_source_boundary", False),
                             "right_source_boundary": r.get("right_source_boundary", False)} for r in self.rows]
        self.provenance = {"schema": SCHEMA, "manifest_sha256": _hash(self.manifest),
                           "split": self.role, "test_loaded": False, "quarantine_loaded": False,
                           "emotion_targets_known": 0, "neutral_targets_known": 0,
                           "motion_target_policy": self.motion_policy,
                           "postprocess_manifest_sha256": _hash(Path(postprocess_manifest)) if postprocess_manifest is not None else None,
                           "speaker_namespace": "HDTF:", "speaker_offset": speaker_offset,
                           "independent_reference_buffer_seconds": self.reference_buffer_seconds,
                           "available_sources": len(self._receipts),
                           "missing_feature_sources": sorted(set(self.unavailable_sources))}
        self._speaker_rows = defaultdict(list)
        for i, row in enumerate(self.rows):
            self._speaker_rows[row["speaker_group"]].append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def _motion_path(self, row: dict) -> Path:
        if self.motion_policy == "sg5":
            return Path(row["motion_processed_path"])
        if self.source_root is not None:
            return self.source_root / "motion_raw" / f"{row['source_id']}.npz"
        return Path(row["motion_path"])

    def _source(self, row: dict) -> dict:
        sid = row["source_id"]
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]
        motion_path, feature_path = self._motion_path(row), Path(row["features_path"])
        receipt = self._receipts[sid]
        if self.verify_hashes:
            target_hash = row["motion_processed_sha256"] if self.motion_policy == "sg5" else receipt.get("motion_sha256")
            if _hash(motion_path) != target_hash:
                raise ValueError(f"HDTF motion artifact hash changed: {sid}")
            if _hash(feature_path) != receipt.get("features_sha256"):
                raise ValueError(f"HDTF feature artifact hash changed: {sid}")
        with np.load(motion_path, allow_pickle=False) as m:
            if self.motion_policy == "sg5":
                if str(m["postprocess_schema"].item()) != "hdtf-motion-postprocess-v2.1" or int(m["candidate_window"]) != 5:
                    raise ValueError("SG5 artifact schema/window differs from selected motion policy")
                if str(m["raw_motion_sha256"].item()) != receipt["motion_sha256"]:
                    raise ValueError("SG5 artifact raw lineage differs from feature receipt")
            data = {key: np.asarray(m[key]).copy() for key in ("motion", "times", "motion_valid", "channel_mask")}
            data["observed"] = np.asarray(m["valid"], bool).copy() if "valid" in m else data["motion_valid"].astype(bool)
        with np.load(feature_path, allow_pickle=False) as f:
            data["audio_features"] = np.asarray(f["audio_features"]).copy()
            data["feature_valid"] = np.asarray(f["feature_valid"], bool).copy()
            feature_times = np.asarray(f["times"], dtype=np.float64)
        n = len(data["times"])
        if data["motion"].shape != (n, 52) or data["audio_features"].shape != (n, 1540):
            raise ValueError(f"HDTF source array dimensions disagree: {sid}")
        if not np.array_equal(data["times"], feature_times) or not np.all(np.diff(feature_times) > 0):
            raise ValueError(f"HDTF audio/motion source clocks disagree: {sid}")
        if data["channel_mask"].shape != (52,):
            raise ValueError("HDTF loader expects one observed-channel mask per source")
        if data["motion_valid"].shape != (n,) or data["feature_valid"].shape != (n,):
            raise ValueError("HDTF source validity dimensions disagree")
        self._cache[sid] = data
        while len(self._cache) > self.cache_sources:
            self._cache.popitem(last=False)
        return data

    def set_extra(self, key: str, values: list[torch.Tensor]) -> None:
        if len(values) != len(self.rows) or any(len(v) != int(self["_lengths"][i]) for i, v in enumerate(values)):
            raise ValueError("HDTF extra cache must match native window lengths")
        self._extras[key] = values

    def batch(self, ids: torch.Tensor | Iterable[int], device: str | torch.device = "cpu", *, keys=None) -> dict:
        ids = list(map(int, ids.detach().cpu().tolist())) if torch.is_tensor(ids) else list(map(int, ids))
        if not ids:
            raise ValueError("cannot load an empty HDTF batch")
        width = max(int(self["_lengths"][i]) for i in ids)
        arrays = {"audio_features": torch.zeros(len(ids), width, 1540),
                  "motion": torch.zeros(len(ids), width, 52),
                  "times": torch.zeros(len(ids), width, dtype=torch.float64),
                  "valid": torch.zeros(len(ids), width, dtype=torch.bool),
                  "motion_valid": torch.zeros(len(ids), width, dtype=torch.bool),
                  "channel_mask": torch.zeros(len(ids), 52, dtype=torch.bool)}
        for j, i in enumerate(ids):
            row = self.rows[i]; data = self._source(row)
            start, end = int(row["start_frame"]), int(row["end_frame"])
            if end > len(data["times"]):
                raise ValueError("HDTF window falls beyond source artifact")
            n = end - start; sl = slice(start, end)
            arrays["audio_features"][j, :n] = torch.from_numpy(data["audio_features"][sl].astype(np.float32))
            arrays["motion"][j, :n] = torch.from_numpy(data["motion"][sl].astype(np.float32))
            arrays["times"][j, :n] = torch.from_numpy(data["times"][sl])
            arrays["times"][j, n:] = arrays["times"][j, n - 1] + torch.arange(1, width - n + 1, dtype=torch.float64) / 25.
            arrays["motion_valid"][j, :n] = torch.from_numpy(data["motion_valid"][sl].astype(bool))
            arrays["valid"][j, :n] = torch.from_numpy((data["motion_valid"][sl] & data["feature_valid"][sl]).astype(bool))
            arrays["channel_mask"][j] = torch.from_numpy(data["channel_mask"].astype(bool))
            expected_start = float(row.get("start_seconds", arrays["times"][j, 0]))
            if not np.isclose(float(arrays["times"][j, 0]), expected_start, atol=1e-7):
                raise ValueError("HDTF manifest window/source clock disagrees")
        arrays["audio_features"] = torch.where(arrays["valid"][..., None], arrays["audio_features"], 0.)
        arrays["content"] = arrays["audio_features"][..., :768]
        for key in ("speaker_id", "emotion_id", "emotion_valid", "intensity_id", "intensity_valid", "dataset_id", "anchors", "anchor_valid"):
            arrays[key] = self[key][ids]
        # This explicit mask makes it impossible to train B0 as though every
        # unlabelled HDTF performance were a neutral utterance.
        arrays["neutral_target_valid"] = torch.zeros(len(ids), dtype=torch.bool)
        for key, values in self._extras.items():
            chunks = [values[i] for i in ids]
            shape = (len(ids), width, *chunks[0].shape[1:])
            value = torch.zeros(shape, dtype=chunks[0].dtype)
            for j, chunk in enumerate(chunks):
                value[j, :len(chunk)] = chunk
            arrays[key] = value
        out = {key: value.to(device) for key, value in arrays.items() if keys is None or key in keys or (key == "audio_features" and "content" in keys)}
        for key in ("clip_id", "sentence_id", "speaker", "source_id", "source_group", "speaker_group", "metadata"):
            if keys is None or key in keys:
                out[key] = [self[key][i] for i in ids]
        return out

    def reference_candidates(self, query_index: int) -> list[int]:
        query = self.rows[int(query_index)]
        candidates = []
        other_sources = []
        buffer = self.reference_buffer_seconds
        for j in self._speaker_rows[query["speaker_group"]]:
            ref = self.rows[j]
            if ref["clip_id"] == query["clip_id"]:
                continue
            if ref["source_group"] != query["source_group"]:
                other_sources.append(j)
                continue
            qstart, qend = float(query["start_seconds"]), float(query["end_seconds"])
            rstart, rend = float(ref["start_seconds"]), float(ref["end_seconds"])
            if rend + buffer <= qstart or qend + buffer <= rstart:
                candidates.append(j)
        return other_sources or candidates

    def context_batch(self, ids: torch.Tensor | Iterable[int], device="cpu") -> dict:
        """Load motion/audio context for event parsing before cropping targets.

        Context never jumps to another source and its absolute timestamps are
        preserved. Events touching this context boundary remain censored by
        ``teacher_targets``. Call ``crop_context_tensor`` only after B0 and
        observed event targets have been derived on these extended windows.
        """
        ids = list(map(int, ids.detach().cpu().tolist())) if torch.is_tensor(ids) else list(map(int, ids))
        if not ids:
            raise ValueError("cannot load an empty HDTF context batch")
        slices = []
        for i in ids:
            row = self.rows[i]; data = self._source(row)
            lo = max(0, int(row.get("context_start_frame", row["start_frame"])))
            hi = min(len(data["times"]), int(row.get("context_end_frame", row["end_frame"])))
            if lo > int(row["start_frame"]) or hi < int(row["end_frame"]):
                raise ValueError("event context does not contain query window")
            slices.append((i, data, lo, hi))
        width = max(hi - lo for _, _, lo, hi in slices)
        out = {"audio_features": torch.zeros(len(ids), width, 1540),
               "motion": torch.zeros(len(ids), width, 52),
               "times": torch.zeros(len(ids), width, dtype=torch.float64),
               "valid": torch.zeros(len(ids), width, dtype=torch.bool),
               "motion_valid": torch.zeros(len(ids), width, dtype=torch.bool),
               "channel_mask": torch.zeros(len(ids), 52, dtype=torch.bool),
               "query_offset": torch.tensor([int(self.rows[i]["start_frame"]) - lo for i, _, lo, _ in slices]),
               "query_lengths": self["_lengths"][ids].clone()}
        for j, (_, data, lo, hi) in enumerate(slices):
            n = hi - lo; sl = slice(lo, hi)
            out["audio_features"][j, :n] = torch.from_numpy(data["audio_features"][sl].astype(np.float32))
            out["motion"][j, :n] = torch.from_numpy(data["motion"][sl].astype(np.float32))
            out["times"][j, :n] = torch.from_numpy(data["times"][sl])
            out["times"][j, n:] = out["times"][j, n - 1] + torch.arange(1, width - n + 1, dtype=torch.float64) / 25.
            out["motion_valid"][j, :n] = torch.from_numpy(data["motion_valid"][sl].astype(bool))
            out["valid"][j, :n] = torch.from_numpy((data["motion_valid"][sl] & data["feature_valid"][sl]).astype(bool))
            out["channel_mask"][j] = torch.from_numpy(data["channel_mask"].astype(bool))
        out["audio_features"] = torch.where(out["valid"][..., None], out["audio_features"], 0.)
        out["content"] = out["audio_features"][..., :768]
        out["source_id"] = [self.rows[i]["source_id"] for i in ids]
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in out.items()}

    @staticmethod
    def crop_context_tensor(tensor: torch.Tensor, context: dict) -> torch.Tensor:
        if tensor.shape[0] != len(context["query_lengths"]) or tensor.ndim < 2:
            raise ValueError("context target must be frame aligned [B,T,...]")
        width = int(context["query_lengths"].max())
        out = torch.zeros((len(context["query_lengths"]), width, *tensor.shape[2:]), dtype=tensor.dtype, device=tensor.device)
        for j, (offset, length) in enumerate(zip(context["query_offset"].tolist(), context["query_lengths"].tolist())):
            out[j, :length] = tensor[j, offset:offset + length]
        return out

    def eligible_reference_indices(self) -> list[int]:
        return [i for i in range(len(self.rows)) if self.reference_candidates(i)]

    def single_references(self, query_ids: torch.Tensor | Iterable[int], device="cpu", *, seed: int = 47) -> dict:
        query_ids = list(map(int, query_ids.detach().cpu().tolist())) if torch.is_tensor(query_ids) else list(map(int, query_ids))
        rng = random.Random(seed)
        selected = [rng.choice(c) if (c := self.reference_candidates(i)) else None for i in query_ids]
        valid = torch.tensor([i is not None for i in selected], dtype=torch.bool)
        present = [i for i in selected if i is not None]
        width = max((int(self["_lengths"][i]) for i in present), default=1)
        out = {"motion": torch.zeros(len(query_ids), width, 52), "valid": torch.zeros(len(query_ids), width, dtype=torch.bool),
               "channel_mask": torch.zeros(len(query_ids), 52, dtype=torch.bool),
               "reference_valid": valid, "clip_id": [], "source_group": [], "times": torch.zeros(len(query_ids), width, dtype=torch.float64)}
        for j, selected_index in enumerate(selected):
            if selected_index is None:
                out["clip_id"].append(None); out["source_group"].append(None)
                continue
            ref = self.batch([selected_index], keys=("motion", "valid", "channel_mask", "times"))
            n = int(self["_lengths"][selected_index])
            for key in ("motion", "valid", "times"):
                out[key][j, :n] = ref[key][0]
            out["channel_mask"][j] = ref["channel_mask"][0]
            out["clip_id"].append(self["clip_id"][selected_index])
            out["source_group"].append(self["source_group"][selected_index])
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in out.items()}
