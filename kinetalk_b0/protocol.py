"""Strict data provenance, label and speaker-disjoint split contracts."""
from __future__ import annotations

import hashlib
from collections import Counter
from itertools import combinations
from pathlib import Path

from .utils import canonical_intensity, jsonl_records

VERSION = "speaker_disjoint_v1"
SPLITS = ("train", "val", "test")
EMOTIONS = ["neutral", "angry", "contempt", "disgust", "fear", "happy", "sad", "surprise"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emotion_name(value):
    name = str(value or "").strip().lower()
    return {"disgusted": "disgust", "fearful": "fear", "surprised": "surprise"}.get(name, name)


def record_index(path):
    result = {}
    for row in jsonl_records(path):
        key = row["clip_id"]
        if key in result:
            raise ValueError(f"Duplicate source record: {key}")
        result[key] = row
    return result


def enrich_pair(pair, records):
    source_id = pair.get("source_clip_id", pair.get("clip_id"))
    reference_id = pair.get("reference_clip_id")
    if source_id not in records or reference_id not in records:
        raise ValueError(f"Missing source/reference record: {source_id}, {reference_id}")
    source, reference = records[source_id], records[reference_id]
    row = dict(pair, clip_id=source_id, source_clip_id=source_id, reference_clip_id=reference_id)
    for field in ("speaker", "sentence_id", "split", "emotion", "intensity", "dataset"):
        value = source.get(field)
        if value is None or value == "":
            raise ValueError(f"Missing source {field}: {source_id}")
        if row.get(field) is not None and row[field] != value:
            raise ValueError(f"Pair/source {field} disagreement: {source_id}")
        row[field] = value
    row["emotion"] = emotion_name(row["emotion"])
    expected_intensity = canonical_intensity(row["intensity"], row["emotion"], 4)
    if "intensity_id" in pair and pair["intensity_id"] != expected_intensity:
        raise ValueError(f"Pair/source intensity_id disagreement: {source_id}")
    row["intensity_id"] = expected_intensity
    for field in ("speaker", "sentence_id", "split", "emotion"):
        if f"reference_{field}" in pair and pair[f"reference_{field}"] != reference.get(field):
            raise ValueError(f"Pair/reference {field} disagreement: {source_id}")
    row["reference_speaker"] = reference.get("speaker")
    row["reference_sentence_id"] = reference.get("sentence_id")
    row["reference_emotion"] = emotion_name(reference.get("emotion"))
    row["reference_split"] = reference.get("split")
    row["source_record"] = dict(source)
    row["reference_record"] = dict(reference)
    row["protocol_version"] = VERSION
    validate_pair(row)
    return row


def validate_pair(row, names=EMOTIONS, levels=4):
    clip = row.get("source_clip_id", row.get("clip_id"))
    for key in ("clip_id", "source_clip_id", "reference_clip_id", "speaker", "sentence_id", "split", "emotion"):
        if row.get(key) is None or row[key] == "":
            raise ValueError(f"Missing {key}: {clip}")
    if row["clip_id"] != row["source_clip_id"] or row["split"] not in SPLITS:
        raise ValueError(f"Invalid clip identity/split: {clip}")
    if emotion_name(row["emotion"]) not in names:
        raise ValueError(f"Unknown emotion {row['emotion']!r}: {clip}")
    expected = canonical_intensity(row.get("intensity"), row["emotion"], levels)
    if row.get("intensity_id") != expected:
        raise ValueError(f"Missing or inconsistent canonical intensity_id: {clip}")
    for key in ("speaker", "sentence_id", "split"):
        if row.get(f"reference_{key}") != row[key]:
            raise ValueError(f"Source/reference {key} mismatch: {clip}")
    if row.get("reference_emotion") != "neutral":
        raise ValueError(f"Teacher reference is not neutral: {clip}")


def load_split(data, split, *, stage1=False):
    if split not in SPLITS:
        raise ValueError(f"Use an explicit train/val/test split, got {split!r}")
    key = ({"train": "stage1_manifest", "val": "stage1_val_manifest", "test": "stage1_test_manifest"}
           if stage1 else {"train": "train_manifest", "val": "val_manifest", "test": "test_manifest"})[split]
    path = data.get(key)
    if not path:
        raise ValueError(f"data.{key} is required; split fallback is prohibited")
    rows = jsonl_records(path)
    if data.get("records_manifest"):
        records = record_index(data["records_manifest"])
        rows = [enrich_pair(row, records) for row in rows]
    if not rows:
        raise ValueError(f"Empty {split} manifest: {path}")
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"data.{key} contains {row.get('split')!r} row; expected {split}: {row.get('source_clip_id')}")
        validate_pair(row, data.get("emotion_classes", EMOTIONS), int(data.get("num_intensity_levels", 4)))
    ids = [row["clip_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate source clips in {path}")
    return rows, Path(path)


def statistics(rows):
    speakers = sorted({r["speaker"] for r in rows})
    return {"n": len(rows), "speaker_count": len(speakers), "speakers": speakers,
            "emotion": dict(sorted(Counter(r["emotion"] for r in rows).items())),
            "intensity_id": dict(sorted(Counter(r["intensity_id"] for r in rows).items())),
            "missing_labels": 0}


def audit_splits(splits):
    report = {"splits": {s: statistics(splits[s]) for s in SPLITS}, "overlap": {}}
    for left, right in combinations(SPLITS, 2):
        speakers = sorted({r["speaker"] for r in splits[left]} & {r["speaker"] for r in splits[right]})
        clipsets = [{r[k] for r in splits[s] for k in ("source_clip_id", "reference_clip_id")} for s in (left, right)]
        clips = sorted(clipsets[0] & clipsets[1])
        if speakers or clips:
            raise ValueError(f"Split leakage {left}/{right}: speakers={speakers}, clips={clips[:10]}")
        report["overlap"][f"{left}/{right}"] = {"speakers": speakers, "source_and_reference_clips": clips}
    return report


def legacy_path(root, kind, clip_id, dataset="mead"):
    root = Path(root)
    paths = [root / kind / dataset / f"{clip_id}.npz", root / kind / f"{clip_id}.npz"]
    if kind == "bs":
        paths += [root.parent / "coeffs_final" / dataset / f"{clip_id}.npz", root.parent / "coeffs_final" / f"{clip_id}.npz"]
    return next((p for p in paths if p.is_file()), None)


def pair_path(row, data):
    raw = Path(row["teacher_artifact"])
    path = raw if raw.is_file() else Path(data["aligned_dtw_root"]) / "pairs" / raw.name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def audit_config(data):
    result = {"version": VERSION}
    for stage1, label in ((True, "stage1"), (False, "stage2_4")):
        splits, manifests = {}, {}
        for split in SPLITS:
            rows, path = load_split(data, split, stage1=stage1)
            splits[split] = rows
            manifests[split] = {"path": str(path), "sha256": sha256(path)}
        result[label] = dict(audit_splits(splits), manifests=manifests)
    # Stage1 and later stages must use the same assignment, including teachers.
    stage_speakers = {}
    for stage in ("stage1", "stage2_4"):
        for split, info in result[stage]["splits"].items():
            for speaker in info["speakers"]:
                if stage_speakers.setdefault(speaker, split) != split:
                    raise ValueError(f"Cross-stage split leakage: {speaker}")
    return result


def require_training_protocol(payload, protocol):
    """Prevent old/all-gated parents from contaminating a new split run."""
    previous = payload.get("data_protocol")
    if previous is None:
        raise ValueError("Parent checkpoint has no split provenance; retrain it under the new protocol")
    for stage in ("stage1", "stage2_4"):
        for split in SPLITS:
            if previous[stage]["manifests"][split]["sha256"] != protocol[stage]["manifests"][split]["sha256"]:
                raise ValueError(f"Parent checkpoint uses a different protocol: {stage}/{split}")
