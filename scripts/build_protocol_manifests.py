"""Enrich gated pairs and preserve the source manifest's speaker split."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.protocol import (
    VERSION, SPLITS, audit_splits, enrich_pair, legacy_path, pair_path, record_index, sha256,
)
from kinetalk_b0.utils import jsonl_records, load_yaml


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build(pairs, records, data):
    enriched = [enrich_pair(row, records) for row in pairs]
    ids = [row["clip_id"] for row in enriched]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source clips in gated input")
    split_rows = {s: [r for r in enriched if r["split"] == s] for s in SPLITS}
    audit_splits(split_rows)
    exclusions, complete = [], []
    for row in enriched:
        artifact = pair_path(row, data)
        with np.load(artifact, allow_pickle=False) as z:
            for key, dim in (("canonical_motion", data["motion_dim"]), ("canonical_content", data["content_dim"])):
                array = z[key]
                if array.ndim != 2 or array.shape[1] != dim or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {key}: {artifact}")
            n = len(z["canonical_motion"])
            if len(z["canonical_content"]) != n or z["teacher_mask"].shape != (n,) or not z["teacher_mask"].any():
                raise ValueError(f"Invalid canonical clock/mask: {artifact}")
        if not data.get("stage1_native_time", False):
            target = Path(data["stage1_target_root"]) / artifact.name
            with np.load(target, allow_pickle=False) as z:
                if z["canonical_neutral_target"].shape != (n, data["motion_dim"]) or not np.isfinite(z["canonical_neutral_target"]).all() or not z["canonical_neutral_mask"].any():
                    raise ValueError(f"Invalid canonical neutral target: {target}")
            row["stage1_target_artifact"] = str(target)
        missing = []
        for clip in (row["source_clip_id"], row["reference_clip_id"]):
            for kind in ("audio", "affect", "bs"):
                if legacy_path(data["legacy_aligned_root"], kind, clip, row["dataset"]) is None:
                    missing.append(f"{kind}:{clip}")
        if missing:
            exclusions.append({"clip_id": row["clip_id"], "split": row["split"], "reason": "missing_inputs", "details": missing})
        else:
            complete.append(row)
    neutral = {r["clip_id"]: r for r in complete if r["emotion"] == "neutral"}
    usable = []
    for row in complete:
        if row["reference_clip_id"] not in neutral:
            exclusions.append({"clip_id": row["clip_id"], "split": row["split"], "reason": "missing_exact_neutral_row"})
        else:
            usable.append(row)
    stages = {"stage1": split_rows, "stage2_4": {s: [r for r in usable if r["split"] == s] for s in SPLITS}}
    report = {name: audit_splits(rows) for name, rows in stages.items()}
    for name, splits in stages.items():
        for split, rows in splits.items():
            if not rows or {r["intensity_id"] for r in rows} != {0, 1, 2, 3}:
                raise ValueError(f"Missing data or canonical intensity classes: {name}/{split}")
    return stages, exclusions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new protocol directory: {args.output_dir}")
    stages, exclusions, report = build(jsonl_records(args.pairs), record_index(args.records), load_yaml(args.config)["data"])
    args.output_dir.mkdir(parents=True)
    report.update({"version": VERSION, "assignment": "preserved source-record speaker split, no resampling",
                   "inputs": {str(p): sha256(p) for p in (args.pairs, args.records, args.config)},
                   "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)), "manifests": {}})
    for stage, splits in stages.items():
        for split, rows in splits.items():
            path = args.output_dir / f"{stage}_{split}.jsonl"
            write_jsonl(path, rows)
            report["manifests"][path.name] = sha256(path)
    write_jsonl(args.output_dir / "exclusions.jsonl", exclusions)
    (args.output_dir / "protocol.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
