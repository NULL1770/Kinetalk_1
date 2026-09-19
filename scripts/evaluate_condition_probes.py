"""Evaluate frozen condition features with fixed nested linear probes.

This measures source-condition separability only. It is not a generation or
perceptual metric, and feature statistics are fitted on the 613 nested fit
clips before scoring the 206 held-sentence clips.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_paper_motion_probes import Probe, label_value, meta, rows, sha256


FEATURES = {
    "affect_global": ("affect_global", 64),
    "affect_global_intensity": ("affect_global_intensity", 65),
    "identity_code": ("identity_code", 128),
    "global": ("global", 65),
    "b9": ("b9", 9),
}


def feature(row, name):
    key, width = FEATURES[name]
    if name == "affect_global_intensity":
        x = np.r_[np.asarray(row["affect_global"], dtype=float).reshape(-1),
                  float(np.asarray(row["affect_intensity"], dtype=float).reshape(-1)[0])]
    else:
        if key not in row:
            raise ValueError(f"missing feature {key}")
        x = np.asarray(row[key], dtype=float).reshape(-1)
    if x.shape != (width,) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite [{width}]")
    return x


def evaluate(dataset: Path, protocol_path: Path, output: Path, name: str):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    dataset_hash = sha256(dataset)
    if protocol.get("dataset_sha256") != dataset_hash:
        raise ValueError("Dataset hash does not match bound protocol")
    payload = torch.load(dataset, map_location="cpu", weights_only=False)
    all_rows = rows(payload)
    by_id = {str(meta(r).get("clip_id")): r for r in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("Duplicate clip identifiers")
    fit_ids = protocol.get("inner_train_ids", protocol.get("fit_ids"))
    val_ids = protocol.get("inner_validation_ids", protocol.get("validation_ids"))
    if not fit_ids or not val_ids or set(fit_ids) & set(val_ids):
        raise ValueError("Nested sentence split ids are required and must be disjoint")
    fit_rows = [by_id[c] for c in fit_ids]
    val_rows = [by_id[c] for c in val_ids]
    X = np.stack([feature(r, name) for r in fit_rows])
    labels = {
        "emotion": [label_value(meta(r).get("emotion"), "emotion") for r in fit_rows],
        "motion_speaker_signature": [label_value(meta(r).get("speaker"), "speaker") for r in fit_rows],
    }
    probes = {k: Probe().fit(X, y) for k, y in labels.items()}
    Xv = np.stack([feature(r, name) for r in val_rows])
    result = {
        "schema": "condition_feature_probe_v2",
        "feature": name,
        "feature_dim": int(X.shape[1]),
        "alpha": 1.0,
        "dataset_sha256": dataset_hash,
        "source_sha256": sha256(Path(__file__)),
        "probe_source_sha256": sha256(Path(__file__).with_name("evaluate_paper_motion_probes.py")),
        "protocol_sha256": sha256(protocol_path),
        "split": {"fit_n": len(fit_rows), "validation_n": len(val_rows),
                  "fit_ids_sha256": hashlib.sha256("\n".join(fit_ids).encode()).hexdigest(),
                  "validation_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest()},
        "fit": {task: probes[task].report(X, labels[task]) for task in probes},
        "inner_validation": {
            "emotion": probes["emotion"].report(Xv, [label_value(meta(r).get("emotion"), "emotion") for r in val_rows]),
            "motion_speaker_signature": probes["motion_speaker_signature"].report(Xv, [label_value(meta(r).get("speaker"), "speaker") for r in val_rows]),
        },
        "limitations": ["source-condition separability only", "not generation quality or human perception",
                        "speaker signature is not appearance identity", "upstream feature exposure is inherited"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    state = output.with_suffix(".probe.npz")
    np.savez_compressed(state, **{f"{task}_{key}": value
        for task, probe in probes.items() for key, value in vars(probe).items()})
    result["probe_state_sha256"] = sha256(state)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature", choices=sorted(FEATURES), required=True)
    args = p.parse_args()
    result = evaluate(args.dataset, args.protocol, args.output, args.feature)
    print(json.dumps({"schema": result["schema"], "feature": args.feature,
                      "validation": {task: {k:v[k] for k in ("accuracy","balanced_accuracy","macro_f1")}
                                     for task,v in result["inner_validation"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
