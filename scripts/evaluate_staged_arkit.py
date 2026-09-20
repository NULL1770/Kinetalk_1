"""Score saved staged-run curves with the audited FaceDiffuser BEAT formulas.

This reads already exposed development archives and metadata only. It neither
generates new motion nor opens native targets. Single-seed interventions and
three-draw full generation are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.arkit_benchmark_report import build_report, score_fullface, write_report


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(run_paths, metadata, output):
    if output.exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf8").splitlines() if line.strip()]
    index = {r["clip_id"]: r for r in rows}
    if len(index) != len(rows) or any(r["split"] != "train" for r in rows):
        raise ValueError("Unique native TRAIN metadata required for these internal archives")
    output.mkdir(parents=True)
    summaries, signature, seen = [], None, set()
    for label, directory in run_paths:
        if label in seen or not label.replace("_", "").isalnum():
            raise ValueError("Unique alphanumeric run labels required")
        seen.add(label)
        path = directory / "curves.pt"
        source = torch.load(path, map_location="cpu", weights_only=False)
        evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf8"))
        if evaluation.get("test_loaded") is not False:
            raise ValueError("Archive lacks explicit non-test evaluation provenance")
        ids = source["clip_id"]
        y, valid, channel, times = [source[k].numpy() for k in ("target", "valid", "channel_mask", "times")]
        if (len(set(ids)) != len(ids) or len(ids) != len(y) or evaluation["clips"] != len(ids)
                or set(ids) - set(index)):
            raise ValueError("Archive clip membership differs from native train metadata")
        if y.ndim != 3 or y.shape[-1] != 52 or valid.shape != y.shape[:2] or times.shape != valid.shape:
            raise ValueError("Invalid archived target, validity, or clock shape")
        if valid.dtype != bool or channel.dtype != bool or channel.shape != (len(ids), 52):
            raise ValueError("Expected original Boolean clip-level channel masks")
        current = (ids, y, valid, channel, times)
        if signature is None:
            signature = current
        else:
            if ids != signature[0]:
                raise ValueError("Run comparison clip order differs")
            for actual, expected in zip(current[1:], signature[1:]):
                np.testing.assert_array_equal(actual, expected)
        groups = {key: [key] for key in source["predictions"]}
        full = [f"{seed}/full" for seed in source["noise_seeds"]]
        if len(full) > 1:
            groups["all_draws/full"] = full
        sources = {"archive": str(path), "archive_sha256": sha(path),
                   "evaluation_sha256": sha(directory / "evaluation.json"),
                   "checkpoint_sha256": sha(directory / "final.pt"),
                   "native_train_metadata_sha256": sha(metadata),
                   "exporter_sha256": sha(Path(__file__))}
        for mode, keys in groups.items():
            predictions = np.stack([source["predictions"][key].numpy() for key in keys])
            scored = []
            for i, cid in enumerate(ids):
                row = index[cid]
                scored.append(score_fullface(predictions[:, i], {
                    "clip_id": cid, "sentence": row["sentence"], "speaker": row["speaker"],
                    "emotion": row["emotion"], "target52": y[i], "valid": valid[i],
                    "times": times[i], "channel_mask": np.broadcast_to(channel[i], y[i].shape),
                }))
            scope = f"{len(ids)} historical internal development clips, one 96-frame window per clip; not paper validation/test"
            report = build_report(scored, scope=scope, sources=sources)
            report.update({"run": label, "mode": mode, "prediction_keys": keys,
                           "conditioning": evaluation["condition_source"],
                           "oracle_intervention": "oracle" in mode,
                           "channel_mask_expansion": "Original native per-clip [52] mask broadcast over time, AND original frame validity",
                           "test_loaded": False, "default_replaced": False})
            write_report(output / label / (mode.replace("/", "_") + ".json"), report)
            summaries.append({"run": label, "mode": mode, "draws": len(keys), "clips": len(ids),
                              **{name: record["value"] for name, record in report["summary"].items()}})
        print(json.dumps({"run": label, "clips": len(ids), "groups": len(groups)}), flush=True)
    with (output / "main_metrics.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    write_report(output / "summary.json", {"scope": "internal development only", "rows": summaries,
                 "test_loaded": False, "matched_targets_masks_clocks": True})
    columns = ["run", "mode", "draws", "arkit_mbe", "arkit_lbe", "arkit_fdd_signed",
               "arkit_fdd_absolute", "supp_upper9_fdd_absolute", "supp_lip23_lbe"]
    lines = ["# Staged-run FaceDiffuser coefficient evaluation", "",
             "405 historical internal development clips, 96-frame windows at 25 fps. Not a paper test table.", "",
             "Same saved target/mask/time tensors across every run; raw coefficients, no clipping.",
             "Intervention comparisons use seed 42. All-draw full results average metrics over three draws per clip.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in summaries:
        if not (row["mode"].startswith("42/") or row["mode"].startswith("all_draws/")):
            continue
        lines.append("| " + " | ".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in columns) + " |")
    lines += ["", "FDD is signed GT minus prediction energy-std, with zero as reference; ABS-FDD is lower-is-better.",
              "Official FDD excludes brows and is invariant to time permutation. Upper9 is supplementary.",
              "Oracle rows use target motion and cannot be reported as audio-only performance.",
              "AV offset/confidence, learned-feature Multimodality, FD and WInD remain pending their validated evaluators.",
              "No same-protocol external baseline training results are supplied here."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=/path/to/completed/run")
    parser.add_argument("--metadata", type=Path, required=True, help="Native train.jsonl; metadata only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run([(value.split("=", 1)[0], Path(value.split("=", 1)[1])) for value in args.run], args.metadata, args.output)
