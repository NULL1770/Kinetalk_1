"""Summarize frozen decoder-bottleneck interventions with paired clip bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "arkit_mbe",
    "arkit_lbe",
    "arkit_fdd_absolute",
    "mouth_jaw/mse",
    "mouth_jaw/centered_mse",
    "mouth_jaw/centered_correlation",
    "jawOpen/mse",
    "jaw_closure_frame_f1",
    "lve_mean_mm_mean",
    "eve_mean_mm_mean",
    "geometry/width_MAE_mm",
    "geometry/width_bias_mm",
    "geometry/aperture_MAE_mm",
    "geometry/aperture_bias_mm",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    per_clip = json.loads(args.per_clip.read_text(encoding="utf-8"))
    if report["schema"] not in ("v5_decoder_bottleneck_v1", "v5_geometry_oracle_sweep_v1"):
        raise ValueError("Unexpected diagnostic schema")
    if report["test_loaded"] or report["query_motion_loaded_for_deployment"]:
        raise ValueError("Report violates diagnostic scope")
    modes = list(per_clip)
    reference = {row["clip_id"]: row for row in per_clip["audio_sample"]}
    if len(reference) != report["clips"]:
        raise ValueError("Clip count differs from completion report")
    rng = np.random.default_rng(args.seed)
    summary = []
    deltas = []
    strata = [("ALL", None)] + [(emotion, emotion) for emotion in sorted({r["emotion"] for r in per_clip["audio_sample"]})]
    for mode in modes:
        rows_by_id = {row["clip_id"]: row for row in per_clip[mode]}
        if set(rows_by_id) != set(reference):
            raise ValueError(f"Clip binding differs for {mode}")
        for label, emotion in strata:
            clip_ids = [cid for cid, row in reference.items() if emotion is None or row["emotion"] == emotion]
            current = [rows_by_id[cid] for cid in clip_ids]
            baseline = [reference[cid] for cid in clip_ids]
            row = {"mode": mode, "emotion": label, "clips": len(current)}
            for key in METRICS:
                row[key] = mean(current, key)
            if label == "ALL":
                probes = report["results"][mode].get("probes", [])
                for index, probe in enumerate(probes):
                    row[f"probe_macro_f1_{index}"] = float(probe["macro_f1"])
            summary.append(row)
            if mode == "audio_sample":
                continue
            for key in METRICS:
                paired = np.asarray([
                    float(rows_by_id[cid][key]) - float(reference[cid][key])
                    for cid in clip_ids
                    if rows_by_id[cid].get(key) is not None and reference[cid].get(key) is not None
                ], dtype=np.float64)
                if not len(paired):
                    continue
                samples = rng.integers(0, len(paired), size=(args.draws, len(paired)))
                boot = paired[samples].mean(axis=1)
                deltas.append({
                    "mode": mode,
                    "emotion": label,
                    "metric": key,
                    "clips": int(len(paired)),
                    "delta_vs_audio_mean": float(paired.mean()),
                    "bootstrap_95_low": float(np.quantile(boot, .025)),
                    "bootstrap_95_high": float(np.quantile(boot, .975)),
                    "bootstrap_draws": args.draws,
                    "seed": args.seed,
                    "resampling_unit": "clip; descriptive, no speaker/training-seed inference",
                })
    write_csv(args.out / "decoder_bottleneck_summary.csv", summary)
    write_csv(args.out / "decoder_bottleneck_paired_deltas.csv", deltas)
    receipt = {
        "schema": "decoder_bottleneck_summary_v1",
        "report_sha256": sha(args.report),
        "per_clip_sha256": sha(args.per_clip),
        "source_sha256": report["source_sha256"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "binding_sha256": report["binding_sha256"],
        "clips": report["clips"],
        "test_loaded": False,
        "diagnostic_only": True,
        "bootstrap": {
            "draws": args.draws,
            "seed": args.seed,
            "unit": "clip",
            "limits": "descriptive intervals only; ignores speaker clustering and training-seed variation",
        },
    }
    (args.out / "decoder_bottleneck_summary_provenance.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **receipt}, indent=2))


if __name__ == "__main__":
    main()
