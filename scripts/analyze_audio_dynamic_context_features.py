"""Recompute paired clip/speaker/sentence bootstrap from saved heldout results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.neutral_affect_metrics import motion_metrics

RUNS = ("run04_latent_cont", "run06_long_context", "run07_acoustic_probe", "run08_content_probe")
REGIONS = {"all": list(range(52)), "upper": list(range(14)), "brows": list(range(41, 46)),
           "mouth": list(range(14, 41)), "jaw17": [17]}


def bootstrap(delta, speakers, sentences, *, seed, repetitions):
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(len(delta), size=(repetitions, len(delta)))].mean(1)
    result = {"mean": float(delta.mean()), "median": float(np.median(delta)),
              "improved_clips": int((delta > 0).sum()), "clips": len(delta),
              "clip_ci95": np.quantile(draws, [.025, .975]).tolist()}
    for name, labels in (("speaker", speakers), ("sentence", sentences)):
        labels = np.asarray(labels)
        groups = [delta[labels == value] for value in np.unique(labels)]
        sampled = rng.integers(len(groups), size=(repetitions, len(groups)))
        means = np.asarray([g.sum() for g in groups])[sampled].sum(1) / np.asarray([len(g) for g in groups])[sampled].sum(1)
        result[name + "_clusters"] = len(groups)
        result[name + "_cluster_ci95"] = np.quantile(means, [.025, .975]).tolist()
    return result


def per_clip(summary, condition, region):
    key = "per_clip_mse" if region == "all" else region + "_per_clip_mse"
    values = np.asarray([s["conditions"][condition][key] for s in summary["seeds"]], dtype=float)
    if values.shape != (3, len(summary["clips"])) or not np.isfinite(values).all():
        raise ValueError("Expected three complete per-clip seed records")
    averaged = values.mean(0)
    np.testing.assert_allclose(averaged, summary["aggregate"][condition]["mean"][key], atol=1e-12)
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/neutral_affect_pilot_20260916"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    summaries = {r: json.loads((args.root / (r + "_eval") / "heldout_summary.json").read_text()) for r in RUNS}
    reference = summaries[RUNS[0]]
    for summary in summaries.values():
        for key in ("clips", "speaker_ids", "sentences"):
            if summary[key] != reference[key]:
                raise ValueError("Cross-run clip pairing mismatch")
        if [s["seed"] for s in summary["seeds"]] != [42, 123, 2026]:
            raise ValueError("Unexpected decoding seeds")
    curve_path = args.root / "robust/heldout_seed42_curves.pt"
    curves = torch.load(curve_path, map_location="cpu", weights_only=False)
    if curves["clip_id"] != reference["clips"]:
        raise ValueError("Raw B0/target clip pairing mismatch")
    metrics = motion_metrics(curves["b0"], curves["target"], curves["b0"], curves["valid"], curves["channel_mask"], curves["times"])
    for summary in summaries.values():
        for key, actual in metrics.items():
            if key in summary["b0"] and actual is not None:
                np.testing.assert_allclose(actual, summary["b0"][key], rtol=1e-6, atol=1e-8)
    mask = curves["valid"].unsqueeze(-1) & curves["channel_mask"].unsqueeze(1)
    b0_mse = {}
    for region, ids in REGIONS.items():
        observed = mask[..., ids]
        difference = (curves["b0"][..., ids].double() - curves["target"][..., ids].double()).square()
        b0_mse[region] = (difference.masked_fill(~observed, 0).sum((1, 2)) / observed.sum((1, 2))).numpy()
    result = {"seed": args.seed, "replicates": args.replicates, "clips": len(reference["clips"]),
              "speakers": len(set(reference["speaker_ids"])), "sentences": len(set(reference["sentences"])),
              "definition": "Positive improvement = alternative MSE minus preferred MSE; average seeds within clip first.",
              "b0_verified_against_all_summaries": True, "runs": {}, "cross_run": {}}
    def compare(preferred, alternative):
        return bootstrap(alternative - preferred, reference["speaker_ids"], reference["sentences"], seed=args.seed, repetitions=args.replicates)
    for run, summary in summaries.items():
        full = summary["aggregate"]["audio_full"]["mean"]
        data = {"points": {k: v for k, v in full.items() if not isinstance(v, list)},
                "affect": summary["affect_diagnostics"], "dynamic": {}, "b0_vs_full": {}}
        for alternative in ("mean", "reverse"):
            data["dynamic"][alternative] = {region: compare(per_clip(summary, "audio_full", region), per_clip(summary, "audio_" + alternative, region)) for region in REGIONS}
        for region in ("mouth", "jaw17"):
            data["b0_vs_full"][region] = compare(per_clip(summary, "audio_full", region), b0_mse[region])
        result["runs"][run] = data
    for preferred, alternative in ((RUNS[1], RUNS[0]), (RUNS[3], RUNS[2])):
        result["cross_run"][preferred + "_vs_" + alternative] = {region: compare(per_clip(summaries[preferred], "audio_full", region), per_clip(summaries[alternative], "audio_full", region)) for region in REGIONS}
    files = [curve_path, Path(__file__), *[args.root / (r + "_eval") / "heldout_summary.json" for r in RUNS]]
    result["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    print(json.dumps({"output": str(args.output), "clips": result["clips"], "b0_verified": True}))


if __name__ == "__main__":
    main()
