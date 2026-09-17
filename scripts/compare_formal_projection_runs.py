"""Compare saved, formally selected runs without selecting or generating models.

Each training seed remains a separate experiment. Generation-noise sufficient
statistics are averaged per clip before paired sentence-cluster resampling.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_formal_projection import GROUPS, SEEDS, assert_metric_agreement, validate_artifacts
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary
from scripts.train_predictable_renderer import basic_metrics, center, sha


REGIONS = ("upper_expression", "brows", "eyes_expression")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def assert_matched_recipes(left, right):
    """A mode change changes basis/head, but not data/RNG/schedule/budget."""
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    for value in (a, b):
        value["args"].pop("mode", None)
        value["args"].pop("output", None)
    if a != b:
        raise ValueError("Matched RRR/PCA recipes differ beyond mode and output")


def check_matching_references(left, right):
    for key in ("clip_id", "sentence_id"):
        if left["q"][key] != right["q"][key]:
            raise ValueError(f"Cross-run reference {key} differs")
    for group, key in (("q", "motion"), ("q", "valid"), ("q", "channel_mask"),
                       ("q", "emotion_id"), ("q", "times"), ("base", "b0"), ("identity", "baseline")):
        if not torch.equal(left[group][key], right[group][key]):
            raise ValueError(f"Cross-run reference {group}/{key} differs")


def load_matching_final_audit(run, summary, hashes):
    matches = []
    for path in sorted(run.glob("*.json")):
        if "audit" not in path.name:
            continue
        value = read_json(path)
        if value.get("schema") != "formal_projection_descriptive_audit_v1":
            continue
        if any(value.get("hashes", {}).get(key) != hashes[key]
               for key in ("summary", "selected_checkpoint", "curves", "cache", "bundle", "checkpoint")):
            raise ValueError(f"Stale/unbound final audit: {path}")
        if value["selected_epoch"] != summary["best_epoch"]:
            raise ValueError("Final audit selected epoch differs")
        matches.append((path, value))
    if not matches:
        raise FileNotFoundError(f"No matching final_audit JSON in {run}")
    first = matches[0]
    if any(value != first[1] for _, value in matches[1:]):
        raise ValueError(f"Multiple different final audits in {run}")
    return first


def trajectory(run, summary, recipe):
    completed = int(summary["completed_epochs"])
    n = int(recipe["data_scope"]["train_clips"])
    batches = math.ceil(n / recipe["args"]["batch_size"])
    previous = 0.
    rows = []
    hashes = {}
    for epoch in range(1, completed + 1):
        path = run / f"epoch{epoch:03d}.json"
        record = read_json(path)
        hashes[path.name] = sha(path)
        if record["epoch"] != epoch or record["step"] != epoch * batches or record["samples_seen"] != n or record["batches"] != batches:
            raise ValueError("Epoch record fails complete-data/step contract")
        elapsed = float(record["elapsed_seconds"])
        if elapsed < previous:
            raise ValueError("Nonmonotone elapsed training clock")
        row = {key: record[key] for key in ("epoch", "step", "samples_seen", "batches", "mean_batch_flow_loss", "teacher_fraction", "elapsed_seconds")}
        row["epoch_wall_seconds_including_development"] = elapsed - previous
        if epoch % recipe["args"]["eval_every"] == 0 or epoch == completed:
            dev_path = run / f"development_epoch{epoch:03d}.json"
            if dev_path.is_file():
                dev = read_json(dev_path)
                hashes[dev_path.name] = sha(dev_path)
                if record["selection"] != dev["selection"]:
                    raise ValueError("Epoch/development selection record differs")
                row["development_selection"] = dev["selection"]
        rows.append(row)
        previous = elapsed
    if summary["optimizer_steps"] != completed * batches:
        raise ValueError("Summary optimizer budget differs from epoch records")
    return {"completed_epochs": completed, "optimizer_steps": summary["optimizer_steps"],
            "selected_epoch": summary["best_epoch"], "selected_epoch_optimizer_steps": summary["best_epoch"] * batches if summary["best_epoch"] else None,
            "total_training_examples_seen": completed * n,
            "training_wall_seconds_including_periodic_development": previous,
            "timing_scope": "Starts after initial original-checkpoint evaluation; includes periodic development, excludes final four-mode audit, never GPU compute-only time",
            "rows": rows, "source_sha256": hashes}


def dynamic_stats(curves, reference, summary):
    q = reference["q"]
    w = q["valid"].float()
    baseline = reference["base"]["b0"].float() + reference["identity"]["baseline"].float()[:, None]
    target = center(q["motion"].float() - baseline, w)
    statistics = {}
    for seed in SEEDS:
        for mode in ("full", "zero", "reverse", "oracle"):
            prediction = curves["motion"][str(seed)][mode]
            assert_metric_agreement(basic_metrics(prediction, reference), summary["after"][str(seed)][mode])
            transformed = center(prediction - baseline, w)
            for region in REGIONS:
                value = clip_statistics(transformed, target, w, GROUPS[region])
                key = mode, region
                statistics[key] = statistics.get(key, np.zeros_like(value)) + value / len(SEEDS)
    return statistics


def compact_run(run, summary, recipe, audit, audit_path, hashes, schedule):
    return {"run": str(run.resolve()), "method": recipe["args"]["mode"], "training_seed": recipe["args"]["seed"],
        "selected_checkpoint_sha256": summary["selected_checkpoint_sha256"],
        "recipe_sha256": summary["recipe_sha256"], "summary_sha256": hashes["summary"],
        "final_audit": {"path": str(audit_path.resolve()), "sha256": sha(audit_path)},
        "frozen_and_head_unchanged": bool(summary["frozen_unchanged"] and summary["head_unchanged"]),
        "has_eligible_checkpoint": summary["has_eligible_checkpoint"],
        "single_seed_engineering_pass": audit["single_seed_engineering_pass"], "checks": audit["checks"],
        "nonneutral_regions": {region: {"full": audit["scores"]["full"]["centered_residual"][region]["nonneutral"],
            "full_vs_zero": audit["paired_dynamic"]["zero"]["nonneutral"][region],
            "full_vs_reverse": audit["paired_dynamic"]["reverse"]["nonneutral"][region]} for region in REGIONS},
        "neutral_raw_relative_vs_zero": {region: audit["neutral_raw_relative_vs_zero"][region] for region in ("upper_expression", "mouth")},
        "mouth_correlation_delta": audit["mouth_correlation_delta"], "teacher_accuracy_delta": audit["teacher_accuracy_delta"],
        "training": schedule}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("Positive bootstrap sample count required")
    output = args.output or args.root / "comparison.json"
    if output.exists():
        raise FileExistsError("Use fresh comparison output; never overwrite selected evidence")
    torch.set_num_threads(4)
    found = {}
    for path in sorted(args.root.glob("*/summary.json")):
        report = read_json(path)
        if report.get("schema") != "formal_predictable_projection_result_v1":
            continue
        prov = read_json(path.parent / "provenance.json")
        key = prov["recipe"]["args"]["mode"], int(prov["recipe"]["args"]["seed"])
        if key in found:
            raise ValueError(f"Duplicate method/seed run: {key}")
        found[key] = path.parent
    required = {("rrr", 46), ("rrr", 47), ("rrr", 48), ("pca", 46)}
    if not required.issubset(found):
        raise ValueError(f"Missing completed prespecified runs: {sorted(required - found.keys())}")
    reports, paired_data = {}, {}
    common_hashes = None
    for key in sorted(required):
        run = found[key]
        summary, recipe, reference, curves, original, hashes, binding = validate_artifacts(run)
        audit_path, audit = load_matching_final_audit(run, summary, hashes)
        if audit["training_seed"] != key[1] or audit["method"] != key[0]:
            raise ValueError("Final audit method/seed differs")
        current_hashes = {name: hashes[name] for name in ("cache", "bundle", "weights", "checkpoint", "config")}
        if common_hashes is None:
            common_hashes = current_hashes
        elif common_hashes != current_hashes:
            raise ValueError("Formal runs do not use identical immutable input artifacts")
        schedule = trajectory(run, summary, recipe)
        tag = f"{key[0]}_seed{key[1]}"
        reports[tag] = compact_run(run, summary, recipe, audit, audit_path, hashes, schedule)
        if key[1] == 46:
            statistics = dynamic_stats(curves, reference, summary)
            # Keep only the matched-pair small statistics and zero reference;
            # never concatenate observations across training seeds.
            paired_data[key[0]] = {"statistics": statistics, "reference": reference,
                "recipe": recipe, "summary": summary,
                "zero_curves": {seed: curves["motion"][str(seed)]["zero"] for seed in SEEDS}}
        del curves, original
        print(json.dumps({"run": tag, "completed_epochs": schedule["completed_epochs"], "steps": schedule["optimizer_steps"],
            "best_epoch": summary["best_epoch"], "upper_gain_vs_zero": reports[tag]["nonneutral_regions"]["upper_expression"]["full_vs_zero"]["r2_improvement"]}), flush=True)
    a, b = paired_data["rrr"], paired_data["pca"]
    assert_matched_recipes(a["recipe"], b["recipe"])
    check_matching_references(a["reference"], b["reference"])
    if any(a["summary"][key] != b["summary"][key] for key in ("completed_epochs", "optimizer_steps", "best_epoch")):
        raise ValueError("Matched RRR46/PCA46 completed/selected training budgets differ")
    if any(not torch.equal(a["zero_curves"][seed], b["zero_curves"][seed]) for seed in SEEDS):
        raise ValueError("Matched RRR/PCA zero-local generation differs")
    q = a["reference"]["q"]
    nonneutral = (q["emotion_id"] != 0).nonzero(as_tuple=True)[0].tolist()
    pair = {region: paired_summary(a["statistics"][("full", region)], b["statistics"][("full", region)],
               q["sentence_id"], nonneutral, samples=args.samples, seed=45) for region in REGIONS}
    report = {"schema": "formal_projection_cross_run_comparison_v1", "source_script_sha256": sha(__file__),
        "root": str(args.root.resolve()), "common_input_sha256": common_hashes, "runs": reports,
        "matched_rrr46_minus_pca46": pair,
        "matched_design_checks": {"same_data_reference_and_zero_curves": True, "same_training_seed_schedule_and_budget": True,
            "same_selected_epoch": True, "recipe_only_mode_and_output_differ": True,
            "rng_note": "Same deterministic CPU training generator and draws for both modes; original runner did not log per-step RNG hashes, so this is design/recipe evidence, not a per-step logged-hash assertion"},
        "bootstrap_samples": args.samples, "bootstrap_seed": 45, "generation_noise_seeds": list(SEEDS),
        "statistical_unit": "Sentence clusters after averaging generation-noise sufficient statistics per clip; each training seed reported separately",
        "scope": "Already-used development set and already-selected checkpoints; descriptive post-selection comparisons, not independent test inference or publication guarantee",
        "refit_or_generation_performed": False, "checkpoint_selection_changed": False, "test_loaded": False,
        "default_replaced": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"RRR46_minus_PCA46": {region: {"delta_r2": values["r2_improvement"], "ci95": values["r2_improvement_ci95"]}
                    for region, values in pair.items()}, "output": str(output)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
