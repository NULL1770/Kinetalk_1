"""Compact, read-only reporting of inner-selected predictable-motion probes.

Ranks and regularization come only from saved inner-CV scores. Outer scores
are reported, never used to select a model. Missing intervals stay missing;
single-model CIs do not establish significance of a paired difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


GROUPS = ("upper_expression", "all_expression", "mouth", "jaw17")
SPLITS = ("train", "internal_heldout", "external_dev")
MODES = ("full", "reverse", "oracle")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_choices(arm):
    """Reconstruct rank selection from inner scores and verify saved choices."""
    selection = arm["inner_selection"]
    if selection.get("outer_heldout_used") is not False:
        raise ValueError("Selection does not explicitly exclude outer heldout data")
    candidates = selection["candidate_scores"]
    chosen_alpha = {}
    for candidate in candidates:
        key = f"{candidate['method']}_rank{candidate['rank']}"
        score = float(candidate["inner_native_motion_mse"])
        if not math.isfinite(score):
            raise ValueError("Nonfinite inner selection score")
        if key not in chosen_alpha or score < chosen_alpha[key]["inner_native_motion_mse"]:
            chosen_alpha[key] = candidate
    if chosen_alpha != selection["selected_per_method_rank"]:
        raise ValueError("Saved per-rank choices do not match inner-CV minima")
    chosen_rank = {}
    for key, candidate in chosen_alpha.items():
        method = candidate["method"]
        if method not in chosen_rank or candidate["inner_native_motion_mse"] < chosen_alpha[chosen_rank[method]]["inner_native_motion_mse"]:
            chosen_rank[method] = key
    if chosen_rank != selection["selected_rank_per_method"]:
        raise ValueError("Saved rank choices do not match inner-CV minima")
    if arm.get("selected_rank_per_method", chosen_rank) != chosen_rank:
        raise ValueError("Top-level saved choices disagree with inner-CV choices")
    for key, candidate in chosen_alpha.items():
        if key not in arm["models"] or arm["models"][key].get("selected_alpha") != candidate["alpha"]:
            raise ValueError(f"Reported model does not use its inner-selected alpha: {key}")
    return chosen_rank, chosen_alpha


def compact_metric(metric):
    if metric is None:
        return {"status": "missing_metric"}
    interval = metric.get("bootstrap_r2_ci95")
    requested = metric.get("bootstrap_requested_samples", 0)
    valid = metric.get("bootstrap_valid_samples", 0)
    if interval is None:
        ci_status = "not_requested" if not requested else "unavailable"
    else:
        ci_status = "available"
    return {
        "status": "available", "r2": metric.get("r2_against_zero"),
        "correlation": metric.get("pooled_centered_correlation"),
        "r2_ci95": interval, "ci_status": ci_status,
        "bootstrap_requested": requested, "bootstrap_valid": valid,
        "energy_ratio": metric.get("energy_ratio"), "clips": metric.get("clips"),
        "sentences": metric.get("sentences"),
    }


def compact_model(arm, key, selection):
    model = arm["models"][key]
    splits = {}
    for split in SPLITS:
        if split not in model:
            splits[split] = {"status": "split_absent"}
            continue
        modes = {}
        for mode in MODES:
            if mode not in model[split]:
                modes[mode] = {"status": "mode_not_evaluated"}
                continue
            population = model[split][mode].get("nonneutral")
            if population is None:
                modes[mode] = {"status": "nonneutral_population_absent"}
                continue
            modes[mode] = {group: compact_metric(population.get(group)) for group in GROUPS}
        splits[split] = modes
    return {"model_key": key, **selection, "population": "nonneutral", "splits": splits}


def r2_at(model, split, group, mode="full"):
    return model["splits"].get(split, {}).get(mode, {}).get(group, {}).get("r2")


def comparison(left, right, label):
    """Positive deltas favor left; no significance claims from marginal CIs."""
    rows = []
    for split in SPLITS:
        for group in GROUPS:
            first, second = r2_at(left, split, group), r2_at(right, split, group)
            if first is None or second is None:
                continue
            rows.append({"split": split, "group": group, "left_r2": first, "right_r2": second,
                         "delta_r2": first - second, "paired_difference_ci95": None,
                         "paired_ci_status": "not_available_in_source_summary"})
    return {"label": label, "left_model": left["model_key"], "right_model": right["model_key"], "rows": rows}


def summarize_run(path):
    path = Path(path)
    path = path / "summary.json" if path.is_dir() else path
    source = read_json(path)
    if source.get("schema") != "predictable_motion_nested_probe_v1":
        raise ValueError(f"Unsupported predictable-motion summary: {path}")
    report = {"source": str(path.resolve()), "source_sha256": source_sha(path), "arms": {}, "comparisons": []}
    provenance = path.parent / "provenance.json"
    if provenance.exists():
        original = read_json(provenance)
        report["provenance"] = {k: original.get(k) for k in
                                ("scope", "selection", "generation_evaluated", "no_main_model_edits")}
    for name, arm in source["arms"].items():
        ranks, alphas = inner_choices(arm)
        selected = {method: compact_model(arm, key, alphas[key]) for method, key in ranks.items()}
        fixed = {method: compact_model(arm, f"{method}_rank8", alphas[f"{method}_rank8"])
                 for method in ("rrr", "pca") if f"{method}_rank8" in alphas}
        # Ridge is a direct full-output baseline, never mislabeled rank 8.
        direct = selected.get("ridge")
        report["arms"][name] = {"input_dim": arm["input_dim"],
                                "selection_target": arm["inner_selection"].get("selection_target"),
                                "selected": selected, "same_rank8": fixed, "direct_ridge": direct}
        if "rrr" in fixed and "pca" in fixed:
            report["comparisons"].append(comparison(fixed["rrr"], fixed["pca"], f"{name}: RRR minus PCA, fixed rank 8"))
        if "rrr" in selected and "pca" in selected:
            report["comparisons"].append(comparison(selected["rrr"], selected["pca"], f"{name}: RRR minus PCA, ranks independently inner-selected"))
    for first, second in (("temporal", "existing_e2v"), ("content_temporal", "content")):
        if first not in report["arms"] or second not in report["arms"]:
            continue
        for family in ("selected", "same_rank8"):
            left, right = report["arms"][first][family], report["arms"][second][family]
            for method in sorted(set(left) & set(right)):
                report["comparisons"].append(comparison(left[method], right[method], f"{first} minus {second}, {family}/{method}"))
    return report


def number(value):
    return "NA" if value is None else f"{value:.5f}"


def console_report(run):
    print(Path(run["source"]).parent)
    print("inner-selected RRR; nonneutral external_dev R2 (upper / all / mouth / jaw), upper CI")
    for arm, value in run["arms"].items():
        model = value["selected"].get("rrr")
        if model is None:
            continue
        metrics = model["splits"].get("external_dev", {}).get("full", {})
        upper = metrics.get("upper_expression", {})
        scores = " / ".join(number(metrics.get(group, {}).get("r2")) for group in GROUPS)
        interval = upper.get("r2_ci95")
        ci = "missing" if interval is None else f"[{interval[0]:.5f}, {interval[1]:.5f}]"
        print(f"  {arm}: rank={model['rank']} alpha={model['alpha']:g}; {scores}; CI={ci}")
    print("Full details: train/internal_heldout/external_dev, rank-8 controls, direct ridge, reverse and oracle in output JSON.")
    print("Paired-difference CIs are unavailable; marginal CIs do not prove an input/method advantage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh summary output path; source results are never overwritten")
    output = {"schema": "predictable_motion_compact_comparison_v1",
              "selection_rule": "Recompute and verify alpha/rank minima exclusively from saved inner-CV native motion MSE",
              "notes": ["All reported target metrics use nonneutral clips and common native motion channels.",
                        "Oracle reads true motion and is a subspace reconstruction diagnostic, not audio inference.",
                        "Direct ridge predicts all target channels; it is not a rank-8 model.",
                        "Rank/input/method differences have no paired CI in source summaries.",
                        "These probes do not evaluate a renderer or audiovisual lip synchronization."],
              "runs": [summarize_run(path) for path in args.results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    for run in output["runs"]:
        console_report(run)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
