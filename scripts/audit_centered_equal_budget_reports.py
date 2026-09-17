"""Read-only equal-budget epoch8/10 comparison from existing sufficient reports.

No checkpoint, cache, target tensor or generation is loaded. The older epoch
reports have no saved curves/checkpoint binding; this limit is explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_projection_centered_probe import validate_centered_pair
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import sha

EPOCHS = (8, 10)
SEEDS = ("42", "123", "2026")
ARMS = ("centered_rollout", "raw_rollout")
MODES = ("full", "zero")
KINDS = ("raw_motion", "centered_residual")
POPULATIONS = ("all", "neutral", "nonneutral")
GROUPS = ("all_expression", "upper_expression", "brows", "eyes_expression", "mouth", "jaw17")
COLUMNS = ("sse", "zero", "sst", "weighted_values", "clips")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate_rng_prefix(centered, raw, *, epochs=10):
    for epoch in range(1, epochs + 1):
        a, b = centered[epoch], raw[epoch]
        for key in ("minibatch_sha256", "noise_time_sha256", "teacher_choice_draw_sha256"):
            value = a.get(key)
            if not isinstance(value, str) or len(value) != 64 or value != b.get(key):
                raise ValueError(f"RNG differs at epoch{epoch}/{key}")
        for key in ("step", "samples_seen", "teacher_fraction"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Training coverage/schedule differs at epoch{epoch}/{key}")
        for record in (a, b):
            if record.get("epoch") != epoch or record.get("step") != epoch * 145 or record.get("samples_seen") != 2315:
                raise ValueError("Expected complete2315clip epochs at145steps/epoch")
            if not np.isfinite(record["teacher_fraction"]) or not 0 <= record["teacher_fraction"] <= 1:
                raise ValueError("Invalid teacher coverage")


def close(actual, expected, name):
    if actual is None or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(f"Report aggregate differs from sufficient statistics: {name}")


def sentence_rows(report):
    sentences = sorted(report["per_sentence"])
    if not sentences:
        raise ValueError("Missing sentence sufficient statistics")
    rows = np.asarray([[report["per_sentence"][s][key] for key in COLUMNS] for s in sentences], dtype=np.float64)
    if not np.isfinite(rows).all() or (rows < 0).any() or (rows[:, 3:] <= 0).any():
        raise ValueError("Invalid nonnegative sentence statistics/counts")
    if not np.equal(rows[:, 3:], np.floor(rows[:, 3:])).all():
        raise ValueError("Native observed counts and clips must be integers")
    total = rows.sum(0)
    for key, expected in (("sse", total[0]), ("target_energy", total[1]), ("sst", total[2]),
                          ("weighted_values", total[3]), ("clips", total[4]), ("sentences", len(sentences)),
                          ("native_mse", total[0] / total[3])):
        close(report.get(key), expected, key)
    if total[1] > 0:
        close(report.get("r2_against_zero"), 1 - total[0] / total[1], "r2")
    for sentence, row in zip(sentences, rows):
        if row[1] > 0:
            close(report["per_sentence"][sentence]["r2_against_zero"], 1 - row[0] / row[1], "sentence r2")
    return sentences, rows


def mean_noise_rows(reports):
    if set(reports) != set(SEEDS):
        raise ValueError("Require the same fixed3noise reports")
    parsed = [sentence_rows(reports[seed]) for seed in SEEDS]
    sentences, first = parsed[0]
    for names, row in parsed[1:]:
        if names != sentences or not np.array_equal(row[:, 1:], first[:, 1:]):
            raise ValueError("Noise reports have different target/count/clip statistics")
    averaged = first.copy()
    averaged[:, 0] = np.stack([rows[:, 0] for _, rows in parsed]).mean(0)
    return sentences, averaged


def summarize(rows):
    total = rows.sum(0)
    return {"native_mse": float(total[0] / total[3]),
        "r2_against_zero": float(1 - total[0] / total[1]) if total[1] > 0 else None,
        "sse": float(total[0]), "target_energy": float(total[1]), "observed_values": int(total[3]),
        "clips": int(total[4]), "sentences": len(rows)}


def compare_rows(candidate, baseline, *, samples=5000):
    names, a = candidate
    bnames, b = baseline
    if names != bnames or not np.array_equal(a[:, 1:], b[:, 1:]):
        raise ValueError("Paired reports have different sentence/target/count/clip statistics")
    delta = a[:, 0] - b[:, 0]
    target, count = a[:, 1].sum(), a[:, 3].sum()
    interval_mse, interval_r2, valid_count = None, None, 0
    if len(names) > 1 and samples > 0:
        grouped = np.column_stack((delta, a[:, 1], a[:, 3]))
        selected = np.random.default_rng(45).integers(len(names), size=(samples, len(names)))
        sums = grouped[selected].sum(1)
        valid = (sums[:, 1] > 0) & (sums[:, 2] > 0)
        valid_count = int(valid.sum())
        if valid_count:
            interval_mse = np.quantile(sums[valid, 0] / sums[valid, 2], [.025, .975]).tolist()
            interval_r2 = np.quantile(-sums[valid, 0] / sums[valid, 1], [.025, .975]).tolist()
    return {"candidate": summarize(a), "baseline": summarize(b),
        "native_mse_delta_candidate_minus_baseline": float(delta.sum() / count),
        "native_mse_delta_ci95": interval_mse,
        "r2_delta_candidate_minus_baseline": float(-delta.sum() / target) if target > 0 else None,
        "r2_delta_ci95": interval_r2,
        "positive_sse_change_sentences": int((delta > 0).sum()), "negative_sse_change_sentences": int((delta < 0).sum()),
        "bootstrap_valid_samples": valid_count, "bootstrap_requested_samples": samples,
        "sign": "Negative native-MSE delta improves; positive R2 delta improves"}


def load_reports(run):
    run = Path(run)
    provenance = read_json(run / "provenance.json")
    recipe = provenance["recipe"]
    if provenance.get("recipe_sha256") != canonical_hash(recipe):
        raise ValueError("Recipe hash mismatch")
    args = recipe["args"]
    if args.get("seed") != 46 or args.get("batch_size") != 16 or args.get("lr") != .0002:
        raise ValueError("Unexpected original training seed/batch/lr")
    if recipe.get("teacher_probability") != .5 or recipe.get("decode_steps") != 12:
        raise ValueError("Unexpected teacher/decode protocol")
    for flag in ("outer280_loaded", "new_identity439_loaded", "test_loaded"):
        if recipe.get(flag) is not False:
            raise ValueError("Forbidden outer/test use")
    scope = recipe.get("data_scope", {})
    if scope.get("fit_clips") != 2315 or scope.get("validation_clips") != 405:
        raise ValueError("Only the locked2315/405internal experiment is allowed")
    records = {epoch: read_json(run / f"epoch{epoch:03}.json") for epoch in range(1, 11)}
    reports = {epoch: read_json(run / f"development_epoch{epoch:03}.json") for epoch in EPOCHS}
    for epoch, report in reports.items():
        if report.get("curve_provenance") is not None or report.get("checkpoint_selection_performed") is not False:
            raise ValueError("Expected original full/zero report-only evidence")
        if report["diagnostics"] != records[epoch]["diagnostics"] or records[epoch]["arm"] != recipe["args"]["arm"]:
            raise ValueError("Development report does not match the epoch record")
        if set(report["noise_reports"]) != set(SEEDS):
            raise ValueError("Unexpected noise set")
        for modes in report["noise_reports"].values():
            if set(modes) != set(MODES):
                raise ValueError("Only actual full/zero historical conditions are available")
    paths = [run / "provenance.json"] + [run / f"epoch{e:03}.json" for e in range(1, 11)] + [run / f"development_epoch{e:03}.json" for e in EPOCHS]
    return {"recipe": recipe, "records": records, "reports": reports,
        "sha256": {p.name: sha(p) for p in paths}}


def audit_reports(centered, raw, *, samples=5000):
    validate_centered_pair(centered["recipe"], raw["recipe"])
    validate_rng_prefix(centered["records"], raw["records"])
    arms = dict(zip(ARMS, (centered, raw)))
    collected, epochs = {}, {}
    for epoch in EPOCHS:
        for arm, data in arms.items():
            reports = data["reports"][epoch]["noise_reports"]
            for mode in MODES:
                for pop in POPULATIONS:
                    for group in GROUPS:
                        for kind in KINDS:
                            item = mean_noise_rows({seed: reports[seed][mode][pop][group][kind] for seed in SEEDS})
                            key = (epoch, arm, mode, pop, group, kind)
                            collected[key] = item
                            reference = collected[(EPOCHS[0], ARMS[0], "full", pop, group, kind)]
                            if item[0] != reference[0] or not np.array_equal(item[1][:, 1:], reference[1][:, 1:]):
                                raise ValueError("Targets/counts changed across epochs/arms/modes")
        # Frozen zero condition must agree for each noise, not just in average.
        for seed in SEEDS:
            for pop in POPULATIONS:
                for group in GROUPS:
                    for kind in KINDS:
                        a = centered["reports"][epoch]["noise_reports"][seed]["zero"][pop][group][kind]
                        b = raw["reports"][epoch]["noise_reports"][seed]["zero"][pop][group][kind]
                        sa, ra = sentence_rows(a); sb, rb = sentence_rows(b)
                        if sa != sb or not np.array_equal(ra, rb):
                            raise ValueError("Frozen zero report differs between arms")
        own = {arm: {pop: {group: {kind: compare_rows(collected[(epoch, arm, "full", pop, group, kind)],
            collected[(epoch, arm, "zero", pop, group, kind)], samples=samples) for kind in KINDS}
            for group in GROUPS} for pop in POPULATIONS} for arm in ARMS}
        between = {mode: {pop: {group: {kind: compare_rows(collected[(epoch, ARMS[0], mode, pop, group, kind)],
            collected[(epoch, ARMS[1], mode, pop, group, kind)], samples=samples) for kind in KINDS}
            for group in GROUPS} for pop in POPULATIONS} for mode in MODES}
        point_diagnostics = {arm: {mode: {"frozen_teacher_accuracy_mean": float(np.mean([
            data["reports"][epoch]["noise_reports"][seed][mode]["frozen_teacher_emotion_accuracy"] for seed in SEEDS])),
            "velocity_mse_per_second_mean": {pop: {group: float(np.mean([
                data["reports"][epoch]["noise_reports"][seed][mode][pop][group]["velocity_mse_per_second"]
                for seed in SEEDS])) for group in GROUPS} for pop in POPULATIONS}}
            for mode in MODES} for arm, data in arms.items()}
        epochs[str(epoch)] = {"same_completed_epochs": epoch, "same_optimizer_steps": epoch * 145,
            "full_minus_same_arm_zero": own, "centered_minus_raw": between,
            "point_only_diagnostics": point_diagnostics,
            "point_diagnostic_note": "Velocity has no sentence sufficient statistics here, so no CI. Frozen training teacher is not independent emotion evidence."}
    return {"schema": "centered_equal_budget_report_only_audit_v1", "epochs": epochs,
        "primary_equal_budget_epoch": 8, "supplemental_actual_checkpoint_epoch": 10,
        "budget_note": "Epoch8 matches the user's later requested budget; it was not the original predeclared final endpoint or selected best. Epoch10 is the last complete saved checkpoint after interruption, not an8epoch model.",
        "bootstrap_samples": samples, "bootstrap_seed": 45, "noise_seeds": list(SEEDS),
        "aggregation": "Average sufficient per-sentence SSE over3noises, retain common targets/counts once, then paired sentence bootstrap; never average generated curves or treat noise as independent samples.",
        "same_rng_epochs1_through10": True, "exact_report_target_and_counts_matched": True,
        "input_sha256_declared": centered["recipe"]["input_sha256"], "data_scope": centered["recipe"]["data_scope"],
        "source_reports_sha256": {arm: data["sha256"] for arm, data in arms.items()},
        "script_sha256": sha(__file__), "checkpoint_loaded": False, "curves_loaded": False,
        "cache_loaded": False, "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "original_epoch_reports_bound_to_checkpoint_or_curves": False,
        "report_integrity_limit": "Original epoch8/10 reports have curve_provenance=null. Hashes record available files now; recipe/RNG/epoch consistency does not retroactively provide missing checkpoint/curve binding.",
        "unavailable": ["epoch8checkpoint reconstruction", "historical raw epoch10curves", "reverse/oracle equal-budget comparison", "per-clip video/curve review", "speaker-stratified uncertainty from these reports"],
        "default_replaced": False, "training_resumed": False, "claimed_final18epoch_completion": False,
        "interpretation": "Report-only internal validation. Improvement over raw loss alone is insufficient: inspect own zero, nonneutral and neutral raw-motion errors, and centered dynamics separately. No all-pass/deployment conclusion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("centered", "raw", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        raise ValueError("Fresh output and positive bootstrap count required")
    report = audit_reports(load_reports(args.centered), load_reports(args.raw), samples=args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "equal_budget_epochs": list(EPOCHS), "evidence": "historical reports only; no epoch8checkpoint or raw10curves"}))


if __name__ == "__main__":
    main()
