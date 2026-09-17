"""Independent fixed-epoch audit of output-motion audio/zero training.

Rebuilds the fit-only initialization, authenticates source and comparison
artifacts, replays all training draws, and regenerates each reported final
checkpoint. Finite-ensemble scores and paired single-GT fidelity stay separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_direct_audio_dynamics as direct_audit
from scripts.audit_audio_conditioned_flow_probe import _positive_ci, _within
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_predictable_motion_predictions import paired_summary, scalar_summary
from scripts.audit_projection_mean_dynamic_tradeoff import paired_component_change
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_teacher_schedule_probe import GROUPS, audit_populations
from scripts.named_motion_distribution_metrics import audit_named_motion_samples
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import basic_metrics, sha, state_hash

SCHEMA = "output_motion_dynamics_v1"
AUDIT_SCHEMA = "output_motion_dynamics_audit_v1"
ARMS = ("audio", "zero")
NOISE_SEEDS = direct_audit.NOISE_SEEDS
MODES = direct_audit.MODES
KINDS = direct_audit.KINDS
COMPARISONS = {
    "audio_full_vs_source_full": ("audio_full", "source_full"),
    "audio_full_vs_own_zero": ("audio_full", "audio_zero"),
    "audio_full_vs_own_reverse": ("audio_full", "audio_reverse"),
    "audio_full_vs_matched_trained_zero": ("audio_full", "zero_zero"),
    "audio_full_vs_direct_l1_full": ("audio_full", "direct_l1_full"),
    "matched_zero_vs_source_full": ("zero_zero", "source_full"),
}
GAIN_COMPARISONS = tuple(name for name in COMPARISONS if name.startswith("audio_full_vs_"))
HASH_KEYS = ("minibatch_sha256", "noise_time_sha256", "choice_draw_sha256")


def same_recipe(recipe):
    value = copy.deepcopy(recipe)
    for key in ("arm", "training_condition"):
        value.pop(key, None)
    return value


def validate_checkpoint(payload, recipe, epoch):
    if (payload.get("schema") != SCHEMA or payload.get("recipe") != recipe
            or payload.get("recipe_sha256") != canonical_hash(recipe)
            or payload.get("completed_epochs") != epoch or payload.get("step") != epoch * 145
            or payload.get("protected_sha256") != recipe.get("protected_sha256")):
        raise ValueError("Output-motion checkpoint protocol differs")
    for key in ("renderer", "encoder"):
        direct_audit._verify_state(payload.get(key), payload.get(key + "_sha256"), key)
    scales = payload.get("motion_scales", {})
    if (set(scales) != {"displacement", "std"}
            or state_hash(scales) != recipe.get("motion_scales_sha256")
            or any(not torch.isfinite(v).all() or (v <= 0).any() for v in scales.values())):
        raise ValueError("Motion scales changed or are invalid")
    if epoch == 0 or recipe["arm"] == "zero":
        if payload["encoder_sha256"] != recipe["initial_encoder_sha256"]:
            raise ValueError("Initial/zero encoder changed")
    if epoch == 0 and any(payload.get(key) != hashlib.sha256().hexdigest() for key in HASH_KEYS):
        raise ValueError("Initial training-draw hashes are nonempty")


def _validate_curve_binding(run, stem, recipe, pointer):
    path = run / (stem + "_curves.provenance.json")
    if (Path(pointer.get("path", "")).resolve() != path
            or pointer.get("sha256") != sha(path)):
        raise ValueError("Curve sidecar pointer differs")
    expected = {"schema": "projection_schedule_curves_provenance_v1",
                "curve_sha256": sha(run / (stem + "_curves.pt")),
                "checkpoint_sha256": sha(run / (stem + ".pt")),
                "recipe_sha256": canonical_hash(recipe),
                "cache_sha256": recipe["input_sha256"]["cache"]}
    if json.loads(path.read_text(encoding="utf8")) != expected:
        raise ValueError("Curve/checkpoint/cache binding differs")


def load_arm(run, arm):
    run = Path(run).resolve()
    inventory = direct_audit._verify_inventory(run)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf8"))
    recipe = provenance.get("recipe", {})
    summary = json.loads((run / "summary.json").read_text(encoding="utf8"))
    digest = canonical_hash(recipe)
    if (recipe.get("schema") != SCHEMA or recipe.get("arm") != arm
            or summary.get("schema") != SCHEMA or summary.get("arm") != arm
            or provenance.get("recipe_sha256") != digest or summary.get("recipe_sha256") != digest):
        raise ValueError("Recipe/schema/arm binding differs")
    expected = {"seed": 46, "epochs": 8, "batch_size": 16, "optimizer": "Adam",
        "renderer_lr": 1e-5, "encoder_lr": 1e-4, "clip_grad_norm": 1.,
        "decode_steps": 12, "eval_noise_seeds": list(NOISE_SEEDS), "eval_modes": list(MODES),
        "loss_weights": {"displacement": 1., "std": 1.},
        "groups": {"brows": list(range(41, 46)), "eyes_expression": [5, 6, 12, 13], "mouth": list(range(14, 41))},
        "std_groups": ["brows", "eyes_expression"], "displacement_floor": .005, "std_floor": .02,
        "std_ddof": 0, "velocity_fps_conversion": False, "all_modules_eval_mode": True,
        "training_condition": "zero-local" if arm == "zero" else "direct-audio-local",
        "default_replaced": False, "test_loaded": False, "checkpoint_selection_performed": False}
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("Fixed output-motion protocol differs")
    if (summary.get("completed_epochs") != 8 or summary.get("optimizer_steps") != 1160
            or summary.get("protected_unchanged") is not True
            or (arm == "zero" and summary.get("encoder_unchanged") is not True)
            or any(summary.get(key) is not False for key in
                   ("test_loaded", "default_replaced", "checkpoint_selection_performed"))):
        raise ValueError("Incomplete/selected or altered protected result")
    records = {epoch: json.loads((run / f"epoch{epoch:03d}.json").read_text(encoding="utf8"))
               for epoch in range(1, 9)}
    checkpoints = {}
    # Every fixed epoch is retained. Check its binding without selecting one.
    for epoch in range(9):
        payload = torch.load(run / f"epoch{epoch:03d}.pt", map_location="cpu", weights_only=False)
        validate_checkpoint(payload, recipe, epoch)
        if epoch and any(payload.get(key) != records[epoch].get(key) for key in HASH_KEYS):
            raise ValueError("Epoch checkpoint/record training draws differ")
        if epoch in (0, 8):
            checkpoints[epoch] = payload
    for name in ("last.pt", "final_epoch008.pt"):
        payload = torch.load(run / name, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, recipe, 8)
        for key in ("renderer_sha256", "encoder_sha256", *HASH_KEYS):
            if payload[key] != checkpoints[8][key]:
                raise ValueError("Epoch8/last/final checkpoint differs")
        if not torch.equal(payload["rng"]["training_generator"], checkpoints[8]["rng"]["training_generator"]):
            raise ValueError("Final checkpoint RNG differs")
    if any(summary.get(key) != checkpoints[8][key] for key in HASH_KEYS):
        raise ValueError("Summary/checkpoint draw evidence differs")
    epoch0_record = json.loads((run / "epoch000_evaluation.json").read_text(encoding="utf8"))
    if (epoch0_record.get("reports") != summary.get("epoch000")
            or epoch0_record.get("source_zero_max_abs_error") != 0
            or epoch0_record.get("curve_provenance") != summary.get("epoch000_curve_provenance")):
        raise ValueError("Epoch0 evaluation binding differs")
    _validate_curve_binding(run, "epoch000", recipe, summary["epoch000_curve_provenance"])
    _validate_curve_binding(run, "final_epoch008", recipe, summary["curve_provenance"])
    return {"run": run, "recipe": recipe, "summary": summary, "records": records,
        "checkpoints": checkpoints, "hashes": inventory,
        "epoch0": torch.load(run / "epoch000_curves.pt", map_location="cpu", weights_only=False),
        "final": torch.load(run / "final_epoch008_curves.pt", map_location="cpu", weights_only=False, mmap=True)}


def audit_sources(recipe, run):
    sources = {str(Path(path).resolve()): digest for path, digest in recipe["source_sha256"].items()}
    runner = Path(importlib.import_module("scripts.train_output_motion_dynamics").__file__).resolve()
    root = runner.parents[1]
    if sources.get(str(runner)) != sha(runner):
        raise ValueError("Imported output-motion runner differs from executed source")
    result = {}
    for name, digest in sources.items():
        path = Path(name)
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
            raise ValueError("Executed source changed or outside project: " + name)
        snapshot = run / "source" / path.relative_to(root)
        if not snapshot.is_file() or sha(snapshot) != digest:
            raise ValueError("Saved training source differs: " + name)
        result[str(snapshot)] = digest
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or not Path(filename).is_absolute():
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
            result[str(path)] = sha(path)
    return result


def validate_rng(arms, loaded):
    from scripts.train_projection_schedule_ablation import draws
    train = loaded["cache"]["splits"]["train"]
    n, shape = len(train["q"]["motion"]), train["q"]["motion"].shape[1:]
    if n != 2315:
        raise ValueError("Prespecified fit count differs")
    generator = torch.Generator().manual_seed(46)
    hashes = {key: hashlib.sha256() for key in HASH_KEYS}
    for epoch in range(1, 9):
        for ids in torch.randperm(n, generator=generator).split(16):
            noise, flow_time, choice = draws(generator, len(ids), shape)
            hashes["minibatch_sha256"].update(ids.numpy().tobytes())
            hashes["noise_time_sha256"].update(noise.numpy().tobytes())
            hashes["noise_time_sha256"].update(flow_time.numpy().tobytes())
            hashes["choice_draw_sha256"].update(choice.numpy().tobytes())
        expected = {"epoch": epoch, "step": epoch * 145,
                    **{key: value.hexdigest() for key, value in hashes.items()}}
        for arm, data in arms.items():
            if any(data["records"][epoch].get(key) != value for key, value in expected.items()):
                raise ValueError(f"Replayed RNG/budget differs: {arm}/{epoch}")
    for arm, data in arms.items():
        payload = data["checkpoints"][8]
        if (any(payload.get(key) != value.hexdigest() for key, value in hashes.items())
                or not torch.equal(payload["rng"]["training_generator"], generator.get_state())):
            raise ValueError("Final RNG evidence differs: " + arm)
    return {**{key: value.hexdigest() for key, value in hashes.items()}, "optimizer_steps": 1160}


def analyze(curves, reference, samples):
    names = set(curves["motion"][str(NOISE_SEEDS[0])])
    needed = {condition for pair in COMPARISONS.values() for condition in pair}
    if not needed.issubset(names):
        raise ValueError("A prespecified comparison condition is missing")
    stats, velocity, components = direct_audit.statistics_for_conditions(curves, reference)
    distributions = audit_named_motion_samples(curves, reference, expected_seeds=NOISE_SEEDS)
    populations, q = audit_populations(reference["q"]), reference["q"]
    scores = {condition: {kind: {group: {population:
        scalar_summary(stats[condition, kind, group][ids]) for population, ids in populations.items()}
        for group in GROUPS} for kind in KINDS} for condition in sorted(names)}
    comparisons = {}
    for label, (candidate, baseline) in COMPARISONS.items():
        comparisons[label] = {
            "conditions": {"candidate": candidate, "baseline": baseline},
            "paired_gt": {population: {group: paired_summary(
                stats[candidate, "centered_residual", group], stats[baseline, "centered_residual", group],
                q["sentence_id"], ids, samples=samples) for group in GROUPS}
                for population, ids in populations.items()},
            "raw_relative": {population: {group: relative_error_audit(
                stats[candidate, "raw_motion", group][:, [0, 3]], stats[baseline, "raw_motion", group][:, [0, 3]],
                q["sentence_id"], ids, samples=samples) for group in GROUPS} for population, ids in populations.items()},
            "velocity_relative": {population: {group: relative_error_audit(
                velocity[candidate, group], velocity[baseline, group], q["sentence_id"], ids, samples=samples)
                for group in GROUPS} for population, ids in populations.items()},
            "mean_relative": {population: {group: relative_error_audit(
                components[candidate, group][:, [2, 3]], components[baseline, group][:, [2, 3]],
                q["sentence_id"], ids, samples=samples) for group in GROUPS} for population, ids in populations.items()},
            "mean_dynamic_decomposition": {population: {group: paired_component_change(
                components[candidate, group], components[baseline, group], q["sentence_id"], ids, samples=samples)
                for group in GROUPS} for population, ids in populations.items()},
            "mouth_correlation_delta": {population: correlation_change(
                scores[candidate]["raw_motion"]["mouth"][population]["pooled_centered_correlation"],
                scores[baseline]["raw_motion"]["mouth"][population]["pooled_centered_correlation"])
                for population in populations},
            "distribution": direct_audit.distribution_pair(distributions, candidate, baseline, q, samples),
        }
    source = comparisons["audio_full_vs_source_full"]
    protection = {
        "eyes_mouth_raw_mse_upper90_within_1pct": all(_within(source["raw_relative"][pop][group], .01)
            for pop in ("neutral", "nonneutral") for group in ("eyes_expression", "mouth")),
        "mouth_correlation_drop_within_005": all(source["mouth_correlation_delta"][pop] is not None
            and source["mouth_correlation_delta"][pop] >= -.005
            for pop in ("neutral", "nonneutral")),
        "mouth_upper_velocity_increase_within_5pct": all(_within(source["velocity_relative"][pop][group], .05,
            "relative_mse_increase") for pop in ("neutral", "nonneutral") for group in ("mouth", "upper_expression")),
        "nonneutral_upper_mean_mse_upper90_within_1pct": _within(source["mean_relative"]["nonneutral"]["upper_expression"], .01),
    }
    paired_checks = {name: _positive_ci(comparisons[name]["paired_gt"]["nonneutral"]["brows"],
                                      "r2_improvement_ci95") for name in GAIN_COMPARISONS}
    distribution_checks = {name: _positive_ci(comparisons[name]["distribution"]["centered_residual"]
        ["nonneutral"]["brows"]["trajectory_energy_score"]["fair"], "improvement_ci95") for name in GAIN_COMPARISONS}
    return {"scores": scores, "comparisons": comparisons, "distribution_scores": distributions,
        "protection_checks": protection, "paired_gt_dynamic_checks": paired_checks,
        "generative_distribution_checks": distribution_checks,
        "paired_gt_dynamic_pass": all(paired_checks.values()) and all(protection.values()),
        "generative_distribution_pass": all(distribution_checks.values()) and all(protection.values()),
        "acceptance_definition": {
            "fidelity": "Nonneutral brow random-sample R2 gain sentence CI95 lower >0 against all five listed baselines plus protection.",
            "distribution": "Nonneutral brow fair trajectory ES improvement sentence CI95 lower >0 against all five baselines plus protection; fair adjacent VS reported separately.",
            "protection_reference": "source_full, fixed before outcomes; inherited 1% raw/mean MSE, .005 mouth correlation and 5% velocity thresholds.",
            "individuals": "Every identity is retained descriptively; pooled success does not certify all people.",
            "scope": "Internal development only; no naturalness, independent emotion, untouched-test, or publication claim."},
        "population_counts": {population: {"clips": len(ids),
            "sentences": len({q["sentence_id"][index] for index in ids}),
            "identities": len({int(q["speaker_id"][index]) for index in ids})}
            for population, ids in populations.items()}}


def correlation_change(candidate, baseline):
    """An undefined constant-trajectory correlation cannot certify protection."""
    return None if candidate is None or baseline is None else candidate - baseline


def compact_report(report):
    value = direct_audit.compact_report(report)
    value["schema"] = "output_motion_dynamics_compact_v1"
    analysis = report["analysis"]
    for key in ("scores", "protection_checks", "paired_gt_dynamic_checks", "generative_distribution_checks",
                "paired_gt_dynamic_pass", "generative_distribution_pass", "acceptance_definition"):
        value[key] = analysis[key]
    value["per_identity"] = {label: {population: groups for population, groups in comparison["paired_gt"].items()
        if population.startswith("speaker_")} for label, comparison in analysis["comparisons"].items()}
    return value


def reconstruct(label, system, encoder, payload, saved, recorded, reference, dev, output, device, restore):
    from scripts.train_direct_audio_dynamics import evaluate
    restore(system, encoder, payload)
    temporary = output.with_name(output.stem + f"_{label}_reconstructed.pt")
    if temporary.exists():
        raise FileExistsError("Fresh reconstruction path required: " + str(temporary))
    actual_reports = evaluate(system, encoder, reference, dev, device, temporary)
    rebuilt = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
    errors = {}
    for seed in NOISE_SEEDS:
        errors[str(seed)] = {}
        for mode in MODES:
            actual, target = rebuilt["motion"][str(seed)][mode], saved["motion"][str(seed)][mode]
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
            errors[str(seed)][mode] = float((actual - target).abs().max())
            assert_metric_agreement(basic_metrics(target, reference), recorded[str(seed)][mode])
            assert_metric_agreement(actual_reports[str(seed)][mode], recorded[str(seed)][mode])
            if actual_reports[str(seed)][mode]["frozen_teacher_emotion_accuracy"] != recorded[str(seed)][mode]["frozen_teacher_emotion_accuracy"]:
                raise ValueError("Frozen training-teacher readout differs")
    evidence = {"max_abs_errors": errors, "temporary_sha256": sha(temporary)}
    del rebuilt
    temporary.unlink()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio", "zero", "direct-l1", "source-curves", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    compact_path = args.compact_output or args.output.with_name("compact_results.json")
    if args.output.exists() or compact_path.exists() or args.samples != 5000:
        raise ValueError("Fresh outputs and fixed 5000 sentence bootstrap samples required")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    arms = {arm: load_arm(getattr(args, arm), arm) for arm in ARMS}
    recipe = arms["audio"]["recipe"]
    if same_recipe(recipe) != same_recipe(arms["zero"]["recipe"]):
        raise ValueError("Matched arms differ beyond declared conditioning")
    for key in ("renderer_sha256", "encoder_sha256"):
        if arms["audio"]["checkpoints"][0][key] != arms["zero"]["checkpoints"][0][key]:
            raise ValueError("Matched arms do not share initialization")
    sources = {arm: audit_sources(data["recipe"], data["run"]) for arm, data in arms.items()}
    historical_l1 = direct_audit.load_arm(args.direct_l1, "dynamics")
    sources["direct_l1"] = direct_audit.audit_sources(historical_l1["recipe"], historical_l1["run"])
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(46)
    from scripts.train_output_motion_dynamics import load_initial_source, fit_output_scales, restore_output
    from scripts.train_direct_audio_dynamics import protected_hash, restore_direct
    loaded, encoder, direct_source = load_initial_source(recipe["direct_source"]["run"], args.device)
    if direct_source != recipe["direct_source"]:
        raise ValueError("Direct initialization source binding differs")
    system = loaded["system"]
    for key in ("input_sha256", "source_adapter_sha256", "data_scope"):
        if loaded[key] != recipe[key] or historical_l1["recipe"][key] != recipe[key]:
            raise ValueError("Source/data binding differs: " + key)
    for key in ("initial_system_sha256", "initial_encoder_sha256", "protected_sha256"):
        if historical_l1["recipe"][key] != recipe[key]:
            raise ValueError("Historical L1 initialization differs")
    scales = fit_output_scales(loaded["cache"]["splits"]["train"])
    if (state_hash(scales) != recipe["motion_scales_sha256"]
            or protected_hash(system) != recipe["protected_sha256"]):
        raise ValueError("Rebuilt fit-only scales/protected source differ")
    for data in arms.values():
        for key, expected in (("renderer", system.renderer.state_dict()), ("encoder", encoder.state_dict())):
            if state_hash(data["checkpoints"][0][key]) != state_hash(expected):
                raise ValueError("Epoch0 does not match restored initialization")
        for checkpoint in data["checkpoints"].values():
            for key, value in scales.items():
                torch.testing.assert_close(checkpoint["motion_scales"][key], value, rtol=0, atol=0)
    rng = validate_rng(arms, loaded)
    direct_audit.validate_rng({"direct_l1": historical_l1}, loaded)
    reference = loaded["cache"]["splits"]["validation"]
    dev = loaded["bundle"]["bundles"]["external_dev"]
    source_curves, source_binding = direct_audit.load_historical_source_curves(
        args.source_curves, loaded, recipe, reference)
    for data in (*arms.values(), historical_l1):
        direct_audit.validate_curves(data["epoch0"], reference, (42,), MODES)
        direct_audit.validate_curves(data["final"], reference, NOISE_SEEDS, MODES)
        for mode in MODES:
            torch.testing.assert_close(data["epoch0"]["motion"]["42"][mode],
                source_curves["motion"]["42"]["zero"], rtol=0, atol=0)
    for seed in NOISE_SEEDS:
        zero_modes = arms["zero"]["final"]["motion"][str(seed)]
        for mode in MODES:
            torch.testing.assert_close(zero_modes[mode], zero_modes["zero"], rtol=0, atol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reconstruction = {}
    for arm, data in arms.items():
        reconstruction[arm] = reconstruct(arm, system, encoder, data["checkpoints"][8], data["final"],
            data["summary"]["final"], reference, dev, args.output, args.device, restore_output)
    reconstruction["direct_l1"] = reconstruct("direct_l1", system, encoder,
        historical_l1["checkpoints"][8], historical_l1["final"], historical_l1["summary"]["final"],
        reference, dev, args.output, args.device, restore_direct)
    combined = {"noise_seeds": list(NOISE_SEEDS), "decode_steps": 12, "motion": {}}
    for seed in NOISE_SEEDS:
        key = str(seed)
        combined["motion"][key] = {"source_full": source_curves["motion"][key]["full"],
            "direct_l1_full": historical_l1["final"]["motion"][key]["full"]}
        for arm, data in arms.items():
            for mode in MODES:
                combined["motion"][key][f"{arm}_{mode}"] = data["final"]["motion"][key][mode]
    analysis = analyze(combined, reference, args.samples)
    report = {"schema": AUDIT_SCHEMA, "script_sha256": sha(__file__),
        "verification": {"inventories_and_source_hashes": True, "recipes_only_differ_by_condition": True,
            "fit_only_feature_statistics_and_motion_scales": True, "protected_source_hash": recipe["protected_sha256"],
            "rng_replayed_all8epochs": True, "choice_draws_replayed": True,
            "epoch0_equals_historical_zero_seed42": True, "zero_encoder_unchanged_all_epochs": True,
            "zero_final_modes_identical": True, "historical_source_system_and_cache_bound": True,
            "audio_zero_direct_l1_final_checkpoint_reconstruction_exact": True,
            "summary_metrics_recomputed": True, "test_loaded": False,
            "checkpoint_selection_performed": False, "default_replaced": False},
        "rng": rng, "source_curves": source_binding, "audit_source_sha256": sources,
        "arm_hashes": {arm: data["hashes"] for arm, data in arms.items()},
        "direct_l1_hashes": historical_l1["hashes"],
        "recipes": {arm: data["recipe"] for arm, data in arms.items()},
        "checkpoint_reconstruction": reconstruction, "analysis": analysis,
        "teacher_readout": {arm: data["summary"]["final"] for arm, data in arms.items()},
        "teacher_note": "Frozen training teacher readout only, not independent emotion/perceptual evidence.",
        "scope": "Fixed 2315-fit/405 internal development; sentence-cluster bootstrap; seeds are not independent examples."}
    save_json(args.output, report); save_json(compact_path, compact_report(report))
    print(json.dumps({"audit": str(args.output.resolve()), "paired_gt_dynamic_pass": analysis["paired_gt_dynamic_pass"],
        "generative_distribution_pass": analysis["generative_distribution_pass"],
        "protection_checks": analysis["protection_checks"]}), flush=True)


if __name__ == "__main__":
    main()
