"""Independent fixed-epoch audit of the two audio-conditioned flow arms.

Single-GT trajectory fidelity and proper finite-ensemble distribution scores
are separate outcomes. Protection is prespecified against frozen uniform-full;
other baselines remain fully reported and cannot replace that reference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audio_flow_metrics import audit_audio_flow_samples
from scripts.audit_audio_activity_gate import velocity_clip_statistics
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.audit_projection_delivery import mean_noise_statistics
from scripts.audit_projection_mean_dynamic_tradeoff import exact_error_components, average_error_components, paired_component_change
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_teacher_schedule_probe import GROUPS, audit_populations
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import basic_metrics, center, observed, sha, state_hash

SCHEMA = "audio_conditioned_flow_probe_v1"
ARMS = ("audio_local", "zero_local")
NOISE_SEEDS = (42, 123, 2026, 7, 19, 73, 211, 997)
MODES = ("full", "zero", "reverse", "oracle")
KINDS = ("raw_motion", "centered_residual")
BASELINES = {"frozen_full": ("frozen", "full"), "frozen_zero": ("frozen", "zero"),
             "own_zero": ("audio_local", "zero"), "own_reverse": ("audio_local", "reverse"),
             "matched_adapted_zero": ("zero_local", "zero")}
REQUIRED_GAIN_BASELINES = ("frozen_full", "own_zero", "own_reverse", "matched_adapted_zero")


def validate_curves(curves, reference):
    if curves.get("noise_seeds") != list(NOISE_SEEDS) or curves.get("decode_steps") != 12:
        raise ValueError("Require the predeclared eight seeds and twelve decode steps")
    if set(curves.get("motion", {})) != set(map(str, NOISE_SEEDS)):
        raise ValueError("Noise keys differ")
    mask = observed(reference["q"])
    for modes in curves["motion"].values():
        if set(modes) != set(MODES):
            raise ValueError("Need full/zero/reverse/oracle for every seed")
        for pred in modes.values():
            if pred.shape != reference["q"]["motion"].shape or not torch.isfinite(pred[mask]).all():
                raise ValueError("Invalid observed motion curves")


def statistics_for_curves(curves, reference):
    """Existing sufficient-statistic contract extended to the fixed eight seeds."""
    validate_curves(curves, reference)
    q = reference["q"]
    target, weight = q["motion"].float(), q["valid"].float()
    common = q["channel_mask"].all(0)
    if not torch.equal(q["channel_mask"], common[None].expand_as(q["channel_mask"])):
        raise ValueError("Require common observed channels")
    if any(c >= len(common) or not common[c] for cc in GROUPS.values() for c in cc):
        raise ValueError("Required expression channels missing")
    baseline = reference["base"]["b0"].float() + reference["identity"]["baseline"].float()[:, None]
    targets = {"raw_motion": target, "centered_residual": center(target - baseline, weight)}
    fixed = {(kind, group): clip_statistics(torch.zeros_like(target), yy, weight, cc)
             for kind, yy in targets.items() for group, cc in GROUPS.items()}
    collected, velocity, components, per_seed = {}, {}, {}, {}
    for seed in NOISE_SEEDS:
        per_seed[str(seed)] = {}
        for mode, pred in curves["motion"][str(seed)].items():
            fields = {"raw_motion": pred, "centered_residual": center(pred - baseline, weight)}
            per_seed[str(seed)][mode] = {}
            for kind, value in fields.items():
                per_seed[str(seed)][mode][kind] = {}
                for group, cc in GROUPS.items():
                    row = clip_statistics(value, targets[kind], weight, cc)
                    collected.setdefault((mode, kind, group), []).append(row)
                    per_seed[str(seed)][mode][kind][group] = row
            for group, cc in GROUPS.items():
                velocity.setdefault((mode, group), []).append(velocity_clip_statistics(pred, reference, cc))
                components.setdefault((mode, group), []).append(exact_error_components(
                    pred, target, observed(q), cc, reference["base"]["b0"], reference["identity"]["baseline"]))
    stats = {key: mean_noise_statistics(rows, fixed[key[1:]]) for key, rows in collected.items()}
    velocities = {}
    for key, rows in velocity.items():
        if any(not np.array_equal(row[:, 1], rows[0][:, 1]) for row in rows):
            raise ValueError("Velocity observation counts differ across seeds")
        mean = np.stack(rows).mean(0); mean[:, 1] = rows[0][:, 1]
        velocities[key] = mean
    return stats, velocities, {key: average_error_components(rows) for key, rows in components.items()}, per_seed


def paired_score_summary(candidate, baseline, sentences, *, samples=5000):
    """Equal-clip score improvement, with sentences as paired bootstrap units."""
    a, b = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or len(a) != len(sentences) or not len(a):
        raise ValueError("Score/sentence pairing differs")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be finite")
    groups = {}
    for left, right, sentence in zip(a, b, sentences):
        groups.setdefault(str(sentence), np.zeros(2))[:] += [right - left, 1]
    ci = None
    if len(groups) > 1 and samples > 0:
        totals = np.stack(list(groups.values()))
        picks = np.random.default_rng(45).integers(len(totals), size=(samples, len(totals)))
        sums = totals[picks].sum(1)
        ci = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975]).tolist()
    return {"candidate_mean": float(a.mean()), "baseline_mean": float(b.mean()),
            "improvement": float((b - a).mean()), "improvement_ci95": ci,
            "clips": len(a), "sentences": len(groups), "bootstrap_samples": samples,
            "aggregation": "Equal clip scores; paired sentence bootstrap; positive improvement is better"}


def _positive_ci(row, key):
    interval = row.get(key)
    return bool(interval is not None and len(interval) == 2 and interval[0] > 0)


def _within(row, threshold, key="one_sided_90_upper"):
    return bool(row is not None and row.get(key) is not None and row[key] <= threshold)


def analyze_curves(arm_curves, reference, *, samples=5000):
    """Analyze immutable curves; never select epochs, fit parameters, or resample GT."""
    if set(arm_curves) != {"frozen", *ARMS}:
        raise ValueError("Need frozen, audio-local, and matched zero-local curves")
    q = reference["q"]; populations = audit_populations(q)
    if not all(pop in populations for pop in ("neutral", "nonneutral")):
        raise ValueError("Both neutral and nonneutral observations required")
    stats, velocity, components, per_seed, distributions = {}, {}, {}, {}, {}
    for arm, curves in arm_curves.items():
        stats[arm], velocity[arm], components[arm], per_seed[arm] = statistics_for_curves(curves, reference)
        distributions[arm] = audit_audio_flow_samples(curves, reference, expected_seeds=NOISE_SEEDS)
    scores = {arm: {mode: {kind: {group: {pop: scalar_summary(stats[arm][mode, kind, group][ids])
        for pop, ids in populations.items()} for group in GROUPS} for kind in KINDS} for mode in MODES} for arm in arm_curves}
    paired, raw, velocity_relative, mean_relative, mean_decomposition, correlation, distribution_comparisons = {}, {}, {}, {}, {}, {}, {}
    for name, (base_arm, base_mode) in BASELINES.items():
        paired[name] = {pop: {group: paired_summary(stats["audio_local"]["full", "centered_residual", group],
            stats[base_arm][base_mode, "centered_residual", group], q["sentence_id"], ids, samples=samples)
            for group in GROUPS} for pop, ids in populations.items()}
        raw[name] = {pop: {group: relative_error_audit(stats["audio_local"]["full", "raw_motion", group][:, [0, 3]],
            stats[base_arm][base_mode, "raw_motion", group][:, [0, 3]], q["sentence_id"], ids, samples=samples)
            for group in GROUPS} for pop, ids in populations.items()}
        velocity_relative[name] = {pop: {group: relative_error_audit(velocity["audio_local"]["full", group],
            velocity[base_arm][base_mode, group], q["sentence_id"], ids, samples=samples)
            for group in GROUPS} for pop, ids in populations.items()}
        mean_relative[name] = {pop: {group: relative_error_audit(components["audio_local"]["full", group][:, [2, 3]],
            components[base_arm][base_mode, group][:, [2, 3]], q["sentence_id"], ids, samples=samples)
            for group in GROUPS} for pop, ids in populations.items()}
        mean_decomposition[name] = {pop: {group: paired_component_change(components["audio_local"]["full", group],
            components[base_arm][base_mode, group], q["sentence_id"], ids, samples=samples)
            for group in GROUPS} for pop, ids in populations.items()}
        correlation[name] = {pop: scores["audio_local"]["full"]["raw_motion"]["mouth"][pop]["pooled_centered_correlation"] -
            scores[base_arm][base_mode]["raw_motion"]["mouth"][pop]["pooled_centered_correlation"] for pop in populations}
        distribution_comparisons[name] = {}
        for kind in KINDS:
            distribution_comparisons[name][kind] = {}
            candidate_populations = distributions["audio_local"]["scores"][kind]["full"]
            for pop, groups in candidate_populations.items():
                ids = distributions["audio_local"]["population_clip_indices"][pop]
                sentences = [q["sentence_id"][i] for i in ids]
                output = {}
                for group, values in groups.items():
                    baseline_values = distributions[base_arm]["scores"][kind][base_mode][pop][group]
                    rows = {}
                    for score in ("trajectory_energy_score", "adjacent_variogram_score"):
                        a, b = values[score], baseline_values[score]
                        score_sentences = sentences
                        if score == "adjacent_variogram_score":
                            if a["included_clip_indices"] != b["included_clip_indices"]:
                                raise ValueError("Variogram observed pairs differ")
                            score_sentences = [sentences[i] for i in a["included_clip_indices"]]
                        rows[score] = {estimator: (paired_score_summary(a[estimator]["per_clip"], b[estimator]["per_clip"],
                            score_sentences, samples=samples) if score_sentences else None) for estimator in ("fair", "empirical")}
                    output[group] = rows
                distribution_comparisons[name][kind][pop] = output
    paired_gt_checks = {"brow_improves_vs_" + baseline: _positive_ci(paired[baseline]["nonneutral"]["brows"], "r2_improvement_ci95")
                        for baseline in REQUIRED_GAIN_BASELINES}
    distribution_checks = {"brow_fair_es_improves_vs_" + baseline: _positive_ci(
        distribution_comparisons[baseline]["centered_residual"]["nonneutral"]["brows"]["trajectory_energy_score"]["fair"],
        "improvement_ci95") for baseline in REQUIRED_GAIN_BASELINES}
    protection = {
        "eyes_mouth_raw_mse_upper90_within_1pct": all(_within(raw["frozen_full"][pop][group], .01)
            for pop in ("neutral", "nonneutral") for group in ("eyes_expression", "mouth")),
        "mouth_correlation_drop_within_005": all(correlation["frozen_full"][pop] >= -.005 for pop in ("neutral", "nonneutral")),
        "mouth_upper_velocity_increase_within_5pct": all(_within(velocity_relative["frozen_full"][pop][group], .05,
            "relative_mse_increase") for pop in ("neutral", "nonneutral") for group in ("mouth", "upper_expression")),
        "nonneutral_upper_mean_mse_upper90_within_1pct": _within(mean_relative["frozen_full"]["nonneutral"]["upper_expression"], .01),
    }
    return {"scores": scores, "paired_gt_dynamic": paired, "raw_mse_relative": raw,
        "velocity_relative": velocity_relative, "mean_mse_relative": mean_relative,
        "raw_mean_dynamic_decomposition": mean_decomposition, "mouth_correlation_delta": correlation,
        "distribution_scores": distributions, "distribution_comparisons": distribution_comparisons,
        "paired_gt_dynamic_checks": paired_gt_checks, "generative_distribution_checks": distribution_checks,
        "protection_checks": protection, "protection_reference": "frozen_full only, fixed before outcomes",
        "paired_gt_dynamic_pass": all(paired_gt_checks.values()) and all(protection.values()),
        "generative_distribution_pass": all(distribution_checks.values()) and all(protection.values()),
        "population_counts": {pop: {"clips": len(ids), "sentences": len({q["sentence_id"][i] for i in ids}),
            "identities": len({int(q["speaker_id"][i]) for i in ids})} for pop, ids in populations.items()},
        "sampling_note": "Mean per-clip error statistics over eight seeds, then sentence bootstrap. Seeds are not independent data examples. R2 is random-sample fidelity, not ensemble-mean R2.",
        "inference_note": "Neither pass proves naturalness, independent emotion quality, untouched-test generalization, or readiness to replace defaults."}


def _same_recipe(recipe):
    value = copy.deepcopy(recipe)
    for key in ("arm", "output"):
        value["args"].pop(key, None)
    return value


def audit_source_hashes(recipe):
    """Bind the imported implementation, not another file named by provenance."""
    modules = ("scripts.train_audio_conditioned_flow_probe", "scripts.audio_flow_metrics",
        "scripts.audit_multiseed_stochasticity", "scripts.audit_teacher_schedule_probe",
        "scripts.audit_audio_activity_gate", "scripts.audit_formal_projection",
        "scripts.audit_predictable_motion_predictions", "scripts.audit_projection_delivery",
        "scripts.audit_projection_mean_dynamic_tradeoff", "scripts.audit_projection_readiness",
        "scripts.train_formal_predictable_projection", "scripts.train_predictable_renderer")
    result = {str(Path(__file__).resolve()): sha(__file__)}
    training_sources = {str(Path(path).resolve()): digest for path, digest in recipe["source_sha256"].items()}
    for name in modules:
        path = Path(importlib.import_module(name).__file__).resolve()
        digest = sha(path)
        if name.startswith("scripts.train_"):
            if training_sources.get(str(path)) != digest:
                raise ValueError("Imported runner/training dependency differs from executed recipe: " + name)
        result[str(path)] = digest
    # Capture every already imported project Python dependency transitively,
    # including metric helpers re-exported by the named modules.
    root = Path(__file__).resolve().parents[1]
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        # torch creates synthetic modules with relative __file__ values such
        # as '_ops.py'. They are not on-disk project sources.
        if not Path(filename).is_absolute():
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(root):
            continue
        digest = sha(path)
        if str(path) in training_sources and training_sources[str(path)] != digest:
            raise ValueError("Imported training source changed: " + str(path))
        result[str(path)] = digest
    return result


def load_arm(run, arm):
    run = Path(run)
    inventory = json.loads((run / "output_hashes.json").read_text(encoding="utf8"))
    for name, digest in inventory.items():
        if sha(run / name) != digest:
            raise ValueError("Saved output/source changed: " + name)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf8"))
    recipe, summary = provenance["recipe"], json.loads((run / "summary.json").read_text(encoding="utf8"))
    digest = canonical_hash(recipe)
    if (recipe.get("schema") != SCHEMA or summary.get("schema") != SCHEMA or recipe["args"].get("arm") != arm
            or summary.get("arm") != arm or provenance.get("recipe_sha256") != digest or summary.get("recipe_sha256") != digest):
        raise ValueError("Recipe/schema/arm binding differs")
    expected = {"seed": 46, "epochs": 8, "batch_size": 16, "lr": 1e-5, "optimizer": "Adam",
                "clip_grad_norm": 1., "teacher_local_probability": 0., "decode_steps": 12,
                "eval_noise_seeds": list(NOISE_SEEDS), "eval_modes": list(MODES),
                "loss": "observed standard flow velocity MSE only", "trainable_parameters": 36864,
                "all_modules_eval_mode": True, "audio_activity_gate": True}
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("Prespecified training/evaluation protocol differs")
    name = recipe.get("trainable", [None])[0]
    if (recipe.get("trainable") != [name] or not isinstance(name, str)
            or not name.startswith("renderer.blocks.") or not name.endswith(".cross_attention.out_proj.weight")):
        raise ValueError("Only the final block output matrix may train")
    for flag in ("outer280_loaded", "new_identity439_loaded", "test_loaded", "default_replaced"):
        if recipe.get(flag) is not False:
            raise ValueError("Data/default contract differs")
    if any(summary.get(key) is not True for key in ("frozen_unchanged", "head_unchanged", "projection_unchanged")):
        raise ValueError("Frozen model contract failed")
    if any(summary.get(key) is not False for key in ("checkpoint_selection_performed", "test_loaded", "default_replaced")):
        raise ValueError("Selection/test/default contract failed")
    if summary.get("completed_epochs") != 8 or summary.get("optimizer_steps") != 1160:
        raise ValueError("Incomplete fixed budget")
    # These are the actual files used at run time, not merely a legacy snapshot.
    for path, digest in recipe["source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Training source changed: " + path)
    records = {e: json.loads((run / f"epoch{e:03d}.json").read_text(encoding="utf8")) for e in range(1, 9)}
    checkpoints, curves, reports = {}, {}, {}
    for epoch, ck_name, curve_name, report_key, binding_key in (
        (0, "epoch000.pt", "epoch000_curves.pt", "epoch000", "epoch000_curve_provenance"),
        (8, "final_epoch008.pt", "final_epoch008_curves.pt", "final", "curve_provenance")):
        ck_path, curve_path = run / ck_name, run / curve_name
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if (ck.get("schema") != SCHEMA or ck.get("recipe") != recipe or ck.get("recipe_sha256") != canonical_hash(recipe)
                or ck.get("completed_epochs") != epoch or ck.get("step") != epoch * 145
                or ck.get("selection") != "none" or ck.get("default_enabled") is not False
                or ck.get("trainable_name") != name or set(ck.get("trainable", {})) != {name}
                or ck["trainable"][name].shape != (192, 192)):
            raise ValueError("Checkpoint protocol differs")
        if (state_hash(ck["head"]) != ck["head_sha256"] or state_hash(ck["local_projection"]) != ck["projection_sha256"]
                or ck["head_sha256"] != recipe["initial_head_sha256"] or ck["projection_sha256"] != recipe["initial_projection_sha256"]):
            raise ValueError("Frozen head/projection differs")
        if epoch == 0 and state_hash(ck["trainable"]) != recipe["initial_trainable_sha256"]:
            raise ValueError("Initial matrix differs")
        for key in ("minibatch_sha256", "noise_time_sha256", "teacher_choice_draw_sha256"):
            expected_hash = hashlib.sha256().hexdigest() if epoch == 0 else records[8][key]
            if ck[key] != expected_hash or (epoch == 8 and summary[key] != expected_hash):
                raise ValueError("Checkpoint/epoch/summary random evidence differs")
        sidecar_path = curve_path.with_name(curve_path.stem + ".provenance.json")
        binding = summary[binding_key]
        if Path(binding["path"]).resolve() != sidecar_path.resolve() or binding["sha256"] != sha(sidecar_path):
            raise ValueError("Curve sidecar binding differs")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf8"))
        expected_binding = {"schema": "projection_schedule_curves_provenance_v1", "curve_sha256": sha(curve_path),
            "checkpoint_sha256": sha(ck_path), "recipe_sha256": canonical_hash(recipe), "cache_sha256": recipe["input_sha256"]["cache"]}
        if any(sidecar.get(key) != value for key, value in expected_binding.items()):
            raise ValueError("Curve/checkpoint/cache provenance differs")
        checkpoints[epoch], curves[epoch], reports[epoch] = ck, torch.load(curve_path, map_location="cpu", weights_only=False), summary[report_key]
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    if (last["completed_epochs"] != 8 or last["step"] != 1160
            or state_hash(last["trainable"]) != state_hash(checkpoints[8]["trainable"])):
        raise ValueError("Last/final checkpoint differs")
    if checkpoints[0]["frozen_state_sha256"] != checkpoints[8]["frozen_state_sha256"]:
        raise ValueError("Frozen model hash changed")
    return {"run": run, "recipe": recipe, "summary": summary, "records": records,
            "checkpoints": checkpoints, "curves": curves, "reports": reports,
            "hashes": {"inventory": sha(run / "output_hashes.json"), **inventory}}


def validate_rng(arms, loaded):
    """Replay the actual historical draw recipe, not only matching saved hashes."""
    from scripts.train_projection_schedule_ablation import draws
    generator = torch.Generator().manual_seed(46)
    n = len(loaded["cache"]["splits"]["train"]["q"]["motion"])
    shape = loaded["cache"]["splits"]["train"]["q"]["motion"].shape[1:]
    batch_hash, noise_hash, choice_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for epoch in range(1, 9):
        order = torch.randperm(n, generator=generator)
        t0 = 0
        for ids in order.split(16):
            noise, flow_time, choose = draws(generator, len(ids), shape)
            batch_hash.update(ids.numpy().tobytes()); noise_hash.update(noise.numpy().tobytes())
            noise_hash.update(flow_time.numpy().tobytes()); choice_hash.update(choose.numpy().tobytes())
            t0 += int((flow_time == 0).sum())
        expected = {"minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
                    "teacher_choice_draw_sha256": choice_hash.hexdigest(), "samples_seen": n,
                    "step": epoch * ((n + 15) // 16), "epoch": epoch, "t0_fraction": t0 / n}
        for arm, data in arms.items():
            if any(data["records"][epoch].get(key) != value for key, value in expected.items()):
                raise ValueError(f"Replayed RNG/budget differs: {arm}/{epoch}")
    for data in arms.values():
        if not torch.equal(data["checkpoints"][8]["rng"]["training_generator"], generator.get_state()):
            raise ValueError("Final training generator state differs from replay")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-local", type=Path, required=True)
    p.add_argument("--zero-local", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=5000)
    args = p.parse_args()
    if args.output.exists() or args.samples != 5000:
        raise ValueError("Fresh output and fixed5000 sentence bootstrap required")
    from scripts.train_audio_conditioned_flow_probe import load_source, configure_adaptation, frozen_except_adaptation_hash, evaluate_probe
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    arms = {arm: load_arm(getattr(args, arm), arm) for arm in ARMS}
    recipe = arms["audio_local"]["recipe"]
    if _same_recipe(recipe) != _same_recipe(arms["zero_local"]["recipe"]):
        raise ValueError("Paired arms differ beyond conditioning and output")
    audit_sources = audit_source_hashes(recipe)
    loaded = load_source(recipe["args"]["source_run"], args.device)
    for key in ("source_recipe", "source_recipe_sha256", "source_provenance_sha256", "source_adapter_sha256",
                "source_summary_sha256", "source_curve_sha256", "source_curve_sidecar_sha256", "input_sha256", "data_scope"):
        if loaded[key] != recipe[key]:
            raise ValueError("Rebuilt source binding differs: " + key)
    system, head = loaded["system"], loaded["head"]
    name, parameter = configure_adaptation(system, expected_dim=192)
    if recipe["trainable"] != [name] or state_hash(system.state_dict()) != recipe["initial_system_sha256"]:
        raise ValueError("Adaptation initialization or final-block identity differs")
    frozen = frozen_except_adaptation_hash(system, name)
    for data in arms.values():
        for ck in data["checkpoints"].values():
            if ck["frozen_state_sha256"] != frozen:
                raise ValueError("Frozen model is not the source model")
    validate_rng(arms, loaded)
    reference = loaded["cache"]["splits"]["validation"]
    dev = loaded["bundle"]["bundles"]["external_dev"]
    for seed in NOISE_SEEDS:
        for mode in MODES:
            if not torch.equal(arms["audio_local"]["curves"][0]["motion"][str(seed)][mode], arms["zero_local"]["curves"][0]["motion"][str(seed)][mode]):
                raise ValueError("Epoch0 baselines differ across arms")
    source_curves = torch.load(Path(recipe["args"]["source_run"]) / "final_epoch008_curves.pt",
                               map_location="cpu", weights_only=False)
    if source_curves.get("noise_seeds") != list(NOISE_SEEDS[:3]) or source_curves.get("decode_steps") != 12:
        raise ValueError("Historical uniform curve protocol differs")
    for seed in NOISE_SEEDS[:3]:
        for mode in MODES:
            if not torch.equal(source_curves["motion"][str(seed)][mode], arms["audio_local"]["curves"][0]["motion"][str(seed)][mode]):
                raise ValueError("Epoch0 fails to reproduce historical uniform source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reproduction = {}
    # Reconstruct source plus each final matrix. Epoch0 is shared and checked above.
    for label, arm, epoch in (("frozen", "audio_local", 0), ("audio_local", "audio_local", 8), ("zero_local", "zero_local", 8)):
        with torch.no_grad():
            parameter.copy_(arms[arm]["checkpoints"][epoch]["trainable"][name].to(parameter))
        if frozen_except_adaptation_hash(system, name) != frozen:
            raise ValueError("Checkpoint restoration changed frozen model")
        saved = arms[arm]["curves"][epoch]
        validate_curves(saved, reference)
        temporary = args.output.with_name(args.output.stem + f"_{label}_reconstructed.pt")
        if temporary.exists():
            raise FileExistsError("Fresh reconstruction path required")
        reports = evaluate_probe(system, head, reference, dev, device=args.device,
            batch_size=recipe["args"]["eval_batch_size"], curves_path=temporary)
        rebuilt = torch.load(temporary, map_location="cpu", weights_only=False)
        errors = {}
        for seed in NOISE_SEEDS:
            errors[str(seed)] = {}
            for mode in MODES:
                torch.testing.assert_close(rebuilt["motion"][str(seed)][mode], saved["motion"][str(seed)][mode], rtol=0, atol=0)
                errors[str(seed)][mode] = float((rebuilt["motion"][str(seed)][mode] - saved["motion"][str(seed)][mode]).abs().max())
                assert_metric_agreement(basic_metrics(saved["motion"][str(seed)][mode], reference), arms[arm]["reports"][epoch][str(seed)][mode])
                assert_metric_agreement(reports[str(seed)][mode], arms[arm]["reports"][epoch][str(seed)][mode])
                if reports[str(seed)][mode]["frozen_teacher_emotion_accuracy"] != arms[arm]["reports"][epoch][str(seed)][mode]["frozen_teacher_emotion_accuracy"]:
                    raise ValueError("Frozen teacher readout differs")
        reproduction[label] = {"max_abs_errors": errors, "curves_sha256": sha(temporary)}
        temporary.unlink()
    curves = {"frozen": arms["audio_local"]["curves"][0], **{arm: data["curves"][8] for arm, data in arms.items()}}
    report = analyze_curves(curves, reference, samples=args.samples)
    teacher = {label: {mode: {"mean_accuracy": float(np.mean([arms[arm]["reports"][epoch][str(seed)][mode]["frozen_teacher_emotion_accuracy"] for seed in NOISE_SEEDS])),
        "per_seed": {str(seed): arms[arm]["reports"][epoch][str(seed)][mode]["frozen_teacher_emotion_accuracy"] for seed in NOISE_SEEDS}}
        for mode in MODES} for label, arm, epoch in (("frozen", "audio_local", 0), ("audio_local", "audio_local", 8), ("zero_local", "zero_local", 8))}
    report.update({"schema": "audio_conditioned_flow_audit_v1", "script_sha256": sha(__file__), "audit_source_sha256": audit_sources,
        "arm_hashes": {arm: data["hashes"] for arm, data in arms.items()}, "recipe": recipe,
        "checkpoint_reconstruction": reproduction, "rng_replayed_all8epochs": True,
        "frozen_source_hash_verified": frozen, "teacher_readout": teacher,
        "teacher_note": "Frozen training motion teacher only; not independent emotion/perceptual evidence and not an acceptance gate.",
        "test_loaded": False, "default_replaced": False, "checkpoint_selection_performed": False})
    save_json(args.output, report)
    print(json.dumps({"paired_gt_dynamic_pass": report["paired_gt_dynamic_pass"],
        "generative_distribution_pass": report["generative_distribution_pass"],
        "protection_checks": report["protection_checks"]}), flush=True)


if __name__ == "__main__":
    main()
