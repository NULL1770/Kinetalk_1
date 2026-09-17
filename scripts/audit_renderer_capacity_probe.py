"""Independent fixed-epoch audit of full-renderer oracle/audio/zero capacity.

The deployment gate is identical to the earlier audio-flow audit. The additional
motion-conditioned arm is a diagnostic; it can never make the audio gate pass.
No fitting, checkpoint selection, time alignment, gain fitting, or best-of-K.
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
from scripts.audit_audio_conditioned_flow_probe import (
    NOISE_SEEDS, MODES, KINDS, analyze_curves, paired_score_summary,
    statistics_for_curves, validate_curves, validate_rng,
)
from scripts.audio_flow_metrics import audit_audio_flow_samples
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_predictable_motion_predictions import paired_summary, scalar_summary
from scripts.audit_projection_mean_dynamic_tradeoff import paired_component_change
from scripts.audit_teacher_schedule_probe import GROUPS, audit_populations
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import basic_metrics, sha, state_hash

SCHEMA = "renderer_capacity_probe_v1"
ARMS = ("oracle_local", "audio_local", "zero_local")


def distribution_pair(candidate, candidate_mode, baseline, baseline_mode, query, *, samples):
    result = {}
    for kind in KINDS:
        result[kind] = {}
        for pop, groups in candidate["scores"][kind][candidate_mode].items():
            ids = candidate["population_clip_indices"][pop]
            if ids != baseline["population_clip_indices"][pop]:
                raise ValueError("Distribution population order differs")
            sentences = [query["sentence_id"][i] for i in ids]
            result[kind][pop] = {}
            for group, values in groups.items():
                output = {}
                for score in ("trajectory_energy_score", "adjacent_variogram_score"):
                    a = values[score]
                    b = baseline["scores"][kind][baseline_mode][pop][group][score]
                    score_sentences = sentences
                    if score == "adjacent_variogram_score":
                        if a["included_clip_indices"] != b["included_clip_indices"]:
                            raise ValueError("Variogram observed pairs differ")
                        score_sentences = [sentences[i] for i in a["included_clip_indices"]]
                    output[score] = {est: paired_score_summary(a[est]["per_clip"], b[est]["per_clip"],
                        score_sentences, samples=samples) if score_sentences else None for est in ("fair", "empirical")}
                result[kind][pop][group] = output
    return result


def analyze_capacity_curves(curves, reference, *, samples=5000):
    if set(curves) != {"frozen", *ARMS}:
        raise ValueError("Need frozen and all three oracle/audio/zero capacity arms")
    # This call preserves the published protection references, tolerances, and
    # single-GT/distribution acceptance rules exactly; oracle never enters it.
    report = analyze_curves({key: curves[key] for key in ("frozen", "audio_local", "zero_local")},
                            reference, samples=samples)
    q = reference["q"]
    populations = audit_populations(q)
    all_stats, all_components = {}, {}
    for arm, values in curves.items():
        all_stats[arm], _, all_components[arm], _ = statistics_for_curves(values, reference)
    oracle_dist = audit_audio_flow_samples(curves["oracle_local"], reference, expected_seeds=NOISE_SEEDS)
    distributions = {**report["distribution_scores"], "oracle_local": oracle_dist}
    report["distribution_scores"]["oracle_local"] = oracle_dist
    report["scores"]["oracle_local"] = {mode: {kind: {group: {
        pop: scalar_summary(all_stats["oracle_local"][mode, kind, group][ids])
        for pop, ids in populations.items()} for group in GROUPS} for kind in KINDS} for mode in MODES}
    pairs = {
        "trained_oracle_vs_frozen_oracle": ("oracle_local", "oracle", "frozen", "oracle"),
        "trained_oracle_vs_own_zero": ("oracle_local", "oracle", "oracle_local", "zero"),
        "trained_oracle_vs_audio_trained_oracle": ("oracle_local", "oracle", "audio_local", "oracle"),
        "trained_oracle_vs_matched_zero": ("oracle_local", "oracle", "zero_local", "zero"),
        "audio_trained_oracle_vs_own_zero": ("audio_local", "oracle", "audio_local", "zero"),
        "audio_trained_oracle_vs_frozen_oracle": ("audio_local", "oracle", "frozen", "oracle"),
        "oracle_trained_audio_vs_frozen_audio": ("oracle_local", "full", "frozen", "full"),
        "oracle_trained_audio_vs_own_zero": ("oracle_local", "full", "oracle_local", "zero"),
    }
    diagnostics = {}
    for label, (ca, cm, ba, bm) in pairs.items():
        diagnostics[label] = {
            "conditions": {"candidate": [ca, cm], "baseline": [ba, bm]},
            "paired_gt_dynamic": {pop: {group: paired_summary(
                all_stats[ca][cm, "centered_residual", group], all_stats[ba][bm, "centered_residual", group],
                q["sentence_id"], ids, samples=samples) for group in GROUPS} for pop, ids in populations.items()},
            "raw_mean_dynamic_decomposition": {pop: {group: paired_component_change(
                all_components[ca][cm, group], all_components[ba][bm, group], q["sentence_id"], ids,
                samples=samples) for group in GROUPS} for pop, ids in populations.items()},
            "distribution_comparisons": distribution_pair(distributions[ca], cm, distributions[ba], bm, q, samples=samples),
        }
    report["capacity_diagnostics"] = diagnostics
    report["oracle_note"] = "oracle_local training and oracle evaluation read true motion. Diagnostic capacity only; excluded from audio deployment acceptance."
    report["acceptance_definition"] = {
        "same_as": "audio_conditioned_flow_audit_v1",
        "paired_gt": "Nonneutral brow full gain 95% sentence CI positive against frozen full, own zero, own reverse, matched trained zero; all four protection checks.",
        "distribution": "Same comparisons with fair trajectory ES gain CI positive; same protections. Adjacent fair VS is reported separately, not silently substituted for ES.",
        "protection": "Eyes/mouth neutral and nonneutral raw MSE one-sided90 upper <=1%; mouth correlation drop <=.005; mouth/upper velocity point increase <=5%; nonneutral upper mean MSE one-sided90 upper <=1%, all relative to frozen full.",
        "individuals": "Every identity is reported descriptively, without using pooled success to certify all identities.",
    }
    return report


def compact_report(report):
    compact = {key: report[key] for key in ("paired_gt_dynamic_pass", "generative_distribution_pass",
        "paired_gt_dynamic_checks", "generative_distribution_checks", "protection_checks", "population_counts",
        "acceptance_definition", "oracle_note")}
    compact["comparisons"] = {}
    for baseline, populations in report["paired_gt_dynamic"].items():
        compact["comparisons"][baseline] = {}
        for population, groups in populations.items():
            compact["comparisons"][baseline][population] = {group: {
                "r2_gain": row["r2_improvement"], "r2_ci": row["r2_improvement_ci95"],
                "raw_mse_relative": report["raw_mse_relative"][baseline][population][group],
                "decomposition": report["raw_mean_dynamic_decomposition"][baseline][population][group],
            } for group, row in groups.items()}
        compact["comparisons"][baseline]["nonneutral_distribution"] = {
            group: {"fair_es": values["trajectory_energy_score"]["fair"],
                    "fair_vs": values["adjacent_variogram_score"]["fair"]}
            for group, values in report["distribution_comparisons"][baseline]["centered_residual"]["nonneutral"].items()}
    compact["capacity_diagnostics"] = {name: {
        "conditions": value["conditions"], "paired_gt_dynamic": value["paired_gt_dynamic"],
        "raw_mean_dynamic_decomposition": value["raw_mean_dynamic_decomposition"],
        "nonneutral_distribution": {group: {"fair_es": scores["trajectory_energy_score"]["fair"],
            "fair_vs": scores["adjacent_variogram_score"]["fair"]} for group, scores in
            value["distribution_comparisons"]["centered_residual"]["nonneutral"].items()},
    } for name, value in report["capacity_diagnostics"].items()}
    compact["scores"] = report["scores"]
    if "teacher_readout" in report:
        compact["teacher_readout"] = report["teacher_readout"]
    compact["interpretation"] = "Raw improvement is separated into mean and temporal errors. Motion oracle is not deployable audio quality. Existing internal development set, no untouched-test or perceptual claim."
    return compact


def same_recipe(recipe):
    result = copy.deepcopy(recipe)
    for key in ("arm", "output"):
        result["args"].pop(key, None)
    return result


def audit_source_hashes(recipe):
    required = ("scripts.train_renderer_capacity_probe", "scripts.train_audio_conditioned_flow_probe",
                "scripts.train_formal_predictable_projection", "scripts.train_predictable_renderer")
    sources = {str(Path(path).resolve()): digest for path, digest in recipe["source_sha256"].items()}
    for name in required:
        path = Path(importlib.import_module(name).__file__).resolve()
        if sources.get(str(path)) != sha(path):
            raise ValueError("Imported runner dependency differs from executed recipe: " + name)
    root = Path(__file__).resolve().parents[1]
    result = {str(Path(__file__).resolve()): sha(__file__)}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or not Path(filename).is_absolute():
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(root):
            digest = sha(path)
            if str(path) in sources and sources[str(path)] != digest:
                raise ValueError("Imported bound source changed: " + str(path))
            result[str(path)] = digest
    return result


def validate_capacity_state(payload, recipe, epoch):
    state = payload.get("capacity", {})
    if (not state or not any(k.startswith("renderer.") for k in state)
            or not any(k.startswith("local_projection.") for k in state)
            or any(not k.startswith(("renderer.", "local_projection.")) for k in state)):
        raise ValueError("Capacity state must contain renderer and local projection only")
    if (state_hash(state) != payload.get("capacity_sha256") or
            state_hash(payload["head"]) != payload.get("head_sha256") or
            payload.get("head_sha256") != recipe["initial_head_sha256"]):
        raise ValueError("Capacity or frozen head content hash differs")
    if epoch == 0 and state_hash(state) != recipe["initial_capacity_sha256"]:
        raise ValueError("Initial capacity differs")
    if (payload.get("schema") != SCHEMA or payload.get("recipe") != recipe
            or payload.get("recipe_sha256") != canonical_hash(recipe)
            or payload.get("completed_epochs") != epoch or payload.get("step") != epoch * 145
            or payload.get("selection") != "none" or payload.get("default_enabled") is not False):
        raise ValueError("Capacity checkpoint protocol differs")


def load_arm(run, arm):
    run = Path(run).resolve()
    inventory = json.loads((run / "output_hashes.json").read_text(encoding="utf8"))
    for name, digest in inventory.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != digest:
            raise ValueError("Saved output/source changed: " + name)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf8"))
    recipe = provenance["recipe"]
    summary = json.loads((run / "summary.json").read_text(encoding="utf8"))
    digest = canonical_hash(recipe)
    if (recipe.get("schema") != SCHEMA or summary.get("schema") != SCHEMA
            or recipe["args"].get("arm") != arm or summary.get("arm") != arm
            or provenance.get("recipe_sha256") != digest or summary.get("recipe_sha256") != digest):
        raise ValueError("Recipe/schema/arm binding differs")
    expected = {"seed": 46, "epochs": 8, "batch_size": 16, "lr": 1e-5, "optimizer": "Adam",
        "clip_grad_norm": 1., "decode_steps": 12, "eval_noise_seeds": list(NOISE_SEEDS),
        "eval_modes": list(MODES), "loss": "observed standard flow velocity MSE only",
        "all_modules_eval_mode": True, "audio_activity_gate": True}
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("Prespecified capacity protocol differs")
    names = recipe.get("trainable", [])
    if (not names or len(set(names)) != len(names) or
            any(not n.startswith(("renderer.", "local_projection.")) for n in names)
            or not any(n.startswith("local_projection.") for n in names)):
        raise ValueError("Only full renderer and local projection may train")
    for flag in ("outer280_loaded", "new_identity439_loaded", "test_loaded", "default_replaced"):
        if recipe.get(flag) is not False:
            raise ValueError("Data/default contract differs")
    if any(summary.get(k) is not True for k in ("frozen_unchanged", "head_unchanged")):
        raise ValueError("Frozen model contract failed")
    if any(summary.get(k) is not False for k in ("checkpoint_selection_performed", "test_loaded", "default_replaced")):
        raise ValueError("Selection/test/default contract failed")
    if summary.get("completed_epochs") != 8 or summary.get("optimizer_steps") != 1160:
        raise ValueError("Incomplete capacity budget")
    for path, digest in recipe["source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Executed training source changed: " + path)
    records = {e: json.loads((run / f"epoch{e:03d}.json").read_text(encoding="utf8")) for e in range(1, 9)}
    checkpoints, curves, reports = {}, {}, {}
    for epoch, stem, report_key, binding_key in ((0, "epoch000", "epoch000", "epoch000_curve_provenance"),
                                                (8, "final_epoch008", "final", "curve_provenance")):
        ck_path, curve_path = run / (stem + ".pt"), run / (stem + "_curves.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        validate_capacity_state(ck, recipe, epoch)
        for key in ("minibatch_sha256", "noise_time_sha256", "teacher_choice_draw_sha256"):
            expected_hash = hashlib.sha256().hexdigest() if epoch == 0 else records[8][key]
            if ck[key] != expected_hash or (epoch == 8 and summary[key] != expected_hash):
                raise ValueError("Checkpoint/epoch/summary random evidence differs")
        sidecar_path = run / (stem + "_curves.provenance.json")
        binding = summary[binding_key]
        if Path(binding["path"]).resolve() != sidecar_path or binding["sha256"] != sha(sidecar_path):
            raise ValueError("Curve sidecar binding differs")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf8"))
        expected_binding = {"schema": "projection_schedule_curves_provenance_v1", "curve_sha256": sha(curve_path),
            "checkpoint_sha256": sha(ck_path), "recipe_sha256": canonical_hash(recipe), "cache_sha256": recipe["input_sha256"]["cache"]}
        if sidecar != expected_binding:
            raise ValueError("Curve/checkpoint/cache provenance differs")
        checkpoints[epoch] = ck
        curves[epoch] = torch.load(curve_path, map_location="cpu", weights_only=False, mmap=True)
        reports[epoch] = summary[report_key]
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    validate_capacity_state(last, recipe, 8)
    if state_hash(last["capacity"]) != state_hash(checkpoints[8]["capacity"]):
        raise ValueError("Last/final capacity differs")
    if checkpoints[0]["frozen_state_sha256"] != checkpoints[8]["frozen_state_sha256"]:
        raise ValueError("Frozen model hash changed")
    return {"run": run, "recipe": recipe, "summary": summary, "records": records,
        "checkpoints": checkpoints, "curves": curves, "reports": reports,
        "hashes": {"inventory": sha(run / "output_hashes.json"), **inventory}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        p.add_argument("--" + arm.replace("_", "-"), type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=5000)
    args = p.parse_args()
    compact_path = args.output.with_name("compact_results.json")
    if args.output.exists() or compact_path.exists() or args.samples != 5000:
        raise ValueError("Fresh audit/compact outputs and 5000 sentence bootstrap required")
    from scripts.train_audio_conditioned_flow_probe import load_source, evaluate_probe
    from scripts.train_renderer_capacity_probe import configure_capacity, frozen_capacity_hash, restore_capacity_checkpoint
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    arms = {arm: load_arm(getattr(args, arm), arm) for arm in ARMS}
    recipe = arms["audio_local"]["recipe"]
    if any(same_recipe(recipe) != same_recipe(data["recipe"]) for data in arms.values()):
        raise ValueError("Capacity arms differ beyond conditioning/output")
    audit_sources = audit_source_hashes(recipe)
    loaded = load_source(recipe["args"]["source_run"], args.device)
    for key in ("source_recipe", "source_recipe_sha256", "source_provenance_sha256", "source_adapter_sha256",
                "source_summary_sha256", "source_curve_sha256", "source_curve_sidecar_sha256", "input_sha256", "data_scope"):
        if loaded[key] != recipe[key]:
            raise ValueError("Rebuilt source binding differs: " + key)
    system, head = loaded["system"], loaded["head"]
    configure_capacity(system)
    actual_names = [n for n, parameter in system.named_parameters() if parameter.requires_grad]
    if actual_names != recipe["trainable"] or state_hash(system.state_dict()) != recipe["initial_system_sha256"]:
        raise ValueError("Full capacity initialization/trainable set differs")
    frozen = frozen_capacity_hash(system)
    for data in arms.values():
        if any(ck["frozen_state_sha256"] != frozen for ck in data["checkpoints"].values()):
            raise ValueError("Frozen model is not source model")
    validate_rng(arms, loaded)
    reference = loaded["cache"]["splits"]["validation"]
    dev = loaded["bundle"]["bundles"]["external_dev"]
    initial = arms["audio_local"]["curves"][0]
    for data in arms.values():
        validate_curves(data["curves"][0], reference)
        for seed in NOISE_SEEDS:
            for mode in MODES:
                if not torch.equal(initial["motion"][str(seed)][mode], data["curves"][0]["motion"][str(seed)][mode]):
                    raise ValueError("Epoch0 baseline differs across capacity arms")
    historical = torch.load(Path(recipe["args"]["source_run"]) / "final_epoch008_curves.pt",
                            map_location="cpu", weights_only=False, mmap=True)
    if historical.get("noise_seeds") != list(NOISE_SEEDS[:3]) or historical.get("decode_steps") != 12:
        raise ValueError("Historical source seed protocol differs")
    for seed in NOISE_SEEDS[:3]:
        for mode in MODES:
            if not torch.equal(historical["motion"][str(seed)][mode], initial["motion"][str(seed)][mode]):
                raise ValueError("Epoch0 does not reproduce historical source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reproduction = {}
    # Rebuild source once and each final full-capacity checkpoint independently.
    labels = [("frozen", "audio_local", 0), *((arm, arm, 8) for arm in ARMS)]
    for label, arm, epoch in labels:
        restore_capacity_checkpoint(system, head, arms[arm]["checkpoints"][epoch], recipe=arms[arm]["recipe"])
        if frozen_capacity_hash(system) != frozen or state_hash(head.state_dict()) != recipe["initial_head_sha256"]:
            raise ValueError("Checkpoint restoration altered frozen system/head")
        saved = arms[arm]["curves"][epoch]
        validate_curves(saved, reference)
        temporary = args.output.with_name(args.output.stem + f"_{label}_reconstructed.pt")
        if temporary.exists():
            raise FileExistsError("Fresh reconstruction path required")
        reports = evaluate_probe(system, head, reference, dev, device=args.device,
            batch_size=recipe["args"]["eval_batch_size"], curves_path=temporary)
        rebuilt = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
        errors = {}
        for seed in NOISE_SEEDS:
            errors[str(seed)] = {}
            for mode in MODES:
                a, b = rebuilt["motion"][str(seed)][mode], saved["motion"][str(seed)][mode]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                errors[str(seed)][mode] = float((a - b).abs().max())
                assert_metric_agreement(basic_metrics(b, reference), arms[arm]["reports"][epoch][str(seed)][mode])
                assert_metric_agreement(reports[str(seed)][mode], arms[arm]["reports"][epoch][str(seed)][mode])
                if reports[str(seed)][mode]["frozen_teacher_emotion_accuracy"] != arms[arm]["reports"][epoch][str(seed)][mode]["frozen_teacher_emotion_accuracy"]:
                    raise ValueError("Frozen teacher readout differs")
        reproduction[label] = {"max_abs_errors": errors, "curves_sha256": sha(temporary)}
        del rebuilt
        temporary.unlink()
    curves = {"frozen": initial, **{arm: data["curves"][8] for arm, data in arms.items()}}
    report = analyze_capacity_curves(curves, reference, samples=args.samples)
    teacher = {label: {mode: {"mean_accuracy": float(np.mean([
        arms[arm]["reports"][epoch][str(seed)][mode]["frozen_teacher_emotion_accuracy"] for seed in NOISE_SEEDS])),
        "per_seed": {str(seed): arms[arm]["reports"][epoch][str(seed)][mode]["frozen_teacher_emotion_accuracy"] for seed in NOISE_SEEDS}}
        for mode in MODES} for label, arm, epoch in labels}
    report.update({"schema": "renderer_capacity_audit_v1", "script_sha256": sha(__file__),
        "audit_source_sha256": audit_sources, "arm_hashes": {arm: data["hashes"] for arm, data in arms.items()},
        "recipe": recipe, "checkpoint_reconstruction": reproduction, "rng_replayed_all8epochs": True,
        "frozen_source_hash_verified": frozen, "teacher_readout": teacher,
        "teacher_note": "Frozen training motion teacher only; not independent emotion/perceptual evidence or acceptance gate.",
        "test_loaded": False, "default_replaced": False, "checkpoint_selection_performed": False})
    save_json(args.output, report)
    save_json(compact_path, compact_report(report))
    print(json.dumps({"paired_gt_dynamic_pass": report["paired_gt_dynamic_pass"],
        "generative_distribution_pass": report["generative_distribution_pass"],
        "protection_checks": report["protection_checks"]}), flush=True)


if __name__ == "__main__":
    main()
