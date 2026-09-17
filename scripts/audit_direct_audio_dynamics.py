"""Independent audit of the matched direct-audio flow and dynamics arms.

This script does not fit, select, or alter a checkpoint. It reconstructs the
bound source, replays the fixed training RNG schedule, regenerates every final
curve, and reports paired trajectory and finite-ensemble diagnostics.
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
from scripts.named_motion_distribution_metrics import audit_named_motion_samples
from scripts.audit_audio_activity_gate import velocity_clip_statistics
from scripts.audit_audio_conditioned_flow_probe import paired_score_summary
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary, scalar_summary
from scripts.audit_projection_delivery import mean_noise_statistics
from scripts.audit_projection_mean_dynamic_tradeoff import (
    average_error_components, exact_error_components, paired_component_change,
)
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_teacher_schedule_probe import GROUPS, audit_populations
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import (
    audio_features, basic_metrics, center, observed, sha, state_hash,
)

SCHEMA = "direct_audio_dynamics_v1"
AUDIT_SCHEMA = "direct_audio_dynamics_audit_v1"
ARMS = ("flow", "dynamics")
NOISE_SEEDS = (42, 123, 2026, 7, 19, 73, 211, 997)
MODES = ("full", "zero", "reverse")
KINDS = ("raw_motion", "centered_residual")
COMPARISONS = {
    "flow_full_vs_source_full": ("flow_full", "source_full"),
    "flow_full_vs_own_zero": ("flow_full", "flow_zero"),
    "flow_full_vs_own_reverse": ("flow_full", "flow_reverse"),
    "dynamics_full_vs_source_full": ("dynamics_full", "source_full"),
    "dynamics_full_vs_own_zero": ("dynamics_full", "dynamics_zero"),
    "dynamics_full_vs_own_reverse": ("dynamics_full", "dynamics_reverse"),
    "dynamics_full_vs_matched_flow_full": ("dynamics_full", "flow_full"),
}


def validate_curves(curves, reference, seeds, modes):
    seeds, modes = list(seeds), tuple(modes)
    if curves.get("noise_seeds") != seeds or curves.get("decode_steps") != 12:
        raise ValueError("Curve seed/decode protocol differs")
    if set(curves.get("motion", {})) != set(map(str, seeds)):
        raise ValueError("Curve seed keys differ")
    mask, shape = observed(reference["q"]), reference["q"]["motion"].shape
    for seed in seeds:
        rows = curves["motion"][str(seed)]
        if set(rows) != set(modes):
            raise ValueError("Curve modes differ")
        for prediction in rows.values():
            if prediction.shape != shape or not torch.isfinite(prediction[mask]).all():
                raise ValueError("Invalid observed motion curve")


def same_recipe(recipe):
    value = copy.deepcopy(recipe)
    value.pop("arm", None)
    return value


def _verify_state(mapping, digest, label):
    if not isinstance(mapping, dict) or not mapping or state_hash(mapping) != digest:
        raise ValueError(label + " state hash differs")
    for name, value in mapping.items():
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(label + " has an invalid tensor: " + name)


def _verify_inventory(run):
    inventory_path = run / "output_hashes.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf8"))
    for name, digest in inventory.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha(path) != digest:
            raise ValueError("Saved output/source changed: " + name)
    return {"inventory": sha(inventory_path), **inventory}


def load_arm(run, arm):
    run = Path(run).resolve()
    inventory = _verify_inventory(run)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf8"))
    recipe = provenance.get("recipe", {})
    summary = json.loads((run / "summary.json").read_text(encoding="utf8"))
    # The executed trainer saves epoch0 separately; summary.json contains
    # only the final report. Do not invent a summary['epoch000'] field.
    epoch0_reports = json.loads((run / "epoch000_evaluation.json").read_text(encoding="utf8"))
    if (set(epoch0_reports) != {"42"}
            or set(epoch0_reports["42"]) != set(MODES)):
        raise ValueError("Epoch0 evaluation seed/mode protocol differs")
    digest = canonical_hash(recipe)
    if (recipe.get("schema") != SCHEMA or recipe.get("arm") != arm
            or summary.get("schema") != SCHEMA or summary.get("arm") != arm
            or provenance.get("recipe_sha256") != digest or summary.get("recipe_sha256") != digest):
        raise ValueError("Recipe/schema/arm binding differs")
    expected = {"seed": 46, "epochs": 8, "batch_size": 16, "optimizer": "Adam",
        "renderer_lr": 1e-5, "encoder_lr": 1e-4, "clip_grad_norm": 1.,
        "decode_steps": 12, "eval_noise_seeds": list(NOISE_SEEDS),
        "eval_modes": list(MODES), "loss_weights": {"centered_upper": 1.},
        "default_replaced": False, "test_loaded": False,
        "checkpoint_selection_performed": False}
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("Fixed direct-audio protocol differs: " + arm)
    if (summary.get("completed_epochs") != 8 or summary.get("optimizer_steps") != 1160
            or summary.get("protected_unchanged") is not True
            or any(summary.get(key) is not False for key in
                   ("test_loaded", "default_replaced", "checkpoint_selection_performed"))):
        raise ValueError("Incomplete or selected direct-audio result")
    records = {epoch: json.loads((run / f"epoch{epoch:03d}.json").read_text(encoding="utf8"))
               for epoch in range(1, 9)}
    checkpoints = {}
    for epoch, name in ((0, "epoch000.pt"), (8, "final_epoch008.pt")):
        payload = torch.load(run / name, map_location="cpu", weights_only=False)
        if (payload.get("schema") != SCHEMA or payload.get("recipe") != recipe
                or payload.get("recipe_sha256") != digest
                or payload.get("completed_epochs") != epoch
                or payload.get("step") != epoch * 145
                or payload.get("protected_sha256") != recipe.get("protected_sha256")):
            raise ValueError("Checkpoint protocol differs: " + name)
        _verify_state(payload.get("renderer"), payload.get("renderer_sha256"), "renderer")
        _verify_state(payload.get("encoder"), payload.get("encoder_sha256"), "encoder")
        if state_hash(payload.get("motion_scales", {})) != recipe.get("motion_scales_sha256"):
            raise ValueError("Checkpoint motion scales differ")
        checkpoints[epoch] = payload
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    if (last.get("completed_epochs") != 8 or last.get("step") != 1160
            or state_hash(last.get("renderer", {})) != checkpoints[8]["renderer_sha256"]
            or state_hash(last.get("encoder", {})) != checkpoints[8]["encoder_sha256"]):
        raise ValueError("Last/final checkpoint differs")
    epoch0 = torch.load(run / "epoch000_curves.pt", map_location="cpu", weights_only=False)
    final = torch.load(run / "final_epoch008_curves.pt", map_location="cpu", weights_only=False, mmap=True)
    binding = summary.get("curve_provenance", {})
    sidecar_path = run / "final_epoch008_curves.provenance.json"
    if Path(binding.get("path", "")).resolve() != sidecar_path or binding.get("sha256") != sha(sidecar_path):
        raise ValueError("Final curve sidecar reference differs")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf8"))
    expected_binding = {"schema": "projection_schedule_curves_provenance_v1",
        "curve_sha256": sha(run / "final_epoch008_curves.pt"),
        "checkpoint_sha256": sha(run / "final_epoch008.pt"),
        "recipe_sha256": digest, "cache_sha256": recipe["input_sha256"]["cache"]}
    if sidecar != expected_binding:
        raise ValueError("Final checkpoint/curve/cache binding differs")
    return {"run": run, "recipe": recipe, "summary": summary, "epoch0_reports": epoch0_reports, "records": records,
        "checkpoints": checkpoints, "epoch0": epoch0, "final": final, "hashes": inventory}


def load_historical_source_curves(path, loaded, recipe, reference):
    """Bind the supplied baseline to the exact source system and cache.

    Merely checking seed42 zero curves would not authenticate the source-full
    comparison or the other seven seeds. Verify the original recipe, epoch0
    checkpoint and curve sidecar before consuming any historical predictions.
    """
    path = Path(path).resolve()
    if path.name != "epoch000_curves.pt":
        raise ValueError("Historical source must be an epoch000 curve artifact")
    run = path.parent
    provenance_path = run / "provenance.json"
    checkpoint_path = run / "epoch000.pt"
    evaluation_path = run / "epoch000_evaluation.json"
    sidecar_path = run / "epoch000_curves.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    historical = provenance.get("recipe", {})
    digest = canonical_hash(historical)
    if (historical.get("schema") != "audio_conditioned_flow_probe_v1"
            or provenance.get("recipe_sha256") != digest):
        raise ValueError("Historical source recipe/schema differs")
    expected_source = {
        "input_sha256": recipe["input_sha256"],
        "source_adapter_sha256": recipe["source_adapter_sha256"],
        "initial_system_sha256": recipe["initial_system_sha256"],
        "initial_head_sha256": state_hash(loaded["head"].state_dict()),
        "decode_steps": 12,
        "eval_noise_seeds": list(NOISE_SEEDS),
        "eval_modes": ["full", "zero", "reverse", "oracle"],
    }
    if any(historical.get(key) != value for key, value in expected_source.items()):
        raise ValueError("Historical baseline belongs to a different source/cache/protocol")
    system = loaded["system"]
    if state_hash(system.state_dict()) != recipe["initial_system_sha256"]:
        raise ValueError("Rebuilt historical source system differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("schema") != historical["schema"]
            or checkpoint.get("recipe") != historical
            or checkpoint.get("recipe_sha256") != digest
            or checkpoint.get("completed_epochs") != 0 or checkpoint.get("step") != 0):
        raise ValueError("Historical epoch0 checkpoint binding differs")
    name = checkpoint.get("trainable_name")
    parameters = dict(system.named_parameters())
    if (name not in parameters
            or state_hash(checkpoint.get("trainable", {})) != state_hash({name: parameters[name]})
            or checkpoint.get("frozen_state_sha256") != state_hash({
                key: value for key, value in system.state_dict().items() if key != name})
            or state_hash(checkpoint.get("local_projection", {})) != state_hash(system.local_projection.state_dict())
            or checkpoint.get("head_sha256") != expected_source["initial_head_sha256"]
            or state_hash(checkpoint.get("head", {})) != expected_source["initial_head_sha256"]):
        raise ValueError("Historical epoch0 weights differ from the rebuilt source")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf8"))
    expected_binding = {"schema": "projection_schedule_curves_provenance_v1",
        "curve_sha256": sha(path), "checkpoint_sha256": sha(checkpoint_path),
        "recipe_sha256": digest, "cache_sha256": recipe["input_sha256"]["cache"]}
    evaluation = json.loads(evaluation_path.read_text(encoding="utf8"))
    pointer = evaluation.get("curve_provenance", {})
    if (sidecar != expected_binding or pointer.get("sha256") != sha(sidecar_path)
            or Path(pointer.get("path", "")).resolve() != sidecar_path):
        raise ValueError("Historical source curve/checkpoint/cache binding differs")
    curves = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    validate_curves(curves, reference, NOISE_SEEDS, expected_source["eval_modes"])
    return curves, {"path": str(path), "sha256": sha(path),
        "recipe_sha256": digest, "checkpoint_sha256": sha(checkpoint_path),
        "sidecar_sha256": sha(sidecar_path), "provenance_sha256": sha(provenance_path),
        "evaluation_sha256": sha(evaluation_path), "source_system_and_cache_bound": True}


def audit_sources(recipe, run):
    training = {str(Path(path).resolve()): digest for path, digest in recipe["source_sha256"].items()}
    runner = Path(importlib.import_module("scripts.train_direct_audio_dynamics").__file__).resolve()
    if training.get(str(runner)) != sha(runner):
        raise ValueError("Imported direct-audio runner differs from executed source")
    result = {str(Path(__file__).resolve()): sha(__file__)}
    runner_root = next((Path(path).resolve().parents[1] for path in training
                        if Path(path).name == "train_direct_audio_dynamics.py"), None)
    if runner_root is None:
        raise ValueError("Training root is absent from source inventory")
    for name, digest in training.items():
        path = Path(name)
        if path.is_file() and sha(path) != digest:
            raise ValueError("Executed source changed: " + name)
        try:
            relative = path.relative_to(runner_root)
        except ValueError as exc:
            raise ValueError("Executed source lies outside training root: " + name) from exc
        snapshot = run / "source" / relative
        if not snapshot.is_file() or sha(snapshot) != digest:
            raise ValueError("Saved source snapshot differs: " + str(relative))
        result[str(snapshot.resolve())] = digest
    return result


def rebuild_fit_only_state(loaded, recipe, checkpoint, device):
    from scripts.train_direct_audio_dynamics import (
        DirectAudioEncoder, configure, feature_statistics, fit_motion_scales, protected_hash,
    )
    system, cache, bundle = loaded["system"], loaded["cache"], loaded["bundle"]
    tr = bundle["bundles"]["internal"]
    features, weight = audio_features(tr), tr["weight"].float()
    mean, std = feature_statistics(features, weight)
    encoder = DirectAudioEncoder(mean, std, system.local_projection.out_features).to(device).eval()
    configure(system)
    scales = fit_motion_scales(cache["splits"]["train"])
    if state_hash(system.state_dict()) != recipe["initial_system_sha256"]:
        raise ValueError("Rebuilt initial system differs")
    if state_hash(encoder.state_dict()) != recipe["initial_encoder_sha256"]:
        raise ValueError("Fit-only feature statistics or encoder initialization differs")
    if state_hash(scales) != recipe["motion_scales_sha256"]:
        raise ValueError("Fit-only motion scales differ")
    if protected_hash(system) != recipe["protected_sha256"]:
        raise ValueError("Protected source hash differs")
    if state_hash(checkpoint["encoder"]) != recipe["initial_encoder_sha256"]:
        raise ValueError("Epoch0 encoder is not the rebuilt fit-only initialization")
    for key, value in scales.items():
        torch.testing.assert_close(checkpoint["motion_scales"][key], value, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["encoder"]["feature_mean"], mean, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["encoder"]["feature_std"], std, rtol=0, atol=0)
    return system, encoder, scales


def validate_rng(arms, loaded):
    from scripts.train_projection_schedule_ablation import draws
    generator = torch.Generator().manual_seed(46)
    train = loaded["cache"]["splits"]["train"]
    n, shape = len(train["q"]["motion"]), train["q"]["motion"].shape[1:]
    batch_hash, noise_hash = hashlib.sha256(), hashlib.sha256()
    for epoch in range(1, 9):
        for ids in torch.randperm(n, generator=generator).split(16):
            noise, flow_time, _ = draws(generator, len(ids), shape)
            batch_hash.update(ids.numpy().tobytes())
            noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
        expected = {"epoch": epoch, "step": epoch * ((n + 15) // 16),
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest()}
        for arm, data in arms.items():
            if any(data["records"][epoch].get(key) != value for key, value in expected.items()):
                raise ValueError(f"Replayed RNG/budget differs: {arm}/{epoch}")
    for arm, data in arms.items():
        payload = data["checkpoints"][8]
        if (payload.get("minibatch_sha256") != batch_hash.hexdigest()
                or payload.get("noise_time_sha256") != noise_hash.hexdigest()
                or not torch.equal(payload["rng"]["training_generator"], generator.get_state())):
            raise ValueError("Final RNG evidence differs: " + arm)
    return {"minibatch_sha256": batch_hash.hexdigest(),
        "noise_time_sha256": noise_hash.hexdigest(), "optimizer_steps": 8 * ((n + 15) // 16)}


def statistics_for_conditions(curves, reference):
    q = reference["q"]
    target, weight, mask = q["motion"].float(), q["valid"].float(), observed(q)
    common = q["channel_mask"].all(0)
    if not torch.equal(q["channel_mask"], common[None].expand_as(q["channel_mask"])):
        raise ValueError("Require common observed channels")
    if any(channel >= len(common) or not common[channel] for cc in GROUPS.values() for channel in cc):
        raise ValueError("Required expression channel is unavailable")
    baseline = reference["base"]["b0"].float() + reference["identity"]["baseline"].float()[:, None]
    targets = {"raw_motion": target, "centered_residual": center(target - baseline, weight)}
    fixed = {(kind, group): clip_statistics(torch.zeros_like(target), truth, weight, channels)
             for kind, truth in targets.items() for group, channels in GROUPS.items()}
    collected, velocities, components = {}, {}, {}
    for seed in NOISE_SEEDS:
        for condition, prediction in curves["motion"][str(seed)].items():
            fields = {"raw_motion": prediction, "centered_residual": center(prediction - baseline, weight)}
            for kind, value in fields.items():
                for group, channels in GROUPS.items():
                    collected.setdefault((condition, kind, group), []).append(
                        clip_statistics(value, targets[kind], weight, channels))
            for group, channels in GROUPS.items():
                velocities.setdefault((condition, group), []).append(
                    velocity_clip_statistics(prediction, reference, channels))
                components.setdefault((condition, group), []).append(exact_error_components(
                    prediction, target, mask, channels, reference["base"]["b0"],
                    reference["identity"]["baseline"]))
    stats = {key: mean_noise_statistics(rows, fixed[key[1:]]) for key, rows in collected.items()}
    velocity = {}
    for key, rows in velocities.items():
        if any(not np.array_equal(row[:, 1], rows[0][:, 1]) for row in rows):
            raise ValueError("Velocity counts differ across noise seeds")
        value = np.stack(rows).mean(0); value[:, 1] = rows[0][:, 1]
        velocity[key] = value
    return stats, velocity, {key: average_error_components(rows) for key, rows in components.items()}


def distribution_pair(distributions, candidate, baseline, query, samples):
    result = {}
    for kind in KINDS:
        result[kind] = {}
        for population, groups in distributions["scores"][kind][candidate].items():
            ids = distributions["population_clip_indices"][population]
            sentences = [query["sentence_id"][index] for index in ids]
            result[kind][population] = {}
            for group, values in groups.items():
                base = distributions["scores"][kind][baseline][population][group]
                result[kind][population][group] = {}
                for score in ("trajectory_energy_score", "adjacent_variogram_score"):
                    left, right = values[score], base[score]
                    units = sentences
                    if score == "adjacent_variogram_score":
                        if left["included_clip_indices"] != right["included_clip_indices"]:
                            raise ValueError("Variogram observation sets differ")
                        units = [sentences[index] for index in left["included_clip_indices"]]
                    result[kind][population][group][score] = {estimator:
                        paired_score_summary(left[estimator]["per_clip"], right[estimator]["per_clip"],
                                             units, samples=samples) if units else None
                        for estimator in ("fair", "empirical")}
    return result


def analyze(curves, reference, samples):
    stats, velocity, components = statistics_for_conditions(curves, reference)
    distributions = audit_named_motion_samples(curves, reference, expected_seeds=NOISE_SEEDS)
    populations, q = audit_populations(reference["q"]), reference["q"]
    conditions = tuple(curves["motion"][str(NOISE_SEEDS[0])])
    scores = {condition: {kind: {group: {population:
        scalar_summary(stats[condition, kind, group][ids]) for population, ids in populations.items()}
        for group in GROUPS} for kind in KINDS} for condition in conditions}
    comparisons = {}
    for label, (candidate, baseline) in COMPARISONS.items():
        comparisons[label] = {
            "conditions": {"candidate": candidate, "baseline": baseline},
            "paired_gt": {population: {group: paired_summary(
                stats[candidate, "centered_residual", group], stats[baseline, "centered_residual", group],
                q["sentence_id"], ids, samples=samples) for group in GROUPS}
                for population, ids in populations.items()},
            "raw_relative": {population: {group: relative_error_audit(
                stats[candidate, "raw_motion", group][:, [0, 3]],
                stats[baseline, "raw_motion", group][:, [0, 3]], q["sentence_id"], ids, samples=samples)
                for group in GROUPS} for population, ids in populations.items()},
            "velocity_relative": {population: {group: relative_error_audit(
                velocity[candidate, group], velocity[baseline, group], q["sentence_id"], ids, samples=samples)
                for group in GROUPS} for population, ids in populations.items()},
            "mean_dynamic_decomposition": {population: {group: paired_component_change(
                components[candidate, group], components[baseline, group], q["sentence_id"], ids,
                samples=samples) for group in GROUPS} for population, ids in populations.items()},
            "distribution": distribution_pair(distributions, candidate, baseline, q, samples),
        }
    return {"scores": scores, "comparisons": comparisons, "distribution_scores": distributions,
        "population_counts": {population: {"clips": len(ids),
            "sentences": len({q["sentence_id"][index] for index in ids}),
            "identities": len({int(q["speaker_id"][index]) for index in ids})}
            for population, ids in populations.items()}}


def compact_report(report):
    selected = {}
    for label, comparison in report["analysis"]["comparisons"].items():
        selected[label] = {}
        for population in ("neutral", "nonneutral"):
            selected[label][population] = {}
            for group in ("brows", "eyes_expression", "mouth", "jaw17"):
                paired = comparison["paired_gt"][population][group]
                distribution = comparison["distribution"]["centered_residual"][population][group]
                selected[label][population][group] = {
                    "r2_improvement": paired["r2_improvement"],
                    "r2_improvement_ci95": paired["r2_improvement_ci95"],
                    "raw_relative": comparison["raw_relative"][population][group],
                    "velocity_relative": comparison["velocity_relative"][population][group],
                    "mean_dynamic_decomposition": comparison["mean_dynamic_decomposition"][population][group],
                    "fair_trajectory_energy": distribution["trajectory_energy_score"]["fair"],
                    "fair_adjacent_variogram": distribution["adjacent_variogram_score"]["fair"],
                }
    return {"schema": "direct_audio_dynamics_compact_v1",
        "verification": report["verification"], "comparisons": selected,
        "population_counts": report["analysis"]["population_counts"],
        "interpretation": "Existing internal development set. Positive paired improvement favors the candidate. No checkpoint selection, untouched-test, naturalness, or publication claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    parser.add_argument("--source-curves", type=Path,
                        default=Path("/root/kinetalk_runs/teacher_schedule_v1/audio_flow_v1/audio_local/epoch000_curves.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    compact_path = args.compact_output or args.output.with_name("compact_results.json")
    if args.output.exists() or compact_path.exists() or args.samples != 5000:
        raise ValueError("Fresh outputs and fixed 5000 sentence bootstrap samples required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    arms = {arm: load_arm(getattr(args, arm), arm) for arm in ARMS}
    recipe = arms["flow"]["recipe"]
    if same_recipe(recipe) != same_recipe(arms["dynamics"]["recipe"]):
        raise ValueError("Matched arms differ beyond the declared loss arm")
    if (arms["flow"]["checkpoints"][0]["renderer_sha256"] !=
            arms["dynamics"]["checkpoints"][0]["renderer_sha256"]
            or arms["flow"]["checkpoints"][0]["encoder_sha256"] !=
            arms["dynamics"]["checkpoints"][0]["encoder_sha256"]):
        raise ValueError("Matched arms do not share initialization")
    audit_source_hashes = {arm: audit_sources(data["recipe"], data["run"]) for arm, data in arms.items()}
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(46)
    from scripts.train_audio_conditioned_flow_probe import load_source
    from scripts.train_direct_audio_dynamics import evaluate, restore_direct
    loaded = load_source(recipe["source_run"], args.device)
    for key in ("input_sha256", "source_adapter_sha256", "data_scope"):
        if loaded[key] != recipe[key]:
            raise ValueError("Rebuilt source binding differs: " + key)
    system, encoder, scales = rebuild_fit_only_state(
        loaded, recipe, arms["flow"]["checkpoints"][0], args.device)
    for data in arms.values():
        if data["checkpoints"][0]["protected_sha256"] != recipe["protected_sha256"]:
            raise ValueError("Epoch0 protected source differs")
        for key, value in scales.items():
            torch.testing.assert_close(data["checkpoints"][0]["motion_scales"][key], value, rtol=0, atol=0)
    rng = validate_rng(arms, loaded)
    reference = loaded["cache"]["splits"]["validation"]
    dev = loaded["bundle"]["bundles"]["external_dev"]
    for data in arms.values():
        validate_curves(data["epoch0"], reference, (42,), MODES)
        validate_curves(data["final"], reference, NOISE_SEEDS, MODES)
    for mode in MODES:
        if not torch.equal(arms["flow"]["epoch0"]["motion"]["42"][mode],
                           arms["dynamics"]["epoch0"]["motion"]["42"][mode]):
            raise ValueError("Epoch0 differs across arms")
    source_curves, source_binding = load_historical_source_curves(
        args.source_curves, loaded, recipe, reference)
    old_zero = source_curves["motion"]["42"]["zero"]
    for arm, data in arms.items():
        for mode in MODES:
            torch.testing.assert_close(data["epoch0"]["motion"]["42"][mode], old_zero, rtol=0, atol=0)
            assert_metric_agreement(basic_metrics(data["epoch0"]["motion"]["42"][mode], reference),
                                    data["epoch0_reports"]["42"][mode])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reconstruction = {}
    for arm, data in arms.items():
        restore_direct(system, encoder, data["checkpoints"][8])
        temporary = args.output.with_name(args.output.stem + f"_{arm}_reconstructed.pt")
        if temporary.exists():
            raise FileExistsError("Fresh reconstruction path required: " + str(temporary))
        rebuilt_reports = evaluate(system, encoder, reference, dev, args.device, temporary)
        rebuilt = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
        errors = {}
        for seed in NOISE_SEEDS:
            errors[str(seed)] = {}
            for mode in MODES:
                saved = data["final"]["motion"][str(seed)][mode]
                actual = rebuilt["motion"][str(seed)][mode]
                torch.testing.assert_close(actual, saved, rtol=0, atol=0)
                errors[str(seed)][mode] = float((actual - saved).abs().max())
                recorded = data["summary"]["final"][str(seed)][mode]
                assert_metric_agreement(basic_metrics(saved, reference), recorded)
                assert_metric_agreement(rebuilt_reports[str(seed)][mode], recorded)
                if rebuilt_reports[str(seed)][mode]["frozen_teacher_emotion_accuracy"] != recorded["frozen_teacher_emotion_accuracy"]:
                    raise ValueError("Frozen teacher readout differs")
        reconstruction[arm] = {"max_abs_errors": errors, "temporary_sha256": sha(temporary)}
        del rebuilt
        temporary.unlink()
    combined = {"noise_seeds": list(NOISE_SEEDS), "decode_steps": 12, "motion": {}}
    for seed in NOISE_SEEDS:
        key = str(seed)
        combined["motion"][key] = {"source_full": source_curves["motion"][key]["full"]}
        for arm, data in arms.items():
            for mode in MODES:
                combined["motion"][key][f"{arm}_{mode}"] = data["final"]["motion"][key][mode]
    analysis = analyze(combined, reference, args.samples)
    report = {"schema": AUDIT_SCHEMA, "script_sha256": sha(__file__),
        "verification": {"inventories_and_source_hashes": True, "recipes_only_differ_by_arm": True,
            "fit_only_feature_statistics_and_motion_scales": True, "protected_source_hash": recipe["protected_sha256"],
            "rng_replayed_all8epochs": True, "epoch0_equals_historical_zero_seed42": True,
            "historical_source_system_and_cache_bound": True,
            "final_checkpoint_reconstruction_exact": True, "summary_metrics_recomputed": True,
            "test_loaded": False, "checkpoint_selection_performed": False, "default_replaced": False},
        "rng": rng, "source_curves": source_binding,
        "audit_source_sha256": audit_source_hashes,
        "arm_hashes": {arm: data["hashes"] for arm, data in arms.items()},
        "recipes": {arm: data["recipe"] for arm, data in arms.items()},
        "checkpoint_reconstruction": reconstruction, "analysis": analysis,
        "scope": "Fixed 2315-fit/405 internal development protocol; sentence-cluster bootstrap; noise seeds are not independent examples."}
    save_json(args.output, report)
    save_json(compact_path, compact_report(report))
    print(json.dumps({"audit": str(args.output.resolve()), "compact": str(compact_path.resolve()),
        "verification": report["verification"]}), flush=True)


if __name__ == "__main__":
    main()
