"""Epoch-based, resumable training of the existing shared dynamic projection.

The complete renderer, audio head, basis and global/identity/B0 paths remain
frozen. A successful run is an experiment, not permission to replace defaults.
No test split is loaded. Cached B0 and original global normalization are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.train_predictable_renderer import (
    PredictableAudioHead, audio_features, basic_metrics, batch_to_device,
    cached_flow, configure_trainable, observed, optimize, projected_affect,
    reverse_controls, sha, state_hash, validate_inputs,
)


SCHEMA = "formal_predictable_projection_resume_v1"
CURVE_PROVENANCE_SCHEMA = "formal_predictable_projection_curves_provenance_v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(cpu_tree(value), temporary)
    temporary.replace(path)


def write_curve_provenance(curves_path, *, recipe_sha256, selected_checkpoint_sha256, cache_sha256):
    """Bind generated curves to the selected adapter and immutable inputs."""
    curves_path = Path(curves_path)
    sidecar = curves_path.with_name(curves_path.stem + ".provenance.json")
    value = {"schema": CURVE_PROVENANCE_SCHEMA, "curve_sha256": sha(curves_path),
             "recipe_sha256": recipe_sha256, "selected_checkpoint_sha256": selected_checkpoint_sha256,
             "cache_sha256": cache_sha256}
    save_json(sidecar, value)
    return {"path": str(sidecar.resolve()), "sha256": sha(sidecar), "schema": CURVE_PROVENANCE_SCHEMA}


def capture_rng(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "training_generator": generator.get_state()}


def restore_rng(state, generator):
    if len(state["torch_cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("Resume CUDA RNG device count differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["training_generator"])


def teacher_probability(epoch, batch_index, batches_per_epoch, teacher_epochs=5):
    """Zero-based epoch; ends at exactly zero on the last warmup batch."""
    if epoch < 0 or batch_index < 0 or batches_per_epoch < 1 or teacher_epochs < 1:
        raise ValueError("Invalid teacher schedule position")
    step = epoch * batches_per_epoch + batch_index
    return .5 * max(0., 1. - step / max(teacher_epochs * batches_per_epoch - 1, 1))


def mean_metric(evaluations, mode, *keys):
    values = []
    for report in evaluations.values():
        value = report[mode]
        for key in keys:
            value = value[key]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Undefined selection metric: {mode}/{keys}")
        values.append(float(value))
    if not values:
        raise ValueError("No development noise evaluations")
    return float(np.mean(values))


def selection_metrics(evaluations, original_class_accuracy):
    """Predeclared point-estimate checkpoint rule, separate from launch CIs."""
    if len(evaluations) != 3:
        raise ValueError("Checkpoint selection requires three fixed generation noise seeds")
    def metric(mode, pop, region, metric_name, field):
        return mean_metric(evaluations, mode, pop, region, metric_name, field)
    full = metric("full", "nonneutral", "upper_expression", "centered_residual", "r2_against_zero")
    zero = metric("zero", "nonneutral", "upper_expression", "centered_residual", "r2_against_zero")
    changes = {}
    for region in ("mouth", "upper_expression"):
        prediction = metric("full", "neutral", region, "raw_motion", "native_mse")
        reference = metric("zero", "neutral", region, "raw_motion", "native_mse")
        if reference <= 0:
            raise ValueError("Nonpositive neutral zero-local baseline error")
        changes[region] = prediction / reference - 1
    correlation_delta = metric("full", "all", "mouth", "raw_motion", "pooled_centered_correlation") - metric(
        "zero", "all", "mouth", "raw_motion", "pooled_centered_correlation")
    class_delta = mean_metric(evaluations, "full", "frozen_teacher_emotion_accuracy") - original_class_accuracy
    velocity_changes = {}
    for population, region in (("neutral", "upper_expression"), ("neutral", "mouth"),
                               ("all", "mouth"), ("nonneutral", "upper_expression")):
        reference = mean_metric(evaluations, "zero", population, region, "velocity_mse_per_second")
        prediction = mean_metric(evaluations, "full", population, region, "velocity_mse_per_second")
        if reference <= 0:
            raise ValueError("Nonpositive zero-local velocity error")
        velocity_changes[f"{population}/{region}"] = prediction / reference - 1
    checks = {"positive_upper_gain": full - zero > 0,
              "neutral_mouth_mse_within_3pct": changes["mouth"] <= .03,
              "neutral_upper_mse_within_3pct": changes["upper_expression"] <= .03,
              "mouth_correlation_drop_within_point01": correlation_delta >= -.01,
              "class_accuracy_drop_within_one_pp": class_delta >= -.01,
              "velocity_degradation_within_5pct": all(v <= .05 for v in velocity_changes.values())}
    return {"eligible": all(checks.values()), "upper_full_minus_zero_r2": full - zero,
            "upper_full_r2": full, "upper_zero_r2": zero, "neutral_raw_mse_relative_change": changes,
            "all_mouth_correlation_full_minus_zero": correlation_delta,
            "velocity_mse_relative_change": velocity_changes,
            "class_accuracy_full_minus_original": class_delta, "checks": checks,
            "scope": "Three-noise mean point estimates for checkpoint selection; final paired sentence uncertainty and launch tests remain separate."}


def frozen_hash(system):
    return state_hash({k: v for k, v in system.state_dict().items() if not k.startswith("local_projection.")})


def restore_checkpoint(payload, *, recipe, system, head, optimizer, generator):
    if payload.get("schema") != SCHEMA or payload.get("recipe_sha256") != canonical_hash(recipe):
        raise ValueError("Resume recipe/config/input hash mismatch")
    if payload.get("recipe") != recipe:
        raise ValueError("Resume recipe differs despite supplied hash")
    if payload["frozen_state_sha256"] != frozen_hash(system):
        raise ValueError("Resume frozen original checkpoint state differs")
    if payload["head_sha256"] != state_hash(head.state_dict()):
        raise ValueError("Resume fixed audio head differs")
    if state_hash(payload["head"]) != payload["head_sha256"]:
        raise ValueError("Corrupt fixed head in resume checkpoint")
    if payload["completed_epochs"] < 0 or payload["step"] < 0:
        raise ValueError("Negative resume progress")
    system.local_projection.load_state_dict(payload["local_projection"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"], generator)


@torch.no_grad()
def evaluate_compact(system, head, split, bundle, *, device, modes=("full", "zero"),
                     seeds=(42, 123, 2026), batch_size=32, decode_steps=12,
                     original=False, curves_path=None):
    system.eval()
    features, weight = audio_features(bundle), bundle["weight"].float()
    reports, saved_curves = {}, {}
    for seed in seeds:
        noise = torch.randn(split["q"]["motion"].shape, generator=torch.Generator().manual_seed(seed))
        curves, logits = {mode: [] for mode in modes}, {mode: [] for mode in modes}
        for start in range(0, len(features), batch_size):
            ids = torch.arange(start, min(start + batch_size, len(features)))
            batch = batch_to_device(split, ids, device)
            q, w = batch["q"], weight[ids].to(device)
            controls = head(features[ids].to(device), w) if head is not None else None
            for mode in modes:
                if original:
                    affect = batch["affect"]
                else:
                    z = controls
                    if mode == "reverse":
                        z = reverse_controls(z, w)
                    elif mode == "oracle":
                        z = head.teacher(bundle["motion_bins"][ids].float().to(device), w)
                    affect = projected_affect(system, batch, z, w, zero=mode == "zero", audio_gate=True)
                pred = system.generate(q["content"], q["valid"], batch["identity"], affect,
                    initial_noise=noise[ids].to(device), steps=decode_steps, base=batch["base"])["motion"]
                curves[mode].append(pred.cpu())
                residual = torch.where(observed(q), pred - batch["base"]["b0"] - batch["identity"]["baseline"][:, None], 0)
                logits[mode].append(system.motion_teacher(residual, q["valid"])["emotion_logits"].cpu())
        reports[str(seed)] = {}
        for mode in modes:
            prediction = torch.cat(curves[mode])
            report = basic_metrics(prediction, split)
            classes = torch.cat(logits[mode]).argmax(-1)
            report["frozen_teacher_emotion_accuracy"] = float((classes == split["q"]["emotion_id"]).float().mean())
            report["emotion_measurement_note"] = "Frozen training teacher, not an independent emotion evaluator."
            reports[str(seed)][mode] = report
            if curves_path is not None:
                saved_curves.setdefault(str(seed), {})[mode] = prediction
        print(json.dumps({"stage": "development_evaluation", "seed": seed, "modes": modes}), flush=True)
    if curves_path is not None:
        save_checkpoint(curves_path, {"noise_seeds": list(seeds), "decode_steps": decode_steps, "motion": saved_curves})
    return reports


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "bundle", "weights", "checkpoint", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=("rrr", "pca"), default="rrr")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-final-curves", action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    if not 10 <= args.min_epochs <= args.max_epochs <= 40:
        raise ValueError("Formal run requires 10 <= min_epochs <= max_epochs <= 40")
    if args.eval_every != 2 or args.teacher_epochs != 5 or args.patience != 5:
        raise ValueError("Predeclared schedule: evaluation every2 epochs, teacher5 epochs, patience5")
    if min(args.batch_size, args.eval_batch_size, args.decode_steps) < 1 or args.lr <= 0:
        raise ValueError("Positive batches, decode steps and learning rate required")
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        raise FileNotFoundError("Resume requires original output directory")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    hashes = {name: sha(getattr(args, name)) for name in ("cache", "bundle", "weights", "checkpoint", "config")}
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    fitted = torch.load(args.weights, map_location="cpu", weights_only=False)
    validate_inputs(cache, bundle, fitted["states"], checkpoint, cfg)
    for name in ("checkpoint", "config", "bundle"):
        if cache["provenance"].get(name + "_sha256") != hashes[name]:
            raise ValueError(f"Cache provenance {name} mismatch")
    if fitted["provenance"].get("bundle_sha256") != hashes["bundle"]:
        raise ValueError("Basis input bundle differs")
    if cfg["data"]["emotion_classes"][0] != "neutral":
        raise ValueError("Gate requires neutral class zero")
    train_query, dev_query = cache["splits"]["train"]["q"], cache["splits"]["validation"]["q"]
    for role, query in (("train", train_query), ("validation", dev_query)):
        if len(set(query["clip_id"])) != len(query["clip_id"]):
            raise ValueError(f"Duplicate {role} query clips")
    if set(train_query["clip_id"]) & set(dev_query["clip_id"]):
        raise ValueError("Training/development query clip overlap")
    if set(train_query["sentence_id"]) & set(dev_query["sentence_id"]):
        raise ValueError("Formal checkpoint-selection development sentences overlap training")
    source_paths = [Path(__file__), Path(__file__).with_name("train_predictable_renderer.py"),
                    Path(__file__).with_name("emotion_ray_metrics.py"), Path(__file__).with_name("train_neutral_affect_pilot.py"),
                    Path(__file__).with_name("train_neutral_affect_audio_ablation.py"),
                    Path(__file__).parents[1] / "kinetalk_b0/models/neutral_affect.py",
                    Path(__file__).parents[1] / "kinetalk_b0/models/dit.py",
                    Path(__file__).parents[1] / "kinetalk_b0/models/model.py",
                    Path(__file__).parents[1] / "kinetalk_b0/models/encoders.py",
                    Path(__file__).parents[1] / "kinetalk_b0/predictable_motion.py",
                    Path(__file__).parents[1] / "kinetalk_b0/utils.py"]
    recipe = {"schema": SCHEMA, "args": {k: str(v.resolve()) if isinstance(v, Path) else v
              for k, v in vars(args).items() if k != "resume"},
              "input_sha256": hashes, "source_sha256": {str(p.resolve()): sha(p) for p in source_paths},
              "torch": str(torch.__version__), "numpy": str(np.__version__), "tf32_matmul": True,
              "noise_seeds": [42, 123, 2026], "audio_activity_gate": "1-softmax(frozen_audio_logits)[neutral]",
              "trainable": ["local_projection.weight"], "loss": "observed_flow_mse_only",
              "selection": "positive nonneutral upper full-zero gain; neutral raw mouth/upper <=3pct degradation; all-mouth correlation drop<=.01; frozen class readout drop<=1pp versus original; neutral-mouth/neutral-upper/all-mouth/nonneutral-upper velocity MSE degradation<=5pct; maximize gain among eligible checkpoints",
              "data_scope": {"train_clips": len(train_query["clip_id"]), "development_clips": len(dev_query["clip_id"]),
                             "train_sentences": len(set(train_query["sentence_id"])), "development_sentences": len(set(dev_query["sentence_id"])),
                             "current_training_sentence_disjoint": True, "pretrained_exposure_excluded": False},
              "new_test_loaded": False}
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    freeze_module(system)
    tr, dev = bundle["bundles"]["internal"], bundle["bundles"]["external_dev"]
    validation = cache["splits"]["validation"]
    state = fitted["states"]["rrr_rank8" if args.mode == "rrr" else "pca_rank8"]
    head = PredictableAudioHead(state, tr["motion_bins"], tr["weight"]).to(args.device).eval()
    freeze_module(head)
    frozen_before, head_before = frozen_hash(system), state_hash(head.state_dict())
    generator = torch.Generator().manual_seed(args.seed)
    if args.resume is None:
        save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe),
            "frozen_state_sha256": frozen_before, "head_sha256": head_before,
            "scope": "Formal-budget parameter-efficient training; no default replacement or publication claim."})
        (args.output / "source").mkdir()
        for path in source_paths:
            shutil.copyfile(path, args.output / "source" / path.name)
        original = evaluate_compact(system, None, validation, dev, device=args.device, modes=("full",), original=True,
                                    batch_size=args.eval_batch_size, decode_steps=args.decode_steps)
        save_json(args.output / "original_development.json", original)
        original_class_accuracy = mean_metric(original, "full", "frozen_teacher_emotion_accuracy")
    configure_trainable(system, seed=args.seed, projection_only=True)
    names = [name for name, parameter in system.named_parameters() if parameter.requires_grad]
    if names != ["local_projection.weight"] or system.local_projection.weight.numel() != 512:
        raise ValueError("Formal protocol requires exactly the existing512-parameter projection")
    params = list(system.local_projection.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    completed_epochs, step, best_epoch, patience_count, elapsed_before = 0, 0, None, 0, 0.
    best_score, latest_selection, best_snapshot = None, None, None
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        restore_checkpoint(resumed, recipe=recipe, system=system, head=head, optimizer=optimizer, generator=generator)
        completed_epochs, step = resumed["completed_epochs"], resumed["step"]
        best_score, best_epoch, patience_count = resumed["best_score"], resumed["best_epoch"], resumed["patience_count"]
        elapsed_before, original_class_accuracy = resumed["elapsed_seconds"], resumed["original_class_accuracy"]
        latest_selection = resumed["latest_selection"]
        best_snapshot = resumed.get("best_snapshot")
        if best_epoch is not None:
            # last.pt is the epoch transaction boundary. If a crash occurred
            # after best.pt but before last.pt, restore the committed best.
            if best_snapshot is None or best_snapshot.get("completed_epochs") != best_epoch:
                raise ValueError("Resume committed best snapshot missing or inconsistent; resume last.pt")
            if best_snapshot.get("recipe_sha256") != canonical_hash(recipe):
                raise ValueError("Resume best snapshot recipe mismatch")
            save_checkpoint(args.output / "best.pt", best_snapshot)
    features, weight = audio_features(tr), tr["weight"].float()
    with torch.no_grad():
        teacher = head.teacher(tr["motion_bins"].float().to(args.device), weight.to(args.device)).cpu()
    train_split = cache["splits"]["train"]
    batch_count = math.ceil(len(features) / args.batch_size)
    if completed_epochs > args.max_epochs or step != completed_epochs * batch_count:
        raise ValueError("Resume epoch/step boundary is inconsistent with complete training epochs")
    shape = train_split["q"]["motion"].shape[1:]
    started = time.time()

    def payload(include_best=True):
        value = {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
                "local_projection": system.local_projection.state_dict(), "head": head.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": capture_rng(generator),
                "completed_epochs": completed_epochs, "step": step, "best_score": best_score,
                "best_epoch": best_epoch, "patience_count": patience_count, "latest_selection": latest_selection,
                "elapsed_seconds": elapsed_before + time.time() - started,
                "original_class_accuracy": original_class_accuracy, "frozen_state_sha256": frozen_before,
                "head_sha256": head_before, "resume_boundary": "completed epoch; next epoch starts from restored training RNG",
                "default_enabled": False}
        if include_best:
            value["best_snapshot"] = best_snapshot
        return value

    # Initial checkpoint also permits recovering a crash before the first epoch.
    if args.resume is None:
        save_checkpoint(args.output / "last.pt", payload())
    stop_reason = "max_epochs"
    for epoch in range(completed_epochs, args.max_epochs):
        if completed_epochs >= args.min_epochs and patience_count >= args.patience:
            stop_reason = "development_patience"
            break
        order = torch.randperm(len(features), generator=generator)
        losses, teacher_counts, seen = [], 0, 0
        for batch_index, ids in enumerate(order.split(args.batch_size)):
            noise = torch.randn((len(ids), *shape), generator=generator)
            flow_time = torch.rand(len(ids), generator=generator)
            flow_time[torch.rand(len(ids), generator=generator) < .2] = 0
            probability = teacher_probability(epoch, batch_index, batch_count, args.teacher_epochs)
            choose = torch.rand(len(ids), generator=generator) < probability
            batch, w = batch_to_device(train_split, ids, args.device), weight[ids].to(args.device)
            with torch.no_grad():
                audio = head(features[ids].to(args.device), w)
                controls = torch.where(choose.to(args.device)[:, None, None], teacher[ids].to(args.device), audio)
            result = cached_flow(system, batch, controls, w, noise.to(args.device), flow_time.to(args.device), audio_gate=True)
            loss = (result["prediction"] - result["velocity_target"])[observed(batch["q"])].square().mean()
            norm = optimize(loss, optimizer, params)
            step += 1; seen += len(ids); teacher_counts += int(choose.sum()); losses.append(float(loss))
            if step % 100 == 0:
                print(json.dumps({"stage": "training", "epoch": epoch + 1, "step": step,
                    "loss": float(loss), "grad_norm": norm, "teacher_probability": probability}), flush=True)
        completed_epochs = epoch + 1
        if seen != len(features) or frozen_hash(system) != frozen_before or state_hash(head.state_dict()) != head_before:
            raise RuntimeError("Epoch coverage or frozen parameter invariant failed")
        evaluation = None
        if completed_epochs % args.eval_every == 0 or completed_epochs == args.max_epochs:
            evaluation = evaluate_compact(system, head, validation, dev, device=args.device,
                batch_size=args.eval_batch_size, decode_steps=args.decode_steps)
            latest_selection = selection_metrics(evaluation, original_class_accuracy)
            improved = latest_selection["eligible"] and (best_score is None or latest_selection["upper_full_minus_zero_r2"] > best_score)
            if improved:
                best_score, best_epoch = latest_selection["upper_full_minus_zero_r2"], completed_epochs
            # Warmup/checkpoint fitting before min_epochs cannot exhaust patience.
            if completed_epochs >= args.min_epochs:
                patience_count = 0 if improved else patience_count + 1
            save_json(args.output / f"development_epoch{completed_epochs:03d}.json", {"selection": latest_selection, "noise_reports": evaluation})
            if improved:
                best_snapshot = cpu_tree(payload(include_best=False))
                save_checkpoint(args.output / "best.pt", best_snapshot)
        record = {"epoch": completed_epochs, "step": step, "samples_seen": seen, "batches": batch_count,
                  "mean_batch_flow_loss": float(np.mean(losses)), "teacher_fraction": teacher_counts / seen,
                  "best_epoch": best_epoch, "best_score": best_score, "patience_count": patience_count,
                  "selection": latest_selection if evaluation is not None else None,
                  "elapsed_seconds": elapsed_before + time.time() - started}
        # One file per completed epoch makes resume idempotent without duplicate JSONL records.
        save_json(args.output / f"epoch{completed_epochs:03d}.json", record)
        save_checkpoint(args.output / "last.pt", payload())
        print(json.dumps(record, allow_nan=False), flush=True)
    selected_path = args.output / ("best.pt" if best_epoch is not None else "last.pt")
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    system.local_projection.load_state_dict(selected["local_projection"], strict=True)
    selected_checkpoint_sha256 = sha(selected_path)
    final = evaluate_compact(system, head, validation, dev, device=args.device,
        modes=("full", "zero", "reverse", "oracle"), batch_size=args.eval_batch_size, decode_steps=args.decode_steps,
        curves_path=args.output / "selected_development_curves.pt" if args.save_final_curves else None)
    curve_provenance = None
    if args.save_final_curves:
        curve_provenance = write_curve_provenance(args.output / "selected_development_curves.pt",
            recipe_sha256=canonical_hash(recipe), selected_checkpoint_sha256=selected_checkpoint_sha256,
            cache_sha256=hashes["cache"])
    save_json(args.output / "summary.json", {"schema": "formal_predictable_projection_result_v1",
        "recipe_sha256": canonical_hash(recipe), "completed_epochs": completed_epochs, "optimizer_steps": step,
        "stop_reason": stop_reason, "selected_checkpoint": str(selected_path.resolve()), "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "selected_development_curves_provenance": curve_provenance,
        "best_epoch": best_epoch, "best_score": best_score, "has_eligible_checkpoint": best_epoch is not None,
        "final_selection": selection_metrics(final, original_class_accuracy), "after": final,
        "frozen_unchanged": frozen_hash(system) == frozen_before, "head_unchanged": state_hash(head.state_dict()) == head_before,
        "default_replaced": False, "new_test_loaded": False,
        "status": "completed_candidate_requires_final_audit" if best_epoch is not None else "completed_no_eligible_checkpoint"})
    print("COMPLETE: formal-budget projection training; default model unchanged", flush=True)


if __name__ == "__main__":
    main()
