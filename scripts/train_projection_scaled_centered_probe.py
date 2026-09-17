"""Matched8epoch uniform versus training-RMS-weighted centered rollout MSE.

Only a fixed per-channel metric changes; both arms retain the raw output,
zero-initialized512parameter projection and frozen model/head/data/RNG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.train_formal_predictable_projection import (
    canonical_hash, capture_rng, evaluate_compact, frozen_hash, save_checkpoint, save_json,
)
from scripts.train_projection_schedule_ablation import (
    curve_binding, diagnostics, draws, read_allowlist, validate_internal_split,
)
from scripts.train_predictable_renderer import (
    PredictableAudioHead, audio_features, batch_to_device, configure_trainable,
    observed, optimize, projected_affect, sha, state_hash, validate_inputs,
)


from scripts.train_projection_centered_rollout_probe import centered_rollout_error, center_observed_error

SCHEMA = "projection_scaled_centered_probe_v1"
ARMS = ("uniform", "train_rms")
LOSS = "observed_12step_centered_rollout_error_squared_times_fixed_channel_weight_divided_by_residual_scale_squared_only"


def fit_training_channel_metric(cache):
    """Touch train tensors only; observed per-clip residual centering in float64."""
    train = cache["splits"]["train"]
    q = train["q"]
    mask = observed(q).cpu()
    target = q["motion"].detach().cpu().double()
    base = train["base"]["b0"].detach().cpu().double()
    identity = train["identity"]["baseline"].detach().cpu().double()
    if base.shape != target.shape or identity.shape != (target.shape[0], target.shape[-1]):
        raise ValueError("Cached training baseline shape differs")
    residual = center_observed_error(target - base - identity[:, None], mask)
    count = mask.sum((0, 1)).double()
    observed_channels = count > 0
    rms = (residual.square().sum((0, 1)) / count.clamp_min(1)).sqrt()
    positive = observed_channels & (rms > 0)
    if not positive.any():
        raise ValueError("No positive observed training residual RMS")
    # Statistical median: midpoint of central values for an even channel count.
    floor = torch.quantile(rms[positive], .5)
    floored = rms.clamp_min(floor)
    inverse = torch.where(observed_channels, floored.square().reciprocal(), 0)
    scale = inverse[observed_channels].mean()
    normalized = inverse / scale
    state = {"rms": rms, "floor": floor, "floored_rms": floored,
        "observed_counts": count, "observed_channels": observed_channels,
        "inverse_variance_mean_observed": scale, "train_rms_weights": normalized.float(),
        "uniform_weights": torch.ones_like(normalized, dtype=torch.float32)}
    if not torch.isfinite(normalized).all() or not torch.all(normalized[observed_channels] > 0):
        raise ValueError("Invalid fixed training metric")
    return state


def channel_metric_report(state):
    return {"schema": "train_only_centered_residual_channel_metric_v1",
        "source_role": "cache.splits.train only;19fit identities",
        "target": "center_observed(target-cached_B0-cached_identity_baseline) per clip/channel",
        "rms": "sqrt(sum observed centered residual squared / observed count) per channel, float64 CPU",
        "floor_policy": "fixed median of positive RMS among observed training channels; even count uses central midpoint",
        "weight_policy": "inverse squared floored RMS divided by observed-channel arithmetic mean; missing channels weight0; uniform arm all1",
        "computed_state": {key: value.tolist() for key, value in state.items()},
        "state_sha256": state_hash(state), "validation_values_used": False,
        "test_values_used": False, "region_specific_weights": False}


def apply_channel_weights(error, channel_weights):
    if channel_weights.shape != (error.shape[-1],) or not torch.isfinite(channel_weights).all() or (channel_weights < 0).any():
        raise ValueError("Finite nonnegative per-channel weights required")
    return error * channel_weights[None, None]


def scaled_centered_rollout_error(system, batch, controls, weight, noise, channel_weights):
    error, prediction = centered_rollout_error(system, batch, controls, weight, noise)
    return apply_channel_weights(error, channel_weights), prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "bundle", "weights", "checkpoint", "config", "output", "fit-ids", "validation-ids", "split-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seed != 46 or args.epochs != 8 or args.lr != 2e-4 or args.batch_size != 16 or args.eval_batch_size < 1:
        raise ValueError("Matched fixed protocol: seed46,8epochs,batch16,Adam lr=.0002")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    input_names = ("cache", "bundle", "weights", "checkpoint", "config", "fit_ids", "validation_ids", "split_lock")
    hashes = {name: sha(getattr(args, name)) for name in input_names}
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    fitted = torch.load(args.weights, map_location="cpu", weights_only=False)
    validate_inputs(cache, bundle, fitted["states"], checkpoint, cfg)
    fit_ids, validation_ids = read_allowlist(args.fit_ids), read_allowlist(args.validation_ids)
    lock = json.loads(args.split_lock.read_text(encoding="utf8"))
    data_scope = validate_internal_split(cache, bundle, lock, fit_ids, validation_ids)
    if len(fit_ids) != 2315 or len(validation_ids) != 405 or len(data_scope["fit_speakers"]) != 19:
        raise ValueError("Require the fixed2315fit/405heldout internal experiment")
    metric_state = fit_training_channel_metric(cache)
    metric_report = channel_metric_report(metric_state)
    selected_weights = metric_state[args.arm + "_weights"]
    selected_weight_hash = state_hash({"weights": selected_weights})
    for name in ("checkpoint", "config", "bundle"):
        if cache["provenance"].get(name + "_sha256") != hashes[name]:
            raise ValueError(f"Changed cache source{name}")
    if fitted["provenance"].get("bundle_sha256") != hashes["bundle"] or cfg["data"]["emotion_classes"][0] != "neutral":
        raise ValueError("Fixed basis or neutral index differs")
    source_paths = [Path(__file__), Path(__file__).with_name("train_projection_centered_rollout_probe.py"), Path(__file__).with_name("train_projection_schedule_ablation.py"),
        Path(__file__).with_name("train_formal_predictable_projection.py"), Path(__file__).with_name("train_predictable_renderer.py"),
        Path(__file__).with_name("emotion_ray_metrics.py"), Path(__file__).with_name("train_neutral_affect_pilot.py"),
        Path(__file__).with_name("train_neutral_affect_audio_ablation.py"),
        Path(__file__).parents[1] / "kinetalk_b0/models/neutral_affect.py", Path(__file__).parents[1] / "kinetalk_b0/models/dit.py",
        Path(__file__).parents[1] / "kinetalk_b0/models/model.py", Path(__file__).parents[1] / "kinetalk_b0/models/encoders.py",
        Path(__file__).parents[1] / "kinetalk_b0/predictable_motion.py", Path(__file__).parents[1] / "kinetalk_b0/utils.py"]
    recipe = {"schema": SCHEMA, "args": {**{key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}, "arm": args.arm},
        "input_sha256": hashes, "source_sha256": {str(path.resolve()): sha(path) for path in source_paths},
        "data_scope": data_scope, "trainable": ["local_projection.weight"], "loss": LOSS,
        "audio_activity_gate": True, "head_basis_scale_frozen": True, "torch": str(torch.__version__),
        "teacher_probability": .5, "decode_steps": 12, "flow_time_draws": "consumed and hashed identically to constant_teacher, unused in rollout loss",
        "loss_centering": "Per-clip/per-channel mean of raw prediction-target error over observed frames only; no detach; raw generated output untouched",
        "mean_anchoring_loss": False, "channel_metric": metric_report,
        "selected_weight_sha256": selected_weight_hash, "checkpoint_selection": "none; final epoch8 primary, epoch2 predeclared auxiliary only",
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "inherited_exposure_note": "This isolates new basis/head/interface fit identities, not historical B0/global training"}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe)})
    save_checkpoint(args.output / "channel_metric.pt", metric_state)
    save_json(args.output / "channel_metric.json", metric_report)
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    if system.residual_scale != .25:
        raise ValueError("Matched residual_scale must remain .25")
    del checkpoint
    configure_trainable(system, seed=args.seed, projection_only=True)
    if [name for name, p in system.named_parameters() if p.requires_grad] != ["local_projection.weight"] or system.local_projection.weight.numel() != 512:
        raise ValueError("Only the original512-parameter projection may train")
    tr, dev = bundle["bundles"]["internal"], bundle["bundles"]["external_dev"]
    head = PredictableAudioHead(fitted["states"]["rrr_rank8"], tr["motion_bins"], tr["weight"]).to(args.device).eval()
    freeze_module(head)
    frozen_before, head_before = frozen_hash(system), state_hash(head.state_dict())
    params = list(system.local_projection.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    features, weight = audio_features(tr), tr["weight"].float()
    with torch.no_grad():
        teacher = head.teacher(tr["motion_bins"].float().to(args.device), weight.to(args.device)).cpu()
    train, validation = cache["splits"]["train"], cache["splits"]["validation"]
    shape = train["q"]["motion"].shape[1:]
    channel_weights = selected_weights.to(args.device)
    batch_hash, noise_hash, choice_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    started, step, epoch_done = time.time(), 0, 0

    def payload():
        return {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
            "local_projection": system.local_projection.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": capture_rng(generator), "step": step, "completed_epochs": epoch_done,
            "frozen_state_sha256": frozen_before, "head_sha256": head_before,
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
            "default_enabled": False, "selection": "none", "channel_metric": metric_state,
            "selected_weight_sha256": selected_weight_hash}

    save_checkpoint(args.output / "last.pt", payload())
    auxiliary_binding = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(features), generator=generator)
        losses, teacher_count, source_errors = [], 0, {"teacher": [0., 0.], "audio": [0., 0.]}
        for ids in order.split(args.batch_size):
            noise, flow_time, choose = draws(generator, len(ids), shape)
            batch_hash.update(ids.numpy().tobytes()); noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
            choice_hash.update(choose.numpy().tobytes())
            use_teacher = choose < .5
            b, w = batch_to_device(train, ids, args.device), weight[ids].to(args.device)
            with torch.no_grad():
                audio = head(features[ids].to(args.device), w)
                controls = torch.where(use_teacher.to(args.device)[:, None, None], teacher[ids].to(args.device), audio)
            error, _ = scaled_centered_rollout_error(system, b, controls, w, noise.to(args.device), channel_weights)
            valid = observed(b["q"])
            loss = error[valid].mean()
            for source, chosen in (("teacher", use_teacher), ("audio", ~use_teacher)):
                mask = valid & chosen.to(args.device)[:, None, None]
                source_errors[source][0] += float(error.detach()[mask].sum())
                source_errors[source][1] += int(mask.sum())
            norm = optimize(loss, optimizer, params)
            losses.append(float(loss)); teacher_count += int(use_teacher.sum()); step += 1
            if step % 100 == 0:
                print(json.dumps({"arm": args.arm, "epoch": epoch + 1, "step": step, "loss": float(loss), "grad_norm": norm}), flush=True)
        epoch_done = epoch + 1
        if frozen_hash(system) != frozen_before or state_hash(head.state_dict()) != head_before:
            raise RuntimeError("Frozen original model/head changed")
        save_checkpoint(args.output / "last.pt", payload())
        evaluation, report_binding = None, None
        if epoch_done % 2 == 0:
            auxiliary = epoch_done == 2
            curve_path = args.output / "epoch002_diagnostic_curves.pt" if auxiliary else None
            if auxiliary:
                save_checkpoint(args.output / "epoch002_auxiliary.pt", payload())
            evaluation = evaluate_compact(system, head, validation, dev, device=args.device, batch_size=args.eval_batch_size,
                modes=("full", "zero", "reverse", "oracle") if auxiliary else ("full", "zero"), curves_path=curve_path)
            if auxiliary:
                auxiliary_binding = curve_binding(curve_path, args.output / "epoch002_auxiliary.pt", recipe, hashes["cache"])
                report_binding = auxiliary_binding
            save_json(args.output / f"development_epoch{epoch_done:03d}.json", {"diagnostics": diagnostics(evaluation),
                "noise_reports": evaluation, "curve_provenance": report_binding, "checkpoint_selection_performed": False})
        record = {"arm": args.arm, "epoch": epoch_done, "step": step, "samples_seen": len(features),
            "mean_scaled_centered_rollout_loss": float(np.mean(losses)), "teacher_fraction": teacher_count / len(features),
            "scaled_centered_rollout_mse_by_condition_source": {name: sums[0] / sums[1] if sums[1] else None for name, sums in source_errors.items()},
            "teacher_probability_last_batch": .5, "elapsed_seconds": time.time() - started,
            "diagnostics": diagnostics(evaluation) if evaluation else None,
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest()}
        save_json(args.output / f"epoch{epoch_done:03d}.json", record)
        print(json.dumps(record), flush=True)
    save_checkpoint(args.output / "final_epoch008.pt", payload())
    curves_path = args.output / "final_epoch008_curves.pt"
    final = evaluate_compact(system, head, validation, dev, device=args.device, batch_size=args.eval_batch_size,
        modes=("full", "zero", "reverse", "oracle"), curves_path=curves_path)
    binding = curve_binding(curves_path, args.output / "final_epoch008.pt", recipe, hashes["cache"])
    save_json(args.output / "summary.json", {"schema": SCHEMA, "recipe_sha256": canonical_hash(recipe), "arm": args.arm,
        "completed_epochs": epoch_done, "optimizer_steps": step, "final": final, "final_diagnostics": diagnostics(final),
        "curve_provenance": binding, "epoch002_auxiliary_curve_provenance": auxiliary_binding,
        "frozen_unchanged": frozen_hash(system) == frozen_before, "head_unchanged": state_hash(head.state_dict()) == head_before,
        "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
        "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
        "checkpoint_selection_performed": False, "outer280_loaded": False, "new_identity439_loaded": False,
        "test_loaded": False, "default_replaced": False, "generated_output_centered": False,
        "channel_metric_sha256": sha(args.output / "channel_metric.pt"), "selected_weight_sha256": selected_weight_hash})
    print("COMPLETE fixed8epoch matched fixed-channel-metric centered-rollout probe", flush=True)


if __name__ == "__main__":
    main()
