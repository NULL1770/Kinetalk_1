"""Fixed-budget centered-error rollout probe; raw generated motion is retained.

Only the final loss changes from the preceding constant-teacher rollout:
remove each clip/channel's observed temporal error mean before one MSE.
No output centering, mean anchoring, extra loss, or region-specific model.
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


SCHEMA = "projection_centered_rollout_probe_v1"
ARM = "constant_teacher_centered_rollout"
LOSS = "observed_12step_rollout_temporally_centered_error_mse_divided_by_residual_scale_squared_only"


def center_observed_error(error, observation_mask):
    """Center each channel over its actual observed frames, preserving grads."""
    if error.ndim != 3 or observation_mask.shape != error.shape or observation_mask.dtype != torch.bool:
        raise ValueError("Expected error and Boolean observation mask with equal[B,T,C]shape")
    if not torch.isfinite(error[observation_mask]).all():
        raise ValueError("Nonfinite observed rollout error")
    clean = torch.where(observation_mask, error, 0)
    count = observation_mask.sum(1, keepdim=True).clamp_min(1)
    mean = clean.sum(1, keepdim=True) / count
    return torch.where(observation_mask, clean - mean, 0)


def centered_rollout_error(system, batch, controls, weight, noise):
    """Raw12-step output returned unchanged; centering applies only to loss."""
    affect = projected_affect(system, batch, controls, weight, audio_gate=True)
    q = batch["q"]
    prediction = system.generate(q["content"], q["valid"], batch["identity"], affect,
        initial_noise=noise, steps=12, base=batch["base"])["motion"]
    centered = center_observed_error(prediction - q["motion"], observed(q))
    return (centered / system.residual_scale).square(), prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "bundle", "weights", "checkpoint", "config", "output", "fit-ids", "validation-ids", "split-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seed != 46 or args.epochs != 18 or args.lr != 2e-4 or args.batch_size != 16 or args.eval_batch_size < 1:
        raise ValueError("Matched fixed protocol: seed46,18epochs,batch16,Adam lr=.0002")
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
    for name in ("checkpoint", "config", "bundle"):
        if cache["provenance"].get(name + "_sha256") != hashes[name]:
            raise ValueError(f"Changed cache source{name}")
    if fitted["provenance"].get("bundle_sha256") != hashes["bundle"] or cfg["data"]["emotion_classes"][0] != "neutral":
        raise ValueError("Fixed basis or neutral index differs")
    source_paths = [Path(__file__), Path(__file__).with_name("train_projection_schedule_ablation.py"),
        Path(__file__).with_name("train_formal_predictable_projection.py"), Path(__file__).with_name("train_predictable_renderer.py"),
        Path(__file__).with_name("emotion_ray_metrics.py"), Path(__file__).with_name("train_neutral_affect_pilot.py"),
        Path(__file__).with_name("train_neutral_affect_audio_ablation.py"),
        Path(__file__).parents[1] / "kinetalk_b0/models/neutral_affect.py", Path(__file__).parents[1] / "kinetalk_b0/models/dit.py",
        Path(__file__).parents[1] / "kinetalk_b0/models/model.py", Path(__file__).parents[1] / "kinetalk_b0/models/encoders.py",
        Path(__file__).parents[1] / "kinetalk_b0/predictable_motion.py", Path(__file__).parents[1] / "kinetalk_b0/utils.py"]
    recipe = {"schema": SCHEMA, "args": {**{key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}, "arm": ARM},
        "input_sha256": hashes, "source_sha256": {str(path.resolve()): sha(path) for path in source_paths},
        "data_scope": data_scope, "trainable": ["local_projection.weight"], "loss": LOSS,
        "audio_activity_gate": True, "head_basis_scale_frozen": True, "torch": str(torch.__version__),
        "teacher_probability": .5, "decode_steps": 12, "flow_time_draws": "consumed and hashed identically to constant_teacher, unused in rollout loss",
        "loss_centering": "Per-clip/per-channel mean of raw prediction-target error over observed frames only; no detach; raw generated output untouched",
        "mean_anchoring_loss": False, "checkpoint_selection": "none; final epoch18 primary, epoch2 predeclared auxiliary only",
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "inherited_exposure_note": "This isolates new basis/head/interface fit identities, not historical B0/global training"}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe)})
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
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
    batch_hash, noise_hash, choice_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    started, step, epoch_done = time.time(), 0, 0

    def payload():
        return {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
            "local_projection": system.local_projection.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": capture_rng(generator), "step": step, "completed_epochs": epoch_done,
            "frozen_state_sha256": frozen_before, "head_sha256": head_before,
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
            "default_enabled": False, "selection": "none"}

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
            error, _ = centered_rollout_error(system, b, controls, w, noise.to(args.device))
            valid = observed(b["q"])
            loss = error[valid].mean()
            for source, chosen in (("teacher", use_teacher), ("audio", ~use_teacher)):
                mask = valid & chosen.to(args.device)[:, None, None]
                source_errors[source][0] += float(error.detach()[mask].sum())
                source_errors[source][1] += int(mask.sum())
            norm = optimize(loss, optimizer, params)
            losses.append(float(loss)); teacher_count += int(use_teacher.sum()); step += 1
            if step % 100 == 0:
                print(json.dumps({"arm": ARM, "epoch": epoch + 1, "step": step, "loss": float(loss), "grad_norm": norm}), flush=True)
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
        record = {"arm": ARM, "epoch": epoch_done, "step": step, "samples_seen": len(features),
            "mean_centered_rollout_loss": float(np.mean(losses)), "teacher_fraction": teacher_count / len(features),
            "centered_rollout_mse_by_condition_source": {name: sums[0] / sums[1] if sums[1] else None for name, sums in source_errors.items()},
            "teacher_probability_last_batch": .5, "elapsed_seconds": time.time() - started,
            "diagnostics": diagnostics(evaluation) if evaluation else None,
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest()}
        save_json(args.output / f"epoch{epoch_done:03d}.json", record)
        print(json.dumps(record), flush=True)
    save_checkpoint(args.output / "final_epoch018.pt", payload())
    curves_path = args.output / "final_epoch018_curves.pt"
    final = evaluate_compact(system, head, validation, dev, device=args.device, batch_size=args.eval_batch_size,
        modes=("full", "zero", "reverse", "oracle"), curves_path=curves_path)
    binding = curve_binding(curves_path, args.output / "final_epoch018.pt", recipe, hashes["cache"])
    save_json(args.output / "summary.json", {"schema": SCHEMA, "recipe_sha256": canonical_hash(recipe), "arm": ARM,
        "completed_epochs": epoch_done, "optimizer_steps": step, "final": final, "final_diagnostics": diagnostics(final),
        "curve_provenance": binding, "epoch002_auxiliary_curve_provenance": auxiliary_binding,
        "frozen_unchanged": frozen_hash(system) == frozen_before, "head_unchanged": state_hash(head.state_dict()) == head_before,
        "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
        "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
        "checkpoint_selection_performed": False, "outer280_loaded": False, "new_identity439_loaded": False,
        "test_loaded": False, "default_replaced": False, "generated_output_centered": False})
    print("COMPLETE fixed18epoch constant-teacher centered-error rollout probe", flush=True)


if __name__ == "__main__":
    main()
