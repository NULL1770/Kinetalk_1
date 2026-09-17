"""Matched audio/zero-local training with output displacement and std losses.

Conventional diagnostic, not a paper reproduction or a stochasticity guarantee.
The previous direct-audio training implementation is imported without changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_audio_conditioned_flow_probe import load_source, NOISE_SEEDS, source_paths
from scripts.train_direct_audio_dynamics import (
    DirectAudioEncoder, MODES, configure, direct_affect, evaluate,
    feature_statistics, protected_hash, restore_direct,
)
from scripts.train_formal_predictable_projection import canonical_hash, capture_rng, save_checkpoint, save_json
from scripts.train_predictable_renderer import audio_features, batch_to_device, observed, optimize, sha, state_hash
from scripts.train_projection_schedule_ablation import draws, curve_binding

SCHEMA = "output_motion_dynamics_v1"
ARMS = ("audio", "zero")
GROUPS = {"brows": list(range(41, 46)), "eyes_expression": [5, 6, 12, 13], "mouth": list(range(14, 41))}
STD_GROUPS = ("brows", "eyes_expression")
LOSS_WEIGHTS = {"displacement": 1., "std": 1.}


def motion_statistics(motion, mask):
    """Observed per-clip std (ddof0), with finite constant-trajectory gradients."""
    if motion.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Motion and Boolean observation mask must match")
    clean = torch.where(mask, motion, 0.)
    count = mask.sum(1)
    mean = clean.sum(1) / count.clamp_min(1)
    centered = torch.where(mask, clean - mean[:, None], 0.)
    # sqrt(sum(x*x)) gives undefined backward at exactly zero; vector_norm
    # implements the finite zero subgradient without altering the std value.
    std = torch.linalg.vector_norm(centered, dim=1) / count.clamp_min(1).sqrt()
    return clean, std, count


@torch.no_grad()
def fit_output_scales(train):
    """Only supplied fit targets enter the two native-motion scales."""
    q = train["q"]
    mask = observed(q)
    clean, std, count = motion_statistics(q["motion"].double(), mask)
    pairs = mask[:, 1:] & mask[:, :-1]
    delta = clean[:, 1:] - clean[:, :-1]
    displacement = (torch.where(pairs, delta.square(), 0.).sum((0, 1)) /
                    pairs.sum((0, 1)).clamp_min(1)).sqrt().clamp_min(.005)
    std_scale = ((std.square() * count).sum(0) / count.sum(0).clamp_min(1)).sqrt().clamp_min(.02)
    for name, channels in GROUPS.items():
        if not pairs[..., channels].any():
            raise ValueError("No observed fit displacement pairs for " + name)
    return {"displacement": displacement.float(), "std": std_scale.float()}


def group_mean(values, mask, channels):
    valid = mask[..., channels]
    if not valid.any():
        raise ValueError("An output-supervision group has no valid observations")
    return values[..., channels][valid].mean()


def generated_output_losses(prediction, batch, scales):
    q = batch["q"]
    mask = observed(q)
    pred, pred_std, count = motion_statistics(prediction, mask)
    target, target_std, _ = motion_statistics(q["motion"], mask)
    pairs = mask[:, 1:] & mask[:, :-1]
    delta_error = (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
    scaled_delta = (delta_error / scales["displacement"]).square()
    displacement = torch.stack([group_mean(scaled_delta, pairs, cc) for cc in GROUPS.values()]).mean()
    std_error = ((pred_std - target_std) / scales["std"]).square()
    std = torch.stack([group_mean(std_error, count >= 2, GROUPS[name]) for name in STD_GROUPS]).mean()
    return {"displacement": displacement, "std": std}


def training_loss(system, encoder, batch, features, weight, scales, noise, flow_time, arm):
    if arm not in ARMS:
        raise ValueError("Unknown output-motion arm")
    q = batch["q"]
    # Build the same encoder in both arms; zero condition has no path to its
    # parameters, and Adam therefore leaves it at the identical initial state.
    local = encoder(features, weight)
    affect = direct_affect(system, batch, local, weight, "zero" if arm == "zero" else "full")
    flow = system.flow(q["motion"], q["content"], q["valid"], batch["identity"], affect,
                       noise=noise, time=flow_time, base=batch["base"])
    losses = {"flow": (flow["prediction"] - flow["velocity_target"])[observed(q)].square().mean()}
    prediction = system.generate(q["content"], q["valid"], batch["identity"], affect,
                                 initial_noise=noise, steps=12, base=batch["base"])["motion"]
    losses.update(generated_output_losses(prediction, batch, scales))
    return losses["flow"] + sum(LOSS_WEIGHTS[k] * losses[k] for k in LOSS_WEIGHTS), losses


def load_initial_source(direct_run, device):
    """Rebuild the fit-only direct initialization and restore only epoch0."""
    direct_run = Path(direct_run).resolve()
    provenance_path, checkpoint_path = direct_run / "provenance.json", direct_run / "epoch000.pt"
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    recipe = provenance["recipe"]
    if (recipe.get("schema") != "direct_audio_dynamics_v1"
            or provenance.get("recipe_sha256") != canonical_hash(recipe)):
        raise ValueError("Direct source recipe differs")
    inventory = json.loads((direct_run / "output_hashes.json").read_text(encoding="utf8"))
    if inventory.get("epoch000.pt") != sha(checkpoint_path) or inventory.get("provenance.json") != sha(provenance_path):
        raise ValueError("Direct source inventory binding differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("recipe") != recipe or checkpoint.get("completed_epochs") != 0
            or checkpoint.get("step") != 0):
        raise ValueError("Require direct epoch000, never its trained result")
    loaded = load_source(recipe["source_run"], device)
    if any(loaded[key] != recipe[key] for key in ("input_sha256", "source_adapter_sha256", "data_scope")):
        raise ValueError("Inherited source input binding differs")
    system, tr = loaded["system"], loaded["bundle"]["bundles"]["internal"]
    if state_hash(system.state_dict()) != recipe["initial_system_sha256"]:
        raise ValueError("Initial source system differs")
    mean, std = feature_statistics(audio_features(tr), tr["weight"].float())
    encoder = DirectAudioEncoder(mean, std, system.local_projection.out_features).to(device).eval()
    if state_hash(encoder.state_dict()) != recipe["initial_encoder_sha256"]:
        raise ValueError("Direct encoder initialization/RNG differs")
    for key, value in (("feature_mean", mean), ("feature_std", std)):
        torch.testing.assert_close(checkpoint["encoder"][key], value, rtol=0, atol=0)
    restore_direct(system, encoder, checkpoint)
    if (state_hash(encoder.state_dict()) != recipe["initial_encoder_sha256"]
            or state_hash(system.state_dict()) != recipe["initial_system_sha256"]):
        raise ValueError("Restored direct epoch0 differs from startup")
    return loaded, encoder, {"run": str(direct_run), "recipe": recipe,
        "provenance_sha256": sha(provenance_path), "checkpoint_sha256": sha(checkpoint_path),
        "inventory_sha256": sha(direct_run / "output_hashes.json")}


def restore_output(system, encoder, payload):
    if payload.get("schema") != SCHEMA or payload.get("recipe_sha256") != canonical_hash(payload["recipe"]):
        raise ValueError("Output-motion checkpoint schema/recipe differs")
    if protected_hash(system) != payload.get("protected_sha256"):
        raise ValueError("Protected source differs")
    for key in ("renderer", "encoder"):
        if state_hash(payload[key]) != payload.get(key + "_sha256"):
            raise ValueError("Invalid saved " + key)
    system.renderer.load_state_dict(payload["renderer"], strict=True)
    encoder.load_state_dict(payload["encoder"], strict=True)
    system.eval(); encoder.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh isolated output required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(46)
    loaded, encoder, direct_source = load_initial_source(args.direct_source, args.device)
    system, cache, bundle = loaded["system"], loaded["cache"], loaded["bundle"]
    train, validation = cache["splits"]["train"], cache["splits"]["validation"]
    tr, dev = bundle["bundles"]["internal"], bundle["bundles"]["external_dev"]
    features, weight = audio_features(tr), tr["weight"].float()
    renderer_params = configure(system)
    parameters = renderer_params + list(encoder.parameters())
    protected, encoder_initial = protected_hash(system), state_hash(encoder.state_dict())
    scales = fit_output_scales(train)
    root = Path(__file__).resolve().parents[1]
    paths = list(dict.fromkeys(source_paths() + [Path(__file__),
        root / "scripts/train_direct_audio_dynamics.py", root / "docs/OUTPUT_MOTION_DYNAMICS_PROTOCOL.md",
        root / "tests/test_output_motion_dynamics.py"]))
    recipe = {"schema": SCHEMA, "arm": args.arm, "direct_source": direct_source,
        "input_sha256": loaded["input_sha256"], "data_scope": loaded["data_scope"],
        "source_adapter_sha256": loaded["source_adapter_sha256"],
        "source_sha256": {str(path.resolve()): sha(path) for path in paths},
        "seed": 46, "epochs": 8, "batch_size": 16, "optimizer": "Adam", "renderer_lr": 1e-5,
        "encoder_lr": 1e-4, "clip_grad_norm": 1., "loss_weights": LOSS_WEIGHTS,
        "groups": GROUPS, "std_groups": list(STD_GROUPS), "displacement_floor": .005,
        "std_floor": .02, "std_ddof": 0, "velocity_fps_conversion": False,
        "initial_system_sha256": state_hash(system.state_dict()), "initial_encoder_sha256": encoder_initial,
        "protected_sha256": protected, "motion_scales_sha256": state_hash(scales),
        "decode_steps": 12, "eval_noise_seeds": list(NOISE_SEEDS), "eval_modes": list(MODES),
        "trainable_parameters": sum(x.numel() for x in parameters), "all_modules_eval_mode": True,
        "time_sampling": "uniform time with independent20% t=0; historical draws",
        "training_condition": "zero-local" if args.arm == "zero" else "direct-audio-local",
        "default_replaced": False, "test_loaded": False, "checkpoint_selection_performed": False}
    args.output.mkdir(parents=True)
    for path in paths:
        dest = args.output / "source" / path.resolve().relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe)})
    optimizer = torch.optim.Adam([{"params": renderer_params, "lr": 1e-5},
                                  {"params": encoder.parameters(), "lr": 1e-4}])
    generator = torch.Generator().manual_seed(46)
    batch_hash, noise_hash, choice_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    epoch_done, step, started = 0, 0, time.time()

    def hashes():
        return {"minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
                "choice_draw_sha256": choice_hash.hexdigest()}

    def payload():
        if protected_hash(system) != protected:
            raise RuntimeError("Protected source changed")
        if args.arm == "zero" and state_hash(encoder.state_dict()) != encoder_initial:
            raise RuntimeError("Zero-local encoder unexpectedly changed")
        if any(m.training for m in system.modules()) or any(m.training for m in encoder.modules()):
            raise RuntimeError("All modules must remain eval")
        return {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
            "renderer": system.renderer.state_dict(), "encoder": encoder.state_dict(),
            "renderer_sha256": state_hash(system.renderer.state_dict()), "encoder_sha256": state_hash(encoder.state_dict()),
            "protected_sha256": protected, "motion_scales": scales, "optimizer": optimizer.state_dict(),
            "rng": capture_rng(generator), "completed_epochs": epoch_done, "step": step, **hashes()}

    save_checkpoint(args.output / "epoch000.pt", payload())
    initial_path = args.output / "epoch000_curves.pt"
    initial = evaluate(system, encoder, validation, dev, args.device, initial_path, seeds=(42,))
    source_curves_path = args.direct_source / "epoch000_curves.pt"
    inventory = json.loads((args.direct_source / "output_hashes.json").read_text(encoding="utf8"))
    if inventory.get("epoch000_curves.pt") != sha(source_curves_path):
        raise ValueError("Direct source initial curves changed")
    source_curves = torch.load(source_curves_path, map_location="cpu", weights_only=False)
    initial_curves = torch.load(initial_path, map_location="cpu", weights_only=False)
    for mode in MODES:
        torch.testing.assert_close(initial_curves["motion"]["42"][mode],
                                   source_curves["motion"]["42"]["zero"], rtol=0, atol=0)
    del source_curves, initial_curves
    initial_binding = curve_binding(initial_path, args.output / "epoch000.pt", recipe, loaded["input_sha256"]["cache"])
    save_json(args.output / "epoch000_evaluation.json", {"reports": initial,
        "curve_provenance": initial_binding, "source_zero_max_abs_error": 0.})
    gpu_scales = {key: value.to(args.device) for key, value in scales.items()}
    for epoch in range(1, 9):
        totals, seen = {}, 0
        for ids in torch.randperm(len(weight), generator=generator).split(16):
            noise, flow_time, choice = draws(generator, len(ids), train["q"]["motion"].shape[1:])
            batch_hash.update(ids.numpy().tobytes())
            noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
            choice_hash.update(choice.numpy().tobytes())
            batch = batch_to_device(train, ids, args.device)
            total, losses = training_loss(system, encoder, batch, features[ids].to(args.device), weight[ids].to(args.device),
                gpu_scales, noise.to(args.device), flow_time.to(args.device), args.arm)
            norm = optimize(total, optimizer, parameters)
            for key, value in losses.items(): totals[key] = totals.get(key, 0.) + float(value.detach()) * len(ids)
            seen += len(ids); step += 1
            if step % 100 == 0:
                print(json.dumps({"arm": args.arm, "step": step, "losses": {k: float(v.detach()) for k, v in losses.items()},
                                  "grad_norm": norm}), flush=True)
        epoch_done = epoch
        state = payload()
        save_checkpoint(args.output / f"epoch{epoch:03d}.pt", state)
        save_checkpoint(args.output / "last.pt", state)
        row = {"epoch": epoch, "step": step, "losses": {k: v / seen for k, v in totals.items()},
               "elapsed_seconds": time.time() - started, **hashes()}
        save_json(args.output / f"epoch{epoch:03d}.json", row); print(json.dumps(row), flush=True)
    checkpoint = args.output / "final_epoch008.pt"
    save_checkpoint(checkpoint, payload())
    curve_path = args.output / "final_epoch008_curves.pt"
    final = evaluate(system, encoder, validation, dev, args.device, curve_path)
    binding = curve_binding(curve_path, checkpoint, recipe, loaded["input_sha256"]["cache"])
    save_json(args.output / "summary.json", {"schema": SCHEMA, "recipe_sha256": canonical_hash(recipe), "arm": args.arm,
        "completed_epochs": epoch_done, "optimizer_steps": step, "epoch000": initial,
        "epoch000_curve_provenance": initial_binding, "final": final, "curve_provenance": binding,
        "protected_unchanged": protected_hash(system) == protected, "encoder_unchanged": state_hash(encoder.state_dict()) == encoder_initial,
        "elapsed_seconds": time.time() - started, **hashes(),
        "test_loaded": False, "default_replaced": False, "checkpoint_selection_performed": False})
    save_json(args.output / "output_hashes.json", {str(x.relative_to(args.output)): sha(x)
        for x in args.output.rglob("*") if x.is_file() and x.name != "output_hashes.json"})
    print("OUTPUT_MOTION_DYNAMICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
