"""Fixed train/development subset diagnosis of audio/text fit accessibility.

Regenerates epoch000 and epoch008 at seed42 on predeclared random128 subsets.
Only predicted conditions drive generation; intensity targets score outputs.
This is a single-seed diagnostic, not model selection or distribution evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.label_guided_intensity import regional_intensity
from scripts import train_audio_text_dynamics as runner
from scripts.train_formal_predictable_projection import canonical_hash, save_json
from scripts.train_predictable_renderer import basic_metrics, batch_to_device, observed, sha, state_hash

SCHEMA = "audio_text_fit_inspection_v1"
SUBSET_SEED = 20260917
NOISE_SEED = 42
SUBSET_SIZE = 128


def fixed_indices(count):
    if count < 1: raise ValueError("Cannot inspect an empty split")
    return torch.randperm(count, generator=torch.Generator().manual_seed(SUBSET_SEED))[:min(count, SUBSET_SIZE)]


def _scalar_scores(prediction, target):
    """Metrics on already selected observed scalar values, in float64."""
    p, y = prediction.double(), target.double()
    error = (p - y).square().mean()
    pc, yc = p - p.mean(), y - y.mean()
    vp, vy = pc.square().mean(), yc.square().mean()
    energy = y.square().mean()
    return {"observed_frames": p.numel(), "mse": float(error),
        "r2_against_target_mean": float(1 - error / vy) if vy > 1e-24 else None,
        "r2_against_zero": float(1 - error / energy) if energy > 1e-24 else None,
        "correlation": float(((pc * yc).mean() / (vp * vy).sqrt()).clamp(-1, 1)) if vp > 1e-24 and vy > 1e-24 else None,
        "prediction_mean": float(p.mean()), "target_mean": float(y.mean()),
        "prediction_variance_ddof0": float(vp), "target_variance_ddof0": float(vy),
        "prediction_variance_ratio": float(vp / vy) if vy > 1e-24 else None,
        "prediction_rms": float(p.square().mean().sqrt()), "target_rms": float(energy.sqrt())}


def intensity_scores(prediction, target, valid):
    """Separate pooled DC/semantic fit from within-clip dynamic fit."""
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 1
            or valid.shape != target.shape or valid.dtype != torch.bool or not valid.any()):
        raise ValueError("Intensity tensors require matching [B,T,1] values and nonempty Boolean mask")
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("Nonfinite observed intensity")
    p, y = torch.where(valid, prediction.double(), 0.), torch.where(valid, target.double(), 0.)
    count = valid.sum(1, keepdim=True)
    pc = torch.where(valid, p - p.sum(1, keepdim=True) / count.clamp_min(1), 0.)
    yc = torch.where(valid, y - y.sum(1, keepdim=True) / count.clamp_min(1), 0.)
    # Single observed frames have no temporal evidence; exclude them from the
    # centered readout while retaining them in the raw report.
    temporal_valid = valid & (count >= 2)
    return {"raw": _scalar_scores(p[valid], y[valid]),
        "clip_centered": _scalar_scores(pc[temporal_valid], yc[temporal_valid]) if temporal_valid.any() else None,
        "valid_clips": int(valid.any(1).sum()), "temporal_clips": int((count >= 2).sum()),
        "definitions": "raw Pearson/R2 center across all observed frames; clip_centered first removes each clip observed mean; ddof=0"}


def score_improvement(candidate, baseline):
    result = {}
    for kind in ("raw", "clip_centered"):
        a, b = candidate[kind], baseline[kind]
        result[kind] = None if a is None or b is None else {
            "mse_improvement": b["mse"] - a["mse"],
            "relative_mse_change": a["mse"] / b["mse"] - 1 if b["mse"] > 0 else None}
    return result


def load_bound_run(run, device):
    run = Path(run).resolve()
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf8"))
    recipe = provenance["recipe"]
    if (recipe.get("schema") != runner.SCHEMA or recipe.get("arm") not in ("text", "no_text")
            or recipe.get("seed") != 46 or recipe.get("epochs") != 8 or recipe.get("smoke_steps") != 0
            or recipe.get("teacher_intensity_probability") != 0. or recipe.get("decode_steps") != 12
            or provenance.get("recipe_sha256") != canonical_hash(recipe)):
        raise ValueError("Require the bound fixed8epoch predicted-condition audio/text run")
    root = Path(runner.__file__).resolve().parents[1]
    for filename, digest in recipe["source_sha256"].items():
        source = Path(filename).resolve()
        if (not source.is_relative_to(root) or sha(source) != digest
                or sha(run / "source" / source.relative_to(root)) != digest):
            raise ValueError("Executed/saved training source changed: " + filename)
    paths = recipe["derived_paths"]
    if set(paths) != {"audio", "text", "targets"}: raise ValueError("Derived cache roles differ")
    for key, path in paths.items():
        if sha(path) != recipe["derived_sha256"][key]: raise ValueError("Derived cache hash changed: " + key)
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(46)
    loaded = runner.load_source(recipe["source_run"], device)
    for key in ("input_sha256", "source_adapter_sha256", "data_scope"):
        if loaded[key] != recipe[key]: raise ValueError("Source binding changed: " + key)
    derived = {key: torch.load(path, map_location="cpu", weights_only=False, mmap=True) for key, path in paths.items()}
    audio, text, targets = [derived[key] for key in ("audio", "text", "targets")]
    runner.validate_inputs(loaded["cache"], audio, text, targets, recipe["input_sha256"]["cache"])
    encoder = runner.make_encoder(audio, text, targets, device)
    system = loaded["system"]; runner.configure(system)
    if (state_hash(system.state_dict()) != recipe["initial_system_sha256"]
            or state_hash(encoder.state_dict()) != recipe["initial_encoder_sha256"]):
        raise ValueError("Fit-only initialization reconstruction differs")
    checkpoints, hashes = {}, {"provenance.json": sha(run / "provenance.json")}
    steps_per_epoch = (len(loaded["cache"]["splits"]["train"]["q"]["valid"]) + 15) // 16
    for epoch in (0, 8):
        path = run / f"epoch{epoch:03d}.pt"
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if (cp.get("recipe") != recipe or cp.get("completed_epochs") != epoch
                or cp.get("step") != epoch * steps_per_epoch):
            raise ValueError("Checkpoint is not the fixed requested epoch")
        runner.restore(system, encoder, cp)
        if epoch == 0 and (state_hash(system.state_dict()) != recipe["initial_system_sha256"]
                or state_hash(encoder.state_dict()) != recipe["initial_encoder_sha256"]):
            raise ValueError("Epoch000 differs from exact rebuilt initialization")
        checkpoints[epoch] = cp; hashes[path.name] = sha(path)
    inventory_path = run / "output_hashes.json"
    inventory_verified = False
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf8"))
        if any(inventory.get(name) != digest for name, digest in hashes.items()):
            raise ValueError("Run inventory does not bind required inspection inputs")
        hashes["output_hashes.json"] = sha(inventory_path); inventory_verified = True
    return run, recipe, loaded, derived, system, encoder, checkpoints, hashes, inventory_verified


@torch.inference_mode()
def inspect_split(system, encoder, checkpoints, recipe, split, audio, text, targets, scales, fit_mean, device):
    ids = fixed_indices(len(split["q"]["valid"]))
    # Match the original evaluation's source of per-clip noise exactly.
    full_noise = torch.randn(split["q"]["motion"].shape, generator=torch.Generator().manual_seed(NOISE_SEED))
    subset = batch_to_device(split, ids, "cpu")
    truth = targets["intensity"][ids].float(); truth_valid = targets["valid"][ids]
    scores = {"fit_constant": intensity_scores(torch.full_like(truth, fit_mean), truth, truth_valid)}
    motion_reports = {}; initial_local_max = None
    for epoch in (0, 8):
        runner.restore(system, encoder, checkpoints[epoch])
        motions, predicted, output_values, output_masks = [], [], [], []
        local_max = 0.
        for selected in ids.split(32):
            batch = batch_to_device(split, selected, device)
            data = runner.slice_inputs(audio, text, targets, selected, device)
            affect, encoded = runner.build_affect(encoder, batch, data, recipe["arm"], "full")
            if not torch.equal(encoded["predicted_intensity"], encoded["driving_intensity"]):
                raise ValueError("Inspection must use predicted intensity, without oracle/static interventions")
            local_max = max(local_max, float(affect["local"].abs().max()))
            q = batch["q"]
            result = system.generate(q["content"], q["valid"], batch["identity"], affect,
                initial_noise=full_noise[selected].to(device), steps=12, base=batch["base"])["motion"]
            intensity, intensity_valid = regional_intensity(result, observed(q) & data["anchor_valid"][:, None],
                                                            data["anchors"], scales.to(device))
            motions.append(result.cpu()); predicted.append(encoded["predicted_intensity"].cpu())
            output_values.append(intensity.cpu()); output_masks.append(intensity_valid.cpu())
        if epoch == 0:
            initial_local_max = local_max
            if initial_local_max != 0.: raise ValueError("Epoch000 local output must be zero")
        output_valid = torch.cat(output_masks)
        if not torch.equal(output_valid, truth_valid): raise ValueError("Generated/target intensity observation masks differ")
        name = f"epoch{epoch:03d}"
        scores[name + "/predicted"] = intensity_scores(torch.cat(predicted), truth, truth_valid)
        scores[name + "/generated_motion"] = intensity_scores(torch.cat(output_values), truth, truth_valid)
        motion_reports[name] = basic_metrics(torch.cat(motions), subset)
        print(json.dumps({"stage": "fit_inspection", "epoch": epoch, "clips": len(ids),
                          "predicted_intensity": scores[name + "/predicted"]}), flush=True)
    comparisons = {"predicted_final_vs_initial": score_improvement(scores["epoch008/predicted"], scores["epoch000/predicted"]),
        "predicted_final_vs_fit_constant": score_improvement(scores["epoch008/predicted"], scores["fit_constant"]),
        "generated_final_vs_initial": score_improvement(scores["epoch008/generated_motion"], scores["epoch000/generated_motion"]),
        "generated_final_vs_fit_constant": score_improvement(scores["epoch008/generated_motion"], scores["fit_constant"])}
    return {"indices": ids.tolist(), "clip_id": subset["q"]["clip_id"], "sentence_id": subset["q"]["sentence_id"],
        "clips": len(ids), "available_clips": len(split["q"]["valid"]), "observed_frames": int(subset["q"]["valid"].sum()),
        "epoch000_local_max_abs": initial_local_max, "motion": motion_reports, "intensity": scores,
        "intensity_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Fresh diagnostic output required")
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Write diagnostic outside immutable training run")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    run, recipe, loaded, derived, system, encoder, checkpoints, hashes, inventory_verified = load_bound_run(args.run, args.device)
    targets = derived["targets"]
    train_targets = targets["splits"]["train"]
    fit_mean = float(train_targets["intensity"][train_targets["valid"]].mean())
    if not np.isfinite(fit_mean): raise ValueError("Invalid fit-only constant intensity")
    result = {}
    for role in ("train", "validation"):
        result[role] = inspect_split(system, encoder, checkpoints, recipe, loaded["cache"]["splits"][role],
            derived["audio"]["splits"][role], derived["text"]["splits"][role], targets["splits"][role],
            targets["scales"], fit_mean, args.device)
    for name, digest in hashes.items():
        if sha(run / name) != digest: raise RuntimeError("Immutable run inputs changed during inspection")
    report = {"schema": SCHEMA, "run": str(run), "arm": recipe["arm"], "recipe_sha256": canonical_hash(recipe),
        "run_input_sha256": hashes, "final_inventory_verified": inventory_verified,
        "derived_sha256": recipe["derived_sha256"], "script_sha256": sha(__file__),
        "selection": {"seed": SUBSET_SEED, "maximum_per_split": SUBSET_SIZE,
                      "rule": "independent randperm from fixed seed separately in each original ordered split; no score selection"},
        "noise_seed": NOISE_SEED, "noise_rule": "draw complete original split shape, then select fixed indices", "decode_steps": 12,
        "fit_constant_intensity": fit_mean, "splits": result,
        "scope": {"target_conditioning": False, "test_loaded": False, "weights_written": False,
                  "default_replaced": False, "checkpoint_selection_performed": False,
                  "interpretation": "Single-seed fit accessibility diagnosis. Poor train fit indicates optimization/capacity failure; a train/dev gap alone does not establish identity generalization."}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print(json.dumps({"complete": True, "output": str(args.output.resolve()), "arm": recipe["arm"]}), flush=True)


if __name__ == "__main__":
    main()
