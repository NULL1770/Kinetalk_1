"""Paired audio-local/zero-local adaptation of one existing DiT output matrix."""
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.train_formal_predictable_projection import (
    canonical_hash, capture_rng, evaluate_compact, frozen_hash,
    save_checkpoint, save_json,
)
from scripts.train_projection_schedule_ablation import (
    curve_binding, diagnostics, draws, read_allowlist, validate_internal_split,
)
from scripts.train_predictable_renderer import (
    PredictableAudioHead, audio_features, batch_to_device, cached_flow,
    observed, optimize, sha, state_hash, validate_inputs,
)

SCHEMA = "audio_conditioned_flow_probe_v1"
ARMS = ("audio_local", "zero_local")
NOISE_SEEDS = (42, 123, 2026, 7, 19, 73, 211, 997)
INPUT_NAMES = ("cache", "bundle", "weights", "checkpoint", "config", "fit_ids",
               "validation_ids", "split_lock")


def configure_adaptation(system, *, expected_dim=None):
    """Freeze everything, without resetting any weights; open one existing matrix."""
    before = state_hash(system.state_dict())
    freeze_module(system)
    system.eval()
    index = len(system.renderer.blocks) - 1
    if index < 0:
        raise ValueError("Renderer must have a final DiT block")
    name = f"renderer.blocks.{index}.cross_attention.out_proj.weight"
    parameter = dict(system.named_parameters())[name]
    if parameter.ndim != 2 or parameter.shape[0] != parameter.shape[1]:
        raise ValueError("Cross-attention output must be a square matrix")
    if expected_dim is not None and parameter.shape != (expected_dim, expected_dim):
        raise ValueError("Unexpected DiT adaptation matrix dimension")
    parameter.requires_grad_(True)
    if [n for n, p in system.named_parameters() if p.requires_grad] != [name]:
        raise ValueError("Only one cross-attention output weight may train")
    if state_hash(system.state_dict()) != before:
        raise ValueError("Adaptation setup changed initial weights")
    return name, parameter


def frozen_except_adaptation_hash(system, name):
    if name not in dict(system.named_parameters()):
        raise ValueError("Adaptation parameter is absent")
    return state_hash({k: v for k, v in system.state_dict().items() if k != name})


def adaptation_flow_loss(system, batch, controls, weight, noise, flow_time, arm):
    if arm not in ARMS:
        raise ValueError("Unknown adaptation arm")
    result = cached_flow(system, batch, controls, weight, noise, flow_time,
                         zero=arm == "zero_local", audio_gate=True)
    return (result["prediction"] - result["velocity_target"])[observed(batch["q"])].square().mean()


def evaluate_probe(system, head, validation, dev, *, device, batch_size=32, curves_path=None):
    return evaluate_compact(system, head, validation, dev, device=device, batch_size=batch_size,
        seeds=NOISE_SEEDS, modes=("full", "zero", "reverse", "oracle"), curves_path=curves_path)


def load_source(source_run, device="cpu"):
    """Restore exact epoch8 uniform adapter and its immutable original inputs."""
    source_run = Path(source_run)
    provenance_path = source_run / "provenance.json"
    adapter_path = source_run / "final_epoch008.pt"
    summary_path = source_run / "summary.json"
    source_curves_path = source_run / "final_epoch008_curves.pt"
    sidecar_path = source_run / "final_epoch008_curves.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    source_recipe = provenance["recipe"]
    recipe_hash = canonical_hash(source_recipe)
    if provenance.get("recipe_sha256") != recipe_hash:
        raise ValueError("Source provenance recipe hash differs")
    expected = {"arm": "uniform", "epochs": 8, "seed": 46, "batch_size": 16}
    if source_recipe.get("schema") != "projection_scaled_centered_probe_v1" or any(
            source_recipe["args"].get(k) != v for k, v in expected.items()):
        raise ValueError("Require the fixed uniform epoch8 source experiment")
    paths = {name: Path(source_recipe["args"][name]) for name in INPUT_NAMES}
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes != source_recipe["input_sha256"]:
        raise ValueError("Original source input hashes differ")
    source_summary = json.loads(summary_path.read_text(encoding="utf8"))
    source_binding = json.loads(sidecar_path.read_text(encoding="utf8"))
    adapter_hash = sha(adapter_path)
    curves_hash = sha(source_curves_path)
    if (source_summary.get("recipe_sha256") != recipe_hash
            or source_summary.get("completed_epochs") != 8
            or source_summary.get("arm") != "uniform"
            or source_summary.get("frozen_unchanged") is not True
            or source_summary.get("head_unchanged") is not True
            or source_summary.get("curve_provenance", {}).get("sha256") != sha(sidecar_path)
            or source_binding.get("checkpoint_sha256") != adapter_hash
            or source_binding.get("curve_sha256") != curves_hash
            or source_binding.get("recipe_sha256") != recipe_hash
            or source_binding.get("cache_sha256") != hashes["cache"]):
        raise ValueError("Source summary/curve/checkpoint hash binding differs")
    cfg = yaml.safe_load(paths["config"].read_text(encoding="utf8"))
    original = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    cache = torch.load(paths["cache"], map_location="cpu", weights_only=False)
    bundle = torch.load(paths["bundle"], map_location="cpu", weights_only=False)
    fitted = torch.load(paths["weights"], map_location="cpu", weights_only=False)
    validate_inputs(cache, bundle, fitted["states"], original, cfg)
    lock = json.loads(paths["split_lock"].read_text(encoding="utf8"))
    fit_ids, val_ids = read_allowlist(paths["fit_ids"]), read_allowlist(paths["validation_ids"])
    data_scope = validate_internal_split(cache, bundle, lock, fit_ids, val_ids)
    if len(fit_ids) != 2315 or len(val_ids) != 405 or len(data_scope["fit_speakers"]) != 19:
        raise ValueError("Require fixed2315fit/405heldout internal experiment")
    for name in ("checkpoint", "config", "bundle"):
        if cache["provenance"].get(name + "_sha256") != hashes[name]:
            raise ValueError("Changed cache provenance: " + name)
    if fitted["provenance"].get("bundle_sha256") != hashes["bundle"]:
        raise ValueError("Fixed basis belongs to different bundle")
    if cfg["data"]["emotion_classes"][0] != "neutral":
        raise ValueError("Gate requires neutral class0")
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if (adapter.get("schema") != source_recipe["schema"] or adapter.get("recipe") != source_recipe
            or adapter.get("recipe_sha256") != recipe_hash or adapter.get("completed_epochs") != 8
            or adapter.get("step") != 8 * ((2315 + 15) // 16)):
        raise ValueError("Source adapter is not the complete uniform epoch8 result")
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(original["model"], strict=True)
    if system.residual_scale != .25 or frozen_hash(system) != adapter["frozen_state_sha256"]:
        raise ValueError("Frozen original model differs from source adapter")
    system.local_projection.load_state_dict(adapter["local_projection"], strict=True)
    tr = bundle["bundles"]["internal"]
    head = PredictableAudioHead(fitted["states"]["rrr_rank8"], tr["motion_bins"], tr["weight"]).to(device).eval()
    expected_head = state_hash(head.state_dict())
    if state_hash(adapter["head"]) != adapter["head_sha256"] or expected_head != adapter["head_sha256"]:
        raise ValueError("Source head/basis/std/scale differs from fixed fitted head")
    head.load_state_dict(adapter["head"], strict=True)
    freeze_module(head)
    freeze_module(system)
    return {"system": system, "head": head, "cache": cache, "bundle": bundle,
        "source_recipe": source_recipe, "input_sha256": hashes, "data_scope": data_scope,
        "source_provenance_sha256": sha(provenance_path), "source_adapter_sha256": adapter_hash,
        "source_recipe_sha256": recipe_hash, "config": cfg,
        "source_summary_sha256": sha(summary_path), "source_curve_sha256": curves_hash,
        "source_curve_sidecar_sha256": sha(sidecar_path),
        "source_head_sha256": expected_head, "source_projection_sha256": state_hash(system.local_projection.state_dict())}


def source_paths():
    root = Path(__file__).resolve().parents[1]
    names = ["scripts/train_audio_conditioned_flow_probe.py", "scripts/train_formal_predictable_projection.py",
        "scripts/train_projection_schedule_ablation.py", "scripts/train_predictable_renderer.py",
        "scripts/train_neutral_affect_audio_ablation.py", "scripts/train_neutral_affect_pilot.py",
        "scripts/neutral_affect_metrics.py", "scripts/emotion_ray_metrics.py",
        "kinetalk_b0/__init__.py", "kinetalk_b0/models/__init__.py", "kinetalk_b0/models/neutral_affect.py",
        "kinetalk_b0/models/dit.py", "kinetalk_b0/models/model.py", "kinetalk_b0/models/encoders.py",
        "kinetalk_b0/models/semantic.py", "kinetalk_b0/predictable_motion.py", "kinetalk_b0/utils.py",
        "kinetalk_b0/emotion_ray.py", "kinetalk_b0/semantic_data.py", "kinetalk_b0/semantic_losses.py", "kinetalk_b0/neutral_data.py",
        "docs/AUDIO_CONDITIONED_FLOW_PROBE.md", "tests/test_audio_conditioned_flow_probe.py"]
    return [root / name for name in names]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-batch-size", type=int, default=32)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh output required; no automatic resume or overwrite")
    if args.eval_batch_size < 1:
        raise ValueError("Positive eval batch size required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(46)
    loaded = load_source(args.source_run, args.device)
    system, head, cache, bundle = (loaded[k] for k in ("system", "head", "cache", "bundle"))
    name, parameter = configure_adaptation(system, expected_dim=192)
    frozen_before = frozen_except_adaptation_hash(system, name)
    head_before, projection_before = state_hash(head.state_dict()), state_hash(system.local_projection.state_dict())
    initial_system_hash = state_hash(system.state_dict())
    paths, root = source_paths(), Path(__file__).resolve().parents[1]
    recipe = {"schema": SCHEMA, "args": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_recipe": loaded["source_recipe"], "source_recipe_sha256": loaded["source_recipe_sha256"],
        "source_provenance_sha256": loaded["source_provenance_sha256"], "source_adapter_sha256": loaded["source_adapter_sha256"],
        "source_summary_sha256": loaded["source_summary_sha256"], "source_curve_sha256": loaded["source_curve_sha256"],
        "source_curve_sidecar_sha256": loaded["source_curve_sidecar_sha256"],
        "input_sha256": loaded["input_sha256"], "source_sha256": {str(path.resolve()): sha(path) for path in paths},
        "data_scope": loaded["data_scope"], "seed": 46, "epochs": 8, "batch_size": 16, "lr": 1e-5,
        "optimizer": "Adam", "clip_grad_norm": 1., "trainable": [name], "trainable_parameters": parameter.numel(),
        "initial_system_sha256": initial_system_hash, "initial_trainable_sha256": state_hash({name: parameter}),
        "initial_projection_sha256": projection_before, "initial_head_sha256": head_before,
        "loss": "observed standard flow velocity MSE only", "audio_activity_gate": True,
        "time_sampling": "uniform U[0,1) with independent20% mass replaced by t=0; historical draws",
        "teacher_local_probability": 0., "teacher_choice_draws": "consumed/hashed for paired historical RNG only; never used",
        "eval_noise_seeds": list(NOISE_SEEDS), "eval_modes": ["full", "zero", "reverse", "oracle"],
        "decode_steps": 12, "checkpoint_selection": "none; epoch0 baseline and fixed final epoch8 only",
        "all_modules_eval_mode": True, "torch": str(torch.__version__),
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "default_replaced": False, "motion_use": "training velocity target only; eval oracle is explicitly target-conditioned",
        "scope": "Training-identity adaptation probe with405internal heldout identities; existing data, no final-test claim."}
    args.output.mkdir(parents=True, exist_ok=False)
    for path in paths:
        dest = args.output / "source" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe)})
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    generator = torch.Generator().manual_seed(46)
    tr, dev = bundle["bundles"]["internal"], bundle["bundles"]["external_dev"]
    features, weight = audio_features(tr), tr["weight"].float()
    train, validation = cache["splits"]["train"], cache["splits"]["validation"]
    shape = train["q"]["motion"].shape[1:]
    batch_hash, noise_hash, choice_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    step, epoch_done, started = 0, 0, time.time()

    def payload():
        return {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
            "trainable": {name: parameter}, "trainable_name": name,
            "local_projection": system.local_projection.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": capture_rng(generator),
            "step": step, "completed_epochs": epoch_done, "frozen_state_sha256": frozen_before,
            "head_sha256": head_before, "projection_sha256": projection_before,
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
            "default_enabled": False, "selection": "none"}

    def assert_frozen():
        if (frozen_except_adaptation_hash(system, name) != frozen_before
                or state_hash(head.state_dict()) != head_before
                or state_hash(system.local_projection.state_dict()) != projection_before):
            raise RuntimeError("A frozen source weight or head changed")
        if any(module.training for module in system.modules()) or any(module.training for module in head.modules()):
            raise RuntimeError("All model modules must remain in eval mode")

    def evaluate(epoch):
        curves = args.output / f"epoch{epoch:03d}_curves.pt" if epoch == 0 else args.output / "final_epoch008_curves.pt"
        ck = args.output / "epoch000.pt" if epoch == 0 else args.output / "final_epoch008.pt"
        result = evaluate_probe(system, head, validation, dev, device=args.device,
                                batch_size=args.eval_batch_size, curves_path=curves)
        binding = curve_binding(curves, ck, recipe, loaded["input_sha256"]["cache"])
        assert_frozen()
        return result, binding

    save_checkpoint(args.output / "epoch000.pt", payload())
    save_checkpoint(args.output / "last.pt", payload())
    baseline, baseline_binding = evaluate(0)
    source_curves = torch.load(args.source_run / "final_epoch008_curves.pt", map_location="cpu", weights_only=False, mmap=True)
    initial_curves = torch.load(args.output / "epoch000_curves.pt", map_location="cpu", weights_only=False, mmap=True)
    source_reproduction = {}
    if source_curves.get("noise_seeds") != [42, 123, 2026] or source_curves.get("decode_steps") != 12:
        raise ValueError("Source baseline sampling protocol differs")
    for seed in ("42", "123", "2026"):
        source_reproduction[seed] = {}
        for mode in ("full", "zero", "reverse", "oracle"):
            a, b = initial_curves["motion"][seed][mode], source_curves["motion"][seed][mode]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            source_reproduction[seed][mode] = float((a - b).abs().max())
    save_json(args.output / "epoch000_evaluation.json", {"reports": baseline, "diagnostics": diagnostics(baseline),
        "curve_provenance": baseline_binding, "source_reproduction_max_abs_error": source_reproduction,
        "checkpoint_selection_performed": False})
    for epoch in range(1, 9):
        order = torch.randperm(len(features), generator=generator)
        sum_loss, count, t0_count = 0., 0, 0
        for ids in order.split(16):
            noise, flow_time, choose = draws(generator, len(ids), shape)
            batch_hash.update(ids.numpy().tobytes()); noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
            choice_hash.update(choose.numpy().tobytes())
            b, w = batch_to_device(train, ids, args.device), weight[ids].to(args.device)
            with torch.no_grad():
                controls = head(features[ids].to(args.device), w)
            loss = adaptation_flow_loss(system, b, controls, w, noise.to(args.device), flow_time.to(args.device), args.arm)
            norm = optimize(loss, optimizer, [parameter])
            observed_count = int(observed(b["q"]).sum())
            sum_loss += float(loss.detach()) * observed_count; count += observed_count
            t0_count += int((flow_time == 0).sum()); step += 1
            if step % 100 == 0:
                print(json.dumps({"arm": args.arm, "epoch": epoch, "step": step,
                    "flow_mse": float(loss.detach()), "grad_norm": norm}), flush=True)
        epoch_done = epoch
        assert_frozen()
        save_checkpoint(args.output / "last.pt", payload())
        row = {"arm": args.arm, "epoch": epoch, "step": step, "mean_observed_flow_mse": sum_loss / count,
            "t0_fraction": t0_count / len(features), "samples_seen": len(features),
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started}
        save_json(args.output / f"epoch{epoch:03d}.json", row); print(json.dumps(row), flush=True)
    save_checkpoint(args.output / "final_epoch008.pt", payload())
    final, final_binding = evaluate(8)
    save_json(args.output / "summary.json", {"schema": SCHEMA, "recipe_sha256": canonical_hash(recipe),
        "arm": args.arm, "completed_epochs": 8, "optimizer_steps": step,
        "epoch000": baseline, "epoch000_diagnostics": diagnostics(baseline), "epoch000_curve_provenance": baseline_binding,
        "final": final, "final_diagnostics": diagnostics(final), "curve_provenance": final_binding,
        "frozen_unchanged": True, "head_unchanged": True, "projection_unchanged": True,
        "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
        "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
        "trainable_delta_rms": float((parameter.detach().cpu() - torch.load(args.output / "epoch000.pt",
            map_location="cpu", weights_only=False)["trainable"][name]).square().mean().sqrt()),
        "checkpoint_selection_performed": False, "test_loaded": False, "default_replaced": False})
    save_json(args.output / "output_hashes.json", {str(path.relative_to(args.output)): sha(path)
        for path in args.output.rglob("*") if path.is_file() and path.name != "output_hashes.json"})
    print("COMPLETE fixed8epoch matched audio-conditioned flow adaptation", flush=True)


if __name__ == "__main__":
    main()
