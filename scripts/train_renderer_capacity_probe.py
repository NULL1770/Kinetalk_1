"""Matched oracle/audio/zero-local flow training with a frozen audio predictor.

Restores uniform epoch8 without changing inherited assets. Only the complete
existing renderer and its shared local projection train; this is a capacity
baseline, not a new architecture. Eight epochs are a diagnostic budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.utils import freeze_module
from scripts.train_audio_conditioned_flow_probe import (
    NOISE_SEEDS, evaluate_probe, load_source, source_paths as inherited_source_paths,
)
from scripts.train_formal_predictable_projection import (
    canonical_hash, capture_rng, cpu_tree, restore_rng, save_checkpoint, save_json,
)
from scripts.train_projection_schedule_ablation import curve_binding, diagnostics, draws
from scripts.train_predictable_renderer import (
    batch_to_device, cached_flow, observed, optimize, sha, state_hash,
)

SCHEMA = "renderer_capacity_probe_v1"
CONTROL_SCHEMA = "renderer_capacity_training_controls_v1"
ARMS = ("oracle_local", "audio_local", "zero_local")
PREFIXES = ("renderer.", "local_projection.")
EPOCHS, SEED, BATCH_SIZE, LR = 8, 46, 16, 1e-5


def capacity_state(system):
    """The complete changed state, using original model key names."""
    return {k: v for k, v in system.state_dict().items() if k.startswith(PREFIXES)}


def frozen_capacity_hash(system):
    return state_hash({k: v for k, v in system.state_dict().items() if not k.startswith(PREFIXES)})


def configure_capacity(system):
    """Open the existing renderer/projection, preserving all source values."""
    before = state_hash(system.state_dict())
    freeze_module(system)
    system.renderer.requires_grad_(True)
    system.local_projection.requires_grad_(True)
    system.eval()
    selected = [(n, p) for n, p in system.named_parameters() if p.requires_grad]
    expected = [n for n, _ in system.named_parameters() if n.startswith(PREFIXES)]
    if [n for n, _ in selected] != expected or not selected:
        raise ValueError("Only complete renderer and local projection may train")
    if state_hash(system.state_dict()) != before:
        raise RuntimeError("Setup altered source weights")
    return [n for n, _ in selected], [p for _, p in selected]


def assert_frozen(system, head, frozen_sha256, head_sha256):
    if frozen_capacity_hash(system) != frozen_sha256 or state_hash(head.state_dict()) != head_sha256:
        raise RuntimeError("Frozen backbone or audio head changed")
    if any(p.requires_grad for p in head.parameters()):
        raise RuntimeError("Audio head must stay frozen")
    if any(m.training for m in system.modules()) or any(m.training for m in head.modules()):
        raise RuntimeError("All modules must remain in eval mode")
    if any(p.requires_grad for n, p in system.named_parameters() if not n.startswith(PREFIXES)):
        raise RuntimeError("A protected parameter is trainable")


def validate_training_controls(record, loaded):
    """Bind fixed student predictions to the exact fit rows and control space."""
    tr, head = loaded["bundle"]["bundles"]["internal"], loaded["head"]
    expected_ids = list(map(str, tr["clip_id"]))
    if record.get("schema") != CONTROL_SCHEMA or record.get("clip_ids") != expected_ids:
        raise ValueError("Training controls schema/fit clip order differs")
    if record.get("source_input_sha256") != loaded["input_sha256"]:
        raise ValueError("Training controls source input hashes differ")
    if record.get("head_sha256") != state_hash(head.state_dict()):
        raise ValueError("Training controls belong to a different fixed head/coordinate system")
    controls, weight = record.get("controls"), record.get("weight")
    expected_weight = tr["weight"].detach().cpu().float()
    shape = (*expected_weight.shape, head.linear.out_features)
    if (not torch.is_tensor(controls) or not controls.is_floating_point()
            or controls.device.type != "cpu" or tuple(controls.shape) != shape):
        raise ValueError("Training controls require finite CPU [fit,K,rank] values")
    if not torch.is_tensor(weight) or not torch.equal(weight.cpu().float(), expected_weight):
        raise ValueError("Training control weights differ from source clock")
    if not torch.isfinite(controls).all() or (controls[expected_weight <= 0] != 0).any():
        raise ValueError("Training controls have nonfinite or nonzero padded values")
    mean = (controls.double() * expected_weight.double()[..., None]).sum(1)
    mean = mean / expected_weight.double().sum(1)[:, None].clamp_min(1)
    if float(mean.abs().max()) > 1e-5:
        raise ValueError("Training controls must remain weighted clip-centered")
    return controls.float().contiguous()


def load_capacity_source(source_run, training_controls=None, device="cpu"):
    """Public loader for independent audit; controls are unnecessary to restore."""
    loaded = load_source(source_run, device)
    if training_controls is not None:
        path = Path(training_controls)
        record = torch.load(path, map_location="cpu", weights_only=False)
        loaded["training_controls"] = validate_training_controls(record, loaded)
        loaded["training_controls_sha256"] = sha(path)
        loaded["training_controls_record"] = record
    return loaded


@torch.no_grad()
def select_training_controls(head, motion_bins, weight, external_controls, arm):
    """Oracle/zero never consume the student prediction values."""
    if arm == "oracle_local":
        return head.teacher(motion_bins, weight).detach()
    if arm == "zero_local":
        return weight.new_zeros((*weight.shape, head.linear.out_features))
    if arm == "audio_local":
        if external_controls is None or external_controls.shape != (*weight.shape, head.linear.out_features):
            raise ValueError("Audio arm requires fixed predictions on the matching bins")
        return external_controls.detach()
    raise ValueError("Unknown capacity arm")


def capacity_flow_loss(system, batch, controls, weight, noise, flow_time, arm):
    if arm not in ARMS:
        raise ValueError("Unknown capacity arm")
    result = cached_flow(system, batch, controls, weight, noise, flow_time,
                         zero=arm == "zero_local", audio_gate=True)
    return (result["prediction"] - result["velocity_target"])[observed(batch["q"])].square().mean()


def checkpoint_payload(system, head, optimizer, generator, recipe, *, step,
                       completed_epochs, frozen_sha256, hashes=None, elapsed_seconds=0.):
    assert_frozen(system, head, frozen_sha256, recipe["initial_head_sha256"])
    capacity = capacity_state(system)
    return cpu_tree({"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
        "capacity": capacity, "capacity_sha256": state_hash(capacity),
        "head": head.state_dict(), "head_sha256": state_hash(head.state_dict()),
        "optimizer": optimizer.state_dict(), "rng": capture_rng(generator),
        "step": step, "completed_epochs": completed_epochs,
        "frozen_state_sha256": frozen_sha256, "initial_system_sha256": recipe["initial_system_sha256"],
        "elapsed_seconds": elapsed_seconds, "default_enabled": False, "selection": "none", **(hashes or {})})


def restore_capacity_checkpoint(system, head, payload, recipe=None, optimizer=None, generator=None):
    """Strictly restore compact changed state on an independently loaded source."""
    saved_recipe = payload.get("recipe")
    if (payload.get("schema") != SCHEMA or not isinstance(saved_recipe, dict)
            or payload.get("recipe_sha256") != canonical_hash(saved_recipe)
            or (recipe is not None and saved_recipe != recipe)):
        raise ValueError("Checkpoint recipe/schema differs")
    if payload.get("selection") != "none" or payload.get("default_enabled") is not False:
        raise ValueError("Checkpoint selection/default contract differs")
    if payload.get("completed_epochs", -1) < 0 or payload.get("step", -1) < 0:
        raise ValueError("Invalid checkpoint progress")
    if payload.get("frozen_state_sha256") != frozen_capacity_hash(system):
        raise ValueError("Frozen source differs from capacity checkpoint")
    expected_head = state_hash(head.state_dict())
    if (payload.get("head_sha256") != expected_head or state_hash(payload["head"]) != expected_head
            or saved_recipe.get("initial_head_sha256") != expected_head):
        raise ValueError("Frozen head differs from capacity checkpoint")
    expected, saved = capacity_state(system), payload.get("capacity", {})
    if set(saved) != set(expected) or state_hash(saved) != payload.get("capacity_sha256"):
        raise ValueError("Capacity checkpoint keys/hash differ")
    for name, tensor in saved.items():
        if (not torch.is_tensor(tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype or not torch.isfinite(tensor).all()):
            raise ValueError("Invalid capacity tensor: " + name)
    state = system.state_dict()
    state.update(saved)
    system.load_state_dict(state, strict=True)
    system.eval(); head.eval()
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if generator is not None:
        restore_rng(payload["rng"], generator)
    return payload


def _link_or_copy(source, target):
    # Fresh output only; a hard link saves storage for immutable common curves.
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def reuse_baseline(baseline_run, output, recipe, checkpoint_path, *, system=None, head=None):
    """Reuse a verified eight-seed epoch0 from this runner or the old probe."""
    baseline_run, output = Path(baseline_run), Path(output)
    prov_path = baseline_run / "provenance.json"
    evaluation_path = baseline_run / "epoch000_evaluation.json"
    curve_path = baseline_run / "epoch000_curves.pt"
    old_ck_path = baseline_run / "epoch000.pt"
    sidecar_path = baseline_run / "epoch000_curves.provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf8"))
    old_recipe = prov["recipe"]
    old_recipe_hash = canonical_hash(old_recipe)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf8"))
    binding = json.loads(sidecar_path.read_text(encoding="utf8"))
    old_ck = torch.load(old_ck_path, map_location="cpu", weights_only=False)
    for key in ("initial_system_sha256", "initial_head_sha256", "input_sha256", "source_adapter_sha256"):
        if old_recipe.get(key) != recipe.get(key):
            raise ValueError("Baseline belongs to a different source: " + key)
    if (prov.get("recipe_sha256") != old_recipe_hash or old_ck.get("recipe_sha256") != old_recipe_hash
            or old_ck.get("recipe") != old_recipe or old_ck.get("completed_epochs") != 0
            or old_ck.get("step") != 0 or old_recipe.get("eval_noise_seeds") != list(NOISE_SEEDS)
            or old_recipe.get("eval_modes") != ["full", "zero", "reverse", "oracle"]
            or old_recipe.get("decode_steps") != 12 or binding.get("recipe_sha256") != old_recipe_hash
            or binding.get("checkpoint_sha256") != sha(old_ck_path)
            or binding.get("curve_sha256") != sha(curve_path)
            or binding.get("cache_sha256") != recipe["input_sha256"]["cache"]
            or evaluation.get("curve_provenance", {}).get("sha256") != sha(sidecar_path)):
        raise ValueError("Baseline checkpoint/curve/protocol binding differs")
    if system is not None:
        if state_hash(system.state_dict()) != recipe["initial_system_sha256"]:
            raise ValueError("Current system is not the declared common initial model")
        if old_ck.get("schema") == SCHEMA:
            if state_hash(old_ck.get("capacity", {})) != state_hash(capacity_state(system)):
                raise ValueError("Baseline initial capacity weights differ")
            if old_ck.get("frozen_state_sha256") != frozen_capacity_hash(system):
                raise ValueError("Baseline protected weights differ")
        elif old_ck.get("schema") == "audio_conditioned_flow_probe_v1":
            name = old_ck.get("trainable_name")
            params = dict(system.named_parameters())
            if name not in params or state_hash(old_ck.get("trainable", {})) != state_hash({name: params[name]}):
                raise ValueError("Historical baseline initial adaptation weight differs")
            if state_hash(old_ck["local_projection"]) != state_hash(system.local_projection.state_dict()):
                raise ValueError("Historical baseline local projection differs")
            frozen_old = state_hash({k: v for k, v in system.state_dict().items() if k != name})
            if old_ck.get("frozen_state_sha256") != frozen_old:
                raise ValueError("Historical baseline frozen weights differ")
        else:
            raise ValueError("Unsupported baseline checkpoint schema")
    if head is not None and (state_hash(old_ck["head"]) != state_hash(head.state_dict())
                             or old_ck.get("head_sha256") != state_hash(head.state_dict())):
        raise ValueError("Baseline frozen head weights differ")
    curves = torch.load(curve_path, map_location="cpu", weights_only=False, mmap=True)
    if curves.get("noise_seeds") != list(NOISE_SEEDS) or curves.get("decode_steps") != 12:
        raise ValueError("Baseline curve sampling differs")
    if set(curves.get("motion", {})) != set(map(str, NOISE_SEEDS)):
        raise ValueError("Baseline curves missing a prescribed seed")
    for seed in map(str, NOISE_SEEDS):
        if set(curves["motion"][seed]) != {"full", "zero", "reverse", "oracle"}:
            raise ValueError("Baseline curves missing a prescribed mode")
    del curves, old_ck
    target = output / "epoch000_curves.pt"
    _link_or_copy(curve_path, target)
    new_binding = curve_binding(target, checkpoint_path, recipe, recipe["input_sha256"]["cache"])
    reuse = {"run": str(baseline_run.resolve()), "provenance_sha256": sha(prov_path),
        "evaluation_sha256": sha(evaluation_path), "curve_sha256": sha(curve_path),
        "checkpoint_sha256": sha(old_ck_path), "curve_sidecar_sha256": sha(sidecar_path)}
    return evaluation["reports"], new_binding, reuse


def source_paths(preparation_script, protocol):
    paths = inherited_source_paths() + [Path(__file__), Path(__file__).resolve().parents[1] /
        "tests/test_renderer_capacity_probe.py", Path(__file__).resolve().parents[1] /
        "scripts/probe_scaled_motion_basis.py", Path(preparation_script), Path(protocol)]
    return list(dict.fromkeys(path.resolve() for path in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "training-controls", "preparation-script", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh output required; no automatic resume or overwrite")
    if args.eval_batch_size < 1:
        raise ValueError("Positive eval batch size required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    loaded = load_capacity_source(args.source_run, args.training_controls, args.device)
    system, head, cache, bundle = (loaded[k] for k in ("system", "head", "cache", "bundle"))
    names, parameters = configure_capacity(system)
    initial_system = state_hash(system.state_dict())
    initial_capacity = state_hash(capacity_state(system))
    frozen_before, head_before = frozen_capacity_hash(system), state_hash(head.state_dict())
    paths = source_paths(args.preparation_script, args.protocol)
    root = Path(__file__).resolve().parents[1]
    control_provenance = loaded["training_controls_record"].get("provenance", {})
    if (control_provenance.get("preparation_source_sha256") != sha(args.preparation_script)
            or control_provenance.get("source_recipe_sha256") != loaded["source_recipe_sha256"]
            or control_provenance.get("source_adapter_sha256") != loaded["source_adapter_sha256"]):
        raise ValueError("Training-control preparation/source provenance differs")
    recipe = {"schema": SCHEMA,
        "args": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        **{k: loaded[k] for k in ("source_recipe", "source_recipe_sha256", "source_provenance_sha256",
            "source_adapter_sha256", "source_summary_sha256", "source_curve_sha256",
            "source_curve_sidecar_sha256", "input_sha256", "data_scope", "training_controls_sha256")},
        "source_sha256": {str(path): sha(path) for path in paths},
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "optimizer": "Adam", "clip_grad_norm": 1., "trainable": names,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "initial_system_sha256": initial_system, "initial_capacity_sha256": initial_capacity,
        "initial_head_sha256": head_before, "frozen_state_sha256": frozen_before,
        "loss": "observed standard flow velocity MSE only", "audio_activity_gate": True,
        "time_sampling": "uniform U[0,1) with independent20% mass replaced by t=0; historical draws",
        "teacher_choice_draws": "consumed/hashed for matched historical RNG only; never used",
        "eval_noise_seeds": list(NOISE_SEEDS), "eval_modes": ["full", "zero", "reverse", "oracle"],
        "decode_steps": 12, "checkpoint_selection": "none; fixed final epoch8 diagnostic",
        "all_modules_eval_mode": True, "torch": str(torch.__version__),
        "training_controls_role": "fixed fit-only cross-fitted student predictions; unchanged head for deployment",
        "coordinate_scope": "shared fit-only U/std/target-scale; not a fully nested OOF representation",
        "motion_use": "flow target; oracle_local training condition only in explicitly oracle arm",
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "default_replaced": False, "scope": "Matched renderer capacity diagnostic; no architecture innovation or final-test claim"}
    # Verify source files before creating output. All paths are recorded verbatim.
    args.output.mkdir(parents=True, exist_ok=False)
    for path in paths:
        relative = path.relative_to(root) if path.is_relative_to(root) else Path("external") / path.name
        dest = args.output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe)})
    optimizer = torch.optim.Adam(parameters, lr=LR)
    generator = torch.Generator().manual_seed(SEED)
    tr, dev = bundle["bundles"]["internal"], bundle["bundles"]["external_dev"]
    train, validation = cache["splits"]["train"], cache["splits"]["validation"]
    weight, motion_bins = tr["weight"].float(), tr["motion_bins"].float()
    external = loaded["training_controls"]
    shape = train["q"]["motion"].shape[1:]
    batch_hash, noise_hash, choice_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    step, epoch_done, started = 0, 0, time.time()

    def hashes():
        return {"minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
                "teacher_choice_draw_sha256": choice_hash.hexdigest()}

    def payload():
        return checkpoint_payload(system, head, optimizer, generator, recipe, step=step,
            completed_epochs=epoch_done, frozen_sha256=frozen_before, hashes=hashes(),
            elapsed_seconds=time.time() - started)

    def evaluate(epoch):
        stem = "epoch000" if epoch == 0 else "final_epoch008"
        curves, checkpoint = args.output / (stem + "_curves.pt"), args.output / (stem + ".pt")
        reports = evaluate_probe(system, head, validation, dev, device=args.device,
                                 batch_size=args.eval_batch_size, curves_path=curves)
        binding = curve_binding(curves, checkpoint, recipe, loaded["input_sha256"]["cache"])
        assert_frozen(system, head, frozen_before, head_before)
        return reports, binding

    save_checkpoint(args.output / "epoch000.pt", payload())
    save_checkpoint(args.output / "last.pt", payload())
    reuse = None
    if args.baseline_run is not None:
        baseline, baseline_binding, reuse = reuse_baseline(args.baseline_run, args.output, recipe,
            args.output / "epoch000.pt", system=system, head=head)
    else:
        baseline, baseline_binding = evaluate(0)
    # Independently check common source reproduction for its three saved seeds.
    source_curves = torch.load(args.source_run / "final_epoch008_curves.pt", map_location="cpu", weights_only=False, mmap=True)
    baseline_curves = torch.load(args.output / "epoch000_curves.pt", map_location="cpu", weights_only=False, mmap=True)
    source_reproduction = {}
    for seed in ("42", "123", "2026"):
        source_reproduction[seed] = {}
        for mode in recipe["eval_modes"]:
            a, b = source_curves["motion"][seed][mode], baseline_curves["motion"][seed][mode]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            source_reproduction[seed][mode] = float((a - b).abs().max())
    del source_curves, baseline_curves
    save_json(args.output / "epoch000_evaluation.json", {"reports": baseline, "diagnostics": diagnostics(baseline),
        "curve_provenance": baseline_binding, "source_reproduction_max_abs_error": source_reproduction,
        "baseline_reuse": reuse, "checkpoint_selection_performed": False})
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(weight), generator=generator)
        sum_loss, count, t0_count = 0., 0, 0
        for ids in order.split(BATCH_SIZE):
            noise, flow_time, choose = draws(generator, len(ids), shape)
            batch_hash.update(ids.numpy().tobytes())
            noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
            choice_hash.update(choose.numpy().tobytes())
            batch, w = batch_to_device(train, ids, args.device), weight[ids].to(args.device)
            # Only audio reads saved prediction values; oracle reads real fit motion.
            mb = motion_bins[ids].to(args.device) if args.arm == "oracle_local" else None
            ac = external[ids].to(args.device) if args.arm == "audio_local" else None
            controls = select_training_controls(head, mb, w, ac, args.arm)
            loss = capacity_flow_loss(system, batch, controls, w, noise.to(args.device), flow_time.to(args.device), args.arm)
            norm = optimize(loss, optimizer, parameters)
            observed_count = int(observed(batch["q"]).sum())
            sum_loss += float(loss.detach()) * observed_count; count += observed_count
            t0_count += int((flow_time == 0).sum()); step += 1
            if step % 100 == 0:
                print(json.dumps({"arm": args.arm, "epoch": epoch, "step": step,
                    "flow_mse": float(loss.detach()), "grad_norm": norm}), flush=True)
        epoch_done = epoch
        assert_frozen(system, head, frozen_before, head_before)
        save_checkpoint(args.output / "last.pt", payload())
        row = {"arm": args.arm, "epoch": epoch, "step": step,
            "mean_observed_flow_mse": sum_loss / count, "t0_fraction": t0_count / len(weight),
            "samples_seen": len(weight), **hashes(), "elapsed_seconds": time.time() - started}
        save_json(args.output / f"epoch{epoch:03d}.json", row)
        print(json.dumps(row), flush=True)
    save_checkpoint(args.output / "final_epoch008.pt", payload())
    final, final_binding = evaluate(EPOCHS)
    capacity_changed = state_hash(capacity_state(system)) != initial_capacity
    save_json(args.output / "summary.json", {"schema": SCHEMA, "recipe_sha256": canonical_hash(recipe),
        "arm": args.arm, "completed_epochs": EPOCHS, "optimizer_steps": step,
        "epoch000": baseline, "epoch000_diagnostics": diagnostics(baseline),
        "epoch000_curve_provenance": baseline_binding, "final": final,
        "final_diagnostics": diagnostics(final), "curve_provenance": final_binding,
        "frozen_unchanged": True, "head_unchanged": True, "capacity_changed": capacity_changed,
        "trainable": names, "capacity_sha256": state_hash(capacity_state(system)),
        "training_controls_sha256": loaded["training_controls_sha256"], **hashes(),
        "elapsed_seconds": time.time() - started, "baseline_reuse": reuse,
        "checkpoint_selection_performed": False, "test_loaded": False, "default_replaced": False})
    save_json(args.output / "output_hashes.json", {str(path.relative_to(args.output)): sha(path)
        for path in args.output.rglob("*") if path.is_file() and path.name != "output_hashes.json"})
    print("COMPLETE fixed8epoch matched renderer capacity probe", flush=True)


if __name__ == "__main__":
    main()
