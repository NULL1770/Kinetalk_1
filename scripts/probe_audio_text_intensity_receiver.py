"""Bounded target-conditioned receiver diagnosis, never an audio deployment.

Two matched arms use true frame intensity or its per-clip mean on a fixed
training subset128. Each trains eight epochs/64 steps from original epoch000.
No development examples select parameters or stopping; old runs stay immutable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.label_guided_intensity import regional_intensity
from scripts import train_audio_text_dynamics as runner
from scripts.inspect_audio_text_fit import fixed_indices, intensity_scores, load_bound_run, SUBSET_SEED
from scripts.audit_projection_mean_dynamic_tradeoff import exact_error_components, component_summary
from scripts.train_formal_predictable_projection import canonical_hash, capture_rng, save_checkpoint, save_json
from scripts.train_predictable_renderer import basic_metrics, batch_to_device, observed, sha, state_hash
from scripts.train_projection_schedule_ablation import draws

SCHEMA = "audio_text_intensity_receiver_probe_v1"
ARMS = ("oracle_frame", "oracle_static")
BASE_MODES = ("frame_gt", "static_gt", "predicted")
DELTA = .25


def receiver_affect(encoder, batch, data, mode, *, delta=0.):
    if mode not in BASE_MODES or delta not in (0., DELTA): raise ValueError("Unknown fixed receiver intervention")
    valid = batch["q"]["valid"]
    args = (data["features"], valid, batch["affect"]["global"], data["tokens"], data["token_valid"])
    initial = encoder(*args, use_text=True)
    predicted = initial["predicted_intensity"]
    if mode == "predicted":
        drive = predicted
    else:
        mask = data["intensity_valid"] & valid[..., None]
        if (data["target_intensity"].shape != predicted.shape or mask.dtype != torch.bool
                or not torch.isfinite(data["target_intensity"][mask]).all()
                or (data["target_intensity"][mask] < 0).any() or not mask.any(1).all()):
            raise ValueError("Every diagnostic clip requires finite observed target intensity")
        truth = torch.where(mask, data["target_intensity"], 0.)
        if mode == "static_gt":
            truth = (truth.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)).expand_as(truth)
        drive = torch.where(mask, truth, predicted)
    drive = torch.where(valid[..., None], drive + delta, 0.)
    output = initial if mode == "predicted" and delta == 0 else encoder(*args, use_text=True, intensity_override=drive)
    return {**batch["affect"], "local": output["local"]}, output


def training_loss(system, encoder, batch, data, scales, noise, flow_time, arm):
    if arm not in ARMS: raise ValueError("Receiver training requires one fixed oracle diagnostic arm")
    affect, output = receiver_affect(encoder, batch, data, "frame_gt" if arm == "oracle_frame" else "static_gt")
    q = batch["q"]
    flow = system.flow(torch.where(observed(q), q["motion"], 0.), q["content"], q["valid"], batch["identity"], affect,
                       noise=noise, time=flow_time, base=batch["base"])
    losses = {"flow": (flow["prediction"] - flow["velocity_target"])[observed(q)].square().mean(),
        "predicted_intensity": runner.masked_huber(output["predicted_intensity"], data["target_intensity"], data["intensity_valid"])}
    generated = system.generate(q["content"], q["valid"], batch["identity"], affect,
                               initial_noise=noise, steps=12, base=batch["base"])["motion"]
    losses.update(runner.generated_losses(generated, batch, data, scales))
    return losses["flow"] + sum(runner.LOSS_WEIGHTS[key] * losses[key] for key in runner.LOSS_WEIGHTS), losses


def motion_readout(prediction, reference):
    report = basic_metrics(prediction, reference)
    mask = observed(reference["q"])
    decomposition = {name: component_summary(exact_error_components(prediction, reference["q"]["motion"], mask,
        channels, reference["base"]["b0"], reference["identity"]["baseline"])) for name, channels in runner.GROUPS.items()}
    values = prediction[mask]
    return {"basic_metrics": report, "raw_mean_dynamic_decomposition": decomposition,
        "domain": {"outside_fraction": float(((values < 0) | (values > 1)).float().mean()),
                   "outside_magnitude": float((torch.relu(-values) + torch.relu(values - 1)).mean())}}


def condition_response(left, right, left_i, right_i, mask, intensity_valid):
    difference = right - left
    intensities = (right_i - left_i)[intensity_valid]
    return {"motion_rms": {name: float(difference[..., channels][mask[..., channels]].square().mean().sqrt())
                            for name, channels in runner.GROUPS.items()},
        "generated_intensity_mean_delta": float(intensities.mean()),
        "generated_intensity_rms_delta": float(intensities.square().mean().sqrt()),
        "generated_intensity_fraction_positive": float((intensities > 1e-6).float().mean()),
        "generated_intensity_fraction_nonnegative": float((intensities >= -1e-6).float().mean()),
        "note": "Larger supplied intensity need not imply positive signed brow coefficients; monotonicity here uses generated expression magnitude."}


@torch.inference_mode()
def evaluate(system, encoder, split, audio, text, targets, scales, ids, noise, device):
    reference = batch_to_device(split, ids, "cpu")
    truth, imask = targets["intensity"][ids].float(), targets["valid"][ids]
    report, outputs, generated_i, conditions = {}, {}, {}, {}
    for mode in BASE_MODES:
        for delta in (0., DELTA):
            name = mode + ("_plus025" if delta else "")
            motion, predicted, driving, intensity, local = [], [], [], [], []
            for selected in ids.split(32):
                batch = batch_to_device(split, selected, device)
                data = runner.slice_inputs(audio, text, targets, selected, device)
                affect, encoded = receiver_affect(encoder, batch, data, mode, delta=delta)
                q = batch["q"]
                value = system.generate(q["content"], q["valid"], batch["identity"], affect,
                    initial_noise=noise[selected].to(device), steps=12, base=batch["base"])["motion"]
                out_i, out_valid = regional_intensity(value, observed(q) & data["anchor_valid"][:, None], data["anchors"], scales)
                if not torch.equal(out_valid, data["intensity_valid"]): raise ValueError("Intensity masks changed")
                motion.append(value.cpu()); intensity.append(out_i.cpu()); local.append(affect["local"].cpu())
                predicted.append(encoded["predicted_intensity"].cpu()); driving.append(encoded["driving_intensity"].cpu())
            outputs[name], generated_i[name], conditions[name] = torch.cat(motion), torch.cat(intensity), torch.cat(local)
            report[name] = {"motion": motion_readout(outputs[name], reference),
                "predicted_intensity": intensity_scores(torch.cat(predicted), truth, imask),
                "driving_intensity": intensity_scores(torch.cat(driving), truth, imask),
                "generated_intensity": intensity_scores(generated_i[name], truth, imask)}
    response = {}
    pairs = [(mode, mode + "_plus025") for mode in BASE_MODES] + [("static_gt", "frame_gt"), ("predicted", "frame_gt")]
    for left, right in pairs:
        value = condition_response(outputs[left], outputs[right], generated_i[left], generated_i[right],
                                   observed(reference["q"]), imask)
        value["local_condition_rms"] = float((conditions[right] - conditions[left])[reference["q"]["valid"]].square().mean().sqrt())
        response[right + "_minus_" + left] = value
    return {"conditions": report, "responses": response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Fresh isolated receiver output required")
    if args.output.resolve().is_relative_to(args.run.resolve()): raise ValueError("Do not write into immutable source run")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    run, source_recipe, loaded, derived, original, original_encoder, source_cp, source_hashes, inventory = load_bound_run(args.run, args.device)
    if source_recipe["arm"] != "text": raise ValueError("Receiver probe starts from the text experiment epoch000")
    runner.restore(original, original_encoder, source_cp[0])
    train = loaded["cache"]["splits"]["train"]
    ids = fixed_indices(len(train["q"]["valid"]))
    if len(ids) != 128: raise ValueError("Prespecified receiver diagnosis requires exactly128 fit clips")
    noise = torch.randn(train["q"]["motion"].shape, generator=torch.Generator().manual_seed(42))
    audio, text, targets = [derived[key]["splits"]["train"] for key in ("audio", "text", "targets")]
    scales = derived["targets"]["scales"].to(args.device).float()
    protected = runner.protected_hash(original)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "scripts/inspect_audio_text_fit.py", root / "scripts/train_audio_text_dynamics.py",
             root / "kinetalk_b0/models/audio_text_affect.py", root / "kinetalk_b0/label_guided_intensity.py",
             root / "scripts/audit_projection_mean_dynamic_tradeoff.py"]
    recipe = {"schema": SCHEMA, "source_run": str(run), "source_recipe_sha256": canonical_hash(source_recipe),
        "source_run_hashes": source_hashes, "source_inventory_verified": inventory,
        "derived_sha256": source_recipe["derived_sha256"], "source_sha256": {str(p): sha(p) for p in paths},
        "subset_seed": SUBSET_SEED, "subset_indices": ids.tolist(), "clip_id": [train["q"]["clip_id"][i] for i in ids],
        "fit_clips": 128, "epochs": 8, "batch_size": 16, "steps": 64, "training_seed": 46, "evaluation_noise_seed": 42,
        "budget_scope": "Fixed128 subset diagnostic:64 optimizer steps per arm; not budget-matched to full2315-clip training1160 steps",
        "renderer_lr": 1e-5, "encoder_lr": 1e-4, "loss_weights": runner.LOSS_WEIGHTS, "decode_steps": 12,
        "amplitude_intervention": DELTA, "target_conditioned_diagnostic_only": True, "deployed_audio_claim": False,
        "development_used": False, "test_loaded": False, "default_replaced": False, "checkpoint_selection_performed": False,
        "oracle_missing_target_rule": "Predicted intensity fallback only where independent anchor makes GT undefined"}
    args.output.mkdir(parents=True)
    for path in paths:
        dest = args.output / "source" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(path.read_bytes())
    save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe)})
    before = evaluate(original, original_encoder, train, audio, text, targets, scales, ids, noise, args.device)
    save_json(args.output / "epoch000_evaluation.json", before)
    results, rng_hashes = {}, {}
    for arm in ARMS:
        started = time.time()
        system, encoder = copy.deepcopy(original), copy.deepcopy(original_encoder)
        renderer_params = runner.configure(system); parameters = renderer_params + list(encoder.parameters())
        optimizer = torch.optim.Adam([{"params": renderer_params, "lr": 1e-5}, {"params": encoder.parameters(), "lr": 1e-4}])
        generator = torch.Generator().manual_seed(46)
        batch_hash, draw_hash = hashlib.sha256(), hashlib.sha256()
        arm_dir = args.output / arm; arm_dir.mkdir()
        steps, epochs = 0, []
        for epoch in range(1, 9):
            losses_sum = {}
            for local_ids in torch.randperm(len(ids), generator=generator).split(16):
                selected = ids[local_ids]
                train_noise, flow_time, _ = draws(generator, len(selected), train["q"]["motion"].shape[1:])
                batch_hash.update(selected.numpy().tobytes()); draw_hash.update(train_noise.numpy().tobytes()); draw_hash.update(flow_time.numpy().tobytes())
                batch = batch_to_device(train, selected, args.device)
                data = runner.slice_inputs(audio, text, targets, selected, args.device)
                total, values = training_loss(system, encoder, batch, data, scales, train_noise.to(args.device), flow_time.to(args.device), arm)
                runner.optimize(total, optimizer, parameters)
                for key, value in values.items(): losses_sum[key] = losses_sum.get(key, 0.) + float(value.detach()) / 8
                steps += 1
            row = {"arm": arm, "epoch": epoch, "steps": steps, "losses": losses_sum,
                   "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": draw_hash.hexdigest()}
            epochs.append(row); print(json.dumps(row), flush=True)
        if steps != 64 or runner.protected_hash(system) != protected: raise RuntimeError("Probe budget/frozen path changed")
        rng_hashes[arm] = {"minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": draw_hash.hexdigest(),
                           "generator_sha256": state_hash({"state": generator.get_state()})}
        payload = {"schema": SCHEMA, "arm": arm, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
            "renderer": system.renderer.state_dict(), "encoder": encoder.state_dict(),
            "renderer_sha256": state_hash(system.renderer.state_dict()), "encoder_sha256": state_hash(encoder.state_dict()),
            "protected_sha256": protected, "optimizer": optimizer.state_dict(), "rng": capture_rng(generator),
            "completed_epochs": 8, "steps": steps, "target_conditioned_diagnostic_only": True, **rng_hashes[arm]}
        checkpoint = arm_dir / "epoch008.pt"; save_checkpoint(checkpoint, payload)
        after = evaluate(system, encoder, train, audio, text, targets, scales, ids, noise, args.device)
        results[arm] = {"epochs": epochs, "final": after, "checkpoint_sha256": sha(checkpoint),
                        "rng_hashes": rng_hashes[arm], "elapsed_seconds": time.time() - started}
        save_json(arm_dir / "evaluation.json", results[arm])
        del optimizer, parameters, renderer_params, system, encoder
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if rng_hashes[ARMS[0]] != rng_hashes[ARMS[1]]: raise RuntimeError("Matched receiver RNG differs")
    for name, digest in source_hashes.items():
        if sha(run / name) != digest: raise RuntimeError("Immutable source run changed")
    save_json(args.output / "summary.json", {"schema": SCHEMA, "recipe": recipe, "initial": before, "arms": results,
        "matched_rng": True, "protected_unchanged": True,
        "interpretation": "Target-conditioned receiving-path diagnostic trained from epoch000. Success is not audio predictability or deployment success. Fixed128 subset64-step training is not budget-matched to the full2315-clip1160-step experiment."})
    save_json(args.output / "output_hashes.json", {str(path.relative_to(args.output)): sha(path)
        for path in args.output.rglob("*") if path.is_file() and path.name != "output_hashes.json"})
    print("AUDIO_TEXT_RECEIVER_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
