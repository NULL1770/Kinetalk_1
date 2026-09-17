"""Eight-epoch training-only OOF test of a 64/192-parameter ridge correction."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.predictable_motion import predict_motion, weighted_clip_center
from kinetalk_b0.temporal_motion_refiner import ConstrainedRidgeRefiner, fit_control_scale
from scripts.audit_predictable_motion_predictions import clip_statistics
from scripts.audit_temporal_audio_refiner_probe import make_report
from scripts.probe_scaled_motion_basis import motion_groups, write_json
from scripts.probe_predictable_motion import intervene_input
from scripts.train_predictable_renderer import audio_features, sha, state_hash
from scripts.train_temporal_audio_refiner_probe import (
    atomic_save, buffer_hash, load_sources, native_loss, predict_batches,
)

ARMS = ("pointwise", "temporal")
SCHEMA = "constrained_audio_refiner_oof_v1"
DEVIATION_WEIGHT = 10.0
LR = .002


def train_arm(model, x, y, weight, fit_ids, *, seed, device, output, provenance,
              fold, arm):
    """Read only fold-fit observations; fixed epoch 8, no checkpoint selection."""
    model = model.to(device)
    xf = x.index_select(0, fit_ids).to(device)
    wf = weight.index_select(0, fit_ids).double().to(device)
    yf = weighted_clip_center(y.index_select(0, fit_ids), wf.cpu()).to(device)
    before = buffer_hash(model)
    with torch.no_grad():
        reference = model(xf, wf)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(seed)
    batch_hash = hashlib.sha256()
    output.mkdir(parents=True, exist_ok=False)
    history, step, started = [], 0, time.time()
    for epoch in range(1, 9):
        order = torch.randperm(len(fit_ids), generator=generator)
        batch_hash.update(fit_ids[order].numpy().tobytes())
        totals = torch.zeros(3, dtype=torch.float64)
        for ids in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xf[ids], wf[ids])
            fit_loss = native_loss(prediction, yf[ids], wf[ids])
            deviation = native_loss(prediction, reference[ids], wf[ids])
            loss = fit_loss + DEVIATION_WEIGHT * deviation
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gradients")
            optimizer.step()
            n = float(wf[ids].sum()) * y.shape[-1]
            totals += torch.tensor([float(fit_loss.detach()) * n,
                                    float(deviation.detach()) * n, n], dtype=torch.float64)
            step += 1
        if buffer_hash(model) != before:
            raise ValueError("Frozen ridge buffers changed")
        row = {"fold": fold, "arm": arm, "epoch": epoch, "step": step,
               "online_native_mse": float(totals[0] / totals[2]) * .25 ** 2,
               "online_ridge_deviation_native_mse": float(totals[1] / totals[2]) * .25 ** 2,
               "batch_order_sha256": batch_hash.hexdigest(),
               "elapsed_seconds": time.time() - started}
        history.append(row)
        atomic_save(output / "last.pt", {"schema": SCHEMA, "provenance": provenance,
            "fold": fold, "arm": arm, "epoch": epoch, "step": step,
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "generator_rng": generator.get_state(),
            "frozen_buffer_sha256": before, "fit_ids": fit_ids,
            "batch_order_sha256": batch_hash.hexdigest(), "history": history,
            "checkpoint_selection": "fixed epoch8 only, no held-out selection"})
        write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
    return model, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "weights", "basis-study", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh output required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(46)
    bundle, channels, speakers, folds, saved, previous, hashes, binding = load_sources(
        args.bundle, args.weights, args.basis_study)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "kinetalk_b0/temporal_motion_refiner.py",
             root / "scripts/train_temporal_audio_refiner_probe.py",
             root / "scripts/audit_temporal_audio_refiner_probe.py",
             root / "kinetalk_b0/predictable_motion.py",
             root / "docs/CONSTRAINED_AUDIO_REFINER_PROBE.md"]
    for path in paths:
        dest = args.output / "source" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    provenance = {"schema": SCHEMA, "args": {k: str(v.resolve()) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, "source_sha256": {str(p): sha(p) for p in paths},
        "input_sha256": hashes, "prior_oof_output_sha256": binding,
        "epochs": 8, "batch_size": 32, "seed": "46+fold", "optimizer": "Adam",
        "lr": LR, "deviation_weight": DEVIATION_WEIGHT,
        "regularization_selection": "fixed a priori; no OOF or inner validation tuning",
        "loss": "native centered motion MSE + 10 * native squared departure from frozen ridge; both divided by .25**2",
        "parameters": {"pointwise": 64, "temporal": 192},
        "native_rank": 8, "alpha": 1., "audio_gate": False,
        "source_roles_accessed": ["bundles.internal"],
        "heldout405_targets_accessed": False, "outer280_targets_accessed": False,
        "new439_targets_accessed": False, "test_targets_accessed": False,
        "renderer_trained": False, "default_changed": False,
        "motion_channel_indices": channels,
        "scope": "Descriptive existing-training three sentence-fold OOF; fixed rank/alpha already chosen previously. Not an independent test or generative diversity experiment."}
    write_json(args.output / "provenance.json", provenance)
    x, y, w = audio_features(bundle), bundle["motion_bins"][..., channels], bundle["weight"]
    target = torch.zeros_like(previous["target"])
    predictions = {"ridge": {m: previous["predictions"]["native"][m].clone()
        for m in ("full", "reverse", "zero", "metric_projection_oracle")}}
    for arm in ARMS:
        predictions[arm] = {m: torch.zeros_like(target) for m in ("full", "reverse")}
    details, reproduction = [], {}
    seen = torch.zeros(len(y), dtype=torch.long)
    for fold, (fit, val) in enumerate(folds):
        state = copy.deepcopy(saved["states"][f"fold{fold}_native"]["transformed_model"])
        scale = fit_control_scale(y, w, fit, state)
        rows, models = {}, {}
        expected = predict_motion(x[fit[:8]], w[fit[:8]], state)
        for arm in ARMS:
            model = ConstrainedRidgeRefiner(state, scale, temporal=arm == "temporal")
            assert sum(p.numel() for p in model.parameters()) == provenance["parameters"][arm]
            with torch.no_grad():
                torch.testing.assert_close(model(x[fit[:8]], w[fit[:8]]), expected, rtol=0, atol=0)
            models[arm], rows[arm] = train_arm(model, x, y, w, fit,
                seed=46 + fold, device=args.device, output=args.output / f"fold{fold}_{arm}",
                provenance=provenance, fold=fold, arm=arm)
            models[arm].cpu()
        if [r["batch_order_sha256"] for r in rows["pointwise"]] != [r["batch_order_sha256"] for r in rows["temporal"]]:
            raise ValueError("Arms saw different batches")
        target[val] = weighted_clip_center(y[val], w[val])
        torch.testing.assert_close(target[val], previous["target"][val], rtol=0, atol=0)
        torch.testing.assert_close(predict_motion(x[val], w[val], state), predictions["ridge"]["full"][val], rtol=0, atol=0)
        for arm in ARMS:
            ck = torch.load(args.output / f"fold{fold}_{arm}/last.pt", weights_only=False, map_location="cpu")
            loaded = ConstrainedRidgeRefiner(state, scale, temporal=arm == "temporal")
            before = buffer_hash(loaded)
            loaded.load_state_dict(ck["model"], strict=True)
            if buffer_hash(loaded) != before or before != ck["frozen_buffer_sha256"]:
                raise ValueError("Frozen state changed in checkpoint")
            models[arm].to(args.device); loaded.to(args.device)
            errors = {}
            for mode, features in (("full", x[val]), ("reverse", intervene_input(x[val], w[val], "reverse"))):
                prediction = predict_batches(models[arm], features, w[val], args.device)
                repeat = predict_batches(loaded, features, w[val], args.device)
                torch.testing.assert_close(repeat, prediction, rtol=0, atol=0)
                predictions[arm][mode][val] = prediction
                errors[mode] = float((repeat - prediction).abs().max())
            reproduction[f"fold{fold}_{arm}"] = errors
            models[arm].cpu()
        seen[val] += 1
        details.append({"fold": fold, "fit_ids": fit.tolist(), "validation_ids": val.tolist(),
                        "history": rows, "control_scale": scale.tolist()})
    if not torch.all(seen == 1):
        raise ValueError("OOF coverage failed")
    native_groups, groups = motion_groups(channels)
    stats = {a: {m: {g: clip_statistics(pred, target, w, cc) for g, cc in groups.items()}
                    for m, pred in modes.items()} for a, modes in predictions.items()}
    curves = {"provenance": provenance, "target": target, "weight": w.double(),
        "predictions": predictions, "statistics": stats, "groups": native_groups,
        "clip_id": bundle["clip_id"], "sentence_id": bundle["sentence_id"],
        "emotion_id": bundle["emotion_id"], "speaker_id": speakers,
        "source_speaker_id": bundle["speaker_id"], "oof_fold": previous["oof_fold"], "folds": details}
    torch.save(curves, args.output / "oof_predictions.pt")
    write_json(args.output / "folds.json", details)
    report = {"schema": SCHEMA + "_audit", "checkpoint_reproduction_max_abs_error": reproduction,
              **make_report(curves)}
    write_json(args.output / "paired_audit.json", report)
    write_json(args.output / "output_hashes.json", {str(p.relative_to(args.output)): sha(p)
        for p in args.output.rglob("*") if p.is_file() and p.name != "output_hashes.json"})
    print(json.dumps({"generation_entry_pass": report["generation_entry_pass"],
        "generation_entry_checks": report["generation_entry_checks"],
        "nonneutral_temporal_vs_ridge": {g: report["comparisons"]["temporal__vs__ridge"]["populations"]["nonneutral"][g]["r2_improvement"]
            for g in ("brows", "eyes_expression", "mouth")},
        "positive_brow_people": report["positive_brow_people"]}), flush=True)
    print("CONSTRAINED_REFINER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
