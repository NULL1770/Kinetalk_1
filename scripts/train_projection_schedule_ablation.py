"""Fixed-budget teacher-schedule intervention on an internal speaker holdout.

Consumes a remapped training-only cache/bundle and a basis refitted solely on
its fit identities. Original outer development/test data must be absent.
Only the existing512-parameter projection trains; epoch18 is the prespecified
comparison and epoch2 is an auxiliary diagnostic, never checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    canonical_hash, capture_rng, evaluate_compact, frozen_hash, mean_metric,
    save_checkpoint, save_json, teacher_probability,
)
from scripts.train_predictable_renderer import (
    PredictableAudioHead, audio_features, batch_to_device, cached_flow,
    configure_trainable, observed, optimize, sha, state_hash, validate_inputs,
)


ARMS = ("constant_teacher", "audio_only", "decay_teacher")
SCHEMA = "projection_schedule_ablation_v1"


def schedule_probability(arm, epoch, batch_index, batches):
    if arm == "constant_teacher":
        return .5
    if arm == "audio_only":
        return 0.
    if arm == "decay_teacher":
        return teacher_probability(epoch, batch_index, batches, 5)
    raise ValueError(f"Unknown schedule arm: {arm}")


def speaker_names(query):
    if "speaker" in query:
        return list(map(str, query["speaker"]))
    result = []
    for clip in query["clip_id"]:
        values = str(clip).split("_")
        if len(values) < 3 or values[0] not in ("mead", "cremad"):
            raise ValueError("Cannot establish speaker identity from metadata")
        result.append("_".join(values[:2]))
    return result


def read_allowlist(path):
    """IDs are clip IDs, never ambiguous row positions after bundle remapping."""
    value = json.loads(Path(path).read_text(encoding="utf8"))
    if isinstance(value, dict):
        value = value.get("clip_ids")
    if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
        raise ValueError("ID JSON must contain a nonempty clip-ID list or {'clip_ids': [...]} ")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate clip IDs in allowlist")
    return value


def validate_internal_split(cache, bundle, split_lock, fit_ids, validation_ids):
    """Require explicit training-only lineage; same sentences are permitted."""
    train, dev = cache["splits"]["train"]["q"], cache["splits"]["validation"]["q"]
    if split_lock.get("schema") != "projection_schedule_internal_split_v1":
        raise ValueError("Need explicit internal-only split lineage schema")
    if split_lock.get("source_role") != "formal_training_queries_only":
        raise ValueError("Internal experiment must derive solely from previous formal training queries")
    if split_lock.get("outer_development_loaded") is not False or split_lock.get("new_identity_development_loaded") is not False:
        raise ValueError("Outer/new-identity development must be excluded")
    for rows, expected, lock_key in ((train, fit_ids, "fit_clip_ids"), (dev, validation_ids, "validation_clip_ids")):
        actual = list(map(str, rows["clip_id"]))
        if actual != expected or actual != split_lock.get(lock_key) or len(actual) != len(set(actual)):
            raise ValueError(f"Remapped cache/allowlist/split lock differs: {lock_key}")
    if set(fit_ids) & set(validation_ids):
        raise ValueError("Fit and heldout clips overlap")
    fit_speakers, heldout_speakers = set(speaker_names(train)), set(speaker_names(dev))
    if fit_speakers & heldout_speakers or len(heldout_speakers) != 3:
        raise ValueError("Require exactly three heldout identities and speaker-disjoint fit")
    if sorted(heldout_speakers) != sorted(split_lock.get("heldout_speakers", [])):
        raise ValueError("Heldout speaker metadata differs")
    if bundle["provenance"].get("internal_split_lock_sha256") != canonical_hash(split_lock):
        raise ValueError("Bundle not bound to internal split metadata")
    return {"fit_clips": len(fit_ids), "validation_clips": len(validation_ids),
        "fit_speakers": sorted(fit_speakers), "heldout_speakers": sorted(heldout_speakers),
        "shared_sentence_count": len(set(train["sentence_id"]) & set(dev["sentence_id"])),
        "scope": "Audio head/U/scale and interface fit identities excluded from heldout; inherited B0/global may have seen them"}


def draws(generator, length, shape):
    """Every arm consumes every draw, including teacher choice for audio-only."""
    noise = torch.randn((length, *shape), generator=generator)
    flow_time = torch.rand(length, generator=generator)
    flow_time[torch.rand(length, generator=generator) < .2] = 0
    choose = torch.rand(length, generator=generator)
    return noise, flow_time, choose


def diagnostics(reports):
    result = {}
    for population in ("nonneutral", "neutral", "all"):
        result[population] = {}
        for region in ("upper_expression", "brows", "eyes_expression", "mouth"):
            values = {mode: mean_metric(reports, mode, population, region, "centered_residual", "r2_against_zero")
                      for mode in next(iter(reports.values())) if mode in ("full", "zero", "reverse", "oracle")}
            values["full_minus_zero_r2"] = values["full"] - values["zero"]
            result[population][region] = values
    return result


def curve_binding(path, checkpoint_path, recipe, cache_hash):
    sidecar = path.with_name(path.stem + ".provenance.json")
    value = {"schema": "projection_schedule_curves_provenance_v1", "curve_sha256": sha(path),
        "checkpoint_sha256": sha(checkpoint_path), "recipe_sha256": canonical_hash(recipe), "cache_sha256": cache_hash}
    save_json(sidecar, value)
    return {"path": str(sidecar.resolve()), "sha256": sha(sidecar)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "bundle", "weights", "checkpoint", "config", "output", "fit-ids", "validation-ids", "split-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seed != 46 or args.epochs != 18 or args.lr != 2e-4 or args.batch_size != 16:
        raise ValueError("Prespecified comparison: seed46,18epochs,batch16,Adam lr=.0002")
    if args.eval_batch_size < 1:
        raise ValueError("Positive evaluation batch size required")
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
    split_lock = json.loads(args.split_lock.read_text(encoding="utf8"))
    data_scope = validate_internal_split(cache, bundle, split_lock, fit_ids, validation_ids)
    for name in ("checkpoint", "config", "bundle"):
        if cache["provenance"].get(name + "_sha256") != hashes[name]:
            raise ValueError(f"Cache {name} provenance mismatch")
    if fitted["provenance"].get("bundle_sha256") != hashes["bundle"]:
        raise ValueError("RRR/PCA basis must be refitted on remapped fit-only bundle")
    if cfg["data"]["emotion_classes"][0] != "neutral":
        raise ValueError("Audio gate requires neutral class zero")
    source_paths = [Path(__file__), Path(__file__).with_name("train_formal_predictable_projection.py"),
        Path(__file__).with_name("train_predictable_renderer.py"), Path(__file__).with_name("emotion_ray_metrics.py"),
        Path(__file__).with_name("train_neutral_affect_pilot.py"), Path(__file__).with_name("train_neutral_affect_audio_ablation.py"),
        Path(__file__).parents[1] / "kinetalk_b0/models/neutral_affect.py", Path(__file__).parents[1] / "kinetalk_b0/models/dit.py",
        Path(__file__).parents[1] / "kinetalk_b0/models/model.py", Path(__file__).parents[1] / "kinetalk_b0/models/encoders.py",
        Path(__file__).parents[1] / "kinetalk_b0/predictable_motion.py",
        Path(__file__).parents[1] / "kinetalk_b0/utils.py"]
    recipe = {"schema": SCHEMA, "args": {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_sha256": hashes, "source_sha256": {str(path.resolve()): sha(path) for path in source_paths},
        "data_scope": data_scope, "trainable": ["local_projection.weight"], "loss": "observed_flow_mse_only",
        "audio_activity_gate": True, "head_basis_scale_frozen": True, "torch": str(torch.__version__),
        "checkpoint_selection": "none; final epoch18 primary, epoch2 predeclared auxiliary only",
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "inherited_exposure_note": "This isolates new basis/head/interface fit identities, not historical B0/global training"}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "provenance.json", {"recipe": recipe, "recipe_sha256": canonical_hash(recipe)})
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    configure_trainable(system, seed=args.seed, projection_only=True)
    if [name for name, p in system.named_parameters() if p.requires_grad] != ["local_projection.weight"] or system.local_projection.weight.numel() != 512:
        raise ValueError("Only the existing512-parameter projection may train")
    tr, dev = bundle["bundles"]["internal"], bundle["bundles"]["external_dev"]
    state = fitted["states"]["rrr_rank8"]
    head = PredictableAudioHead(state, tr["motion_bins"], tr["weight"]).to(args.device).eval()
    freeze_module(head)
    frozen_before, head_before = frozen_hash(system), state_hash(head.state_dict())
    params = list(system.local_projection.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    features, weight = audio_features(tr), tr["weight"].float()
    with torch.no_grad():
        teacher = head.teacher(tr["motion_bins"].float().to(args.device), weight.to(args.device)).cpu()
    train_split, validation = cache["splits"]["train"], cache["splits"]["validation"]
    batches = math.ceil(len(features) / args.batch_size)
    shape = train_split["q"]["motion"].shape[1:]
    batch_hash, noise_hash, choice_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    started, step, completed_epochs = time.time(), 0, 0

    def payload():
        return {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
            "local_projection": system.local_projection.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": capture_rng(generator), "step": step, "completed_epochs": completed_epochs,
            "frozen_state_sha256": frozen_before, "head_sha256": head_before,
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
            "default_enabled": False, "selection": "none"}

    save_checkpoint(args.output / "last.pt", payload())
    auxiliary_binding = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(features), generator=generator)
        losses, teacher_count = [], 0
        source_errors = {"teacher": [0., 0.], "audio": [0., 0.]}
        for batch_index, ids in enumerate(order.split(args.batch_size)):
            noise, flow_time, choose = draws(generator, len(ids), shape)
            batch_hash.update(ids.numpy().tobytes()); noise_hash.update(noise.numpy().tobytes()); noise_hash.update(flow_time.numpy().tobytes())
            choice_hash.update(choose.numpy().tobytes())
            probability = schedule_probability(args.arm, epoch, batch_index, batches)
            use_teacher = choose < probability
            b, w = batch_to_device(train_split, ids, args.device), weight[ids].to(args.device)
            with torch.no_grad():
                audio = head(features[ids].to(args.device), w)
                controls = torch.where(use_teacher.to(args.device)[:, None, None], teacher[ids].to(args.device), audio)
            result = cached_flow(system, b, controls, w, noise.to(args.device), flow_time.to(args.device), audio_gate=True)
            error = (result["prediction"] - result["velocity_target"]).square()
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
        completed_epochs = epoch + 1
        if frozen_hash(system) != frozen_before or state_hash(head.state_dict()) != head_before:
            raise RuntimeError("Frozen renderer/B0/identity/global/head changed")
        save_checkpoint(args.output / "last.pt", payload())
        evaluation, report_binding = None, None
        if completed_epochs % 2 == 0:
            auxiliary = completed_epochs == 2
            curve_path = args.output / "epoch002_diagnostic_curves.pt" if auxiliary else None
            if auxiliary:
                save_checkpoint(args.output / "epoch002_auxiliary.pt", payload())
            evaluation = evaluate_compact(system, head, validation, dev, device=args.device,
                batch_size=args.eval_batch_size, modes=("full", "zero", "reverse", "oracle") if auxiliary else ("full", "zero"),
                curves_path=curve_path)
            if auxiliary:
                auxiliary_binding = curve_binding(curve_path, args.output / "epoch002_auxiliary.pt", recipe, hashes["cache"])
                report_binding = auxiliary_binding
            save_json(args.output / f"development_epoch{completed_epochs:03d}.json",
                {"diagnostics": diagnostics(evaluation), "noise_reports": evaluation, "curve_provenance": report_binding,
                 "checkpoint_selection_performed": False})
        record = {"arm": args.arm, "epoch": completed_epochs, "step": step, "samples_seen": len(features),
            "mean_flow_loss": float(np.mean(losses)), "teacher_fraction": teacher_count / len(features),
            "flow_mse_by_condition_source": {name: values[0] / values[1] if values[1] else None for name, values in source_errors.items()},
            "flow_source_comparison_note": "Teacher/audio minibatches use different conditioning; pooled training loss is not an identical objective-distribution comparison across schedules",
            "teacher_probability_last_batch": schedule_probability(args.arm, epoch, batches - 1, batches),
            "elapsed_seconds": time.time() - started, "diagnostics": diagnostics(evaluation) if evaluation else None,
            "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
            "teacher_choice_draw_sha256": choice_hash.hexdigest()}
        save_json(args.output / f"epoch{completed_epochs:03d}.json", record)
        print(json.dumps(record), flush=True)
    save_checkpoint(args.output / "final_epoch018.pt", payload())
    curves_path = args.output / "final_epoch018_curves.pt"
    final = evaluate_compact(system, head, validation, dev, device=args.device, batch_size=args.eval_batch_size,
        modes=("full", "zero", "reverse", "oracle"), curves_path=curves_path)
    binding = curve_binding(curves_path, args.output / "final_epoch018.pt", recipe, hashes["cache"])
    save_json(args.output / "summary.json", {"schema": SCHEMA, "recipe_sha256": canonical_hash(recipe),
        "arm": args.arm, "completed_epochs": completed_epochs, "optimizer_steps": step, "final": final,
        "final_diagnostics": diagnostics(final), "curve_provenance": binding, "epoch002_auxiliary_curve_provenance": auxiliary_binding,
        "frozen_unchanged": frozen_hash(system) == frozen_before, "head_unchanged": state_hash(head.state_dict()) == head_before,
        "minibatch_sha256": batch_hash.hexdigest(), "noise_time_sha256": noise_hash.hexdigest(),
        "teacher_choice_draw_sha256": choice_hash.hexdigest(), "elapsed_seconds": time.time() - started,
        "checkpoint_selection_performed": False, "outer280_loaded": False, "new_identity439_loaded": False,
        "test_loaded": False, "default_replaced": False})
    print("COMPLETE fixed18-epoch internal identity teacher-schedule intervention", flush=True)


if __name__ == "__main__":
    main()
