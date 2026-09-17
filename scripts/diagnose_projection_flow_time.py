"""Read-only fixed-training-sample flow-time diagnosis for six saved adapters.

No gradients, optimizer, saved-rollout loading, checkpoint selection, or held-out
motion access. Same-fit rollouts are generated only for paired diagnostics.
Regions are metric groups only; the model is unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from scripts.train_formal_predictable_projection import canonical_hash, frozen_hash, save_json
from scripts.train_projection_schedule_ablation import ARMS, SCHEMA, read_allowlist
from scripts.prepare_teacher_schedule_probe import take_cache_split
from scripts.train_predictable_renderer import (
    PredictableAudioHead, audio_activity_gate, audio_features, basic_metrics, batch_to_device,
    cached_flow, projected_affect, sha, state_hash,
)


TIMES = (0., .25, .5, .75, .9)
SEEDS = (42, 123, 2026)
EMOTIONS = (0, 1, 5, 6)
REGIONS = {"upper_expression": [5, 6, 12, 13, 41, 42, 43, 44, 45], "mouth": list(range(14, 41))}


def select_metadata_ids(query, per_emotion=16, seed=20260922):
    clips = list(map(str, query["clip_id"]))
    labels = query["emotion_id"].tolist()
    if len(clips) != len(labels) or len(set(clips)) != len(clips):
        raise ValueError("Duplicate or inconsistent fit metadata")
    selected = []
    for emotion in EMOTIONS:
        pool = [i for i, value in enumerate(labels) if int(value) == emotion]
        pool.sort(key=lambda i: (hashlib.sha256(f"{seed}:{clips[i]}".encode()).hexdigest(), clips[i]))
        if len(pool) < per_emotion:
            raise ValueError(f"Emotion{emotion} requires{per_emotion}fit clips")
        selected.extend(pool[:per_emotion])
    return torch.tensor(selected, dtype=torch.long)


def error_sums(prediction, target, valid, channel_mask, channels):
    indices = torch.as_tensor(channels, device=prediction.device)
    error = (prediction[..., indices] - target[..., indices]).double().square()
    mask = valid[..., None] & channel_mask[:, None, indices]
    if not torch.isfinite(error[mask]).all():
        raise ValueError("Nonfinite observed flow error")
    return float(error[mask].sum()), int(mask.sum())


def rms_sums(values, weight):
    weight = weight.double()
    mask = weight > 0
    values = torch.where(mask[..., None], values.double(), 0)
    return float((values.square() * weight[..., None]).sum()), float(weight.sum()) * values.shape[-1]


def common_recipe(recipe):
    value = copy.deepcopy(recipe)
    for key in ("arm", "output"):
        value["args"].pop(key, None)
    return value


@torch.no_grad()
def diagnose(args):
    recipes, checkpoints, source_hashes = {}, {}, {}
    for arm in ARMS:
        run = getattr(args, arm)
        provenance_path = run / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf8"))
        recipe = provenance["recipe"]
        if recipe.get("schema") != SCHEMA or recipe["args"]["arm"] != arm or canonical_hash(recipe) != provenance["recipe_sha256"]:
            raise ValueError("Schedule provenance/schema mismatch")
        if any(recipe.get(flag) is not False for flag in ("outer280_loaded", "new_identity439_loaded", "test_loaded")):
            raise ValueError("Schedule source includes forbidden outer targets")
        recipes[arm] = recipe
        source_hashes[arm] = {"provenance": sha(provenance_path)}
        checkpoints[arm] = {}
        for epoch, name in ((2, "epoch002_auxiliary.pt"), (18, "final_epoch018.pt")):
            path = run / name
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("recipe") != recipe or checkpoint.get("recipe_sha256") != canonical_hash(recipe) or checkpoint.get("completed_epochs") != epoch:
                raise ValueError("Fixed checkpoint epoch/recipe mismatch")
            checkpoints[arm][epoch] = checkpoint
            source_hashes[arm][str(epoch)] = sha(path)
    first = recipes[ARMS[0]]
    if any(common_recipe(value) != common_recipe(first) for value in recipes.values()):
        raise ValueError("Schedule recipes differ beyond arm/output")
    paths = {key: Path(first["args"][key]) for key in first["input_sha256"]}
    for key, path in paths.items():
        if sha(path) != first["input_sha256"][key]:
            raise ValueError(f"Changed source {key}")
    lock = json.loads(paths["split_lock"].read_text(encoding="utf8"))
    if lock.get("source_role") != "formal_training_queries_only":
        raise ValueError("Fit source must be formal-training queries only")
    # mmap avoids faulting heldout tensor pages. Only train/internal fields
    # below are indexed; byte-hashing does not interpret validation targets.
    cache = torch.load(paths["cache"], map_location="cpu", weights_only=False, mmap=True)
    bundle = torch.load(paths["bundle"], map_location="cpu", weights_only=False, mmap=True)
    split, train = cache["splits"]["train"], bundle["bundles"]["internal"]
    fit_ids = read_allowlist(paths["fit_ids"])
    if split["q"]["clip_id"] != fit_ids or train["clip_id"] != fit_ids or lock["fit_clip_ids"] != fit_ids:
        raise ValueError("Internal fit clip order differs")
    if bundle["provenance"]["internal_split_lock_sha256"] != canonical_hash(lock):
        raise ValueError("Fit bundle not bound to internal split")
    weights = torch.load(paths["weights"], map_location="cpu", weights_only=False)
    if weights["provenance"]["bundle_sha256"] != first["input_sha256"]["bundle"]:
        raise ValueError("Fixed basis source differs")
    state = weights["states"]["rrr_rank8"]
    if not torch.equal(state["train_ids"], torch.arange(len(fit_ids))):
        raise ValueError("Fixed head not fitted to exact internal fit set")
    cfg = yaml.safe_load(paths["config"].read_text(encoding="utf8"))
    original = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    system = NeutralAffectSystem(cfg).to(args.device).eval()
    system.load_state_dict(original["model"], strict=True)
    freeze_module(system)
    del original
    head = PredictableAudioHead(state, train["motion_bins"], train["weight"]).to(args.device).eval()
    freeze_module(head)
    backbone_hash, head_hash = frozen_hash(system), state_hash(head.state_dict())
    selected = select_metadata_ids(split["q"])
    selected_reference = take_cache_split(split, selected)
    selected_clips = [fit_ids[int(i)] for i in selected]
    selected_metadata = [{"clip_id": fit_ids[int(i)], "sentence_id": str(split["q"]["sentence_id"][int(i)]),
        "speaker": str(split["q"]["speaker"][int(i)]), "emotion_id": int(split["q"]["emotion_id"][i])} for i in selected]
    features = audio_features(train)
    weight = train["weight"].float()
    shape = (len(selected), *split["q"]["motion"].shape[1:])
    noise = {seed: torch.randn(shape, generator=torch.Generator().manual_seed(seed)) for seed in SEEDS}
    noise_hashes = {str(seed): state_hash({"noise": values}) for seed, values in noise.items()}
    populations = {"all": list(range(len(selected))),
        "neutral": [i for i, row in enumerate(selected_metadata) if row["emotion_id"] == 0],
        "nonneutral": [i for i, row in enumerate(selected_metadata) if row["emotion_id"] != 0]}
    rows = {}
    for arm in ARMS:
        rows[arm] = {}
        for epoch in (2, 18):
            checkpoint = checkpoints[arm][epoch]
            if checkpoint["frozen_state_sha256"] != backbone_hash or checkpoint["head_sha256"] != head_hash or state_hash(checkpoint["head"]) != head_hash:
                raise ValueError("Checkpoint backbone/head differs from fixed inputs")
            system.local_projection.load_state_dict(checkpoint["local_projection"], strict=True)
            projection = system.local_projection.weight.detach().cpu().double()
            norm = {"frobenius": float(projection.norm()), "spectral": float(torch.linalg.matrix_norm(projection, 2))}
            totals, amplitude = {}, {}
            rollout = {seed: {mode: [] for mode in ("full", "zero", "oracle")} for seed in SEEDS}
            for start in range(0, len(selected), args.batch_size):
                local_ids = torch.arange(start, min(start + args.batch_size, len(selected)))
                ids = selected[local_ids]
                batch = batch_to_device(split, ids, args.device)
                w = weight[ids].to(args.device)
                controls = {"audio": head(features[ids].to(args.device), w),
                    "teacher": head.teacher(train["motion_bins"][ids].float().to(args.device), w)}
                gate = audio_activity_gate(batch["affect"])
                for mode, z in controls.items():
                    fields = {"controls": z, "controls_gated": z * gate[:, None, None],
                        "projected_controls": system.local_projection(z),
                        "projected_controls_gated": system.local_projection(z * gate[:, None, None])}
                    projected = projected_affect(system, batch, z, w, audio_gate=True)["local"]
                    fields["renderer_local_gated"] = projected
                    for field, values in fields.items():
                        ww = batch["q"]["valid"].float() if field == "renderer_local_gated" else w
                        for population, pop_ids in populations.items():
                            keep = torch.tensor([int(i) in pop_ids for i in local_ids], device=args.device)
                            if keep.any():
                                sums = rms_sums(values[keep], ww[keep])
                                key = mode, field, population
                                old = amplitude.get(key, (0., 0.))
                                amplitude[key] = (old[0] + sums[0], old[1] + sums[1])
                q = batch["q"]
                for seed in SEEDS:
                    eps = noise[seed][local_ids].to(args.device)
                    for mode in ("full", "zero", "oracle"):
                        z = controls["teacher" if mode == "oracle" else "audio"]
                        affect = projected_affect(system, batch, z, w, zero=mode == "zero", audio_gate=True)
                        prediction = system.generate(q["content"], q["valid"], batch["identity"], affect,
                            initial_noise=eps, steps=12, base=batch["base"])["motion"]
                        rollout[seed][mode].append(prediction.cpu())
                    for time_value in TIMES:
                        time_tensor = torch.full((len(ids),), time_value, device=args.device)
                        for mode, z in controls.items():
                            out = cached_flow(system, batch, z, w, eps, time_tensor, audio_gate=True)
                            regions = {"all_observed": list(range(q["motion"].shape[-1])), **REGIONS}
                            for population, pop_ids in populations.items():
                                keep = torch.tensor([int(i) in pop_ids for i in local_ids], device=args.device)
                                if not keep.any():
                                    continue
                                for region, channels in regions.items():
                                    value = error_sums(out["prediction"][keep], out["velocity_target"][keep], q["valid"][keep], q["channel_mask"][keep], channels)
                                    key = seed, time_value, mode, population, region
                                    old = totals.get(key, (0., 0))
                                    totals[key] = (old[0] + value[0], old[1] + value[1])
            per_seed, averaged = {}, {}
            for seed in SEEDS:
                per_seed[str(seed)] = {str(t): {mode: {pop: {region: totals[(seed, t, mode, pop, region)][0] / totals[(seed, t, mode, pop, region)][1]
                    for region in ("all_observed", *REGIONS)} for pop in populations} for mode in controls} for t in TIMES}
            for t in TIMES:
                averaged[str(t)] = {mode: {pop: {region: {
                    "scaled_velocity_mse": float(np.mean([per_seed[str(seed)][str(t)][mode][pop][region] for seed in SEEDS])),
                    "native_motion_velocity_mse": float(np.mean([per_seed[str(seed)][str(t)][mode][pop][region] for seed in SEEDS])) * system.residual_scale ** 2,
                    "observed_values_per_noise": totals[(SEEDS[0], t, mode, pop, region)][1]}
                    for region in ("all_observed", *REGIONS)} for pop in populations} for mode in controls}
            closed_loop = {}
            for seed in SEEDS:
                closed_loop[str(seed)] = {}
                for mode, predictions in rollout[seed].items():
                    metrics = basic_metrics(torch.cat(predictions), selected_reference)
                    closed_loop[str(seed)][mode] = {pop: {region: {
                        kind: {key: metrics[pop][region][kind][key] for key in ("native_mse", "r2_against_zero", "pooled_centered_correlation")}
                        for kind in ("raw_motion", "centered_residual")}
                        for region in ("upper_expression", "brows", "eyes_expression", "mouth")} for pop in populations}
            closed_loop_mean = {mode: {pop: {region: {kind: {
                key: float(np.mean([closed_loop[str(seed)][mode][pop][region][kind][key] for seed in SEEDS]))
                for key in ("native_mse", "r2_against_zero", "pooled_centered_correlation")}
                for kind in ("raw_motion", "centered_residual")}
                for region in ("upper_expression", "brows", "eyes_expression", "mouth")} for pop in populations}
                for mode in ("full", "zero", "oracle")}
            rows[arm][str(epoch)] = {"checkpoint_sha256": source_hashes[arm][str(epoch)], "projection_norm": norm,
                "condition_rms": {mode: {field: {pop: float(np.sqrt(amplitude[(mode, field, pop)][0] / amplitude[(mode, field, pop)][1]))
                    for pop in populations} for field in ("controls", "controls_gated", "projected_controls", "projected_controls_gated", "renderer_local_gated")}
                    for mode in ("audio", "teacher")}, "flow_mse_noise_mean": averaged, "flow_scaled_mse_per_noise": per_seed,
                "same_fit_clips_closed_loop": {"decode_steps": 12, "noise_mean_points": closed_loop_mean, "per_noise": closed_loop,
                    "aggregation": "Arithmetic mean of each per-noise point metric, including correlation; not correlation pooled over noise",
                    "scope": "Same64 fitting clips and exact same initial noise as flow-time probes; no heldout/generalization interpretation"}}
            if frozen_hash(system) != backbone_hash or state_hash(head.state_dict()) != head_hash or any(p.grad is not None for p in system.parameters()):
                raise RuntimeError("Read-only/no-gradient invariant failed")
            print(json.dumps({"arm": arm, "epoch": epoch, "projection_norm": norm,
                "audio_upper_flow_mse_by_t": {str(t): averaged[str(t)]["audio"]["nonneutral"]["upper_expression"]["scaled_velocity_mse"] for t in TIMES}}), flush=True)
    report = {"schema": "projection_flow_time_diagnostic_v1", "source_sha256": sha(__file__),
        "input_sha256": first["input_sha256"], "checkpoint_sources": source_hashes,
        "fit_only_probe": True, "selection": {"metadata_hash_seed": 20260922, "clips_per_emotion": 16,
            "indices": selected.tolist(), "metadata": selected_metadata, "selected_clip_order_sha256": canonical_hash(selected_clips)},
        "noise_seeds": list(SEEDS), "noise_tensor_sha256": noise_hashes, "times": list(TIMES),
        "populations": {pop: len(ids) for pop, ids in populations.items()}, "rows": rows,
        "residual_scale": system.residual_scale,
        "units_note": "native_motion_velocity_mse = scaled velocity MSE * residual_scale^2; derivative is w.r.t. flow time, not seconds of facial motion",
        "scope": "Same fixed64 fitting clips, same noise for all arms/checkpoints/times and twelve-step closed-loop rollout. Training-state versus rollout diagnostic on identical fit distribution; no heldout/generalization claim or CIs treating noise as independent samples",
        "gradient_or_parameter_update": False, "heldout_target_values_accessed": False,
        "outer280_loaded": False, "new_identity439_loaded": False, "test_loaded": False,
        "closed_loop_curves_loaded": False, "same_fit_closed_loop_generated": True, "large_curves_saved": False,
        "checkpoint_selection_changed": False}
    save_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError("Fresh output and positive batch size required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    diagnose(args)


if __name__ == "__main__":
    main()
