"""Compare a label-associated expression basis with rank3 motion PCA.

Training labels only define one fixed full-channel subspace. Fresh audio-only
students predict its centered low-rate trajectories; no class label enters the
student, and no generator is changed. This is a proxy-target diagnostic, not a
claim that label-associated motion has been causally disentangled.
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.probe_dynamic_predictability import (
    OUTPUT_GAIN, bin_frames, center_controls, fit_pca_target, fit_target_scale,
    fresh_student, sentence_split, target_metrics, training_only_audio, weighted_mse,
)
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, optimize, prepare,
    select, sha, write_json,
)

SPLIT_SEED = 45
RANK = 3
TARGET_NAMES = ("label_expression", "unsupervised_pca3")


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _json(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json(item) for item in value]
    return value


def fit_label_basis(residual, valid, channel_mask, speaker_id, emotion_id, intensity_id, train_ids, stride=4):
    """Fit three neutral-anchored label directions with equal cell weighting.

    Frames average within clip, clips within speaker/emotion/level, levels
    within speaker/emotion, and speakers within emotion. The three resulting
    directions are NOT centered again: their origin is the neutral anchor.
    Heldout labels never participate in this fit.
    """
    if residual.ndim != 3 or valid.shape != residual.shape[:2] or channel_mask.shape != (len(residual), residual.shape[-1]):
        raise ValueError("Invalid motion/frame/channel mask shapes")
    train_ids = train_ids.to(residual.device)
    selected = channel_mask[train_ids].all(0).nonzero(as_tuple=True)[0]
    if len(selected) < RANK or not channel_mask[:, selected].all():
        raise ValueError("Insufficient common training channels or heldout missing fitted channels")
    clean = torch.where(valid[..., None], residual[:, :, selected], 0)
    if not valid.any(1).all() or not torch.isfinite(clean).all():
        raise ValueError("Each clip needs finite observed residual frames")
    means = clean.double().sum(1) / valid.sum(1)[:, None]
    speakers = sorted(int(value) for value in speaker_id[train_ids].unique())
    emotions = sorted(int(value) for value in emotion_id[train_ids].unique() if int(value) != 0)
    if len(emotions) != RANK:
        raise ValueError("The planned diagnostic requires exactly three nonneutral training emotions")
    neutral_means, speaker_directions, counts = {}, {}, []
    for speaker in speakers:
        train_speaker = train_ids[speaker_id[train_ids] == speaker]
        neutral = train_speaker[emotion_id[train_speaker] == 0]
        if not len(neutral):
            raise ValueError(f"Training speaker {speaker} lacks a neutral query anchor")
        neutral_means[speaker] = means[neutral].mean(0)
        counts.append({"speaker_id": speaker, "emotion_id": 0, "intensity_id": None, "n": len(neutral), "indices": neutral.cpu().tolist()})
    directions = []
    for emotion in emotions:
        emotion_train = train_ids[emotion_id[train_ids] == emotion]
        levels = sorted(int(value) for value in intensity_id[emotion_train].unique())
        if any(level < 0 for level in levels):
            raise ValueError("Label basis requires valid nonneutral intensity levels")
        per_speaker = []
        for speaker in speakers:
            level_directions = []
            for level in levels:
                members = train_ids[(speaker_id[train_ids] == speaker) & (emotion_id[train_ids] == emotion) & (intensity_id[train_ids] == level)]
                if not len(members):
                    raise ValueError(f"Missing planned training cell speaker={speaker}, emotion={emotion}, level={level}")
                level_directions.append(means[members].mean(0) - neutral_means[speaker])
                counts.append({"speaker_id": speaker, "emotion_id": emotion, "intensity_id": level, "n": len(members), "indices": members.cpu().tolist()})
            per_speaker.append(torch.stack(level_directions).mean(0))
        speaker_directions[emotion] = torch.stack(per_speaker)
        directions.append(speaker_directions[emotion].mean(0))
    directions = torch.stack(directions)
    _, singular, vh = torch.linalg.svd(directions, full_matrices=False)
    if float(singular[-1]) <= max(float(singular[0]) * 1e-7, 1e-10):
        raise ValueError("The three label directions are rank deficient; do not invent extra axes")
    basis = vh.T
    peak = basis.abs().argmax(0)
    signs = basis[peak, torch.arange(RANK, device=basis.device)].sign()
    basis = (basis * signs).to(residual.dtype)
    binned, weight = bin_frames(clean, valid, stride)
    # Neutral mean subtraction would be algebraically canceled by clip
    # centering. It defines the basis origin above, not a forced zero target.
    target = center_controls(binned @ basis, weight)
    state = {"basis": basis, "channel_indices": selected, "directions": directions,
             "singular_values": singular, "emotion_ids": emotions, "speaker_ids": speakers,
             "neutral_means": {str(key): value for key, value in neutral_means.items()},
             "per_speaker_emotion_directions": {str(key): value for key, value in speaker_directions.items()},
             "training_cells": counts, "fit_train_indices": train_ids.clone(),
             "clip_mean_projection": means.to(basis.dtype) @ basis}
    return target, weight, state


def basis_diagnostics(basis, channel_indices, target, weight, train_ids):
    """Region names annotate the learned basis; they never alter fitting/loss."""
    energy = (target[train_ids].square() * weight[train_ids, :, None]).sum((0, 1)) / weight[train_ids].sum()
    regions = {"eyes": range(14), "mouth_jaw": range(14, 41), "brows": range(41, 46), "other": range(46, 52)}
    rows = []
    for axis in range(basis.shape[-1]):
        values = basis[:, axis]
        top = values.abs().argsort(descending=True)[:8]
        rows.append({"axis": axis + 1, "training_dynamic_energy": float(energy[axis]),
                     "squared_loading_mass": {name: float(values[[i for i, original in enumerate(channel_indices) if int(original) in region]].square().sum()) for name, region in regions.items()},
                     "top_loadings": [{"channel_index": int(channel_indices[i]), "loading": float(values[i])} for i in top]})
    return {"axes": rows, "training_dynamic_energy": float(energy.sum()),
            "energy_weighted_squared_loading_mass": {name: sum(row["training_dynamic_energy"] * row["squared_loading_mass"][name] for row in rows) / float(energy.sum()) if float(energy.sum()) > 0 else None for name in regions}}


def grouped_metrics(prediction, target, weight, query, indices, scale):
    groups = {"all": indices, "neutral": indices[query["emotion_id"][indices] == 0],
              "nonneutral": indices[query["emotion_id"][indices] != 0]}
    for emotion in sorted(int(value) for value in query["emotion_id"][indices].unique()):
        groups[f"emotion_{emotion}"] = indices[query["emotion_id"][indices] == emotion]
    result = {}
    for group, ids in groups.items():
        if not len(ids):
            continue
        q = select(query, ids)
        summary = target_metrics(prediction[ids], target[ids], weight[ids], q["clip_id"], q["sentence_id"], scale)
        native_prediction, native_target = prediction[ids] * scale, target[ids] * scale
        mse = weighted_mse(native_prediction, native_target, weight[ids])
        zero = weighted_mse(torch.zeros_like(native_target), native_target, weight[ids])
        summary["native_mse"] = float(mse)
        summary["native_r2_against_zero"] = float(1 - mse / zero) if float(zero) > 1e-15 else None
        summary["native_prediction_energy"] = float(weighted_mse(torch.zeros_like(native_prediction), native_prediction, weight[ids]))
        summary["by_axis"] = [{"axis": axis + 1, **target_metrics(prediction[ids, :, axis:axis + 1], target[ids, :, axis:axis + 1], weight[ids], q["clip_id"], q["sentence_id"], scale[axis:axis + 1])} for axis in range(RANK)]
        # Avoid repeated per-clip lists while retaining the aggregate per-axis values.
        for axis in summary["by_axis"]:
            axis.pop("per_clip")
        result[group] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "audio" or checkpoint.get("audio_source") != "acoustic":
        raise ValueError("Use the frozen run07 acoustic audio checkpoint")
    if any(cfg.get(key) != checkpoint.get("config", {}).get(key) for key in ("data", "model")):
        raise ValueError("Use run07 effective_config.yaml; only the fresh student rank changes")
    cache_hash = sha(args.data / "train.pt")
    if checkpoint.get("provenance", {}).get("train_sha256") != cache_hash:
        raise ValueError("Source cache differs from the checkpoint's data224 cache")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.get("device", "cuda"))
    dataset = NeutralAffectDataset(args.data / "train.pt")
    train_cpu, heldout_cpu = sentence_split([q["sentence_id"] for q in dataset.queries], SPLIT_SEED)
    ids = {"train": train_cpu.to(device), "heldout": heldout_cpu.to(device)}
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    frozen_hash = state_hash(system.state_dict())
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        query, _ = prepare(system, dataset, device)
        query, feature_stats = training_only_audio(query, dataset.cache["audio_stats"], ids["train"])
        refs = dataset.identity_references
        if sorted(refs) != list(range(len(refs))) or len({len(value) for value in refs.values()}) != 1:
            raise ValueError("Enrollment requires contiguous IDs and equal reference counts")
        reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
        ref_residual = torch.where(observed(reference), reference["motion"] - system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": ref_residual.reshape(len(refs), len(refs[0]), *ref_residual.shape[1:]),
                                             "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        residual = affect_residual(query, select(identities, query["speaker_id"]))
        label_target, weight, label_basis = fit_label_basis(residual, query["valid"], query["channel_mask"], query["speaker_id"],
                                                           query["emotion_id"], query["intensity_id"], ids["train"], system.motion_teacher.stride)
        pca_target, pca_weight, pca_basis = fit_pca_target(residual, query["valid"], query["channel_mask"], ids["train"], RANK, system.motion_teacher.stride)
        torch.testing.assert_close(weight, pca_weight, rtol=0, atol=0)
        torch.testing.assert_close(label_basis["channel_indices"], pca_basis["channel_indices"], rtol=0, atol=0)
        if residual.shape[-1] == 52 and len(label_basis["channel_indices"]) != 51:
            raise ValueError("The planned real-data probe requires all 51 observed channels")
        targets = {"label_expression": label_target, "unsupervised_pca3": pca_target}
        bases = {"label_expression": label_basis, "unsupervised_pca3": pca_basis}
        scales = {name: fit_target_scale(value, weight, ids["train"]) for name, value in targets.items()}
        control_times, _ = bin_frames(query["times"][..., None], query["valid"], system.motion_teacher.stride)
    student_cfg = copy.deepcopy(cfg)
    student_cfg["model"]["affect_rank"] = RANK
    student = fresh_student(student_cfg, args.seed, query["audio"].shape[-1], device)
    initial_hash = state_hash(student.state_dict())
    splits = {}
    for name, indices in (("train", train_cpu), ("heldout", heldout_cpu)):
        rows = [dataset.queries[int(index)] for index in indices]
        splits[name] = {"n": len(rows), "indices": indices.tolist(), "sentences": sorted({q["sentence_id"] for q in rows}),
                        "clips": [q["clip_id"] for q in rows], "artifact_sha256": {q["clip_id"]: q["metadata"]["artifact_sha256"] for q in rows}}
    diagnostics = {name: basis_diagnostics(bases[name]["basis"], bases[name]["channel_indices"], target, weight, ids["train"]) for name, target in targets.items()}
    provenance = {"schema": "label_expression_dynamics_probe_v1", "seed": args.seed, "split_seed": SPLIT_SEED,
                  "steps_per_target": args.steps, "rank": RANK, "output_gain": OUTPUT_GAIN,
                  "original_config": cfg, "student_config": student_cfg, "split": splits,
                  "checkpoint_sha256": sha(args.checkpoint), "train_cache_sha256": cache_hash,
                  "source_sha256": {"probe_label_expression_dynamics.py": sha(__file__), "probe_dynamic_predictability.py": sha(Path(__file__).with_name("probe_dynamic_predictability.py")),
                                    "neutral_affect.py": sha(Path(__file__).resolve().parents[1] / "kinetalk_b0/models/neutral_affect.py")},
                  "frozen_state_sha256": frozen_hash, "initial_audio_state_sha256": initial_hash,
                  "parameter_count": sum(p.numel() for p in student.parameters()), "trainable_parameter_count": sum(p.numel() for p in student.parameters() if p.requires_grad),
                  "target_fit": "Only training clips: within-speaker neutral clip mean anchor, equal clip/level/speaker weighting per nonneutral emotion, uncentered rank3 SVD of three neutral-relative directions. No regional weighting or channel split.",
                  "student_input": "Original 83D acoustic features only; invert old cache normalization, then fit mean/std only on new training sentences. No emotion/intensity/identity label input.",
                  "feature_stats": _json(feature_stats), "target_scaling": "Each target axis RMS from training only; identical gain and dynamic MSE across arms",
                  "temporal_target": "Stride4 means of residual projected into fixed basis, subtract per-clip valid-weighted temporal mean. Neutral clips keep their observed dynamics.",
                  "scope": "No generator changes. Fresh rank3 TCN. Frozen B0/identity/teacher may have seen these sentences; no complete-system heldout claim. Identity correction does not prove causal identity/emotion/content disentanglement.",
                  "basis_diagnostics": diagnostics, "label_fit": _json({key: value for key, value in label_basis.items() if key != "clip_mean_projection"}),
                  "pca_explained_training_variance": float(pca_basis["explained_variance_ratio"])}
    write_json(args.output / "provenance.json", provenance)
    (args.output / "effective_student_config.yaml").write_text(yaml.safe_dump(student_cfg, sort_keys=False), encoding="utf8")
    report, curves, batch_hashes = {}, {}, []
    for name in TARGET_NAMES:
        student = fresh_student(student_cfg, args.seed, query["audio"].shape[-1], device)
        if state_hash(student.state_dict()) != initial_hash:
            raise RuntimeError("Student arms must start with identical parameters")
        target = targets[name] / scales[name]
        params = [p for p in student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(cfg["training"].get("audio_lr", .0005)), weight_decay=1e-5)
        rng, digest = torch.Generator(device="cpu").manual_seed(args.seed), hashlib.sha256()
        torch.manual_seed(args.seed)
        student.train()
        for step in range(args.steps):
            sampled = torch.randint(len(train_cpu), (int(cfg["training"].get("batch_size", 8)),), generator=rng)
            digest.update(sampled.numpy().tobytes())
            batch = ids["train"][sampled.to(device)]
            predicted = OUTPUT_GAIN * student(query["audio"][batch], query["valid"][batch])["controls"]
            loss = weighted_mse(predicted, target[batch], weight[batch])
            norm = optimize(loss, optimizer, params)
            if step % 200 == 0 or step + 1 == args.steps:
                record = {"target": name, "step": step + 1, "loss": float(loss.detach()), "grad_norm": norm, "time": time.time()}
                with (args.output / "training.jsonl").open("a", encoding="utf8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)
        student.eval()
        with torch.no_grad():
            prediction = OUTPUT_GAIN * student(query["audio"], query["valid"])["controls"]
        report[name] = {split: grouped_metrics(prediction, target, weight, query, indices, scales[name]) for split, indices in ids.items()}
        report[name].update(minibatch_sha256=digest.hexdigest(), initial_audio_state_sha256=initial_hash,
                            target_scale=scales[name].cpu().tolist(), basis_diagnostics=diagnostics[name])
        batch_hashes.append(digest.hexdigest())
        curves[name] = {"target": targets[name].cpu(), "prediction": (prediction * scales[name]).cpu(),
                        "target_normalized": target.cpu(), "prediction_normalized": prediction.cpu(), "scale": scales[name].cpu()}
        torch.save({"model": student.cpu().state_dict(), "student_config": student_cfg, "target": name, "target_scale": scales[name].cpu(),
                    "output_gain": OUTPUT_GAIN, "basis": _cpu(bases[name]), "feature_stats": feature_stats, "cached_audio_stats": dataset.cache["audio_stats"],
                    "seed": args.seed, "initial_audio_state_sha256": initial_hash}, args.output / (name + ".pt"))
        write_json(args.output / "summary.json", report)
    if len(set(batch_hashes)) != 1 or state_hash(system.state_dict()) != frozen_hash or any(p.grad is not None for p in system.parameters()):
        raise RuntimeError("Matched-batch or frozen-system invariant failed")
    torch.save({"targets": curves, "bases": _cpu(bases), "weight": weight.cpu(), "control_times": control_times[..., 0].cpu(),
                "train_indices": train_cpu, "heldout_indices": heldout_cpu, "clips": query["clip_id"], "sentences": query["sentence_id"],
                "emotion_id": query["emotion_id"].cpu(), "intensity_id": query["intensity_id"].cpu(), "speaker_id": query["speaker_id"].cpu()}, args.output / "curves.pt")
    report["checks"] = {"identical_initial_students": True, "identical_minibatches": True, "frozen_teacher_b0_identity_unchanged": True,
                        "new_student_sentence_disjoint": True, "labels_only_fit_training_basis": True, "no_ground_truth_label_student_input": True}
    write_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
