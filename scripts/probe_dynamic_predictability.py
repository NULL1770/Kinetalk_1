"""Sentence-disjoint audio diagnostics for four fixed dynamic targets.

Only train.pt is split here; the earlier development/audit clips are never fit.
This probes target predictability, not emotion truth or generated motion quality.
Cached acoustic normalization is inverted before fitting new statistics only on
training sentences. PCA/target scales also use training; students are always fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import LowRateAffectEncoder, NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_feature_probe import fit_feature_stats, prepare_feature_batch
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, optimize, prepare,
    select, sha, write_json,
)

TARGET_NAMES = ("teacher_controls", "smoothed_controls", "residual_pca", "smoothed_residual_pca")
OUTPUT_GAIN = 3.0


def training_only_audio(query, cached_stats, train_ids):
    """Invert the existing affine transform, then refit on this training split."""
    audio = query["audio"]
    mean, std = [torch.as_tensor(cached_stats[key], dtype=audio.dtype, device=audio.device) for key in ("mean", "std")]
    if (mean.shape != (audio.shape[-1],) or std.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Invalid cached acoustic statistics")
    raw = {**query, "audio": torch.where(query["valid"][..., None], audio * std + mean, 0)}
    stats = fit_feature_stats(select(raw, train_ids), "acoustic")
    return prepare_feature_batch(raw, "acoustic", stats), stats


def sentence_split(sentence_ids, seed, heldout_fraction=.2):
    """Metadata-only grouping: the same sentence never crosses the split."""
    sentences = sorted(set(str(value) for value in sentence_ids))
    if len(sentences) < 4 or not 0 < heldout_fraction < 1:
        raise ValueError("At least four sentence groups and a valid fraction are required")
    order = sorted(sentences, key=lambda value: (hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), value))
    count = min(len(sentences) - 2, max(1, round(len(sentences) * heldout_fraction)))
    heldout = set(order[:count])
    train_ids = torch.tensor([i for i, value in enumerate(sentence_ids) if str(value) not in heldout])
    heldout_ids = torch.tensor([i for i, value in enumerate(sentence_ids) if str(value) in heldout])
    return train_ids, heldout_ids


def center_controls(values, weight):
    if values.ndim != 3 or weight.shape != values.shape[:2] or (weight < 0).any() or not (weight.sum(1) > 0).all():
        raise ValueError("Controls require matching nonnegative weights and nonempty clips")
    mask = weight > 0
    if not torch.isfinite(values[mask]).all():
        raise ValueError("Observed controls must be finite")
    clean = torch.where(mask[..., None], values, 0)
    mean = (clean * weight[..., None]).sum(1, keepdim=True) / weight.sum(1)[:, None, None]
    return torch.where(mask[..., None], clean - mean, 0)


def bin_frames(values, valid, stride):
    """Average on original fixed frame bins, preserving partial-bin weights."""
    if stride < 1 or values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("Invalid frame binning inputs")
    clean = torch.where(valid[..., None], values, 0)
    if not torch.isfinite(clean).all():
        raise ValueError("Valid frames must be finite")
    extra = (-values.shape[1]) % stride
    weight = F.pad(valid.to(values.dtype), (0, extra)).reshape(len(values), -1, stride).sum(2)
    binned = F.pad(clean, (0, 0, 0, extra)).reshape(len(values), -1, stride, values.shape[-1]).sum(2)
    return binned / weight.clamp_min(1)[..., None], weight


def smooth_controls(values, weight):
    """Fixed five-bin triangular smoothing, followed by clip centering."""
    centered = center_controls(values, weight)
    kernel = values.new_tensor([1., 2., 3., 2., 1.]).reshape(1, 1, 5)
    denominator = F.conv1d(weight[:, None], kernel, padding=2)
    numerator = F.conv1d((centered * weight[..., None]).transpose(1, 2),
                         kernel.expand(values.shape[-1], -1, -1), padding=2, groups=values.shape[-1])
    return center_controls((numerator / denominator.clamp_min(1)).transpose(1, 2), weight)


def fit_pca_target(residual, valid, channel_mask, train_ids, rank, stride):
    """Fit an unwhitened diagnostic PCA only on observed training channels.

    Missing channels never become measurements. A heldout clip missing one of
    the fitted PCA channels fails explicitly, rather than filling it with zero.
    """
    selected = channel_mask[train_ids].all(0).nonzero(as_tuple=True)[0]
    if len(selected) < rank:
        raise ValueError("Too few commonly observed training channels for PCA rank")
    if not channel_mask[:, selected].all():
        raise ValueError("A heldout clip lacks a fitted PCA channel; masked PCA is not supported")
    binned, weight = bin_frames(residual[:, :, selected], valid, stride)
    centered = center_controls(binned, weight)
    values = centered[train_ids].double().reshape(-1, len(selected))
    sample_weight = weight[train_ids].double().reshape(-1)
    covariance = values.T @ (values * sample_weight[:, None]) / sample_weight.sum()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.arange(len(eigenvalues) - 1, -1, -1, device=eigenvalues.device)
    eigenvalues, eigenvectors = eigenvalues[order].clamp_min(0), eigenvectors[:, order]
    basis = eigenvectors[:, :rank]
    # Deterministic signs make artifacts and refits easier to compare.
    maxima = basis.abs().argmax(0)
    signs = torch.sign(basis[maxima, torch.arange(rank, device=basis.device)]).clamp(min=-1, max=1)
    basis = (basis * signs).to(residual.dtype)
    target = center_controls(centered @ basis, weight)
    state = {"basis": basis, "channel_indices": selected, "eigenvalues": eigenvalues,
             "explained_variance_ratio": eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-15),
             "training_weighted_frames": sample_weight.sum(), "fit_train_indices": train_ids.clone()}
    return target, weight, state


def fit_target_scale(values, weight, train_ids):
    """Per-axis RMS equals std because every clip has weighted zero mean."""
    target, w = values[train_ids], weight[train_ids]
    scale = ((target.square() * w[..., None]).sum((0, 1)) / w.sum()).sqrt().clamp_min(1e-4)
    return scale


def weighted_mse(prediction, target, weight):
    return ((prediction - target).square() * weight[..., None]).sum() / (weight.sum() * target.shape[-1])


def target_metrics(prediction, target, weight, clip_ids, sentence_ids, scale):
    """Normalized errors plus equally weighted per-clip/per-axis correlations."""
    mse = weighted_mse(prediction, target, weight)
    zero = weighted_mse(torch.zeros_like(target), target, weight)
    rows, correlations = [], []
    for index, clip_id in enumerate(clip_ids):
        pred, truth, w = prediction[index], target[index], weight[index]
        pmean = (pred * w[:, None]).sum(0) / w.sum()
        tmean = (truth * w[:, None]).sum(0) / w.sum()
        pc, tc = pred - pmean, truth - tmean
        pvar, tvar = (pc.square() * w[:, None]).sum(0), (tc.square() * w[:, None]).sum(0)
        usable = (pvar > 1e-10) & (tvar > 1e-10)
        corr = (pc * tc * w[:, None]).sum(0)[usable] / (pvar[usable] * tvar[usable]).sqrt()
        correlations.extend(corr.detach().cpu().tolist())
        cmse = float(weighted_mse(pred[None], truth[None], w[None]))
        czero = float(weighted_mse(torch.zeros_like(truth)[None], truth[None], w[None]))
        rows.append({"clip_id": str(clip_id), "sentence_id": str(sentence_ids[index]),
                     "normalized_mse": cmse, "zero_mse": czero, "r2_against_zero": 1 - cmse / czero if czero > 1e-12 else None,
                     "temporal_correlation": float(corr.mean()) if len(corr) else None,
                     "correlation_axes": int(usable.sum()), "weighted_frames": float(w.sum())})
    native = target * scale
    native_energy = float(weighted_mse(torch.zeros_like(native), native, weight))
    return {"n": len(target), "normalized_mse": float(mse), "zero_mse": float(zero),
            "r2_against_zero": float(1 - mse / zero) if float(zero) > 1e-12 else None,
            "temporal_correlation": sum(correlations) / len(correlations) if correlations else None,
            "correlation_axes": len(correlations), "native_target_energy": native_energy,
            "normalized_target_energy": float(zero),
            "target_fraction_beyond_output_bound": float(((target.abs() > 2 * OUTPUT_GAIN) * weight[..., None]).sum() / (weight.sum() * target.shape[-1])),
            "per_clip": rows}


def fresh_student(cfg, seed, input_dim, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        student = LowRateAffectEncoder(cfg, input_dim, motion=False)
    for name in ("global_head", "emotion_classifier", "intensity_classifier"):
        getattr(student, name).requires_grad_(False)
    return student.to(device)


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
    if checkpoint.get("stage") != "audio" or checkpoint.get("audio_source") != "acoustic" or "feature_stats" not in checkpoint:
        raise ValueError("Use the run07 acoustic feature-probe audio checkpoint")
    saved_cfg = checkpoint.get("config", {})
    if any(cfg.get(key) != saved_cfg.get(key) for key in ("data", "model")):
        raise ValueError("Use the checkpoint's effective model/data configuration")
    source_hash = sha(args.data / "train.pt")
    if source_hash != checkpoint.get("provenance", {}).get("train_sha256"):
        raise ValueError("Training cache hash differs from the checkpoint's data224 cache")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.get("device", "cuda"))
    dataset = NeutralAffectDataset(args.data / "train.pt")
    train_cpu, heldout_cpu = sentence_split([q["sentence_id"] for q in dataset.queries], args.seed)
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
        if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
            raise ValueError("Enrollment requires contiguous IDs and equal reference counts")
        reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
        residual = torch.where(observed(reference), reference["motion"] - system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": residual.reshape(len(refs), len(refs[0]), *residual.shape[1:]),
                                             "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        residual = affect_residual(query, select(identities, query["speaker_id"]))
        teacher = system.motion_teacher(residual, query["valid"])
        weight = teacher["control_weight"]
        targets = {"teacher_controls": center_controls(teacher["controls"], weight),
                   "smoothed_controls": smooth_controls(teacher["controls"], weight)}
        targets["residual_pca"], pca_weight, pca = fit_pca_target(residual, query["valid"], query["channel_mask"],
                                                               ids["train"], system.motion_teacher.rank, system.motion_teacher.stride)
        torch.testing.assert_close(weight, pca_weight, rtol=0, atol=0)
        targets["smoothed_residual_pca"] = smooth_controls(targets["residual_pca"], weight)
        scales = {name: fit_target_scale(value, weight, ids["train"]) for name, value in targets.items()}
        times, _ = bin_frames(query["times"][..., None], query["valid"], system.motion_teacher.stride)
    split_report = {}
    for name, indices in (("train", train_cpu), ("heldout", heldout_cpu)):
        rows = [dataset.queries[int(index)] for index in indices]
        counts = {}
        for row in rows:
            key = f"{row['speaker']}/emotion{int(row['emotion_id'])}/level{int(row['intensity_id'])}"
            counts[key] = counts.get(key, 0) + 1
        split_report[name] = {"n": len(rows), "indices": indices.tolist(), "sentence_ids": sorted({str(q["sentence_id"]) for q in rows}),
                              "clip_ids": [q["clip_id"] for q in rows], "cell_counts": counts,
                              "artifact_sha256": {q["clip_id"]: q["metadata"]["artifact_sha256"] for q in rows}}
    student = fresh_student(cfg, args.seed, query["audio"].shape[-1], device)
    initial_hash = state_hash(student.state_dict())
    provenance = {"schema": "dynamic_predictability_probe_v1", "seed": args.seed, "steps_per_target": args.steps,
                  "config": cfg, "checkpoint_sha256": sha(args.checkpoint), "train_cache_sha256": source_hash,
                  "script_sha256": sha(__file__), "frozen_state_sha256": frozen_hash, "initial_audio_state_sha256": initial_hash,
                  "parameter_count": sum(p.numel() for p in student.parameters()),
                  "trainable_parameter_count": sum(p.numel() for p in student.parameters() if p.requires_grad),
                  "split": split_report, "target_scale_fit": "new_training_sentences_only",
                  "input_transform": "invert_cached_audio_stats_then_fit_new_training_sentences_only; run07_feature_stats_not_used",
                  "feature_stats": {k: v.cpu().tolist() if torch.is_tensor(v) else v for k, v in feature_stats.items()},
                  "input_statistics_caveat": "Inverting the original float32 cache transform restores acoustic features up to rounding error; original stats are not fitted again.",
                  "scope": "Fresh audio students; frozen teacher/B0/identity may have seen internal holdout sentences. Prior external development and audit clips are not loaded. Same enrolled people; no full-system generalization claim.",
                  "loss": "one weighted MSE between standardized target and 3*student.controls; no global/semantic loss",
                  "output_gain": OUTPUT_GAIN, "smoothing": "5 low-rate bins, triangular [1,2,3,2,1], valid frame count weighted; recentered",
                  "pca": {"rank": system.motion_teacher.rank, "stride": system.motion_teacher.stride,
                          "channel_indices": pca["channel_indices"].cpu().tolist(),
                          "explained_variance_ratio": float(pca["explained_variance_ratio"]),
                          "fitted_on": "new_training_sentences_only; unscaled observed residual channels; no emotion interpretation"},
                  "comparison_limit": "PCA and teacher coordinates represent different signal content and native energy. Standardized R2 is relative to each target's zero baseline, not directly equivalent emotional quality."}
    write_json(args.output / "provenance.json", provenance)
    report, curves, batch_digests = {}, {}, []
    for name in TARGET_NAMES:
        student = fresh_student(cfg, args.seed, query["audio"].shape[-1], device)
        if state_hash(student.state_dict()) != initial_hash:
            raise RuntimeError("Target arms must start with identical student parameters")
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
                row = {"target": name, "step": step + 1, "loss": float(loss.detach()), "grad_norm": norm, "time": time.time()}
                with (args.output / "training.jsonl").open("a", encoding="utf8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
        student.eval()
        with torch.no_grad():
            prediction = OUTPUT_GAIN * student(query["audio"], query["valid"])["controls"]
        arm_report = {}
        for split_name, indices in ids.items():
            selected = select(query, indices)
            arm_report[split_name] = target_metrics(prediction[indices], target[indices], weight[indices],
                                                   selected["clip_id"], selected["sentence_id"], scales[name])
        batch_digests.append(digest.hexdigest())
        arm_report["minibatch_sha256"] = digest.hexdigest()
        arm_report["initial_audio_state_sha256"] = initial_hash
        arm_report["target_scale"] = scales[name].cpu().tolist()
        report[name] = arm_report
        curves[name] = {"target": targets[name].cpu(), "prediction": (prediction * scales[name]).cpu(),
                        "target_normalized": target.cpu(), "prediction_normalized": prediction.cpu(), "scale": scales[name].cpu()}
        torch.save({"model": student.cpu().state_dict(), "config": cfg, "target": name, "output_gain": OUTPUT_GAIN,
                    "target_scale": scales[name].cpu(), "feature_stats": feature_stats, "cached_audio_stats": dataset.cache["audio_stats"],
                    "seed": args.seed, "initial_audio_state_sha256": initial_hash}, args.output / (name + ".pt"))
        write_json(args.output / "summary.json", report)
    if len(set(batch_digests)) != 1 or state_hash(system.state_dict()) != frozen_hash or any(p.grad is not None for p in system.parameters()):
        raise RuntimeError("Matched batches or frozen teacher/B0/identity invariant failed")
    torch.save({"targets": curves, "weight": weight.cpu(), "control_times": times[..., 0].cpu(),
                "train_indices": train_cpu, "heldout_indices": heldout_cpu, "clips": query["clip_id"],
                "sentences": query["sentence_id"], "pca": {k: v.cpu() for k, v in pca.items()}}, args.output / "curves.pt")
    report["checks"] = {"identical_initial_students": True, "identical_minibatches": True,
                        "frozen_teacher_b0_identity_unchanged": True, "new_student_sentence_disjoint": True}
    write_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
