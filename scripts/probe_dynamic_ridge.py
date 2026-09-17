"""Nested sentence-disjoint ridge diagnostic for the scalar upper-face field.

There is no new model path or renderer update. Binned audio/content features
are centered per clip using inputs alone. Three inner sentence folds select
ridge alpha; every feature scale is fit on that fold's training inputs only.
The outer heldout is evaluated once after alpha selection, without calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.probe_dynamic_predictability import (
    bin_frames, center_controls, fit_target_scale, sentence_split, weighted_mse,
)
from scripts.probe_expression_energy_field import energy_metrics, fit_expression_energy_target
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, prepare, select, sha,
    write_json,
)


DEFAULT_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def expression_only_field(residual, valid, channel_mask, stride):
    """Scalar label from squint/wide/brows, excluding gaze and blink nuisance.

    Same abs-after-binning convention as upper_l1. A supervision definition,
    not a generator channel split or a psychological emotion ground truth.
    """
    selected = [5, 6, 12, 13, 41, 42, 43, 44, 45]
    if not channel_mask[:, selected].all():
        raise ValueError("Expression target needs observed selected controllers")
    binned, weight = bin_frames(residual[:, :, selected], valid, stride)
    return center_controls(binned.abs().mean(-1, keepdim=True), weight), weight


def sentence_folds(sentence_ids, outer_train_ids, seed=45, n_folds=3):
    """Return global index pairs using outer-training sentence metadata only."""
    ids = [int(i) for i in outer_train_ids]
    groups = sorted({str(sentence_ids[i]) for i in ids})
    if n_folds < 2 or len(groups) < n_folds:
        raise ValueError("Each inner fold needs at least one distinct training sentence")
    ordered = sorted(groups, key=lambda s: (hashlib.sha256(
        f"inner:{seed}:{s}".encode()).hexdigest(), s))
    pairs = []
    for fold in range(n_folds):
        validation = set(ordered[fold::n_folds])
        pairs.append((torch.tensor([i for i in ids if str(sentence_ids[i]) not in validation]),
                      torch.tensor([i for i in ids if str(sentence_ids[i]) in validation])))
    return pairs


def centered_features(frames, valid, stride):
    """Float64 binning and clip centering, with no dataset-fitted transform."""
    values, weight = bin_frames(frames.double(), valid, stride)
    return center_controls(values, weight), weight


def ridge_path(features, target, weight, train_ids, alphas):
    """Fit all alphas with one eigendecomposition of the weighted mean Gram.

    Features must already have weighted zero mean per clip. RMS is then the
    training standard deviation; no global intercept/mean is fitted. Targets
    are kept in native units, so target statistics cannot affect alpha fitting.
    """
    if target.shape[-1] != 1 or features.shape[:2] != target.shape[:2]:
        raise ValueError("Ridge diagnostic requires aligned inputs and scalar targets")
    if not alphas or any(not float(a) > 0 for a in alphas):
        raise ValueError("Ridge alphas must be positive")
    ids = torch.as_tensor(train_ids, device=features.device)
    x, y, w = features[ids].double(), target[ids].double(), weight[ids].double()
    if not len(ids) or not (w.sum() > 0):
        raise ValueError("Training data must contain observed bins")
    if not torch.isfinite(x[w > 0]).all() or not torch.isfinite(y[w > 0]).all():
        raise ValueError("Observed training inputs and targets must be finite")
    x = torch.where(w[..., None] > 0, x, 0)
    y = torch.where(w[..., None] > 0, y, 0)
    std = ((x.square() * w[..., None]).sum((0, 1)) / w.sum()).sqrt().clamp_min(1e-8)
    x = (x / std).reshape(-1, x.shape[-1])
    y, w = y.reshape(-1, 1), w.reshape(-1)
    gram = x.T @ (x * w[:, None]) / w.sum()
    rhs = x.T @ (y * w[:, None]) / w.sum()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    # Tiny negative values are roundoff in a positive-semidefinite Gram.
    eigenvalues = eigenvalues.clamp_min(0)
    projected = eigenvectors.T @ rhs
    coefficients = [eigenvectors @ (projected / (eigenvalues[:, None] + float(a)))
                    for a in alphas]
    return coefficients, std


def predict(features, weight, coefficients, std):
    """A single linear scalar readout; restore exact weighted clip centering."""
    return center_controls((features.double() / std) @ coefficients, weight.double())


def nested_select_alpha(features, target, weight, sentence_ids, train_ids,
                        alphas=DEFAULT_ALPHAS, seed=45, n_folds=3):
    """Select alpha without reading any outer heldout feature or target."""
    alphas = tuple(float(a) for a in alphas)
    squared_errors = [0.0 for _ in alphas]
    total_weight = 0.0
    fold_rows = []
    for fold, (fit_cpu, validation_cpu) in enumerate(sentence_folds(
            sentence_ids, train_ids, seed, n_folds)):
        fit_ids = fit_cpu.to(features.device)
        validation_ids = validation_cpu.to(features.device)
        coefficients, std = ridge_path(features, target, weight, fit_ids, alphas)
        rows = []
        for j, coefficient in enumerate(coefficients):
            prediction = predict(features[validation_ids], weight[validation_ids], coefficient, std)
            mse = float(weighted_mse(prediction, target[validation_ids], weight[validation_ids]))
            squared_errors[j] += mse * float(weight[validation_ids].sum())
            rows.append({"alpha": alphas[j], "native_mse": mse})
        total_weight += float(weight[validation_ids].sum())
        fold_rows.append({"fold": fold, "fit_indices": fit_cpu.tolist(),
            "validation_indices": validation_cpu.tolist(),
            "fit_sentences": sorted({str(sentence_ids[int(i)]) for i in fit_cpu}),
            "validation_sentences": sorted({str(sentence_ids[int(i)]) for i in validation_cpu}),
            "feature_std_fit_indices": fit_cpu.tolist(), "scores": rows})
    scores = [{"alpha": a, "inner_native_mse": error / total_weight}
              for a, error in zip(alphas, squared_errors)]
    # Stable first-minimum choice is determined solely by the inner folds.
    best = min(range(len(alphas)), key=lambda j: scores[j]["inner_native_mse"])
    return alphas[best], {"folds": fold_rows, "scores": scores,
                        "aggregation": "pooled observed-bin weight across sentence-disjoint folds"}


def intervene_features(features, weight, mode):
    """Mask-safe heldout input interventions; no target values are used."""
    altered = torch.zeros_like(features)
    if mode == "zero":
        return altered
    if mode not in ("reverse", "shuffle"):
        raise ValueError("Intervention must be zero, reverse, or shuffle")
    if mode == "shuffle" and len(features) < 2:
        raise ValueError("Shuffle requires at least two heldout clips")
    for i in range(len(features)):
        destination = (weight[i] > 0).nonzero(as_tuple=True)[0]
        if mode == "reverse":
            altered[i, destination] = features[i, destination.flip(0)]
        else:
            source_idx = (i + 1) % len(features)
            source = features[source_idx, weight[source_idx] > 0]
            positions = torch.linspace(0, len(source) - 1, len(destination),
                                       device=features.device).round().long()
            altered[i, destination] = source[positions]
    return center_controls(altered, weight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mode", choices=("audio_only", "content_only"), default="audio_only")
    parser.add_argument("--target", choices=("upper_l1", "upper_expression_l1", "jaw17_motion"), default="upper_l1")
    parser.add_argument("--split-seed", type=int, default=45)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "audio":
        raise ValueError("Checkpoint must be an audio-stage checkpoint")
    cache_hash = sha(args.data / "train.pt")
    if cache_hash != checkpoint.get("provenance", {}).get("train_sha256"):
        raise ValueError("Training cache differs from checkpoint provenance")
    torch.set_num_threads(4)
    device = torch.device(cfg.get("device", "cuda"))
    dataset = NeutralAffectDataset(args.data / "train.pt")
    train_cpu, heldout_cpu = sentence_split([q["sentence_id"] for q in dataset.queries], args.split_seed)
    train_ids, heldout_ids = train_cpu.to(device), heldout_cpu.to(device)
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    frozen_hash = state_hash(system.state_dict())
    with torch.no_grad():
        query, _ = prepare(system, dataset, device)
        if query["audio"].shape[-1] != 768 or query["content"].shape[-1] != 768:
            raise ValueError("This probe requires 768-D emotion2vec and content caches")
        refs = dataset.identity_references
        if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
            raise ValueError("Contiguous speaker IDs and equal reference counts required")
        reference = device_batch([r for s in sorted(refs) for r in refs[s]], device)
        ref_residual = torch.where(observed(reference), reference["motion"] -
            system.base(reference["content"], reference["valid"])["b0"], 0)
        identity = cached_identity(system, {"residual": ref_residual.reshape(
            len(refs), len(refs[0]), *ref_residual.shape[1:]),
            "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        residual = affect_residual(query, select(identity, query["speaker_id"]))
        channels = (list(range(14)) + list(range(41, 46)) if args.target == "upper_l1" else [17])
        if not query["channel_mask"][:, channels].all():
            raise ValueError("Target requires all selected channels to be observed")
        stride = system.motion_teacher.stride
        if args.target == "upper_l1":
            target, weight, _ = fit_expression_energy_target(residual, query["valid"],
                query["channel_mask"], query["speaker_id"], query["emotion_id"],
                query["intensity_id"], train_ids, stride, mode="upper_l1")
        elif args.target == "upper_expression_l1":
            target, weight = expression_only_field(residual, query["valid"], query["channel_mask"], stride)
        else:
            target, weight = centered_features(query["motion"][:, :, 17:18], query["valid"], stride)
        key = "audio" if args.mode == "audio_only" else "content"
        features, feature_weight = centered_features(query[key], query["valid"], stride)
        target, weight = target.double(), weight.double()
        if not torch.equal(weight, feature_weight):
            raise RuntimeError("Input and target bin weights differ")
        alpha, inner = nested_select_alpha(features, target, weight, query["sentence_id"],
            train_cpu, args.alphas, args.split_seed, args.inner_folds)
        coefficients, std = ridge_path(features, target, weight, train_ids, [alpha])
        coefficient = coefficients[0]
        prediction = predict(features, weight, coefficient, std)
        # Outer-train scale is only for reporting; the native target was fit above.
        scale = fit_target_scale(target, weight, train_ids)
        local_ids = torch.arange(len(heldout_ids), device=device)
        heldout_query = select(query, heldout_ids)
        interventions = {}
        for mode in ("zero", "reverse", "shuffle"):
            altered = predict(intervene_features(features[heldout_ids], weight[heldout_ids], mode),
                              weight[heldout_ids], coefficient, std)
            interventions[mode] = energy_metrics(altered / scale, target[heldout_ids] / scale,
                weight[heldout_ids], heldout_query, local_ids, scale)
            interventions[mode]["prediction_change_mse"] = float(weighted_mse(
                altered, prediction[heldout_ids], weight[heldout_ids]))
        if state_hash(system.state_dict()) != frozen_hash:
            raise RuntimeError("Frozen main model state changed")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "dynamic_ridge_probe_v1", "mode": args.mode, "target": args.target,
        "split_seed": args.split_seed, "stride": stride, "selected_alpha": alpha,
        "inner_validation": inner, "train": energy_metrics(prediction / scale, target / scale,
            weight, query, train_ids, scale),
        "heldout": energy_metrics(prediction / scale, target / scale, weight, query, heldout_ids, scale),
        "heldout_input_interventions": interventions, "main_model_unchanged": True,
        "input_dim": int(features.shape[-1]), "output_dim": 1, "bias": False,
        "dtype": "float64", "alpha_convention": "weighted mean Gram + alpha I",
        "source_sha256": {"script": sha(Path(__file__))}}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "provenance.json", {"schema": summary["schema"],
        "checkpoint_sha256": sha(args.checkpoint), "train_cache_sha256": cache_hash,
        "main_model_sha256": frozen_hash, "split_seed": args.split_seed,
        "mode": args.mode, "target": args.target,
        "train_indices": train_cpu.tolist(), "heldout_indices": heldout_cpu.tolist(),
        "train_sentences": sorted({str(query["sentence_id"][int(i)]) for i in train_cpu}),
        "heldout_sentences": sorted({str(query["sentence_id"][int(i)]) for i in heldout_cpu}),
        "feature_std": std.cpu().tolist(), "feature_std_fit_indices": train_cpu.tolist(),
        "target_scale": scale.cpu().tolist(), "selected_alpha": alpha,
        "selection": f"{args.inner_folds} inner training-sentence folds; no outer-heldout fitting or selection",
        "cache_normalization": "positive affine cache transforms cancel after clip centering and inner-training standardization",
        "target_note": ("jaw motion channel17 positive control, no residual subtraction"
                        if args.target == "jaw17_motion" else
                        "Scalar expression controller proxy of frozen B0/identity residual; not ground-truth emotion"),
        "inner_validation": inner})
    effective = {**cfg, "probe": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output / "effective_config.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True), encoding="utf8")
    torch.save({"schema": summary["schema"], "mode": args.mode, "target": args.target,
        "coefficient": coefficient.cpu(), "feature_std": std.cpu(),
        "native_coefficient": (coefficient / std[:, None]).cpu(),
        "target_scale": scale.cpu(), "stride": stride, "alpha": alpha,
        "preprocessing": "bin frames, weighted clip-center, divide feature_std; scalar native output"},
        args.output / "weights.pt")
    print(json.dumps({k: summary[k] for k in ("mode", "target", "selected_alpha", "split_seed")}
        | {s: {k: summary[s][k] for k in ("r2_against_zero", "temporal_correlation")}
           for s in ("train", "heldout")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
