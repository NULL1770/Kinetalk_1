"""Training-internal OOF diagnosis of the metric used by a shared rank8 RRR.

This adds no generator, gate, region head, text input, rank search, or loss.
Only bundles.internal is indexed. The existing three sentence folds are reused;
alpha1/rank8 were previously selected on these data, so this is a descriptive
internal diagnostic, not an untouched or unbiased hyperparameter evaluation.

Scaled fitting uses Y D, D=sqrt(mean-normalized inverse floored channel RMS^2).
Predictions are decoded and divided by D before all reported motion metrics.
The oracle is a target-conditioned metric projection, NOT a guaranteed native
MSE upper bound. D^-1 U is generally not Euclidean-orthonormal and must not be
dropped into the current PredictableAudioHead or treated as a deployment head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.predictable_motion import (
    decode_controls, fit_motion_path, motion_target, predict_motion,
    weighted_clip_center,
)
from scripts.audit_predictable_motion_predictions import (
    clip_statistics, paired_summary, scalar_summary,
)
from scripts.probe_predictable_motion import intervene_input
from scripts.train_predictable_renderer import audio_features, sha, state_hash


ARMS = ("native", "scaled")
MODES = ("full", "reverse", "metric_projection_oracle", "zero")
ALPHA, RANK = 1.0, 8
INTERPRETATION = (
    "Descriptive internal diagnostic: alpha1/rank8 were previously selected on "
    "these training folds; not an untouched test or unbiased hyperparameter "
    "evaluation. No renderer trained or default changed. No audio gate. "
    "All scores use original motion units after inverse scaling. "
    "Metric-projection oracle reads validation targets only after each fold fit; "
    "it is not audio inference or a guaranteed native-MSE bound. The inverse "
    "basis is not generally Euclidean-orthonormal and is incompatible with "
    "unmodified PredictableAudioHead teacher projection."
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf8")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _ids(value, n, name):
    ids = torch.as_tensor(value)
    if (ids.ndim != 1 or ids.dtype == torch.bool or ids.is_floating_point()
            or ids.is_complex() or not len(ids)):
        raise ValueError(f"{name} must be a nonempty integer vector")
    ids = ids.cpu().long()
    if ids.min() < 0 or ids.max() >= n or len(ids.unique()) != len(ids):
        raise ValueError(f"{name} has duplicate/out-of-range indices")
    return ids


def validate_folds(folds, sentence_ids):
    """Require the saved, complete three-fold sentence partition, not a new one."""
    n = len(sentence_ids)
    if len(folds) != 3:
        raise ValueError("Exactly three saved sentence folds are required")
    counts = torch.zeros(n, dtype=torch.long)
    result = []
    for number, fold in enumerate(folds):
        if fold["fold"] != number:
            raise ValueError("Saved folds must be numbered 0,1,2")
        fit = _ids(fold["fit_indices"], n, "fit_indices")
        val = _ids(fold["validation_indices"], n, "validation_indices")
        if sorted(fit.tolist() + val.tolist()) != list(range(n)):
            raise ValueError("Fold fit/validation must form a disjoint complete partition")
        fit_sentences = sorted({str(sentence_ids[i]) for i in fit.tolist()})
        val_sentences = sorted({str(sentence_ids[i]) for i in val.tolist()})
        if set(fit_sentences) & set(val_sentences):
            raise ValueError("Fold has overlapping fit/validation sentences")
        if (fit_sentences != fold["fit_sentences"]
                or val_sentences != fold["validation_sentences"]):
            raise ValueError("Saved fold sentences do not match clip indices")
        counts[val] += 1
        result.append((fit, val))
    if not torch.equal(counts, torch.ones_like(counts)):
        raise ValueError("Every training clip must be validated exactly once")
    return result


def fit_channel_metric(y, weight):
    """Accept fold-fit tensors only; positive-RMS median uses midpoint quantile."""
    centered = weighted_clip_center(y, weight)
    w = weight.detach().cpu().double()
    total = w.sum()
    if not total > 0:
        raise ValueError("Metric fit requires positive training weight")
    rms = ((centered.square() * w[..., None]).sum((0, 1)) / total).sqrt()
    positive = rms[rms > 0]
    if not len(positive) or not torch.isfinite(rms).all():
        raise ValueError("Metric fit requires positive finite motion RMS")
    floor = torch.quantile(positive, .5)
    # Algebraically identical after mean normalization, without inverse-square
    # overflow for very small, otherwise valid training motion amplitudes.
    metric = (floor / rms.clamp_min(floor)).square()
    metric = metric / metric.mean()
    if not torch.isfinite(metric).all() or not (metric > 0).all():
        raise ValueError("Channel metric overflow/underflow")
    return {"rms": rms, "floor": floor, "metric": metric,
            "sqrt_metric": metric.sqrt(), "training_weight": total}


def fit_fold_arm(x, y, weight, fit_ids, arm):
    """Slice before inspecting observations: validation NaNs cannot affect fit."""
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    ids = _ids(fit_ids, len(y), "fit_ids")
    xf = x.index_select(0, ids.to(x.device))
    yf = y.index_select(0, ids.to(y.device)).detach().cpu().double()
    wf = weight.index_select(0, ids.to(weight.device)).detach().cpu().double()
    if arm == "scaled":
        metric = fit_channel_metric(yf, wf)
    else:
        metric = {"metric": torch.ones(y.shape[-1], dtype=torch.float64),
                  "sqrt_metric": torch.ones(y.shape[-1], dtype=torch.float64)}
    model = fit_motion_path(xf, yf * metric["sqrt_metric"], wf,
                            torch.arange(len(ids)), [ALPHA], [RANK],
                            methods=("rrr",))[0]
    if arm == "scaled":
        model.update(target_definition="weighted_clip_center(real_motion * sqrt_metric) @ basis",
                     preprocessing="weighted clip centering; train feature RMS; fold-fit scaled motion coordinates",
                     objective="weighted_mean_scaled_motion_MSE_sum_over_channels + alpha * scaled_coefficient_Frobenius_squared",
                     coordinate_system="transformed motion; must inverse-scale decoded predictions")
    # The nested library state operates in transformed coordinates for scaled.
    # Keep native decoding explicit; never advertise this as a renderer state.
    return {"schema": "scaled_motion_basis_diagnostic_fold_v1", "arm": arm,
            "fit_ids": ids, "metric": metric, "transformed_model": model,
            "inverse_native_basis": model["basis"] / metric["sqrt_metric"][:, None],
            "teacher_analysis_basis": model["basis"] * metric["sqrt_metric"][:, None],
            "deployment_compatible": False}


def predict_native(x, weight, fitted, *, reverse=False):
    """Audio-only path. No target, emotion, or identity argument is accepted."""
    if reverse:
        x = intervene_input(x, weight, "reverse")
    return (predict_motion(x, weight, fitted["transformed_model"])
            / fitted["metric"]["sqrt_metric"])


def metric_projection_oracle(y, weight, fitted):
    scale = fitted["metric"]["sqrt_metric"]
    model = fitted["transformed_model"]
    return decode_controls(motion_target(y.double() * scale, weight, model), model) / scale


def motion_groups(channels):
    native = {"all_expression": list(channels),
              "upper_expression": [5, 6, 12, 13, 41, 42, 43, 44, 45],
              "brows": list(range(41, 46)), "eyes_expression": [5, 6, 12, 13],
              "mouth": list(range(14, 41)), "jaw17": [17]}
    native = {k: [c for c in v if c in channels] for k, v in native.items()}
    native = {k: v for k, v in native.items() if v}
    return native, {k: [channels.index(c) for c in v] for k, v in native.items()}


def basis_report(fitted, groups):
    model, metric = fitted["transformed_model"], fitted["metric"]
    u, b = model["basis"], fitted["inverse_native_basis"]
    eye = torch.eye(RANK, dtype=torch.float64)
    def fractions(basis, spectral=False):
        mass = basis.square()
        if spectral:
            mass = mass * model["spectrum"][:RANK]
        mass = mass.sum(1)
        total = mass.sum()
        return {k: float(mass[c].sum() / total) if total > 0 else None
                for k, c in groups.items()}
    return {"metric": metric["metric"].tolist(),
            "rms": metric["rms"].tolist() if "rms" in metric else None,
            "rms_floor": float(metric["floor"]) if "floor" in metric else None,
            "metric_sha256": state_hash(metric),
            "model_tensor_sha256": state_hash({k: v for k, v in model.items() if torch.is_tensor(v)}),
            "transformed_gram_max_abs_error": float((u.T @ u - eye).abs().max()),
            "inverse_native_gram_max_abs_error": float((b.T @ b - eye).abs().max()),
            "metric_min": float(metric["metric"].min()),
            "metric_max": float(metric["metric"].max()),
            "transformed_coordinate_loading_fraction": fractions(u),
            "inverse_native_coordinate_loading_fraction": fractions(b),
            "transformed_spectrum_weighted_loading_fraction": fractions(u, True),
            "loading_note": "Coordinate/energy summaries, not independent causal contributions; groups overlap."}


def training_inputs(source, fitted, *, expected_clips=2315, expected_speakers=19):
    """Do not enumerate bundles or read any external_dev tensor/metadata."""
    bundle = source["bundles"]["internal"]
    n = len(bundle["clip_id"])
    if n != expected_clips or len(set(bundle["clip_id"])) != n:
        raise ValueError("Training clip count/uniqueness differs from locked fit set")
    if (not torch.equal(source["train_ids"], torch.arange(n))
            or len(source["heldout_ids"])):
        raise ValueError("Requires prepared data_locked internal-only training indices")
    if not torch.equal(fitted["states"]["rrr_rank8"]["train_ids"], torch.arange(n)):
        raise ValueError("Saved fit state has different training indices")
    state = fitted["states"]["rrr_rank8"]
    if state["rank"] != RANK or state["alpha"] != ALPHA or state["method"] != "rrr":
        raise ValueError("Saved reference must be RRR alpha1/rank8")
    channels = list(state["motion_channel_indices"])
    cm = bundle["channel_mask"].bool()
    if cm.ndim != 2 or len(cm) != n or not torch.equal(cm, cm[:1].expand_as(cm)):
        raise ValueError("Requires the audited common observed channel layout")
    expected = [c for c in range(cm.shape[1]) if cm[0, c] and c not in NUISANCE_CHANNELS_52]
    if channels != expected or bundle["groups"]["all_expression"] != expected:
        raise ValueError("Motion channels differ from locked native fit")
    speakers = []
    for clip in bundle["clip_id"]:
        match = re.match(r"^(mead_[^_]+)_", str(clip), re.IGNORECASE)
        if not match:
            raise ValueError("Locked MEAD clip IDs are required for speaker labels")
        speakers.append(match.group(1))
    if len(set(speakers)) != expected_speakers or len(torch.unique(bundle["speaker_id"])) != expected_speakers:
        raise ValueError("Training identity count differs from locked fit set")
    if len(bundle["sentence_id"]) != n or bundle["emotion_id"].shape != (n,):
        raise ValueError("Training metadata lengths differ")
    folds = validate_folds(fitted["selection"]["folds"], bundle["sentence_id"])
    return bundle, channels, speakers, folds


def run_oof(bundle, channels, folds, saved_folds=None):
    """Fit exactly two arms in each existing fold, with no tuning or full fit."""
    x, y, weight = audio_features(bundle), bundle["motion_bins"][..., channels], bundle["weight"]
    n = len(y)
    target = torch.zeros(y.shape, dtype=torch.float64)
    native_groups, groups = motion_groups(channels)
    predictions = {arm: {mode: torch.zeros_like(target) for mode in MODES} for arm in ARMS}
    states, fold_reports = {}, []
    seen = torch.zeros(n, dtype=torch.long)
    oof_fold = torch.full((n,), -1, dtype=torch.long)
    for number, (fit_ids, val_ids) in enumerate(folds):
        if (seen[val_ids] != 0).any():
            raise ValueError("OOF prediction would overwrite a validation clip")
        # Finish both fits before using validation observations for scoring.
        fold_fits = {arm: fit_fold_arm(x, y, weight, fit_ids, arm) for arm in ARMS}
        xv, yv, wv = x[val_ids], y[val_ids], weight[val_ids]
        target[val_ids] = weighted_clip_center(yv, wv)
        for arm, fitted in fold_fits.items():
            predictions[arm]["full"][val_ids] = predict_native(xv, wv, fitted)
            predictions[arm]["reverse"][val_ids] = predict_native(xv, wv, fitted, reverse=True)
            predictions[arm]["metric_projection_oracle"][val_ids] = metric_projection_oracle(yv, wv, fitted)
            fitted["validation_ids"] = val_ids.clone()
            fitted["motion_channel_indices"] = list(channels)
            states[f"fold{number}_{arm}"] = fitted
            fold_rows = {mode: scalar_summary(clip_statistics(predictions[arm][mode][val_ids],
                target[val_ids], wv, groups["all_expression"])) for mode in MODES}
            expected = None
            if arm == "native" and saved_folds is not None:
                matches = [r for r in saved_folds[number]["scores"]
                           if r["method"] == "rrr" and r["rank"] == RANK and r["alpha"] == ALPHA]
                if len(matches) != 1:
                    raise ValueError("Missing unique saved native fold score")
                expected = matches[0]["native_motion_mse"]
                if not np.isclose(fold_rows["full"]["native_mse"], expected, rtol=1e-9, atol=1e-12):
                    raise ValueError(f"Native fold{number} fails saved score reproduction")
            fold_reports.append({"fold": number, "arm": arm, "fit_clips": len(fit_ids),
                "validation_clips": len(val_ids), "all_expression": fold_rows,
                "saved_native_mse": expected, "basis": basis_report(fitted, groups)})
            print(json.dumps({"fold": number, "arm": arm,
                              "native_mse": fold_rows["full"]["native_mse"]}), flush=True)
        seen[val_ids] += 1
        oof_fold[val_ids] = number
    if not torch.equal(seen, torch.ones_like(seen)):
        raise ValueError("Incomplete OOF coverage")
    statistics = {arm: {mode: {group: clip_statistics(value, target, weight, ids)
                  for group, ids in groups.items()} for mode, value in modes.items()}
                  for arm, modes in predictions.items()}
    return {"target": target, "weight": weight.detach().cpu().double(),
            "predictions": predictions, "statistics": statistics, "states": states,
            "fold_reports": fold_reports, "groups": native_groups, "oof_fold": oof_fold}


def paired_report(statistics, sentence_ids, emotion_ids, speakers, *, samples=5000, seed=45):
    n = len(sentence_ids)
    emotion = torch.as_tensor(emotion_ids).tolist()
    populations = {"all": list(range(n)),
                   "nonneutral": [i for i, e in enumerate(emotion) if e != 0],
                   "neutral": [i for i, e in enumerate(emotion) if e == 0]}
    comparisons = []
    for arm in ARMS:
        for baseline in ("zero", "reverse", "metric_projection_oracle"):
            comparisons.append((f"{arm}_full__vs__{baseline}", (arm, "full"), (arm, baseline)))
        comparisons.append((f"{arm}_metric_projection_oracle__vs__zero",
                            (arm, "metric_projection_oracle"), (arm, "zero")))
    for mode in ("full", "reverse", "metric_projection_oracle"):
        comparisons.append((f"scaled__vs__native_{mode}", ("scaled", mode), ("native", mode)))
    result = {}
    for name, left, right in comparisons:
        a, b = statistics[left[0]][left[1]], statistics[right[0]][right[1]]
        def compare(ids):
            return {group: paired_summary(a[group], b[group], sentence_ids, ids,
                                         samples=samples, seed=seed) for group in a}
        result[name] = {"candidate": list(left), "baseline": list(right),
            "populations": {p: compare(ids) for p, ids in populations.items()},
            "by_speaker": {speaker: {p: compare([i for i in ids if speakers[i] == speaker])
                for p, ids in populations.items()} for speaker in sorted(set(speakers))}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1 or args.threads < 1:
        parser.error("samples and threads must be positive")
    if (args.bundle.resolve().parent.name != "data_locked"
            or args.weights.resolve().parent != args.bundle.resolve().parent
            or args.bundle.name != "diagnostic_bundle.pt"
            or args.weights.name != "content_temporal_weights.pt"):
        parser.error("Use the paired diagnostic_bundle.pt/content_temporal_weights.pt in data_locked")
    if args.output.exists():
        parser.error("Output must be a fresh directory")
    torch.set_num_threads(args.threads)
    inputs = {"bundle": sha(args.bundle), "weights": sha(args.weights)}
    fitted = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    if fitted["provenance"]["bundle_sha256"] != inputs["bundle"]:
        raise ValueError("Weights/bundle byte hash mismatch")
    source = torch.load(args.bundle, map_location="cpu", weights_only=False, mmap=True)
    for key in ("schema", "internal_split_lock_sha256", "fit_source_indices", "cv_seed", "fixed_rank"):
        if source["provenance"][key] != fitted["provenance"][key]:
            raise ValueError(f"Bundle/weights provenance mismatch: {key}")
    bundle, channels, speakers, folds = training_inputs(source, fitted)
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.output / "source"
    source_dir.mkdir()
    paths = [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "kinetalk_b0/predictable_motion.py",
             Path(__file__).with_name("audit_predictable_motion_predictions.py"),
             Path(__file__).with_name("probe_predictable_motion.py"),
             Path(__file__).with_name("train_predictable_renderer.py"),
             Path(__file__).with_name("train_neutral_affect_audio_ablation.py")]
    source_hashes = {str(p.resolve()): sha(p) for p in paths}
    for path in paths:
        shutil.copyfile(path, source_dir / path.name)
    provenance = {"schema": "scaled_motion_basis_oof_v1", "interpretation": INTERPRETATION,
        "input_sha256": inputs, "source_sha256": source_hashes, "alpha": ALPHA, "rank": RANK,
        "arms": list(ARMS), "audio_features": ["content", "middle", "prosody"], "audio_gate": False,
        "fit_components": ["fold-fit channel metric (scaled only)", "fold-fit audio RMS", "RRR basis", "ridge coefficients"],
        "metric_formula": "m=1/max(centered_RMS,median(positive_centered_RMS))^2; m/=mean(m)",
        "metric_objective_note": "Scaling weights both output MSE and output-column coefficient ridge penalty, under the same rank8 constraint.",
        "source_roles_accessed": ["bundles.internal"],
        "heldout405_tensor_values_accessed": False, "outer280_tensor_values_accessed": False,
        "new_identity439_loaded": False, "test_targets_loaded": False,
        "container_note": "Complete input files byte-hashed and mmap metadata loaded; only internal training tensor values indexed.",
        "clip_count": len(bundle["clip_id"]), "speaker_count": len(set(speakers)),
        "sentence_count": len(set(bundle["sentence_id"])), "motion_channel_indices": channels,
        "split_lock_sha256": source["provenance"]["internal_split_lock_sha256"],
        "saved_folds_sha256": canonical_hash(fitted["selection"]["folds"]),
        "bootstrap_samples": args.samples, "bootstrap_seed": args.seed, "cluster_unit": "sentence",
        "bootstrap_scope": "Conditional on saved OOF predictions; no refitting/selection uncertainty. Exploratory region/person breakdowns, not independent confirmatory tests.",
        "generalization_scope": "Cross-sentence within the nineteen fitted identities, not unseen-identity generalization.",
        "reverse_definition": "Reverse valid-bin input order, keep destination weights, then recenter; not an exact physical-time reversal for partial bins.",
        "renderer_trained": False, "default_changed": False, "deployment_compatible": False}
    write_json(args.output / "provenance.json", provenance)
    output = run_oof(bundle, channels, folds, fitted["selection"]["folds"])
    states = output.pop("states")
    torch.save({"provenance": provenance, "states": states}, args.output / "fold_states.pt")
    output.update(clip_id=list(bundle["clip_id"]), sentence_id=list(bundle["sentence_id"]),
                  emotion_id=bundle["emotion_id"].clone(), speaker_id=list(speakers),
                  source_speaker_id=bundle["speaker_id"].clone(), provenance=provenance)
    torch.save(output, args.output / "oof_predictions.pt")
    report = {**provenance, "groups": output["groups"], "folds": output["fold_reports"],
        "comparisons": paired_report(output["statistics"], output["sentence_id"], output["emotion_id"],
                                     speakers, samples=args.samples, seed=args.seed)}
    write_json(args.output / "paired_audit.json", report)
    write_json(args.output / "output_hashes.json", {name: sha(args.output / name) for name in
               ("provenance.json", "fold_states.pt", "oof_predictions.pt", "paired_audit.json")})
    print(json.dumps({"output": str(args.output), "status": "complete", "clips": len(speakers)}), flush=True)


if __name__ == "__main__":
    main()
