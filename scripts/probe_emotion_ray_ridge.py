"""Nested training-sentence ridge audit of a frozen emotion-ray bundle.

Each inner fold refits expression directions from its training clip means,
then selects one shared scalar ridge head per audio source. Evaluation reuses
the original outer-training directions and targets. No heldout calibration,
renderer update, global-emotion update, or learned conditional branch is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_ray import fit_emotion_rays, direction_from_probs
from scripts.probe_dynamic_ridge import sentence_folds, ridge_path, predict
from scripts.probe_dynamic_predictability import weighted_mse
from scripts.train_emotion_ray_pilot import evaluate_field, temporal_intervention
from scripts.train_neutral_affect_pilot import sha, write_json


DEFAULT_ALPHAS = (.01, .1, 1., 10., 100.)
UPPER_EXPRESSION_CHANNELS = (5, 6, 12, 13, 41, 42, 43, 44, 45)


def prepare_supervision(diagnostic, supervision="full_expression"):
    """Optionally rebuild the signed upper-expression diagnostic in isolation.

    This changes supervision and evaluation only, never a generation output
    channel. Outer directions are refit from outer-training clip means. Inner
    folds subsequently refit on exactly the same restricted channel mask.
    Original cached tensors, probabilities, and feature statistics are intact.
    """
    if supervision == "full_expression":
        return diagnostic
    if supervision != "upper_expression":
        raise ValueError("supervision must be full_expression or upper_expression")
    diagnostic = deepcopy(diagnostic)
    internal = diagnostic["bundles"]["internal"]
    if internal["motion_bins"].shape[-1] != 52:
        raise ValueError("upper_expression supervision requires the 52-controller layout")
    selected = torch.zeros(52, dtype=torch.bool)
    selected[list(UPPER_EXPRESSION_CHANNELS)] = True
    for bundle in diagnostic["bundles"].values():
        if bundle["motion_bins"].shape[-1] != 52:
            raise ValueError("All upper_expression bundles need the 52-controller layout")
        local_selected = selected.to(bundle["motion_bins"].device)
        bundle["channel_mask"] = bundle["channel_mask"] & local_selected[None]
        bundle["motion_bins"] = torch.where(local_selected[None, None], bundle["motion_bins"], 0)
        bundle["residual_clip_means"] = torch.where(local_selected[None], bundle["residual_clip_means"], 0)
    original_state = diagnostic["rays"]
    state = fit_fold_rays(internal, diagnostic["train_ids"],
        int(original_state["num_emotions"]), int(original_state["stride"]))
    diagnostic["rays"] = state
    common = state["observed_channels"]
    for bundle in diagnostic["bundles"].values():
        if not bundle["channel_mask"][:, common].all():
            raise ValueError("Evaluation lacks an observed training upper-expression channel")
        bundle["motion_bins"] = torch.where(common[None, None], bundle["motion_bins"], 0)
        bundle["residual_clip_means"] = torch.where(common[None], bundle["residual_clip_means"], 0)
        direction = state["rays"][bundle["emotion_id"]]
        bundle["true_direction"] = direction
        bundle["actual_direction"] = bundle["probability"] @ state["rays"]
        bundle["scalar"] = (bundle["motion_bins"] * direction[:, None]).sum(-1, keepdim=True)
        bundle["proxy"] = bundle["scalar"] * direction[:, None]
        bundle["nonneutral"] = state["active"][bundle["emotion_id"]]
        _, bundle["direction_info"] = direction_from_probs(bundle["probability"], state, return_info=True)
        bundle["groups"] = {"upper_expression": common.nonzero(as_tuple=True)[0].tolist()}
    return diagnostic


def fit_fold_rays(bundle, fit_ids, num_emotions=8, stride=4):
    """Refit directions using fold-training means only, including neutrals.

    Ray fitting originally averages each valid clip before averaging by
    speaker/emotion. A one-frame sequence containing that clip mean is
    equivalent, and requires no full motion cache or checkpoint reload.
    """
    means = bundle["residual_clip_means"]
    valid = torch.ones((len(means), 1), dtype=torch.bool, device=means.device)
    return fit_emotion_rays(means[:, None], valid, bundle["channel_mask"],
        bundle["speaker_id"], bundle["emotion_id"], fit_ids,
        num_emotions=num_emotions, stride=stride)


def _active_ids(bundle, state, ids):
    ids = torch.as_tensor(ids, dtype=torch.long, device=bundle["emotion_id"].device)
    return ids[state["active"][bundle["emotion_id"][ids]]]


def nested_select_alpha(bundle, source, train_ids, alphas=DEFAULT_ALPHAS,
                        seed=45, n_folds=3, num_emotions=8, stride=4):
    """Select alpha from fold-fitted targets and feature scales, never heldout.

    Neutral clips establish expression directions. The scalar fit and MSE
    selection use only active nonneutral clips, since a zero neutral direction
    makes neutral amplitude unidentifiable. Inference direction supplies the
    neutral/confidence suppression, without a GT gate on the scalar head.
    """
    alphas = tuple(float(value) for value in alphas)
    features, weight = bundle["features"][source], bundle["weight"]
    pooled_errors = [0. for _ in alphas]
    total_weight = 0.
    fold_rows = []
    for index, (fit_ids, validation_ids) in enumerate(sentence_folds(
            bundle["sentence_id"], train_ids, seed=seed, n_folds=n_folds)):
        state = fit_fold_rays(bundle, fit_ids, num_emotions, stride)
        active_fit = _active_ids(bundle, state, fit_ids)
        active_validation = _active_ids(bundle, state, validation_ids)
        if not len(active_fit) or not len(active_validation):
            raise ValueError("Each inner fold needs active nonneutral fitting and validation clips")
        # Build only this fold's targets: outer-heldout motion values never
        # participate in projection, fitting, scaling, or model selection.
        target = bundle["motion_bins"].new_zeros((*weight.shape, 1))
        used_ids = torch.cat((active_fit, active_validation))
        directions = state["rays"][bundle["emotion_id"][used_ids]]
        target[used_ids] = (bundle["motion_bins"][used_ids] * directions[:, None]).sum(-1, keepdim=True)
        coefficients, std = ridge_path(features, target, weight, active_fit, alphas)
        rows = []
        validation_weight = float(weight[active_validation].sum())
        for j, coefficient in enumerate(coefficients):
            estimated = predict(features[active_validation], weight[active_validation], coefficient, std)
            error = float(weighted_mse(estimated, target[active_validation], weight[active_validation]))
            pooled_errors[j] += error * validation_weight
            rows.append({"alpha": alphas[j], "native_mse": error})
        total_weight += validation_weight
        fold_rows.append({"fold": index, "ray_fit_indices": fit_ids.tolist(),
            "fit_indices": active_fit.tolist(), "validation_indices": active_validation.tolist(),
            "all_validation_indices": validation_ids.tolist(),
            "fit_sentences": sorted({str(bundle["sentence_id"][int(i)]) for i in fit_ids}),
            "validation_sentences": sorted({str(bundle["sentence_id"][int(i)]) for i in validation_ids}),
            "feature_std_fit_indices": active_fit.tolist(),
            "active_emotions": state["active"].nonzero(as_tuple=True)[0].tolist(),
            "ray_raw_norms": state["raw_norms"].tolist(),
            "validation_weight": validation_weight, "scores": rows})
    scores = [{"alpha": alpha, "inner_native_mse": error / total_weight}
              for alpha, error in zip(alphas, pooled_errors)]
    best = min(range(len(alphas)), key=lambda j: scores[j]["inner_native_mse"])
    return alphas[best], {"folds": fold_rows, "scores": scores,
        "aggregation": "pooled active-nonneutral observed-bin MSE across training-sentence folds",
        "target_fitting": "refit neutral-relative emotion rays independently in every inner training fold"}


def _subset_bundle(bundle, ids):
    """Subset clip tensors/metadata without slicing unrelated channel groups."""
    count = len(bundle["scalar"])
    result = {}
    for key, value in bundle.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            result[key] = value[ids]
        elif key in ("sentence_id", "clip_id"):
            result[key] = [value[int(i)] for i in ids]
        else:
            result[key] = value
    return result


def evaluate_split(prediction, bundle, ids, bootstrap_samples=1000):
    """Intervene on the scalar alone, keeping each clip's direction fixed."""
    local = _subset_bundle(bundle, ids)
    local_ids = torch.arange(len(ids), dtype=torch.long)
    scalar, weight = prediction[ids], bundle["weight"][ids]
    report = {
        "same_curve_gt_direction": evaluate_field(scalar, local["true_direction"], local, local_ids, bootstrap_samples),
        "same_curve_audio_direction": evaluate_field(scalar, local["actual_direction"], local, local_ids, bootstrap_samples),
        "interventions": {},
    }
    for mode in ("zero", "reverse", "shuffle"):
        altered = temporal_intervention(scalar, weight, mode)
        report["interventions"][mode] = {
            "same_curve_gt_direction": evaluate_field(altered, local["true_direction"], local, local_ids, bootstrap_samples),
            "same_curve_audio_direction": evaluate_field(altered, local["actual_direction"], local, local_ids, bootstrap_samples),
        }
    return report


def run_probe(diagnostic, output, *, bundle_path=None, alphas=DEFAULT_ALPHAS,
              seed=45, n_folds=3, bootstrap_samples=1000,
              supervision="full_expression"):
    """Run both frozen feature sources; existing output directories are refused."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    diagnostic = prepare_supervision(diagnostic, supervision)
    internal = diagnostic["bundles"]["internal"]
    external = diagnostic["bundles"]["external_dev"]
    train_ids = torch.as_tensor(diagnostic["train_ids"], dtype=torch.long)
    heldout_ids = torch.as_tensor(diagnostic["heldout_ids"], dtype=torch.long)
    if set(train_ids.tolist()) & set(heldout_ids.tolist()):
        raise ValueError("Training and heldout indices overlap")
    train_sentences = {str(internal["sentence_id"][int(i)]) for i in train_ids}
    heldout_sentences = {str(internal["sentence_id"][int(i)]) for i in heldout_ids}
    dev_sentences = {str(value) for value in external["sentence_id"]}
    if train_sentences & heldout_sentences or (train_sentences | heldout_sentences) & dev_sentences:
        raise ValueError("Sentence partitions overlap")
    state = diagnostic["rays"]
    final_fit_ids = _active_ids(internal, state, train_ids)
    if not len(final_fit_ids):
        raise ValueError("No outer-training active nonneutral clips")
    # Detect stale or mispaired bundles before fitting a final head.
    expected = (internal["motion_bins"][train_ids] *
                state["rays"][internal["emotion_id"][train_ids], None]).sum(-1, keepdim=True)
    torch.testing.assert_close(expected, internal["scalar"][train_ids])
    splits = {"train": (internal, train_ids), "internal_heldout": (internal, heldout_ids),
              "external_dev": (external, torch.arange(len(external["scalar"])))}
    schema = "emotion_ray_nested_ridge_v1"
    report = {"schema": schema, "arms": {}, "generation_evaluated": False,
        "supervision": supervision, "mouth_protection_evaluated": False,
        "frozen_backbone_loaded_or_updated": False,
        "selection": "Inner training-sentence MSE only; rays and feature scale refit in each fold",
        "scalar_head": "One shared bias-free linear scalar per source; no emotion conditioning",
        "evaluation": ("Original outer-training GT-ray proxy and motion; GT/actual audio directions paired"
            if supervision == "full_expression" else
            "Upper-only signed ray refit on outer-training means; upper motion only; mouth protection cannot be assessed"),
        "scope": diagnostic.get("provenance", {}).get("scope"),
        "train_indices": train_ids.tolist(), "internal_heldout_indices": heldout_ids.tolist()}
    weights = {"schema": schema, "rays": state, "arms": {}, "supervision": supervision,
        "preprocessing": "binned, weighted clip-centered features; train RMS standardization; scalar linear readout"}
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), root / "scripts/probe_dynamic_ridge.py",
        root / "scripts/train_emotion_ray_pilot.py", root / "scripts/emotion_ray_metrics.py",
        root / "kinetalk_b0/emotion_ray.py", root / "scripts/probe_dynamic_predictability.py"]
    (output / "source").mkdir()
    for path in source_paths:
        shutil.copyfile(path, output / "source" / path.name)
    provenance = {"schema": schema, "bundle": str(bundle_path) if bundle_path else None,
        "supervision": supervision,
        "supervision_channels": (list(UPPER_EXPRESSION_CHANNELS) if supervision == "upper_expression"
                                 else state["observed_channels"].nonzero(as_tuple=True)[0].tolist()),
        "supervision_note": ("Single scalar diagnostic restricted to upper expression; no new generator channels. "
                             "Outer/inner rays refit on their training clips only. Does not evaluate mouth protection."
                             if supervision == "upper_expression" else "Original full-expression diagnostic target"),
        "mouth_protection_evaluated": False,
        "bundle_sha256": sha(Path(bundle_path)) if bundle_path else None,
        "parent_provenance": diagnostic.get("provenance", {}),
        "source_sha256": {str(path.relative_to(root)): sha(path) for path in source_paths},
        "seed": seed, "inner_folds": n_folds, "alphas": list(alphas),
        "bootstrap_samples": bootstrap_samples, "ray_fit_indices": train_ids.tolist(),
        "final_feature_std_fit_indices": final_fit_ids.tolist(),
        "neutral_handling": "Neutral clips fit ray anchors; scalar fits active nonneutral clips; inference keeps audio direction",
        "heldout_used_for_selection_or_calibration": False,
        "generation_evaluated": False}
    write_json(output / "provenance.json", provenance)
    for source in ("acoustic", "content"):
        alpha, inner = nested_select_alpha(internal, source, train_ids, alphas, seed, n_folds,
            num_emotions=int(state["num_emotions"]), stride=int(state["stride"]))
        coefficients, std = ridge_path(internal["features"][source], internal["scalar"],
            internal["weight"], final_fit_ids, [alpha])
        coefficient = coefficients[0]
        predictions = {"internal": predict(internal["features"][source], internal["weight"], coefficient, std),
                       "external_dev": predict(external["features"][source], external["weight"], coefficient, std)}
        arm = {"selected_alpha": alpha, "inner_validation": inner, "splits": {}}
        for name, (bundle, ids) in splits.items():
            prediction = predictions["external_dev" if name == "external_dev" else "internal"]
            arm["splits"][name] = evaluate_split(prediction, bundle, ids, bootstrap_samples)
            row = arm["splits"][name]
            print(json.dumps({"source": source, "alpha": alpha, "split": name,
                "scalar_r2": row["same_curve_gt_direction"]["scalar_nonneutral"]["r2_against_zero"],
                "gt_proxy_r2": row["same_curve_gt_direction"]["proxy_nonneutral"]["r2_against_zero"],
                "audio_proxy_r2": row["same_curve_audio_direction"]["proxy_nonneutral"]["r2_against_zero"]}), flush=True)
        report["arms"][source] = arm
        weights["arms"][source] = {"coefficient": coefficient, "feature_std": std,
            "native_coefficient": coefficient / std[:, None], "alpha": alpha,
            "feature_std_fit_indices": final_fit_ids, "predictions": predictions}
        write_json(output / "summary.json", report)
        torch.save(weights, output / "weights.pt")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--supervision", choices=("full_expression", "upper_expression"),
                        default="full_expression")
    args = parser.parse_args()
    torch.set_num_threads(4)
    diagnostic = torch.load(args.bundle, map_location="cpu", weights_only=False)
    run_probe(diagnostic, args.output, bundle_path=args.bundle, alphas=args.alphas,
              seed=args.seed, n_folds=args.inner_folds, bootstrap_samples=args.bootstrap_samples,
              supervision=args.supervision)


if __name__ == "__main__":
    main()
