"""Nested sentence-heldout predictable-motion diagnostic, with no model edits.

All targets are real centered motion. Each inner fold refits feature scales,
RRR/PCA bases and ridge weights; original motion MSE selects hyperparameters.
Evaluation never chooses ranks, calibrates amplitude, or supplies motion to an
audio prediction. Oracle motion projection is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.predictable_motion import (
    bin_centered_frames, decode_controls, fit_motion_path, motion_target,
    predict_motion, weighted_clip_center,
)
from scripts.emotion_ray_metrics import field_metrics
from scripts.probe_dynamic_ridge import sentence_folds


DEFAULT_ALPHAS = (.01, .1, 1., 10., 100.)
DEFAULT_RANKS = (2, 4, 8)
ARMS = ("existing_e2v", "temporal", "content", "content_temporal")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def model_key(method, rank):
    return f"{method}_rank{rank}"


def native_sse(prediction, target, weight):
    target = weighted_clip_center(target, weight)
    w = weight.detach().cpu().double()
    if prediction.shape != target.shape or not torch.isfinite(prediction[w > 0]).all():
        raise ValueError("prediction must be finite and match the motion target")
    error = torch.where(w[..., None] > 0, prediction - target, 0)
    return float((error.square() * w[..., None]).sum()), float(w.sum()) * target.shape[-1]


def nested_select(x, y, weight, sentence_ids, train_ids, *, alphas=DEFAULT_ALPHAS,
                  ranks=DEFAULT_RANKS, folds=3, seed=45):
    """Choose alpha per method/rank, then rank using only inner motion error.

    The function never reads outer-heldout tensor values. No target basis or
    feature scale is shared across inner fits. Scores are pooled real-motion
    SSE divided by observed channel-weight, not per-model latent accuracy.
    """
    rows, accumulated = [], {}
    for fold, (fit_ids, validation_ids) in enumerate(sentence_folds(sentence_ids, train_ids, seed, folds)):
        states = fit_motion_path(x, y, weight, fit_ids, alphas, ranks)
        scores = []
        for state in states:
            prediction = predict_motion(x[validation_ids], weight[validation_ids], state)
            error, denominator = native_sse(prediction, y[validation_ids], weight[validation_ids])
            key = (state["method"], state["rank"], state["alpha"])
            record = accumulated.setdefault(key, [0., 0.])
            record[0] += error
            record[1] += denominator
            scores.append({"method": state["method"], "rank": state["rank"], "alpha": state["alpha"],
                           "native_motion_mse": error / denominator})
        rows.append({"fold": fold, "fit_indices": fit_ids.tolist(), "validation_indices": validation_ids.tolist(),
            "fit_sentences": sorted({str(sentence_ids[int(i)]) for i in fit_ids}),
            "validation_sentences": sorted({str(sentence_ids[int(i)]) for i in validation_ids}),
            "all_fitted_components": "feature RMS, motion basis, audio ridge coefficients", "scores": scores})
    candidates = [{"method": method, "rank": rank, "alpha": alpha, "inner_native_motion_mse": value[0] / value[1]}
                  for (method, rank, alpha), value in accumulated.items()]
    selected = {}
    for candidate in candidates:
        key = model_key(candidate["method"], candidate["rank"])
        if key not in selected or candidate["inner_native_motion_mse"] < selected[key]["inner_native_motion_mse"]:
            selected[key] = candidate
    rank_selected = {}
    for key, candidate in selected.items():
        method = candidate["method"]
        if method not in rank_selected or candidate["inner_native_motion_mse"] < selected[rank_selected[method]]["inner_native_motion_mse"]:
            rank_selected[method] = key
    return {"selected_per_method_rank": selected, "selected_rank_per_method": rank_selected,
            "candidate_scores": candidates, "folds": rows,
            "selection_target": "pooled native real-motion MSE on all observed expression channels, including neutral",
            "outer_heldout_used": False}


def _clip_ids(bundle):
    value = bundle.get("clip_id", bundle.get("clips"))
    if not isinstance(value, (list, tuple)) or len(value) != len(bundle["weight"]):
        raise ValueError("bundle requires one clip_id per motion clip")
    values = [str(v) for v in value]
    if len(set(values)) != len(values):
        raise ValueError("duplicate bundle clip IDs")
    return values


def load_aligned_sidecar(bundle, path, stride):
    """Strict sidecar adapter used by the CLI and leakage/alignment tests."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    clips = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(clips, list):
        raise ValueError("audio sidecar must contain a clips list")
    ids = _clip_ids(bundle)
    by_id = {str(c["clip_id"]): c for c in clips}
    if len(by_id) != len(clips) or set(by_id) != set(ids):
        raise ValueError("sidecar and bundle clip IDs must match exactly without duplicates")
    expected = bundle["weight"].double()
    collected = {"middle": [], "final": [], "prosody": []}
    for i, clip_id in enumerate(ids):
        clip = by_id[clip_id]
        valid, times = torch.as_tensor(clip["valid"]), torch.as_tensor(clip["times"])
        if valid.ndim != 1 or valid.dtype != torch.bool or times.shape != valid.shape:
            raise ValueError("sidecar valid/times geometry is invalid")
        if not torch.isfinite(times).all() or not (times[1:] > times[:-1]).all():
            raise ValueError("sidecar times must be finite and strictly increasing")
        for key, dimension in (("middle", 768), ("final", 768), ("prosody", 4)):
            frames = torch.as_tensor(clip[key])
            if frames.shape != (len(valid), dimension):
                raise ValueError(f"{clip_id} {key} dimensions differ from expected native-frame geometry")
            values, weight = bin_centered_frames(frames[None], valid[None], stride)
            if weight.shape[1:] != expected[i].shape or not torch.equal(weight[0], expected[i]):
                raise ValueError(f"{clip_id} sidecar valid-frame bin weights differ from motion bundle")
            collected[key].append(values)
    result = {key: torch.cat(value, 0) for key, value in collected.items()}
    metadata = {k: v for k, v in payload.items() if k != "clips"}
    return result, metadata


def final_feature_agreement(raw_final, cached_features, weight):
    """Report cache/extraction consistency after per-axis affine normalization.

    Used only to audit extraction, never for training or model selection.
    Cached features may have a prior per-feature positive affine transform.
    """
    a, b = weighted_clip_center(raw_final, weight), weighted_clip_center(cached_features, weight)
    if a.shape != b.shape:
        raise ValueError("final sidecar and cached emotion features must have matching geometry")
    w = weight.double()[..., None]
    va, vb = (a.square() * w).sum((0, 1)), (b.square() * w).sum((0, 1))
    valid = (va > 1e-12) & (vb > 1e-12)
    correlation = (a * b * w).sum((0, 1))[valid] / (va[valid] * vb[valid]).sqrt()
    return {"valid_features": int(valid.sum()), "mean_axis_correlation": float(correlation.mean()) if len(correlation) else None,
            "min_axis_correlation": float(correlation.min()) if len(correlation) else None,
            "note": "affine-invariant diagnostic only; extraction dtype may differ; no equality assumption"}


def intervene_input(features, weight, mode):
    """Time-only input controls, with destination emotion/identity held fixed."""
    if mode == "full":
        return features
    result = torch.zeros_like(features)
    if mode == "zero":
        return result
    if mode not in ("reverse", "shuffle"):
        raise ValueError("unknown intervention")
    if mode == "shuffle" and len(features) < 2:
        return result
    for i in range(len(features)):
        positions = (weight[i] > 0).nonzero(as_tuple=True)[0]
        if not len(positions):
            continue
        if mode == "reverse":
            result[i, positions] = features[i, positions.flip(0)]
        else:
            source = (i + 1) % len(features)
            active = features[source, weight[source] > 0]
            if len(active):
                indices = torch.linspace(0, len(active)-1, len(positions)).round().long()
                result[i, positions] = active[indices]
    return weighted_clip_center(result, weight)


def score_fields(prediction, target, weight, sentence_ids, emotion_ids, groups, *, bootstrap=0, per_sentence=False):
    result = {}
    populations = {"all": torch.arange(len(target)), "nonneutral": (emotion_ids != 0).nonzero(as_tuple=True)[0],
                   "neutral": (emotion_ids == 0).nonzero(as_tuple=True)[0]}
    for population, ids in populations.items():
        if not len(ids):
            result[population] = None
            continue
        result[population] = {}
        for name, channels in groups.items():
            if not channels:
                continue
            scores = field_metrics(prediction[ids][:, :, channels], target[ids][:, :, channels], weight[ids],
                [sentence_ids[int(i)] for i in ids], bootstrap_samples=bootstrap if population == "nonneutral" else 0)
            if not per_sentence or population != "nonneutral":
                scores.pop("per_sentence")
            result[population][name] = scores
    return result


def assert_partition(source):
    bundle = source["bundles"]["internal"]
    train, heldout = source["train_ids"].long(), source["heldout_ids"].long()
    if len(set(train.tolist()) & set(heldout.tolist())) or sorted(train.tolist()+heldout.tolist()) != list(range(len(bundle["weight"]))):
        raise ValueError("outer indices must form a disjoint complete partition")
    train_sentences = {str(bundle["sentence_id"][int(i)]) for i in train}
    heldout_sentences = {str(bundle["sentence_id"][int(i)]) for i in heldout}
    external = source["bundles"]["external_dev"]
    if train_sentences & heldout_sentences:
        raise ValueError("outer train and heldout sentence overlap")
    if set(_clip_ids(bundle)) & set(_clip_ids(external)):
        raise ValueError("internal and external clip overlap")
    if set(map(str,bundle["sentence_id"])) & set(map(str,external["sentence_id"])):
        raise ValueError("internal and external sentence overlap")
    return train, heldout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audio-sidecars", type=Path,
                        help="Optional raw sidecars; omit when bundle already contains middle/prosody bins")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source = torch.load(args.bundle, map_location="cpu", weights_only=False)
    train_ids, hold_ids = assert_partition(source)
    bundles = source["bundles"]
    stride = int(source.get("rays", {}).get("stride", 4))
    channels = bundles["internal"]["groups"]["all_expression"]
    if bundles["external_dev"]["groups"] != bundles["internal"]["groups"]:
        raise ValueError("all splits must share identical observed evaluation channels")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source").mkdir()
    paths = [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "kinetalk_b0/predictable_motion.py",
             Path(__file__).resolve().parent / "emotion_ray_metrics.py", Path(__file__).resolve().parent / "probe_dynamic_ridge.py"]
    for path in paths:
        shutil.copyfile(path, args.output / "source" / path.name)
    audio, sidecar_meta, agreement = {}, {}, {}
    if args.audio_sidecars is not None:
        for key, filename in (("internal", "train.pt"), ("external_dev", "heldout.pt")):
            audio[key], sidecar_meta[key] = load_aligned_sidecar(bundles[key], args.audio_sidecars / filename, stride)
            agreement[key] = final_feature_agreement(audio[key]["final"], bundles[key]["features"]["acoustic"], bundles[key]["weight"])
    else:
        for key, bundle in bundles.items():
            values = bundle["features"]
            if any(k not in values for k in ("middle", "prosody")):
                raise ValueError("bundle lacks middle/prosody; supply --audio-sidecars")
            audio[key] = {k: weighted_clip_center(values[k], bundle["weight"]) for k in ("middle", "prosody")}
            if audio[key]["middle"].shape[-1] != 768 or audio[key]["prosody"].shape[-1] != 4:
                raise ValueError("prebinned middle/prosody dimensions must be 768/4")
            sidecar_meta[key] = {"source": "features already binned in diagnostic bundle"}
            agreement[key] = {"status": "not repeated", "reason": "raw final features not supplied"}
    features = {arm: {} for arm in args.arms}
    for key, bundle in bundles.items():
        available = {"existing_e2v": bundle["features"]["acoustic"].double(),
            "temporal": torch.cat([audio[key]["middle"], audio[key]["prosody"]], -1),
            "content": bundle["features"]["content"].double(),
            "content_temporal": torch.cat([bundle["features"]["content"].double(), audio[key]["middle"], audio[key]["prosody"]], -1)}
        for arm in features:
            features[arm][key] = available[arm]
    provenance = {"schema": "predictable_motion_nested_probe_v1", "args": {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "bundle_sha256": sha(args.bundle), "audio_sidecar_sha256": ({name: sha(args.audio_sidecars / name) for name in ("train.pt","heldout.pt")} if args.audio_sidecars is not None else None),
        "source_sha256": {str(p): sha(p) for p in paths}, "source_bundle_provenance": source.get("provenance"),
        "train_indices": train_ids.tolist(), "internal_heldout_indices": hold_ids.tolist(),
        "target_channels": channels, "stride": stride, "sidecar_metadata": sidecar_meta,
        "scope": "same enrolled identities; development data previously inspected; source frozen-backbone exposure retained from bundle",
        "selection": "inner training sentences only; every fold refits feature RMS and RRR/PCA basis; native motion MSE including neutral",
        "motion_teacher": "real motion projected onto train-fitted U; audio predictions never become target labels",
        "no_main_model_edits": True, "generation_evaluated": False}
    write_json(args.output / "provenance.json", provenance)
    report = {"schema": provenance["schema"], "feature_agreement": agreement, "arms": {},
        "interpretation": "latent fit is not success; use common native-motion upper-face and lip metrics with zero/reverse controls"}
    internal = bundles["internal"]
    target = internal["motion_bins"][:, :, channels]
    weight = internal["weight"]
    for arm, feature_sets in features.items():
        started = time.time()
        print(json.dumps({"stage": "inner_fit", "arm": arm, "input_dim": feature_sets["internal"].shape[-1]}), flush=True)
        selection = nested_select(feature_sets["internal"], target, weight, internal["sentence_id"], train_ids,
            alphas=args.alphas, ranks=args.ranks, folds=args.inner_folds, seed=args.seed)
        selected = selection["selected_per_method_rank"]
        needed_alphas = sorted({v["alpha"] for v in selected.values()})
        states = fit_motion_path(feature_sets["internal"], target, weight, train_ids, needed_alphas, args.ranks)
        states = [s for s in states if selected[model_key(s["method"], s["rank"])]["alpha"] == s["alpha"]]
        arm_report = {"input_dim": feature_sets["internal"].shape[-1], "inner_selection": selection,
            "selected_rank_per_method": selection["selected_rank_per_method"], "fixed_rank8": [k for k,v in selected.items() if v["rank"] == 8], "models": {}}
        saved_states, saved_predictions = {}, {}
        for state in states:
            key = model_key(state["method"], state["rank"])
            chosen = key in selection["selected_rank_per_method"].values()
            evaluations, saved_predictions[key] = {}, {}
            for split, bundle_key, ids in (("train", "internal", train_ids), ("internal_heldout", "internal", hold_ids),
                    ("external_dev", "external_dev", torch.arange(len(bundles["external_dev"]["weight"])))):
                if not len(ids):
                    continue
                bundle = bundles[bundle_key]
                xx, ww = feature_sets[bundle_key][ids], bundle["weight"][ids]
                yy = weighted_clip_center(bundle["motion_bins"][ids], ww)
                sentence_ids = [str(bundle["sentence_id"][int(i)]) for i in ids]
                modes = ("full", "oracle", "zero") if split == "train" else ("full", "oracle", "zero", "reverse", "shuffle")
                scores, predictions = {}, {}
                for mode in modes:
                    pred = torch.zeros_like(yy)
                    if mode == "oracle":
                        value = decode_controls(motion_target(yy[:, :, channels], ww, state), state)
                    else:
                        value = predict_motion(intervene_input(xx, ww, mode), ww, state)
                    pred[:, :, channels] = value
                    predictions[mode] = pred.float()
                    boot = args.bootstrap if mode == "full" and split != "train" and (chosen or state["rank"] == 8) else 0
                    scores[mode] = score_fields(pred, yy, ww, sentence_ids, bundle["emotion_id"][ids], bundle["groups"],
                        bootstrap=boot, per_sentence=mode == "full" and split != "train")
                evaluations[split] = scores
                # All candidates retain scalar metrics. Keep curves only for
                # validation of inner-selected/fixed-rank8 models, avoiding
                # redundant copies of the expanded training cache on disk.
                if split != "train" and (chosen or state["rank"] == 8):
                    saved_predictions[key][split] = {"motion": predictions, "target": yy.float(), "weight": ww,
                        "emotion_id": bundle["emotion_id"][ids], "sentence_id": sentence_ids,
                        "clip_id": [_clip_ids(bundle)[int(i)] for i in ids]}
            evaluations["selected_alpha"] = state["alpha"]
            arm_report["models"][key] = evaluations
            saved_states[key] = {**state, "motion_channel_indices": channels}
            print(json.dumps({"arm": arm, "model": key, "alpha": state["alpha"], "inner_selected_rank": chosen,
                "heldout_upper_r2": evaluations["internal_heldout"]["full"]["nonneutral"]["upper_expression"]["r2_against_zero"] if "internal_heldout" in evaluations else None,
                "external_upper_r2": evaluations["external_dev"]["full"]["nonneutral"]["upper_expression"]["r2_against_zero"]}), flush=True)
        arm_report["elapsed_seconds"] = time.time() - started
        report["arms"][arm] = arm_report
        torch.save({"states": saved_states, "selection": selection, "provenance": provenance}, args.output / (arm + "_weights.pt"))
        torch.save(saved_predictions, args.output / (arm + "_predictions.pt"))
        write_json(args.output / "summary.json", report)
    print("COMPLETE: nested real-motion evaluation; no renderer or main checkpoint changed", flush=True)


if __name__ == "__main__":
    main()
