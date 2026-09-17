"""Read-only target geometry audit: every oracle here uses query GT motion.

This does not train a model, fit audio labels, or establish predictability.
Group-specific scalars and per-clip PCA are evaluation-only diagnostics of the
single shared scalar hypothesis, not proposed model region channels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import EMOTIONS
from scripts.emotion_ray_metrics import field_metrics


CHANNEL_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def divide(n, d):
    return float(n / d) if float(d) > 1e-15 else None


def temporal_agreement(a, b, weight):
    """Pooled and per-clip weighted temporal correlations, no amplitude fitting."""
    w = weight[..., None]
    denom = w.sum(1, keepdim=True).clamp_min(1)
    a = a - (a * w).sum(1, keepdim=True) / denom
    b = b - (b * w).sum(1, keepdim=True) / denom
    va, vb = (a.square() * w).sum((1, 2)), (b.square() * w).sum((1, 2))
    cov = (a * b * w).sum((1, 2))
    usable = (va > 1e-12) & (vb > 1e-12)
    per_clip = cov[usable] / (va[usable] * vb[usable]).sqrt()
    active = (a.abs() > 1e-5) & (b.abs() > 1e-5) & (w > 0)
    return {
        "pooled_centered_correlation": divide(cov.sum(), (va.sum() * vb.sum()).sqrt()),
        "mean_clip_correlation": float(per_clip.mean()) if len(per_clip) else None,
        "median_clip_correlation": float(per_clip.median()) if len(per_clip) else None,
        "negative_correlation_clips": int((per_clip < 0).sum()),
        "correlatable_clips": len(per_clip),
        "weighted_same_sign_fraction": divide((w * active * ((a * b) > 0)).sum(), (w * active).sum()),
    }


def compact_metrics(pred, target, weight, sentence_ids):
    m = field_metrics(pred, target, weight, sentence_ids, bootstrap_samples=0)
    return {k: m[k] for k in ("native_mse", "zero_mse", "r2_against_zero",
        "pooled_centered_correlation", "energy_ratio", "prediction_energy",
        "target_energy", "sse", "clips", "sentences", "per_sentence")}


def per_clip_pca(y, weight, sentence_ids, clip_ids, emotion):
    """Weighted GT SVD-equivalent oracle: each clip chooses its own spatial axes."""
    energies = []
    rows = []
    for i in range(len(y)):
        x = y[i] * weight[i].sqrt()[:, None]
        eig = torch.linalg.eigvalsh(x.T @ x).flip(0).clamp_min(0)
        total = eig.sum()
        energies.append([float(total), float(eig[:1].sum()), float(eig[:2].sum())])
        rows.append({"clip_id": clip_ids[i], "sentence_id": sentence_ids[i],
            "emotion": EMOTIONS[int(emotion[i])], "target_energy": float(total),
            "rank1_coverage": divide(eig[:1].sum(), total),
            "rank2_coverage": divide(eig[:2].sum(), total)})
    values = torch.tensor(energies, dtype=torch.float64)
    return {"definition": "per-clip GT weighted PCA; axes refit on each evaluated motion; no audio predictability claim",
        "rank1_pooled_coverage": divide(values[:, 1].sum(), values[:, 0].sum()),
        "rank2_pooled_coverage": divide(values[:, 2].sum(), values[:, 0].sum()),
        "rank1_median_clip_coverage": float(torch.tensor([r["rank1_coverage"] for r in rows if r["rank1_coverage"] is not None]).median()),
        "rank2_median_clip_coverage": float(torch.tensor([r["rank2_coverage"] for r in rows if r["rank2_coverage"] is not None]).median()),
        "per_clip": rows}


def audit_subset(bundle, ids):
    selected = ids[bundle["nonneutral"][ids]]
    if not len(selected):
        return None
    y = bundle["motion_bins"][selected].double()
    weight = bundle["weight"][selected].double()
    ray = bundle["true_direction"][selected].double()
    scalar = bundle["scalar"][selected].double()
    sentence_ids = [bundle["sentence_id"][int(i)] for i in selected]
    clip_ids = [bundle["clip_id"][int(i)] for i in selected]
    emo = bundle["emotion_id"][selected]
    torch.testing.assert_close((y * ray[:, None]).sum(-1, keepdim=True), scalar,
                               atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ray.square().sum(-1), torch.ones(len(ray), dtype=ray.dtype),
                               atol=1e-6, rtol=1e-5)
    all_energy = (scalar.square() * weight[..., None]).sum()
    result = {"nonneutral_active_clips": len(selected), "sentences": sorted(set(sentence_ids)),
              "groups": {}, "temporal_agreement": {}}
    group_scalars = {}
    grouped = bundle["groups"]
    group_indices = {**grouped,
        "other_expression": sorted(set(grouped["all_expression"]) - set(grouped["upper_expression"]) - set(grouped["mouth"]))}
    for name, channels in group_indices.items():
        if not channels:
            continue
        group_y, group_ray = y[:, :, channels], ray[:, channels]
        norm2 = group_ray.square().sum(-1)
        projected = (group_y * group_ray[:, None]).sum(-1, keepdim=True)
        best_scalar = torch.where(norm2[:, None, None] > 1e-12,
                                  projected / norm2[:, None, None].clamp_min(1e-12), 0)
        group_scalars[name] = best_scalar
        best = best_scalar * group_ray[:, None]
        shared = scalar * group_ray[:, None]
        mse_shared = compact_metrics(shared, group_y, weight, sentence_ids)
        mse_best = compact_metrics(best, group_y, weight, sentence_ids)
        # Orthogonal projection must never increase that group's GT SSE.
        if mse_best["r2_against_zero"] is not None and mse_best["r2_against_zero"] < -1e-8:
            raise AssertionError("group-optimal scalar is worse than zero")
        result["groups"][name] = {
            "channels": channels,
            "global_scalar_same_ray": mse_shared,
            "group_optimal_scalar_same_ray": mse_best,
            "global_minus_group_optimal_sse": mse_shared["sse"] - mse_best["sse"],
            "mean_ray_squared_mass": float(norm2.mean()),
            "group_projection_covariance_share_in_global_scalar": divide(
                (projected * scalar * weight[..., None]).sum(), all_energy),
            "group_projection_variance_fraction_of_global_scalar": divide(
                (projected.square() * weight[..., None]).sum(), all_energy),
        }
    for left, right in (("upper_expression", "mouth"), ("all_expression", "upper_expression"), ("all_expression", "mouth")):
        if left in group_scalars and right in group_scalars:
            result["temporal_agreement"][left + "__vs__" + right] = temporal_agreement(
                group_scalars[left], group_scalars[right], weight)
    if "upper_expression" in group_indices:
        channels = group_indices["upper_expression"]
        result["upper_per_clip_gt_pca"] = per_clip_pca(y[:, :, channels], weight, sentence_ids, clip_ids, emo)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = torch.load(args.bundle, map_location="cpu", weights_only=False)
    rays = source["rays"]
    report = {"schema": "emotion_ray_geometry_audit_v1",
        "source_bundle_sha256": sha(args.bundle), "script_sha256": sha(__file__),
        "scope": "GT target geometry only; no training, no audio prediction, no renderer evaluation",
        "definitions": {
            "global_scalar": "a=<centered motion, unit emotion ray>, fitted on all observed expression channels",
            "group_optimal_scalar": "a_g=<motion_g,ray_g>/||ray_g||^2, query-motion oracle specific to the evaluated group",
            "covariance_share": "E[<motion_g,ray_g> * global_scalar] / E[global_scalar^2]; disjoint group shares sum to one",
            "pca": "per-clip GT axes chosen on evaluated motion, stricter spatial capacity oracle than static emotional ray",
        }, "ray_controller_contributions": {}, "splits": {}}
    groups = source["bundles"]["internal"]["groups"]
    active_ids = rays["active"].nonzero(as_tuple=True)[0]
    active_rays = rays["rays"][active_ids].double()
    report["static_ray_cosine"] = {"emotion_order": [EMOTIONS[int(e)] for e in active_ids],
        "matrix": (active_rays @ active_rays.T).tolist()}
    for eid in rays["active"].nonzero(as_tuple=True)[0].tolist():
        ray = rays["rays"][eid].double()
        order = ray.square().argsort(descending=True)
        report["ray_controller_contributions"][EMOTIONS[eid]] = {
            "group_squared_mass": {k: float(ray[v].square().sum()) for k,v in groups.items()},
            "controllers": [{"index": int(j), "name": CHANNEL_NAMES[j], "coefficient": float(ray[j]),
                "squared_mass": float(ray[j].square())} for j in order.tolist() if ray[j].abs() > 1e-7]}
    for name, key, ids in (("train", "internal", source["train_ids"]),
        ("internal_heldout", "internal", source["heldout_ids"]),
        ("external_dev", "external_dev", torch.arange(len(source["bundles"]["external_dev"]["scalar"])))):
        bundle = source["bundles"][key]
        row = {"all_clips": len(ids),
            "all_sentences": len({bundle["sentence_id"][int(i)] for i in ids}),
            "neutral_clips": int((bundle["emotion_id"][ids] == 0).sum()),
            "inactive_nonneutral_clips": int(((bundle["emotion_id"][ids] != 0) & ~bundle["nonneutral"][ids]).sum()),
            "pooled": audit_subset(bundle, ids), "per_emotion": {}}
        for eid in bundle["emotion_id"][ids].unique().tolist():
            own = ids[bundle["emotion_id"][ids] == eid]
            if rays["active"][eid]:
                row["per_emotion"][EMOTIONS[eid]] = audit_subset(bundle, own)
        report["splits"][name] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    for name, item in report["splits"].items():
        row = item["pooled"]
        upper = row["groups"]["upper_expression"]
        print(json.dumps({"split": name,
            "all_global_ray_r2": row["groups"]["all_expression"]["global_scalar_same_ray"]["r2_against_zero"],
            "upper_global_scalar_r2": upper["global_scalar_same_ray"]["r2_against_zero"],
            "upper_own_scalar_r2": upper["group_optimal_scalar_same_ray"]["r2_against_zero"],
            "upper_mouth_correlation": row["temporal_agreement"]["upper_expression__vs__mouth"]["pooled_centered_correlation"],
            "upper_gt_pca_rank1": row["upper_per_clip_gt_pca"]["rank1_pooled_coverage"],
            "upper_gt_pca_rank2": row["upper_per_clip_gt_pca"]["rank2_pooled_coverage"]}), flush=True)


if __name__ == "__main__":
    main()
