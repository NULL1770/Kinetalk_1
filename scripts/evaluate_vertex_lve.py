"""Evaluate mesh-space errors from ARKit coefficients and a declared rig.

The evaluator never infers geometry from coefficient arrays. It requires a
neutral mesh and 52 matching blendshape deltas. Inputs are ``[T,52]`` or
``[S,T,52]`` predictions and ``[T,52]`` targets.

Both conventions encountered in the literature are emitted:
``facediffuser_mve`` is mean per-vertex Euclidean error; the official
FaceDiffuser-code lip metric is ``facediffuser_lve_sq`` (maximum squared
lip-vertex error per frame, averaged over time); ``vertex_lve_sqrt`` is the
same maximum-over-lips metric with the square root retained, matching the
metric description used by EmoTalk. Squared metrics have squared coordinate
units and must not be reported as millimetres.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _vertices(coeff: np.ndarray, neutral: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Map [T,52] ARKit coefficients to [T,V,3] vertices."""
    return neutral[None] + np.einsum("tc,cvd->tvd", coeff, deltas, optimize=True)


def _check_mask(name: str, value, n_vertices: int) -> np.ndarray:
    if value is None:
        return np.ones(n_vertices, dtype=bool)
    value = np.asarray(value, dtype=bool)
    if value.shape != (n_vertices,) or not value.any():
        raise ValueError(f"{name} must be a non-empty [V] boolean mask")
    return value


def evaluate(prediction, target, valid, channel_mask, neutral, deltas,
             lip_mask=None, eye_forehead_mask=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    channel_mask = np.asarray(channel_mask, dtype=bool)
    neutral = np.asarray(neutral, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    if prediction.ndim == 2:
        prediction = prediction[None]
    if prediction.ndim != 3 or prediction.shape[-1] != 52:
        raise ValueError("prediction must be [T,52] or [S,T,52]")
    if target.shape != prediction.shape[1:] or valid.shape != (target.shape[0],):
        raise ValueError("target/valid shapes do not match prediction")
    if channel_mask.shape == (52,):
        support = np.broadcast_to(channel_mask, target.shape)
    elif channel_mask.shape == target.shape:
        # Prepared ARKit clips may carry a time-varying observation mask.
        # Preserve it instead of silently treating an intermittently missing
        # coefficient as a valid zero target.
        support = channel_mask
    else:
        raise ValueError("channel_mask must have shape [52] or [T,52]")
    if neutral.ndim != 2 or neutral.shape[1] != 3:
        raise ValueError("neutral_vertices must have shape [V,3]")
    if deltas.shape != (52, neutral.shape[0], 3):
        raise ValueError("blendshape_deltas must have shape [52,V,3]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite coefficient input")
    if not valid.any():
        raise ValueError("no valid frames")

    # Apply the channel mask identically to prediction and target. Unobserved
    # channels therefore cannot create a geometry difference.
    observed = valid[:, None] & support
    target_masked = np.where(observed, target, 0.0)
    pred_masked = np.where(observed[None], prediction, 0.0)
    gt = _vertices(target_masked, neutral, deltas)
    samples = np.stack([_vertices(x, neutral, deltas) for x in pred_masked], axis=0)

    # [S,T_valid,V] Euclidean vertex distances.
    distance = np.linalg.norm(samples - gt[None], axis=-1)[:, valid]
    if distance.size == 0:
        raise ValueError("no valid frames")

    all_mask = np.ones(neutral.shape[0], dtype=bool)
    lip_mask = _check_mask("lip_mask", lip_mask, neutral.shape[0])
    eye_forehead_mask = _check_mask("eye_forehead_mask", eye_forehead_mask, neutral.shape[0])

    # FaceDiffuser MVE official code: mean over vertices and then frames.
    facediffuser_mve = distance[:, :, all_mask].mean(axis=(1, 2))
    # Official FaceDiffuser LVE code uses max squared distance over lip verts.
    lip_sq = np.square(distance[:, :, lip_mask])
    facediffuser_lve_sq = lip_sq.max(axis=-1).mean(axis=-1)
    # EmoTalk-style textual definition: max Euclidean distance per frame.
    vertex_lve_sqrt = distance[:, :, lip_mask].max(axis=-1).mean(axis=-1)
    eye_forehead_eve = distance[:, :, eye_forehead_mask].max(axis=-1).mean(axis=-1)
    # Diagnostic only; do not label this LVE/MVE in the paper.
    mean_vertex_l2 = distance.mean(axis=(1, 2))

    return {
        "schema": "vertex_metrics_v3",
        "samples": int(prediction.shape[0]),
        "vertices": int(neutral.shape[0]),
        "valid_frames": int(valid.sum()),
        "facediffuser_mve": facediffuser_mve.tolist(),
        "facediffuser_lve_sq": facediffuser_lve_sq.tolist(),
        "vertex_lve_sqrt": vertex_lve_sqrt.tolist(),
        "eye_forehead_eve_sqrt": eye_forehead_eve.tolist(),
        "mean_vertex_l2": mean_vertex_l2.tolist(),
        "facediffuser_mve_mean": float(facediffuser_mve.mean()),
        "facediffuser_lve_sq_mean": float(facediffuser_lve_sq.mean()),
        "vertex_lve_sqrt_mean": float(vertex_lve_sqrt.mean()),
        "eye_forehead_eve_sqrt_mean": float(eye_forehead_eve.mean()),
        "mean_vertex_l2_mean": float(mean_vertex_l2.mean()),
        "coordinate_units": "as provided by the rig (convert to mm before reporting)",
        "definitions": {
            "facediffuser_mve": "mean_t mean_v ||pred_t(v)-gt_t(v)||_2",
            "facediffuser_lve_sq": "mean_t max_v_in_lips ||pred_t(v)-gt_t(v)||_2^2",
            "vertex_lve_sqrt": "mean_t max_v_in_lips ||pred_t(v)-gt_t(v)||_2",
            "eye_forehead_eve_sqrt": "mean_t max_v_in_eye_forehead ||pred_t(v)-gt_t(v)||_2",
        },
        "channel_mask_used": True,
        "channel_mask_shape": list(channel_mask.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="npz: prediction,target,valid,channel_mask")
    parser.add_argument("--rig", type=Path, required=True,
                        help="npz: neutral_vertices,blendshape_deltas")
    parser.add_argument("--regions", type=Path, default=None,
                        help="optional npz with lip_mask and eye_forehead_mask")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as z:
        values = {k: z[k] for k in ("prediction", "target", "valid", "channel_mask")}
    with np.load(args.rig, allow_pickle=False) as z:
        neutral = z["neutral_vertices"]
        deltas = z["blendshape_deltas"]

    regions = {}
    if args.regions is not None:
        with np.load(args.regions, allow_pickle=False) as z:
            regions = {k: z[k] for k in ("lip_mask", "eye_forehead_mask") if k in z}
    result = evaluate(values["prediction"], values["target"], values["valid"],
                      values["channel_mask"], neutral, deltas, **regions)
    result["input"] = str(args.input.resolve())
    result["rig"] = str(args.rig.resolve())
    if args.regions is not None:
        result["regions"] = str(args.regions.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
