"""Evaluate mesh-space errors from ARKit coefficients and a declared rig.

The evaluator never infers geometry from coefficient arrays. It requires a
neutral mesh and 52 matching blendshape deltas. Inputs are ``[T,52]`` or
``[S,T,52]`` predictions and ``[T,52]`` targets.

Both conventions encountered in the literature are emitted:
``facediffuser_mve`` is mean per-vertex Euclidean error; the official
FaceDiffuser-code lip metric is ``facediffuser_lve_sq`` (maximum squared
lip-vertex error per frame, averaged over time); ``vertex_lve_sqrt`` is the
same maximum-over-lips metric with the square root retained, matching the
metric description used by EmoTalk. Squared metrics retain squared coordinate
units; the report records the conversion used for paper-facing millimetre
values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _vertices(coeff: np.ndarray, neutral: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Map [T,52] ARKit coefficients to [T,V,3] vertices."""
    return neutral[None] + np.einsum("tc,cvd->tvd", coeff, deltas, optimize=True)


def _array_sha256(value: np.ndarray) -> str:
    """Hash an array by shape, dtype and contiguous bytes.

    The shape and dtype are included so that a byte-identical view with a
    different interpretation cannot silently be treated as the same rig.
    """
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rig_identity(neutral: np.ndarray, deltas: np.ndarray, *,
                 coordinate_scale_to_mm: float = 1.0,
                 coordinate_unit: str = "rig") -> dict:
    """Return a reproducible identity for the fixed coefficient-to-mesh rig."""
    neutral = np.asarray(neutral)
    deltas = np.asarray(deltas)
    payload = {
        "neutral_sha256": _array_sha256(neutral),
        "blendshape_deltas_sha256": _array_sha256(deltas),
        "coordinate_scale_to_mm": float(coordinate_scale_to_mm),
        "coordinate_unit": str(coordinate_unit),
        "vertices": int(neutral.shape[0]) if neutral.ndim == 2 else None,
        "channels": int(deltas.shape[0]) if deltas.ndim == 3 else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    payload["rig_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _check_mask(name: str, value, n_vertices: int) -> np.ndarray:
    if value is None:
        return np.ones(n_vertices, dtype=bool)
    value = np.asarray(value, dtype=bool)
    if value.shape != (n_vertices,) or not value.any():
        raise ValueError(f"{name} must be a non-empty [V] boolean mask")
    return value


def evaluate(prediction, target, valid, channel_mask, neutral, deltas,
             lip_mask=None, eye_forehead_mask=None, expression_mask=None,
             fdd_mask=None, coordinate_scale_to_mm=1.0,
             coordinate_unit="rig", coefficient_support=None):
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
    if not valid.any():
        raise ValueError("no valid frames")

    # Apply the channel mask identically to prediction and target. Unobserved
    # channels therefore cannot create a geometry difference.
    observed = valid[:, None] & support
    # Missing channels/frames may legitimately carry NaN or inf sentinels;
    # only supervised values are required to be finite.  This matches the
    # coefficient-space evaluator and prevents a display-only fill from
    # becoming a vertex metric failure.
    if not np.isfinite(target[observed]).all() or not np.isfinite(prediction[:, observed]).all():
        raise ValueError("nonfinite coefficient input on observed values")
    target_masked = np.where(observed, target, 0.0)
    pred_masked = np.where(observed[None], prediction, 0.0)
    gt = _vertices(target_masked, neutral, deltas)
    samples = np.stack([_vertices(x, neutral, deltas) for x in pred_masked], axis=0)

    # [S,T_valid,V] Euclidean vertex distances.
    distance = np.linalg.norm(samples - gt[None], axis=-1)[:, valid]
    if distance.size == 0:
        raise ValueError("no valid frames")

    if not np.isfinite(coordinate_scale_to_mm) or coordinate_scale_to_mm <= 0:
        raise ValueError("coordinate_scale_to_mm must be finite and positive")

    all_mask = np.ones(neutral.shape[0], dtype=bool)
    lip_mask = _check_mask("lip_mask", lip_mask, neutral.shape[0])
    if expression_mask is not None and eye_forehead_mask is not None:
        a = np.asarray(expression_mask, dtype=bool)
        b = np.asarray(eye_forehead_mask, dtype=bool)
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError("expression_mask and eye_forehead_mask disagree")
    if expression_mask is None:
        expression_mask = eye_forehead_mask
    expression_mask = _check_mask("expression_mask", expression_mask, neutral.shape[0])
    if fdd_mask is None:
        fdd_mask = expression_mask
    fdd_mask = _check_mask("fdd_mask", fdd_mask, neutral.shape[0])

    # Distances are kept in the rig's native coordinates internally and then
    # converted once for all reported geometric errors.  This avoids mixing
    # a mesh exported in metres with a mesh exported in millimetres.
    distance = distance * float(coordinate_scale_to_mm)

    # FaceDiffuser MVE official code: mean over vertices and then frames.
    facediffuser_mve = distance[:, :, all_mask].mean(axis=(1, 2))
    # Official FaceDiffuser LVE code uses max squared distance over lip verts.
    lip_sq = np.square(distance[:, :, lip_mask])
    facediffuser_lve_sq = lip_sq.max(axis=-1).mean(axis=-1)
    # EmoTalk-style textual definition: max Euclidean distance per frame.
    vertex_lve_sqrt = distance[:, :, lip_mask].max(axis=-1).mean(axis=-1)
    eye_forehead_eve = distance[:, :, expression_mask].max(axis=-1).mean(axis=-1)
    # Paper-style region metrics used by FaceFormer/EmoTalk tables report the
    # mean Euclidean error over the region, rather than the max-vertex
    # diagnostic above.  Keep both so that the protocol is explicit.
    lve_mean = distance[:, :, lip_mask].mean(axis=(1, 2))
    eve_mean = distance[:, :, expression_mask].mean(axis=(1, 2))

    # Vertex-space FDD follows the FaceDiffuser energy definition: compute the
    # per-frame squared displacement energy in a declared region, take its
    # population temporal standard deviation, then compare prediction to GT.
    # It is intentionally order-invariant and therefore is not a lip/eyebrow
    # timing metric.  Incomplete coefficient frames are excluded entirely so
    # masked fills cannot change the energy statistic.
    required = np.ones(52, dtype=bool) if coefficient_support is None else np.asarray(coefficient_support)
    if required.shape != (52,) or required.dtype != bool or not required.any():
        raise ValueError('coefficient_support must declare nonempty Boolean [52] train-protocol support')
    if np.any(support[:, ~required]):
        raise ValueError('Observation mask includes coefficients outside the declared scoring support')
    complete = valid & np.all(support[:, required], axis=1)
    fdd_signed = None
    fdd_absolute = None
    if int(complete.sum()) >= 2:
        gt_region = (gt[:, fdd_mask, :] - neutral[None, fdd_mask, :]) * float(coordinate_scale_to_mm)
        pred_region = (samples[:, :, fdd_mask, :] - neutral[None, None, fdd_mask, :]) * float(coordinate_scale_to_mm)
        gt_energy = np.square(gt_region).sum(axis=(1, 2))
        pred_energy = np.square(pred_region[:, complete]).sum(axis=(2, 3))
        gt_std = float(np.std(gt_energy[complete], ddof=0))
        pred_std = np.std(pred_energy, axis=1, ddof=0)
        fdd_signed = gt_std - pred_std
        fdd_absolute = np.abs(fdd_signed)
    elif int(complete.sum()):
        # Keep the shape useful for downstream aggregation while declaring the
        # metric unavailable.  No one-frame temporal standard deviation is
        # reported as a valid FDD.
        fdd_signed = np.full(prediction.shape[0], np.nan, dtype=np.float64)
        fdd_absolute = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    fdd_computed = fdd_signed is not None and bool(np.isfinite(fdd_signed).all())
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
        "lve_mean_mm": lve_mean.tolist(),
        "eve_mean_mm": eve_mean.tolist(),
        "vertex_fdd_signed_mm2": None if fdd_signed is None else fdd_signed.tolist(),
        "vertex_fdd_absolute_mm2": None if fdd_absolute is None else fdd_absolute.tolist(),
        "mean_vertex_l2": mean_vertex_l2.tolist(),
        "facediffuser_mve_mean": float(facediffuser_mve.mean()),
        "facediffuser_lve_sq_mean": float(facediffuser_lve_sq.mean()),
        "vertex_lve_sqrt_mean": float(vertex_lve_sqrt.mean()),
        "eye_forehead_eve_sqrt_mean": float(eye_forehead_eve.mean()),
        "lve_mean_mm_mean": float(lve_mean.mean()),
        "eve_mean_mm_mean": float(eve_mean.mean()),
        "vertex_fdd_signed_mm2_mean": float(fdd_signed.mean()) if fdd_computed else None,
        "vertex_fdd_absolute_mm2_mean": float(fdd_absolute.mean()) if fdd_computed else None,
        "vertex_fdd_status": "computed" if fdd_computed else "pending",
        "fdd_complete_frames": int(complete.sum()),
        "coefficient_support": required.tolist(),
        "unobserved_coefficient_policy": "fixed excluded coefficients remain neutral for both prediction and target; FDD requires all declared supported coefficients",
        "mean_vertex_l2_mean": float(mean_vertex_l2.mean()),
        "coordinate_units": str(coordinate_unit),
        "coordinate_scale_to_mm": float(coordinate_scale_to_mm),
        "reported_distance_unit": "mm",
        "reported_fdd_unit": "mm^2",
        "rig_identity": rig_identity(neutral, deltas,
                                      coordinate_scale_to_mm=coordinate_scale_to_mm,
                                      coordinate_unit=coordinate_unit),
        "definitions": {
            "facediffuser_mve": "mean_t mean_v ||pred_t(v)-gt_t(v)||_2",
            "facediffuser_lve_sq": "mean_t max_v_in_lips ||pred_t(v)-gt_t(v)||_2^2",
            "vertex_lve_sqrt": "mean_t max_v_in_lips ||pred_t(v)-gt_t(v)||_2",
            "eye_forehead_eve_sqrt": "mean_t max_v_in_expression ||pred_t(v)-gt_t(v)||_2",
            "lve_mean_mm": "mean_t mean_v_in_lips ||pred_t(v)-gt_t(v)||_2, reported in mm",
            "eve_mean_mm": "mean_t mean_v_in_expression ||pred_t(v)-gt_t(v)||_2, reported in mm",
            "vertex_fdd_signed_mm2": "std_t(sum_v_in_fdd_region ||gt_t(v)-neutral(v)||_2^2) - std_t(sum_v_in_fdd_region ||pred_t(v)-neutral(v)||_2^2)",
            "vertex_fdd_absolute_mm2": "absolute value of vertex_fdd_signed_mm2; lower is better",
        },
        "channel_mask_used": True,
        "channel_mask_shape": list(channel_mask.shape),
        "region_masks": {
            "lip_vertices": int(lip_mask.sum()),
            "expression_vertices": int(expression_mask.sum()),
            "fdd_vertices": int(fdd_mask.sum()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="npz: prediction,target,valid,channel_mask")
    parser.add_argument("--rig", type=Path, required=True,
                        help="npz: neutral_vertices,blendshape_deltas")
    parser.add_argument("--regions", type=Path, default=None,
                        help="optional npz with lip_mask and expression_mask (eye_forehead_mask is accepted as a legacy alias)")
    parser.add_argument("--coordinate-scale-to-mm", type=float, default=None,
                        help="multiply rig coordinates by this value before reporting distance errors; defaults to rig metadata or 1")
    parser.add_argument("--coordinate-unit", default=None,
                        help="declared source coordinate unit; defaults to rig metadata or 'rig'")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as z:
        values = {k: z[k] for k in ("prediction", "target", "valid", "channel_mask")}
    with np.load(args.rig, allow_pickle=False) as z:
        neutral = z["neutral_vertices"]
        deltas = z["blendshape_deltas"]
        rig_scale = float(np.asarray(z["coordinate_scale_to_mm"]).item()) if "coordinate_scale_to_mm" in z else 1.0
        rig_unit = str(np.asarray(z["coordinate_unit"]).item()) if "coordinate_unit" in z else "rig"
    coordinate_scale_to_mm = rig_scale if args.coordinate_scale_to_mm is None else args.coordinate_scale_to_mm
    coordinate_unit = rig_unit if args.coordinate_unit is None else args.coordinate_unit

    regions = {}
    if args.regions is not None:
        with np.load(args.regions, allow_pickle=False) as z:
            regions = {k: z[k] for k in ("lip_mask", "expression_mask", "eye_forehead_mask", "fdd_mask") if k in z}
    result = evaluate(values["prediction"], values["target"], values["valid"],
                      values["channel_mask"], neutral, deltas,
                      coordinate_scale_to_mm=coordinate_scale_to_mm,
                      coordinate_unit=coordinate_unit, **regions)
    result["input"] = str(args.input.resolve())
    result["rig"] = str(args.rig.resolve())
    if args.regions is not None:
        result["regions"] = str(args.regions.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
