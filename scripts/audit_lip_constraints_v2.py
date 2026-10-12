"""Audit geometry-only lip interval assets without fitting data or predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v2.phonetic import LipConstraintAsset, SCHEMA


def main():
    p = argparse.ArgumentParser(); p.add_argument("--asset", type=Path, required=True)
    a = p.parse_args()
    asset = LipConstraintAsset.load(a.asset)
    with np.load(a.asset, allow_pickle=False) as d:
        pairs = d["pair_local_indices"]
        xy = d["pairing_xy_distance_mm"]
        j = d["jacobian_mm"]
        rest = d["rest_gap_mm"]
        neutral = d["neutral_mm"]
        delta = d["blendshape_deltas_mm"]
        lip = d["lip_mask"]
        upper = d["upper_candidates"]
        lower = d["lower_candidates"]
        channels = list(d["channels"])
    # Verify the exported affine observer against full rig geometry, including
    # nonzero rest offset. Synthetic coefficients do not constitute motion QA.
    rng = np.random.default_rng(1701)
    coefficients = rng.uniform(0, .4, (20,52))
    vertices = neutral[None] + np.einsum("bc,cvd->bvd", coefficients, delta)
    explicit_gap = vertices[:,pairs[:,0],2]-vertices[:,pairs[:,1],2]
    affine_error = float(np.max(np.abs(explicit_gap-asset.observe(coefficients))))
    singular = np.linalg.svd(j, compute_uv=False)
    source = json.loads(a.asset.with_suffix(".json").read_text(encoding="utf8"))
    rig_sha = hashlib.sha256(Path(source["rig"]).read_bytes()).hexdigest()
    # Front/plan/side views expose mismatched depth, inverted pairs and coarse
    # skin-group selection. This view must not be called a contact calibration.
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    for ax, (u,v), name in zip(axes, ((0,2),(0,1),(1,2)), ("Front: X/Z", "Plan: X/Y", "Side: Y/Z")):
        ax.scatter(neutral[lip,u],neutral[lip,v],s=8,color=".75",label="Lip region")
        ax.scatter(neutral[upper,u],neutral[upper,v],s=12,color="#a23f32",label="Upper candidates")
        ax.scatter(neutral[lower,u],neutral[lower,v],s=12,color="#2969ac",label="Lower candidates")
        for k, (up,lo) in enumerate(pairs):
            ax.plot(neutral[[up,lo],u],neutral[[up,lo],v],"-o",lw=1.2,ms=3,color="#249448")
            ax.annotate(str(k),((neutral[up,u]+neutral[lo,u])/2,(neutral[up,v]+neutral[lo,v])/2),fontsize=7)
        ax.set_title(name);ax.set_xlabel(f"{'XYZ'[u]} (mm)");ax.set_ylabel(f"{'XYZ'[v]} (mm)")
        ax.set_aspect("equal");ax.grid(alpha=.15)
    axes[0].legend(fontsize=7,loc="best")
    fig.suptitle("Provisional rig gap observations; no physiological contact or phonetic calibration")
    fig.tight_layout()
    plot_path = a.asset.with_name(a.asset.stem + "_geometry.png")
    fig.savefig(plot_path,dpi=160);plt.close(fig)
    checks = {
        "schema": SCHEMA,
        "pair_count": int(len(rest)),
        "jacobian_rank": int(np.linalg.matrix_rank(j)),
        "jacobian_singular_values_mm": singular.tolist(),
        "jacobian_rank_at_0p01_mm": int((singular>.01).sum()),
        "jacobian_finite": bool(np.isfinite(j).all()),
        "rest_gap_finite": bool(np.isfinite(rest).all()),
        "pairs_unique": bool(len(set(map(tuple, pairs))) == len(pairs)),
        "pair_distance_max_mm": float(xy.max()),
        "rest_gap_range_mm": [float(rest.min()), float(rest.max())],
        "asset_sha256": asset.digest,
        "rig_sha256_matches_provenance": rig_sha == source["rig_sha256"],
        "affine_geometry_max_error_mm": affine_error,
        "jawOpen_gap_derivative_range_mm": [float(j[:,channels.index("jawOpen")].min()),float(j[:,channels.index("jawOpen")].max())],
        "geometry_plot": str(plot_path.resolve()),
        "provisional": asset.provisional,
        "physiological_contact_claim": False,
        "data_dependent_calibration": False,
        "status": "PASS_GEOMETRY_ONLY" if (np.isfinite(j).all() and np.isfinite(rest).all() and
            affine_error<.001 and rig_sha==source["rig_sha256"]) else "FAIL",
    }
    report = a.asset.with_name(a.asset.stem + "_audit.json")
    report.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(checks, ensure_ascii=False))
    if checks["status"] != "PASS_GEOMETRY_ONLY":
        raise ValueError("Rig observation audit failed")


if __name__ == "__main__":
    main()
