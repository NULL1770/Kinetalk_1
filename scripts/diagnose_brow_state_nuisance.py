"""Same-clock nuisance associations on the 32 locked fresh-forward fit clips.

These measurements share a tracker with the brow coefficients. They are
diagnostics, not independent truth, causal confidence, or training labels.
No interpolation, lag search, target fitting, or state-teacher dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "brow_state_nuisance_diagnostic_v1"
FPS = 25.0
GROUPS = {"raise": [43, 44, 45], "down": [41, 42]}
NUISANCES = ["blink_left", "blink_right", "eye_aperture_right", "eye_aperture_left",
             "pose_yaw", "pose_pitch", "pose_roll"]
BLINK_THRESHOLD, BLINK_RADIUS = .5, 2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf8")


def continuous_runs(valid, adjacent):
    """Return half-open observed runs; clock gaps also split runs."""
    start = None
    for index, supported in enumerate(valid):
        if start is not None and (not supported or (index and not adjacent[index - 1])):
            yield start, index
            start = None
        if supported and start is None:
            start = index
    if start is not None:
        yield start, len(valid)


def dilate_within_runs(seed, observed, adjacent, radius=BLINK_RADIUS):
    output = np.zeros(len(observed), dtype=bool)
    for left, right in continuous_runs(observed, adjacent):
        for index in np.flatnonzero(seed[left:right]) + left:
            output[max(left, index - radius):min(right, index + radius + 1)] = True
    return output


def correlation(x, y, support):
    mask = np.asarray(support, bool) & np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 3:
        return {"pearson": None, "pairs": int(len(x)), "reason": "fewer_than_three_pairs"}
    a, b = x - x.mean(), y - y.mean()
    scale_a, scale_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if scale_a <= 1e-12 or scale_b <= 1e-12:
        return {"pearson": None, "pairs": int(len(x)), "reason": "constant_difference"}
    return {"pearson": float(np.clip(np.dot(a, b) / scale_a / scale_b, -1., 1.)),
            "pairs": int(len(x)), "reason": None}


def _optional(arrays, key, shape):
    if key not in arrays:
        return np.full(shape, np.nan), "missing_array"
    value = np.asarray(arrays[key], np.float64)
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
    return value.copy(), None


def diagnose_clip(arrays, source_id):
    """Return JSON diagnostics and plot-ready NPZ arrays without mutating input."""
    for key in ("coeffs", "valid", "times"):
        if key not in arrays:
            raise ValueError("required fresh-forward array missing: " + key)
    coeffs = np.asarray(arrays["coeffs"], np.float64)
    valid = np.asarray(arrays["valid"])
    times = np.asarray(arrays["times"], np.float64)
    n = len(times)
    if times.ndim != 1 or not n or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be a nonempty finite strictly increasing vector")
    if coeffs.shape != (n, 52) or valid.shape != (n,) or valid.dtype != np.bool_:
        raise ValueError("coeffs[T,52] and Boolean valid[T] required")
    geometry, geometry_missing = _optional(arrays, "geometry", (n, 4))
    pose, pose_missing = _optional(arrays, "head_pose", (n, 3))
    clock_adjacent = np.isclose(np.diff(times), 1 / FPS, rtol=0, atol=1e-7)
    adjacent = valid[:-1] & valid[1:] & clock_adjacent
    nuisance = np.column_stack([coeffs[:, 0], coeffs[:, 7], geometry[:, 2:4], pose])
    nuisance_valid = valid[:, None] & np.isfinite(nuisance)
    nuisance_diff = np.diff(nuisance, axis=0)
    # Euler components use the shortest signed angular displacement in degrees.
    nuisance_diff[:, 4:] = (nuisance_diff[:, 4:] + 180.) % 360. - 180.
    nuisance_pairs = adjacent[:, None] & nuisance_valid[:-1] & nuisance_valid[1:]
    nuisance_diff[~nuisance_pairs] = np.nan
    availability = {}
    for column, name in enumerate(NUISANCES):
        missing = geometry_missing if 2 <= column < 4 else pose_missing if column >= 4 else None
        count = int(nuisance_valid[:, column].sum())
        availability[name] = {"observed_frames": count, "observed_pairs": int(nuisance_pairs[:, column].sum()),
                              "missing_reason": missing or ("no_finite_valid_observations" if not count else None)}

    # Require both blink sides to be observed, so NaN never implies a negative.
    blink_observed = nuisance_valid[:, :2].all(axis=1)
    blink_seed = blink_observed & np.any(nuisance[:, :2] > BLINK_THRESHOLD, axis=1)
    blink_near = dilate_within_runs(blink_seed, blink_observed, adjacent)
    blink_pairs = adjacent & blink_observed[:-1] & blink_observed[1:]
    # A transition is near a blink only when both endpoints lie in the ±2 window.
    blink_near_pairs = blink_pairs & blink_near[:-1] & blink_near[1:]
    output = {"times": times.copy(), "valid": valid.copy(), "adjacent_valid": adjacent,
              "transition_times": times[1:].copy(), "nuisance_names": np.asarray(NUISANCES),
              "nuisance_values": nuisance, "nuisance_valid": nuisance_valid,
              "nuisance_differences": nuisance_diff, "nuisance_pair_valid": nuisance_pairs,
              "blink_observed": blink_observed, "blink_seed": blink_seed,
              "blink_neighborhood": blink_near, "blink_observed_pairs": blink_pairs,
              "blink_neighborhood_pairs": blink_near_pairs}
    groups = {}
    for group, channels in GROUPS.items():
        value = coeffs[:, channels]
        group_valid = valid & np.isfinite(value).all(axis=1)
        pair_valid = adjacent & group_valid[:-1] & group_valid[1:]
        mean = value.mean(axis=1)
        difference = np.diff(mean)
        energy = np.sum(np.diff(value, axis=0) ** 2, axis=1)
        difference[~pair_valid] = np.nan
        energy[~pair_valid] = np.nan
        observed_pairs = pair_valid & blink_pairs
        near_pairs = observed_pairs & blink_near_pairs
        total_energy = float(energy[pair_valid].sum())
        observed_energy = float(energy[observed_pairs].sum())
        near_energy = float(energy[near_pairs].sum())
        groups[group] = {
            "channels": channels, "observed_frames": int(group_valid.sum()),
            "observed_pairs": int(pair_valid.sum()),
            "difference_correlations": {name: correlation(difference, nuisance_diff[:, i],
                                                            pair_valid & nuisance_pairs[:, i])
                                        for i, name in enumerate(NUISANCES)},
            "blink_neighborhood_energy": {
                "all_observed_brow_energy": total_energy,
                "blink_observable_brow_energy": observed_energy,
                "near_blink_brow_energy": near_energy,
                "near_blink_fraction": near_energy / observed_energy if observed_energy > 0 else None,
                "reason": None if observed_energy > 0 else "no_positive_blink_observable_brow_energy",
                "blink_observable_brow_pairs": int(observed_pairs.sum()),
                "near_blink_brow_pairs": int(near_pairs.sum()),
                "blink_observable_pair_fraction": float(observed_pairs.sum() / pair_valid.sum()) if pair_valid.any() else None,
                "blink_observable_energy_fraction": observed_energy / total_energy if total_energy > 0 else None,
            },
        }
        output.update({group + "_mean": mean, group + "_valid": group_valid,
                       group + "_difference": difference, group + "_pair_valid": pair_valid,
                       group + "_difference_energy": energy})
    report = {"schema": SCHEMA, "source_id": source_id, "frames": n, "valid_frames": int(valid.sum()),
              "clock_gap_transitions": int((~clock_adjacent).sum()),
              "valid_runs": [[int(a), int(b)] for a, b in continuous_runs(valid, adjacent)],
              "availability": availability, "blink_above_threshold_frames": int(blink_seed.sum()),
              "blink_neighborhood_frames": int(blink_near.sum()), "groups": groups,
              "independent_ground_truth": False, "causal_confidence": False,
              "used_for_training": False}
    return report, output


def _verified(audit_dir, manifest, relative):
    path = audit_dir / relative
    record = manifest.get(relative)
    if not isinstance(record, dict) or not path.is_file():
        raise ValueError("missing locked input or manifest entry: " + relative)
    if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha(path):
        raise ValueError("locked input hash/size mismatch: " + relative)
    return path


def run(audit_dir, output):
    audit_dir, output = Path(audit_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be a fresh directory")
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf8"))
    selection_path = _verified(audit_dir, manifest, "selection.json")
    provenance_path = _verified(audit_dir, manifest, "provenance.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    if selection.get("schema") != "tracking_reset_selection_v1":
        raise ValueError("requires tracking-reset locked selection")
    if provenance.get("schema") != "tracking_reset_order_diagnostic_v1" or provenance.get("selection_sha256") != sha(selection_path):
        raise ValueError("tracking provenance/selection mismatch")
    if provenance.get("fps") != FPS or "fresh_forward" not in provenance.get("arms", []):
        raise ValueError("requires 25 Hz fresh_forward arm")
    rows = selection["clips"]
    ids = [row["clip_id"] for row in rows]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("requires exactly32 unique locked fit clips")
    if any(not isinstance(cid, str) or cid in ("", ".", "..") or "/" in cid or "\\" in cid or ":" in cid for cid in ids):
        raise ValueError("unsafe clip id")
    inputs = {cid: _verified(audit_dir, manifest, "arrays/" + cid + "_fresh_forward.npz") for cid in ids}
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir()
    (output / "clips").mkdir()
    protocol = {"schema": SCHEMA, "source_arm": "fresh_forward", "fps": FPS,
                "selection_sha256": sha(selection_path), "source_manifest_sha256": sha(audit_dir / "manifest.json"),
                "source_provenance_sha256": sha(provenance_path), "code_sha256": sha(__file__),
                "brow_groups": GROUPS, "nuisance_names": NUISANCES,
                "correlation": "Pearson of signed group-mean first difference and nuisance first difference on identical adjacent native frames; no lag search",
                "pose_difference": "shortest signed Euler component displacement in degrees",
                "blink": {"channels": [0, 7], "strict_threshold": BLINK_THRESHOLD, "radius_frames": BLINK_RADIUS,
                          "missing": "both sides must be finite to classify a frame",
                          "dilation": "within contiguous valid, blink-observed, 25Hz runs only",
                          "transition_membership": "both endpoints inside dilated neighborhood"},
                "energy": "sum of squared channel first differences; blink fraction denominator uses only blink-observable valid pairs",
                "missing_policy": "missing geometry/pose or nonfinite observations excluded and reported; constant correlations are null",
                "independent_ground_truth": False, "causal_confidence": False,
                "used_for_training": False, "query_motion_used": False}
    write_json(output / "protocol.json", protocol)
    records, plot_arrays = [], []
    for row in rows:
        cid = row["clip_id"]
        with np.load(inputs[cid], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("coeffs", "valid", "times", "geometry", "head_pose") if key in archive}
        report, values = diagnose_clip(arrays, cid)
        report["metadata"] = {key: row[key] for key in ("speaker_name", "emotion", "sentence", "video_sha256") if key in row}
        report["source_npz_sha256"] = manifest["arrays/" + cid + "_fresh_forward.npz"]["sha256"]
        write_json(output / "clips" / (cid + ".json"), report)
        np.savez_compressed(output / "arrays" / (cid + ".npz"), **values)
        records.append(report)
        plot_arrays.append(values)
    pooled = {}
    for group in GROUPS:
        pairs = np.concatenate([a[group + "_pair_valid"] for a in plot_arrays])
        delta = np.concatenate([a[group + "_difference"] for a in plot_arrays])
        nuisance_delta = np.concatenate([a["nuisance_differences"] for a in plot_arrays])
        nuisance_pairs = np.concatenate([a["nuisance_pair_valid"] for a in plot_arrays])
        energy_rows = [r["groups"][group]["blink_neighborhood_energy"] for r in records]
        summed = {key: sum(r[key] for r in energy_rows) for key in
                  ("all_observed_brow_energy", "blink_observable_brow_energy", "near_blink_brow_energy",
                   "blink_observable_brow_pairs", "near_blink_brow_pairs")}
        denominator = summed["blink_observable_brow_energy"]
        summed["near_blink_fraction"] = summed["near_blink_brow_energy"] / denominator if denominator > 0 else None
        summed["blink_observable_pair_fraction"] = summed["blink_observable_brow_pairs"] / int(pairs.sum()) if pairs.any() else None
        total_energy = summed["all_observed_brow_energy"]
        summed["blink_observable_energy_fraction"] = denominator / total_energy if total_energy > 0 else None
        pooled[group] = {"observed_pairs": int(pairs.sum()), "blink_neighborhood_energy": summed,
                         "difference_correlations": {name: correlation(delta, nuisance_delta[:, i], pairs & nuisance_pairs[:, i])
                                                     for i, name in enumerate(NUISANCES)}}
    summary = {"schema": SCHEMA, "clip_count": len(records), "frames": sum(r["frames"] for r in records),
               "pooled_groups": pooled, "pooling": "concatenate precomputed within-clip differences; never difference across clips",
               "independent_ground_truth": False, "causal_confidence": False, "used_for_training": False,
               "clips": records, "protocol_sha256": sha(output / "protocol.json")}
    write_json(output / "summary.json", summary)
    # Verify the read-only sources did not change while producing diagnostics.
    for relative in ("selection.json", "provenance.json", *("arrays/" + cid + "_fresh_forward.npz" for cid in ids)):
        _verified(audit_dir, manifest, relative)
    write_json(output / "manifest.json", {p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                                         for p in sorted(output.rglob("*")) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.audit, args.output)
    print(json.dumps({"schema": SCHEMA, "clip_count": result["clip_count"], "pooled_groups": result["pooled_groups"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
