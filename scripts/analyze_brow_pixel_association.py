"""Read-only, same-clock description; no truth gate or threshold search."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np

GROUPS = {"raise": [2, 3, 4], "down": [0, 1]}
SIDES = ["right", "left", "bilateral_mean"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def verify(root, required=()):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if not set(required).issubset(manifest):
        raise ValueError("required consumed input omitted from manifest")
    for name, rec in manifest.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]:
            raise ValueError("frozen input mismatch: " + str(p))
    return sha(root / "manifest.json")


def pearson(x, y):
    if len(x) < 3:
        return None
    a, b = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denom, -1, 1)) if denom > 1e-12 else None


def stat(x, y, mask):
    mask = mask & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if not len(x):
        return {"pairs": 0, "pearson_expected_sign": None, "bs_rms": None, "pixel_rms": None,
                "raw_sign_agreement": None, "absolute_product_weighted_sign_agreement": None}
    weights = np.abs(x * y)
    return {"pairs": int(len(x)), "pearson_expected_sign": pearson(x, y),
            "bs_rms": float(np.sqrt(np.square(x).mean())), "pixel_rms": float(np.sqrt(np.square(y).mean())),
            "raw_sign_agreement": float(np.mean(x * y > 0)),
            "absolute_product_weighted_sign_agreement": float(weights[x * y > 0].sum() / weights.sum()) if weights.sum() else None}



def aligned_evidence(p, s, n, expected_frames=None):
    count = len(p["valid"])
    if (expected_frames is not None and count != expected_frames
            or p["times"].shape != (count,)
            or p["corrected_normalized"].shape != (count, 2, 2)
            or p["corrected_observed"].shape != (count, 2)
            or not np.array_equal(p["valid"], s["valid"])
            or not np.array_equal(p["valid"], n["valid"])
            or not np.array_equal(p["times"], n["times"])
            or not np.allclose(p["times"], np.arange(count) / 25, rtol=0, atol=1e-8)):
        raise ValueError("native clocks, shapes or masks differ")
    if not np.array_equal(p["corrected_observed"], np.isfinite(p["corrected_normalized"]).all(2)):
        raise ValueError("pixel observation mask differs from finite pixel movement")
    if (n["blink_neighborhood_pairs"].shape != (count - 1,)
            or n["blink_observed_pairs"].shape != (count - 1,)
            or np.any(n["blink_neighborhood_pairs"] & ~n["blink_observed_pairs"])):
        raise ValueError("invalid blink pair masks")
    expected_pairs = np.zeros(count, bool)
    expected_pairs[1:] = p["valid"][1:] & p["valid"][:-1]
    if not np.array_equal(p["pair_valid"], expected_pairs) or np.any(n["blink_observed_pairs"] & ~expected_pairs[1:]):
        raise ValueError("difference crosses missing frame or clip boundary")
    dy = p["corrected_normalized"][1:, :, 1]
    dy = np.column_stack([dy, dy.mean(1)])
    support = np.column_stack([p["corrected_observed"][1:], p["corrected_observed"][1:].all(1)])
    arms = {"all": p["pair_valid"][1:],
            "blink_near": n["blink_neighborhood_pairs"],
            "nonblink": n["blink_observed_pairs"] & ~n["blink_neighborhood_pairs"]}
    if np.any(arms["blink_near"] & arms["nonblink"]):
        raise ValueError("blink strata overlap")
    return dy, support, arms


def directional_pixel(group, dy):
    if group not in GROUPS:
        raise ValueError("unsupported brow group")
    return (-1 if group == "raise" else 1) * np.asarray(dy)


def run(pixel_dir, state_dir, nuisance_dir, output):
    PIXEL, STATE, NUISANCE, OUT = map(Path, (pixel_dir, state_dir, nuisance_dir, output))
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("fresh output required")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {"pixel": verify(PIXEL, ["protocol.json"]), "states": verify(STATE, ["clips.json", "protocol.json"]), "nuisance": verify(NUISANCE, ["protocol.json"])}
    rows = json.loads((STATE / "clips.json").read_text(encoding="utf8"))
    if len(rows) != 32 or len({r["source_id"] for r in rows}) != 32:
        raise ValueError("requires fixed32 clips")
    ids = [r["source_id"] for r in rows]
    if any(not isinstance(cid, str) or cid in ("", ".", "..") or any(c in cid for c in "/\\:") for cid in ids):
        raise ValueError("unsafe source_id")
    for directory in (PIXEL, STATE, NUISANCE):
        verify(directory, ["arrays/" + cid + ".npz" for cid in ids])
    pixel_protocol = json.loads((PIXEL / "protocol.json").read_text(encoding="utf8"))
    nuisance_protocol = json.loads((NUISANCE / "protocol.json").read_text(encoding="utf8"))
    if pixel_protocol["selection_sha256"] != nuisance_protocol["selection_sha256"]:
        raise ValueError("locked fit selection differs")
    write(OUT / "protocol.json", {
        "schema": "pixel_bs_direction_analysis_v1", "input_manifest_sha256": manifest_hashes,
        "same_clock": "difference ending at frame t equals pixel-flow array[t]; no lag optimization",
        "sign": "image y increases downwards: raise compares ΔBS to −dy, down compares ΔBS to +dy",
        "blink": "reuse frozen >.5±2-frame neighborhood, both endpoints; rest only blink-observable pairs",
        "correlation": "signed Pearson of adjacent first differences; raw and SG5 smooth coefficients separately",
        "descriptive_discordance": "upper-half within32 mean-group BS RMS and lower-half pixel RMS on nonblink shared support; ranks for inspection only, no label acceptance threshold",
        "joint_support": "positive expected-sign bilateral correlation, with per-side and per-clip numbers; not semantic GT or causal evidence",
        "episode": "onset/release direction measured separately by actual net BS and summed pixel displacement on shared support; no clip-level sign substitute",
        "threshold_fitting": False, "independent_ground_truth": False, "training_labels_changed": False,
        "code_sha256": sha(__file__),
    })
    records, pooled, segments = [], {}, []
    for group in GROUPS:
        for side in SIDES:
            for source in ("raw", "smooth"):
                for arm in ("all", "blink_near", "nonblink"):
                    pooled[group, side, source, arm] = [[], []]

    for row in rows:
        cid = row["source_id"]
        with np.load(PIXEL / "arrays" / (cid + ".npz"), allow_pickle=False) as z:
            p = {k: z[k].copy() for k in z.files}
        with np.load(STATE / "arrays" / (cid + ".npz"), allow_pickle=False) as z:
            s = {k: z[k].copy() for k in z.files}
        with np.load(NUISANCE / "arrays" / (cid + ".npz"), allow_pickle=False) as z:
            n = {k: z[k].copy() for k in z.files}
        dy, support, arms = aligned_evidence(p, s, n)
        result = {"source_id": cid, "metadata": row["metadata"], "groups": {}}
        for group, channels in GROUPS.items():
            sign = directional_pixel(group, 1.)
            values = {source: np.diff(s[source + "5"][:, channels].mean(1)) for source in ("raw", "smooth")}
            out = {}
            for j, side in enumerate(SIDES):
                out[side] = {}
                for source, x in values.items():
                    out[side][source] = {}
                    for arm, mask in arms.items():
                        observed = support[:, j] & mask & np.isfinite(x) & np.isfinite(dy[:, j])
                        out[side][source][arm] = stat(x, sign * dy[:, j], observed)
                        target = pooled[group, side, source, arm]
                        target[0].append(x[observed])
                        target[1].append(sign * dy[observed, j])
            result["groups"][group] = out
            for kind in ("transitions", "episodes"):
                for index, segment in enumerate(row["groups"][group][kind]):
                    intervals = [("transition", segment["start"], segment["stop"] - 1)] if kind == "transitions" else [
                        ("onset", segment["start"], segment["peak"]),
                        ("release", segment["release"], segment["stop"] - 1)]
                    for phase, start, end in intervals:
                        mask = np.zeros(len(dy), bool)
                        mask[start:end] = True
                        observed = mask & support[:, 2] & arms["all"]
                        nonblink = observed & arms["nonblink"]
                        x = values["smooth"]
                        measured = {arm: stat(x, sign * dy[:, 2], chosen) for arm, chosen in [("all", observed), ("nonblink", nonblink)]}
                        net_bs = float(x[observed].sum())
                        net_pixel = float(sign * dy[observed, 2].sum())
                        segments.append({"source_id": cid, "group": group, "kind": kind, "index": index,
                                         "phase": phase, "start": start, "end": end,
                                         "phase_identified": kind == "episodes", "conflict": segment.get("conflict", False),
                                         "left_censored": segment["left_censored"], "right_censored": segment["right_censored"],
                                         "expected_pairs": end - start, "matched_pairs": int(observed.sum()),
                                         "net_smooth_bs_on_shared_pairs": net_bs,
                                         "net_direction_adjusted_pixel_on_shared_pairs": net_pixel,
                                         "net_direction_agrees": bool(net_bs * net_pixel > 0) if observed.any() else None,
                                         "metrics": measured})
        records.append(result)

    aggregate = {}
    consistency = {}
    ranked = {}
    for group in GROUPS:
        aggregate[group], consistency[group] = {}, {}
        for side in SIDES:
            aggregate[group][side] = {}
            consistency[group][side] = {}
            for source in ("raw", "smooth"):
                aggregate[group][side][source] = {}
                consistency[group][side][source] = {}
                for arm in ("all", "blink_near", "nonblink"):
                    x, y = [np.concatenate(a) for a in pooled[group, side, source, arm]]
                    aggregate[group][side][source][arm] = stat(x, y, np.ones(len(x), bool))
                    correlations = [r["groups"][group][side][source][arm]["pearson_expected_sign"] for r in records]
                    correlations = [v for v in correlations if v is not None]
                    consistency[group][side][source][arm] = {"available_clips": len(correlations),
                                                            "positive_expected_sign_clips": sum(v > 0 for v in correlations),
                                                            "median_correlation": float(np.median(correlations)) if correlations else None,
                                                            "q25_q75": np.quantile(correlations, [.25, .75]).tolist() if correlations else None}
        inspect = []
        for r in records:
            m = r["groups"][group]["bilateral_mean"]["smooth"]["nonblink"]
            if m["pairs"]:
                inspect.append({"source_id": r["source_id"], **m,
                                "right_correlation": r["groups"][group]["right"]["smooth"]["nonblink"]["pearson_expected_sign"],
                                "left_correlation": r["groups"][group]["left"]["smooth"]["nonblink"]["pearson_expected_sign"]})
        by_bs = sorted(inspect, key=lambda r: -r["bs_rms"])
        by_pixel = sorted(inspect, key=lambda r: -r["pixel_rms"])
        bs_rank = {r["source_id"]: i + 1 for i, r in enumerate(by_bs)}
        px_rank = {r["source_id"]: i + 1 for i, r in enumerate(by_pixel)}
        for r in inspect:
            r["bs_activity_rank_descending"] = bs_rank[r["source_id"]]
            r["pixel_activity_rank_descending"] = px_rank[r["source_id"]]
        half = len(inspect) // 2
        ranked[group] = {
            "all_ranked_by_bs_rms": sorted(inspect, key=lambda r: r["bs_activity_rank_descending"]),
            "high_bs_low_pixel_for_review": [r for r in inspect if r["bs_activity_rank_descending"] <= half and r["pixel_activity_rank_descending"] > half],
            "both_sides_expected_sign_by_bilateral_correlation": sorted([r for r in inspect if r["right_correlation"] is not None and r["left_correlation"] is not None and r["right_correlation"] > 0 and r["left_correlation"] > 0], key=lambda r: -r["pearson_expected_sign"]),
        }

    segment_summary = {}
    for group in GROUPS:
        segment_summary[group] = {}
        for kind in ("transitions", "episodes"):
            selected = [r for r in segments if r["group"] == group and r["kind"] == kind]
            segment_summary[group][kind] = {"flanks_or_transitions": len(selected),
                                           "net_direction_agrees": sum(r["net_direction_agrees"] is True for r in selected),
                                           "pixel_missing": sum(r["matched_pairs"] == 0 for r in selected),
                                           "phase_semantics_known": kind == "episodes"}
    result = {"schema": "pixel_bs_direction_analysis_v1", "clip_count": 32, "aggregate": aggregate,
              "per_clip_consistency": consistency, "ranked_inspection": ranked,
              "segment_summary": segment_summary, "segments": segments, "clips": records,
              "independent_ground_truth": False, "causal_confidence": False,
              "manifest_inputs_verified": manifest_hashes, "protocol_sha256": sha(OUT / "protocol.json")}
    write(OUT / "analysis.json", result)
    write(OUT / "manifest.json", {p.relative_to(OUT).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--nuisance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pixel, args.states, args.nuisance, args.output)
    print(json.dumps({"clip_count": result["clip_count"], "aggregate_bilateral": {g: result["aggregate"][g]["bilateral_mean"] for g in GROUPS}, "segment_summary": result["segment_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
