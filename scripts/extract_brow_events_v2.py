"""Second frozen fit-only brow excursion teacher; not semantic ground truth.

This extractor is deliberately independent of generated/query motion. It replaces
the proposal's unspecified velocity hysteresis with prominence and four-frame
quiet supports. ``protocol.json`` is written before the real arrays are loaded.
No post-hoc threshold search or global interpolation across invalid frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

SCHEMA = "fixed_brow_excursion_teacher_v2"
GROUPS = {"raise": [2, 3, 4], "down": [0, 1]}
BROW52 = [41, 42, 43, 44, 45]
PROTOCOL = {
    "schema": SCHEMA, "fps": 25.0, "channels": list(range(5)), "groups": GROUPS,
    "input_channels": "exactly5 brow-order channels, or52 ARKit channels selected at [41,42,43,44,45]; all other shapes rejected",
    "savgol_window": 5, "savgol_polyorder": 2, "edge_exclusion": 2,
    "absolute_prominence_floor": .035, "noise_mad_multiplier": 6.0,
    "noise_proxy": "unscaled MAD of group mean raw minus SG5/2, pooled fit valid interior frames",
    "quiet_range_fraction_of_threshold": .25, "support_low_fraction_of_prominence": .1,
    "minimum_onset_frames": 4, "minimum_release_frames": 4, "minimum_hold_frames": 4,
    "apex": "exact numerical plateau from scipy.find_peaks; zero duration allowed",
    "group_projection": "arithmetic mean; inspect positive peaks and negative valleys separately",
    "supports": "nearest four-frame low-range window on each side, below peak minus .9 prominence; windows may not cross invalid runs",
    "overlap": "same-group complete candidates that overlap or have fewer than four interevent quiet frames are composite; different groups may overlap and remain separate unless phase boundaries meet merge tolerance; unsupported/censored candidates never veto complete candidates",
    "merge": "different groups and all four phase boundaries within two frames; preserve joint signed marks",
    "edges": "left/right extrema in residual spans outside complete supported excursions and missing supports are retained as censored; do not count the two flanks of an explained pulse again as opposite-sign edge excursions",
    "waiting": "complement intervals carry left/right censoring; intervals adjacent to rejected candidates are unsupported, not reliable HOLD",
    "deviation_from_proposal": "prominence plus quiet supports replaces unspecified joint-speed hysteresis; all thresholds/timing unchanged from v1",
    "deviation_from_v1": "remove unsupported/censored blanket veto; restrict conflicting-complete gate to shared groups; unsupported masks exclude accepted event frames; this is an implementation correction after v1 audit, not an independent experiment or semantic validation",
    "verified_semantic_ground_truth": False, "query_motion_used": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def valid_runs(valid):
    mask = np.asarray(valid, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def brow5(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] not in (5, 52):
        raise ValueError("raw must have exactly5 brow or52 ARKit channels")
    return raw.copy() if raw.shape[1] == 5 else raw[:, BROW52].copy()


def smooth_valid(raw, valid):
    raw = brow5(raw)
    if len(valid) != len(raw):
        raise ValueError("valid must have T entries")
    if not np.isfinite(raw[np.asarray(valid, bool)]).all():
        raise ValueError("nonfinite valid brow coefficients")
    result = raw.copy()
    for left, right in valid_runs(valid):
        if right-left >= 5:
            result[left:right] = savgol_filter(raw[left:right], 5, 2, axis=0, mode="interp")
    return result


def fit_noise_thresholds(clips):
    """clips is a fit-only iterable of (raw, valid); no per-query adaptation."""
    residuals = {name: [] for name in GROUPS}
    clip_count = 0
    for raw, valid in clips:
        clip_count += 1
        raw = brow5(raw)
        smooth = smooth_valid(raw, valid)
        for left, right in valid_runs(valid):
            if right-left < 5:
                continue
            for name, channels in GROUPS.items():
                residuals[name].extend((np.asarray(raw)[left+2:right-2, channels].mean(axis=1)
                                       - smooth[left+2:right-2, channels].mean(axis=1)).tolist())
    result = {}
    for name, values in residuals.items():
        a = np.asarray(values, dtype=float)
        mad = float(np.median(np.abs(a-np.median(a)))) if len(a) else 0.
        result[name] = {"threshold": max(.035, 6*mad), "residual_mad": mad,
                        "residual_samples": len(a), "fit_clip_count": clip_count,
                        "noise_is_independent_measurement": False}
    return result


def _quiet_support(z, peak, prominence, threshold, left):
    limit = z[peak]-.9*prominence
    windows = range(peak-4, -1, -1) if left else range(peak+1, len(z)-3)
    for start in windows:
        values = z[start:start+4]
        if len(values) == 4 and np.ptp(values) <= .25*threshold and max(values) <= limit:
            return start, start+3
    return None


def _event_from_bounds(smooth, bounds, groups, source_id, run_index):
    start, peak, release, end = map(int, bounds)
    base = smooth[start]
    apex = smooth[peak]
    final = smooth[end]
    return {"source_id": source_id, "run_index": run_index, "start": start, "peak": peak,
            "release": release, "end": end, "d_onset": peak-start, "d_apex": release-peak,
            "d_release": end-release, "baseline5": base.tolist(), "p5": (apex-base).tolist(),
            "r5": (final-base).tolist(), "peak5": apex.tolist(), "terminal5": final.tolist(), "groups": groups,
            "left_censored": False, "right_censored": False,
            "verified_semantic_ground_truth": False}


def _one_group(z, threshold, offset, source_id, run_index, group):
    candidates = []
    for direction in (1, -1):
        signed = z*direction
        peaks, props = find_peaks(signed, prominence=threshold, plateau_size=(1, None))
        for index, peak in enumerate(peaks):
            prominence = float(props["prominences"][index])
            left = _quiet_support(signed, int(peak), prominence, threshold, True)
            right = _quiet_support(signed, int(peak), prominence, threshold, False)
            p, q = int(props["left_edges"][index]), int(props["right_edges"][index])
            start, end = (left[1] if left else 0), (right[0] if right else len(z)-1)
            reasons = []
            if left is None: reasons.append("left_censored_no_quiet_support")
            if right is None: reasons.append("right_censored_no_quiet_support")
            if p-start < 4: reasons.append("onset_shorter_than_four_frames")
            if end-q < 4: reasons.append("release_shorter_than_four_frames")
            candidates.append({"source_id": source_id, "run_index": run_index, "group": group,
                "direction": direction, "prominence": prominence, "threshold": threshold,
                "start": start+offset, "peak": p+offset, "release": q+offset, "end": end+offset,
                "left_censored": left is None, "right_censored": right is None,
                "reasons": reasons, "accepted": not reasons})
    # Ordinary peak finding drops boundary extrema. Only residual edge spans
    # outside supported complete excursions count, so an isolated pulse is not
    # spuriously counted again as two opposite-sign censored excursions.
    complete = [c for c in candidates if c["accepted"]]
    left_stop = min(c["start"]-offset for c in complete) if complete else len(z)-1
    right_start = max(c["end"]-offset for c in complete) if complete else 0
    for direction in (1, -1):
        signed = z*direction
        for side in ("left", "right"):
            boundary = 0 if side == "left" else len(z)-1
            stop = left_stop if side == "left" else len(z)-1
            begin = right_start if side == "right" else 0
            segment = signed[begin:stop+1]
            if len(segment) < 2: continue
            trough = begin+int(np.argmin(segment))
            prominence = float(signed[boundary]-signed[trough])
            if prominence < threshold: continue
            start, end = sorted((boundary, trough))
            candidates.append({"source_id": source_id, "run_index": run_index, "group": group,
                "direction": direction, "prominence": prominence, "threshold": threshold,
                "start": start+offset, "peak": boundary+offset, "release": boundary+offset,
                "end": end+offset, "left_censored": side == "left", "right_censored": side == "right",
                "reasons": [side+"_censored_boundary_extremum"], "accepted": False})
    return candidates


def _merge_candidates(candidates, smooth, source_id, run_index):
    good = [c for c in candidates if c["accepted"]]
    groups = []
    for c in sorted(good, key=lambda c: (c["start"], c["peak"], c["group"])):
        match = next((g for g in groups if all(x["group"] != c["group"] for x in g)
                      and max(abs(g[0][key]-c[key]) for key in ("start", "peak", "release", "end")) <= 2), None)
        if match is None: groups.append([c])
        else: match.append(c)
    # Close/overlapping incompatible groups are a composite component. Reject
    # the whole connected component, including a large candidate containing two.
    bounds = [(min(c["start"] for c in g), max(c["end"] for c in g)) for g in groups]
    bad = set()
    for i, (a, b) in enumerate(bounds):
        for j, (c, d) in enumerate(bounds):
            shared = {x["group"] for x in groups[i]} & {x["group"] for x in groups[j]}
            if i < j and shared and min(b, d)+4 > max(a, c): bad.update((i, j))
    events = []
    for index, g in enumerate(groups):
        if index in bad:
            for c in g:
                c["accepted"] = False; c["reasons"].append("overlapping_or_insufficient_hold_composite")
            continue
        phase = [int(round(np.mean([c[k] for c in g]))) for k in ("start", "peak", "release", "end")]
        e = _event_from_bounds(smooth, phase,
             [{"group": c["group"], "direction": c["direction"], "prominence": c["prominence"]} for c in g], source_id, run_index)
        events.append(e)
    return sorted(events, key=lambda e: e["start"])


def extract_clip(raw, valid, thresholds, source_id, fps=25.0):
    """Return (JSON report, raw/smooth/masks arrays); bounds are frame indices.

    ``peak`` is onset end, ``release`` is apex end, and ``end`` is release end.
    P5/R5 use the original coefficient coordinate and ordering [downL, downR,
    innerUp, outerUpL, outerUpR], not logit units. Durations count transitions.
    """
    raw = brow5(raw)
    valid = np.asarray(valid, dtype=bool)
    smooth = smooth_valid(raw, valid)
    all_candidates, events, waits = [], [], []
    stage_counts = {g: {"candidates": 0, "complete_support_before_merge": 0,
                       "accepted_after_same_group_conflict": 0} for g in GROUPS}
    support_mask = np.zeros(len(raw), bool); event_mask = support_mask.copy()
    unsupported_mask = support_mask.copy(); eligible = support_mask.copy()
    runs = valid_runs(valid)
    for run_index, (left, right) in enumerate(runs):
        lo, hi = left+2, right-2
        if hi <= lo: continue
        eligible[lo:hi] = True
        if hi-lo < 9:
            unsupported_mask[lo:hi] = True
            continue
        candidates = []
        for group, channels in GROUPS.items():
            value = thresholds[group]
            threshold = float(value["threshold"] if isinstance(value, dict) else value)
            if not np.isfinite(threshold) or threshold < .035:
                raise ValueError("fixed fit threshold must be finite and >= .035")
            candidates.extend(_one_group(smooth[lo:hi, channels].mean(axis=1), threshold, lo, source_id, run_index, group))
        for g in GROUPS:
            stage_counts[g]["candidates"] += sum(c["group"] == g for c in candidates)
            stage_counts[g]["complete_support_before_merge"] += sum(c["group"] == g and c["accepted"] for c in candidates)
        current = _merge_candidates(candidates, smooth, source_id, run_index)
        for g in GROUPS:
            stage_counts[g]["accepted_after_same_group_conflict"] += sum(c["group"] == g and c["accepted"] for c in candidates)
        for c in candidates:
            if not c["accepted"]:
                unsupported_mask[max(lo, c["start"]):min(hi, c["end"]+1)] = True
        for e in current:
            event_mask[e["start"]:e["end"]+1] = True
            support_mask[e["start"]-3:e["start"]+1] = True
            support_mask[e["end"]:e["end"]+4] = True
        unsupported_mask[event_mask] = False
        points = [(lo, None)] + [(e["end"], e) for e in current]
        for index, (start, prior) in enumerate(points):
            next_event = current[index] if index < len(current) else None
            end = next_event["start"] if next_event else hi-1
            if end < start: continue
            waits.append({"source_id": source_id, "run_index": run_index, "start": start, "end": end,
                          "duration_frames": end-start, "left_censored": prior is None,
                          "right_censored": next_event is None,
                          "supported_hold": bool(end-start >= 4 and not unsupported_mask[start:end+1].any()),
                          "reason": "unsupported_motion_present" if unsupported_mask[start:end+1].any() else "below_event_threshold_not_semantically_verified"})
        all_candidates.extend(candidates); events.extend(current)
    valid_diff = valid[1:] & valid[:-1]
    raw_energy = np.sum(np.diff(raw, axis=0)**2, axis=1)
    smooth_energy = np.sum(np.diff(smooth, axis=0)**2, axis=1)
    total = float(smooth_energy[valid_diff].sum())
    def energy(mask):
        return float(smooth_energy[valid_diff & mask[1:] & mask[:-1]].sum())
    reasons = {}
    for c in all_candidates:
        for reason in c["reasons"]: reasons[reason] = reasons.get(reason, 0)+1
    summary = {"candidate_count": len(all_candidates), "complete_event_count": len(events),
               "accepted_group_candidate_count": sum(c["accepted"] for c in all_candidates),
               "censored_candidate_count": sum(c["left_censored"] or c["right_censored"] for c in all_candidates),
               "unsupported_candidate_count": sum(not c["accepted"] for c in all_candidates),
               "rejection_reasons": reasons, "valid_frames": int(valid.sum()), "eligible_frames": int(eligible.sum()),
               "event_frames": int(event_mask.sum()), "unsupported_frames": int(unsupported_mask.sum()),
               "raw_derivative_energy": float(raw_energy[valid_diff].sum()), "smooth_derivative_energy": total,
               "event_derivative_energy": energy(event_mask), "unsupported_derivative_energy": energy(unsupported_mask),
               "event_energy_fraction": energy(event_mask)/total if total else 0.,
               "unsupported_energy_fraction": energy(unsupported_mask)/total if total else 0.,
               "event_rate_hz": len(events)/(valid.sum()/fps) if valid.sum() else 0.,
               "supported_hold_frames": sum(w["duration_frames"] for w in waits if w["supported_hold"]),
               "stage_counts": stage_counts}
    report = {"schema": SCHEMA, "source_id": source_id, "fps": fps, "runs": runs,
              "verified_semantic_ground_truth": False, "events": events, "candidates": all_candidates,
              "waits": waits, "summary": summary}
    return report, {"raw5": raw, "smooth5": smooth, "valid": valid, "eligible": eligible,
                    "event_mask": event_mask, "unsupported_mask": unsupported_mask, "quiet_support_mask": support_mask}


def run(sidecars, audit, output):
    sidecars, audit, output = map(Path, (sidecars, audit, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be a fresh directory; never overwrite an existing protocol/run")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    protocol.update({"selection_sha256": sha(audit/"selection.json"),
                     "sidecar_manifest_sha256": sha(sidecars/"manifest.json"),
                     "extractor_sha256": sha(__file__), "protocol_frozen_before_array_load": True})
    write_json(output/"protocol.json", protocol)
    selection = json.loads((audit/"selection.json").read_text(encoding="utf-8"))
    migration = json.loads((sidecars/"migration_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((sidecars/"manifest.json").read_text(encoding="utf-8"))
    if migration["source_arm"] != "fresh_forward" or migration["verified_ground_truth"] or migration["incorporated_into_training"]:
        raise ValueError("requires isolated, unadopted fresh_forward sidecars")
    if len(selection["clips"]) != 32 or len({r["clip_id"] for r in selection["clips"]}) != 32:
        raise ValueError("requires the locked 32-clip fit audit selection")
    if migration["locked_selection_sha256"] != protocol["selection_sha256"]:
        raise ValueError("sidecars and fit selection mismatch")
    clips = []
    for row in sorted(selection["clips"], key=lambda r: r["clip_id"]):
        source_id = row["clip_id"]
        if Path(source_id).name != source_id or "/" in source_id or "\\" in source_id:
            raise ValueError("unsafe source_id")
        path = sidecars/"raw"/(source_id+".npz")
        rec = manifest["raw/"+source_id+".npz"]
        if sha(path) != rec["sha256"] or path.stat().st_size != rec["bytes"]:
            raise ValueError("sidecar source hash/size mismatch")
        with np.load(path, allow_pickle=False) as data:
            raw, valid = data["coeffs"].copy(), data["valid"].copy()
        clips.append((row, raw, valid))
    thresholds = fit_noise_thresholds((raw, valid) for _, raw, valid in clips)
    write_json(output/"thresholds.json", thresholds)
    reports = []
    for row, raw, valid in clips:
        report, arrays = extract_clip(raw, valid, thresholds, row["clip_id"])
        report["metadata"] = {k: row[k] for k in ("sentence", "speaker", "emotion", "speaker_name", "video", "video_sha256")}
        write_json(output/"clips"/(row["clip_id"]+".json"), report)
        (output/"arrays").mkdir(exist_ok=True)
        np.savez_compressed(output/"arrays"/(row["clip_id"]+".npz"), **arrays)
        reports.append(report)
    sums = {k: sum(r["summary"][k] for r in reports) for k in
            ("candidate_count", "complete_event_count", "censored_candidate_count", "unsupported_candidate_count",
             "valid_frames", "eligible_frames", "event_frames", "unsupported_frames", "supported_hold_frames",
             "raw_derivative_energy", "smooth_derivative_energy", "event_derivative_energy", "unsupported_derivative_energy")}
    total = sums["smooth_derivative_energy"]
    sums.update({"event_energy_fraction": sums["event_derivative_energy"]/total if total else 0.,
                 "unsupported_energy_fraction": sums["unsupported_derivative_energy"]/total if total else 0.,
                 "event_rate_hz": sums["complete_event_count"]/(sums["valid_frames"]/25),
                 "clips_with_complete_events": sum(bool(r["events"]) for r in reports)})
    sums["stage_counts"] = {g: {k: sum(r["summary"]["stage_counts"][g][k] for r in reports)
        for k in ("candidates", "complete_support_before_merge", "accepted_after_same_group_conflict")} for g in GROUPS}
    result = {"schema": SCHEMA, "status": "extraction_complete_semantic_review_required", "clip_count": len(reports),
              "summary": sums, "thresholds": thresholds, "verified_semantic_ground_truth": False,
              "training_started": False, "original_data_modified": False,
              "clips": {r["source_id"]: r["summary"] for r in reports}}
    write_json(output/"summary.json", result)
    write_json(output/"events.json", {"schema": SCHEMA, "events": [e for r in reports for e in r["events"]],
               "waits": [w for r in reports for w in r["waits"]], "verified_semantic_ground_truth": False})
    write_json(output/"teacher.json", {"schema": SCHEMA, "protocol": protocol, "thresholds": thresholds,
        "clips": [{"source_id": r["source_id"], "metadata": r["metadata"], "events": r["events"],
                   "waits": r["waits"], "diagnostics": r["summary"], "candidates": r["candidates"]} for r in reports],
        "summary": sums, "verified_semantic_ground_truth": False, "training_started": False})
    write_json(output/"manifest.json", {str(p.relative_to(output)).replace("\\", "/"):
               {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecars", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.sidecars, args.audit, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
