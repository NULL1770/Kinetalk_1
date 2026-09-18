"""Group-wise observed states with explicit phase ambiguity and boundary censoring.

Offline diagnostic teacher, NOT semantic ground truth or an inference input.
Revision 2 supersedes the invalidated v2-candidate relabeling draft. Local
stationarity and directed displacement are measured anew on frozen v2 arrays;
missing v2 quiet support is never relabelled as boundary censoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import extract_brow_events_v2 as v2

SCHEMA = "brow_observed_state_v3_r3"
GROUP_NAMES = list(v2.GROUPS)
STATE_NAMES = ["UNKNOWN", "HOLD", "ONSET", "ACTIVE", "RELEASE", "STATIC", "TRANSITION", "UNSUPPORTED"]
UNKNOWN, HOLD, ONSET, ACTIVE, RELEASE, STATIC, TRANSITION, UNSUPPORTED = range(8)
PROTOCOL = {
    "schema": SCHEMA, "groups": v2.GROUPS, "state_names": STATE_NAMES,
    "fps": 25, "window": 4, "quiet_window_range_fraction": .25,
    "stable_component_total_range_fraction": .5, "minimum_transition_frames": 4,
    "minimum_net_displacement_fraction": 1., "minimum_net_over_total_variation": .8,
    "return_displacement_ratio_complete_only": [.5, 1.5],
    "half_open_intervals": True, "edge_guard_frames": 2,
    "partial_transition": "boundary-to-stable or stable-to-boundary directed displacement; observed duration is a lower bound",
    "phase_identification": "opposite adjacent transitions around observed stable plateau; isolated transitions have unknown onset/release phase",
    "zero_apex": "one prominent extremum between observed supports or true boundaries, each flank >=4 transitions and net/travel >=.8; plateau is optional",
    "overlapping_episode_phase_conflicts": "mark conflict and exclude phase loss; never choose by ordering or score",
    "static": "stable without episode baseline/apex evidence; never infer dynamic ACTIVE from absolute level",
    "known_mask": "mechanical state evidence only; use phase_known separately; neither implies semantic correctness",
    "source": "32 previously examined fit-only clips; no held-out generalization claim",
    "audio_training_allowed": False, "verified_semantic_ground_truth": False,
}


def _stable_spans(y, threshold):
    quiet = np.zeros(len(y), bool)
    for i in range(len(y)-3):
        if np.ptp(y[i:i+4]) <= .25*threshold:
            quiet[i:i+4] = True
    # Overlapping quiet windows can concatenate a slow ramp. Test the entire
    # connected component; do not partition it into many false static states.
    return [(a, b) for a, b in v2.valid_runs(quiet) if np.ptp(y[a:b]) <= .5*threshold]


def _transition(y, a, b, threshold, left_support, right_support, offset, boundary):
    if b <= a:
        return None
    delta = float(y[b]-y[a])
    travel = float(np.abs(np.diff(y[a:b+1])).sum())
    if b-a < 4 or abs(delta) < threshold or abs(delta)/max(travel, 1e-15) < .8:
        return None
    return {"start": a+offset, "stop": b+offset+1, "start_value": float(y[a]),
            "end_value": float(y[b]), "delta": delta, "trend": 1 if delta > 0 else -1,
            "net_over_total_variation": abs(delta)/max(travel, 1e-15),
            "left_censored": left_support is None, "right_censored": right_support is None,
            "left_boundary_kind": boundary[0] if left_support is None else "observed_stable",
            "right_boundary_kind": boundary[1] if right_support is None else "observed_stable",
            "duration_frames_observed": b-a,
            "duration_type": "complete" if left_support is not None and right_support is not None else "lower_bound",
            "left_support": left_support, "right_support": right_support}


def _pulse_in_gap(z, a, b, threshold, ls, rs, offset, boundary, run_index):
    """A zero-duration apex need not contain a four-frame quiet plateau."""
    events = []
    for sign in (1, -1):
        peaks, props = v2.find_peaks(sign*z[a:b+1], prominence=threshold, plateau_size=(1, None))
        for k, peak in enumerate(peaks):
            p, q = a+int(props["left_edges"][k]), a+int(props["right_edges"][k])
            up = _transition(z, a, p, threshold, ls, -1, offset, boundary)
            down = _transition(z, q, b, threshold, -1, rs, offset, boundary)
            if up is None or down is None or up["trend"] != sign or down["trend"] != -sign:
                continue
            if ls is not None and rs is not None and not .5 <= abs(down["delta"]/up["delta"]) <= 1.5:
                continue
            events.append({"run_index": run_index, "start": offset+a, "peak": offset+p,
                           "release": offset+q, "stop": offset+b+1, "direction": sign,
                           "peak_delta_observed": up["delta"], "return_delta_observed": down["delta"],
                           "left_censored": up["left_censored"], "right_censored": down["right_censored"],
                           "left_boundary_kind": up["left_boundary_kind"], "right_boundary_kind": down["right_boundary_kind"],
                           "onset_duration_type": up["duration_type"], "release_duration_type": down["duration_type"],
                           "conflict": False, "shape_source": "gap_extremum", "left_support": ls, "right_support": rs})
    return events


def _episode_claims(ep, stable_records):
    claims = [(ep["start"], ep["peak"], ONSET, ep["direction"]),
              (ep["peak"], ep["release"]+1, ACTIVE, ep["direction"]),
              (ep["release"]+1, ep["stop"], RELEASE, ep["direction"])]
    if ep["left_support"] is not None:
        sr = stable_records[ep["left_support"]]
        claims.append((sr["start"], ep["start"], HOLD, 0))
    if ep["right_support"] is not None:
        sr = stable_records[ep["right_support"]]
        claims.append((ep["stop"], sr["stop"], HOLD, 0))
    return claims


def extract_group(y, valid, eligible, threshold):
    n = len(y)
    labels = np.full(n, UNKNOWN, np.int8)
    phase = np.zeros(n, bool)
    trend = np.zeros(n, np.int8)
    trend_known = np.zeros(n, bool)
    direction = np.zeros(n, np.int8)
    conflict = np.zeros(n, bool)
    censored = np.zeros(n, bool)
    reconstruction = np.full(n, np.nan)
    segments, transitions, episodes = [], [], []
    for run_index, (a, b) in enumerate(v2.valid_runs(valid)):
        lo, hi = a+2, b-2
        if hi <= lo:
            continue
        # Input eligibility must agree with the v2 guard, including gaps.
        if not eligible[lo:hi].all():
            raise ValueError("v2 eligible mask differs from valid-run SG guard")
        z = y[lo:hi]
        stable = _stable_spans(z, threshold)
        labels[lo:hi] = UNSUPPORTED
        stable_records = []
        boundary = ("clip_start_with_sg_guard" if a == 0 else "invalid_gap_left_with_sg_guard",
                    "clip_end_with_sg_guard" if b == n else "invalid_gap_right_with_sg_guard")
        for s, e in stable:
            labels[lo+s:lo+e] = STATIC
            trend_known[lo+s:lo+e] = True
            reconstruction[lo+s:lo+e] = np.median(z[s:e])
            rec = {"start": lo+s, "stop": lo+e, "run_index": run_index,
                   "level": float(np.median(z[s:e])), "range": float(np.ptp(z[s:e])),
                   "left_censored": s == 0, "right_censored": e == len(z),
                   "left_boundary_kind": boundary[0] if s == 0 else "observed_transition_or_unknown",
                   "right_boundary_kind": boundary[1] if e == len(z) else "observed_transition_or_unknown",
                   "role": "static_phase_unidentified"}
            stable_records.append(rec)
        pairs = []
        if stable:
            if stable[0][0] > 0:
                pairs.append((0, stable[0][0], None, 0))
            pairs.extend((stable[i][1]-1, stable[i+1][0], i, i+1) for i in range(len(stable)-1))
            if stable[-1][1] < len(z):
                pairs.append((stable[-1][1]-1, len(z)-1, len(stable)-1, None))
        else:
            pairs = [(0, len(z)-1, None, None)]
        run_transitions, gap_episodes = [], []
        for start, end, ls, rs in pairs:
            t = _transition(z, start, end, threshold, ls, rs, lo, boundary)
            if t is None:
                gap_episodes.extend(_pulse_in_gap(z, start, end, threshold, ls, rs, lo, boundary, run_index))
                continue
            t["run_index"] = run_index
            run_transitions.append(t)
            x, endx = t["start"], t["stop"]
            labels[x:endx] = TRANSITION
            trend[x:endx] = t["trend"]
            trend_known[x:endx] = True
            censored[x:endx] = t["left_censored"] or t["right_censored"]
            q = np.linspace(0., 1., endx-x)
            h = q**3*(10-15*q+6*q*q)
            reconstruction[x:endx] = t["start_value"]+t["delta"]*h
        # Adjacent directed changes identify an excursion only when they
        # share the SAME observed plateau and have opposed, comparable deltas.
        claims = []
        for ep in gap_episodes:
            episodes.append(ep)
            claims.append((ep, _episode_claims(ep, stable_records)))
            x, p, q, e = ep["start"], ep["peak"], ep["release"], ep["stop"]-1
            for u, v in ((x, p), (q, e)):
                h = np.linspace(0., 1., v-u+1); h = h**3*(10-15*h+6*h*h)
                reconstruction[u:v+1] = y[u]+(y[v]-y[u])*h
                trend[u:v+1] = 1 if y[v] > y[u] else -1
            reconstruction[p:q+1] = np.median(y[p:q+1])
            trend[p:q+1] = 0
            trend_known[x:e+1] = True
        for first, second in zip(run_transitions, run_transitions[1:]):
            si = first["right_support"]
            if si is None or si != second["left_support"] or first["trend"] == second["trend"]:
                continue
            ratio = abs(second["delta"]/first["delta"])
            if not (first["left_censored"] or second["right_censored"]) and not .5 <= ratio <= 1.5:
                continue
            ep = {"run_index": run_index, "start": first["start"], "peak": first["stop"]-1,
                  "release": second["start"], "stop": second["stop"],
                  "direction": first["trend"], "peak_delta_observed": first["delta"],
                  "return_delta_observed": second["delta"],
                  "left_censored": first["left_censored"], "right_censored": second["right_censored"],
                  "left_boundary_kind": first["left_boundary_kind"],
                  "right_boundary_kind": second["right_boundary_kind"],
                  "onset_duration_type": first["duration_type"], "release_duration_type": second["duration_type"],
                  "conflict": False, "shape_source": "observed_plateau",
                  "left_support": first["left_support"], "right_support": second["right_support"]}
            episodes.append(ep)
            claims.append((ep, _episode_claims(ep, stable_records)))
        options = [set() for _ in range(n)]
        for ep, ec in claims:
            for s, e, code, sign in ec:
                for f in range(s, e):
                    options[f].add((code, sign))
        for f in range(lo, hi):
            if len(options[f]) > 1:
                conflict[f] = True
                labels[f] = UNSUPPORTED
            elif options[f]:
                labels[f], direction[f] = next(iter(options[f]))
                phase[f] = True
        for ep, ec in claims:
            ep["conflict"] = bool(conflict[ep["start"]:ep["stop"]].any())
            if ep["left_censored"] or ep["right_censored"]:
                censored[ep["start"]:ep["stop"]] = True
        # Trend is segment-level direction of change, not a frame derivative.
        trend[np.isin(labels, [STATIC, HOLD, ACTIVE])] = 0
        segments.extend(stable_records)
        transitions.extend(run_transitions)
    known = np.isin(labels, [HOLD, ONSET, ACTIVE, RELEASE, STATIC, TRANSITION]) & eligible & valid & ~conflict
    phase &= known
    trend_known &= eligible & valid & ~conflict
    reconstruction[~known] = np.nan
    return {"labels": labels, "known_mask": known, "phase_known": phase,
            "trend": trend, "trend_known": trend_known, "direction": direction,
            "conflict": conflict, "censored_mask": censored & known,
            "reconstruction": reconstruction}, {"stable_segments": segments, "transitions": transitions, "episodes": episodes}


def build_clip(report, arrays, thresholds=None):
    raw = np.asarray(arrays["raw5"], float)
    smooth = np.asarray(arrays["smooth5"], float)
    valid = np.asarray(arrays["valid"])
    eligible = np.asarray(arrays["eligible"])
    if valid.dtype != bool or eligible.dtype != bool or valid.ndim != 1 or raw.shape != (len(valid), 5) or smooth.shape != raw.shape:
        raise ValueError("requires native T,5 raw/smooth and Boolean masks")
    expected = np.zeros(len(valid), bool)
    for a, b in v2.valid_runs(valid):
        if b-a > 4:
            expected[a+2:b-2] = True
    if not np.array_equal(expected, eligible) or not np.isfinite(smooth[valid]).all():
        raise ValueError("eligible guard or finite-value contract failed")
    thresholds = thresholds or {g: .035 for g in GROUP_NAMES}
    results, groups = [], {}
    for g in GROUP_NAMES:
        t = thresholds[g]
        threshold = float(t["threshold"] if isinstance(t, dict) else t)
        if not np.isfinite(threshold) or threshold < .035:
            raise ValueError("invalid frozen threshold")
        a, r = extract_group(smooth[:, v2.GROUPS[g]].mean(1), valid, eligible, threshold)
        results.append(a)
        groups[g] = r
    output = {k: np.stack([a[k] for a in results], axis=1) for k in results[0]}
    output.update(valid=valid.copy(), eligible=eligible.copy(), raw5=raw.copy(), smooth5=smooth.copy())
    row = {"schema": SCHEMA, "source_id": report["source_id"], "groups": groups,
           "group_names": GROUP_NAMES, "state_names": STATE_NAMES,
           "v2_complete_events": len(report.get("events", [])), "verified_semantic_ground_truth": False}
    return row, output


def run(teacher_dir, output):
    teacher_dir, output = Path(teacher_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("fresh output required")
    teacher = json.loads((teacher_dir/"teacher.json").read_text(encoding="utf8"))
    manifest = json.loads((teacher_dir/"manifest.json").read_text(encoding="utf8"))
    if teacher["schema"] != "fixed_brow_excursion_teacher_v2" or len(teacher["clips"]) != 32:
        raise ValueError("locked 32-clip v2 teacher required")
    ids = [c["source_id"] for c in teacher["clips"]]
    if len(set(ids)) != 32 or any(Path(s).name != s or s in (".", "..") or "/" in s or "\\" in s for s in ids):
        raise ValueError("unique safe source ids required")
    for name in ["teacher.json"]+[f"arrays/{s}.npz" for s in ids]:
        p = teacher_dir/name
        if v2.sha(p) != manifest[name]["sha256"] or p.stat().st_size != manifest[name]["bytes"]:
            raise ValueError("parent hash/size mismatch: "+name)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {**PROTOCOL, "parent_teacher_sha256": v2.sha(teacher_dir/"teacher.json"),
                "parent_manifest_sha256": v2.sha(teacher_dir/"manifest.json"),
                "extractor_sha256": v2.sha(__file__), "thresholds": teacher["thresholds"],
                "protocol_written_before_array_load": True}
    v2.write_json(output/"protocol.json", protocol)
    (output/"arrays").mkdir()
    rows = []
    for c in teacher["clips"]:
        with np.load(teacher_dir/"arrays"/(c["source_id"]+".npz"), allow_pickle=False) as z:
            r, a = build_clip(c, {k: z[k].copy() for k in z.files}, teacher["thresholds"])
        r["metadata"] = c["metadata"]
        np.savez_compressed(output/"arrays"/(r["source_id"]+".npz"), **a)
        v2.write_json(output/"clips"/(r["source_id"]+".json"), r)
        rows.append(r)
    v2.write_json(output/"clips.json", rows)
    v2.write_json(output/"manifest.json", {p.relative_to(output).as_posix(): {"sha256": v2.sha(p), "bytes": p.stat().st_size}
                  for p in sorted(output.rglob("*")) if p.is_file()})
    return {"schema": SCHEMA, "clips": len(rows), "training_started": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-v2", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.teacher_v2, args.output)))
