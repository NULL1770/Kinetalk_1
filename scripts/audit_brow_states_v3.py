"""Independently check v3 arrays, phase masks, provenance and reconstruction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_brow_states_v3 import (SCHEMA, STATE_NAMES, GROUP_NAMES,
    HOLD, ONSET, ACTIVE, RELEASE, STATIC, TRANSITION)
from scripts.extract_brow_events_v2 import GROUPS, valid_runs, sha, write_json


def validate_arrays(a):
    valid, eligible = a["valid"], a["eligible"]
    if valid.dtype != bool or valid.ndim != 1 or eligible.shape != valid.shape or eligible.dtype != bool:
        raise ValueError("Boolean native masks required")
    expected = np.zeros_like(valid)
    for left, right in valid_runs(valid):
        if right-left > 4: expected[left+2:right-2] = True
    if not np.array_equal(expected, eligible): raise ValueError("incorrect half-open eligibility")
    n = len(valid)
    for k in ("raw5", "smooth5"):
        if a[k].shape != (n, 5) or not np.isfinite(a[k][valid]).all():
            raise ValueError("finite native raw5/smooth5 T,5 required")
    keys = ("labels", "known_mask", "phase_known", "trend", "trend_known", "direction", "conflict", "censored_mask", "reconstruction")
    if any(a[k].shape != (n, 2) for k in keys): raise ValueError("all group arrays require shape T,2")
    for k in ("known_mask", "phase_known", "trend_known", "conflict", "censored_mask"):
        if a[k].dtype != bool: raise ValueError("Boolean group masks required")
    labels, known, phase = a["labels"], a["known_mask"], a["phase_known"]
    if not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels >= len(STATE_NAMES))).any():
        raise ValueError("invalid labels")
    expected_known = np.isin(labels, [HOLD, ONSET, ACTIVE, RELEASE, STATIC, TRANSITION]) & eligible[:, None] & ~a["conflict"]
    if not np.array_equal(known, expected_known): raise ValueError("known state/eligibility conflict")
    if not np.array_equal(phase, np.isin(labels, [HOLD, ONSET, ACTIVE, RELEASE]) & known):
        raise ValueError("phase must be identified, valid and conflict-free")
    if (a["censored_mask"] & ~known).any() or (a["trend_known"] & (~eligible[:, None] | a["conflict"])).any():
        raise ValueError("invalid censor/trend mask")
    if not np.isin(a["trend"], [-1, 0, 1]).all() or not np.isin(a["direction"], [-1, 0, 1]).all():
        raise ValueError("invalid signed direction")
    dynamic_phase = phase & np.isin(labels, [ONSET, ACTIVE, RELEASE])
    if (a["direction"][dynamic_phase] == 0).any():
        raise ValueError("identified dynamic phase requires nonzero event direction")
    if not np.isfinite(a["reconstruction"][known]).all() or not np.isnan(a["reconstruction"][~known]).all():
        raise ValueError("reconstruction must be finite exactly on known states")


def audit(states_dir, output):
    states_dir, output = Path(states_dir), Path(output)
    if output.exists(): raise FileExistsError("fresh audit output required")
    manifest = json.loads((states_dir/"manifest.json").read_text(encoding="utf8"))
    for name, record in manifest.items():
        p = (states_dir/name).resolve()
        if not p.is_relative_to(states_dir.resolve()) or sha(p) != record["sha256"] or p.stat().st_size != record["bytes"]:
            raise ValueError("state manifest mismatch")
    clips = json.loads((states_dir/"clips.json").read_text(encoding="utf8"))
    ids = [c["source_id"] for c in clips]
    if any(not isinstance(s, str) or Path(s).name != s or s in ("", ".", "..") or "/" in s or "\\" in s for s in ids):
        raise ValueError("unsafe source_id")
    required = {"clips.json", "protocol.json"} | {f"arrays/{s}.npz" for s in ids} | {f"clips/{s}.json" for s in ids}
    if not required.issubset(manifest):
        raise ValueError("required input missing from manifest")
    protocol = json.loads((states_dir/"protocol.json").read_text(encoding="utf8"))
    if protocol["schema"] != SCHEMA or protocol.get("groups") != GROUPS or protocol.get("state_names") != STATE_NAMES:
        raise ValueError("protocol schema mismatch")
    if len({c["source_id"] for c in clips}) != len(clips) or any(c["schema"] != SCHEMA for c in clips):
        raise ValueError("duplicate source or schema mismatch")
    totals = {g: {"states": {s: 0 for s in STATE_NAMES}, "known": 0, "phase_known": 0,
                     "eligible": 0, "conflict": 0, "transitions": 0, "episodes": 0,
                     "complete_nonconflicting_episodes": 0, "censored_episodes": 0,
                     "known_squared_error": 0., "known_target_centered_energy": 0.,
                     "total_derivative_energy": 0., "known_derivative_energy": 0.,
                     "phase_derivative_energy": 0.} for g in GROUP_NAMES}
    per_clip = []
    for row in clips:
        if row != json.loads((states_dir/"clips"/(row["source_id"]+".json")).read_text(encoding="utf8")):
            raise ValueError("per-clip JSON differs from combined JSON")
        with np.load(states_dir/"arrays"/(row["source_id"]+".npz"), allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        validate_arrays(a)
        rec = {"source_id": row["source_id"], "groups": {}}
        for j, g in enumerate(GROUP_NAMES):
            t = totals[g]; known, phase = a["known_mask"][:, j], a["phase_known"][:, j]
            labels = a["labels"][:, j]; gt = a["smooth5"][:, GROUPS[g]].mean(1)
            t["eligible"] += int(a["eligible"].sum()); t["known"] += int(known.sum())
            t["phase_known"] += int(phase.sum()); t["conflict"] += int(a["conflict"][:, j].sum())
            for i, name in enumerate(STATE_NAMES): t["states"][name] += int((labels == i).sum())
            gr = row["groups"][g]
            t["transitions"] += len(gr["transitions"]); t["episodes"] += len(gr["episodes"])
            t["complete_nonconflicting_episodes"] += sum(not e["left_censored"] and not e["right_censored"] and not e["conflict"] for e in gr["episodes"])
            t["censored_episodes"] += sum(e["left_censored"] or e["right_censored"] for e in gr["episodes"])
            for kind in ("stable_segments", "transitions", "episodes"):
                for seg in gr[kind]:
                    if not 0 <= seg["start"] < seg["stop"] <= len(gt) or not a["eligible"][seg["start"]:seg["stop"]].all():
                        raise ValueError("segment crosses gap or guard")
                    runs = valid_runs(a["valid"])
                    ri = seg["run_index"]
                    if not 0 <= ri < len(runs) or not runs[ri][0]+2 <= seg["start"] < seg["stop"] <= runs[ri][1]-2:
                        raise ValueError("segment run_index mismatch")
                    if kind == "episodes" and not seg["start"] <= seg["peak"] <= seg["release"] < seg["stop"]:
                        raise ValueError("episode phase bounds invalid")
                    if kind in ("transitions", "episodes"):
                        supports = [ss for ss in gr["stable_segments"] if ss["run_index"] == ri]
                        for sk in ("left_support", "right_support"):
                            si = seg[sk]
                            if si is not None and not 0 <= si < len(supports):
                                raise ValueError("segment support index invalid")
            error = float(np.square(a["reconstruction"][known, j]-gt[known]).sum())
            centered = float(np.square(gt[known]-gt[known].mean()).sum()) if known.any() else 0.
            t["known_squared_error"] += error; t["known_target_centered_energy"] += centered
            pair = a["eligible"][1:] & a["eligible"][:-1]
            energy = np.square(np.diff(a["smooth5"][:, GROUPS[g]], axis=0)).sum(1)
            t["total_derivative_energy"] += float(energy[pair].sum())
            t["known_derivative_energy"] += float(energy[pair & known[1:] & known[:-1]].sum())
            t["phase_derivative_energy"] += float(energy[pair & phase[1:] & phase[:-1]].sum())
            rec["groups"][g] = {"known_frames": int(known.sum()), "phase_frames": int(phase.sum()),
                                 "episodes": len(gr["episodes"]), "known_rmse": float(np.sqrt(error/known.sum())) if known.any() else None}
        per_clip.append(rec)
    for t in totals.values():
        t["observed_state_fraction"] = t["known"]/max(t["eligible"], 1)
        t["identified_phase_fraction"] = t["phase_known"]/max(t["eligible"], 1)
        t["known_rmse"] = float(np.sqrt(t["known_squared_error"]/t["known"])) if t["known"] else None
        t["known_energy_fraction"] = t["known_derivative_energy"]/max(t["total_derivative_energy"], 1e-15)
        t["phase_energy_fraction"] = t["phase_derivative_energy"]/max(t["total_derivative_energy"], 1e-15)
    result = {"schema": SCHEMA+"_audit", "clip_count": len(clips), "groups": totals,
              "clips": per_clip, "audio_training_allowed": False, "reconstruction_uses_target_boundaries": True,
              "reason": "descriptive fit-only state reconstruction; semantic/naturalness and audio prediction not validated",
              "source_manifest_sha256": sha(states_dir/"manifest.json")}
    write_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); r = audit(args.states, args.output)
    print(json.dumps({k: v for k, v in r.items() if k != "clips"}))
