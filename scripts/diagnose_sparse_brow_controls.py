"""Engineering-only sparse brow event controls.

This diagnostic deliberately does **not** fit a teacher, train a network, read
audio timing, or claim natural motion.  It injects a tiny, fixed synthetic mark
bank into the saved independent global logit levels for the first neutral query
of four people.  The purpose is to check that a sparse event lifecycle can be
composed into the existing 52-channel curves while preserving the other 47
channels bit-for-bit.

The synthetic bank contains equal raise/down components:
``raise=[0,0,.10,.07,.07]`` and ``down=[.10,.10,0,0,0]`` in the five brow
channels ``browDownLeft, browDownRight, browInnerUp, browOuterUpLeft,
browOuterUpRight``.  Both return to the independent level.  Onset/apex/release
durations are fixed to 12/8/18 frames.  Waiting hazards and near-zero scales
are fixed constants, not measurements from the dataset.  Outputs are suitable
for the display renderer only; all provenance states engineering-only status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_dynamic_rig_comparison as rig
from scripts import sparse_brow_event_process as event


FPS = 25
PERSON_PREFIXES = ("mead_M003", "mead_M005", "mead_W011", "mead_W014")
QUERY_SUFFIX = "_neutral_L1_007"
BROW5 = np.asarray([41, 42, 43, 44, 45], dtype=np.int64)
MODE_NAMES = ("static baseline", "synthetic sparse event")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def synthetic_model() -> dict:
    """Return fixed marks; no values are estimated from a recording."""
    # The mark vector is peak5, level-relative return5, log(onset),
    # log1p(apex), log(release).  Zero scales make each component exact while
    # the component choice remains an auditable independent random event.
    raise_peak = [0., 0., .10, .07, .07]
    down_peak = [.10, .10, 0., 0., 0.]
    zero_return = [0.] * 5
    def component(peak):
        loc = peak + zero_return + [np.log(12.), np.log1p(8.), np.log(18.)]
        return {"weight": 1., "df": 5., "loc": loc, "scale": np.zeros((13, 13)).tolist()}
    return event.validate_model({
        "schema_version": event.SCHEMA,
        "fps": FPS,
        "mark_space": "coefficient_delta",
        "coefficient_eps": 1e-4,
        "components": [component(raise_peak), component(down_peak)],
        # Fixed engineering hazards (one-second bins), not fit statistics.
        "wait_hazard": [.55, .80, 1.0], "wait_bin_frames": 25,
        "minimum_hold_frames": 25,
        "max_peak_offset": [.15] * 5, "max_return_offset": [0.] * 5,
        "duration_bounds": {"onset": [12, 12], "apex": [8, 8], "release": [18, 18]},
    })


def _load_prior(prior_root: Path):
    with (prior_root / "query_selection.json").open(encoding="utf8") as f:
        selection = json.load(f)
    with (prior_root / "protocol.json").open(encoding="utf8") as f:
        protocol = json.load(f)
    with (prior_root / "status.json").open(encoding="utf8") as f:
        status = json.load(f)
    saved = torch.load(prior_root / "predictions.pt", map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or "curves" not in saved:
        raise ValueError("saved prior predictions.pt has no curves")
    expected = [f + QUERY_SUFFIX for f in PERSON_PREFIXES]
    rows = {row["clip_id"]: row for row in selection["queries"]}
    if any(cid not in rows for cid in expected):
        raise ValueError("locked query selection does not contain all four neutral first queries")
    return protocol, status, saved, [rows[cid] for cid in expected]


def _source_npz(fullface_root: Path, clip_id: str) -> Path:
    candidates = [fullface_root / "video_npz" / (clip_id + ".npz"),
                  fullface_root / (clip_id + ".npz")]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing fullface NPZ for {clip_id}")


def _base_frame(source: Path) -> tuple[np.ndarray, dict]:
    with np.load(source, allow_pickle=False) as data:
        channels = data["channels"].tolist()
        names = data["mode_names"].tolist()
        motions = np.asarray(data["motions"], dtype=np.float32)
        valid = np.asarray(data["valid"])
        if channels != rig.ARKIT_NAMES or valid.dtype != np.bool_ or motions.ndim != 3 or motions.shape[2] != 52:
            raise ValueError(f"invalid 52-channel fullface source: {source}")
        if not valid.any():
            raise ValueError(f"source has no valid frame: {source}")
        baseline_candidates = [i for i, name in enumerate(names) if name == "run12 audio baseline"]
        if len(baseline_candidates) != 1:
            raise ValueError("fullface source lacks run12 baseline mode")
        baseline_index = baseline_candidates[0]
        if not np.isfinite(motions[baseline_index, valid]).all():
            raise ValueError(f"source baseline has nonfinite valid frames: {source}")
        first = int(np.flatnonzero(valid)[0])
        return motions[baseline_index, first].copy(), {"channels": channels, "source_modes": names,
            "source_valid_frames": int(valid.sum()), "first_valid_index": first,
            "baseline_mode_index": baseline_index}


def _make_curve(base: np.ndarray, level: np.ndarray, seed: int, key: str, frames: int, model: dict):
    if frames != 10 * FPS:
        raise ValueError("The fixed control schedule requires exactly 250 frames")
    # The saved independent level controls the initial brow state. Other 47
    # channels remain the static first-valid baseline from the frozen display.
    process = event.SparseBrowEventProcess(model, level, seed, key)
    output = np.broadcast_to(base, (frames, 52)).copy()
    output[:, BROW5] = expit(level[:5]).astype(np.float32)
    chunks = []
    # Explicit 4s RUN, 1s HOLD, 1s RELEASE, 4s RUN schedule.
    chunks.append(process.sample(4 * FPS))
    process.command("HOLD")
    chunks.append(process.sample(1 * FPS))
    process.command("RELEASE")
    chunks.append(process.sample(1 * FPS))
    process.command("RUN")
    chunks.append(process.sample(4 * FPS))
    sampled = np.concatenate([chunk["values"] for chunk in chunks], axis=0)
    output[:, BROW5] = sampled[:, :5].astype(np.float32)
    return output, process, chunks


def _long_control_checks(model, level, seed, key, frames):
    amplitudes = (0., .5, 1., 1.5)
    gains = []
    for amplitude in amplitudes:
        state = event.SparseBrowEventProcess(model, level, seed, key, {"amplitude": amplitude})
        gains.append((state, state.sample(frames)))
    normal_state, normal = gains[2]
    zero_exact = np.array_equal(gains[0][1]["logits"], np.broadcast_to(level, (frames, 9)))
    same_clock = all(np.array_equal(out["phase"], normal["phase"]) and
                     np.array_equal(out["event_index"], normal["event_index"]) for _, out in gains)
    excursions = [float(np.max(out["values"][:, :5] - expit(level[:5]))) for _, out in gains]
    activity_counts, activity_states = [], []
    for activity in (.5, 1., 2.):
        state = event.SparseBrowEventProcess(model, level, seed, key, {"activity": activity})
        state.sample(frames)
        activity_counts.append(len(state.events)); activity_states.append(state)
    common = min(activity_counts)
    aligned_marks = all([row["raw_mark"] for row in state.events[:common]] ==
                        [row["raw_mark"] for row in normal_state.events[:common]] for state in activity_states)
    complete = [row for row in normal_state.events if row["status"] == "complete"]
    terminal_exact = all(np.array_equal(row["return_logit"], level[:5]) for row in complete)
    adjacent_hold = (normal["phase"][1:] == "HOLD") & (normal["phase"][:-1] == "HOLD")
    checks = {"zero_amplitude_exact": zero_exact, "amplitude_same_clock": same_clock,
              "amplitude_excursions": excursions,
              "amplitude_excursion_nondecreasing": all(a <= b + 1e-14 for a, b in zip(excursions, excursions[1:])),
              "activity_counts": activity_counts,
              "activity_count_nondecreasing": activity_counts == sorted(activity_counts),
              "same_index_marks_across_activity": aligned_marks,
              "complete_return_level_exact": terminal_exact,
              "hold_exact": bool((np.diff(normal["logits"], axis=0)[adjacent_hold] == 0).all()),
              "eye4_exact": np.array_equal(normal["logits"][:, 5:], np.broadcast_to(level[5:], (frames, 4))),
              "finite": bool(np.isfinite(normal["values"]).all()),
              "bounded": bool(((normal["values"] >= 0) & (normal["values"] <= 1)).all())}
    return normal_state, normal, {**checks, "passed": bool(all(v for v in checks.values() if isinstance(v, (bool, np.bool_))))}


def _save_html(output: Path, records: list[dict], model: dict, duration_sec: int, seeds: list[int]):
    body = ["<h1>Sparse brow event controls (engineering diagnostic)</h1>",
            "<p><b>Not trained, not audio-conditioned, not a teacher fit, and not a naturalness result.</b></p>",
            "<p>Fixed synthetic raise/down marks are used only to verify event lifecycle and 52-channel composition."
            " The global nine-dimensional levels come from the saved independent prior; other 47 channels are static"
            " first-valid baseline values.</p>", "<ul>"]
    for row in records:
        body.append(f"<li><code>{row['clip_id']}</code>: <a href='{row['npz']}'>NPZ</a>; "
                    f"events={row['event_count']}, protected47_exact={row['protected47_exact']}</li>")
    body += ["</ul>", f"<p>Long safety sweep: {len(seeds)} seeds × {duration_sec}s × 4 people; fixed wait hazard.</p>",
             "<p>The next required step is a validated visible-action teacher and audio timing model.</p>"]
    (output / "index.html").write_text("<!doctype html><meta charset='utf-8'>" + "\n".join(body), encoding="utf8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--fullface-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--duration-sec", type=int, default=60)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("output must be a fresh directory")
    if args.seeds < 1 or args.duration_sec < 1:
        parser.error("seeds and duration-sec must be positive")
    args.output.mkdir(parents=True)
    (args.output / "controls").mkdir()
    (args.output / "status.json").write_text(json.dumps({"status": "running", "engineering_only": True}), encoding="utf8")
    protocol, status, saved, picks = _load_prior(args.prior_root)
    model = synthetic_model()
    model_hash = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    records = []
    jobs = []
    long_stats = []
    for pick in picks:
        cid = pick["clip_id"]
        source = _source_npz(args.fullface_root, cid)
        base, source_meta = _base_frame(source)
        curves = saved["curves"][cid]
        level = np.asarray(curves["level"], dtype=np.float64)
        if level.shape != (9,) or not np.isfinite(level).all():
            raise ValueError(f"missing finite independent level for {cid}")
        output, process, chunks = _make_curve(base, level, 42, cid, 10 * FPS, model)
        static = np.broadcast_to(base, output.shape).copy()
        static[:, BROW5] = expit(level[:5]).astype(np.float32)
        protected = np.delete(np.arange(52), BROW5)
        if not np.array_equal(output[:, protected], static[:, protected]):
            raise RuntimeError("protected 47 channels changed")
        path = args.output / "controls" / f"{cid}.npz"
        times = np.arange(10 * FPS, dtype=np.float64) / FPS
        np.savez_compressed(path, channels=np.asarray(rig.ARKIT_NAMES), mode_names=np.asarray(MODE_NAMES),
                            clip_id=np.asarray(cid), noise_seed=np.asarray(42), times=times,
                            valid=np.ones(len(times), dtype=bool), motions=np.stack([static, output]),
                            schedule=np.asarray("RUN4s,HOLD1s,RELEASE1s,RUN4s"),
                            phase=np.concatenate([chunk["phase"] for chunk in chunks]),
                            event_index=np.concatenate([chunk["event_index"] for chunk in chunks]),
                            command=np.repeat(np.asarray(["RUN", "HOLD", "RELEASE", "RUN"]), [100, 25, 25, 100]),
                            engineering_only=np.asarray(True), trained=np.asarray(False))
        event_count = len(process.events)
        records.append({"clip_id": cid, "source_npz": source.as_posix(), "source_sha256": sha(source),
                        "npz": path.relative_to(args.output).as_posix(), "npz_sha256": sha(path),
                        "level": level.tolist(), "source_meta": source_meta, "event_count": event_count,
                        "event_provenance": process.events, "wait_provenance": process.waits,
                        "control_provenance": process.controls, "protected47_exact": True,
                        "query_motion_used": False, "audio_timing_used": False})
        jobs.append({"input": path.relative_to(args.output).as_posix(), "output": cid,
                     "fps": FPS, "columns": 2, "tile_size": 360, "samples": 16,
                     "max_frames": 0, "expected_video": cid + "/comparison.mp4", "audio": None})
        for seed in range(args.seeds):
            long_process, long_out, checks = _long_control_checks(model, level, seed, cid, args.duration_sec * FPS)
            long_stats.append({"clip_id": cid, "seed": seed, "frames": len(long_out["values"]),
                               "events": len(long_process.events),
                               "phase_fraction": {name: float(np.mean(long_out["phase"] == name))
                                                   for name in ("HOLD", "ONSET", "APEX", "RELEASE")},
                               "max_abs_brow_logit": float(np.max(np.abs(long_out["logits"][:, :5] - level[:5]))),
                               "nonfinite": bool(not np.isfinite(long_out["values"]).all()), "checks": checks})
    manifest = {"schema": "sparse_brow_engineering_controls_v1", "status": "engineering_only",
                "trained": False, "teacher_fit": False, "audio_conditioned": False,
                "naturalness_certified": False, "query_motion_used": False, "fps": FPS,
                "schedule": "4s RUN, 1s HOLD, 1s RELEASE, 4s RUN", "model": model,
                "model_sha256": model_hash, "prior_root": args.prior_root.as_posix(),
                "prior_status": status, "prior_protocol_sha256": sha(args.prior_root / "protocol.json"),
                "prior_predictions_sha256": sha(args.prior_root / "predictions.pt"),
                "query_selection_sha256": sha(args.prior_root / "query_selection.json"),
                "saved_prior_status": protocol.get("schema"), "query_selection": picks,
                "records": records, "long_stats": long_stats, "script_sha256": sha(Path(__file__)),
                "sampler_sha256": sha(Path(event.__file__)), "fit_type": "synthetic_fixed_constants",
                "query_gt_in_source_container": True, "query_gt_used_for_parameters": False,
                "selection_rule": "First saved metadata query of each fixed person; no outcome filtering",
                "all_engineering_checks_pass": all(row["checks"]["passed"] for row in long_stats)}
    (args.output / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf8")
    (args.output / "long_stats.json").write_text(json.dumps(_jsonable(long_stats), indent=2), encoding="utf8")
    _save_html(args.output, records, model, args.duration_sec, list(range(args.seeds)))
    (args.output / "render_jobs.json").write_text(json.dumps({"jobs": jobs, "rendered": False,
        "engineering_only": True, "paths_relative_to": "directory containing this JSON"}, indent=2), encoding="utf8")
    (args.output / "status.json").write_text(json.dumps({"status": "complete", "engineering_only": True,
        "trained": False, "formal_prior_fitted": False, "rendered": False,
        "engineering_checks_pass": manifest["all_engineering_checks_pass"]}, indent=2), encoding="utf8")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(records),
                      "long_runs": len(long_stats), "status": "engineering_only"}))


if __name__ == "__main__":
    main()
