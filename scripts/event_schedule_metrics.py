"""Scores coarse excursion schedules using the fit-only event teacher.

The labels describe complete positive excursions, never displacement from a
query or population anchor. Schedule diagnostics complement ARKit metrics;
they do not certify semantics or naturalness.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from scripts.event_schedule_teacher import (
    FPS, GROUPS, EventThresholds, as_numpy as _as_numpy, clip_arrays,
    fit_event_thresholds, extract_event_labels, extract_schedule, fit_teacher,
)


def _clip_row(row):
    target, valid, observed = clip_arrays(row)
    pred = row.get("samples", row.get("prediction", row.get("predictions")))
    if pred is None:
        raise ValueError("predictions are required; target is never a fallback")
    pred = _as_numpy(pred).astype(np.float64, copy=False)
    if pred.ndim == 2:
        pred = pred[None]
    if pred.ndim != 3 or pred.shape[0] < 1 or pred.shape[1:] != target.shape:
        raise ValueError("samples/prediction must be nonempty [K,T,9] matching target")
    if not np.isfinite(pred[:,observed]).all():
        raise ValueError("observed predictions must be finite")
    return target, pred, valid, observed


def evaluate_clip(row: Mapping, thresholds: EventThresholds) -> dict:
    target, predictions, valid, observed = _clip_row(row)
    truth = extract_event_labels(target, valid, thresholds, observed)
    draws_all = [extract_event_labels(p, valid, thresholds, observed) for p in predictions]
    out = {"groups": {}}
    eps = 1e-6
    requested = _as_numpy(row.get("score_mask", valid))
    if requested.dtype != bool or requested.shape != valid.shape:
        raise ValueError("score_mask must be Boolean [T]")
    for group in GROUPS:
        gt = truth["groups"][group]
        draws = np.stack([p["groups"][group]["active"] for p in draws_all])
        # Common truth-known support is fixed across arms. Predictions with
        # incomplete events remain zero: they cannot hide bad outputs by
        # shrinking the score support using their own known masks.
        use = valid & requested & gt["known"]
        if not use.any():
            out["groups"][group] = {"status": "pending_no_known_frames", "score_frames": 0,
                                     "brier": None, "nll": None, "duration_error_s": None,
                                     "truth_event_count": len(gt["events"]),
                                     "pred_event_count_mean": float(np.mean([len(d["groups"][group]["events"]) for d in draws_all]))}
            continue
        y = gt["active"].astype(np.float64)
        prob = draws.mean(axis=0)
        p = np.clip(prob[use], eps, 1-eps); yy = y[use]
        out["groups"][group] = {
            "status": "ok", "score_frames": int(use.sum()),
            "brier": float(np.mean((prob[use]-yy)**2)),
            "nll": float(-np.mean(yy*np.log(p)+(1-yy)*np.log1p(-p))),
            "duration_error_s": float(abs(prob[use].sum()-yy.sum())/thresholds.fps),
            "truth_active_frames": int(yy.sum()), "pred_active_frames_mean": float(prob[use].sum()),
            "truth_event_count": len(gt["events"]),
            "pred_event_count_mean": float(np.mean([len(d["groups"][group]["events"]) for d in draws_all])),
        }
    return out


def aggregate_clip_results(results: Mapping[str, Mapping]) -> dict:
    out = {"groups": {}}
    keys = ("brier", "nll", "duration_error_s")
    for group in GROUPS:
        rows = [r["groups"][group] for r in results.values()]
        good = [r for r in rows if r["status"] == "ok"]
        out["groups"][group] = {k: float(np.mean([r[k] for r in good])) if good else None for k in keys}
        out["groups"][group].update({"scored_clips": len(good), "pending_clips": len(rows)-len(good)})
    all_rows = [r["groups"][g] for r in results.values() for g in GROUPS if r["groups"][g]["status"] == "ok"]
    out["macro"] = {k: float(np.mean([r[k] for r in all_rows])) if all_rows else None for k in keys}
    out["status"] = "ok" if all_rows else "pending_no_known_frames"
    out["clips"] = len(results)
    return out


def load_curves(path: Path) -> dict[str, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "clips" in payload:
        return dict(payload["clips"])
    if isinstance(payload, dict) and "target" in payload:
        return {str(payload.get("clip_id", "clip0")): payload}
    raise ValueError(f"unsupported curves payload: {path}")


def evaluate_arms(fit: Mapping[str, Mapping], arms: Mapping[str, Mapping[str, Mapping]], *, fps=FPS, output=None):
    if not arms:
        raise ValueError("at least one evaluation arm required")
    ids = set(next(iter(arms.values())))
    if set(fit) & ids:
        raise ValueError("fit and evaluation clip IDs must be disjoint")
    for name, clips in arms.items():
        if set(clips) != ids:
            raise ValueError("all arms must contain exactly the same clip IDs")
        if set(fit) & set(clips):
            raise ValueError("fit and evaluation clip IDs must be disjoint")
    # Reference and scoring support must also be bit-identical across arms.
    first = next(iter(arms.values()))
    for clips in arms.values():
        for cid in ids:
            for a,b in zip(clip_arrays(first[cid]), clip_arrays(clips[cid])):
                if not np.array_equal(a,b, equal_nan=True):
                    raise ValueError("all arms must use the same targets/valid/observed masks")
            if not np.array_equal(first[cid].get("score_mask", clip_arrays(first[cid])[1]),
                                  clips[cid].get("score_mask", clip_arrays(clips[cid])[1])):
                raise ValueError("all arms must use identical score_mask")
    thresholds = fit_event_thresholds(fit.values(), fps=fps)
    report = {"schema": "event_schedule_report_v2", "thresholds": asdict(thresholds), "arms": {}, "protocol": {
        "fit_clips": len(fit), "thresholds_fit_on": "fit raw-minus-smooth noise only", "fps": fps,
        "no_eval_target_used_for_threshold": True, "fit_eval_ids_disjoint": True,
        "evaluation_ids": sorted(ids), "condition": "active4, phase4, duration_seconds4; no amplitude/anchor",
    }}
    for name, clips in arms.items():
        per_clip = {cid: evaluate_clip(clips[cid], thresholds) for cid in sorted(ids)}
        report["arms"][name] = {"summary": aggregate_clip_results(per_clip), "per_clip": per_clip}
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "CURVES"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=FPS)
    args = parser.parse_args(argv)
    fit = load_curves(args.fit)
    arms = {name: load_curves(Path(path)) for name, path in args.arm}
    evaluate_arms(fit, arms, fps=args.fps, output=args.output)
    print(json.dumps({"output": str(args.output), "arms": list(arms)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
