"""Independent audit of a completed six-arm visual-semantic pilot.

This audit only consumes the sealed run directory: predictions.pt contains
targets, baseline and generated samples, scales.pt contains training-fitted
metric scales, and reports.json contains the submitted summaries. It does not
load the dataset or fit anything. The complete report uses the shared metric
implementation. Primary ES, dynamic energy and speed are also independently
implemented below; shared-scorer reproduction alone is not independent evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.joint_motion_metrics import score_clip, summarize

ARMS = ("va_oracle", "va_audio", "va_static", "posterior_oracle", "posterior_audio", "posterior_static")
GROUP_NAMES = ("up", "down", "squint", "wide")
SCHEMA = "visual_semantic_pilot_audit_v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _finite_nested(value):
    if isinstance(value, dict):
        return all(_finite_nested(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_nested(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _max_abs(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float('inf')
        keys = set(a) | set(b)
        vals = [_max_abs(a[k], b[k]) for k in keys if k in a and k in b]
        return max(vals, default=0.)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return max((_max_abs(x, y) for x, y in zip(a, b)), default=0.)
    if a is None or b is None:
        return 0. if a is None and b is None else float("inf")
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return 0. if a == b else float("inf")


def independent_primary(clips, ids, arm, scale):
    """Separate NumPy implementation of the primary trajectory diagnostics."""
    groups = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
    pred_energy = np.zeros(4); target_energy = np.zeros(4)
    pred_speed = 0.; target_speed = 0.; pred_count = 0; target_count = 0
    rows = []
    for cid in ids:
        clip = clips[cid]; mask = np.asarray(clip['valid'], bool)
        x = np.asarray(clip['samples'][arm], np.float64) / scale
        y = np.asarray(clip['target'], np.float64) / scale
        k = len(x)
        if k < 2 or not np.isfinite(x[:, mask]).all() or not np.isfinite(y[mask]).all():
            raise ValueError('Invalid independently scored samples')
        xc = np.zeros_like(x); yc = np.zeros_like(y)
        edge = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
        for left, right in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
            xc[:, left:right] = x[:, left:right] - x[:, left:right].mean(axis=1, keepdims=True)
            yc[left:right] = y[left:right] - y[left:right].mean(axis=0, keepdims=True)
            dx = np.diff(x[:, left:right], axis=1); dy = np.diff(y[left:right], axis=0)
            pred_speed += float((dx * dx).sum() / 9); target_speed += float((dy * dy).sum() / 9)
            pred_count += k * max(right-left-1, 0); target_count += max(right-left-1, 0)
        scores = {}
        for name, a, b in [('raw', x[:, mask], y[mask]), ('centered', xc[:, mask], yc[mask])]:
            flat = a.reshape(k, -1); truth = b.reshape(-1)
            distance = np.linalg.norm(flat-truth, axis=1).mean() / np.sqrt(flat.shape[1])
            pairwise = np.linalg.norm(flat[:, None]-flat[None, :], axis=-1).sum()
            scores[name] = float(distance - pairwise/(2*k*(k-1)*np.sqrt(flat.shape[1])))
        for i, group in enumerate(groups):
            pred_energy[i] += np.square(xc[:, mask][:, :, group]).sum() / k
            target_energy[i] += np.square(yc[mask][:, group]).sum()
        rows.append({'clip_id': cid, 'sentence': clip['metadata']['sentence'], **scores})
    return {'joint_fair_es': {name: float(np.mean([r[name] for r in rows])) for name in ('raw', 'centered')},
            'rms_ratio': [float(np.sqrt(a/b)) if b > 0 else None for a,b in zip(pred_energy,target_energy)],
            'speed_rms': float(np.sqrt(pred_speed/pred_count)) if pred_count else None,
            'target_speed_rms': float(np.sqrt(target_speed/target_count)) if target_count else None,
            'rows': rows}


def _clip_score(clip, samples, scales):
    target = _numpy(clip["target"]).astype(np.float64)
    valid = np.asarray(clip["valid"], bool)
    samples = _numpy(samples).astype(np.float64)
    if samples.ndim != 3 or samples.shape[1:] != target.shape:
        raise ValueError("sample [K,T,9] and target [T,9] shape mismatch")
    return score_clip(samples, target, valid, scales)


def _baseline_score(clip, scales, count):
    baseline = _numpy(clip["baseline52"]).astype(np.float64)[:, [41, 42, 43, 44, 45, 5, 6, 12, 13]]
    return _clip_score(clip, np.broadcast_to(baseline, (count,) + baseline.shape).copy(), scales)


def _clip_ids_for_role(clips, role):
    if role == "holdout":
        return [cid for cid, row in clips.items() if row["metadata"].get("split") == "holdout"]
    # The run deliberately stores exactly four fit examples. Preserve the
    # report's fixed list if present, without selecting by observed scores.
    return [cid for cid, row in clips.items() if row["metadata"].get("split") == "train"]


def _sentence_bootstrap(rows, key, seed=20260918, draws=2000):
    by_sentence = {}
    for row in rows:
        by_sentence.setdefault(row["sentence"], []).append(row[key])
    sentences = sorted(by_sentence)
    if not sentences:
        return {"sentences": 0, "draws": draws, "mean": None, "q025": None, "q975": None}
    values = np.asarray([np.mean(by_sentence[s]) for s in sentences], float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return {"sentences": len(sentences), "draws": draws, "seed": seed,
            "mean": float(values.mean()), "q025": float(np.quantile(samples, .025)),
            "q975": float(np.quantile(samples, .975)), "clip_values": values.tolist()}


def _difference_bootstrap(rows, left, right, seed=20260918):
    return _sentence_bootstrap([{"sentence": r["sentence"], "value": r[left] - r[right]} for r in rows], "value", seed)


def _rows_for_role(clips, ids, arm, scales):
    rows = []
    for cid in ids:
        clip = clips[cid]
        if arm not in clip["samples"]:
            raise ValueError(f"missing {arm} samples for {cid}")
        score = _clip_score(clip, clip["samples"][arm], scales)
        rows.append({"clip_id": cid, "sentence": clip["metadata"]["sentence"],
                     "speaker": clip["metadata"].get("speaker"), "emotion": clip["metadata"].get("emotion"), **score})
    return rows


def _summary_compare(actual, reported):
    if reported is None:
        return {"available": False, "max_abs_difference": None, "matches": False}
    delta = _max_abs(actual, reported)
    return {"available": True, "max_abs_difference": float(delta), "matches": bool(delta <= 1e-9)}


def audit(run_dir, student_report=None, output=None, bootstrap_draws=2000):
    run_dir, output = Path(run_dir), Path(output or Path(run_dir) / "audit.json")
    if output.exists():
        raise FileExistsError("fresh audit output required")
    pred_path, scales_path = run_dir / "predictions.pt", run_dir / "scales.pt"
    report_path = Path(student_report) if student_report else run_dir / "reports.json"
    for path in (pred_path, scales_path, report_path):
        if not path.is_file():
            raise ValueError("missing run input: " + str(path))
    packed = torch.load(pred_path, map_location="cpu", weights_only=False)
    scales = torch.load(scales_path, map_location="cpu", weights_only=False)
    reported = json.loads(report_path.read_text(encoding="utf8"))
    if packed.get("schema") != "visual_semantic_condition_pilot_v1":
        raise ValueError("unexpected predictions schema")
    if "metric_scale" not in scales:
        raise ValueError("training metric_scale missing")
    metric_scale = _numpy(scales["metric_scale"]).astype(np.float64)
    if metric_scale.shape != (9,) or not np.isfinite(metric_scale).all() or not (metric_scale > 0).all():
        raise ValueError("invalid metric scales")
    clips = packed.get("clips", {})
    if not clips:
        raise ValueError("no packaged clip curves")
    if any(not isinstance(cid, str) for cid in clips):
        raise ValueError("unsafe clip id")
    arm_reports, all_rows = {}, {}
    for arm in ARMS:
        arm_reports[arm] = {}
        all_rows[arm] = {}
        for role in ("holdout", "fit_examples"):
            ids = _clip_ids_for_role(clips, role)
            if role == "fit_examples":
                # Only four fit examples are packaged; do not turn all train
                # clips into an apparent in-sample score.
                ids = ids[:4]
            rows = _rows_for_role(clips, ids, arm, metric_scale)
            all_rows[arm][role] = rows
            summary = summarize(rows) if rows else None
            submitted = reported.get(arm, {}).get(role, {}).get("summary")
            arm_reports[arm][role] = {"clip_count": len(rows), "rows": rows, "recomputed_summary": summary,
                                       "reported_summary_compare": _summary_compare(summary, submitted)}
    # Static repeated baseline is independently scored as a reference. It is
    # not one of the six trained arms and is never compared as a learned arm.
    baseline_rows = {}
    for role in ("holdout", "fit_examples"):
        ids = _clip_ids_for_role(clips, role)[:4] if role == "fit_examples" else _clip_ids_for_role(clips, role)
        rows = []
        for cid in ids:
            clip = clips[cid]; score = _baseline_score(clip, metric_scale, 4)
            rows.append({"clip_id": cid, "sentence": clip["metadata"]["sentence"], **score})
        baseline_rows[role] = {"rows": rows, "summary": summarize(rows) if rows else None}
    paired = {}
    for role in ("holdout", "fit_examples"):
        paired[role] = {}
        for kind, left, right in (("va_audio_vs_static", "va_audio", "va_static"),
                                  ("posterior_audio_vs_static", "posterior_audio", "posterior_static"),
                                  ("va_oracle_vs_audio", "va_oracle", "va_audio"),
                                  ("posterior_oracle_vs_audio", "posterior_oracle", "posterior_audio")):
            left_rows = {r["clip_id"]: r for r in all_rows[left][role]}
            right_rows = {r["clip_id"]: r for r in all_rows[right][role]}
            diffs = []
            for cid in left_rows:
                # Fair ES is the primary metric for multimodal samples.
                diffs.append({"sentence": left_rows[cid]["sentence"], "value": left_rows[cid]["joint_fair_es"]["raw"] - right_rows[cid]["joint_fair_es"]["raw"],
                              "clip_id": cid})
            paired[role][kind] = {"lower_is_better": True, "bootstrap": _sentence_bootstrap(diffs, "value", draws=bootstrap_draws),
                                  "clip_differences": diffs}
    independent = {}
    for arm in ARMS:
        value = independent_primary(clips, _clip_ids_for_role(clips, 'holdout'), arm, metric_scale)
        reference = reported.get(arm, {}).get('holdout', {}).get('summary')
        if reference is not None:
            delta = max(_max_abs(value['joint_fair_es'], reference['joint_fair_es']),
                        _max_abs(value['rms_ratio'], reference['rms_ratio']),
                        abs(value['speed_rms']-reference['speed']['all']['rms']),
                        abs(value['target_speed_rms']-reference['speed']['reference_all']['rms']))
            value['max_report_difference'] = delta
            if delta > 1e-9: raise ValueError('Independent primary mismatch: '+arm)
        independent[arm] = value
    independent_pairs = {}
    for kind in ('va', 'posterior'):
        a = independent[kind+'_audio']['rows']; b = independent[kind+'_static']['rows']
        independent_pairs[kind] = {metric: _sentence_bootstrap(
            [{'sentence': x['sentence'], 'value': x[metric]-y[metric]} for x,y in zip(a,b)],
            'value', draws=bootstrap_draws) for metric in ('raw','centered')}
    result = {"schema": SCHEMA, "run_dir": str(run_dir.resolve()),
              "inputs": {"predictions_sha256": sha(pred_path), "scales_sha256": sha(scales_path), "student_report_sha256": sha(report_path),
                         "student_report_recomputed": str(report_path.resolve())},
              "sample_seeds": packed.get("protocol", {}).get("sample_seeds"), "arm_names": list(ARMS),
              "holdout_count": len(_clip_ids_for_role(clips, "holdout")), "fit_example_count": len(_clip_ids_for_role(clips, "fit_examples")[:4]),
              "arms": arm_reports, "baseline": baseline_rows, "paired": paired,
              "independent_primary": independent, "independent_audio_minus_static": independent_pairs,
              "independent_recompute": {"used_dataset": False, "used_teacher": False, "used_405_or_test": False,
                                         "used_reported_curves": False, "scorer": "scripts.joint_motion_metrics.score_clip/summarize",
                                         "reported_reverse_arm": "not independently recomputed because reports contain summaries but not reverse curves"},
              "all_values_finite_json": True}
    if not _finite_nested(result):
        raise ValueError("nonfinite audit result")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    conclusion = output.with_name("audit_conclusion.md")
    lines = ["# Visual semantic pilot independent audit", "", f"Holdout clips: {result['holdout_count']}; packaged fit examples: {result['fit_example_count']}.", "", "All scores below are recomputed from packaged target/baseline/sample curves and training-fitted metric scales. They are not a new training result.", ""]
    for role in ("holdout", "fit_examples"):
        lines.append(f"## {role}")
        for kind, value in paired[role].items():
            b = value["bootstrap"]
            lines.append(f"- {kind}: sentence bootstrap mean ES difference {b['mean'] if b['mean'] is not None else 'NA'}; 95% interval [{b['q025'] if b['q025'] is not None else 'NA'}, {b['q975'] if b['q975'] is not None else 'NA'}].")
    lines += ["", "A positive audio-minus-static or oracle-minus-audio ES difference is worse because ES is lower-is-better; inspect the sign before claiming improvement.", "", "This audit does not certify naturalness, causal semantics, or paper-level generalization."]
    conclusion.write_text("\n".join(lines) + "\n", encoding="utf8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--student-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    result = audit(args.run, args.student_report, args.output, args.bootstrap_draws)
    print(json.dumps({"schema": SCHEMA, "holdout": result["holdout_count"], "fit_examples": result["fit_example_count"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
