"""Fixed paired sentence-bootstrap audit of saved motion predictions only.

This script never fits a model, chooses a rank from development results, or
calibrates amplitudes. Positive improvements mean the candidate beats the
paired baseline on exactly the same native-motion targets and valid bins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import EMOTIONS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_paired(left, right):
    for key in ("clip_id", "sentence_id"):
        if list(left[key]) != list(right[key]):
            raise ValueError(f"predictions are not paired: {key}")
    for key in ("target", "weight", "emotion_id"):
        if left[key].shape != right[key].shape or not torch.equal(left[key], right[key]):
            raise ValueError(f"predictions are not paired: {key}")


def speaker_ids(curves):
    if "speaker_id" in curves:
        return [str(v) for v in curves["speaker_id"]], "saved speaker_id"
    result = []
    for clip in curves["clip_id"]:
        match = re.match(r"^(mead_[^_]+)_", str(clip), re.IGNORECASE)
        if not match:
            raise ValueError(f"cannot recover a MEAD speaker from {clip}; require saved speaker_id")
        result.append(match.group(1))
    return result, "parsed MEAD speaker prefix from clip_id"


def clip_statistics(prediction, target, weight, channels):
    if prediction.shape != target.shape or weight.shape != target.shape[:2]:
        raise ValueError("incompatible prediction/target/weight shapes")
    p, y = prediction[:, :, channels].double(), target[:, :, channels].double()
    w = weight.double()
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("invalid bin weights")
    mask = w > 0
    if not torch.isfinite(p[mask]).all() or not torch.isfinite(y[mask]).all():
        raise ValueError("nonfinite observed motion")
    p, y = torch.where(mask[..., None], p, 0), torch.where(mask[..., None], y, 0)
    ww = w[..., None]
    denominator = w.sum(1)[:, None, None].clamp_min(1)
    pc = torch.where(mask[..., None], p - (p * ww).sum(1, keepdim=True) / denominator, 0)
    yc = torch.where(mask[..., None], y - (y * ww).sum(1, keepdim=True) / denominator, 0)
    return torch.stack([
        ((p-y).square()*ww).sum((1,2)), (y.square()*ww).sum((1,2)),
        (p.square()*ww).sum((1,2)), w.sum(1)*len(channels),
        (pc*yc*ww).sum((1,2)), (pc.square()*ww).sum((1,2)), (yc.square()*ww).sum((1,2)),
    ], -1).numpy()


def scalar_summary(rows):
    error, target, pred, count, cov, vp, vy = rows.sum(0)
    return {"native_mse": float(error/count) if count else None,
        "r2_against_zero": float(1-error/target) if target > 0 else None,
        "prediction_energy_ratio": float(pred/target) if target > 0 else None,
        "prediction_rms_amplitude_ratio": float(np.sqrt(pred/target)) if target > 0 else None,
        "pooled_centered_correlation": float(np.clip(cov/np.sqrt(vp*vy),-1,1)) if vp > 0 and vy > 0 else None,
        "prediction_rms": float(np.sqrt(pred/count)) if count else None,
        "target_rms": float(np.sqrt(target/count)) if count else None,
        "target_energy": float(target), "sse": float(error), "weighted_values": float(count)}


def paired_summary(candidate, baseline, sentence_ids, indices, *, samples=5000, seed=45):
    ids = np.asarray(indices, dtype=np.int64)
    if not len(ids):
        return None
    a, b = candidate[ids], baseline[ids]
    if not np.array_equal(a[:, 1], b[:, 1]) or not np.array_equal(a[:, 3], b[:, 3]):
        raise ValueError("paired statistics have different targets/weights")
    difference = b[:, 0]-a[:, 0]
    target, count = a[:, 1].sum(), a[:, 3].sum()
    groups = {}
    for local, global_id in enumerate(ids):
        row = groups.setdefault(str(sentence_ids[global_id]), np.zeros(3))
        row += [difference[local], a[local,1], a[local,3]]
    ci_mse, ci_r2, valid_samples = None, None, 0
    if len(groups) >= 2 and samples > 0:
        totals = np.stack(list(groups.values()))
        picked = np.random.default_rng(seed).integers(len(totals), size=(samples,len(totals)))
        sums = totals[picked].sum(1)
        valid = (sums[:,1] > 0) & (sums[:,2] > 0)
        valid_samples = int(valid.sum())
        if valid_samples:
            ci_mse = np.quantile(sums[valid,0]/sums[valid,2],[.025,.975]).tolist()
            ci_r2 = np.quantile(sums[valid,0]/sums[valid,1],[.025,.975]).tolist()
    return {"clips": len(ids), "sentences": len(groups), "candidate": scalar_summary(a), "baseline": scalar_summary(b),
        "native_mse_improvement": float(difference.sum()/count) if count > 0 else None,
        "r2_improvement": float(difference.sum()/target) if target > 0 else None,
        "native_mse_improvement_ci95": ci_mse, "r2_improvement_ci95": ci_r2,
        "positive_sentence_count": sum(float(v[0]) > 0 for v in groups.values()),
        "negative_sentence_count": sum(float(v[0]) < 0 for v in groups.values()),
        "bootstrap_valid_samples": valid_samples, "bootstrap_requested_samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    torch.set_num_threads(4)
    summary_path = args.results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    provenance = json.loads((args.results/"provenance.json").read_text(encoding="utf8"))
    common = provenance["target_channels"]
    groups = {"all_expression": common, "upper_expression": [i for i in [5,6,12,13,41,42,43,44,45] if i in common],
        "brows": [i for i in range(41,46) if i in common],
        "eyes_expression": [i for i in [5,6,12,13] if i in common],
        "mouth": [i for i in range(14,41) if i in common], "jaw17": [17] if 17 in common else []}
    selected = {arm: row["selected_rank_per_method"]["rrr"] for arm,row in summary["arms"].items()}
    fixed = {arm:[key] for arm,key in selected.items()}
    fixed["content_temporal"] = list(dict.fromkeys(fixed["content_temporal"] + ["rrr_rank8", "pca_rank8", summary["arms"]["content_temporal"]["selected_rank_per_method"]["ridge"]]))
    curves, hashes = {}, {}
    for arm, keys in fixed.items():
        path = args.results / (arm+"_predictions.pt")
        if not path.exists() or path.stat().st_size < 100:
            raise ValueError(f"prediction artifact missing/incomplete: {path}")
        values = torch.load(path, map_location="cpu", weights_only=False)
        for key in keys:
            curves[arm+":"+key] = values[key]["external_dev"]
        hashes[arm] = sha(path)
        del values
    reference = next(iter(curves.values()))
    for value in curves.values():
        ensure_paired(reference,value)
    speakers, speaker_source = speaker_ids(reference)
    emotion = reference["emotion_id"].tolist()
    n = len(emotion)
    populations = {"nonneutral": [i for i,e in enumerate(emotion) if e != 0],
                   "neutral": [i for i,e in enumerate(emotion) if e == 0]}
    stats = {}
    for name,value in curves.items():
        for mode in ("full","zero","reverse","shuffle"):
            stats[(name,mode)] = {group:clip_statistics(value["motion"][mode],value["target"],value["weight"],channels)
                                 for group,channels in groups.items() if channels}
    comparisons = []
    for arm,key in selected.items():
        name = arm+":"+key
        for mode in ("zero","reverse","shuffle"):
            comparisons.append((name+"__vs__"+mode,(name,"full"),(name,mode)))
    focus = "content_temporal:"+selected["content_temporal"]
    for arm in ("content", "temporal"):
        name = arm+":"+selected[arm]
        comparisons.append((focus+"__vs__"+name,(focus,"full"),(name,"full")))
    for key in ("pca_rank8",summary["arms"]["content_temporal"]["selected_rank_per_method"]["ridge"]):
        name = "content_temporal:"+key
        comparisons.append((focus+"__vs__"+name,(focus,"full"),(name,"full")))
    rrr8 = "content_temporal:rrr_rank8"
    if focus != rrr8:
        comparisons.append((rrr8+"__vs__content_temporal:pca_rank8",(rrr8,"full"),("content_temporal:pca_rank8","full")))
    report = {"schema":"predictable_motion_paired_audit_v1", "source_summary_sha256":sha(summary_path),
        "script_sha256":sha(__file__), "prediction_sha256":hashes, "bootstrap_samples":args.samples,
        "bootstrap_seed":args.seed, "cluster_unit":"sentence", "selection":selected,
        "scope":"fixed saved predictions; development audit with exploratory group breakdowns, no hyperparameter or model reselection",
        "pairing":"clip_id/sentence_id order and target/weight/emotion tensors exactly equal",
        "speaker_source":speaker_source, "clip_count":n, "sentence_count":len(set(reference["sentence_id"])),
        "speaker_count":len(set(speakers)), "groups":groups, "comparisons":{}}
    for name,left,right in comparisons:
        output = {"candidate":left,"baseline":right,"populations":{},"by_emotion":{},"by_speaker":{}}
        for population,ids in populations.items():
            output["populations"][population] = {group:paired_summary(stats[left][group],stats[right][group],reference["sentence_id"],ids,samples=args.samples,seed=args.seed) for group in stats[left]}
        # Breakdowns are diagnostics; do not select a model/scale per group.
        for eid in sorted(set(emotion)):
            ids = [i for i,e in enumerate(emotion) if e==eid]
            output["by_emotion"][EMOTIONS[eid]] = {group:paired_summary(stats[left][group],stats[right][group],reference["sentence_id"],ids,samples=0) for group in stats[left]}
        for speaker in sorted(set(speakers)):
            ids = [i for i,s in enumerate(speakers) if s==speaker and emotion[i]!=0]
            output["by_speaker"][speaker] = {group:paired_summary(stats[left][group],stats[right][group],reference["sentence_id"],ids,samples=0) for group in stats[left]}
        for breakdown in ("by_emotion", "by_speaker"):
            for group_rows in output[breakdown].values():
                for group, detail in list(group_rows.items()):
                    if detail is not None:
                        group_rows[group] = {k: detail[k] for k in ("clips", "sentences",
                            "native_mse_improvement", "r2_improvement", "positive_sentence_count", "negative_sentence_count")}
                        group_rows[group].update(candidate_r2=detail["candidate"]["r2_against_zero"],
                            baseline_r2=detail["baseline"]["r2_against_zero"],
                            candidate_amplitude_ratio=detail["candidate"]["prediction_rms_amplitude_ratio"])
        report["comparisons"][name]=output
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    for name,row in report["comparisons"].items():
        if name.startswith(focus):
            print(json.dumps({"comparison":name,"nonneutral":{k:{q:row["populations"]["nonneutral"][k][q] for q in ("r2_improvement","r2_improvement_ci95")} for k in ("upper_expression","brows","eyes_expression")}}),flush=True)


if __name__ == "__main__":
    main()
