"""Prepare provenance-bound continuous upper-face motion without VA conditions.

Only metadata-selected historical-fit exports are read. ``valid`` is the audio
deployment support; ``motion_mask`` is observation support for training/scoring.
Never use ``prepare_segments`` to decide deployment support: use
``contiguous_runs(clip['valid'])`` instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_sparse_brow_teacher import checked, sha

SCHEMA = "continuous_motion_dataset_v1"
UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]
ROLES = ("train", "holdout")


def contiguous_runs(mask):
    """Return half-open true intervals; reject nonbinary or nonvector masks."""
    values = torch.as_tensor(mask).detach().cpu()
    if values.ndim != 1 or not ((values == 0) | (values == 1)).all():
        raise ValueError("A one-dimensional binary mask is required")
    values = values.bool()
    edges = torch.diff(torch.cat((torch.tensor([False]), values, torch.tensor([False]))).int())
    return list(zip(torch.where(edges == 1)[0].tolist(), torch.where(edges == -1)[0].tolist()))


def _mask(values, shape, label):
    values = np.asarray(values)
    if values.shape != shape or not np.isin(values, [0, 1]).all():
        raise ValueError(f"Invalid {label} shape or values: {values.shape}")
    return values.astype(bool)


def _channels(values, frames):
    values = np.asarray(values)
    if values.shape == (52,):
        values = np.broadcast_to(values, (frames, 52))
    return _mask(values, (frames, 52), "channel mask").copy()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _safe_selection(selected):
    rows = selected.get("clips", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nonempty fixed selection is required")
    seen = set()
    for row in rows:
        cid = row.get("clip_id")
        if (not isinstance(cid, str) or not cid or cid in (".", "..") or
                any(c in cid for c in "/\\:") or cid in seen):
            raise ValueError("Unique safe clip IDs are required")
        if row.get("split") not in ROLES:
            raise ValueError("Only historical-fit train/holdout roles are accepted")
        for key in ("sentence", "speaker", "speaker_name", "emotion"):
            if key not in row:
                raise ValueError("Missing selection metadata: " + key)
        seen.add(cid)
    train = {row["sentence"] for row in rows if row["split"] == "train"}
    holdout = {row["sentence"] for row in rows if row["split"] == "holdout"}
    if train & holdout:
        raise ValueError("Training and diagnostic holdout sentences overlap")
    if not train:
        raise ValueError("Training members are required for fit-only statistics")
    return rows


def _joint_mask(clip):
    valid = torch.as_tensor(clip["valid"]).bool()
    motion_mask = torch.as_tensor(clip["motion_mask"]).bool()
    if motion_mask.shape != (len(valid), 9):
        raise ValueError("Expected [T,9] motion observation mask")
    return valid & motion_mask.all(-1)


def fit_statistics(clips):
    """Train only, equal clip weighting; frames are uniform within each clip.

    Residual scale is RMS about the AUDIO baseline (not target centering).
    Audio moments use native-valid frames, independently of motion observability.
    Context is concatenated global affect/intensity, identity, and audio b9.
    """
    fit = [c for c in clips if c["split"] == "train"]
    if not fit:
        raise ValueError("No train clips for statistics")
    audio_first, audio_second, contexts, residual_second, residual_ids = [], [], [], [], []
    for clip in fit:
        valid = torch.as_tensor(clip["valid"]).bool()
        audio = torch.as_tensor(clip["features"])[valid].double()
        context = torch.as_tensor(clip["context"]).double()
        if not len(audio) or not torch.isfinite(audio).all() or not torch.isfinite(context).all():
            raise ValueError("Finite native audio/context required for statistics")
        audio_first.append(audio.mean(0))
        audio_second.append(audio.square().mean(0))
        contexts.append(context)
        joint = _joint_mask(clip)
        if joint.any():
            residual = (torch.as_tensor(clip["motion9"])[joint].double()
                        - torch.as_tensor(clip["b9"]).double())
            if not torch.isfinite(residual).all():
                raise ValueError("Nonfinite observed motion")
            residual_second.append(residual.square().mean(0))
            residual_ids.append(clip["clip_id"])
    if not residual_second:
        raise ValueError("No jointly observed training motion")
    mean = torch.stack(audio_first).mean(0)
    variance = (torch.stack(audio_second).mean(0) - mean.square()).clamp_min(0)
    context = torch.stack(contexts)
    context_mean = context.mean(0)
    result = {
        "residual_scale": torch.stack(residual_second).mean(0).sqrt().clamp_min(.02).float(),
        "audio_mean": mean.float(), "audio_scale": variance.sqrt().clamp_min(.01).float(),
        "context_mean": context_mean.float(),
        "context_scale": (context - context_mean).square().mean(0).sqrt().clamp_min(.01).float(),
        "train_clip_ids": [c["clip_id"] for c in fit],
        "residual_train_clip_ids": residual_ids,
        "residual_no_joint_observation_clip_ids": [c["clip_id"] for c in fit if c["clip_id"] not in residual_ids],
        "weighting": "equal clips; equal supported frames within clip; residual RMS has no target centering",
        "residual_scale_floor": .02, "audio_scale_floor": .01, "context_scale_floor": .01,
        "audio_support": "native valid only, independent of target mask",
        "residual_support": "native valid AND all nine observed channels",
    }
    return result


def prepare_segments(clips, stats, role, max_frames=200):
    """Supervised segments, never deployment windows; return (segments, report).

    Fixed, nonoverlapping chunks from each jointly observed run. No gap crossing,
    no end padding/duplication. Tails shorter than five frames are counted/dropped.
    """
    if role not in ROLES or not isinstance(max_frames, int) or max_frames < 5:
        raise ValueError("train/holdout role and max_frames >= 5 required")
    segments, reports = [], []
    for clip in clips:
        if clip["split"] != role:
            continue
        joint = _joint_mask(clip)
        report = {"clip_id": clip["clip_id"], "native_frames": len(joint),
                  "native_valid_frames": int(clip["valid"].sum()),
                  "joint_observed_frames": int(joint.sum()), "runs": 0,
                  "kept_frames": 0, "segments": 0, "dropped_short_frames": 0,
                  "dropped_short_segments": 0}
        for start, end in contiguous_runs(joint):
            report["runs"] += 1
            for left in range(start, end, max_frames):
                right = min(left + max_frames, end)
                if right - left < 5:
                    report["dropped_short_frames"] += right - left
                    report["dropped_short_segments"] += 1
                    continue
                residual = (clip["motion9"][left:right] - clip["b9"]) / stats["residual_scale"]
                audio = (clip["features"][left:right] - stats["audio_mean"]) / stats["audio_scale"]
                context = (clip["context"] - stats["context_mean"]) / stats["context_scale"]
                if not all(torch.isfinite(x).all() for x in (residual, audio, context)):
                    raise ValueError("Nonfinite normalized supervised segment")
                metadata = {**clip.get("metadata", {}), "clip_id": clip["clip_id"],
                            "split": role, "sentence": clip["sentence"], "start": left,
                            "end": right, "run_start": start, "run_end": end,
                            "support": "joint observed, training/scoring only"}
                segments.append({"residual": residual.float(), "audio": audio.float(),
                                 "context": context.float(), "b9": clip["b9"].clone(),
                                 "metadata": metadata})
                report["kept_frames"] += right - left
                report["segments"] += 1
        report["unobserved_native_valid_frames"] = report["native_valid_frames"] - report["joint_observed_frames"]
        reports.append(report)
    sums = {key: sum(r[key] for r in reports) for key in (
        "native_frames", "native_valid_frames", "joint_observed_frames", "runs", "kept_frames",
        "segments", "dropped_short_frames", "dropped_short_segments", "unobserved_native_valid_frames")}
    return segments, {"role": role, "max_frames": max_frames, "min_frames": 5,
                      "clips": reports, "totals": sums,
                      "excluded_clip_ids": [r["clip_id"] for r in reports if not r["segments"]]}


def build(selection, baseline, output):
    selection, baseline, output = Path(selection), Path(baseline), Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    selected, manifest = _read(selection), _read(baseline / "manifest.json")
    checked(baseline / "provenance.json", manifest["provenance.json"])
    source = _read(baseline / "provenance.json")
    if source.get("selection_sha256") != sha(selection):
        raise ValueError("Selection hash differs from frozen baseline provenance")
    if (source.get("query_motion_inference") is not False or source.get("inference_frozen") is not True
            or source.get("test_loaded") is not False or source.get("dev405_indexed") is not False):
        raise ValueError("Frozen acoustic-only historical-fit provenance required")
    rows = _safe_selection(selected)
    source_rows = {r["clip_id"]: r for r in source["clips"]}
    if len(source_rows) != len(source["clips"]) or set(source_rows) != {r["clip_id"] for r in rows}:
        raise ValueError("Selected/exported membership differs")
    clips, reports = [], []
    for row in rows:
        cid, exported = row["clip_id"], source_rows[row["clip_id"]]
        if any(row[key] != exported.get(key) for key in ("sentence", "speaker", "speaker_name", "emotion")):
            raise ValueError("Selected/exported metadata differs: " + cid)
        aname, vname = "arrays/" + cid + ".npz", "video_npz/" + cid + ".npz"
        for name, key in ((aname, "arrays"), (vname, "video_npz")):
            checked(baseline / name, manifest[name])
            if exported.get(key) != name or exported.get(key + "_sha256") != manifest[name]["sha256"]:
                raise ValueError("Per-clip provenance/manifest differs: " + cid)
        with np.load(baseline / aname, allow_pickle=False) as z:
            b = {k: z[k].copy() for k in z.files}
        with np.load(baseline / vname, allow_pickle=False) as z:
            v = {k: z[k].copy() for k in z.files}
        n = len(b["valid"])
        valid = _mask(b["valid"], (n,), "native valid")
        if not valid.any():
            raise ValueError("Empty native audio support: " + cid)
        channels = _channels(b["channel_mask"], n)
        if (str(b["clip_id"].item()) != cid or str(v["clip_id"].item()) != cid
                or not np.array_equal(valid, _mask(v["valid"], (n,), "video native valid"))
                or not np.array_equal(channels, _channels(v["channel_mask"], n))
                or not np.array_equal(b["times"], v["times"])
                or not np.allclose(b["times"], np.arange(n) / 25, atol=1e-7, rtol=0)):
            raise ValueError("Native arrays/reference clock, mask or identity differs: " + cid)
        modes = v["mode_names"].tolist()
        target_name, baseline_name = "native coefficient reference (not inference input)", "run12 frozen audio baseline"
        if modes.count(target_name) != 1 or modes.count(baseline_name) != 1:
            raise ValueError("Unique motion reference and acoustic baseline required")
        target = v["motions"][modes.index(target_name)]
        if (target.shape != (n, 52) or b["baseline52"].shape != (n, 52)
                or b["features"].shape != (n, 1540)
                or not np.array_equal(v["motions"][modes.index(baseline_name)], b["baseline52"])):
            raise ValueError("Unexpected native shape or mismatched frozen baseline")
        observed = valid[:, None] & channels
        if (not np.isfinite(target[observed]).all() or not np.isfinite(b["features"][valid]).all()
                or not np.isfinite(b["baseline52"]).all()):
            raise ValueError("Nonfinite observed motion or acoustic input/output")
        global_code = np.concatenate((b["affect_global"].reshape(-1), b["affect_intensity"].reshape(-1)))
        identity = b["identity_code"].reshape(-1)
        b9 = b["baseline52"][valid][:, UPPER].mean(0)
        context = np.concatenate((global_code, identity, b9))
        if not len(identity) or not len(global_code) or not np.isfinite(context).all():
            raise ValueError("Finite nonempty global/identity context required")
        metadata = {**row, "native_metadata": exported.get("native_metadata"),
                    "source_arrays_sha256": manifest[aname]["sha256"],
                    "source_reference_sha256": manifest[vname]["sha256"],
                    "audio_relative_path": str(b["audio_relative_path"].item()),
                    "audio_sha256": str(b["audio_sha256"].item()),
                    "audio_offset_seconds": float(b["audio_offset_seconds"].item())}
        clip = {**row, "metadata": metadata}
        for key, value in {"features": b["features"], "global": global_code,
                "affect_global": b["affect_global"], "affect_intensity": b["affect_intensity"],
                "identity_code": identity, "b9": b9, "context": context,
                "motion9": target[:, UPPER], "target52": target, "baseline52": b["baseline52"]}.items():
            clip[key] = torch.from_numpy(np.asarray(value).copy()).float()
        clip.update(valid=torch.from_numpy(valid), times=torch.from_numpy(b["times"].copy()),
                    channel_mask=torch.from_numpy(channels), motion_mask=torch.from_numpy(observed[:, UPPER]))
        clips.append(clip)
        reports.append({"clip_id": cid, "split": row["split"], "frames": n,
                        "native_valid_frames": int(valid.sum()),
                        "joint_observed_frames": int(_joint_mask(clip).sum())})
    dims = {(c["global"].numel(), c["identity_code"].numel(), c["context"].numel()) for c in clips}
    if len(dims) != 1:
        raise ValueError("Condition dimensions vary across clips")
    stats = fit_statistics(clips)
    provenance = {"schema": SCHEMA, "selection_sha256": sha(selection),
        "baseline_manifest_sha256": sha(baseline / "manifest.json"),
        "baseline_provenance_sha256": sha(baseline / "provenance.json"),
        "code_sha256": sha(__file__), "upper_channels": UPPER,
        "condition_dimensions": dict(zip(("global", "identity", "context"), next(iter(dims)))),
        "b9_support": "frozen audio baseline mean over native-valid frames; no target mean",
        "deployment_support": "native audio valid; independent of target channel mask",
        "motion_support": "all nine channels jointly observed, contiguous runs only",
        "va_used": False, "sealed_test_loaded": False, "dev405_indexed": False,
        "scope": "Previously exposed historical-fit internal development, not independent final evaluation"}
    payload = {"schema": SCHEMA, "clips": clips, "stats": stats, "provenance": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = {**provenance, "dataset_sha256": sha(output), "clips": reports,
               "statistics": {k: v.tolist() if torch.is_tensor(v) else v for k, v in stats.items()}}
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.selection, args.baseline, args.output)
    print(json.dumps({"schema": SCHEMA, "clips": len(result["clips"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
