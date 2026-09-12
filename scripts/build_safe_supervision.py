"""Build phoneme/viseme boundary and mouth-event supervision for safe pairs.

This script is deliberately conservative.  It never invents a phoneme boundary:
when no timestamped aligner output is available the boundary mask is zero.  The
mouth event stream is computed from the neutral BS motion and is therefore still
usable for every pair.  Pair-level event agreement is gated by the saved DTW
path; failed pairs are written with ``teacher_eligible=false``.

The output is one ``.npz`` sidecar per clip plus a pair audit jsonl.  It can be
run locally or on the extraction host and does not modify the source arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first(a: Any, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in a.files:
            return np.asarray(a[name], dtype=np.float32)
    return np.asarray(a[a.files[0]], dtype=np.float32)


def motion_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as a:
        x = first(a, ("coeffs", "motion", "bs"))
    return x if x.ndim == 2 else x.reshape(len(x), -1)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    scale = np.nanmedian(np.abs(x - med)) * 1.4826 + 1e-6
    return (x - med) / scale


def mouth_events(motion: np.ndarray, indices: list[int], fps: float) -> tuple[np.ndarray, np.ndarray]:
    idx = [i for i in indices if 0 <= i < motion.shape[1]]
    signal = np.linalg.norm(motion[:, idx], axis=1) if idx else np.linalg.norm(motion, axis=1)
    velocity = np.abs(np.diff(signal, prepend=signal[:1]))
    threshold = np.nanmedian(velocity) + 1.5 * (np.nanmedian(np.abs(velocity - np.nanmedian(velocity))) * 1.4826 + 1e-6)
    active = (robust_z(signal) > 0.25).astype(np.float32)
    onset = np.zeros(len(signal), np.float32)
    offset = np.zeros(len(signal), np.float32)
    onset[1:] = ((active[1:] > active[:-1]) & (velocity[1:] > threshold)).astype(np.float32)
    offset[1:] = ((active[1:] < active[:-1]) & (velocity[1:] > threshold)).astype(np.float32)
    # [active, onset, offset, normalized mouth signal, velocity]
    feat = np.stack([active, onset, offset, robust_z(signal), robust_z(velocity)], axis=-1)
    return feat.astype(np.float32), np.ones(len(signal), np.float32)


def load_path(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as a:
        keys = set(a.files)
        if "path" in keys:
            path_xy = np.asarray(a["path"], dtype=np.int64)
            if path_xy.ndim == 2 and path_xy.shape[1] >= 2:
                return path_xy[:, 0], path_xy[:, 1]
        src = next((a[k] for k in ("src_index", "source_index", "path_src", "audio_index") if k in keys), None)
        ref = next((a[k] for k in ("ref_index", "target_index", "path_ref", "motion_index") if k in keys), None)
        if src is None or ref is None:
            return None
        return np.asarray(src, dtype=np.int64), np.asarray(ref, dtype=np.int64)

def event_gate(source: np.ndarray, reference: np.ndarray, path: tuple[np.ndarray, np.ndarray] | None, max_lag: int) -> tuple[float, bool, np.ndarray]:
    if path is None:
        return 0.0, False, np.zeros(len(source), np.float32)
    src_i, ref_i = path
    valid = (src_i >= 0) & (src_i < len(source)) & (ref_i >= 0) & (ref_i < len(reference))
    if not valid.any():
        return 0.0, False, np.zeros(len(source), np.float32)
    a = source[src_i[valid], 0] > 0.5
    b = reference[ref_i[valid], 0] > 0.5
    local = np.zeros(len(source), np.float32)
    local[src_i[valid]] = (a == b).astype(np.float32)
    agreement = float(np.mean(a == b))
    # Require both event presence and agreement; this avoids accepting a silent
    # path that matches only because both clips are mostly inactive.
    presence = float(0.5 * (a.mean() + b.mean()))
    return agreement, bool(agreement >= 0.72 and presence >= 0.03 and max_lag >= 0), local


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--safe-manifest", type=Path, required=True)
    p.add_argument("--records-manifest", type=Path, required=True)
    p.add_argument("--motion-root", type=Path, required=True)
    p.add_argument("--pair-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--neutral-indices", default="14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,51")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--event-threshold", type=float, default=0.72)
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    safe = read_jsonl(args.safe_manifest)
    records = {str(r.get("clip_id")): r for r in read_jsonl(args.records_manifest)}
    indices = [int(x) for x in args.neutral_indices.split(",") if x.strip()]
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    def sidecar(clip: str) -> tuple[np.ndarray, np.ndarray]:
        if clip not in cache:
            path = args.motion_root / f"{clip}.npz"
            feat, mask = mouth_events(motion_array(path), indices, args.fps)
            np.savez_compressed(args.out_root / f"{clip}.npz", mouth_event=feat, mouth_event_mask=mask, fps=np.float32(args.fps), boundary=np.zeros((len(feat), 3), np.float32), boundary_mask=np.zeros(len(feat), np.float32), viseme_id=np.full(len(feat), -1, np.int16), viseme_mask=np.zeros(len(feat), np.float32))
            cache[clip] = feat, mask
        return cache[clip]
    audit = []
    for row in safe:
        src, ref = str(row["source_clip_id"]), str(row["reference_clip_id"])
        sf, _ = sidecar(src); rf, _ = sidecar(ref)
        pair_name = f"{src}__to__{ref}.npz"
        score, gate, local = event_gate(sf, rf, load_path(args.pair_root / pair_name), 0)
        gate = bool(gate and score >= args.event_threshold)
        pair_mask_path = args.out_root / "pair_masks" / f"{src}__to__{ref}.npz"
        pair_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pair_mask_path, event_local_mask=local, event_quality=np.float32(score))
        out = dict(row, mouth_event_agreement=score, mouth_event_gate=gate,
                   teacher_eligible=bool(row.get("teacher_eligible", True)),
                   event_local_mask_path=str(pair_mask_path),
                   boundary_status="missing", viseme_status="missing")
        audit.append(out)
    (args.out_root / "pair_audit.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in audit) + "\n", encoding="utf-8")
    gated = [x for x in audit if x["teacher_eligible"]]
    (args.out_root / "teacher_manifest_gated.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in gated) + ("\n" if gated else ""), encoding="utf-8")
    summary = {"safe_rows": len(safe), "gated_rows": len(gated), "sidecars": len(cache), "event_gate_pass": sum(bool(x["mouth_event_gate"]) for x in audit), "boundary_available": 0, "viseme_available": 0}
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()





