"""Materialize neutral reference motion on each pair's canonical clock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_motion_path(record: dict[str, Any], motion_root: Path | None) -> Path:
    candidates: list[Path] = []
    for raw in (record.get("npz_final"), record.get("bs"), record.get("npz")):
        if not raw:
            continue
        value = str(raw).replace("\\", "/")
        path = Path(value)
        candidates.append(path)
        if motion_root is not None:
            candidates.extend((motion_root / PurePosixPath(value), motion_root / PurePosixPath(value).name))
    for raw in (record.get("bs_rel"), record.get("motion_rel")):
        if raw and motion_root is not None:
            candidates.append(motion_root / PurePosixPath(str(raw).replace("\\", "/")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing neutral motion for {record.get('clip_id')}; tried: {candidates}")


def materialize_target(pair_path: Path, reference_record: dict[str, Any], motion_root: Path | None, fps: float):
    with np.load(pair_path, allow_pickle=False) as pair:
        times = np.asarray(pair["canonical_times"], dtype=np.float64)
        geometry_mask = np.asarray(pair["teacher_mask"], dtype=bool)
    motion_path = resolve_motion_path(reference_record, motion_root)
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = np.asarray(archive["coeffs"] if "coeffs" in archive.files else archive[archive.files[0]], dtype=np.float32)
    if motion.ndim != 2 or len(motion) < 2:
        raise ValueError(f"Invalid neutral motion shape in {motion_path}: {motion.shape}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    motion_times = np.arange(len(motion), dtype=np.float64) / float(fps)
    target = np.stack(
        [np.interp(times, motion_times, motion[:, channel], left=motion[0, channel], right=motion[-1, channel])
         for channel in range(motion.shape[1])], axis=1).astype(np.float32)
    mask = geometry_mask & (times >= motion_times[0]) & (times <= motion_times[-1])
    if not mask.any():
        raise ValueError("neutral target has no valid canonical frames")
    return target, mask, times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = {str(row["clip_id"]): row for row in read_jsonl(args.records)}
    rows = read_jsonl(args.manifest)
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = str(row["source_clip_id"]); reference = str(row["reference_clip_id"])
        pair_path = args.pair_root / Path(str(row["teacher_artifact"])).name
        try:
            if not pair_path.is_file():
                raise FileNotFoundError(f"Missing pair artifact: {pair_path}")
            if reference not in records:
                raise KeyError(f"Reference clip is absent from records: {reference}")
            target, mask, times = materialize_target(pair_path, records[reference], args.motion_root, args.fps)
            name = f"{source}__to__{reference}.npz"
            np.savez_compressed(args.out / name, canonical_neutral_target=target, canonical_neutral_mask=mask,
                                canonical_times=times, source_clip_id=source, reference_clip_id=reference)
            output.append(dict(row, stage1_target_artifact=str(args.out / name),
                               stage1_target_kind="neutral_reference_on_canonical_times",
                               stage1_target_mask_frames=int(mask.sum())))
        except Exception as exc:
            failures.append({"row": index, "source": source, "reference": reference, "error": str(exc)})
    (args.out / "stage1_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps({"input_rows": len(rows), "output_rows": len(output), "bad_rows": len(failures), "bad": failures[:20]}, indent=2), encoding="utf-8")
    print(json.dumps({"input_rows": len(rows), "output_rows": len(output), "bad_rows": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
