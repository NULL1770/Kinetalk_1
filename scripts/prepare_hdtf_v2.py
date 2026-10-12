"""Inventory/extract HDTF without copying videos or reusing unknown audio tensors."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v2.hdtf import (ExtractOptions, atomic_json, atomic_jsonl, inventory_one,
                              options_signature, process_source)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path(r"E:\final_detaset\HDTF"))
    ap.add_argument("--output", type=Path, default=Path(r"E:\final_detaset\HDTF_processed_v2_20261010"))
    ap.add_argument("--model", type=Path, default=Path(r"D:\实验室项目\新实验\数据集\models\face_landmarker.task"))
    ap.add_argument("--rig", type=Path, default=Path(__file__).resolve().parents[1] / "final_experiment/rig/fixed_arkit_rig_20260923.npz")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--start-seconds", type=float, default=0)
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--source-ids", default="")
    ap.add_argument("--min-free-gb", type=float, default=5)
    ap.add_argument("--timeout-hours", type=float, default=18)
    args = ap.parse_args()
    if not args.input.is_dir() or not args.model.is_file() or not args.rig.is_file():
        ap.error("input/model/rig path does not exist")
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < args.min_free_gb * 2**30:
        ap.error("insufficient free storage")
    paths = sorted((args.input / "videos").glob("*.mp4"))
    if not paths:
        ap.error("no source videos")
    with ThreadPoolExecutor(max_workers=min(8, args.workers)) as pool:
        inventory = list(pool.map(lambda p: inventory_one(p, args.seed), paths))
    atomic_jsonl(args.output / "inventory.jsonl", inventory)
    by_split = Counter(r["split"] for r in inventory)
    total_seconds = sum(r.get("duration", 0) for r in inventory)
    estimated_bytes = total_seconds * (16000 * 2 + 25 * (52 * 4 + 70)) * 1.2
    atomic_json(args.output / "inventory_summary.json",
                {"sources": len(inventory), "split_sources": dict(by_split),
                 "identity_groups": len({r["speaker_group"] for r in inventory}),
                 "named_identity_groups": len({r["speaker_group"] for r in inventory if r["split"] != "quarantine"}),
                 "total_seconds": total_seconds, "hours": total_seconds / 3600,
                 "estimated_audio_raw_motion_gb": estimated_bytes / 2**30,
                 "free_gb": shutil.disk_usage(args.output).free / 2**30,
                 "identity_verification": "named_filename_groups_only; Radio quarantined; manual face audit still required"})
    print(json.dumps({"inventory_sources": len(inventory), "split_sources": dict(by_split),
                      "hours": total_seconds / 3600, "estimated_gb": estimated_bytes / 2**30}), flush=True)
    if args.inventory_only:
        return 0
    if shutil.disk_usage(args.output).free < estimated_bytes + args.min_free_gb * 2**30:
        ap.error("estimated source-level audio/motion exceeds free space reserve")
    chosen = [r for r in inventory if r["inventory_status"] == "ok" and r["duration"] > args.start_seconds]
    if args.source_ids:
        ids = set(args.source_ids.split(","))
        chosen = [r for r in chosen if r["source_id"] in ids]
    # Balance early extraction across identity groups; quarantine sources come last.
    chosen.sort(key=lambda r: (r["split"] == "quarantine", r["speaker_group"], r["source_id"]))
    first, remaining, seen = [], [], set()
    for rec in chosen:
        if rec["speaker_group"] not in seen:
            first.append(rec)
            seen.add(rec["speaker_group"])
        else:
            remaining.append(rec)
    chosen = first + remaining
    if args.limit:
        chosen = chosen[:args.limit]
    if not chosen:
        ap.error("no eligible sources match request")
    opts = ExtractOptions(model=str(args.model.resolve()), rig=str(args.rig.resolve()), output=str(args.output.resolve()),
                          start_seconds=args.start_seconds, max_seconds=args.max_seconds)
    signature, provenance = options_signature(opts)
    atomic_json(args.output / "run_binding.json", {"signature": signature, "provenance": provenance,
                "pid": os.getpid(), "started_unix": time.time(), "requested_sources": len(chosen),
                "argv": sys.argv, "timeout_hours": args.timeout_hours})
    started = time.monotonic()
    outcomes = []
    windows = []
    pool = ProcessPoolExecutor(max_workers=args.workers)
    pending = {pool.submit(process_source, rec, opts, signature, provenance): rec for rec in chosen}
    summary = {"pid": os.getpid(), "state": "running", "completed_sources": 0,
               "requested_sources": len(chosen), "outcome_counts": {}, "frames": 0,
               "eligible_windows": 0, "elapsed_seconds": 0., "updated_unix": time.time()}
    atomic_json(args.output / "status.json", summary)
    try:
      while pending:
        finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["updated_unix"] = time.time()
        atomic_json(args.output / "status.json", summary)
        if time.monotonic() - started > args.timeout_hours * 3600:
            summary["state"] = "timeout"
            atomic_json(args.output / "status.json", summary)
            for proc in pool._processes.values():
                proc.terminate()
            raise TimeoutError("run exceeded configured hard timeout")
        if shutil.disk_usage(args.output).free < args.min_free_gb * 2**30:
            summary["state"] = "storage_reserve_stop"
            atomic_json(args.output / "status.json", summary)
            for proc in pool._processes.values():
                proc.terminate()
            raise OSError("storage reserve exhausted")
        for future in finished:
            rec = pending.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                result = {"source_id": rec["source_id"], "processing_status": "error", "error": f"{type(exc).__name__}: {exc}"[:1000]}
                atomic_json(args.output / "errors" / f"{rec['source_id']}.json", result)
            outcomes.append(result)
            if result["processing_status"] in {"ok", "cached"}:
                (args.output / "errors" / f"{result['source_id']}.json").unlink(missing_ok=True)
                path = args.output / "windows_by_source" / f"{result['source_id']}.jsonl"
                windows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
            summary = {"pid": os.getpid(), "state": "running", "completed_sources": len(outcomes),
                       "requested_sources": len(chosen), "outcome_counts": dict(Counter(r["processing_status"] for r in outcomes)),
                       "frames": sum(r.get("n_frames", 0) for r in outcomes),
                       "eligible_windows": sum(r.get("n_training_eligible_windows", 0) for r in outcomes),
                       "elapsed_seconds": time.monotonic() - started, "updated_unix": time.time(),
                       "last_source": result["source_id"]}
            atomic_json(args.output / "status.json", summary)
            ordered_outcomes = sorted(outcomes, key=lambda r: r["source_id"])
            ordered_windows = sorted(windows, key=lambda r: (r["source_id"], r["start_frame"]))
            atomic_jsonl(args.output / "processed_sources.jsonl", ordered_outcomes)
            atomic_jsonl(args.output / "quality_windows.jsonl", ordered_windows)
            atomic_jsonl(args.output / "training_windows.jsonl", [r for r in ordered_windows if r["split"] != "quarantine"])
            print(json.dumps({"done": len(outcomes), "requested": len(chosen), "source": result["source_id"],
                              "status": result["processing_status"], "frames": result.get("n_frames"),
                              "quality": result.get("quality_ratio"), "windows": result.get("n_training_eligible_windows"),
                              "fps": result.get("frames_per_second"), "error": result.get("error")}), flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    ordered_outcomes = sorted(outcomes, key=lambda r: r["source_id"])
    ordered_windows = sorted(windows, key=lambda r: (r["source_id"], r["start_frame"]))
    atomic_jsonl(args.output / "processed_sources.jsonl", ordered_outcomes)
    atomic_jsonl(args.output / "quality_windows.jsonl", ordered_windows)
    atomic_jsonl(args.output / "training_windows.jsonl", [r for r in ordered_windows if r["split"] != "quarantine"])
    summary["state"] = "complete"
    summary["window_splits"] = dict(Counter(r["split"] for r in windows))
    summary["mean_source_quality"] = sum(r.get("quality_ratio", 0) for r in outcomes) / max(len(outcomes), 1)
    atomic_json(args.output / "status.json", summary)
    return 1 if any(r["processing_status"] == "error" for r in outcomes) else 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("GLOG_minloglevel", "2")
    raise SystemExit(main())
