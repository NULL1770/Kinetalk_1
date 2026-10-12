"""Audit/apply conservative HDTF SG candidates without touching raw NPZ files."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v2.hdtf import atomic_json, atomic_jsonl, sha256
from kinetalk_v2.hdtf_postprocess import Policy, aggregate_audit, load_motion, process_file

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"E:\final_detaset\HDTF_processed_v2_20261010"))
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--mode", choices=("audit", "apply"), default="audit")
    ap.add_argument("--candidate", type=int, choices=(3,5,7), default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--source-ids", default="")
    ap.add_argument("--watch-minutes",type=float,default=0,
                    help="Repeated resumable snapshots, maximum 120 minutes")
    ap.add_argument("--poll-seconds",type=float,default=60)
    args = ap.parse_args()
    if args.watch_minutes:
        if not 0 < args.watch_minutes <= 120:ap.error('watch duration must be <=120 minutes')
        args.output=args.output or(args.root/f'motion_processed_sg{args.candidate}')
        deadline=time.monotonic()+args.watch_minutes*60
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(args.root),
                 '--output',str(args.output),'--mode',args.mode,'--candidate',str(args.candidate)]
        if args.limit:command.extend(['--limit',str(args.limit)])
        if args.source_ids:command.extend(['--source-ids',args.source_ids])
        runs=0
        while time.monotonic()<deadline:
            completed=subprocess.run(command)
            runs+=1
            extract_status_path=args.root/'status.json'
            extract_status=json.loads(extract_status_path.read_text(encoding='utf-8')) if extract_status_path.exists() else {}
            state='source_complete' if extract_status.get('state')=='complete' else 'watching'
            atomic_json(args.output/'watcher_status.json',{'pid':os.getpid(),'state':state,
                        'completed_snapshots':runs,'last_exit_code':completed.returncode,
                        'source_state':extract_status.get('state'),'source_completed':extract_status.get('completed_sources'),
                        'updated_unix':time.time(),'watch_minutes':args.watch_minutes})
            if state=='source_complete':return completed.returncode
            time.sleep(min(max(args.poll_seconds,10),max(deadline-time.monotonic(),0)))
        atomic_json(args.output/'watcher_status.json',{'pid':os.getpid(),'state':'duration_complete',
                    'completed_snapshots':runs,'updated_unix':time.time(),'watch_minutes':args.watch_minutes})
        return 0
    root = args.root
    done = []
    for path in sorted((root / "sources").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("processing_status") not in {"ok", "cached"} or rec.get("split") == "quarantine": continue
        if not args.include_test and rec.get("split") == "test": continue
        if args.source_ids and rec["source_id"] not in set(args.source_ids.split(",")): continue
        # A postprocessor never reads a partially written source receipt.
        if not Path(rec["motion_path"]).is_file(): continue
        done.append(rec)
    if args.limit: done = done[:args.limit]
    if not done: ap.error("no completed non-quarantine sources")
    policy = Policy(candidate_window=args.candidate)
    out_root = args.output or (root / f"motion_processed_sg{args.candidate}")
    audit_rows, errors = [], []
    for rec in done:
        try:
            # Audit candidate with identical guard policy. This is per-source,
            # not a synthetic reconstruction or an emotion label.
            out = out_root / "motion" / f"{rec['source_id']}.npz"
            row = process_file(Path(rec["motion_path"]), out, policy, source_record=rec)
            row["split"] = rec["split"]
            audit_rows.append(row)
        except Exception as exc:
            errors.append({"source_id": rec["source_id"], "error": f"{type(exc).__name__}: {exc}"})
    if not audit_rows: ap.error(f"all sources failed: {errors[:1]}")
    by_split = {s: [r for r in audit_rows if r["split"] == s] for s in ("train", "val", "test")}
    summary = aggregate_audit([r for r in audit_rows if r["split"] in {"train", "val"}], args.candidate)
    summary.update({"mode": args.mode, "source_root": str(root.resolve()), "output_root": str(out_root.resolve()),
                    "include_test": args.include_test, "errors": errors,
                    "split_counts": {s: len(rows) for s, rows in by_split.items()},
                    "started_unix": time.time(), "policy": vars(policy)})
    summary['train_only'] = aggregate_audit(by_split['train'],args.candidate) if by_split['train'] else None
    summary['validation_only'] = aggregate_audit(by_split['val'],args.candidate) if by_split['val'] else None
    summary['policy_selection_roles'] = 'train; validation reported separately; test excluded'
    summary['code_sha256'] = sha256(Path(__file__).resolve().parents[1]/'kinetalk_v2/hdtf_postprocess.py')
    atomic_json(out_root / "audit_summary.json", summary)
    atomic_jsonl(out_root / "processed_sources.jsonl", sorted(audit_rows, key=lambda r: r["source_id"]))
    source_map={r['source_id']:r for r in audit_rows}
    windows=[]
    for sid,rec in source_map.items():
        path=root/'windows_by_source'/f'{sid}.jsonl'
        if not path.exists():continue
        for line in path.read_text(encoding='utf-8').splitlines():
            win=json.loads(line)
            if win['split'] not in {'train','val'}:continue
            win.update(motion_processed_path=rec['motion_processed_path'],
                       motion_processed_sha256=rec['motion_processed_sha256'],
                       postprocess_schema=rec['postprocess_schema'],postprocess_signature=rec['postprocess_signature'],
                       raw_motion_sha256=rec['raw_motion_sha256'])
            windows.append(win)
    atomic_jsonl(out_root/'windows_processed.jsonl',sorted(windows,key=lambda r:(r['source_id'],r['start_frame'])))
    atomic_json(out_root/'policy_binding.json',{'schema':summary['schema'],'policy':vars(policy),
               'code_sha256':summary['code_sha256'],'source_root':str(root.resolve()),
               'role':'explicit_opt_in_only','raw_mutated':False,'no_test_parameter_selection':True,
               'audit_summary_sha256':sha256(out_root/'audit_summary.json')})
    print(json.dumps({k: summary[k] for k in ("mode", "candidate_window", "n_sources", "n_frames", "split_counts", "mean_fallback_fraction", "mean_smoothed_fraction", "median_velocity_ratio", "median_acceleration_ratio", "median_jaw_peak_retention", "median_jaw_closure_rate_delta", "median_blink_left_peak_retention", "median_blink_right_peak_retention", "errors")}, ensure_ascii=False), flush=True)
    return 1 if errors else 0

if __name__ == "__main__": raise SystemExit(main())
