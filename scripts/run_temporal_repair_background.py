"""Durable one-phase launcher; never starts duplicate/replacement outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('align','direct','soft'),required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();code=Path(__file__).resolve().parents[1]
    args.root.mkdir(parents=True,exist_ok=True)
    run=args.root/args.phase
    if run.exists() and not args.resume:raise FileExistsError('Fresh phase directory required')
    command=[sys.executable,'-u',str(code/'scripts/train_temporal_repair.py'),
        '--source-run','/root/kinetalk_runs/teacher_schedule_v1/scaled_centered_v1/uniform',
        '--audio','/root/kinetalk_runs/audio_text_v1/data/audio.pt',
        '--targets','/root/kinetalk_runs/audio_text_v1/data/intensity_targets/targets.pt',
        '--enrollment','/root/kinetalk_runs/teacher_schedule_v1/data_locked/enrollment.jsonl',
        '--native-root','/root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask',
        '--trained-run','/root/kinetalk_full_staged_20260917/run12',
        '--output',str(run),'--phase',args.phase,'--epochs','12']
    if args.phase!='align':command+=['--align-run',str(args.root/'align')]
    if args.resume:command+=['--resume']
    path=args.root/(args.phase+'_process.json')
    def write(value):
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
    started=datetime.now(timezone.utc).isoformat()
    child=subprocess.Popen(command,cwd=code)
    write({'status':'running','pid':child.pid,'started_utc':started,'command':command})
    result=child.wait()
    write({'status':'complete' if result==0 else 'failed','pid':child.pid,'exit_code':result,
           'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'command':command})
    raise SystemExit(result)


if __name__=='__main__':main()
