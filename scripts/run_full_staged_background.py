"""Launch target: durable process state and explicit exit status for staged run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8');temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();code=Path(__file__).resolve().parents[1]
    command=[sys.executable,'-u',str(code/'scripts/train_full_staged.py'),
        '--source-run','/root/kinetalk_runs/teacher_schedule_v1/scaled_centered_v1/uniform',
        '--audio','/root/kinetalk_runs/audio_text_v1/data/audio.pt',
        '--targets','/root/kinetalk_runs/audio_text_v1/data/intensity_targets/targets.pt',
        '--enrollment','/root/kinetalk_runs/teacher_schedule_v1/data_locked/enrollment.jsonl',
        '--native-root','/root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask',
        '--output',str(args.run),'--epochs','12']
    if args.resume:command+=['--resume']
    started=datetime.now(timezone.utc).isoformat()
    status=args.run.parent/(args.run.name+'_process.json')
    child=subprocess.Popen(command,cwd=code)
    write(status,{'state':'training','pid':child.pid,'started_utc':started,'command':command})
    result=child.wait()
    record={'state':'complete' if result==0 else 'failed','training_exit_code':result,'pid':child.pid,
        'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'command':command}
    if result==0:
        exporter=code/'scripts/export_full_staged_examples.py'
        if exporter.is_file():
            exported=subprocess.run([sys.executable,str(exporter),'--run',str(args.run),
                '--previous-visual','/root/kinetalk_runs/audio_text_v1/visual','--output',str(args.run/'visual')],cwd=code)
            record['visual_export_exit_code']=exported.returncode
        else:record['visual_export']='not installed; saved curves remain available'
    write(status,record)
    raise SystemExit(result)


if __name__=='__main__':main()
