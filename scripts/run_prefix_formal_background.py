"""Detached fixed-budget prefix formal supervisor with persistent exit status."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh formal output required')
    code=Path(__file__).resolve().parents[1];root=Path('/root/kinetalk_full_staged_20260917')
    command=[sys.executable,'-u',str(code/'scripts/train_prefix_formal.py'),
        '--source-run','/root/kinetalk_runs/teacher_schedule_v1/scaled_centered_v1/uniform',
        '--audio','/root/kinetalk_runs/audio_text_v1/data/audio.pt',
        '--targets','/root/kinetalk_runs/audio_text_v1/data/intensity_targets/targets.pt',
        '--enrollment','/root/kinetalk_runs/teacher_schedule_v1/data_locked/enrollment.jsonl',
        '--native-root','/root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask',
        '--trained-run',str(root/'run12'),'--history-run',str(root/'history12'),
        '--centered-run',str(root/'repair12/centered_prior'),'--pilot-run',str(root/'prefix_pilot30'),
        '--output',str(args.output),'--epochs','12']
    status=args.output.with_name(args.output.name+'_process.json');started=datetime.now(timezone.utc).isoformat()
    def write(value):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(status)
    child=subprocess.Popen(command,cwd=code)
    write({'status':'running','pid':child.pid,'started_utc':started,'command':command})
    result=child.wait()
    write({'status':'complete' if result==0 else 'failed','pid':child.pid,'exit_code':result,'started_utc':started,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'command':command})
    raise SystemExit(result)


if __name__=='__main__':main()
