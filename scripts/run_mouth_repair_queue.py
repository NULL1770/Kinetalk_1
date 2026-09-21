"""Detached smoke -> fixed-budget mouth recovery -> gated protected stages."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh queue path required')
    a.output.mkdir(parents=True)
    if shutil.disk_usage(a.output).free < 800_000_000:raise OSError('Require 800 MB free for recovery and protected checkpoint output')
    code=Path(__file__).resolve().parents[1]; records=[]
    try:
        for label in ('smoke','recovery'):
            command=[sys.executable,'-u',str(code/'scripts/recover_paper_mouth.py'),
                '--data',str(a.data),'--source',str(a.source),'--output',str(a.output/label),'--epochs',str(a.epochs)]
            command += ['--smoke'] if label=='smoke' else ['--continue-protected']
            with (a.output/(label+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=code,stdout=log,stderr=subprocess.STDOUT)
                write(a.output/'queue_status.json',{'status':'running','stage':label,'pid':child.pid,
                    'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),'test_loaded':False})
                rc=child.wait()
            records.append({'stage':label,'exit_code':rc});write(a.output/'process_records.json',records)
            if rc:raise RuntimeError(f'{label} exited {rc}')
            status=json.loads((a.output/label/'status.json').read_text())
            if label=='smoke' and not (a.output/label/'evaluation.json').exists():raise RuntimeError('Smoke evaluation missing')
        write(a.output/'queue_status.json',{'status':'complete','result':status,'records':records,
            'default_replaced':False,'test_loaded':False,'quality_claim':'See acceptance result; completion is not success'})
    except BaseException as exc:
        write(a.output/'queue_status.json',{'status':'failed','exception':repr(exc),'records':records,'test_loaded':False})
        raise


if __name__=='__main__':main()
