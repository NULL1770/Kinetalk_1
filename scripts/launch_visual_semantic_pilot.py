"""Detached bounded build/student/six-arm training pipeline with real exit status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');temp.replace(path)


def supervise(args):
    args.output.mkdir(parents=True,exist_ok=True)
    commands=[
        [sys.executable,'-u','-m','scripts.build_visual_semantic_dataset','--selection',str(args.selection),'--baseline',str(args.baseline),'--teacher',str(args.teacher),'--output',str(args.output/'dataset.pt')],
        [sys.executable,'-u','-m','scripts.fit_visual_semantic_student','--dataset',str(args.output/'dataset.pt'),'--output',str(args.output/'student')],
        [sys.executable,'-u','-m','scripts.train_visual_semantic_pilot','--dataset',str(args.output/'dataset.pt'),'--student',str(args.output/'student/predictions.pt'),'--output',str(args.output/'six_arm30'),'--epochs','30','--device','cuda'],
    ]
    start=time.time();status=args.output/'pipeline_status.json'
    for name,command in zip(('dataset','student','training'),commands):
        write(status,{'state':'running','phase':name,'pid':os.getpid(),'start':start,'command':command})
        with (args.output/(name+'.log')).open('w',encoding='utf8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
            write(status,{'state':'running','phase':name,'pid':os.getpid(),'child_pid':process.pid,'start':start,'command':command})
            code=process.wait()
        if code:
            write(status,{'state':'failed','phase':name,'exit_code':code,'elapsed_seconds':time.time()-start})
            return code
    write(status,{'state':'complete','exit_code':0,'elapsed_seconds':time.time()-start})
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('selection','baseline','teacher','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--supervise',action='store_true');a=p.parse_args()
    if a.supervise:raise SystemExit(supervise(a))
    if a.output.exists():raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--supervise']
    for n in ('selection','baseline','teacher','output'):cmd+=['--'+n,str(getattr(a,n).resolve())]
    with (a.output/'supervisor.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(a.output/'launch.json',{'pid':child.pid,'command':cmd,'launched_at':time.time(),'completion_not_yet_verified':True})
    print(json.dumps({'pid':child.pid,'output':str(a.output)}))
