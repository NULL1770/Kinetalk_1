"""Wait for declared completion files, then run frozen evaluations and tables."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_text());root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    states={j['name']:{'status':'waiting'} for j in plan['jobs']};running={};handles={}
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0'}
    while True:
        for name,(proc,job) in list(running.items()):
            code=proc.poll()
            if code is None:continue
            states[name].update(status='complete' if code==0 and Path(job['marker']).is_file() else 'failed',exit_code=code,finished=time.time())
            handles.pop(name).close();del running[name]
        for job in plan['jobs']:
            name=job['name']
            if states[name]['status']!='waiting':continue
            if Path(job['marker']).exists():states[name]['status']='complete';continue
            if any(Path(p).exists() for p in job.get('failure_files',[])):
                states[name]['status']='dependency_failed';continue
            if not all(Path(p).is_file() for p in job.get('requires',[])):continue
            if len(running)>=plan.get('max_concurrent',1):continue
            path=root/(name+'.log');handles[name]=path.open('ab')
            proc=subprocess.Popen(job['argv'],env=env,stdin=subprocess.DEVNULL,stdout=handles[name],stderr=subprocess.STDOUT,start_new_session=True)
            running[name]=(proc,job);states[name].update(status='running',pid=proc.pid,started=time.time(),log=str(path))
            print(json.dumps({'event':'launched','job':name,'pid':proc.pid}),flush=True)
        write(root/'queue_status.json',{'updated':time.time(),'jobs':states})
        if all(s['status'] in ('complete','failed','dependency_failed') for s in states.values()):break
        time.sleep(20)
    write(root/'queue_complete.json',{'all_passed':all(s['status']=='complete' for s in states.values()),'jobs':states})


if __name__=='__main__':main()
