"""Run an explicit experiment DAG and record process exits durably.

Each job specifies argv, dependencies, output and a completion marker. A zero
exit without its marker is a failure. No job is automatically retrained or
relabelled as complete. Run this supervisor under nohup on the training host.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf8');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args(); plan=json.loads(args.plan.read_text())
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs={j['name']:j for j in plan['jobs']}; state={};running={};logs={}
    if (output/'supervisor_status.json').exists():
        raise FileExistsError('Existing supervisor state: inspect live jobs before restarting')
    for name,job in jobs.items():
        for dep in job.get('after',[]):
            if dep not in jobs:raise ValueError('Unknown dependency '+dep)
        state[name]={'status':'pending','argv':job['argv'],'marker':job['marker']}
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
         'NUMEXPR_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'}
    while True:
        for name,process in list(running.items()):
            code=process.poll()
            if code is None: continue
            status='complete' if code==0 and Path(jobs[name]['marker']).is_file() else 'failed'
            state[name].update(status=status,exit_code=code,finished=time.time())
            logs.pop(name).close();del running[name]
        for name,job in jobs.items():
            if state[name]['status']!='pending':continue
            dependencies=[state[d]['status'] for d in job.get('after',[])]
            if any(s in ('failed','dependency_failed') for s in dependencies):
                state[name]['status']='dependency_failed';continue
            if not all(s=='complete' for s in dependencies):continue
            # Ablations share a lane; the four primary methods can train in
            # parallel while auxiliary evaluations use their declared lane.
            lane=job.get('lane')
            if lane and any(jobs[n].get('lane')==lane for n in running):continue
            if len(running)>=plan.get('max_concurrent',4):continue
            log=output/(name+'.log'); logs[name]=log.open('ab')
            process=subprocess.Popen(job['argv'],cwd=plan['code_root'],env={**env,**job.get('env',{})},
                stdin=subprocess.DEVNULL,stdout=logs[name],stderr=subprocess.STDOUT,start_new_session=True)
            running[name]=process;state[name].update(status='running',pid=process.pid,started=time.time(),log=str(log))
            print(json.dumps({'event':'launched','job':name,'pid':process.pid}),flush=True)
        write(output/'supervisor_status.json',{'updated':time.time(),'jobs':state})
        if not running and not any(s['status']=='pending' for s in state.values()):break
        time.sleep(20)
    write(output/'supervisor_complete.json',{'jobs':state,'all_passed':all(s['status']=='complete' for s in state.values())})


if __name__=='__main__':main()
