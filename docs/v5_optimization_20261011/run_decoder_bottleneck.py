"""One detached, bounded frozen diagnostic after the existing state queue."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path('/root/kinetalk_v5_20261011')
CODE=ROOT/'code_decoder_state'
SCRIPT=ROOT/'decoder_bottleneck_tools/audit_v5_decoder_bottleneck.py'
PY='/root/miniconda3/bin/python'
CHECKPOINT=ROOT/'state_raw_direct/final.pt'
EXPECTED='310fe9fa348ecb7f6210fc854f3c867dc055e470db00be1d2dc38722016df702'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,d):
    temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(d,indent=2));temporary.replace(p)


def main():
    with (ROOT/'decoder_bottleneck.lock').open('x') as f: f.write(str(os.getpid()))
    state=ROOT/'decoder_bottleneck_state.json';stage='waiting_state_queue';child=None
    try:
        deadline=time.monotonic()+3600
        while not (ROOT/'decoder_state_complete.json').is_file():
            prior=json.loads((ROOT/'decoder_state_pilot_state.json').read_text())
            if prior['status']=='failed': raise RuntimeError('Prior queue failed: '+str(prior))
            os.kill(prior['pid'],0)
            if time.monotonic()>deadline: raise TimeoutError('State wait exceeded1hour')
            write(state,dict(status='waiting',stage=stage,pid=os.getpid(),updated_unix=time.time()));time.sleep(15)
        if sha(CHECKPOINT)!=EXPECTED: raise ValueError('Predeclared raw-direct checkpoint changed')
        manifest=json.loads((ROOT/'decoder_state_source_hashes.json').read_text())
        if any(sha(CODE/n)!=h for n,h in manifest.items()): raise ValueError('Decoder snapshot changed')
        script_sha=sha(SCRIPT)
        env=os.environ.copy();env.update(PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for stage,limit in [('decoder_bottleneck_smoke',4),('decoder_bottleneck_full',128)]:
            command=[PY,'-u',str(SCRIPT),'--binding',str(ROOT/'binding.json'),'--checkpoint',str(CHECKPOINT),
                '--output',str(ROOT/stage),'--limit',str(limit),'--device','cuda']
            started=time.monotonic()
            with (ROOT/(stage+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    write(state,dict(status='running',stage=stage,pid=os.getpid(),child_pid=child.pid,
                        elapsed_seconds=time.monotonic()-started,updated_unix=time.time()))
                    if time.monotonic()-started>15*60:
                        child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise TimeoutError('Owned diagnostic stage exceeded15minutes')
                    time.sleep(15)
                if child.returncode: raise RuntimeError('Diagnostic failed: '+stage+' exit '+str(child.returncode))
                child=None
            receipt=json.loads((ROOT/stage/'complete.json').read_text())
            if receipt['test_loaded'] or receipt['checkpoint_sha256']!=EXPECTED:raise ValueError('Diagnostic binding differs')
            if sha(CHECKPOINT)!=EXPECTED or sha(SCRIPT)!=script_sha: raise ValueError('Frozen input changed')
        write(ROOT/'decoder_bottleneck_complete.json',dict(status='complete',checkpoint_sha256=EXPECTED,
            source_sha256=script_sha,diagnostic_only=True,test_loaded=False,promotion=False))
        write(state,dict(status='complete',stage=stage,pid=os.getpid(),updated_unix=time.time()))
    except Exception as e:
        write(state,dict(status='failed',stage=stage,pid=os.getpid(),child_pid=child.pid if child else None,error=repr(e),updated_unix=time.time()))
        raise


if __name__=='__main__':main()
