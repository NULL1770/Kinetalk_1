"""Wait for matched pilots, then train/evaluate the annotation-aware HDTF route."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path('/root/kinetalk_v5_20261011'); CODE=ROOT/'code_content_style_route'; PY='/root/miniconda3/bin/python'
STATE=ROOT/'content_style_route_state.json'; LOCK=ROOT/'content_style_route.lock'

def write(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2),encoding='utf8'); tmp.replace(path)

def main():
    with LOCK.open('x') as f:f.write(str(os.getpid()))
    stage='waiting_for_matched_pilots'
    try:
        deadline=time.monotonic()+2*3600
        while not (ROOT/'mixed_pilot_queue_complete.json').exists():
            prior=json.loads((ROOT/'mixed_pilot_queue_state.json').read_text())
            if prior['status']=='failed':raise RuntimeError(prior.get('error','matched queue failed'))
            try:os.kill(prior['pid'],0)
            except ProcessLookupError:raise RuntimeError('matched queue exited without completion')
            if time.monotonic()>deadline:raise TimeoutError('matched queue timeout')
            write(STATE,dict(status='waiting',stage=stage,pid=os.getpid(),updated_unix=time.time()));time.sleep(30)
        common=[PY,'-u','scripts/train_v5_mixed.py','--binding',str(ROOT/'binding.json'),
          '--checkpoint',str(ROOT/'geometry_rollout_mean/final.pt'),'--output',str(ROOT/'content_style_route'),
          '--seed','47','--epochs','1','--steps','256','--eval-limit','128','--mead-batch-size','8',
          '--hdtf-batch-size','4','--hdtf-weight',str(1/3),'--hdtf-scope','content_style','--rollout-mean-weight','.5',
          '--hdtf-manifest',str(ROOT/'hdtf_binding_final/training_windows_verified.jsonl'),
          '--hdtf-features','/root/kinetalk_unified_v2_20261010/hdtf_full/features',
          '--hdtf-source-root','/root/kinetalk_unified_v2_20261010/hdtf_full']
        jobs=[('train',common),('development',[PY,'-u','scripts/evaluate_v5_candidate.py','--binding',str(ROOT/'binding.json'),
          '--checkpoint',str(ROOT/'content_style_route/final.pt'),'--output',str(ROOT/'content_style_route_eval'),
          '--selection-json',str(ROOT/'diagnostic_seed47/selection.json')]),('dynamics',[PY,'-u','scripts/evaluate_v4_dynamics.py',
          '--binding',str(ROOT/'binding.json'),'--checkpoint',str(ROOT/'content_style_route/final.pt'),
          '--output',str(ROOT/'content_style_route_dynamics'),'--limit','128','--seeds','47','91','137'])]
        for suffix,cmd in jobs:
            stage='content_style_'+suffix;write(STATE,dict(status='running',stage=stage,pid=os.getpid(),updated_unix=time.time()))
            with (ROOT/(stage+'.log')).open('x') as log:subprocess.run(cmd,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(ROOT/'content_style_route_complete.json',dict(status='complete',test_loaded=False,promotion=False,training_started=True,pid=os.getpid()))
        write(STATE,dict(status='complete',stage=stage,updated_unix=time.time()))
    except Exception as e:
        write(STATE,dict(status='failed',stage=stage,error=repr(e),updated_unix=time.time()));raise

if __name__=='__main__':main()
