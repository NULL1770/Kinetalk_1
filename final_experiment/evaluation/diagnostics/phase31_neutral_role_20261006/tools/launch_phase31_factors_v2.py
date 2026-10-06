"""Sequential read-only D0 workers with durable status, no model changes."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/root/kinetalk_phase31_neutral_role_20261006')
state=root/'factor_launch_v2.json'
assert not state.exists(),'Do not duplicate launch'
results={}
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
for model in ('neutral','native'):
    argv=['/root/miniconda3/bin/python','-u',str(root/'tools/frozen_factor_diagnostic_v2.py'),
          '--binding',str(root/'factor_binding_v3.json'),'--model',model,
          '--protocol',str(root/'tools/phase30_ua_protocol.py'),'--output',str(root/f'factors_{model}_v2')]
    state.write_text(json.dumps(dict(status='running',current=model,argv=argv,results=results,
        pid=os.getpid(),updated_at=time.time(),test_loaded=False,training_performed=False),indent=2))
    with (root/f'factors_{model}_v2.log').open('w') as log:
        result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
    results[model]={'exit_code':result.returncode}
state.write_text(json.dumps(dict(status='complete' if all(v['exit_code']==0 for v in results.values()) else 'failed',
    results=results,pid=os.getpid(),updated_at=time.time(),test_loaded=False,training_performed=False),indent=2))
