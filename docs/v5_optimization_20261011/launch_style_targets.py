"""Detach one CPU frozen style-target diagnostic after script upload."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
root=Path('/root/kinetalk_v5_20261011');code=root/'code_style_targets'
if (root/'style_targets.log').exists() or (root/'style_targets').exists():
    raise FileExistsError('Target-style diagnostic already dispatched')
cmd=['/root/miniconda3/bin/python','-u','scripts/audit_v5_style_targets.py',
    '--binding',str(root/'binding.json'),'--output',str(root/'style_targets'),
    '--limit','64','--device','cpu']
for arm in ('mead_matched','content_style_route','posture_direct','posture_privileged'):
    if not (root/arm/'complete.json').is_file():raise FileNotFoundError('Training incomplete: '+arm)
    cmd+=['--checkpoint',str(root/arm/'final.pt')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (root/'style_targets.log').open('x') as log:
    worker=subprocess.Popen(cmd,cwd=code,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=worker.pid,device='cpu',detached=True,matched_pairs=64,checkpoints=4,
    source_sha256=hashlib.sha256((code/'scripts/audit_v5_style_targets.py').read_bytes()).hexdigest(),
    test_loaded=False,training=False,promotion=False)))
