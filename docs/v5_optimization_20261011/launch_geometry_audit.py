"""Run remotely after the audit source is uploaded to its isolated snapshot."""
from pathlib import Path
import os
import hashlib
import json
import subprocess
root=Path('/root/kinetalk_v5_20261011');code=root/'code_geometry_audit'
if (root/'geometry_route_audit.log').exists() or (root/'geometry_route_audit').exists():
    raise FileExistsError('Geometry route diagnostic already dispatched')
cmd=['/root/miniconda3/bin/python','-u','scripts/audit_v5_geometry.py',
    '--binding',str(root/'binding.json'),'--output',str(root/'geometry_route_audit'),
    '--limit','128','--device','cpu']
for arm in ('mead_matched','mixed_matched','content_style_route','posture_control'):
    cmd+=['--checkpoint',str(root/arm/'final.pt')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (root/'geometry_route_audit.log').open('x') as log:
    worker=subprocess.Popen(cmd,cwd=code,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=worker.pid,device='cpu',detached=True,clips=128,checkpoints=4,
    source_sha256=hashlib.sha256((code/'scripts/audit_v5_geometry.py').read_bytes()).hexdigest(),
    training=False,test_loaded=False,promotion=False)))
