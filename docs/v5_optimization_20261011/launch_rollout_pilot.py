from pathlib import Path
import json
import subprocess
root = Path('/root/kinetalk_v5_20261011')
if (root / 'geometry_rollout_mean').exists(): raise FileExistsError('Existing run')
command = ['/root/miniconda3/bin/python', '-u', str(root / 'run_rollout_pilot.py')]
with (root / 'rollout_queue.log').open('w') as log:
    proc = subprocess.Popen(command, cwd=root / 'code', stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps({'pid': proc.pid, 'detached': True, 'command': command, 'updates': 256}))
