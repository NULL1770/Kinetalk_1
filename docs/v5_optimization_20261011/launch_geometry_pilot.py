from pathlib import Path
import json
import subprocess

root = Path('/root/kinetalk_v5_20261011')
if (root / 'pilot_queue.pid').exists(): raise FileExistsError('Pilot already dispatched')
command = ['/root/miniconda3/bin/python', '-u', str(root / 'run_matched_geometry_pilot.py')]
with (root / 'pilot_queue.log').open('w') as log:
    process = subprocess.Popen(command, cwd=root / 'code', stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / 'pilot_queue.pid').write_text(str(process.pid))
receipt = {'pid': process.pid, 'command': command, 'detached': True, 'arms': 2,
           'updates_per_arm': 256, 'same_initial_checkpoint': True, 'test_loaded': False}
(root / 'pilot_queue_launch.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
