from pathlib import Path
import json
import os
import subprocess

root = Path('/root/kinetalk_v5_20261011')
if (root / 'diagnostic_seed47').exists() or (root / 'diagnostic.pid').exists():
    raise FileExistsError('Do not relaunch an existing diagnostic')
command = ['/root/miniconda3/bin/python', '-u', 'scripts/audit_v4_controls.py',
           '--binding', str(root / 'binding.json'),
           '--checkpoint', '/root/kinetalk_v4_20261011/mead_formal_seed47/final.pt',
           '--output', str(root / 'diagnostic_seed47'), '--limit', '128', '--seed', '47']
with (root / 'diagnostic.log').open('w') as log:
    process = subprocess.Popen(command, cwd=root / 'code', stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / 'diagnostic.pid').write_text(str(process.pid))
receipt = {'command': command, 'cwd': str(root / 'code'), 'pid': process.pid,
           'detached': True, 'output': str(root / 'diagnostic_seed47'), 'test_loaded': False}
(root / 'diagnostic_launch.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
