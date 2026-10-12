"""Run remotely after uploading and verifying the new snapshot files."""
from pathlib import Path
import hashlib
import json
import subprocess
root = Path('/root/kinetalk_v5_20261011')
code = root / 'code_posture_pilot'
if (root / 'posture_pilot.lock').exists() or (root / 'posture_pilot_queue.log').exists():
    raise FileExistsError('Posture pilot already dispatched')
expected = json.loads((root / 'posture_upload_hashes.json').read_text())
for name, digest in expected.items():
    if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
        raise ValueError('Uploaded posture source differs: ' + name)
hashes = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
(root / 'posture_pilot_source_hashes.json').write_text(json.dumps(hashes, indent=2))
with (root / 'posture_pilot_queue.log').open('x') as log:
    worker = subprocess.Popen(['/root/miniconda3/bin/python', '-u', str(root / 'run_posture_pilots.py')],
        cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=worker.pid, detached=True, waits_for='route_followup_complete.json',
    smoke_updates=16, updates_per_arm=256, arms=3, full_development_per_arm=1367,
    sampling_seeds=[47,91,137], training_seeds=[47], promotion=False)))
