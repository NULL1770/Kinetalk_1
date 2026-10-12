"""Verify uploads and detach the geometry queue; preserve all live snapshots."""
from pathlib import Path
import hashlib
import json
import subprocess
root=Path('/root/kinetalk_v5_20261011');code=root/'code_acoustic_geometry'
if (root/'acoustic_geometry_pilot.lock').exists() or (root/'acoustic_geometry_queue.log').exists():
    raise FileExistsError('Acoustic geometry queue already dispatched')
expected=json.loads((root/'acoustic_geometry_upload_hashes.json').read_text())
for name,digest in expected.items():
    if hashlib.sha256((code/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Acoustic geometry upload differs: '+name)
sources={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
(root/'acoustic_geometry_source_hashes.json').write_text(json.dumps(sources,indent=2))
with (root/'acoustic_geometry_queue.log').open('x') as log:
    worker=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(root/'run_acoustic_geometry_pilots.py')],
        cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=worker.pid,detached=True,waits_for='posture_pilot_complete.json',
    smoke_updates=16,arms=3,updates_per_arm=256,full_development_per_arm=1367,
    training_seeds=[47],sampling_seeds=[47,91,137],promotion=False,paper_result=False)))
