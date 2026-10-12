"""Verify isolated source upload, then detach the matched decoder-state queue."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path('/root/kinetalk_v5_20261011')
code = root/'code_decoder_state'
if (root/'decoder_state_pilot.lock').exists() or (root/'decoder_state_queue.log').exists():
    raise FileExistsError('Decoder-state queue already dispatched')
expected = json.loads((root/'decoder_state_upload_hashes.json').read_text())
for name, digest in expected.items():
    path = root/'run_decoder_state_pilots.py' if name == 'run_decoder_state_pilots.py' else code/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Upload hash differs: '+name)
subprocess.run(['/root/miniconda3/bin/python', '-m', 'py_compile',
    str(code/'kinetalk_v4/unified_model_v4.py'), str(code/'scripts/train_v5_mixed.py'),
    str(root/'run_decoder_state_pilots.py')], check=True)
sources = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
(root/'decoder_state_source_hashes.json').write_text(json.dumps(sources, indent=2))
with (root/'decoder_state_queue.log').open('x') as log:
    worker = subprocess.Popen(['/root/miniconda3/bin/python', '-u', str(root/'run_decoder_state_pilots.py')],
        cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=worker.pid, detached=True, waits_for='acoustic_geometry_complete.json',
    new_parameters=0, arms=3, updates_per_arm=256, full_development=1367,
    promotion=False, paper_result=False, test_loaded=False)))
