from pathlib import Path
import hashlib
import json
import shutil
import subprocess

root = Path('/root/kinetalk_v5_20261011')
code = root / 'code_pilot_followup'
if code.exists() or (root / 'pilot_followup.lock').exists():
    raise FileExistsError('Follow-up already dispatched; inspect state instead')
shutil.copytree(root / 'code', code)
sources = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in code.rglob('*.py')}
(root / 'pilot_followup_source_hashes.json').write_text(json.dumps(sources, indent=2))
with (root / 'pilot_followup.log').open('x') as log:
    process = subprocess.Popen(['/root/miniconda3/bin/python', '-u', str(root / 'run_pilot_followup.py')],
        cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, detached=True, waits_for='mixed_pilot_queue_complete.json',
    full_development=True, style_factor_controls=True, new_training=False)))
