from pathlib import Path
import hashlib
import json
import shutil
import subprocess

root = Path('/root/kinetalk_v5_20261011')
code = root / 'code_mixed_pilot'
if code.exists() or (root / 'mixed_pilot_queue.lock').exists():
    raise FileExistsError('Mixed pilots already dispatched; inspect state instead')
shutil.copytree(root / 'code', code)
files = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in code.rglob('*.py')}
(root / 'mixed_pilot_source_hashes.json').write_text(json.dumps(files, indent=2))
with (root / 'mixed_pilot_queue.log').open('x') as log:
    process = subprocess.Popen(['/root/miniconda3/bin/python', '-u', str(root / 'run_mixed_pilots.py')],
        cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(pid=process.pid, detached=True, smoke_updates=32,
                     matched_updates_per_arm=256, code=str(code), long_run_queued=False)))
