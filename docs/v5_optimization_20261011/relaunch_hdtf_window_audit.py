from pathlib import Path
import json
import subprocess
root = Path('/root/kinetalk_v5_20261011')
command = ['/root/miniconda3/bin/python', '-u', 'scripts/audit_hdtf_binding_v5.py',
    '--manifest', str(root / 'hdtf_windows.jsonl'), '--features', '/root/kinetalk_unified_v2_20261010/hdtf_full/features',
    '--source-root', '/root/kinetalk_unified_v2_20261010/hdtf_full', '--output', str(root / 'hdtf_binding_audit_windows')]
if (root / 'hdtf_binding_audit_windows').exists(): raise FileExistsError('No duplicate audit')
with (root / 'hdtf_audit_windows.log').open('w') as log:
    proc = subprocess.Popen(command, cwd=root / 'code', stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / 'hdtf_audit_windows.pid').write_text(str(proc.pid))
print(json.dumps({'pid': proc.pid, 'command': command, 'fix': 'quality applies to query windows, not full partly failed recording'}))
