"""Credential-free request wrapper around the existing private SSH transport."""
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]


def transport(request):
    r = subprocess.run([sys.executable, str(PROJECT/'.codex-finalizer/remote_ops.py')],
        input=json.dumps(request), text=True, encoding='utf8', errors='replace',capture_output=True,
        cwd=PROJECT,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:] or r.stdout[-3000:])
    return r.stdout


if __name__=='__main__':
    print(transport(json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))))
