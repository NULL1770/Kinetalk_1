"""Private-session transport; never includes credentials in experiment sources."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def request(payload):
    proc = subprocess.run([sys.executable, str(ROOT / '.codex-finalizer/remote_ops.py')],
                          input=json.dumps(payload), text=True, encoding='utf8',
                          errors='replace', capture_output=True, cwd=ROOT)
    if proc.returncode:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout

if __name__ == '__main__':
    print(request(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))))
