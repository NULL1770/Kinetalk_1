"""Detached serial queue; immutable sources, owned-process timeout, durable receipts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(code, manifest):
    for rel, digest in manifest.items():
        if sha(code / rel) != digest:
            raise RuntimeError('Immutable source changed: ' + rel)


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    root, code = Path(plan['output']), Path(plan['code_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'queue_state.json').exists():
        raise FileExistsError('Existing queue evidence; no implicit restart')
    sources = json.loads(Path(plan['source_manifest']).read_text())
    verify(code, sources)
    state = dict(status='running', started=time.time(), supervisor_pid=os.getpid(), jobs=[],
                 test_loaded=False, promotion=False)
    write(root / 'queue_state.json', state)
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
           'CUDA_VISIBLE_DEVICES': '0', 'PYTHONUNBUFFERED': '1'}
    current = None
    try:
        for job in plan['jobs']:
            verify(code, sources)
            log_path = root / (job['name'] + '.log')
            entry = dict(name=job['name'], argv=job['argv'], marker=job['marker'],
                         status='running', started=time.time(), log=str(log_path))
            with log_path.open('xb') as log:
                current = subprocess.Popen(job['argv'], cwd=code, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                entry['pid'] = current.pid
                state['jobs'].append(entry)
                write(root / 'queue_state.json', state)
                print(json.dumps(entry), flush=True)
                timed_out = False
                while current.poll() is None:
                    if time.time() - entry['started'] > job.get('timeout_seconds', 2700):
                        timed_out = True
                        os.killpg(current.pid, signal.SIGTERM)
                        try:
                            current.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(current.pid, signal.SIGKILL)
                        break
                    state['updated'] = time.time()
                    write(root / 'queue_state.json', state)
                    time.sleep(5)
                exit_code = current.wait()
                marker = Path(job['marker'])
                marker_status = json.loads(marker.read_text()).get('status') if marker.is_file() else None
                entry.update(exit_code=exit_code, finished=time.time(), timeout=timed_out,
                    status='complete' if exit_code == 0 and marker_status == 'complete' else 'failed')
                current = None
                write(root / 'queue_state.json', state)
                if entry['status'] != 'complete':
                    raise RuntimeError('Stage failed: ' + job['name'])
        verify(code, sources)
        state.update(status='complete', finished=time.time())
        write(root / 'complete.json', state)
    except BaseException as exc:
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
        state.update(status='failed', error=str(exc), finished=time.time())
        write(root / 'failed.json', state)
        raise
    finally:
        write(root / 'queue_state.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True, type=Path)
    run(p.parse_args().plan)
