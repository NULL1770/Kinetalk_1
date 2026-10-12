"""Server follow-up scoring after the bounded matched pilots; no model fitting."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path('/root/kinetalk_v5_20261011')
CODE = ROOT / 'code_pilot_followup'
STATE = ROOT / 'pilot_followup_state.json'
PY = '/root/miniconda3/bin/python'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    tmp.replace(path)


def main():
    with (ROOT / 'pilot_followup.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    stage = 'waiting_for_matched_pilots'
    try:
        deadline = time.monotonic() + 2*3600
        while not (ROOT / 'mixed_pilot_queue_complete.json').exists():
            write(STATE, dict(status='waiting', stage=stage, pid=os.getpid(), updated_unix=time.time()))
            prior = json.loads((ROOT / 'mixed_pilot_queue_state.json').read_text())
            if prior['status'] == 'failed':
                raise RuntimeError('Matched pilot queue failed: ' + prior.get('error', 'inspect logs'))
            try:
                os.kill(prior['pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('Matched pilot queue exited without completion')
            if time.monotonic() > deadline:
                raise TimeoutError('Matched pilots did not finish in two hours')
            time.sleep(30)
        for arm in ('mead_matched', 'mixed_matched'):
            checkpoint = ROOT / arm / 'final.pt'
            jobs = [('full_development', [PY, '-u', 'scripts/evaluate_v5_candidate.py',
                '--binding', str(ROOT / 'binding.json'), '--checkpoint', str(checkpoint),
                '--output', str(ROOT / (arm + '_full_eval'))]),
                ('factor_controls', [PY, '-u', 'scripts/audit_v5_controls.py',
                '--binding', str(ROOT / 'binding.json'), '--checkpoint', str(checkpoint),
                '--output', str(ROOT / (arm + '_controls')), '--limit', '32'])]
            for suffix, command in jobs:
                stage = arm + '_' + suffix
                write(STATE, dict(status='running', stage=stage, pid=os.getpid(), updated_unix=time.time()))
                with (ROOT / (stage + '.log')).open('x') as log:
                    subprocess.run(command, cwd=CODE, check=True, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
        write(ROOT / 'pilot_followup_complete.json', dict(status='complete', test_loaded=False,
            promotion=False, training_started=False, pid=os.getpid()))
        write(STATE, dict(status='complete', stage=stage, updated_unix=time.time()))
    except Exception as error:
        write(STATE, dict(status='failed', stage=stage, error=repr(error), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
