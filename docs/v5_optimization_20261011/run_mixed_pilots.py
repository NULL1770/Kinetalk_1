"""Detached bounded data-expansion comparison; stop before a longer run."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/root/kinetalk_v5_20261011')
CODE = ROOT / 'code_mixed_pilot'
STATE = ROOT / 'mixed_pilot_queue_state.json'
PY = '/root/miniconda3/bin/python'
CHECKPOINT = ROOT / 'geometry_rollout_mean/final.pt'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def main():
    lock = ROOT / 'mixed_pilot_queue.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    binding = ROOT / 'binding.json'
    common = [PY, '-u', 'scripts/train_v5_mixed.py', '--binding', str(binding),
              '--checkpoint', str(CHECKPOINT), '--seed', '47', '--rollout-mean-weight', '.5']
    hdtf = ['--hdtf-manifest', str(ROOT / 'hdtf_binding_final/training_windows_verified.jsonl'),
            '--hdtf-features', '/root/kinetalk_unified_v2_20261010/hdtf_full/features',
            '--hdtf-source-root', '/root/kinetalk_unified_v2_20261010/hdtf_full']
    jobs = [('mixed_smoke', common + ['--output', str(ROOT / 'mixed_smoke'),
              '--epochs', '1', '--steps', '32', '--eval-limit', '32',
              '--hdtf-weight', str(1/3), '--mead-batch-size', '8', '--hdtf-batch-size', '4'] + hdtf)]
    for arm, mixed in [('mead_matched', False), ('mixed_matched', True)]:
        jobs += [(arm + '_train', common + ['--output', str(ROOT / arm), '--epochs', '1',
                  '--steps', '256', '--eval-limit', '128', '--mead-batch-size', '8' if mixed else '12',
                  '--hdtf-batch-size', '4', '--hdtf-weight', str(1/3) if mixed else '0'] + (hdtf if mixed else [])),
                 (arm + '_development', [PY, '-u', 'scripts/evaluate_v5_candidate.py',
                  '--binding', str(binding), '--checkpoint', str(ROOT / arm / 'final.pt'),
                  '--output', str(ROOT / (arm + '_eval')),
                  '--selection-json', str(ROOT / 'diagnostic_seed47/selection.json')]),
                 (arm + '_dynamics', [PY, '-u', 'scripts/evaluate_v4_dynamics.py',
                  '--binding', str(binding), '--checkpoint', str(ROOT / arm / 'final.pt'),
                  '--output', str(ROOT / (arm + '_dynamics')), '--limit', '128',
                  '--seeds', '47', '91', '137'])]
    write(ROOT / 'mixed_pilot_jobs.json', dict(jobs=jobs, code=str(CODE),
          checkpoint=str(CHECKPOINT), smoke_updates=32, matched_updates=256,
          batch_examples=12, matching='same initialization, optimizer updates and total examples; domain composition differs',
          limitations=['Continuation pilots, not three-contribution architectural ablations.',
                       'Different-length domain windows mean FLOPs are only approximately matched.',
                       'HDTF perceptual synchronization remains unverified.'],
          test_loaded=False, paper_result=False, long_run_queued=False))
    stage = 'preflight'
    try:
        for stage, command in jobs:
            write(STATE, dict(status='running', stage=stage, pid=os.getpid(), updated_unix=time.time()))
            with (ROOT / (stage + '.log')).open('x') as log:
                subprocess.run(command, cwd=CODE, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            if stage == 'mixed_smoke':
                smoke = json.loads((ROOT / 'mixed_smoke/complete.json').read_text())
                if smoke['updates'] != 32 or smoke['unique_hdtf_people'] < 2:
                    raise RuntimeError('Mixed smoke did not cover multiple independent people')
                if smoke['protocol']['hdtf_emotion_targets'] or smoke['protocol']['hdtf_neutral_targets']:
                    raise RuntimeError('Unknown HDTF targets entered supervision')
        write(ROOT / 'mixed_pilot_queue_complete.json', dict(status='complete', test_loaded=False,
              promotion=False, long_run_queued=False, pid=os.getpid()))
        write(STATE, dict(status='complete', stage=stage, updated_unix=time.time()))
    except Exception as error:
        write(STATE, dict(status='failed', stage=stage, error=repr(error), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
