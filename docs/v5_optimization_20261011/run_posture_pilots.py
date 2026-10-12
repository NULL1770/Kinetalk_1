"""Detached matched posture/teacher pilot; wait for existing route scoring."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/root/kinetalk_v5_20261011')
CODE = ROOT / 'code_posture_pilot'
STATE = ROOT / 'posture_pilot_state.json'
PY = '/root/miniconda3/bin/python'
CHECKPOINT = ROOT / 'mead_matched/final.pt'
EXPECTED_SHA = 'c9189be8050a1b536054bfc11df265d0c0898488a21791638228f9682b937c83'
ARMS = ('posture_control', 'posture_direct', 'posture_privileged')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (ROOT / 'posture_pilot.lock').open('x') as handle:
        handle.write(str(os.getpid()))
    stage = 'waiting_for_route_scoring'
    child = None
    try:
        deadline = time.monotonic() + 4*3600
        while not (ROOT / 'route_followup_complete.json').exists():
            prior = json.loads((ROOT / 'route_followup_state.json').read_text())
            if prior['status'] == 'failed':
                raise RuntimeError('Prior route scoring failed: ' + prior.get('error', 'inspect log'))
            if prior['status'] != 'complete':
                try:
                    os.kill(prior['pid'], 0)
                except ProcessLookupError:
                    raise RuntimeError('Prior route scoring exited without completion')
            if time.monotonic() > deadline:
                raise TimeoutError('Prior queue wait exceeded four hours')
            write(STATE, dict(status='waiting', stage=stage, pid=os.getpid(), updated_unix=time.time()))
            time.sleep(30)
        if sha(CHECKPOINT) != EXPECTED_SHA:
            raise ValueError('Immutable matched MEAD source checkpoint differs')
        sources = json.loads((ROOT / 'posture_pilot_source_hashes.json').read_text())
        if any(sha(CODE / name) != digest for name, digest in sources.items()):
            raise ValueError('Posture pilot source snapshot differs from launch')
        binding = ROOT / 'binding.json'
        common = [PY, '-u', 'scripts/train_v5_mixed.py', '--binding', str(binding),
                  '--checkpoint', str(CHECKPOINT), '--seed', '47', '--epochs', '1',
                  '--hdtf-weight', '0', '--mead-batch-size', '12', '--rollout-mean-weight', '.5']
        posture = ['--posture-adapter', '--posture-weight', '1']
        privileged = posture + ['--teacher-guidance-mode', 'posture', '--teacher-posture-weight', '.5']
        jobs = [('posture_smoke', common + privileged + ['--output', str(ROOT / 'posture_smoke'),
                 '--steps', '16', '--eval-limit', '32'])]
        for arm, extra in [(ARMS[0], ['--teacher-guidance-mode', 'frame']),
                           (ARMS[1], posture + ['--teacher-guidance-mode', 'none']),
                           (ARMS[2], privileged)]:
            jobs.append((arm + '_train', common + extra + ['--output', str(ROOT / arm),
                         '--steps', '256', '--eval-limit', '128']))
            checkpoint = ROOT / arm / 'final.pt'
            for suffix, script, more in [
                ('subset_eval', 'evaluate_v5_candidate.py', ['--selection-json', str(ROOT / 'diagnostic_seed47/selection.json')]),
                ('state', 'audit_v5_state_coordinates.py', ['--limit', '128', '--device', 'cuda']),
                ('full_eval', 'evaluate_v5_candidate.py', []),
                ('dynamics', 'evaluate_v4_dynamics.py', ['--limit', '128', '--seeds', '47', '91', '137']),
                ('controls', 'audit_v5_controls.py', ['--limit', '32'])]:
                jobs.append((arm + '_' + suffix, [PY, '-u', 'scripts/' + script,
                    '--binding', str(binding), '--checkpoint', str(checkpoint),
                    '--output', str(ROOT / (arm + '_' + suffix))] + more))
        write(ROOT / 'posture_pilot_jobs.json', dict(jobs=jobs, source_checkpoint_sha256=EXPECTED_SHA,
            source_snapshot=str(CODE), matched_updates=256, seed=47, examples_per_update=12,
            hdtf_in_pilot=False, prior_hdtf_route_preserved=True, test_loaded=False,
            matching='identical source/common weights, TRAIN order/references, optimizer steps and rollout seeds',
            teacher_isolation='direct and privileged arms share adapter/posture objective/teacher anchor; only pooled guidance differs',
            limitations=['Control versus direct jointly changes head, posture objective and legacy guidance.',
                         'One continuation seed; three sampling seeds are not independent training seeds.',
                         'Pooled head and distillation are established tools; novelty not demonstrated.'],
            promotion=False, paper_result=False))
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        matched_receipts = []
        for stage, command in jobs:
            started = time.monotonic()
            with (ROOT / (stage + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    write(STATE, dict(status='running', stage=stage, pid=os.getpid(), child_pid=child.pid,
                        stage_elapsed_seconds=time.monotonic()-started, updated_unix=time.time()))
                    if time.monotonic()-started > 45*60:
                        child.terminate()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        raise TimeoutError('Owned stage exceeded declared 45-minute hard timeout: ' + stage)
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError('Stage failed with exit code ' + str(child.returncode) + ': ' + stage)
                child = None
            if stage == 'posture_smoke' or stage.endswith('_train'):
                directory = 'posture_smoke' if stage == 'posture_smoke' else stage.removesuffix('_train')
                complete = json.loads((ROOT / directory / 'complete.json').read_text())
                if complete['protocol']['source_checkpoint_sha256'] != EXPECTED_SHA:
                    raise ValueError('Training source changed')
                if directory != 'posture_control' and not complete['posture_head_gradient_verified']:
                    raise RuntimeError('New head gradient gate failed')
                if complete['protocol']['test_loaded'] or complete['protocol']['hdtf_windows']:
                    raise RuntimeError('Unexpected data in isolated posture pilot')
                if stage == 'posture_smoke' and complete['updates'] != 16:
                    raise RuntimeError('Smoke budget differs')
                if stage.endswith('_train'):
                    matched_receipts.append({
                        'updates': complete['updates'],
                        'b0': complete['frozen_b0_sha256'],
                        'references': complete['protocol']['training_references_sha256'],
                        'fit': complete['protocol']['fit_indices_sha256'],
                        'train_sequence': complete['mead_training_sequence_sha256']})
                    if complete['updates'] != 256 or any(row != matched_receipts[0] for row in matched_receipts):
                        raise ValueError('Matched posture arms differ in budget/data/reference/B0 receipts')
            if sha(CHECKPOINT) != EXPECTED_SHA:
                raise ValueError('Pilot modified original checkpoint')
        write(ROOT / 'posture_pilot_complete.json', dict(status='complete', pid=os.getpid(),
            test_loaded=False, promotion=False, paper_result=False, arms=ARMS,
            matched_data_reference_b0_receipts=matched_receipts))
        write(STATE, dict(status='complete', stage=stage, updated_unix=time.time()))
    except Exception as error:
        write(STATE, dict(status='failed', stage=stage, error=repr(error), updated_unix=time.time(),
                         child_pid=child.pid if child is not None else None))
        raise


if __name__ == '__main__':
    main()
