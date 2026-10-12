"""Isolate actual decoder-state supervision after the acoustic queue completes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/root/kinetalk_v5_20261011')
CODE = ROOT / 'code_decoder_state'
PY = '/root/miniconda3/bin/python'
SOURCE = ROOT / 'mead_matched/final.pt'
EXPECTED = 'c9189be8050a1b536054bfc11df265d0c0898488a21791638228f9682b937c83'
ARMS = ('state_raw_direct', 'state_decoder_direct', 'state_decoder_privileged')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    with (ROOT / 'decoder_state_pilot.lock').open('x') as handle:
        handle.write(str(os.getpid()))
    status = ROOT / 'decoder_state_pilot_state.json'
    stage = 'waiting_acoustic_completion'
    child = None
    try:
        deadline = time.monotonic() + 4*3600
        while not ((ROOT / 'acoustic_geometry_complete.json').is_file() or
                   (ROOT / 'acoustic_geometry_pilot_state.json').is_file() and
                   json.loads((ROOT / 'acoustic_geometry_pilot_state.json').read_text()).get('status') == 'failed'):
            prior = json.loads((ROOT / 'acoustic_geometry_pilot_state.json').read_text())
            if prior['status'] == 'failed':
                raise RuntimeError('Acoustic queue failed: '+prior.get('error', ''))
            try:
                os.kill(prior['pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('Acoustic queue exited without completion or failure receipt')
            if time.monotonic() > deadline:
                raise TimeoutError('Prior queue wait exceeded four hours')
            write(status, dict(status='waiting', stage=stage, pid=os.getpid(), updated_unix=time.time()))
            time.sleep(15)
        if sha(SOURCE) != EXPECTED:
            raise ValueError('Matched source differs')
        acoustic_degraded = not (ROOT / 'acoustic_geometry_complete.json').is_file()
        if acoustic_degraded:
            prior = json.loads((ROOT / 'acoustic_geometry_pilot_state.json').read_text())
            if prior.get('stage') != 'geometry_acoustic_mean_train':
                raise RuntimeError('Acoustic queue failed before the optional mean arm')
            required = (
                ROOT/'geometry_coarse/complete.json', ROOT/'geometry_coarse_full_eval/complete.json',
                ROOT/'geometry_coarse_dynamics/complete.json', ROOT/'geometry_coarse_controls/complete.json',
                ROOT/'geometry_coarse_geometry/complete.json', ROOT/'geometry_acoustic/complete.json',
                ROOT/'geometry_acoustic_full_eval/complete.json', ROOT/'geometry_acoustic_dynamics/complete.json',
                ROOT/'geometry_acoustic_controls/complete.json', ROOT/'geometry_acoustic_geometry/complete.json')
            if any(not path.is_file() for path in required):
                raise RuntimeError('Missing complete coarse/native acoustic arm')
        files = json.loads((ROOT / 'decoder_state_source_hashes.json').read_text())
        if any(sha(CODE/name) != digest for name, digest in files.items()):
            raise ValueError('Immutable decoder-state snapshot differs')
        common = [PY, '-u', 'scripts/train_v5_mixed.py', '--binding', str(ROOT/'binding.json'),
            '--checkpoint', str(SOURCE), '--seed', '47', '--epochs', '1', '--hdtf-weight', '0',
            '--mead-batch-size', '12', '--rollout-mean-weight', '.5', '--posture-weight', '1']
        guide = ['--state-supervision-space', 'decoder', '--teacher-guidance-mode', 'posture',
                 '--teacher-posture-weight', '.5']
        jobs = [('state_decoder_smoke', common+guide+['--steps', '16', '--eval-limit', '32',
                 '--output', str(ROOT/'state_decoder_smoke')])]
        for arm, space, mode, teacher_weight in zip(ARMS, ('raw', 'decoder', 'decoder'),
                ('none', 'none', 'posture'), ('0', '0', '.5')):
            extra = ['--state-supervision-space', space, '--teacher-guidance-mode', mode,
                     '--teacher-posture-weight', teacher_weight]
            jobs.append((arm+'_train', common+extra+['--steps', '256', '--eval-limit', '128',
                         '--output', str(ROOT/arm)]))
            checkpoint = ROOT/arm/'final.pt'
            for suffix, script, args in (
                ('subset_eval', 'evaluate_v5_candidate.py', ['--selection-json', str(ROOT/'diagnostic_seed47/selection.json')]),
                ('state', 'audit_v5_state_coordinates.py', ['--limit', '128', '--device', 'cuda']),
                ('full_eval', 'evaluate_v5_candidate.py', []),
                ('dynamics', 'evaluate_v4_dynamics.py', ['--limit', '128', '--seeds', '47', '91', '137']),
                ('controls', 'audit_v5_controls.py', ['--limit', '32']),
                ('geometry', 'audit_v5_geometry.py', ['--limit', '128', '--device', 'cpu'])):
                jobs.append((arm+'_'+suffix, [PY, '-u', 'scripts/'+script, '--binding', str(ROOT/'binding.json'),
                    '--checkpoint', str(checkpoint), '--output', str(ROOT/(arm+'_'+suffix))]+args))
        jobs.append(('state_target_style', [PY, '-u', 'scripts/audit_v5_style_targets.py',
            '--binding', str(ROOT/'binding.json'), '--output', str(ROOT/'state_target_style'),
            '--device', 'cpu', '--limit', '64']+
            [value for arm in ARMS for value in ('--checkpoint', str(ROOT/arm/'final.pt'))]))
        write(ROOT/'decoder_state_jobs.json', dict(jobs=jobs, source_checkpoint_sha256=EXPECTED,
            matched_updates=256, training_seeds=[47], sampling_seeds=[47,91,137], new_parameters=0,
            test_loaded=False, promotion=False, paper_result=False))
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        baseline = json.loads((ROOT/'posture_control/complete.json').read_text())
        receipts = []
        for stage, command in jobs:
            start = time.monotonic()
            with (ROOT/(stage+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    write(status, dict(status='running', stage=stage, pid=os.getpid(), child_pid=child.pid,
                        stage_elapsed_seconds=time.monotonic()-start, updated_unix=time.time()))
                    if time.monotonic()-start > 45*60:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                        raise TimeoutError('Owned stage exceeded declared45minute hard timeout: '+stage)
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError('Stage failed: '+stage+' exit '+str(child.returncode))
                child = None
            if stage.endswith('_train') or stage == 'state_decoder_smoke':
                arm = stage.removesuffix('_train')
                result = json.loads((ROOT/arm/'complete.json').read_text())
                protocol = result['protocol']
                if protocol['source_checkpoint_sha256'] != EXPECTED or protocol['test_loaded'] or protocol['hdtf_windows']:
                    raise ValueError('Unexpected source/data scope')
                if result['frozen_b0_sha256'] != baseline['frozen_b0_sha256']:
                    raise ValueError('Frozen B0 differs')
                if protocol['posture_extension_added'] or protocol['geometry_extension_added']:
                    raise ValueError('This pilot must add no parameters')
                if stage == 'state_decoder_smoke' and result['updates'] != 16:
                    raise ValueError('Smoke budget differs')
                if stage.endswith('_train'):
                    if result['updates'] != 256:
                        raise ValueError('Training budget differs')
                    matching = dict(initial=protocol['common_initial_parameters_sha256'],
                        references=protocol['training_references_sha256'], fit=protocol['fit_indices_sha256'],
                        sequence=result['mead_training_sequence_sha256'], B0=result['frozen_b0_sha256'])
                    if any(matching != prior for prior in receipts):
                        raise ValueError('Matched arms differ in initial/data/reference/B0')
                    if any(protocol[key] != baseline['protocol'][key] for key in
                           ('training_references_sha256', 'fit_indices_sha256')):
                        raise ValueError('Prior continuation scope differs')
                    if matching['sequence'] != baseline['mead_training_sequence_sha256']:
                        raise ValueError('Prior continuation order differs')
                    receipts.append(matching)
            if sha(SOURCE) != EXPECTED:
                raise ValueError('Source checkpoint mutated')
        write(ROOT/'decoder_state_complete.json', dict(status='complete', receipts=receipts,
            acoustic_partial=acoustic_degraded, failed_optional_arm='geometry_acoustic_mean' if acoustic_degraded else None,
            new_parameters=0, test_loaded=False, promotion=False, paper_result=False))
        write(status, dict(status='complete', stage=stage, pid=os.getpid(), updated_unix=time.time()))
    except Exception as error:
        write(status, dict(status='failed', stage=stage, error=repr(error), pid=os.getpid(),
            child_pid=child.pid if child else None, updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
