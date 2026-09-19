"""Run independent event and mouth experiments sequentially, without promotion.

Uses the current Python interpreter. Paths relative to --root are resolved
against that experiment root; --output must be a new directory under it.
Scientific gate failure is reported separately from a crashed/incomplete stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = 'event_mouth_sequence_v1'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def read_status(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf8'))
    except (OSError, ValueError):
        return None


def resolved(root, path):
    return (path if path.is_absolute() else root / path).resolve()


def stages(args):
    event = [sys.executable, '-u', '-m', 'scripts.train_event_schedule_pipeline',
             '--dataset', str(args.dataset), '--source-run', str(args.source_run),
             '--output', str(args.output / 'event'), '--device', args.device,
             '--receiver-steps', str(args.event_receiver_steps),
             '--predictor-steps', str(args.event_predictor_steps)]
    mouth = [sys.executable, '-u', '-m', 'scripts.train_mouth_residual_candidate',
             '--dataset', str(args.dataset), '--reference-protocol', str(args.source_run / 'protocol.json'),
             '--output', str(args.output / 'mouth'), '--device', args.device,
             '--steps', str(args.mouth_steps), '--seeds', '42', '123', '2026']
    if args.smoke:
        event.append('--smoke')
        mouth = mouth[:-3] + ['42']
    return [('event', event, 'complete'), ('mouth', mouth, 'completed')]


def validate(args):
    if min(args.event_receiver_steps, args.event_predictor_steps, args.mouth_steps) < 1:
        raise ValueError('Every stage budget must be positive')
    args.root = args.root.resolve()
    if not args.root.is_dir():
        raise NotADirectoryError(args.root)
    for name in ('code_root', 'dataset', 'source_run', 'output'):
        setattr(args, name, resolved(args.root, getattr(args, name)))
    if args.output == args.root or not args.output.is_relative_to(args.root):
        raise ValueError('--output must be a fresh child directory under --root')
    if args.output.exists():
        raise FileExistsError('Existing output is never reused: ' + str(args.output))
    for path in (args.dataset, args.source_run / 'protocol.json',
                 args.code_root / 'scripts/train_event_schedule_pipeline.py',
                 args.code_root / 'scripts/train_mouth_residual_candidate.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = json.loads((args.source_run / 'protocol.json').read_text(encoding='utf8'))
    if reference.get('schema') != 'bounded_audio_experiment_v1':
        raise ValueError('--source-run must bind bounded_audio_experiment_v1 protocol')


def execute_stage(name, command, expected_state, args, shared):
    directory = args.output / 'supervisor'
    path = directory / (name + '_status.json')
    log_path = directory / (name + '.log')
    child_status = args.output / name / 'status.json'
    started_wall, started_clock = time.time(), time.monotonic()
    record = {'name': name, 'state': 'starting', 'command': command, 'cwd': str(args.code_root),
              'started': started_wall, 'pid': None, 'exit_code': None, 'log': str(log_path),
              'child_status_path': str(child_status), 'default_replaced': False}
    def publish():
        record['updated'] = time.time()
        record['duration_seconds'] = time.monotonic() - started_clock
        shared['stages'][name] = record.copy()
        shared['active_stage'] = name
        shared['updated'] = record['updated']
        atomic_json(path, record)
        atomic_json(args.output / 'status.json', shared)
    publish()
    process = None
    try:
        environment = os.environ.copy()
        environment['PYTHONUNBUFFERED'] = '1'
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, cwd=args.code_root, stdout=log,
                                       stderr=subprocess.STDOUT, env=environment)
            record.update(state='running', pid=process.pid)
            publish()
            while process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                record['child_status'] = read_status(child_status)
                publish()
            record['exit_code'] = process.returncode
        record['child_status'] = read_status(child_status)
        child_state = (record['child_status'] or {}).get('state')
        record['completed_child_state'] = child_state
        record['state'] = 'completed' if process.returncode == 0 and child_state == expected_state else 'failed'
        if record['state'] == 'failed':
            record['failure_reason'] = ('process_nonzero_exit' if process.returncode != 0
                                        else 'missing_or_incomplete_child_status')
        if name == 'event':
            record['scientific_decision'] = read_status(args.output / name / 'decision.json')
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record.update(state='interrupted', exit_code=None if process is None else process.returncode,
                      failure_reason='supervisor_interrupted')
        publish()
        raise
    except Exception as exc:
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}',
                      failure_reason='supervisor_stage_exception')
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record['exit_code'] = None if process is None else process.returncode
    publish()
    return record


def run(args):
    validate(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'supervisor').mkdir()
    plan = stages(args)
    created = time.time()
    shared = {'schema': SCHEMA, 'state': 'running', 'supervisor_pid': os.getpid(),
              'started': created, 'updated': created, 'active_stage': None, 'stages': {},
              'default_replaced': False, 'smoke': args.smoke,
              'training_completed_is_not_scientific_success': True}
    atomic_json(args.output / 'protocol.json', {
        'schema': SCHEMA, 'root': str(args.root), 'code_root': str(args.code_root),
        'dataset': str(args.dataset), 'source_run': str(args.source_run),
        'python': sys.executable, 'smoke': args.smoke, 'device': args.device,
        'commands': {name: command for name, command, _ in plan},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy': 'Run event then independent mouth. Event failure does not skip mouth. No retries, resume, overwrite, or default promotion.',
        'gate_policy': 'A completed scientific experiment with rejected gates is not a process failure. Scientific decision is copied into stage state.',
        'incomplete_policy': 'Exit 0 without the expected complete/completed child status counts as failure.',
        'smoke_policy': 'Smoke is separately labeled and never a replacement for formal budgets/seeds.'})
    atomic_json(args.output / 'status.json', shared)
    try:
        for name, command, expected_state in plan:
            execute_stage(name, command, expected_state, args, shared)
    except KeyboardInterrupt:
        shared.update(state='interrupted', active_stage=None, updated=time.time(),
                      duration_seconds=time.time() - created, exit_code=130)
        atomic_json(args.output / 'status.json', shared)
        return 130
    failed = [name for name, row in shared['stages'].items() if row['state'] != 'completed']
    code = 1 if failed else 0
    shared.update(state='failed' if failed else 'completed', active_stage=None,
                  updated=time.time(), duration_seconds=time.time() - created,
                  failed_stages=failed, exit_code=code)
    atomic_json(args.output / 'status.json', shared)
    atomic_json(args.output / 'summary.json', shared)
    return code


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'code-root', 'dataset', 'source-run', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--event-receiver-steps', type=int, default=2000)
    p.add_argument('--event-predictor-steps', type=int, default=1500)
    p.add_argument('--mouth-steps', type=int, default=1500)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true',
                   help='Reduced validation/seeds; set all three step flags explicitly for a short smoke.')
    return p


if __name__ == '__main__':
    try:
        sys.exit(run(parser().parse_args()))
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
