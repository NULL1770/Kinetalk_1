"""Detach the sentence-validation audio-adapter pipeline and record its exit."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MIN_FREE_BYTES = 1024 ** 3
RUNNER_MODULE = 'scripts.train_prior_audio_adapter'
LAUNCHER_MODULE = 'scripts.launch_prior_audio_adapter'
SCALAR_FLAGS = ('ae_steps', 'prior_steps', 'adapter_steps', 'batch_size', 'valid_sentences', 'max_delta')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def thread_environment():
    return {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4'}


def _cmdline_matches(command, output):
    """Check both this experiment's executable tokens and the exact output."""
    if not command:
        return False
    ours = any(token in (RUNNER_MODULE, LAUNCHER_MODULE)
               or Path(token).name in ('launch_prior_audio_adapter.py', 'train_prior_audio_adapter.py')
               for token in command)
    if not ours:
        return False
    wanted = str(Path(output).resolve())
    for index, token in enumerate(command):
        candidate = None
        if token == '--output' and index + 1 < len(command):
            candidate = command[index + 1]
        elif token.startswith('--output='):
            candidate = token[len('--output='):]
        if candidate is not None and str(Path(candidate).resolve()) == wanted:
            return True
    return False


def process_matches_output(pid, output, proc_root=None):
    if type(pid) is not int or pid <= 0:
        return False
    if proc_root is not None or sys.platform.startswith('linux'):
        path = Path('/proc' if proc_root is None else proc_root) / str(pid) / 'cmdline'
        try:
            command = [part.decode('utf8', errors='replace') for part in path.read_bytes().split(b'\0') if part]
        except (FileNotFoundError, ProcessLookupError):
            return False
        except PermissionError as exc:
            raise RuntimeError(f'Cannot verify existing process {pid}; refusing duplicate launch') from exc
    else:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError('Process verification requires Linux /proc or psutil') from exc
        try:
            command = psutil.Process(pid).cmdline()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except psutil.AccessDenied as exc:
            raise RuntimeError(f'Cannot verify existing process {pid}; refusing duplicate launch') from exc
    return _cmdline_matches(command, output)


def existing_processes(output):
    candidates = set()
    for name in ('launch.json', 'pipeline_status.json'):
        path = Path(output) / name
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text(encoding='utf8'))
        except (ValueError, OSError) as exc:
            raise RuntimeError(f'Cannot read launch ownership record: {path}') from exc
        if not isinstance(record, dict):
            raise RuntimeError(f'Invalid launch ownership record: {path}')
        for key in ('pid', 'supervisor_pid', 'child_pid'):
            pid = record.get(key)
            if type(pid) is int and pid > 0:
                candidates.add(pid)
    return sorted(pid for pid in candidates if process_matches_output(pid, output))


def require_cuda(device):
    if not device.startswith('cuda'):
        return {'device': device, 'cuda_checked': False}
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no background process was started')
    selected = torch.device(device)
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f'CUDA device is unavailable: {device}')
    torch.empty(1, device=selected)
    return {'device': device, 'cuda_checked': True, 'cuda_name': torch.cuda.get_device_name(index)}


def validate_budgets(args):
    for key in ('ae_steps', 'prior_steps', 'adapter_steps', 'batch_size', 'valid_sentences'):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (isinstance(args.max_delta, bool) or not isinstance(args.max_delta, (int, float))
            or not math.isfinite(args.max_delta) or args.max_delta <= 0):
        raise ValueError('max_delta must be a finite positive scalar')
    if (not isinstance(args.milestones, (list, tuple)) or not args.milestones
            or any(type(step) is not int or step < 1 or step > args.adapter_steps for step in args.milestones)
            or list(args.milestones) != sorted(set(args.milestones))
            or args.milestones[-1] != args.adapter_steps):
        raise ValueError('milestones must be increasing unique positive steps ending at adapter_steps')


def preflight(args):
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.code_root = args.code_root.resolve()
    if not args.dataset.is_file():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    if not args.code_root.is_dir():
        raise FileNotFoundError(f'Code root not found: {args.code_root}')
    validate_budgets(args)
    if args.output.exists():
        if not args.output.is_dir():
            raise ValueError('Output exists and is not a directory')
        if not args.resume:
            raise FileExistsError('Output already exists; use --resume explicitly')
        running = existing_processes(args.output)
        if running:
            raise RuntimeError(f'Existing run is still alive for this output: {running}')
    elif args.resume:
        raise FileNotFoundError('--resume requires an existing output directory')
    disk_path = args.output
    while not disk_path.exists():
        disk_path = disk_path.parent
    free = shutil.disk_usage(disk_path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f'At least 1 GiB free disk is required; available bytes: {free}')
    return {**require_cuda(args.device), 'free_bytes': free, 'dataset': str(args.dataset),
            'output': str(args.output), 'code_root': str(args.code_root), 'resume': args.resume}


def _training_flags(args):
    flags = []
    for key in SCALAR_FLAGS:
        flags.extend(('--' + key.replace('_', '-'), str(getattr(args, key))))
    flags.extend(('--milestones', *(str(step) for step in args.milestones)))
    if args.resume:
        flags.append('--resume')
    return flags


def runner_command(args):
    return [sys.executable, '-u', '-m', RUNNER_MODULE,
            '--dataset', str(args.dataset), '--output', str(args.output), '--device', args.device,
            *_training_flags(args)]


def _file_fingerprint(path):
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def supervise(args, *, command=None):
    """Wait for the real child; only a fresh terminal status plus exit 0 completes."""
    validate_budgets(args)
    args.output.mkdir(parents=True, exist_ok=True)
    command = runner_command(args) if command is None else command
    start = time.time()
    status_path = args.output / 'pipeline_status.json'
    runner_status = args.output / 'status.json'
    old_fingerprint = _file_fingerprint(runner_status)
    state = {'state': 'running', 'pid': os.getpid(), 'supervisor_pid': os.getpid(), 'child_pid': None,
             'exit_code': None, 'start': start, 'end': None, 'command': command,
             'output': str(args.output), 'resume': args.resume}
    write_json(status_path, state)
    try:
        with (args.output / 'training.log').open('a' if args.resume else 'w', encoding='utf8') as log:
            child = subprocess.Popen(command, cwd=args.code_root, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=thread_environment())
            state['child_pid'] = child.pid
            write_json(status_path, state)
            code = child.wait()
        state['exit_code'] = code
        payload = None
        if runner_status.exists() and _file_fingerprint(runner_status) != old_fingerprint:
            try:
                payload = json.loads(runner_status.read_text(encoding='utf8'))
            except (ValueError, OSError) as exc:
                state['reason'] = 'Runner status could not be read: ' + str(exc)
        else:
            state['reason'] = 'Runner did not write a fresh status.json'
        runner_state = payload.get('state') if isinstance(payload, dict) else None
        state['runner_state'] = runner_state
        if code == 0 and runner_state in ('complete', 'stopped'):
            state['state'] = runner_state
            if runner_state == 'stopped':
                state['reason'] = payload.get('reason', 'Runner stopped at a quality gate; not a successful experiment')
        else:
            state['state'] = 'failed'
            state.setdefault('reason', f'Child exit {code}, runner state {runner_state!r}')
    except Exception as exc:
        state['state'] = 'failed'
        state['reason'] = f'{type(exc).__name__}: {exc}'
        state['exit_code'] = state['exit_code'] if state['exit_code'] is not None else 1
    state['end'] = time.time()
    state['elapsed_seconds'] = state['end'] - start
    write_json(status_path, state)
    return 0 if state['state'] in ('complete', 'stopped') else (state['exit_code'] or 1)


def launch(args):
    checks = preflight(args)
    if args.check_only:
        return {**checks, 'state': 'checked', 'started': False}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    command = [sys.executable, '-u', '-m', LAUNCHER_MODULE, '--supervise',
               '--dataset', str(args.dataset), '--output', str(args.output),
               '--code-root', str(args.code_root), '--device', args.device, *_training_flags(args)]
    record = {'state': 'launching', 'pid': None, 'child_pid': None, 'command': command,
              'start': time.time(), 'end': None, 'exit_code': None, 'checks': checks,
              'completion_not_yet_verified': True}
    write_json(args.output / 'launch.json', record)
    try:
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        with (args.output / 'supervisor.log').open('a' if args.resume else 'w', encoding='utf8') as log:
            child = subprocess.Popen(command, cwd=args.code_root, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True,
                                     env=thread_environment(), **kwargs)
        record.update(state='launched', pid=child.pid)
    except Exception as exc:
        record.update(state='failed', end=time.time(), exit_code=1, reason=f'{type(exc).__name__}: {exc}')
        write_json(args.output / 'launch.json', record)
        raise
    write_json(args.output / 'launch.json', record)
    return {'state': 'launched', 'pid': child.pid, 'output': str(args.output), 'started': True}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--code-root', type=Path, default=Path.cwd())
    result.add_argument('--device', default='cuda')
    result.add_argument('--ae-steps', type=int, default=4000)
    result.add_argument('--prior-steps', type=int, default=14000)
    result.add_argument('--adapter-steps', type=int, default=3000)
    result.add_argument('--milestones', type=int, nargs='+', default=[1000, 3000])
    result.add_argument('--batch-size', type=int, default=24)
    result.add_argument('--valid-sentences', type=int, default=5)
    result.add_argument('--max-delta', type=float, default=.35)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--check-only', action='store_true')
    result.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.supervise:
        if args.check_only:
            raise ValueError('--supervise and --check-only cannot be combined')
        return supervise(args)
    print(json.dumps(launch(args), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
