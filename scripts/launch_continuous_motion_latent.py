"""Detach one bounded latent-motion run and preserve its actual terminal state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MIN_FREE_BYTES = 1024 ** 3
RUNNER_MODULE = 'scripts.train_continuous_motion_latent'
LAUNCHER_MODULE = 'scripts.launch_continuous_motion_latent'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def thread_environment():
    return {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4'}


def _cmdline_matches(command, output):
    """Match argument tokens, never a PID or an output-path substring alone."""
    if not command:
        return False
    is_ours = any(token in (RUNNER_MODULE, LAUNCHER_MODULE)
                  or Path(token).name in ('launch_continuous_motion_latent.py',
                                         'train_continuous_motion_latent.py')
                  for token in command)
    if not is_ours:
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
        # Local Windows checks also compare command tokens; production uses /proc.
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError('Process command verification requires Linux /proc or psutil') from exc
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
    # Allocate once to catch a driver/runtime mismatch before detaching.
    torch.empty(1, device=selected)
    return {'device': device, 'cuda_checked': True, 'cuda_name': torch.cuda.get_device_name(index)}


def preflight(args):
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.code_root = args.code_root.resolve()
    if not args.dataset.is_file():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    if not args.code_root.is_dir():
        raise FileNotFoundError(f'Code root not found: {args.code_root}')
    for key in ('ae_steps', 'prior_steps', 'adapt_steps', 'batch_size'):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(f'{key} must be a positive integer')
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


def runner_command(args):
    command = [sys.executable, '-u', '-m', RUNNER_MODULE,
               '--dataset', str(args.dataset), '--output', str(args.output), '--device', args.device]
    for key in ('ae_steps', 'prior_steps', 'adapt_steps', 'batch_size'):
        command.extend(('--' + key.replace('_', '-'), str(getattr(args, key))))
    if args.resume:
        command.append('--resume')
    return command


def _file_fingerprint(path):
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def supervise(args, *, command=None):
    """Run synchronously inside the detached supervisor; command permits fake-child tests."""
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
               '--code-root', str(args.code_root), '--device', args.device]
    for key in ('ae_steps', 'prior_steps', 'adapt_steps', 'batch_size'):
        command.extend(('--' + key.replace('_', '-'), str(getattr(args, key))))
    if args.resume:
        command.append('--resume')
    # Write ownership before Popen and replace it with the returned process PID.
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
    result.add_argument('--prior-steps', type=int, default=8000)
    result.add_argument('--adapt-steps', type=int, default=6000)
    result.add_argument('--batch-size', type=int, default=24)
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
