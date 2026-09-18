"""Preflight and detach the gated joint-motion experiment on the migrated host.

The driver owns creation of its fresh output directory. Launcher metadata and
logs deliberately live outside it, and completion requires a zero exit code
and the driver's terminal status, never just an existing PID file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


DEFAULT_ROOT = Path('/root/kinetalk_joint_20260918')
DEFAULT_NATIVE = Path('/root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf8'))
    except (OSError, ValueError):
        return None


def existing_ancestor(path):
    path = Path(path)
    while not path.exists():
        if path == path.parent:
            raise ValueError('No existing filesystem parent: ' + str(path))
        path = path.parent
    return path


def require_file(path, label):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'{label} must be a nonempty file: {path}')


def preflight(args, command):
    require_file(args.python, 'Python executable')
    require_file(args.code_root / 'scripts/train_joint_motion_prior.py', 'Training driver')
    require_file(args.code_root / 'docs/JOINT_MOTION_PRIOR_PROTOCOL_20260918.md', 'Protocol')
    for name in ('audio', 'targets', 'native_manifest', 'audio_checkpoint'):
        require_file(getattr(args, name), name)
    if not args.native_root.is_dir():
        raise ValueError('Missing native data directory: ' + str(args.native_root))
    for name in ('manifest.json', 'complete.json'):
        require_file(args.delta_dir / name, 'Complete audio delta ' + name)
    if args.output.exists():
        raise FileExistsError('Fresh experiment output required: ' + str(args.output))
    if args.log.exists():
        raise FileExistsError('Refusing to overwrite existing run log: ' + str(args.log))
    if (args.output in (args.launch_dir, args.log) or args.output in args.launch_dir.parents
            or args.output in args.log.parents):
        raise ValueError('Launch directory and log must be outside the driver output directory')
    space = {}
    for label, path in (('output', args.output), ('launch', args.launch_dir), ('log', args.log)):
        parent = existing_ancestor(path.parent)
        free = shutil.disk_usage(parent).free
        space[label] = {'filesystem_parent': str(parent), 'free_bytes': free}
        if free < args.min_free_gib * 1024**3:
            raise RuntimeError(f'Insufficient free space for {label}: {free / 1024**3:.2f} GiB; '
                               f'require {args.min_free_gib:.2f} GiB')
    if args.device != 'cuda' and not args.device.startswith('cuda:'):
        raise ValueError('This launcher requires an available CUDA device; use --device cuda or cuda:N')
    probe = (
        'import json,sys,numpy,torch; '
        'available=torch.cuda.is_available(); '
        "print(json.dumps({'cuda_available':available,'torch':torch.__version__,'python':sys.executable}),flush=True); "
        "sys.exit('CUDA unavailable: this instance is in no-GPU mode. Restart it with a GPU and rerun the launcher.') if not available else None; "
        'x=torch.zeros(1,device=sys.argv[1]); torch.cuda.synchronize(x.device); '
        "print(json.dumps({'device':str(x.device),'device_name':torch.cuda.get_device_name(x.device)}),flush=True)"
    )
    result = subprocess.run([str(args.python), '-c', probe, args.device], cwd=args.code_root,
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('CUDA/dependency preflight failed; no training was launched.\n'
                           + result.stdout.strip() + '\n' + result.stderr.strip())
    return {'checked_utc': utc_now(), 'space': space, 'cuda_probe': result.stdout.strip(),
            'command': command}


def supervise(plan_path):
    """Detached parent records the actual child exit and driver terminal status."""
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError('Invalid launch plan')
    record = Path(plan_path).with_name('launch.json')
    state = dict(plan, status='supervisor_started', supervisor_pid=os.getpid(), started_utc=utc_now())
    write_json(record, state)
    proc = None
    try:
        if Path(plan['output']).exists():
            raise FileExistsError('Output appeared before launch; refusing to reuse it')
        with Path(plan['log']).open('x', encoding='utf8') as log:
            proc = subprocess.Popen(plan['command'], cwd=plan['code_root'],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
            state.update(status='running', pid=proc.pid)
            Path(plan_path).with_name('pid').write_text(str(proc.pid) + '\n', encoding='ascii')
            write_json(record, state)
            code = proc.wait()
        driver = read_json(Path(plan['output']) / 'status.json')
        complete = code == 0 and isinstance(driver, dict) and driver.get('status') == 'complete'
        state.update(status='complete' if complete else 'failed', exit_code=code,
                     finished_utc=utc_now(), driver_status=driver)
        if code == 0 and not complete:
            state['error'] = 'Child exited zero without a complete driver status'
        write_json(record, state)
        return 0 if complete else 1
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
        if proc is not None:
            state['child_poll'] = proc.poll()
        write_json(record, state)
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--python', type=Path, default=Path('/root/miniconda3/bin/python'))
    parser.add_argument('--audio', type=Path, default=Path('/root/kinetalk_runs/audio_text_v1/data/audio.pt'))
    parser.add_argument('--targets', type=Path,
                        default=Path('/root/kinetalk_runs/audio_text_v1/data/intensity_targets/targets.pt'))
    parser.add_argument('--native-root', type=Path, default=DEFAULT_NATIVE)
    parser.add_argument('--native-manifest', type=Path)
    parser.add_argument('--delta-dir', type=Path)
    parser.add_argument('--audio-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--launch-dir', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs, disk and CUDA; do not launch')
    parser.add_argument('--min-free-gib', type=float, default=2.0)
    parser.add_argument('--startup-seconds', type=float, default=10.0)
    parser.add_argument('--_supervise', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._supervise:
        return args
    if args.min_free_gib <= 0 or not 1 <= args.startup_seconds <= 60:
        parser.error('--min-free-gib must be positive and --startup-seconds must be in [1, 60]')
    root = args.experiment_root.resolve()
    args.code_root = args.code_root or root / 'code'
    args.native_manifest = args.native_manifest or args.native_root / 'train.jsonl'
    args.delta_dir = args.delta_dir or root / 'full_native_audio_delta'
    args.audio_checkpoint = args.audio_checkpoint or root / 'run12/audio/final.pt'
    args.output = args.output or root / ('joint_prior_smoke' if args.smoke else 'joint_prior_formal30')
    args.log = args.log or args.output.with_name(args.output.name + '.run.log')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    args.launch_dir = args.launch_dir or root / 'launches' / (args.output.name + '_' + stamp)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    return args


def main():
    args = parse_args()
    if args._supervise:
        return supervise(args._supervise)
    command = [str(args.python), '-u', str(args.code_root / 'scripts/train_joint_motion_prior.py')]
    for name in ('audio', 'targets', 'native_root', 'native_manifest', 'delta_dir', 'audio_checkpoint', 'output'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    command.extend(['--device', args.device])
    if args.smoke:
        command.append('--smoke')
    checks = preflight(args, command)
    if args.check_only:
        print(json.dumps({'status': 'preflight_passed', **checks}, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.launch_dir.mkdir(parents=True, exist_ok=False)
    plan = {'command': command, 'code_root': str(args.code_root), 'output': str(args.output),
            'log': str(args.log), 'launch_dir': str(args.launch_dir), 'preflight': checks,
            'created_utc': utc_now(), 'smoke': args.smoke}
    plan_path = args.launch_dir / 'plan.json'
    write_json(plan_path, plan)
    record = args.launch_dir / 'launch.json'
    write_json(record, dict(plan, status='launching'))
    with (args.launch_dir / 'supervisor.log').open('x', encoding='utf8') as supervisor_log:
        supervisor = subprocess.Popen([str(args.python), '-u', str(Path(__file__).resolve()),
                                       '--_supervise', str(plan_path)], cwd=args.code_root,
                                      stdin=subprocess.DEVNULL, stdout=supervisor_log,
                                      stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    deadline = time.monotonic() + args.startup_seconds
    state = None
    while time.monotonic() < deadline:
        state = read_json(record)
        if state and state.get('status') in ('complete', 'failed'):
            break
        if supervisor.poll() is not None:
            break
        time.sleep(.25)
    state = read_json(record) or {}
    driver = read_json(args.output / 'status.json')
    if state.get('status') == 'complete':
        outcome = 'complete'
    elif state.get('status') == 'running' and supervisor.poll() is None and isinstance(driver, dict):
        # The live supervisor is waiting on the real child; the driver has also
        # written its own state. This is stronger than trusting a stale PID file.
        outcome = 'startup_verified'
    elif state.get('status') == 'running' and supervisor.poll() is None:
        outcome = 'spawned_not_yet_verified'
    else:
        outcome = 'failed'
    print(json.dumps({'status': outcome, 'launch_record': str(record), 'log': str(args.log),
                      'output': str(args.output), 'pid': state.get('pid'),
                      'supervisor_pid': supervisor.pid, 'driver_status': driver,
                      'error': state.get('error')}, indent=2))
    return 1 if outcome == 'failed' else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'Launch failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
