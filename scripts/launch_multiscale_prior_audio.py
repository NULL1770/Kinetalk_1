"""Detached supervised launch of multiscale pilot and gated continuation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'source-run', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--code-root', type=Path, default=Path.cwd())
    p.add_argument('--pilot-steps', type=int, default=1000)
    p.add_argument('--final-steps', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--supervise', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve(); args.code_root = args.code_root.resolve()
    base = [sys.executable, '-u', '-m']
    flags = ['--dataset', str(args.dataset.resolve()), '--source-run', str(args.source_run.resolve()),
             '--output', str(args.output), '--pilot-steps', str(args.pilot_steps),
             '--final-steps', str(args.final_steps), '--batch-size', str(args.batch_size)]
    if not args.supervise:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        if not (0 < args.pilot_steps <= args.final_steps and args.batch_size > 0): raise ValueError('Invalid budgets')
        if args.output.exists(): raise FileExistsError('Fresh output required; refusing duplicate launch')
        if not args.dataset.is_file(): raise FileNotFoundError(args.dataset)
        for name in ('protocol.json','ae_final.pt','prior_final.pt','fit_stats.pt'):
            if not (args.source_run/name).is_file(): raise FileNotFoundError(args.source_run/name)
        if shutil.disk_usage(args.output.parent).free < 1024**3: raise RuntimeError('Less than 1 GiB free')
        args.output.mkdir()
        command = base+['scripts.launch_multiscale_prior_audio', '--supervise', '--code-root', str(args.code_root)]+flags
        with (args.output/'supervisor.log').open('w') as log:
            env = {**os.environ, 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4'}
            process = subprocess.Popen(command, cwd=args.code_root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env,
                **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}))
        info = {'state':'launched', 'supervisor_pid':process.pid, 'command':command, 'time':time.time()}
        write(args.output/'launch.json',info); print(json.dumps(info)); return 0
    command = base+['scripts.train_multiscale_prior_audio']+flags
    # The launcher owns an otherwise empty output; the runner accepts it only
    # via resume and still writes/binds a new immutable protocol itself.
    command += ['--resume']
    state = {'state':'running', 'supervisor_pid':os.getpid(), 'start':time.time(), 'command':command}
    try:
        with (args.output/'training.log').open('w') as log:
            process = subprocess.Popen(command, cwd=args.code_root, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid; write(args.output/'pipeline_status.json',state)
            code = process.wait()
        status = json.loads((args.output/'status.json').read_text())
        state.update(state='complete' if code == 0 and status['state'] == 'complete' else 'failed',
                     exit_code=code, runner_status=status)
    except Exception as exc:
        state.update(state='failed',error=str(exc),exit_code=1)
    state['end'] = time.time(); write(args.output/'pipeline_status.json',state)
    return 0 if state['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
