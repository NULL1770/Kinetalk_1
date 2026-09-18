"""Preflight and detach the fixed-clock paired experiment on migrated instance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_joint_prior_background as base


def main():
    modes = argparse.ArgumentParser(add_help=False)
    exclusive = modes.add_mutually_exclusive_group()
    exclusive.add_argument('--residual', action='store_true')
    exclusive.add_argument('--prosody', action='store_true')
    exclusive.add_argument('--activity', action='store_true')
    exclusive.add_argument('--controlled', action='store_true')
    modes.add_argument('--source-run', type=Path)
    mode, remaining = modes.parse_known_args()
    sys.argv[1:] = remaining
    residual, prosody, activity, controlled = mode.residual, mode.prosody, mode.activity, mode.controlled
    args = base.parse_args()
    if args._supervise:
        return base.supervise(args._supervise)
    name = ('controlled_prior_smoke' if args.smoke else 'controlled_prior') if controlled else (('activity_condition_smoke' if args.smoke else 'activity_condition30') if activity else (
        ('clocked_prosody_smoke' if args.smoke else 'clocked_prosody30') if prosody else (
        ('clocked_residual_smoke' if args.smoke else 'clocked_residual30') if residual else (
        'clocked_prior_smoke' if args.smoke else 'clocked_prior_pair30'))))
    # Respect explicitly supplied output/log/launch choices, adapting old defaults.
    if '--output' not in sys.argv:
        args.output = args.experiment_root/name
    if '--log' not in sys.argv:
        args.log = args.output.with_name(args.output.name+'.run.log')
    if '--launch-dir' not in sys.argv:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        args.launch_dir = args.experiment_root/'launches'/(name+'_'+stamp)
    driver = args.code_root/('scripts/run_controlled_prior.py' if controlled else ('scripts/train_activity_condition.py' if activity else (
        'scripts/train_prosody_clocked_residual.py' if prosody else (
        'scripts/train_clocked_residual_prior.py' if residual else 'scripts/train_clocked_motion_prior.py'))))
    base.require_file(driver, 'Fixed-clock training driver')
    base.require_file(args.code_root/'docs/CLOCKED_MOTION_PRIOR_PROTOCOL_20260918.md', 'Fixed-clock protocol')
    if residual or prosody:
        base.require_file(args.code_root/'docs/CLOCKED_RESIDUAL_PROTOCOL_20260918.md', 'Residual protocol')
    if prosody:
        base.require_file(args.code_root/'docs/CLOCKED_PROSODY_PROTOCOL_20260918.md', 'Prosody protocol')
    if activity:
        base.require_file(args.code_root/'docs/ACTIVITY_CONDITION_PROTOCOL_20260918.md', 'Activity protocol')
    if controlled:
        base.require_file(args.code_root/'docs/SUPERVISION_NATURAL_PRIOR_PROTOCOL_20260918.md', 'Controlled prior protocol')
    command = [str(args.python), '-u', str(driver)]
    for key in ('audio','targets','native_root','native_manifest','delta_dir','audio_checkpoint','output'):
        command.extend(['--'+key.replace('_','-'), str(getattr(args,key))])
    command.extend(['--device', args.device])
    if prosody or activity or controlled:
        source = mode.source_run or args.experiment_root/'clocked_residual30'
        base.require_file(source/'status.json', 'Completed frozen residual source')
        command.extend(['--source-run', str(source)])
    elif mode.source_run is not None:
        raise ValueError('--source-run is only valid with --prosody, --activity or --controlled')
    if args.smoke:
        command.append('--smoke')
    checks = base.preflight(args, command)
    if args.check_only:
        print(json.dumps({'status':'preflight_passed',**checks},indent=2)); return 0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.log.parent.mkdir(parents=True,exist_ok=True)
    args.launch_dir.mkdir(parents=True,exist_ok=False)
    plan={'command':command,'code_root':str(args.code_root),'output':str(args.output),
          'log':str(args.log),'launch_dir':str(args.launch_dir),'preflight':checks,
          'created_utc':base.utc_now(),'smoke':args.smoke}
    plan_path=args.launch_dir/'plan.json'; base.write_json(plan_path,plan)
    with (args.launch_dir/'supervisor.log').open('x',encoding='utf8') as log:
        proc=subprocess.Popen([str(args.python),str(Path(__file__).resolve()),'--_supervise',str(plan_path)],
            cwd=args.code_root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    print(json.dumps({'status':'launched','supervisor_pid':proc.pid,'launch':str(args.launch_dir/'launch.json'),
                      'output':str(args.output),'log':str(args.log)},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
