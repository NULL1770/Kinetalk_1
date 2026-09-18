"""Preflight and detach the fixed-clock paired experiment on migrated instance."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_joint_prior_background as base


def main():
    residual = '--residual' in sys.argv
    if residual:
        sys.argv.remove('--residual')
    args = base.parse_args()
    if args._supervise:
        return base.supervise(args._supervise)
    name = ('clocked_residual_smoke' if args.smoke else 'clocked_residual30') if residual else (
        'clocked_prior_smoke' if args.smoke else 'clocked_prior_pair30')
    # Respect explicitly supplied output/log/launch choices, adapting old defaults.
    if '--output' not in sys.argv:
        args.output = args.experiment_root/name
    if '--log' not in sys.argv:
        args.log = args.output.with_name(args.output.name+'.run.log')
    if '--launch-dir' not in sys.argv:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        args.launch_dir = args.experiment_root/'launches'/(name+'_'+stamp)
    driver = args.code_root/('scripts/train_clocked_residual_prior.py' if residual else 'scripts/train_clocked_motion_prior.py')
    base.require_file(driver, 'Fixed-clock training driver')
    base.require_file(args.code_root/'docs/CLOCKED_MOTION_PRIOR_PROTOCOL_20260918.md', 'Fixed-clock protocol')
    if residual:
        base.require_file(args.code_root/'docs/CLOCKED_RESIDUAL_PROTOCOL_20260918.md', 'Residual protocol')
    command = [str(args.python), '-u', str(driver)]
    for key in ('audio','targets','native_root','native_manifest','delta_dir','audio_checkpoint','output'):
        command.extend(['--'+key.replace('_','-'), str(getattr(args,key))])
    command.extend(['--device', args.device])
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
