"""Detached fixed two-arm native-context training supervisor."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delta-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() != args.resume:
        raise ValueError('Fresh output required unless explicitly resuming')
    root = Path('/root/kinetalk_full_staged_20260917')
    code = Path(__file__).resolve().parents[1]
    native = '/root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask'
    command = [sys.executable, '-u', str(code/'scripts/train_native_context.py'),
        '--source-run', '/root/kinetalk_runs/teacher_schedule_v1/scaled_centered_v1/uniform',
        '--audio', '/root/kinetalk_runs/audio_text_v1/data/audio.pt',
        '--targets', '/root/kinetalk_runs/audio_text_v1/data/intensity_targets/targets.pt',
        '--enrollment', '/root/kinetalk_runs/teacher_schedule_v1/data_locked/enrollment.jsonl',
        '--native-root', native, '--native-manifest', native+'/train.jsonl',
        '--trained-run', str(root/'run12'), '--history-run', str(root/'history12'),
        '--centered-run', str(root/'repair12/centered_prior'), '--context-run', str(root/'context12'),
        '--split-report', str(root/'repair12/scale_diagnosis/report.json'),
        '--initial-diagnostic', str(root/'initial_history_fit128'),
        '--delta-dir', str(args.delta_dir), '--output', str(args.output), '--epochs', '30']
    if args.smoke:
        command.append('--smoke')
    if args.resume:
        command.append('--resume')
    if args.stop_after_epoch:
        command.extend(['--stop-after-epoch', str(args.stop_after_epoch)])
    record = args.output.with_name(args.output.name+'_process.json')

    def write(value):
        temp = record.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2), encoding='utf8')
        temp.replace(record)

    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.Popen(command, cwd=code)
    write({'status': 'running', 'pid': proc.pid, 'started_utc': started, 'command': command})
    result = proc.wait()
    status = 'failed' if result else ('smoke_checkpoint_stop' if args.stop_after_epoch else 'complete')
    write({'status': status, 'pid': proc.pid, 'exit_code': result,
           'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'command': command})
    raise SystemExit(result)


if __name__ == '__main__':
    main()
