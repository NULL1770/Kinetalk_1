from pathlib import Path
import json
import subprocess
import time

root = Path('/root/kinetalk_v5_20261011')
python = '/root/miniconda3/bin/python'
cwd = root / 'code'
commands = []
for arm in ('continued_control', 'geometry_repaired'):
    train = [python, '-u', 'scripts/train_v5_geometry_repair.py', '--binding', str(root / 'binding.json'),
             '--checkpoint', '/root/kinetalk_v4_20261011/mead_formal_seed47/final.pt',
             '--output', str(root / arm), '--epochs', '2', '--limit', '1024', '--eval-limit', '128']
    if arm == 'geometry_repaired': train.append('--repair')
    commands.append((arm + '_train', train))
    evaluate = [python, '-u', 'scripts/evaluate_v5_candidate.py', '--binding', str(root / 'binding.json'),
                '--checkpoint', str(root / arm / 'final.pt'), '--output', str(root / (arm + '_eval')),
                '--selection-json', str(root / 'diagnostic_seed47/selection.json')]
    commands.append((arm + '_eval', evaluate))
for name, command in commands:
    state = {'status': 'running', 'phase': name, 'started_unix': time.time(), 'command': command}
    (root / 'pilot_queue_state.json').write_text(json.dumps(state, indent=2))
    with (root / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state.update(status='failed', exit_code=result.returncode)
        (root / 'pilot_queue_state.json').write_text(json.dumps(state, indent=2))
        raise SystemExit(result.returncode)
(root / 'pilot_queue_state.json').write_text(json.dumps({'status': 'complete', 'finished_unix': time.time()}))
