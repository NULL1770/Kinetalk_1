from pathlib import Path
import json
import subprocess
root = Path('/root/kinetalk_v5_20261011'); py = '/root/miniconda3/bin/python'
arm = 'geometry_rollout_mean'
commands = [
    [py, '-u', 'scripts/train_v5_geometry_repair.py', '--binding', str(root / 'binding.json'),
     '--checkpoint', '/root/kinetalk_v4_20261011/mead_formal_seed47/final.pt',
     '--output', str(root / arm), '--epochs', '2', '--limit', '1024', '--eval-limit', '128',
     '--repair', '--rollout-mean-weight', '.5'],
    [py, '-u', 'scripts/evaluate_v5_candidate.py', '--binding', str(root / 'binding.json'),
     '--checkpoint', str(root / arm / 'final.pt'), '--output', str(root / (arm + '_eval')),
     '--selection-json', str(root / 'diagnostic_seed47/selection.json')],
]
for i, command in enumerate(commands):
    with (root / (arm + f'_{i}.log')).open('w') as log:
        p = subprocess.run(command, cwd=root / 'code', stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT)
    if p.returncode: raise SystemExit(p.returncode)
(root / (arm + '_queue_complete.json')).write_text(json.dumps({'status': 'complete', 'arms': 1}))
