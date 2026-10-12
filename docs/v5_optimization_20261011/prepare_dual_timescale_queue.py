"""Prepare the isolated snapshot's manifest and dispatch its smoke/pilot/evaluations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/kinetalk_v5_20261012_dual')
PARENT = Path('/root/kinetalk_v5_20261011/state_raw_direct/final.pt')
BINDING = Path('/root/kinetalk_v5_20261011/binding.json')
PYTHON = '/root/miniconda3/bin/python'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


assert sha(PARENT) == '310fe9fa348ecb7f6210fc854f3c867dc055e470db00be1d2dc38722016df702'
assert sha(BINDING) == 'd30f10075b704c4cb228629549820056c952533fa5c4507666ab8640a1fa6087'
assert not (ROOT / 'queue_plan.json').exists(), 'Fresh queue required'
manifest = {str(path.relative_to(ROOT)): sha(path) for path in sorted(ROOT.rglob('*.py'))}
manifest_path = ROOT / 'source_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
jobs = []


def add(name, script, params, marker):
    jobs.append(dict(name=name, argv=[PYTHON, str(ROOT/'scripts'/script), *params],
                     marker=str(marker), timeout_seconds=2700))


def training(name, arm, updates, eval_limit=128):
    out = ROOT/'runs'/name
    add(name, 'train_dual_timescale_v5.py', ['--binding', str(BINDING), '--checkpoint', str(PARENT),
        '--output', str(out), '--arm', arm, '--updates', str(updates), '--eval-limit', str(eval_limit),
        '--batch-size', '12', '--lr', '5e-5', '--seed', '47'], out/'complete.json')


training('guided_smoke16', 'guided', 16, 16)
for arm in ('control', 'direct', 'guided'):
    training(arm, arm, 256)
for arm in ('control', 'direct', 'guided'):
    ck = ROOT/'runs'/arm/'final.pt'
    def evaluation(name, script, extra):
        out = ROOT/'evaluations'/(arm+'_'+name)
        add(arm+'_'+name, script, ['--binding', str(BINDING), '--checkpoint', str(ck),
            '--output', str(out), *extra], out/'complete.json')
    evaluation('full1367', 'evaluate_v5_candidate.py', ['--limit', '0'])
    evaluation('dynamics128', 'evaluate_v4_dynamics.py', ['--limit', '128', '--seeds', '47', '91', '137'])
    evaluation('controls32', 'audit_v5_controls.py', ['--limit', '32'])
    evaluation('geometry128', 'audit_v5_geometry.py', ['--limit', '128'])
    evaluation('style64', 'audit_v5_style_targets.py', ['--limit', '64', '--device', 'cuda'])
plan = dict(output=str(ROOT/'queue'), code_root=str(ROOT), source_manifest=str(manifest_path),
            source_checkpoint_sha256=sha(PARENT), binding_sha256=sha(BINDING), jobs=jobs,
            test_loaded=False, promotion=False)
plan_path = ROOT/'queue_plan.json'
plan_path.write_text(json.dumps(plan, indent=2))
(ROOT/'queue').mkdir()
with (ROOT/'queue'/'supervisor.log').open('xb') as log:
    process = subprocess.Popen([PYTHON, str(ROOT/'scripts'/'run_dual_timescale_queue.py'),
        '--plan', str(plan_path)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(supervisor_pid=process.pid, root=str(ROOT), jobs=len(jobs),
    parent_sha256=sha(PARENT), binding_sha256=sha(BINDING), source_files=len(manifest),
    stages='guided smoke16, three matched256-update arms, then full/dynamics/control/geometry/style',
    test_loaded=False, promotion=False)
(ROOT/'launch_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
