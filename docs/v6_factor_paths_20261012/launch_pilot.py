"""Smoke-gated, immutable two-seed feasibility pilot and automatic evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/root/kinetalk_v6_20261012_paths')
PYTHON='/root/miniconda3/bin/python'
PARENT=Path('/root/kinetalk_v5_20261011/state_raw_direct/final.pt')
BINDING=Path('/root/kinetalk_v5_20261011/binding.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

smoke=json.loads((ROOT/'runs/posterior_smoke16/complete.json').read_text())
assert smoke['status']=='complete' and smoke['updates']==16 and smoke['head_gradients_verified']
assert not smoke['protocol']['test_loaded']
assert sha(ROOT/'runs/posterior_smoke16/final.pt')==smoke['checkpoint_sha256']
for name,checksum in json.loads((ROOT/'source_manifest.json').read_text()).items():
    assert sha(ROOT/name)==checksum
if (ROOT/'pilot_plan.json').exists():raise FileExistsError('pilot already dispatched')
jobs=[]
def add(name,script,args,marker,timeout=2700):
    jobs.append(dict(name=name,argv=[PYTHON,str(ROOT/'scripts'/script),*args],
                     marker=str(marker),timeout_seconds=timeout))

for seed in (47,91):
    for arm in ('direct','posterior'):
        name=f'{arm}_seed{seed}';out=ROOT/'runs'/name
        add(name,'train_factor_paths_v6.py',['--binding',str(BINDING),'--checkpoint',str(PARENT),
            '--output',str(out),'--arm',arm,'--updates','768','--batch-size','12',
            '--seed',str(seed),'--lr','3e-4','--eval-limit','128','--bootstrap-updates','128'],out/'complete.json')
        checkpoint=out/'final.pt'
        # Evaluate each completed arm before the next run; final checkpoints fixed.
        for suffix,script,extra in (
            ('full1367','evaluate_v5_candidate.py',['--limit','0']),
            ('dynamics128','evaluate_v4_dynamics.py',['--limit','128','--seeds','47','91','137']),
            ('controls32','audit_v5_controls.py',['--limit','32']),
            ('geometry128','audit_v5_geometry.py',['--limit','128']),
            ('style64','audit_v5_style_targets.py',['--limit','64','--device','cuda'])):
            ev=ROOT/'evaluations'/(name+'_'+suffix)
            add(name+'_'+suffix,script,['--binding',str(BINDING),'--checkpoint',str(checkpoint),
                '--output',str(ev),*extra],ev/'complete.json')
plan=dict(output=str(ROOT/'pilot_queue'),code_root=str(ROOT),source_manifest=str(ROOT/'source_manifest.json'),
    source_checkpoint_sha256=sha(PARENT),binding_sha256=sha(BINDING),jobs=jobs,
    test_loaded=False,promotion=False)
(ROOT/'pilot_plan.json').write_text(json.dumps(plan,indent=2));(ROOT/'pilot_queue').mkdir()
with (ROOT/'pilot_queue/supervisor.log').open('xb') as log:
    p=subprocess.Popen([PYTHON,str(ROOT/'scripts/run_dual_timescale_queue.py'),'--plan',str(ROOT/'pilot_plan.json')],
        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(supervisor_pid=p.pid,status='pilot_launched',root=str(ROOT),jobs=len(jobs),
    arms=['direct','posterior'],training_seeds=[47,91],updates_per_arm=768,
    pilot_plan_sha256=sha(ROOT/'pilot_plan.json'),smoke_checkpoint_sha256=smoke['checkpoint_sha256'],
    test_loaded=False,promotion=False)
(ROOT/'pilot_launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
