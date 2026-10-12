"""Read-only inspection before selecting the retained joint model."""
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'docs/v6_factor_paths_20261012'))
from remote_transport import transport

REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
import torch
root=Path('/root/kinetalk_v5_20261012_dual')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checkpoint=root/'runs/control/final.pt'
expected='372dbc2a23913743cb67452bf6f5756e3be195f81c468f2a1f972afbb79c9add'
assert sha(checkpoint)==expected
binding=Path('/root/kinetalk_v5_20261011/binding.json')
assert sha(binding)=='d30f10075b704c4cb228629549820056c952533fa5c4507666ab8640a1fa6087'
sys.path.insert(0,str(root))
from scripts.load_candidate_model import load_candidate_model
ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
model=load_candidate_model(ck,'cpu').eval()
assert type(model).__name__=='KineTalkV4Model'
assert tuple(model.decoder.output.weight.shape)==(52,192)
assert not any(k.startswith('backbone.') or k.startswith('student_logvar.') for k in ck['model'])
manifest=json.loads((root/'source_manifest.json').read_text())
sources={k:v for k,v in manifest.items() if k.startswith(('kinetalk_b0/','kinetalk_v2/','kinetalk_v4/'))}
for name,digest in sources.items():
    assert sha(root/name)==digest, name
queue=json.loads(Path('/root/kinetalk_v6_20261012_paths/pilot_queue/queue_state.json').read_text())
assert queue['status']=='stopped'
active=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try: argv=(entry/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError): continue
    if any(arg.startswith(b'/root/kinetalk_v6_20261012_paths/') for arg in argv):
        active.append({'pid':int(entry.name),'argv':[x.decode(errors='replace') for x in argv if x]})
assert not active,active
binding_data=json.loads(binding.read_text())
result=dict(schema='joint_model_rollback_preflight_v1',checkpoint=str(checkpoint),
    checkpoint_sha256=expected,binding=str(binding),binding_sha256=sha(binding),
    source_root=str(root),source_manifest_sha256=sha(root/'source_manifest.json'),
    model_family=type(model).__name__,strict_load=True,joint_output_shape=[52,192],
    config=ck['config'],source_hashes=sources,v6_queue_status=queue['status'],
    v6_active_processes=active,binding_keys=sorted(binding_data),test_loaded=False)
print(json.dumps(result))
'''

if __name__ == '__main__':
    result = json.loads(transport({'command': "/root/miniconda3/bin/python - <<'PY'\n" + REMOTE + "\nPY", 'timeout': 120}))
    output = Path(__file__).with_name('preflight.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: result[k] for k in ('model_family', 'strict_load', 'joint_output_shape', 'v6_queue_status', 'v6_active_processes')}))
