"""Read-only status and SHA-bound receipts; never dispatch or implicitly resume."""
import json
from pathlib import Path
from remote_transport import transport

code='''from pathlib import Path
import json,hashlib,time
r=Path('/root/kinetalk_v6_20261012_paths')
queue=json.loads((r/'pilot_queue/queue_state.json').read_text())
result={'schema':'v6_pilot_status_v1','time':time.time(),'root':str(r),'queue':queue,'runs':{},'test_loaded':False}
for name in ['direct_seed47','posterior_seed47','direct_seed91','posterior_seed91']:
 d=r/'runs'/name
 for marker in ['complete.json','state.json']:
  if (d/marker).exists():
   result['runs'][name]=json.loads((d/marker).read_text());break
print(json.dumps(result))
'''
status=json.loads(transport({'command':"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY"}))
Path(__file__).with_name('latest_status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
print(json.dumps(dict(status=status['queue']['status'],runs={k:{field:v.get(field) for field in
    ['status','step','updates','total_updates','elapsed_seconds','loss','gradient_norm']} for k,v in status['runs'].items()})))
transport({'download':[['/root/kinetalk_v6_20261012_paths/source_manifest.json',
                       'docs/v6_factor_paths_20261012/source_manifest_remote.json']]})
