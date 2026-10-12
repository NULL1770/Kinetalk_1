"""Collect only completed immutable v6 stages, validating each transferred SHA."""
import hashlib
import json
from pathlib import Path
from remote_transport import transport,PROJECT

ROOT=Path(__file__).parent/'evidence'
code='''from pathlib import Path
import hashlib,json
r=Path('/root/kinetalk_v6_20261012_paths')
queue=json.loads((r/'pilot_queue/queue_state.json').read_text())
items=[]
for job in queue['jobs']:
 if job['status']!='complete':continue
 d=Path(job['marker']).parent
 for p in sorted(d.rglob('*')):
  if not p.is_file():continue
  if p.suffix=='.json' or (p.name=='final.pt' and d.parent.name=='runs') or p.suffix=='.npz':
   if p.name=='state.json':continue
   items.append(dict(remote=str(p),local=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(schema='v6_completed_stage_collection_v1',files=items,queue_status=queue['status'],completed_jobs=[j['name'] for j in queue['jobs'] if j['status']=='complete'],test_loaded=False)))
'''
receipt=json.loads(transport({'command':"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY"}))
transfer=[]
for item in receipt['files']:
    path=ROOT/item['local']
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Existing immutable evidence differs '+str(path))
    else:transfer.append([item['remote'],str(path.relative_to(PROJECT))])
if transfer:transport({'download':transfer,'download_prefetch_requests':4})
for item in receipt['files']:
    path=ROOT/item['local']
    if path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Transfer verification failed '+str(path))
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/'collection_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8',newline='\n')
print(json.dumps(dict(status='verified',files=len(receipt['files']),new=len(transfer),jobs=receipt['completed_jobs'])))
