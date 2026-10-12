"""Collect completed immutable matched-pilot evidence, checking remote/local SHA."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent / 'dual_evidence'


def transport(request):
    result = subprocess.run([sys.executable, str(PROJECT/'.codex-finalizer/remote_ops.py')],
        input=json.dumps(request), text=True, encoding='utf8', capture_output=True, cwd=PROJECT)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return result.stdout


def main():
    source = '''
from pathlib import Path
import hashlib,json
r=Path('/root/kinetalk_v5_20261012_dual')
c=r/'queue/complete.json'
if not c.is_file() or json.loads(c.read_text())['status']!='complete':
 raise RuntimeError('Queue incomplete')
items=[]
for name in ['queue/complete.json','queue/queue_state.json','launch_receipt.json',
             'source_manifest.json','upload_hashes.json','queue_plan.json']:
 items.append((r/name,name))
for arm in ['control','direct','guided']:
 for p in sorted((r/'runs'/arm).glob('*')):
  if p.name in ['final.pt','history.json','protocol.json','complete.json']:
   items.append((p,str(p.relative_to(r))))
 for suffix in ['full1367','dynamics128','controls32','geometry128','style64']:
  d=r/'evaluations'/(arm+'_'+suffix)
  if not (d/'complete.json').is_file():raise RuntimeError('Missing '+str(d))
  for p in sorted(d.glob('*.json')):
   items.append((p,str(p.relative_to(r))))
  if suffix=='full1367':
   for p in sorted((d/'render_inputs').glob('*.npz')):items.append((p,str(p.relative_to(r))))
manifest=[dict(remote=str(p),local=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,n in items]
print(json.dumps(dict(schema='dual_timescale_collection_v1',files=manifest,source_root=str(r),test_loaded=False,remote_mutated=False)))
'''
    manifest = json.loads(transport({'command':"/root/miniconda3/bin/python - <<'PY'\n"+source+"\nPY",'timeout':120}))
    transfers = []
    for item in manifest['files']:
        p=ROOT/item['local']
        if p.is_file():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Existing evidence differs '+str(p))
        else:
            transfers.append([item['remote'],str(p.relative_to(PROJECT))])
    if transfers:transport({'download':transfers,'download_prefetch_requests':4})
    for item in manifest['files']:
        p=ROOT/item['local']
        if p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Transfer hash mismatch '+str(p))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'collection_receipt.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(dict(status='collected',files=len(manifest['files']),new_transfers=len(transfers))))


if __name__=='__main__':main()
