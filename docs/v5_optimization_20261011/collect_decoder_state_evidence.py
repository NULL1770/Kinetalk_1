"""Collect immutable completed decoder-state artifacts through the existing SSH helper.

No credentials are duplicated. The remote side only reads the named experiment;
no model, source, dataset or process is modified. Collection fails on incomplete
queues or any local/remote hash mismatch.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ARMS = ('state_raw_direct', 'state_decoder_direct', 'state_decoder_privileged')


def transport(request):
    result = subprocess.run([sys.executable, str(PROJECT/'.codex-finalizer/remote_ops.py')],
        input=json.dumps(request), text=True, encoding='utf8', capture_output=True, cwd=PROJECT)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
    return result.stdout


def main():
    source = '''
from pathlib import Path
import hashlib,json
r=Path('/root/kinetalk_v5_20261011')
arms=('state_raw_direct','state_decoder_direct','state_decoder_privileged')
complete=r/'decoder_state_complete.json'
if not complete.is_file():
 raise RuntimeError('Queue incomplete: '+(r/'decoder_state_pilot_state.json').read_text())
items=[]
for name in ['decoder_state_complete.json','decoder_state_pilot_state.json','decoder_state_jobs.json',
             'decoder_state_source_hashes.json','decoder_state_upload_hashes.json']:
 items.append((r/name,name))
for arm in arms:
 items += [(r/arm/'complete.json',arm+'_train_complete.json'),(r/arm/'final.pt',arm+'_final.pt')]
 for suffix in ['subset_eval','state','full_eval','dynamics','controls','geometry']:
  d=r/(arm+'_'+suffix)
  if not (d/'complete.json').is_file(): raise RuntimeError('Missing stage receipt: '+str(d))
  items.append((d/'complete.json',arm+'_'+suffix+'_complete.json'))
  if suffix in ('controls','geometry'):
   files=[p for p in d.glob('*.json') if p.name not in ('state.json','complete.json')]
   if len(files)!=1: raise ValueError('Ambiguous report '+str(d))
   items.append((files[0],arm+'_'+suffix+'_report.json'))
  else: items.append((d/'report.json',arm+'_'+suffix+'_report.json'))
  if suffix=='full_eval':
   for p in sorted((d/'render_inputs').glob('*.npz')):
    items.append((p,arm+'_render_inputs/'+p.name))
 style=r/'state_target_style'
 if style.exists():
  for p in sorted(style.glob('*.json')):
   if all(str(p)!=str(x[0]) for x in items): items.append((p,'state_target_style/'+p.name))
manifest=[dict(remote=str(p),local=name,bytes=p.stat().st_size,
 sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,name in items]
print(json.dumps(dict(schema='decoder_state_collection_v1',files=manifest,
 source_root=str(r),test_loaded=False,remote_mutated=False)))
'''
    manifest = json.loads(transport({'command': "/root/miniconda3/bin/python - <<'PY'\n"+source+"\nPY", 'timeout': 120}))
    transfers=[]
    for item in manifest['files']:
        path=ROOT/item['local']
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Existing local evidence differs: '+str(path))
        else:
            transfers.append([item['remote'],str(path.relative_to(PROJECT))])
    if transfers:
        transport({'download':transfers,'download_prefetch_requests':4})
    for item in manifest['files']:
        p=ROOT/item['local']
        if p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Hash mismatch after transfer: '+str(p))
    (ROOT/'decoder_state_collection_receipt.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(dict(status='collected',files=len(manifest['files']),new_transfers=len(transfers),
                         test_loaded=False,remote_mutated=False)))


if __name__=='__main__':
    main()
