from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/root/kinetalk_v5_20261011');code=root/'code_content_style_route'
if code.exists() or (root/'content_style_route.lock').exists():raise FileExistsError('route already dispatched')
shutil.copytree(root/'code',code)
# The caller uploads the latest route files to root/code before dispatch.
required=(code/'scripts/train_v5_mixed.py',code/'kinetalk_v4/domain_routes.py')
if not all(p.is_file() for p in required):raise FileNotFoundError('route source snapshot incomplete')
hashes={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}
(root/'content_style_route_source_hashes.json').write_text(json.dumps(hashes,indent=2))
with (root/'content_style_route_queue.log').open('x') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(root/'run_content_style_route.py')],cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,detached=True,updates=256,scope='content_style',waits_for='mixed_pilot_queue_complete.json')))
