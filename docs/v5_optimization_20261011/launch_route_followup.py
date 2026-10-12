from pathlib import Path
import json,subprocess
root=Path('/root/kinetalk_v5_20261011')
if (root/'route_followup.lock').exists() or (root/'route_followup.log').exists():raise FileExistsError('route follow-up already dispatched')
with (root/'route_followup.log').open('x') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(root/'run_route_followup.py')],cwd=root/'code_content_style_route',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,detached=True,waits_for=['content_style_route_complete.json','pilot_followup_complete.json'],full_development=True,factor_controls=True,promotion=False)))
