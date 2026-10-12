"""Run remotely to create a fresh source snapshot before explicit file upload."""
from pathlib import Path
import json
import shutil
root = Path('/root/kinetalk_v5_20261011')
code = root / 'code_posture_pilot'
if code.exists() or (root / 'posture_pilot.lock').exists():
    raise FileExistsError('Posture pilot already prepared/dispatched')
shutil.copytree(root / 'code_content_style_route', code)
print(json.dumps(dict(snapshot=str(code), source='code_content_style_route', uploaded_changes_pending=True)))
