"""Fresh CPU-only diagnostic snapshot, independent of live training."""
from pathlib import Path
import shutil
import json
root=Path('/root/kinetalk_v5_20261011');code=root/'code_geometry_audit'
if code.exists():raise FileExistsError('Geometry audit snapshot already exists')
shutil.copytree(root/'code_posture_pilot',code)
print(json.dumps(dict(snapshot=str(code),device='cpu',source='code_posture_pilot')))
