"""Prepare a separate snapshot without changing the running posture pilots."""
from pathlib import Path
import json
import shutil
root=Path('/root/kinetalk_v5_20261011');code=root/'code_acoustic_geometry'
if code.exists() or (root/'acoustic_geometry_pilot.lock').exists():
    raise FileExistsError('Acoustic geometry pilot already prepared')
shutil.copytree(root/'code_posture_pilot',code)
print(json.dumps(dict(snapshot=str(code),source='code_posture_pilot',pending_upload=True)))
