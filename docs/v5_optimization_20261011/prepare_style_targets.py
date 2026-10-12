"""Separate CPU diagnostic snapshot; all model weights remain frozen."""
from pathlib import Path
import shutil
import json
root=Path('/root/kinetalk_v5_20261011');code=root/'code_style_targets'
if code.exists():raise FileExistsError('Target-style snapshot already exists')
shutil.copytree(root/'code_acoustic_geometry',code)
print(json.dumps(dict(snapshot=str(code),source='code_acoustic_geometry',device='cpu')))
