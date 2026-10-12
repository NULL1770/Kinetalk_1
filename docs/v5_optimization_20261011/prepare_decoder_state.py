"""Make a fresh snapshot; all active snapshots remain immutable."""
from pathlib import Path
import shutil
import json

root = Path('/root/kinetalk_v5_20261011')
code = root/'code_decoder_state'
if code.exists() or (root/'decoder_state_pilot.lock').exists():
    raise FileExistsError('Decoder-state pilot already prepared')
shutil.copytree(root/'code_acoustic_geometry', code)
source = root/'code_style_targets/scripts/audit_v5_style_targets.py'
shutil.copy2(source, code/'scripts/audit_v5_style_targets.py')
print(json.dumps(dict(snapshot=str(code), source='code_acoustic_geometry', pending_upload=True)))
