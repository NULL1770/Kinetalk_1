from pathlib import Path
import json
import shutil

root = Path('/root/kinetalk_v5_20261011')
if root.exists():
    raise FileExistsError('Fresh v5 workspace required')
root.mkdir()
shutil.copytree('/root/kinetalk_v4_20261011/code_eval', root / 'code')
shutil.copy2('/root/kinetalk_v4_20261011/binding.json', root / 'binding.json')
shutil.copy2('/root/kinetalk_v4_20261011/lip_geometry.npz', root / 'lip_geometry.npz')
print(json.dumps({'root': str(root), 'created': True, 'original_untouched': True}))
