"""Preserve relative paths; no credentials or model checkpoints in upload."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[2]
sources = []
for folder in ('kinetalk_v4', 'kinetalk_v5'):
    sources.extend(root.joinpath(folder).rglob('*.py'))
for name in ('train_dual_timescale_v5.py', 'load_candidate_model.py', 'evaluate_v5_candidate.py',
             'evaluate_v4_dynamics.py', 'audit_v5_controls.py', 'audit_v5_state_coordinates.py',
             'audit_v5_geometry.py', 'audit_v5_style_targets.py'):
    sources.append(root/'scripts'/name)
sources.append(root/'tests'/'test_dual_timescale_v5.py')
artifact = root/'candidate_v5_upload_corrected.zip'
hashes = {}
with zipfile.ZipFile(artifact, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for path in sources:
        relative = path.relative_to(root).as_posix()
        z.write(path, relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('run_dual_timescale_queue.py', 'prepare_dual_timescale_queue.py'):
        path = Path(__file__).parent/name
        relative = 'scripts/'+name
        z.write(path, relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    z.writestr('upload_hashes.json', json.dumps(hashes, indent=2))
print(json.dumps(dict(path=str(artifact), files=len(hashes),
    sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())))
