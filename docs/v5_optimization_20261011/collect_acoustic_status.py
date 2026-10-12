"""Read-only compact status for the immutable acoustic geometry pilot."""
from pathlib import Path
import json

ROOT = Path('/root/kinetalk_v5_20261011')
status = ROOT / 'acoustic_geometry_pilot_state.json'
print(status.read_text())
for arm in ('geometry_coarse', 'geometry_acoustic', 'geometry_acoustic_mean'):
    for suffix in ('/complete.json', '_full_eval/report.json', '_geometry/report.json'):
        path = ROOT / (arm + suffix)
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        row = dict(arm=arm, report=suffix, clips=data.get('clips'), updates=data.get('updates'),
                   checkpoint=data.get('checkpoint_sha256'))
        if 'results' in data:
            value = data['results'].get('v4_sample/clip_all')
            if value:
                row['metrics'] = {key: value['metrics'].get(key) for key in
                    ('lve_mean_mm_mean', 'vertex_lve_sqrt_mean', 'eye_forehead_eve_sqrt_mean',
                     'jaw_closure_frame_f1', 'arkit_mbe', 'arkit_lbe')}
                row['probe'] = value['probes'][0]['macro_f1']
                row['angry'] = value['probes'][0]['per_class']['angry']['f1']
        print(json.dumps(row))
