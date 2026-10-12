"""Keep all geometry pipeline stages and classes with report provenance."""
from pathlib import Path
import csv
import hashlib
import json

root=Path(__file__).resolve().parent
files=dict(MEAD='geometry_mead.json',HDTF_all='geometry_mixed.json',
           HDTF_route='geometry_route.json',posture_control='geometry_posture_control.json')
reports={name:json.loads((root/path).read_text()) for name,path in files.items()}
first=next(iter(reports.values()))
if any(r['indices']!=first['indices'] or r['binding_sha256']!=first['binding_sha256']
       or r['source_sha256']!=first['source_sha256'] or r['test_loaded']
       or r['query_motion_role']!='scoring only' for r in reports.values()):
    raise ValueError('Geometry protocols differ')
rows=[]
for name,r in reports.items():
    for emotion,modes in r['summary'].items():
        for mode,axes in modes.items():
            for axis,values in axes.items():
                rows.append(dict(arm=name,emotion=emotion,stage=mode,axis=axis,clips=r['clips'],**values))
with (root/'geometry_pipeline_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
provenance=dict(schema='v5_geometry_pipeline_table_v1',sources={name:dict(file=files[name],
    report_sha256=hashlib.sha256((root/files[name]).read_bytes()).hexdigest(),
    checkpoint_sha256=r['checkpoint_sha256']) for name,r in reports.items()},
    indices=first['indices'],binding_sha256=first['binding_sha256'],source_sha256=first['source_sha256'],
    rows=len(rows),test_loaded=False,trained_ablation=False,
    limits=first['limits'])
(root/'geometry_pipeline_provenance.json').write_text(json.dumps(provenance,indent=2))
for r in rows:
    if r['emotion'] in ('ALL','angry') and r['axis'] in ('width_mm','aperture_mm'):
        print(json.dumps({k:r[k] for k in ('arm','emotion','stage','axis','MAE','bias','centered_MSE')}))
