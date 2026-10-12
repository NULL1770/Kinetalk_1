"""Compact remote status, run through the existing pinned SSH bridge."""
from pathlib import Path
import json
import os
import subprocess
import time

root = Path('/root/kinetalk_v5_20261011')
result = {'now_unix': time.time(), 'files': {}}
names = ['posture_pilot_state.json', 'posture_pilot_complete.json',
         'geometry_route_audit/state.json', 'geometry_route_audit/complete.json']
for arm in ('posture_control', 'posture_direct', 'posture_privileged'):
    names += [arm + '/state.json', arm + '/complete.json',
              arm + '_full_eval/state.json', arm + '_full_eval/report.json',
              arm + '_state/report.json', arm + '_dynamics/report.json',
              arm + '_controls/report.json']
for name in names:
    path = root / name
    if not path.is_file():
        result['files'][name] = {'exists': False}
        continue
    data = json.loads(path.read_text())
    summary = {'exists': True, 'bytes': path.stat().st_size}
    if name.endswith('complete.json'):
        summary.update({k:v for k,v in data.items() if k not in ('protocol','history')})
    elif name.endswith('report.json'):
        summary.update({k:data[k] for k in ('schema','clips','checkpoint_sha256','test_loaded') if k in data})
        if 'results' in data and 'v4_sample/clip_all' in data['results']:
            view=data['results']['v4_sample/clip_all']
            summary['sample_clip_all'] = {k:view['metrics'][k] for k in (
                'lve_mean_mm_mean','eye_forehead_eve_sqrt_mean','jaw_closure_frame_f1')}
            summary['sample_clip_all'].update(probe_F1=view['probes'][0]['macro_f1'],
                angry_F1=view['probes'][0]['per_class']['angry']['f1'])
        elif 'summary' in data:
            summary['summary']={k:v.get('macro_mse') for k,v in data['summary'].items()
                if isinstance(v,dict) and 'macro_mse' in v}
        elif 'status' in data:
            summary['status']=data['status']
    else:
        summary.update(data)
    result['files'][name] = summary
for pid in (8413,9406):
    try:
        os.kill(pid,0); alive=True
    except ProcessLookupError:
        alive=False
    result[str(pid)+'_alive']=alive
result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw',
    '--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
result['disk']=subprocess.run(['df','-h','/root','/root/autodl-tmp'],capture_output=True,text=True).stdout.strip()
print(json.dumps(result))
