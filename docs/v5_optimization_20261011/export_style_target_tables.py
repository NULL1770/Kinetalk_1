"""Target-direction statistics, preserving every policy and evidence hash."""
from pathlib import Path
import json
from export_control_dynamics_tables import export
root=Path(__file__).resolve().parent
files={name:'style_targets_'+name+'.json' for name in ('mead','route','direct','privileged')}
reports={name:json.loads((root/path).read_text()) for name,path in files.items()}
first=reports['mead']
if any(r['selection']!=first['selection'] or r['test_loaded'] or r['source_sha256']!=first['source_sha256']
       or r['binding_sha256']!=first['binding_sha256'] for r in reports.values()):
    raise ValueError('Target-style scope differs')
rows=[]
for arm,r in reports.items():
    for policy,v in r['summary'].items():
        for region in ('all51','mouth','brows','eyes'):
            for statistic in ('mean','centered_rms','q90_q10','displacement_rms'):
                prefix=region+'/'+statistic
                rows.append(dict(arm=arm,policy=policy,region=region,statistic=statistic,
                    pairs=r['matched_pairs'],target_error_before=v[prefix+'_before'],
                    target_error_after=v[prefix+'_after'],improved_pair_fraction=v[prefix+'_improved'],
                    probe1_label_preserved=v['probe1_label_preserved'],
                    probe1_emotion_correct_after=v['probe1_emotion_correct_after'],
                    jaw_closure_disagreement=v['jaw_closure_disagreement']))
export('style_target_statistics.csv',rows,files.values(),first['limits'])
