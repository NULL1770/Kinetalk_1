"""Export a fixed seed42, metadata-selected comparison from saved raw NPZ files."""
from pathlib import Path
import json
import numpy as np


def build(root=Path('artifacts/event_schedule_20260919')):
    cid=json.loads((root/'review/manifest.json').read_text())[1]['clip_id']
    same24=root/'receiver_bottleneck_audit'
    repaired=root/'audits/receiver_support_repair_20260919/receiver_null_full_support'
    entries=[('GT reference',same24/'source_prior/npz',0),
             ('Frozen baseline',same24/'source_prior/npz',1),
             ('AE oracle',same24/'ae_oracle/npz',2),
             ('Source prior seed42',same24/'source_prior/npz',2),
             ('Old null seed42',root/'runs/event_mouth_formal_20260919_v1/event/control/null/npz',2),
             ('Full support seed42',repaired/'npz',2)]
    arrays=[]; first=None
    for label,path,index in entries:
        with np.load(path/(cid+'.npz'),allow_pickle=False) as z:
            current={k:z[k].copy() for k in z.files}
        if first is None:first=current
        for field in ('valid','times','channels','clip_id','native_valid'):
            if not np.array_equal(current[field],first[field]):raise ValueError('Different clock or rig: '+field)
        if not np.array_equal(current['motions'][1],first['motions'][1]):raise ValueError('Baseline differs')
        arrays.append(current['motions'][index])
    first.update(motions=np.stack(arrays),mode_names=np.array([r[0] for r in entries]))
    dest=root/'support_review'
    dest.mkdir(exist_ok=True)
    output=dest/(cid+'.npz')
    np.savez_compressed(output,**first)
    (dest/'manifest.json').write_text(json.dumps({'clip_id':cid,'frames':len(first['times']),
        'selection':'Second clip in existing metadata-fixed two-example manifest; same seed42, no metric selection',
        'audio':'not attached; no AV certification','panels':[r[0] for r in entries]},indent=2),encoding='utf8')
    print(output)


if __name__=='__main__':build()
