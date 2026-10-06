"""SHA-verify completed small frozen diagnostics and summarize cell means."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.frozen_factor_diagnostic import sha


def average(rows, field, region, columns):
    values = [r[field][region] for r in rows if r.get(field) is not None]
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    return {name: float(x[np.isfinite(x[:, i]), i].mean()) if np.isfinite(x[:, i]).any() else None
            for i, name in enumerate(columns)}


def summarize(root):
    result = dict(schema='phase31_small_factor_readout_v1', training_performed=False,
        test_loaded=False, default_replaced=False,
        scope='66 fixed validation speaker/emotion/intensity cells; intervention draw42 only, original3draws exactly replayed. Descriptive diagnosis, not full performance or 772D results.', models={})
    for name in ('neutral', 'native'):
        folder = root/f'factors_{name}_v2'
        state = json.loads((folder/'state.json').read_text())
        if state['status'] != 'complete':
            raise ValueError('Unfinished model: ' + name)
        for filename, key in [('report.json','report_sha256'),('fixed_exports.npz','exports_sha256'),('replay.json','replay_sha256')]:
            if sha(folder/filename) != state[key]:
                raise ValueError('Artifact hash differs: ' + name + '/' + filename)
        r = json.loads((folder/'report.json').read_text())
        if not r['replay_bit_exact'] or not r['model_state_unchanged'] or r['test_loaded']:
            raise ValueError('Frozen contract failed')
        if len(r['selected_indices']) != 66 or len(set(r['selected_indices'])) != 66:
            raise ValueError('Sampling contract differs')
        rows = r['rows']; modes = sorted({row['mode'] for row in rows})
        if len(rows) != 66*13*2:
            raise ValueError('Missing intervention rows')
        original_groups = {}; responses = {}
        for policy in ('raw','clip_all'):
            for emotion in [None]+list(range(8)):
                selected=[x for x in rows if x['policy']==policy and x['mode']=='original'
                          and (emotion is None or x['emotion']==emotion)]
                key=f'{policy}/'+('all' if emotion is None else str(emotion))
                original_groups[key] = dict(cells=len(selected),
                    b0={region:average(selected,'b0_score',region,r['region_columns']) for region in r['regions']},
                    final={region:average(selected,'target_score',region,r['region_columns']) for region in r['regions']})
            for mode in modes:
                selected=[x for x in rows if x['policy']==policy and x['mode']==mode]
                if len(selected)!=66:raise ValueError('Incomplete mode')
                responses[f'{policy}/{mode}']=dict(cells=66,
                    response={region:average(selected,'response',region,r['response_columns']) for region in r['regions']},
                    target_score={region:average(selected,'target_score',region,r['region_columns']) for region in r['regions']})
        reference_summary={}
        for role in ('train','validation'):
            ref=[x for x in r['reference_rows'] if x['role']==role]
            reference_summary[role]=dict(speakers=len(ref),**{
                key:float(np.mean([x[key] for x in ref])) for key in
                ('same_code_rms','same_bias_rms','other_code_rms_mean','other_bias_rms_mean')})
        result['models'][name]=dict(report_sha256=state['report_sha256'],exports_sha256=state['exports_sha256'],
            binding_sha256=r['binding_sha256'],selected_clip_ids=r['selected_clip_ids'],
            original_groups=original_groups,responses=responses,reference_summary=reference_summary)
    if result['models']['neutral']['selected_clip_ids']!=result['models']['native']['selected_clip_ids']:
        raise ValueError('Compared samples differ')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh summary required')
    result=summarize(a.root)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    for name,r in result['models'].items():
        print(json.dumps(dict(model=name,neutral_jaw=r['original_groups']['clip_all/0'],
            reference_summary=r['reference_summary'],
            intervention_jaw={k:v['response']['jaw'] for k,v in r['responses'].items() if k.startswith('raw/')})))
