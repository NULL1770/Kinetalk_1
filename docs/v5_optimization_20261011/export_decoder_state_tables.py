"""Export matched state-supervision evidence; no model selection or test reads."""
from pathlib import Path
import csv
import hashlib
import json
from export_posture_tables import METRICS

ROOT=Path(__file__).resolve().parent
ARMS=('state_raw_direct','state_decoder_direct','state_decoder_privileged')


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf8'))


def export(name,rows):
    if not rows: raise ValueError('Empty table: '+name)
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    receipts={arm:read(arm+'_train_complete.json') for arm in ARMS}
    first=receipts[ARMS[0]]
    match_fields=('source_checkpoint_sha256','binding_sha256','training_references_sha256',
                  'fit_indices_sha256','common_initial_parameters_sha256')
    for arm,r in receipts.items():
        p=r['protocol']
        if r['updates']!=256 or p['seed']!=47 or p['test_loaded'] or p['hdtf_windows']:
            raise ValueError('Unexpected training budget/data: '+arm)
        if any(p[k]!=first['protocol'][k] for k in match_fields): raise ValueError('Unmatched source: '+arm)
        if r['mead_training_sequence_sha256']!=first['mead_training_sequence_sha256'] or r['frozen_b0_sha256']!=first['frozen_b0_sha256']:
            raise ValueError('Unmatched B0/order: '+arm)
    reports={arm:read(arm+'_full_eval_report.json') for arm in ARMS}
    f=reports[ARMS[0]]
    if f['clips']!=1367: raise ValueError('Incomplete development table')
    rows=[]
    for arm,d in reports.items():
        if d['indices']!=f['indices'] or d['baseline_sha256']!=f['baseline_sha256'] or d['test_loaded']:
            raise ValueError('Unmatched query/baseline: '+arm)
        if d['checkpoint_sha256']!=read(arm+'_full_eval_complete.json')['checkpoint_sha256']:
            raise ValueError('Report/complete binding mismatch')
        for mode in ('v4_sample','v4_modal'):
            for policy in ('raw','clip_all'):
                v=d['results'][mode+'/'+policy]; p=v['probes'][0]
                rows.append(dict(arm=arm,mode=mode,policy=policy,clips=d['clips'],
                    checkpoint_sha256=d['checkpoint_sha256'],
                    **{k:v['metrics'][key] for k,key in METRICS.items()},
                    emotion_probe_F1=p['macro_f1'],angry_probe_F1=p['per_class']['angry']['f1']))
    for prefix in ('phase53_default','phase53_reference_A'):
        for policy in ('raw','clip_all'):
            v=f['results'][prefix+'/'+policy]; p=v['probes'][0]
            rows.append(dict(arm=prefix,mode=prefix,policy=policy,clips=f['clips'],checkpoint_sha256=f['baseline_sha256'],
                **{k:v['metrics'][key] for k,key in METRICS.items()},emotion_probe_F1=p['macro_f1'],angry_probe_F1=p['per_class']['angry']['f1']))
    export('decoder_state_full_development.csv',rows)
    state_rows=[]; dynamic_rows=[]; geometry_rows=[]
    selected=None
    for arm in ARMS:
        d=read(arm+'_state_report.json')
        if selected is None: selected=d['indices']
        if d['indices']!=selected or d['test_loaded']: raise ValueError('Unmatched state subset')
        for mode,v in d['summary'].items():
            state_rows.append(dict(arm=arm,mode=mode,clips=d['clips'],macro_MSE=v['macro_mse'],
                mean_bias_MSE=sum(v['bias_mse'])/len(v['bias_mse']),
                centered_MSE=sum(v['centered_mse'])/len(v['centered_mse']),
                oracle=mode.startswith('teacher_oracle')))
        dyn=read(arm+'_dynamics_report.json')
        if dyn['selected_indices']!=selected or dyn['test_loaded']: raise ValueError('Unmatched dynamics subset')
        for mode,v in dyn['results'].items():
            direction=v.get('expression_direction',{})
            for event,e in v['realized_event_distributions'].items():
                dynamic_rows.append(dict(arm=arm,mode=mode,event=event,clips=dyn['clips'],
                    direction_MAE=direction.get('mean_axis_mae'),cosine_error=direction.get('direction_cosine_error'),
                    event_rate_per_minute=e['complete_events_per_minute'],
                    rate_abs_error=e.get('complete_events_per_minute_abs_error'),
                    duration_W1=e.get('duration_seconds_wasserstein1'),amplitude_W1=e.get('amplitude_wasserstein1')))
        g=read(arm+'_geometry_report.json')
        if g['indices']!=selected or g['test_loaded']: raise ValueError('Unmatched geometry subset')
        for emotion in ('ALL','angry','happy'):
            for stage,values in g['summary'][emotion].items():
                for axis,v in values.items():
                    geometry_rows.append(dict(arm=arm,emotion=emotion,stage=stage,coordinate=axis,
                        MAE_mm=v['MAE'],bias_mm=v['bias'],centered_MSE=v['centered_MSE'],supported_clips=v['supported_clips']))
    export('decoder_state_coordinates.csv',state_rows)
    export('decoder_state_dynamics.csv',dynamic_rows)
    export('decoder_state_geometry.csv',geometry_rows)
    sources=[arm+suffix for arm in ARMS for suffix in ('_train_complete.json','_full_eval_report.json','_state_report.json','_dynamics_report.json','_geometry_report.json')]
    (ROOT/'decoder_state_tables_provenance.json').write_text(json.dumps(dict(schema='decoder_state_tables_v1',
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
        matched_fields=match_fields,training_seeds=[47],sampling_seeds=[47,91,137],test_loaded=False,promotion=False,
        limits=['One continuation seed; sampling seeds do not provide training replication.',
                'Raw-to-decoder contrast changes supervision coordinates, not architecture.',
                'Direct-to-privileged contrast isolates accepted pooled guidance.',
                'Motion probes, event parser and jaw closure are proxies, not psychological dynamics/phonetic truth.']),indent=2),encoding='utf8')
    print(json.dumps(dict(status='exported',full_rows=len(rows),state_rows=len(state_rows),dynamic_rows=len(dynamic_rows),geometry_rows=len(geometry_rows))))


if __name__=='__main__':
    main()
