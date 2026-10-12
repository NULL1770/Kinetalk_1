"""Equal-scope posture/privileged-teacher evidence; preserve all output policies."""
from pathlib import Path
import csv
import hashlib
import json
from export_control_dynamics_tables import export

ROOT=Path(__file__).resolve().parent
ARMS=('posture_control','posture_direct','posture_privileged')
METRICS=dict(lip_mean_distance_mm='lve_mean_mm_mean',vertex_LVE_sqrt_mean_mm='vertex_lve_sqrt_mean',
    upper_face_EVE_mm='eye_forehead_eve_sqrt_mean',closure_proxy_F1='jaw_closure_frame_f1',
    MBE='arkit_mbe',LBE='arkit_lbe',jaw_correlation='jawOpen/centered_correlation',
    brows_centered_MSE='brows/centered_mse',brows_posture_bias_MSE='brows/mean_bias_mse')


def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))


def main():
    required=[arm+suffix for arm in ARMS for suffix in ('_train_complete.json','_full_report.json',
        '_subset_report.json','_state_report.json','_dynamics.json','_factor_controls.json')]
    missing=[name for name in required if not (ROOT/name).is_file() or not (ROOT/name).stat().st_size]
    if missing:
        print(json.dumps(dict(status='pending',missing=missing)));return
    receipts={arm:read(arm+'_train_complete.json') for arm in ARMS}
    first=receipts[ARMS[0]]
    fields=('source_checkpoint_sha256','training_references_sha256','fit_indices_sha256','binding_sha256')
    for arm,r in receipts.items():
        if r['updates']!=256 or r['frozen_b0_sha256']!=first['frozen_b0_sha256'] or r['protocol']['test_loaded']:
            raise ValueError('Posture budget/B0/data scope differs')
        if any(r['protocol'][k]!=first['protocol'][k] for k in fields):raise ValueError('Posture reference/fit/source differs')
        if r['mead_training_sequence_sha256']!=first['mead_training_sequence_sha256']:
            raise ValueError('Posture training order differs')
    limits=['Direct vs privileged isolates supported pooled teacher guidance.',
        'Control vs direct jointly changes new head, posture objective and removal of framewise guidance.',
        'One continuation training seed; three event sampling seeds are not three training seeds.',
        'Geometry and whole-clip emotion probes do not measure psychological framewise emotion truth.',
        'Scores are development-only; no novelty, statistical significance, style accuracy or promotion claim.']
    for scope in ('full','subset'):
        names={arm:arm+'_'+scope+'_report.json' for arm in ARMS}
        names['source_MEAD']='mead_'+('full_report.json' if scope=='full' else 'matched_report.json')
        reports={arm:read(name) for arm,name in names.items()};f=reports[ARMS[0]]
        if any(r['indices']!=f['indices'] or r['baseline_sha256']!=f['baseline_sha256'] or r['test_loaded']
               for r in reports.values()):raise ValueError('Development query/baseline scope differs')
        rows=[]
        for arm,r in [('Phase53_default',f),('Phase53_reference_A',f)]+list(reports.items()):
            prefix='phase53_default' if arm=='Phase53_default' else 'phase53_reference_A' if arm=='Phase53_reference_A' else None
            for mode in ([prefix] if prefix else ['v4_sample','v4_modal']):
                for policy in ('raw','clip_all'):
                    v=r['results'][mode+'/'+policy];p=v['probes'][0]
                    rows.append(dict(arm=arm,mode=mode,policy=policy,clips=r['clips'],
                        **{k:v['metrics'][key] for k,key in METRICS.items()},emotion_probe_F1=p['macro_f1'],
                        angry_probe_F1=p['per_class']['angry']['f1']))
        export('posture_comparison_'+scope+'.csv',rows,names.values(),limits)
    state_rows=[];direction_rows=[];region_rows=[];event_rows=[];style_rows=[]
    for arm in ARMS:
        state=read(arm+'_state_report.json')
        for mode,values in state['summary'].items():
            for axis,name in enumerate(state['coordinate_names']):
                state_rows.append(dict(arm=arm,mode=mode,axis=name,clips=state['clips'],
                    bias_MSE=values['bias_mse'][axis],centered_MSE=values['centered_mse'][axis],
                    target_centered_variance=values['target_centered_variance'][axis]))
        dynamic=read(arm+'_dynamics.json')
        for mode,values in dynamic['results'].items():
            if 'expression_direction' in values:
                direction_rows.append(dict(arm=arm,mode=mode,clips=dynamic['clips'],
                    state_MAE=values['expression_direction']['mean_axis_mae'],
                    direction_cosine_error=values['expression_direction']['direction_cosine_error']))
            for region,v in values['regional_dynamics_macro_clip'].items():
                region_rows.append(dict(arm=arm,mode=mode,region=region,**v))
            for event,v in values['realized_event_distributions'].items():
                event_rows.append(dict(arm=arm,mode=mode,event=event,
                    rate_per_minute=v['complete_events_per_minute'],rate_absolute_error=v.get('complete_events_per_minute_abs_error'),
                    duration_mean_seconds=v['duration_seconds_mean'],amplitude_mean=v['amplitude_mean'],
                    duration_wasserstein1=v.get('duration_seconds_wasserstein1'),amplitude_wasserstein1=v.get('amplitude_wasserstein1')))
        control=read(arm+'_factor_controls.json')
        for condition in ('same_person_B','donor','same_person_B_fixed_schedule','donor_fixed_schedule'):
            v=control['summary'][condition]['content_proxies']
            style_rows.append(dict(arm=arm,condition=condition,clips=control['clips'],
                closure_proxy_flip_fraction=v['jaw_closure_proxy_flip_fraction_at_05']['mean'],
                jaw_derivative_correlation=v['jaw_derivative_correlation']['mean'],
                jaw_range_ratio=v['jaw_range_ratio']['mean']))
    for filename,rows,suffix in (
        ('posture_state_decomposition.csv',state_rows,'_state_report.json'),
        ('posture_expression_direction.csv',direction_rows,'_dynamics.json'),
        ('posture_regional_dynamics.csv',region_rows,'_dynamics.json'),
        ('posture_event_distributions.csv',event_rows,'_dynamics.json'),
        ('posture_reference_sensitivity.csv',style_rows,'_factor_controls.json')):
        export(filename,rows,[arm+suffix for arm in ARMS],limits+[
            'Teacher coordinate rows are privileged GT-motion oracles, not deployment results.',
            'Closure threshold flips under style changes are sensitivity, not phoneme mistakes.',
            'Event rates/geometry use weak ARKit descriptions, not physiological or psychological truth.'])
    (ROOT/'posture_matching_receipt.json').write_text(json.dumps(dict(status='matched',updates=256,
        fields=fields,source_checkpoint=first['protocol']['source_checkpoint_sha256'],
        sequence=first['mead_training_sequence_sha256'],B0=first['frozen_b0_sha256'],
        files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in required},
        test_loaded=False,novelty_verified=False,promotion=False),indent=2))


if __name__=='__main__':main()
