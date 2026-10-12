"""Source-bound matched-pilot tables, retaining raw/clipped/modal and every seed."""
from pathlib import Path
import csv
import hashlib
import json
from export_posture_tables import METRICS

ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/'dual_evidence'
ARMS=('control','direct','guided')


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def report(arm,kind):
    folder=EVIDENCE/'evaluations'/(arm+'_'+kind)
    if not (folder/'complete.json').is_file():raise ValueError('Incomplete '+str(folder))
    files=([folder/'report.json'] if (folder/'report.json').is_file() else
           [p for p in folder.glob('*.json') if p.name not in ('complete.json','state.json')])
    if len(files)!=1:raise ValueError('Ambiguous '+str(folder))
    return read(files[0])
def export(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    receipts={a:read(EVIDENCE/'runs'/a/'complete.json') for a in ARMS}
    fields=('source_checkpoint_sha256','binding_sha256','fit_indices_sha256',
            'training_references_sha256','initial_parent_parameters_sha256')
    first=receipts['control']
    for a,r in receipts.items():
        if r['updates']!=256 or r['protocol']['seed']!=47 or r['protocol']['test_loaded']:
            raise ValueError('Budget/scope differs')
        if any(r['protocol'][k]!=first['protocol'][k] for k in fields):raise ValueError('Unmatched '+a)
        if r['mead_training_sequence_sha256']!=first['mead_training_sequence_sha256'] or r['frozen_b0_sha256']!=first['frozen_b0_sha256']:
            raise ValueError('Order/B0 differs')
        if sha(EVIDENCE/'runs'/a/'final.pt')!=r['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    full={a:report(a,'full1367') for a in ARMS};f=full['control']
    rows=[];probe_rows=[]
    for a,d in full.items():
        if d['clips']!=1367 or d['indices']!=f['indices'] or d['baseline_sha256']!=f['baseline_sha256'] or d['test_loaded']:
            raise ValueError('Development scope differs')
        if d['checkpoint_sha256']!=receipts[a]['checkpoint_sha256']:raise ValueError('Report binding differs')
        keys=[k for k in d['results'] if k.startswith('v4_') or a=='control']
        for key in keys:
            mode,policy=key.split('/');v=d['results'][key];p=v['probes'][0]
            label=a if mode.startswith('v4_') else mode
            rows.append(dict(arm=label,mode=mode,policy=policy,clips=1367,
                **{k:v['metrics'][old] for k,old in METRICS.items()},
                emotion_probe_F1=p['macro_f1'],angry_probe_F1=p['per_class']['angry']['f1']))
            for n,probe in enumerate(v['probes']):
                for emotion,e in probe['per_class'].items():
                    probe_rows.append(dict(arm=label,mode=mode,policy=policy,probe=n+1,
                        macro_F1=probe['macro_f1'],emotion=emotion,**e))
    export('dual_full_development.csv',rows);export('dual_emotion_probes.csv',probe_rows)
    dyn=[];region=[];geometry=[];style=[];controls=[]
    subsets={}
    for a in ARMS:
        d=report(a,'dynamics128');subsets[a]=d['selected_indices']
        if d['selected_indices']!=subsets['control'] or d['test_loaded']:raise ValueError('Dynamics scope differs')
        for mode,v in d['results'].items():
            direction=v.get('expression_direction',{})
            for event,e in v['realized_event_distributions'].items():
                execution=v.get('schedule_execution_consistency',{}).get(event,{})
                dyn.append(dict(arm=a,mode=mode,event=event,clips=d['clips'],
                    direction_MAE=direction.get('mean_axis_mae'),cosine_error=direction.get('direction_cosine_error'),
                    **e,execution_F1=execution.get('onset_f1')))
            for name,e in v['regional_dynamics_macro_clip'].items():region.append(dict(arm=a,mode=mode,region=name,**e))
        g=report(a,'geometry128')
        if g['indices']!=subsets['control'] or g['test_loaded']:raise ValueError('Geometry scope differs')
        for emotion,v in g['summary'].items():
            for stage,axes in v.items():
                for axis,e in axes.items():geometry.append(dict(arm=a,emotion=emotion,stage=stage,axis=axis,**e))
        s=report(a,'style64')
        if a=='control':style_selection=s['selection']
        if s['selection']!=style_selection or s['test_loaded']:raise ValueError('Style scope differs')
        for policy,v in s['summary'].items():
            for metric,value in v.items():style.append(dict(arm=a,policy=policy,pairs=s['matched_pairs'],metric=metric,value=value))
        c=report(a,'controls32')
        for intervention,v in c['summary'].items():
            for metric,e in v['content_proxies'].items():
                controls.append(dict(arm=a,intervention=intervention,metric=metric,**e))
    for name,values in [('dual_dynamics.csv',dyn),('dual_regions.csv',region),('dual_geometry.csv',geometry),
                         ('dual_style.csv',style),('dual_controls.csv',controls)]:export(name,values)
    provenance=dict(schema='dual_timescale_tables_v1',matched_fields=fields,
        checkpoint_sha256={a:r['checkpoint_sha256'] for a,r in receipts.items()},
        source_manifest_sha256=sha(EVIDENCE/'source_manifest.json'),
        collection_receipt_sha256=sha(EVIDENCE/'collection_receipt.json'),
        training_seeds=[47],sampling_seeds=[47,91,137],development_clips=1367,
        dynamic_clips=128,style_pairs=64,test_loaded=False,promotion=False,
        limits=['Same-budget continuation; added modules show negligible mouth effect.',
                'Sampling seeds are not independent training seeds.',
                'Shared weak event parser and whole-clip probes are not independent emotion timing truth.',
                'Three development speakers limit style generalization.',
                'Baseline Phase53 training budget differs; its differences are descriptive.'])
    (ROOT/'dual_tables_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print(json.dumps(dict(status='exported',full_rows=len(rows),dynamic_rows=len(dyn),geometry_rows=len(geometry))))


if __name__=='__main__':main()
