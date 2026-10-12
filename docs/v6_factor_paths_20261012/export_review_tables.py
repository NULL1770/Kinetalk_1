"""Completed fixed-final stages only; include failures of the modeling hypothesis."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent;PROJECT=ROOT.parents[1];EVIDENCE=ROOT/'evidence'
METRICS=dict(lip_mean_mm='lve_mean_mm_mean',vertex_LVE_mm='vertex_lve_sqrt_mean',
    upper_face_EVE_mm='eye_forehead_eve_sqrt_mean',closure_F1='jaw_closure_frame_f1',
    jaw_correlation='jawOpen/centered_correlation',MBE='arkit_mbe',LBE='arkit_lbe')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def report(folder):
    if not (folder/'complete.json').exists():raise ValueError('not completed')
    if (folder/'report.json').exists():return read(folder/'report.json')
    files=[p for p in folder.glob('*.json') if p.name not in ('complete.json','state.json')]
    if len(files)!=1:raise ValueError('ambiguous report')
    return read(files[0])


def main():
    completed=[];sources=[]
    for arm in ('direct_seed47','posterior_seed47','direct_seed91','posterior_seed91'):
        path=EVIDENCE/'evaluations'/(arm+'_full1367')/'report.json'
        if path.exists():
            r=read(path)
            if r['clips']!=1367 or r['test_loaded']:raise ValueError('scope differs')
            if sha(EVIDENCE/'runs'/arm/'final.pt')!=r['checkpoint_sha256']:raise ValueError('checkpoint differs')
            completed.append((arm,r));sources.append(path)
    if not completed:raise ValueError('no completed full evaluation')
    old_path=PROJECT/'docs/v5_optimization_20261011/dual_evidence/evaluations/control_full1367/report.json'
    old=read(old_path);sources.append(old_path)
    full=[];probes=[]
    for arm,r in [('v5_control',old)]+completed:
        if r['indices']!=old['indices'] or r['baseline_sha256']!=old['baseline_sha256']:
            raise ValueError('query/baseline differs')
        for key,v in r['results'].items():
            if arm!='v5_control' and key.startswith('phase53'):continue
            mode,policy=key.split('/');label=arm if mode.startswith('v4_') else mode
            p=v['probes'][0]
            full.append(dict(arm=label,mode=mode,policy=policy,clips=1367,
                **{n:v['metrics'][k] for n,k in METRICS.items()},
                emotion_F1=p['macro_f1'],angry_F1=p['per_class']['angry']['f1'],
                checkpoint_sha256=r['checkpoint_sha256']))
            for ordinal,probe in enumerate(v['probes']):
                for emotion,score in probe['per_class'].items():
                    probes.append(dict(arm=label,mode=mode,policy=policy,probe=ordinal+1,
                        emotion=emotion,macro_F1=probe['macro_f1'],**score))
    write_csv('full_development.csv',full);write_csv('emotion_probes.csv',probes)
    dynamic=[];geometry=[];style=[];controls=[]
    for arm,_ in completed:
        for kind in ('dynamics128','geometry128','style64','controls32'):
            folder=EVIDENCE/'evaluations'/(arm+'_'+kind)
            if not (folder/'complete.json').exists():continue
            r=report(folder)
            if r['test_loaded']:raise ValueError('test data')
            sources.extend(folder.glob('*.json'))
            if kind=='dynamics128':
                for mode,v in r['results'].items():
                    direction=v.get('expression_direction')
                    if direction:dynamic.append(dict(arm=arm,mode=mode,clips=r['clips'],
                        mean_axis_MAE=direction['mean_axis_mae'],cosine_error=direction['direction_cosine_error']))
            elif kind=='geometry128':
                for emotion,v in r['summary'].items():
                    for stage,axes in v.items():
                        for axis,score in axes.items():geometry.append(dict(arm=arm,emotion=emotion,stage=stage,axis=axis,**score))
            elif kind=='style64':
                for policy,v in r['summary'].items():
                    for metric,value in v.items():style.append(dict(arm=arm,policy=policy,pairs=r['matched_pairs'],metric=metric,value=value))
            elif kind=='controls32':
                for intervention,v in r['summary'].items():
                    for metric,score in v['content_proxies'].items():controls.append(dict(arm=arm,intervention=intervention,metric=metric,**score))
    for name,rows in [('dynamics.csv',dynamic),('geometry.csv',geometry),('style.csv',style),('controls.csv',controls)]:write_csv(name,rows)
    provenance=dict(schema='v6_completed_fixed_final_tables_v1',completed_arms=[a for a,_ in completed],
        source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in sources},
        test_loaded=False,promotion=False,limits=[
            'v6 new decoder trained768 updates; v5 comparator continues an older jointly trained decoder256 updates. Cross-family differences are descriptive.',
            'Phase53 train budget/reference count differs; report one-reference Phase53 separately.',
            'No checkpoint selection or model promotion; incomplete arms omitted and recorded.',
            'Whole-clip motion probes and shared weak event parser are auxiliary evidence, not psychological temporal truth.'])
    (ROOT/'table_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8',newline='\n')
    print(json.dumps(dict(completed=provenance['completed_arms'],sample_clip_rows=[r for r in full if r['mode'] in ('v4_sample','phase53_default') and r['policy']=='clip_all'])))


if __name__=='__main__':main()
