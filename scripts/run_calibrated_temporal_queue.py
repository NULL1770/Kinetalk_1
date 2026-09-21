"""Gated calibrated full model and paired temporal upper-face experiment."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_formal_predictable_projection import save_json
from scripts.extract_emotion2vec_pilot import sha
from scripts.compact_native_curves import persist_native_curves


def persist_stage_artifacts(source,destination,length_by_id):
    destination.mkdir(parents=True,exist_ok=True)
    record=persist_native_curves(source/'curves.pt',destination/'native_curves.pt',length_by_id)
    record['reports']={}
    for path in source.glob('*.json'):
        target=destination/path.name
        shutil.copy2(path,target)
        if sha(path)!=sha(target):raise IOError('Artifact report backup differs')
        record['reports'][path.name]=sha(target)
    save_json(destination/'persistence.json',record)
    return record


def _record_ids(rows):
    if not isinstance(rows,list) or not rows:
        raise ValueError('Paired records must be a nonempty list')
    if any(not isinstance(row,dict) or not isinstance(row.get('clip_id'),str) or not row['clip_id'] for row in rows):
        raise ValueError('Every paired record requires a nonempty clip_id')
    ids=[row['clip_id'] for row in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate paired clip_id')
    return set(ids)


def interval(left,right,path):
    left_ids,right_ids=_record_ids(left),_record_ids(right)
    other={x['clip_id']:x for x in right};groups={}
    if left_ids!=right_ids:raise ValueError('Paired clip membership differs')
    for row in left:
        b=other[row['clip_id']]
        if not isinstance(row.get('sentence'),str) or not row['sentence']:raise ValueError('Missing paired sentence cluster')
        scales=row.get('scales')
        if (not isinstance(scales,list) or len(scales)!=9 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in scales)):
            raise ValueError('Paired scales must be nine finite positive values')
        if row['sentence']!=b['sentence'] or row['scales']!=b['scales']:raise ValueError('Paired metadata differs')
        x,y=row,b
        try:
            for key in path:x,y=x[key],y[key]
        except (KeyError,TypeError) as exc:raise ValueError('Missing comparison metric: '+'.'.join(path)) from exc
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,y)):
            raise ValueError('Comparison metrics must be finite numbers')
        delta=x-y
        if not math.isfinite(delta):raise ValueError('Nonfinite paired metric difference')
        groups.setdefault(row['sentence'],[]).append(delta)
    means=np.array([np.mean(v) for v in groups.values()]);n=np.array([len(v) for v in groups.values()])
    ix=np.random.default_rng(20260921).integers(len(n),size=(2000,len(n)))
    boot=(means[ix]*n[ix]).sum(1)/n[ix].sum(1)
    if not np.isfinite(boot).all():raise ValueError('Nonfinite paired bootstrap statistic')
    ci=np.quantile(boot,[.025,.975]).tolist()
    return {'audio_minus_control':float((means*n).sum()/n.sum()),'sentence_cluster_95ci':ci,
        'clusters':len(n),'clips':int(n.sum()),'passed':bool(len(n)>1 and ci[1]<0)}


def assess(output,artifacts,*,expected_ids=None):
    def read(path):return json.loads(path.read_text())
    if expected_ids is None:
        expected_ids=read(output/'queue_plan.json').get('validation_clip_ids')
    if (not isinstance(expected_ids,list) or not expected_ids
            or any(not isinstance(cid,str) or not cid for cid in expected_ids)
            or len(set(expected_ids))!=len(expected_ids)):
        raise ValueError('Acceptance requires unique validation clip IDs bound before training')
    expected=set(expected_ids)
    real=read(artifacts/'audio/dynamics/temporal_full.json')
    controls={'trained_static':read(artifacts/'static/dynamics/temporal_full.json'),
        'same_static':read(artifacts/'audio/dynamics/temporal_static_state.json'),
        'reverse_conditions':read(artifacts/'audio/dynamics/temporal_reverse_audio.json')}
    for name,rows in {'audio':real,**controls}.items():
        if _record_ids(rows)!=expected:raise ValueError(name+' does not cover the complete declared validation partition')
        for row in rows:
            if row.get('sample_count')!=3:raise ValueError(name+' requires exactly three generated draws per clip')
            if type(row.get('valid_frames')) is not int or row['valid_frames']<2:
                raise ValueError(name+' has insufficient native frame support')
    source={row['clip_id']:row for row in real}
    for name,rows in controls.items():
        for row in rows:
            base=source[row['clip_id']]
            if any(row.get(key)!=base.get(key) for key in ('valid_frames','valid_runs','sample_count')):
                raise ValueError(name+' scoring frame/draw support differs')
    paired={name:{'centered_es':interval(real,rows,['joint_fair_es','centered']),
        'variogram':interval(real,rows,['variogram','aggregate'])} for name,rows in controls.items()}
    benchmark={name:read(output/name/'dynamics/benchmark_summary.json') for name in ('audio','static')}
    mouth={name:read(output/name/'dynamics/mouth_protection.json') for name in ('audio','static')}
    modes=('full','base','static_state','oracle_state','reverse_audio');seeds=(42,123,2026)
    expected_keys={f'{seed}/{mode}' for seed in seeds for mode in modes}
    for arm in ('audio','static'):
        if not isinstance(mouth[arm],dict) or set(mouth[arm])!=expected_keys:
            raise ValueError(arm+' mouth protection must cover every declared seed and mode')
        if any(not isinstance(g,dict) or type(g.get('passed')) is not bool for g in mouth[arm].values()):
            raise ValueError(arm+' mouth protection verdicts must be Boolean')
        for mode in modes:
            record=benchmark[arm].get(mode)
            if not isinstance(record,dict) or record.get('prediction_keys')!=[f'{seed}/{mode}' for seed in seeds]:
                raise ValueError(arm+' benchmark draw binding differs for '+mode)
    checks={'paired_centered_es':all(x['centered_es']['passed'] for x in paired.values()),
        'paired_variogram':all(x['variogram']['passed'] for x in paired.values()),
        'mouth_preserved':all(g['passed'] is True for arm in mouth.values() for g in arm.values())}
    result={'schema':'calibrated_temporal_acceptance_v1','checks':checks,'quantitative_timing_passed':all(checks.values()),
        'paired':paired,'benchmark':benchmark,'test_loaded':False,'default_replaced':False,
        'visual_review_pending':True,'independent_emotion_and_AV_pending':True,
        'validation_clips':len(expected),'scope':'complete declared development queries; timing and prior quality separately assessed',
        'caution':'Passing numeric timing checks alone does not certify naturalness or paper readiness'}
    save_json(output/'acceptance.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','calibration','output','artifacts','smoke-output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=12);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    status=json.loads((a.calibration/'evaluation.json').read_text())
    if status.get('passed') is not True:raise ValueError('Reference-calibrated mouth gate must pass first')
    if a.output.exists():raise FileExistsError('Fresh queue output required')
    if shutil.disk_usage(a.output.parent).free<1_200_000_000:raise OSError('Need at least 1.2 GB persistent checkpoint and compact curve space')
    a.output.mkdir(parents=True);a.artifacts.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.data/'manifest.json').read_text())
    validation_ids=[row['clip_id'] for row in manifest['roles']['val']['query']]
    native_lengths={row['clip_id']:row['frames'] for row in manifest['roles']['val']['query']}
    if len(set(validation_ids))!=len(validation_ids) or not validation_ids:raise ValueError('Invalid validation membership in manifest')
    plan={'schema':'calibrated_temporal_queue_v1','calibration_sha256':sha(a.calibration/'final.pt'),
        'data_index_sha256':sha(a.data/'index.json'),'epochs':a.epochs,'test_loaded':False,
        'manifest_sha256':manifest['manifest_sha256'],'validation_clip_ids':validation_ids,
        'steps':['real CUDA smoke identity/teacher/audio/temporal dynamics','identity200 teacher12 audio12',
                 'audio temporal dynamics','independent static temporal dynamics','paired acceptance'],
        'artifacts':str(a.artifacts),'checkpoint_output':str(a.output)}
    plan['durable_artifacts']=str(a.output/'native_artifacts')
    plan['artifact_policy']='lossless native-frame values persisted after each completed arm; padded originals may be volatile'
    save_json(a.output/'queue_plan.json',plan);records=[];started=time.monotonic()
    common=[sys.executable,'-u',str(root/'scripts/train_full_staged.py'),'--paper-data',str(a.data),
        '--protect-mouth','--compact','--seed','47','--batch-size','16','--epochs',str(a.epochs)]
    jobs=[('smoke',common+['--output',str(a.smoke_output),'--smoke','--temporal-upper',
            '--start-stage','identity','--stage-checkpoint',str(a.calibration/'final.pt')]),
          ('protected',common+['--output',str(a.output/'protected'),'--artifact-dir',str(a.artifacts/'protected'),
            '--start-stage','identity','--end-stage','audio','--identity-epochs','200','--stage-checkpoint',str(a.calibration/'final.pt')])]
    for arm in ('audio','static'):
        jobs.append((arm,common+['--output',str(a.output/arm),'--artifact-dir',str(a.artifacts/arm),
            '--temporal-upper','--start-stage','dynamics','--condition-mode',arm,
            '--stage-checkpoint',str(a.output/'protected/audio/final.pt')]))
    try:
        for name,cmd in jobs:
            with (a.output/(name+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                save_json(a.output/'queue_status.json',{'status':'running','stage':name,'pid':child.pid,'command':cmd,
                    'started_utc':datetime.now(timezone.utc).isoformat(),'test_loaded':False})
                rc=child.wait()
            records.append({'stage':name,'exit_code':rc});save_json(a.output/'process_records.json',records)
            if rc:raise RuntimeError(name+' stopped: see log/quality gate; exit '+str(rc))
            if name!='smoke':
                stage='audio' if name=='protected' else 'dynamics'
                persist_stage_artifacts(a.artifacts/name/stage,a.output/'native_artifacts'/name/stage,native_lengths)
        result=assess(a.output,a.output/'native_artifacts')
        save_json(a.output/'queue_status.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,
            'quantitative_timing_passed':result['quantitative_timing_passed'],'default_replaced':False,'test_loaded':False,
            'artifacts':str(a.output/'native_artifacts'),'review_pending':True})
    except BaseException as exc:
        save_json(a.output/'queue_status.json',{'status':'failed','exception':repr(exc),'records':records,
            'elapsed_seconds':time.monotonic()-started,'test_loaded':False});raise


if __name__=='__main__':main()
