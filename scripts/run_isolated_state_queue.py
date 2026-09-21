"""Run independent state repair, gate, then frozen-state paired residuals."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_formal_predictable_projection import save_json,canonical_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.train_isolated_audio_state import paired_ci
from scripts.run_calibrated_temporal_queue import interval
from scripts.train_isolated_residual import acceptance_checks,SEEDS,MODES
from scripts.prepare_paper_full_data import validate_manifest


STATE_CHECKS={'mean_no_worse','mouth_preserved','mbe_no_worse','internal_emotion_no_large_drop',
              'train_state_fit','val_state_better_than_static','val_state_better_than_reverse'}
RESIDUAL_CHECKS={'mouth_preserved','nonupper43_exact','upper_mean_preserved',
                 'mbe_no_worse_than_stage4','teacher_drop_at_most_3pp','frozen_models_unchanged'}


def exact_checks(checks,names,label):
    if not isinstance(checks,dict) or set(checks)!=names or any(type(v) is not bool for v in checks.values()):
        raise ValueError(label+' gate schema mismatch')


def source_inventory(root):
    paths=sorted({*root.joinpath('scripts').glob('*.py'),*root.joinpath('kinetalk_b0').rglob('*.py')})
    return {p.relative_to(root).as_posix():sha(p) for p in paths}


def validate_stage_complete(out,name,plan,code_root):
    """Bind completed work to the recipe before either accepting or skipping it."""
    read=lambda p:json.loads(p.read_text())
    status=read(out/'status.json');protocol=read(out/'protocol.json')
    kind,arm=name.split('_',1)
    if status.get('status')!='complete' or status.get('test_loaded') is not False:
        raise ValueError('Stage is not complete and test-free: '+name)
    wanted_schema='isolated_mean_state_v1' if kind=='state' else 'isolated_dc_residual_training_v1'
    required={'schema':wanted_schema,'source_sha256':plan['source_sha256'],'mode':arm,
        'epochs':plan[kind+'_epochs'],'smoke':False,'test_loaded':False,'default_replaced':False,
        'train_clips':plan['train_clips'],'validation_clip_ids':plan['validation_ids'],
        'noise_seeds':list(SEEDS),'seed':47,'batch_size':16,'selection':'fixed final epoch'}
    if any(protocol.get(k)!=v for k,v in required.items()):raise ValueError('Completed stage protocol mismatch: '+name)
    if (protocol.get('data',{}).get('manifest_sha256')!=plan['manifest_sha256']
            or protocol.get('data',{}).get('index_sha256')!=plan['data_index_sha256']):
        raise ValueError('Completed stage data differs')
    sources=protocol.get('sources')
    if not isinstance(sources,dict) or not sources:raise ValueError('Missing stage source hashes')
    for source,digest in sources.items():
        if plan['sources'].get(source)!=digest or sha(code_root/source)!=digest or sha(out/'source'/source)!=digest:
            raise ValueError('Completed stage source changed: '+source)
    if kind=='residual':
        state_path=out.parent/'state_audio/final.pt'
        state_protocol=read(out.parent/'state_audio/protocol.json')
        if (protocol.get('state_checkpoint_sha256')!=sha(state_path)
                or protocol.get('state_protocol_sha256')!=canonical_hash(state_protocol)
                or protocol.get('evaluation_modes')!=list(MODES)):
            raise ValueError('Residual state backbone/evaluation binding differs')
    for filename,field in [('final.pt','final_sha256'),('native_curves.pt','native_curves_sha256')]:
        if status.get(field)!=sha(out/filename):raise ValueError('Completed artifact hash differs: '+filename)
    checkpoint=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    digest=canonical_hash(protocol)
    if checkpoint.get('protocol')!=protocol or checkpoint.get('protocol_sha256')!=digest or checkpoint.get('test_loaded') is not False:
        raise ValueError('Final checkpoint protocol mismatch')
    del checkpoint
    last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    if last.get('epoch')!=required['epochs'] or last.get('protocol_sha256')!=digest:
        raise ValueError('Last checkpoint incomplete/protocol mismatch')
    del last
    curves=torch.load(out/'native_curves.pt',map_location='cpu',weights_only=False)
    if (curves.get('compaction_schema')!='native_curve_pack_v1'
            or curves.get('clip_id')!=plan['validation_ids'] or curves.get('noise_seeds')!=list(SEEDS)):
        raise ValueError('Native curve membership/seed metadata differs')
    modes=('full','base','static','reverse') if kind=='state' else MODES
    expected_keys={f'{s}/{m}' for s in SEEDS for m in modes}
    if set(curves.get('predictions',{}))!=expected_keys:raise ValueError('Native curve seed/mode coverage differs')
    lengths=curves.get('native_lengths');expected_lengths=[plan['validation_metadata'][cid]['frames'] for cid in plan['validation_ids']]
    if lengths!=expected_lengths:raise ValueError('Native frame lengths differ')
    valid=curves['valid'];frames=sum(lengths)
    if valid.dtype!=torch.bool or valid.shape!=(frames,):raise ValueError('Native mask dimensions differ')
    if curves['target'].shape!=(frames,52) or curves['channel_mask'].shape!=(len(lengths),52):
        raise ValueError('Native coefficient/channel support differs')
    if any(v.shape!=(frames,52) or not torch.isfinite(v[valid]).all() for v in curves['predictions'].values()):
        raise ValueError('Incomplete/nonfinite prediction curves')
    support={};offset=0
    for cid,length in zip(plan['validation_ids'],lengths):
        mask=valid[offset:offset+length];offset+=length
        runs=int((mask & ~torch.cat((torch.zeros(1,dtype=torch.bool),mask[:-1]))).sum())
        edges=np.diff(np.r_[False,mask.numpy(),False].astype(np.int8))
        intervals=[[int(l),int(r)] for l,r in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
        count=int(mask.sum());meta=plan['validation_metadata'][cid]
        if count!=meta['valid_frames'] or count<2:raise ValueError('Manifest native valid frame support differs')
        support[cid]={'valid_frames':count,'valid_runs':runs,'sentence':meta['sentence'],
            'valid_intervals':intervals,'valid_sha256':canonical_hash(mask.tolist())}
    del curves
    report=read(out/'evaluation.json')
    if report.get('clips')!=len(plan['validation_ids']) or report.get('mode')!=arm or report.get('test_loaded') is not False:
        raise ValueError('Evaluation scope differs')
    reports={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*.json'))
             if 'source' not in p.relative_to(out).parts and p.name!='completion_verified.json'}
    record={'schema':'isolated_stage_completion_v1','stage':name,'protocol_sha256':digest,
        'final_sha256':status['final_sha256'],'native_curves_sha256':status['native_curves_sha256'],
        'reports':reports,'support':support}
    persisted=out/'completion_verified.json'
    if persisted.exists() and read(persisted)!=record:raise ValueError('Previously verified completed evidence changed')
    save_json(persisted,record)
    return support


def checked_rows(rows,expected,value='mse'):
    if not expected or len(set(expected))!=len(expected):raise ValueError('Invalid expected validation membership')
    if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError('Missing full validation rows')
    if any(not isinstance(x,dict) or not isinstance(x.get('clip_id'),str) for x in rows):raise ValueError('Invalid validation row')
    ids=[x['clip_id'] for x in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(expected):raise ValueError('Validation membership differs')
    if value is not None and any(type(x.get(value)) not in (int,float) or not np.isfinite(x[value]) for x in rows):raise ValueError('Nonfinite scoring row')
    return {x['clip_id']:x for x in rows}


def check_support(left,right,expected_support=None):
    for cid,x in left.items():
        y=right[cid]
        if not isinstance(x.get('sentence'),str) or not x['sentence'] or x['sentence']!=y.get('sentence'):
            raise ValueError('Pair sentence mismatch')
        if (type(x.get('valid_frames')) is not int or x['valid_frames']<2
                or x['valid_frames']!=y.get('valid_frames')):raise ValueError('Pair native support differs')
        runs=x.get('valid_runs')
        if runs!=y.get('valid_runs'):raise ValueError('Pair native support differs')
        if isinstance(runs,list):
            if (not runs or any(not isinstance(r,list) or len(r)!=2 or any(type(v) is not int for v in r)
                    or r[0]<0 or r[1]<=r[0] for r in runs)
                    or any(a[1]>=b[0] for a,b in zip(runs,runs[1:]))
                    or sum(r-l for l,r in runs)!=x['valid_frames']):raise ValueError('Invalid native intervals')
            count=len(runs)
            if expected_support is not None and runs!=expected_support[cid]['valid_intervals']:raise ValueError('Native mask intervals differ')
        elif type(runs) is int:count=runs
        else:raise ValueError('Invalid native run count')
        if count<1 or count>x['valid_frames']:raise ValueError('Insufficient native support')
        if expected_support is not None and (count!=expected_support[cid]['valid_runs']
                or x['valid_frames']!=expected_support[cid]['valid_frames']):raise ValueError('Native curve support differs')
        if expected_support is not None and x['sentence']!=expected_support[cid]['sentence']:raise ValueError('Manifest sentence differs')


def assess_state(root,expected,support=None):
    read=lambda p:json.loads(p.read_text())
    a=read(root/'state_audio/evaluation.json');s=read(root/'state_static/evaluation.json')
    left=checked_rows(a['per_clip_state_mse']['full'],expected);right=checked_rows(s['per_clip_state_mse']['full'],expected)
    check_support(left,right,support)
    delta=[];sentences=[]
    for cid in expected:
        x,y=left[cid],right[cid]
        if x['sentence']!=y['sentence']:raise ValueError('Pair sentence mismatch')
        delta.append(x['mse']-y['mse']);sentences.append(x['sentence'])
    paired=paired_ci(delta,sentences)
    exact_checks(a.get('checks'),STATE_CHECKS,'State')
    exact_checks(s.get('checks'),STATE_CHECKS,'Static state')
    if any(r.get('test_loaded') is not False or r.get('mode')!=arm or r.get('clips')!=len(expected)
           for arm,r in [('audio',a),('static',s)]):raise ValueError('State evaluation scope mismatch')
    passed=all(a['checks'].values()) and paired['passed']
    result={'checks':a['checks'],'independent_static':paired,'passed':passed,'test_loaded':False,'default_replaced':False}
    save_json(root/'state_acceptance.json',result);return result


def assess_residual(root,expected,support=None):
    read=lambda p:json.loads(p.read_text())
    a=root/'residual_audio';s=root/'residual_static'
    real=read(a/'scores/residual/temporal_full.json')
    controls={'independent_static':read(s/'scores/residual/temporal_full.json'),
        'same_static':read(a/'scores/residual/temporal_static.json'),
        'reverse':read(a/'scores/residual/temporal_reverse.json')}
    reference=checked_rows(real,expected,None)
    for rows in (real,*controls.values()):
        compared=checked_rows(rows,expected,None);check_support(reference,compared,support)
        if any(type(x.get('sample_count')) is not int or x['sample_count']!=3 for x in rows):raise ValueError('Three draws required')
    pairs={k:{'es':interval(real,v,['joint_fair_es','centered']),
              'variogram':interval(real,v,['variogram','aggregate'])} for k,v in controls.items()}
    reports={k:read(root/f'residual_{k}/evaluation.json') for k in ('audio','static')}
    local={k:r['checks'] for k,r in reports.items()}
    for arm,report in reports.items():
        exact_checks(local[arm],RESIDUAL_CHECKS,'Residual')
        if report.get('mode')!=arm or report.get('clips')!=len(expected) or report.get('test_loaded') is not False:
            raise ValueError('Residual evaluation scope mismatch')
        if report.get('protection_passed') is not all(local[arm].values()):raise ValueError('Residual protection verdict inconsistency')
        recomputed=acceptance_checks(report,read(root/f'residual_{arm}/benchmark_summary.json'))
        if any(local[arm][k]!=v for k,v in recomputed.items()):raise ValueError('Residual protection checks differ from metrics')
    passed=all(all(c.values()) for c in local.values()) and all(v['passed'] for x in pairs.values() for v in x.values())
    result={'paired':pairs,'local_checks':local,'quantitative_passed':passed,'visual_and_independent_emotion_AV_pending':True,
        'test_loaded':False,'default_replaced':False}
    save_json(root/'acceptance.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--state-epochs',type=int,default=100);p.add_argument('--residual-epochs',type=int,default=40)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.output.exists() and not a.resume:raise FileExistsError('Fresh queue required')
    if not a.output.exists() and shutil.disk_usage(a.output.parent).free<1_200_000_000:raise OSError('Need1.2GB checkpoint/native curves space')
    a.output.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[1]
    manifest=json.loads((a.data/'manifest.json').read_text());validate_manifest(manifest)
    ids=[x['clip_id'] for x in manifest['roles']['val']['query']]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Invalid complete validation membership')
    plan={'schema':'isolated_state_queue_v1','data_index_sha256':sha(a.data/'index.json'),
        'source_sha256':sha(a.source),'manifest_sha256':manifest['manifest_sha256'],'validation_ids':ids,
        'validation_metadata':{x['clip_id']:{k:x[k] for k in ('frames','valid_frames','sentence')} for x in manifest['roles']['val']['query']},
        'train_clips':len(manifest['roles']['train']['query']),'sources':source_inventory(root),
        'state_epochs':a.state_epochs,'residual_epochs':a.residual_epochs,'test_loaded':False,'default_replaced':False,
        'queue_sha256':sha(__file__),'start_utc':datetime.now(timezone.utc).isoformat()}
    if a.resume:
        prior=json.loads((a.output/'plan.json').read_text())
        if any(prior[k]!=v for k,v in plan.items() if k!='start_utc'):raise ValueError('Queue resume recipe mismatch')
    else:save_json(a.output/'plan.json',plan)
    jobs=[]
    for arm in ('audio','static'):
        jobs.append(('state_'+arm,[sys.executable,'-u',str(root/'scripts/train_isolated_audio_state.py'),
            '--data',str(a.data),'--source',str(a.source),'--output',str(a.output/('state_'+arm)),
            '--mode',arm,'--epochs',str(a.state_epochs)]))
    for arm in ('audio','static'):
        # Identical accepted deterministic backbone for both residual arms.
        jobs.append(('residual_'+arm,[sys.executable,'-u',str(root/'scripts/train_isolated_residual.py'),
            '--data',str(a.data),'--source',str(a.source),'--state-checkpoint',str(a.output/'state_audio/final.pt'),
            '--output',str(a.output/('residual_'+arm)),'--mode',arm,'--epochs',str(a.residual_epochs)]))
    records=[];started=time.monotonic();supports={}
    try:
        for name,cmd in jobs:
            if name=='residual_audio':
                if supports['state_audio']!=supports['state_static']:raise ValueError('State native masks differ across arms')
                gate=assess_state(a.output,ids,supports['state_audio'])
                if not gate['passed']:
                    save_json(a.output/'status.json',{'status':'stopped_quality_gate','gate':'state_acceptance.json',
                        'elapsed_seconds':time.monotonic()-started,'test_loaded':False,'default_replaced':False})
                    return
            out=a.output/name
            if out.exists():
                state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                if state.get('status')=='complete':
                    supports[name]=validate_stage_complete(out,name,plan,root)
                    records.append({'stage':name,'resumed_completed':True});continue
                if not a.resume:raise FileExistsError(out)
                cmd+=['--resume']
            with (a.output/(name+'.log')).open('a') as f:
                child=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
                save_json(a.output/'status.json',{'status':'running','stage':name,'pid':child.pid,'command':cmd,'test_loaded':False})
                rc=child.wait()
            records.append({'stage':name,'exit_code':rc});save_json(a.output/'records.json',records)
            if rc:raise RuntimeError(f'{name} failed, exit {rc}')
            supports[name]=validate_stage_complete(out,name,plan,root)
        if any(x!=supports['state_audio'] for x in supports.values()):raise ValueError('Native mask support differs across stages')
        result=assess_residual(a.output,ids,supports['state_audio'])
        save_json(a.output/'status.json',{'status':'complete','quantitative_passed':result['quantitative_passed'],
            'elapsed_seconds':time.monotonic()-started,'test_loaded':False,'default_replaced':False})
    except BaseException as e:
        save_json(a.output/'status.json',{'status':'failed','error':repr(e),'records':records,'test_loaded':False});raise


if __name__=='__main__':main()
