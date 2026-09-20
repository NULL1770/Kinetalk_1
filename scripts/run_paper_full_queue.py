"""Durable full-stage run, independent static control and paired evaluation.

Runs sequentially on one GPU, writes explicit exit codes, and never consumes
test assets. Existing successful stages are retained on queue resume.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_formal_predictable_projection import save_json
from scripts.extract_emotion2vec_pilot import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--identity-epochs',type=int,default=200);p.add_argument('--resume',action='store_true')
    a=p.parse_args();code=Path(__file__).resolve().parents[1]
    if a.output.exists() and not a.resume:raise FileExistsError('Fresh queue output required')
    a.output.mkdir(parents=True,exist_ok=True);a.artifacts.mkdir(parents=True,exist_ok=True)
    args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
    plan={'args':args,'data_index_sha256':sha(a.data/'index.json'),'runner_sha256':sha(__file__),
          'train_manifest_sha256':json.loads((a.data/'manifest.json').read_text())['manifest_sha256'],
          'order':['full five-stage audio model','independent static dynamics control','paired validation report'],
          'test_loaded':False,'default_replaced':False}
    planpath=a.output/'queue_plan.json'
    if planpath.exists() and json.loads(planpath.read_text())!=plan:raise ValueError('Queue resume plan differs')
    save_json(planpath,plan)
    started=time.time();records=[]
    try:
        for label in ('audio','static'):
            run=a.output/label;artifacts=a.artifacts/label
            summary=run/'summary.json'
            if summary.exists() and json.loads(summary.read_text()).get('status')=='complete':
                records.append({'arm':label,'status':'already_complete'});continue
            free=shutil.disk_usage(a.output).free
            # 5 stage checkpoints + one atomic resume checkpoint need <600 MB
            # with this model. Check actual available space before launching.
            required=650_000_000 if label=='audio' else 270_000_000
            if free<required:raise OSError(f'Insufficient persistent free bytes: {free} < {required}')
            cmd=[sys.executable,'-u',str(code/'scripts/train_full_staged.py'),'--paper-data',str(a.data),
                 '--output',str(run),'--artifact-dir',str(artifacts),'--compact',
                 '--epochs',str(a.epochs),'--identity-epochs',str(a.identity_epochs),'--seed','47',
                 '--condition-mode',label]
            if label=='static':cmd+=['--start-stage','dynamics','--stage-checkpoint',str(a.output/'audio/audio/final.pt')]
            if run.exists():
                if not a.resume or not (run/'last.pt').exists():raise FileExistsError('Incomplete existing run requires valid --resume')
                cmd+=['--resume']
            with (a.output/(label+'.log')).open('a') as log:
                process=subprocess.Popen(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT)
                save_json(a.output/'queue_status.json',{'status':'running','arm':label,'child_pid':process.pid,
                    'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat(),'test_loaded':False})
                rc=process.wait()
            records.append({'arm':label,'exit_code':rc,'command':cmd})
            save_json(a.output/'process_records.json',records)
            if rc:raise RuntimeError(f'{label} training failed with exit code {rc}; inspect {label}.log')
        compare(a.output,a.artifacts)
        save_json(a.output/'queue_status.json',{'status':'complete','elapsed_seconds':time.time()-started,
                    'results':'PAIRED_REPORT.md','test_loaded':False,'default_replaced':False})
    except BaseException as exc:
        save_json(a.output/'queue_status.json',{'status':'failed','exception':repr(exc),'records':records,
                    'elapsed_seconds':time.time()-started,'test_loaded':False})
        raise


def compare(root,artifacts):
    reports={label:json.loads((root/label/'dynamics/benchmark_summary.json').read_text()) for label in ('audio','static')}
    rows=[]
    for label,value in reports.items():
        for mode in ('full','static_state','oracle_state','reverse_audio'):
            r=value[mode];rows.append({'arm':label,'mode':mode,
                **{key:v['value'] for key,v in r['coefficient'].items()},'temporal':r['temporal']})
    save_json(root/'paired_results.json',{'scope':'446 paper validation queries; full native sequences',
        'rows':rows,'test_loaded':False,'no_quality_claim_from_training_completion':True})
    # Resample canonical sentence clusters; draws never inflate sample count.
    left=json.loads((artifacts/'audio/dynamics/temporal_full.json').read_text())
    right=json.loads((artifacts/'static/dynamics/temporal_full.json').read_text())
    controls={r['clip_id']:r for r in right}
    if set(controls)!={r['clip_id'] for r in left}:raise ValueError('Paired temporal memberships differ')
    paired={}
    for key in ('raw','centered'):
        values={}
        for row in left:
            other=controls[row['clip_id']]
            if row['sentence']!=other['sentence'] or row['scales']!=other['scales']:raise ValueError('Pair metadata/scales differ')
            a,b=row['joint_fair_es'][key],other['joint_fair_es'][key]
            if a is not None and b is not None:values.setdefault(row['sentence'],[]).append(a-b)
        means=np.array([np.mean(v) for v in values.values()]);counts=np.array([len(v) for v in values.values()])
        rng=np.random.default_rng(20260920);ix=rng.integers(len(means),size=(2000,len(means)))
        boot=(means[ix]*counts[ix]).sum(1)/counts[ix].sum(1)
        paired[key]={'audio_minus_static':float((means*counts).sum()/counts.sum()),
                     'sentence_cluster_95ci':np.quantile(boot,[.025,.975]).tolist(),
                     'clusters':len(means),'clips':int(counts.sum()),'negative_favors_audio':True}
    save_json(root/'paired_intervals.json',paired)
    cols=['arm','mode','arkit_mbe','arkit_lbe','arkit_fdd_signed','arkit_fdd_absolute','supp_upper9_fdd_absolute']
    lines=['# Paired paper validation diagnostic','',
           'All 446 validation clips; three fixed draws, matching targets, full native clocks and masks.',
           'The static arm is separately trained from the same stage-4 checkpoint and upper-flow initialization.',
           'Its local audio and h0 are both pooled over time. Global audio and independent neutral identity remain.',
           'Oracle rows use target state; neither oracle nor validation is a final test result.','',
           '| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in rows:lines.append('| '+' | '.join(str(row[k]) if isinstance(row[k],str) else f'{row[k]:.6f}' for k in cols)+' |')
    for key,row in paired.items():
        lines += ['',f"{key} fair ES audio - independently trained static: {row['audio_minus_static']:.6f}; sentence-cluster 95% CI {row['sentence_cluster_95ci']}."]
    lines+=['','Timing/distribution diagnostics, raw out-of-domain rates, identity and global readouts are saved alongside coefficient results.',
            'FDD excludes brows and is invariant to time permutation. It cannot establish audio timing.',
            'Audio/static differences require paired per-sentence intervals and visual review before claims.',
            'AV synchronization, learned feature evaluations and external baseline retraining remain unfinished.']
    (root/'PAIRED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
