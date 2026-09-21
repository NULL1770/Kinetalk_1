"""Run the gated KineTalk experiment, then the declared FaceDiffuser adapter."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_formal_predictable_projection import save_json
from scripts.extract_emotion2vec_pilot import sha
from scripts.run_isolated_state_queue import source_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','source','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh suite output required')
    args.output.mkdir(parents=True)
    code=Path(__file__).resolve().parents[1]
    sources=source_inventory(code)
    source_sha=sha(args.source);data_sha=sha(args.data/'index.json')
    def verify_inputs():
        if source_inventory(code)!=sources:raise RuntimeError('Suite code changed after launch')
        if sha(args.source)!=source_sha or sha(args.data/'index.json')!=data_sha:
            raise RuntimeError('Suite source/data changed after launch')
    jobs=[('kinetalk',[sys.executable,'-u',str(code/'scripts/run_isolated_state_queue.py'),
        '--data',str(args.data),'--source',str(args.source),'--output',str(args.output/'kinetalk'),
        '--state-epochs','100','--residual-epochs','40']),
        ('facediffuser',[sys.executable,'-u',str(code/'scripts/train_facediffuser_arkit.py'),
        '--data',str(args.data),'--source',str(args.source),'--output',str(args.output/'facediffuser'),
        '--epochs','100','--batch-size','16','--seed','47'])]
    save_json(args.output/'plan.json',{'schema':'isolated_baseline_suite_v1','sources':sources,
        'source_sha256':source_sha,'data_index_sha256':data_sha,
        'jobs':jobs,'started_utc':datetime.now(timezone.utc).isoformat(),'test_loaded':False,
        'quality_stop_policy':'KineTalk failed state gate skips residual; baseline still runs. Process errors stop suite.'})
    records=[];started=time.monotonic()
    try:
        for name,command in jobs:
            verify_inputs()
            with (args.output/(name+'.log')).open('w') as stream:
                process=subprocess.Popen(command,cwd=code,stdout=stream,stderr=subprocess.STDOUT)
                save_json(args.output/'status.json',{'status':'running','stage':name,'pid':process.pid,
                    'command':command,'test_loaded':False,'default_replaced':False})
                rc=process.wait()
            records.append({'stage':name,'exit_code':rc})
            save_json(args.output/'records.json',records)
            if rc:raise RuntimeError(name+' failed; inspect stage log')
            verify_inputs()
            result=json.loads((args.output/name/'status.json').read_text())
            allowed=('complete','stopped_quality_gate') if name=='kinetalk' else ('complete',)
            if result.get('status') not in allowed:raise RuntimeError(name+' ended without final status')
        save_json(args.output/'status.json',{'status':'complete','records':records,
            'elapsed_seconds':time.monotonic()-started,'test_loaded':False,'default_replaced':False,
            'quality_pass_claimed':False,'review_required':True})
    except BaseException as error:
        save_json(args.output/'status.json',{'status':'failed','error':repr(error),
            'records':records,'test_loaded':False,'default_replaced':False})
        raise


if __name__=='__main__':main()
