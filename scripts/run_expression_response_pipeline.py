"""Finite durable training -> frozen evaluation pipeline, no promotion/test use."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import train,parser,write,sha
from scripts.evaluate_expression_response import evaluate


def main(root):
    root=Path(root);lock=root/'pipeline.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    state=root/'pipeline_state.json'
    try:
        binding=json.loads((root/'binding.json').read_text())
        assert json.loads((root/'preflight_v2_state.json').read_text())['status']=='complete'
        assert json.loads((root/'runtime_checks.json').read_text())['passed']
        assert json.loads((root/'smoke_v2/smoke.json').read_text())['passed']
        assert json.loads((root/'eval_smoke/state.json').read_text())['status']=='complete'
        launch=json.loads((root/'launch_contract.json').read_text())
        assert sha(__file__)==launch['pipeline_sha256']
        assert sha(root/'binding.json')==launch['binding_sha256']
        for n,h in binding['source_files'].items():assert sha(root/'code'/n)==h,n
        write(state,{'status':'training','pid':os.getpid(),'started_at':time.time(),'test_loaded':False})
        a=parser().parse_args(['--binding',str(root/'binding.json'),'--output',str(root/'seed47'),
                              '--epochs','24','--seed','47','--batch-size','16'])
        runtime=train(a)
        write(state,{'status':'evaluating','pid':os.getpid(),'test_loaded':False})
        evaluate(binding,root/'seed47',root/'evaluation',runtime=runtime)
        write(state,{'status':'complete','pid':os.getpid(),'finished_at':time.time(),
                     'test_loaded':False,'default_replaced':False,'rendering':'independent local collector queue'})
    except Exception as exc:
        write(state,{'status':'failed','type':type(exc).__name__,'message':str(exc),
                     'traceback':traceback.format_exc(),'test_loaded':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
