"""Finite hash-bound refinement and original-protocol evaluation."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.refine_expression_prior import parser,train
from scripts.train_expression_response import write,sha
from scripts.evaluate_expression_response import evaluate


def main(root,resume=False):
    root=Path(root);lock=root/'pipeline.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    state=root/'pipeline_state.json'
    try:
        binding=json.loads((root/'binding.json').read_text())
        contract=json.loads((root/'launch_contract.json').read_text())
        assert sha(root/'binding.json')==contract['binding_sha256']
        assert sha(__file__)==contract['pipeline_sha256']
        for name,h in binding['source_files'].items():assert sha(root/'code'/name)==h,name
        for name in ('runtime_checks.json','smoke_v2/smoke.json'):
            assert json.loads((root/name).read_text())['passed']
        assert json.loads((root/'preflight_v2_state.json').read_text())['status']=='complete'
        assert json.loads((root/'eval_smoke/state.json').read_text())['status']=='complete'
        args=['--binding',str(root/'binding.json'),'--output',str(root/'seed47'),
              '--matching',binding['matching'],'--epochs','8','--seed','47','--batch-size','16']
        if resume:args+=['--resume']
        write(state,{'status':'training','pid':os.getpid(),'started_at':time.time(),'test_loaded':False})
        runtime=train(parser().parse_args(args))
        write(state,{'status':'evaluating','pid':os.getpid(),'test_loaded':False})
        evaluate(binding,root/'seed47',root/'evaluation',runtime=runtime)
        write(state,{'status':'complete','pid':os.getpid(),'finished_at':time.time(),
                     'test_loaded':False,'default_replaced':False})
    except Exception as exc:
        write(state,{'status':'failed','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();main(a.root,a.resume)
