"""Finite, immutable-source receiver experiment followed by the unchanged evaluator."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.adapt_expression_receiver import parser,train
from scripts.train_expression_response import write,sha
from scripts.evaluate_expression_response import evaluate


def main(root,resume=False):
    root=Path(root)
    fd=os.open(root/'pipeline.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    state=root/'pipeline_state.json'
    try:
        binding=json.loads((root/'binding.json').read_text())
        contract=json.loads((root/'launch_contract.json').read_text())
        assert sha(root/'binding.json')==contract['binding_sha256']
        assert sha(__file__)==contract['pipeline_sha256']
        for name,h in binding['source_files'].items():assert sha(root/'code'/name)==h,name
        for name in ('runtime_checks.json','smoke/smoke.json'):
            assert json.loads((root/name).read_text())['passed']
        assert json.loads((root/'preflight_state.json').read_text())['status']=='complete'
        assert json.loads((root/'eval_smoke/state.json').read_text())['status']=='complete'
        args=['--binding',str(root/'binding.json'),'--output',str(root/'seed47'),
              '--mode',binding['mode'],'--epochs','8','--seed','47','--batch-size','16']
        if resume:args+=['--resume']
        write(state,dict(status='training',pid=os.getpid(),started_at=time.time(),test_loaded=False))
        runtime=train(parser().parse_args(args))
        write(state,dict(status='evaluating',pid=os.getpid(),test_loaded=False))
        evaluate(binding,root/'seed47',root/'evaluation',runtime=runtime)
        write(state,dict(status='complete',pid=os.getpid(),finished_at=time.time(),test_loaded=False,default_replaced=False))
    except Exception as exc:
        write(state,dict(status='failed',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();main(a.root,a.resume)
