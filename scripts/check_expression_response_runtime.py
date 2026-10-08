"""Real-data gradient and exact-resume checks, including deployment adaptation."""
import argparse,copy,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import (configure,load_runtime,cache_base,reference_batch,
    repeated_query,training_reference_views,objective,write,state_digest,development_fold)
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig,motion_objective


def run(binding_path,checkpoint_path,output,device='cuda'):
    configure(47);binding=json.loads(Path(binding_path).read_text());data,base=load_runtime(binding,device)
    fold=development_fold(data['splits']['train'],data['fit_sids'])
    ids=sorted(fold['train'],key=lambda i:int(data['splits']['train']['_lengths'][i]),reverse=True)[:16]
    cache_base(data,base,device,{'train':ids,'validation':[]})
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    started=time.time()
    ck=torch.load(checkpoint_path,map_location=device,weights_only=False)
    model=ExpressionResponse(ResponseConfig(**ck['config']),ck['model']['feature_mean'],
                            ck['model']['feature_std'],ck['model']['scales']).to(device)
    model.load_state_dict(ck['model']);model.train();base_digest=state_digest(base.stage1)
    b=data['splits']['train'].batch(torch.tensor(ids),device)
    reference_mode=ck.get('reference_training','single')
    ref_source=reference_batch(data,b,device)
    legacy=reference_batch(data,b,device,two_subsets=True)
    assert all(torch.equal(v,training_reference_views(ref_source)[k]) for k,v in legacy.items())
    refs=training_reference_views(ref_source,reference_mode,0);b=repeated_query(b)
    style=model.encode_style(refs)['code'];prior=model.audio_prior(b['audio_features'],b['valid'])
    posterior=model.motion_posterior(b['motion'],b['b0'],style,b['valid'],b['channel_mask'],b['times'])
    def check_no_prior(distribution):
        y=model.decode(b['b0'],distribution,style,b['valid'],sample=True)
        loss,_=motion_objective(y,b['motion'],b['valid'],b['channel_mask'],b['times'],model.scales)
        gs=torch.autograd.grad(loss,list(model.prior.parameters()),allow_unused=True,retain_graph=True)
        assert all(g is None or not g.any() for g in gs)
    check_no_prior(posterior);check_no_prior({k:v.detach() for k,v in prior.items()})
    initial=copy.deepcopy(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    rng=torch.Generator(device=device).manual_seed(123);snap=rng.get_state()
    def update(step):
        opt.zero_grad(set_to_none=True)
        refs=training_reference_views(ref_source,reference_mode,step)
        loss,logs=objective(model,b,refs,beta=.01,prior_weight=.5,sample=True,generator=rng,weights=None)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        for part in (model.prior,model.posterior,model.decoder,model.style):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in part.parameters())
        opt.step()
        return float(loss.detach())
    first=update(0)
    saved={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':rng.get_state()}
    second=update(1);expected=copy.deepcopy(model.state_dict())
    model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.set_state(saved['rng'])
    replay=update(1)
    assert second==replay
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())
    assert state_digest(base.stage1)==base_digest and all(p.grad is None for p in base.stage1.parameters())
    model.eval()
    normal=model.predict(b['audio_features'],b['b0'],b['valid'],refs)
    poison=b['audio_features'].clone();poison[...,:768]=float('nan')
    actual=model.predict(poison,b['b0'],b['valid'],refs)
    assert torch.equal(normal,actual)
    report={'passed':True,'reconstruction_prior_gradient_zero':True,'neutral_frozen':True,
            'all_four_modules_receive_correct_objectives':True,'exact_optimizer_rng_resume':True,
            'hubert_nan_independent':True,'reference_training':reference_mode,
            'center_local':model.cfg.center_local,'losses':[first,second,replay],'test_loaded':False}
    report.update(longest_fit_batch=len(ids),native_frames=b['valid'].shape[1],
                  seconds=time.time()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
                  legacy_single_reference_batch_exact=True)
    write(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();run(a.binding,a.checkpoint,a.output)
