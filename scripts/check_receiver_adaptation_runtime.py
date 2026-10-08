"""Real GPU longest TRAIN batch: decoder-only gradients and exact stochastic replay."""
import argparse, copy, io, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.adapt_expression_receiver import freeze_conditions,frozen_digest,objective,inherited_budget
from scripts.refine_expression_prior import restore_model
from scripts.train_expression_response import (configure,load_runtime,cache_base,development_fold,
    reference_batch,state_digest,write)


def run(binding_path,output,device='cuda'):
    configure(47); device=torch.device(device)
    binding=json.loads(Path(binding_path).read_text(encoding='utf8'))
    data,base=load_runtime(binding,device)
    fold=development_fold(data['splits']['train'],data['fit_sids'])
    ids=sorted(fold['train'],key=lambda i:int(data['splits']['train']['_lengths'][i]),reverse=True)[:16]
    cache_base(data,base,device,{'train':ids,'validation':[]})
    m,parent=restore_model(binding,device);params=freeze_conditions(m)
    frozen,neutral=frozen_digest(m),state_digest(base.stage1)
    b=data['splits']['train'].batch(torch.tensor(ids),device);refs=reference_batch(data,b,device)
    with torch.no_grad():
        before=m.predict(b['audio_features'],b['b0'],b['valid'],refs)
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    rng=torch.Generator(device=device).manual_seed(40047)
    order_rng=torch.Generator().manual_seed(30047)
    torch.cuda.reset_peak_memory_stats();began=time.time()
    def update():
        opt.zero_grad(set_to_none=True)
        loss,_=objective(m,b,refs,binding['mode'],rng)
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('decoder.'))
        assert all(p.grad is None for p in base.stage1.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
        opt.step();return float(loss.detach())
    first=update()
    saved=io.BytesIO()
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),sample_rng=rng.get_state(),
                    order_rng=order_rng.get_state()),saved)
    second=update();expected=copy.deepcopy(m.state_dict());next_order=torch.randperm(31,generator=order_rng)
    saved.seek(0);ck=torch.load(saved,map_location=device,weights_only=False)
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
    rng.set_state(ck['sample_rng'].cpu());order_rng.set_state(ck['order_rng'].cpu())
    replay=update()
    assert second==replay and all(torch.equal(v,m.state_dict()[k]) for k,v in expected.items())
    assert torch.equal(next_order,torch.randperm(31,generator=order_rng))
    assert frozen_digest(m)==frozen and state_digest(base.stage1)==neutral==parent['neutral_digest']
    m.eval()
    with torch.no_grad():
        normal=m.predict(b['audio_features'],b['b0'],b['valid'],refs)
        poison=b['audio_features'].clone();poison[...,:768]=float('nan')
        assert torch.equal(normal,m.predict(poison,b['b0'],b['valid'],refs))
        assert not torch.equal(normal,before) and torch.isfinite(normal).all()
    write(output,dict(passed=True,mode=binding['mode'],frozen_conditions_exact=True,neutral_frozen=True,
        only_decoder_gradients=True,hubert_nan_independent=True,exact_optimizer_sample_order_restore=True,
        losses=[first,second,replay],longest_fit_batch=len(ids),native_frames=b['valid'].shape[1],
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-began,
        inherited_budget=inherited_budget(parent),test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.binding,a.output)
