"""Real longest-fit-batch gradient, source isolation, and exact resume gate."""
import argparse,copy,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.refine_expression_prior import freeze_receiver,frozen_digests,objective,restore_model
from scripts.train_expression_response import (configure,load_runtime,cache_base,development_fold,
    reference_batch,training_reference_views,repeated_query,state_digest,write)


def run(binding_path,output,device='cuda'):
    configure(47);bnd=json.loads(Path(binding_path).read_text(encoding='utf8'))
    data,base=load_runtime(bnd,device);fold=development_fold(data['splits']['train'],data['fit_sids'])
    ids=sorted(fold['train'],key=lambda i:int(data['splits']['train']['_lengths'][i]),reverse=True)[:16]
    cache_base(data,base,device,{'train':ids,'validation':[]})
    m,parent=restore_model(bnd,device);params=freeze_receiver(m);frozen=frozen_digests(m)
    b=data['splits']['train'].batch(torch.tensor(ids),device);r=reference_batch(data,b,device);b=repeated_query(b)
    neutral=state_digest(base.stage1);mode=bnd['matching'];refmode=bnd['reference_training']
    refs=training_reference_views(r,refmode,0)
    with torch.no_grad():
        before=m.predict(b['audio_features'],b['b0'],b['valid'],refs)
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    torch.cuda.reset_peak_memory_stats();began=time.time()
    original=m.decoder.forward
    def forbidden(*args,**kwargs):raise AssertionError('Refinement objective invoked motion decoder')
    m.decoder.forward=forbidden
    def update(step):
        opt.zero_grad(set_to_none=True)
        refs=training_reference_views(r,refmode,step)
        loss,_=objective(m,b,refs,mode)
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        for part in (m.prior,m.emotion_head,m.intensity_head):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in part.parameters())
        assert all(p.grad is None for part in (m.posterior,m.style,m.decoder,base.stage1) for p in part.parameters())
        opt.step();return float(loss.detach())
    first=update(0);saved=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    second=update(1);expected=copy.deepcopy(m.state_dict())
    m.load_state_dict(saved);opt.load_state_dict(optimizer);replay=update(1)
    assert second==replay and all(torch.equal(v,m.state_dict()[k]) for k,v in expected.items())
    m.decoder.forward=original
    assert frozen_digests(m)==frozen and state_digest(base.stage1)==neutral
    m.eval()
    with torch.no_grad():
        normal=m.predict(b['audio_features'],b['b0'],b['valid'],refs)
        poisoned=b['audio_features'].clone();poisoned[...,:768]=float('nan')
        assert torch.equal(normal,m.predict(poisoned,b['b0'],b['valid'],refs))
        assert not torch.equal(before,normal)
    write(output,{'passed':True,'matching':mode,'reference_training':refmode,'center_local':m.cfg.center_local,
        'frozen_teacher_style_decoder_exact':True,'neutral_frozen':True,'motion_decoder_not_in_objective':True,
        'prior_and_semantic_heads_receive_gradients':True,'hubert_nan_independent':True,
        'exact_optimizer_restore':True,'losses':[first,second,replay],'longest_fit_batch':len(ids),
        'native_frames':b['valid'].shape[1],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'seconds':time.time()-began,'test_loaded':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.binding,a.output)
