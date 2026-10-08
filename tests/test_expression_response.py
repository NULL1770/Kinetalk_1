import io
import pytest
import torch
from kinetalk_b0.models.expression_response import (
    ExpressionResponse, ResponseConfig, motion_objective, normalized_kl, native_interpolate)

torch.set_num_threads(2)


def fixture(prior_variance='learned'):
    torch.manual_seed(19)
    model=ExpressionResponse(ResponseConfig(hidden=16,decoder_hidden=24,global_dim=8,
        local_dim=4,style_dim=8,prior_variance=prior_variance),torch.zeros(772),torch.ones(772),torch.ones(52)*.2)
    valid=torch.ones(2,9,dtype=torch.bool);valid[1,7:]=False
    audio=torch.randn(2,9,1540);base=torch.rand(2,9,52,requires_grad=True)
    refs={'motion':torch.rand(2,2,7,52),'b0':torch.rand(2,2,7,52),
          'valid':torch.ones(2,2,7,dtype=torch.bool),'channel_mask':torch.ones(2,2,52,dtype=torch.bool)}
    times=torch.arange(9,dtype=torch.float64)[None].expand(2,-1)/25
    return model,audio,base,valid,refs,times


def test_prior_has_no_content_access_even_nan():
    m,a,b,v,r,t=fixture();a.requires_grad_()
    p=m.audio_prior(a,v);changed=a.detach().clone();changed[...,:768]=float('nan')
    other=m.audio_prior(changed,v)
    for k in p:torch.testing.assert_close(p[k],other[k],rtol=0,atol=0)
    g=torch.autograd.grad(p['g_mean'].sum()+p['u_mean'].sum(),a)[0]
    assert not g[...,:768].any() and not g[~v].any() and g[...,768:].abs().sum()>0


@pytest.mark.parametrize('prior_variance',['learned','unit'])
def test_gradients_obey_training_responsibilities(prior_variance):
    m,a,b,v,r,t=fixture(prior_variance);p=m.audio_prior(a,v);s=m.encode_style(r)['code']
    ch=torch.ones(2,52,dtype=torch.bool)
    q=m.motion_posterior(torch.rand_like(b),b,s,v,ch,t)
    y=m.decode(b,q,s,v,sample=True)
    y[...,14:41].square().sum().backward()
    assert b.grad is None and all(x.grad is None for x in m.prior.parameters())
    for part in (m.posterior,m.style,m.decoder):
        assert any(x.grad is not None and x.grad.abs().sum()>0 for x in part.parameters())
    assert (m.decoder.output.weight.grad[14:41].abs().sum(1)>0).all()
    m.zero_grad(set_to_none=True);p=m.audio_prior(a,v);s=m.encode_style(r)['code']
    detached={k:x.detach() for k,x in p.items()}
    m.decode(b,detached,s,v).square().sum().backward()
    assert all(x.grad is None for x in m.prior.parameters())
    m.zero_grad(set_to_none=True)
    p=m.audio_prior(a,v);q=m.motion_posterior(torch.rand_like(b),b,s,v,ch,t)
    normalized_kl(q,p).backward()
    for part in (m.posterior,m.prior):
        assert any(x.grad is not None and x.grad.abs().sum()>0 for x in part.parameters())


def test_padding_nan_and_reference_set_invariance():
    m,a,b,v,r,t=fixture();m.eval()
    with torch.no_grad():
        y=m.predict(a,b,v,r)
        reversed_refs={k:x.flip(1) for k,x in r.items()}
        torch.testing.assert_close(y,m.predict(a,b,v,reversed_refs),rtol=1e-6,atol=1e-6)
        a[~v]=float('nan');b=b.detach();b[~v]=float('nan')
        torch.testing.assert_close(y,m.predict(a,b,v,r),rtol=0,atol=0)
        ap=torch.cat((a,torch.full((2,4,1540),float('nan'))),1)
        bp=torch.cat((b,torch.full((2,4,52),float('nan'))),1)
        vp=torch.cat((v,torch.zeros(2,4,dtype=torch.bool)),1)
        torch.testing.assert_close(y,m.predict(ap,bp,vp,r)[:,:9],rtol=1e-5,atol=1e-6)


def test_prior_mean_save_restore_and_private_randomness():
    m,a,b,v,r,t=fixture();m.eval();y=m.predict(a,b,v,r)
    buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
    restored=fixture()[0].eval();restored.load_state_dict(torch.load(buf,weights_only=True))
    torch.testing.assert_close(y,restored.predict(a,b,v,r),rtol=0,atol=0)
    rng=torch.Generator().manual_seed(11);state=rng.get_state()
    sampled=m.predict(a,b,v,r,sample=True,generator=rng);rng.set_state(state)
    torch.testing.assert_close(sampled,m.predict(a,b,v,r,sample=True,generator=rng),rtol=0,atol=0)
    assert not torch.equal(y,sampled)


def test_kl_mask_and_length_normalization():
    m,a,b,v,r,t=fixture();p=m.audio_prior(a,v)
    assert abs(float(normalized_kl(p,p).detach()))<1e-7
    q={k:x.clone() for k,x in p.items()};q['g_mean']=q['g_mean']+1;q['u_mean']=q['u_mean']+1
    loss=normalized_kl(q,p)
    qp={};pp={}
    for source,dest in ((q,qp),(p,pp)):
        for k,x in source.items():
            if k.startswith('u_'):dest[k]=torch.cat((x,x),1)
            else:dest[k]=x
    torch.testing.assert_close(loss,normalized_kl(qp,pp))
    assert p['g_logvar'].min()>=m.cfg.logvar_min and p['g_logvar'].max()<=m.cfg.logvar_max


def test_velocity_does_not_bridge_gap_or_missing_channel():
    pred=torch.zeros(1,4,52);pred[:,2:]=1;target=torch.zeros_like(pred)
    times=torch.tensor([[0.,.04,.2,.24]],dtype=torch.float64)
    valid=torch.ones(1,4,dtype=torch.bool);channels=torch.ones(1,52,dtype=torch.bool)
    _,parts=motion_objective(pred,target,valid,channels,times,torch.ones(52))
    assert parts['velocity']==0
    channels[:,1]=False;target[:,:,1]=float('nan');pred[:,:,1]=float('nan')
    value,_=motion_objective(pred,target,valid,channels,times,torch.ones(52))
    assert torch.isfinite(value)


def test_native_interpolation_retains_fixed_clock():
    z=torch.tensor([[[0.],[2.],[4.]]]);mask=torch.ones(1,3,dtype=torch.bool)
    y=native_interpolate(z,mask,torch.ones(1,6,dtype=torch.bool),2)
    torch.testing.assert_close(y.flatten(),torch.tensor([0.,.5,1.5,2.5,3.5,4.]))


def test_no_query_gt_in_deployment_signature_and_no_legacy_parameters():
    import inspect
    m,*_=fixture()
    assert list(inspect.signature(m.predict).parameters)==['audio','base','valid','refs','sample','generator']
    assert not any(s in k for k in m.state_dict() for s in ['h0','prototype','flow','speaker_embedding'])


def test_observed_nan_fails_closed():
    m,a,b,v,r,t=fixture();a[0,0,800]=float('nan')
    with pytest.raises(ValueError):m.audio_prior(a,v)


def test_unit_prior_removes_only_prior_variance_rows_and_matches_initialization():
    old,a,b,v,r,t=fixture();model,*_=fixture('unit');cfg=model.cfg
    assert model.prior.global_head.out_features==cfg.global_dim
    assert model.prior.local_head.out_features==cfg.local_dim
    assert model.posterior.global_head.out_features==2*cfg.global_dim
    assert model.posterior.local_head.out_features==2*cfg.local_dim
    before=old.state_dict()
    for k,value in model.state_dict().items():
        expected=before[k][:len(value)] if k.startswith(('prior.global_head.','prior.local_head.')) else before[k]
        torch.testing.assert_close(value,expected,rtol=0,atol=0)
    p=model.audio_prior(a,v);other=old.audio_prior(a,v)
    for k in ('g_mean','u_mean','u_mask'):torch.testing.assert_close(p[k],other[k],rtol=0,atol=0)
    assert torch.equal(p['g_logvar'],torch.zeros_like(p['g_logvar']))
    assert torch.equal(p['u_logvar'],torch.zeros_like(p['u_logvar']))
    q={k:x.clone() for k,x in p.items()};q['g_mean']+=2;q['u_mean']+=2
    # Unit variances: KL is half squared mean error, normalized by factor/length/dim.
    torch.testing.assert_close(normalized_kl(q,p),torch.tensor(2.))
    q['u_mean'][~p['u_mask']]=1000.
    torch.testing.assert_close(normalized_kl(q,p),torch.tensor(2.))
    rng=torch.Generator().manual_seed(11);state=rng.get_state()
    y=model.predict(a,b,v,r,sample=True,generator=rng)
    restored=ExpressionResponse(ResponseConfig(**model.checkpoint_config()),
        model.feature_mean,model.feature_std,model.scales)
    restored.load_state_dict(model.state_dict(),strict=True);rng.set_state(state)
    torch.testing.assert_close(y,restored.predict(a,b,v,r,sample=True,generator=rng),rtol=0,atol=0)


def test_unit_prior_still_rejects_content_access_even_nan():
    model,a,b,v,r,t=fixture('unit');a.requires_grad_();p=model.audio_prior(a,v)
    changed=a.detach().clone();changed[...,:768]=float('nan');other=model.audio_prior(changed,v)
    for k in p:torch.testing.assert_close(p[k],other[k],rtol=0,atol=0)
    g=torch.autograd.grad(p['g_mean'].sum()+p['u_mean'].sum(),a)[0]
    assert not g[...,:768].any() and g[...,768:].abs().sum()>0
