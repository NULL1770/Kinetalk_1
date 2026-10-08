import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from tests.test_expression_response import fixture
from scripts.refine_expression_prior import distribution_match,freeze_receiver,frozen_digests,objective,scientific_args
from kinetalk_b0.models.expression_response import normalized_kl


def latent_pair(m,a,v):
    p={k:z.detach().clone().requires_grad_(z.dtype!=torch.bool) for k,z in m.audio_prior(a,v).items()}
    q={k:z.detach().clone() for k,z in p.items()}
    q['g_mean']+=2.;q['u_mean']+=torch.randn_like(q['u_mean'])
    return q,p


@pytest.mark.parametrize('center',[False,True])
def test_mean_gradient_does_not_depend_on_prior_variance(center):
    m,a,b,v,r,t=fixture();m.cfg=replace(m.cfg,center_local=center)
    q,p=latent_pair(m,a,v)
    x=distribution_match(m,q,p,v,'mean_scale')
    grads=torch.autograd.grad(x,[p['g_mean'],p['u_mean']])
    other={k:z.detach().clone().requires_grad_(z.dtype!=torch.bool) for k,z in p.items()}
    other['g_logvar']=torch.full_like(other['g_logvar'],2.,requires_grad=True)
    other['u_logvar']=torch.full_like(other['u_logvar'],2.,requires_grad=True)
    y=distribution_match(m,q,other,v,'mean_scale')
    later=torch.autograd.grad(y,[other['g_mean'],other['u_mean']])
    for left,right in zip(grads,later):torch.testing.assert_close(left,right,rtol=0,atol=0)
    torch.testing.assert_close(distribution_match(m,q,p,v,'kl'),normalized_kl(q,p),rtol=0,atol=0)


def test_native_mask_padding_and_centering_preserved():
    m,a,b,v,r,t=fixture();m.cfg=replace(m.cfg,center_local=True)
    v[0,3:5]=False;q,p=latent_pair(m,a,v)
    expected=distribution_match(m,q,p,v,'mean_scale')
    q['u_mean']=q['u_mean']+5
    q['u_mean'][~q['u_mask']]=float('nan');q['u_logvar'][~q['u_mask']]=float('nan')
    # Masked nonfinite variance must be removed before exp, including gradients.
    with torch.no_grad():p['u_logvar'][~p['u_mask']]=float('nan')
    actual=distribution_match(m,q,p,v,'mean_scale')
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    vp=torch.cat((v,torch.zeros(2,4,dtype=torch.bool)),1)
    torch.testing.assert_close(actual,distribution_match(m,q,p,vp,'mean_scale'),rtol=1e-5,atol=1e-6)
    actual.backward()
    for k,x in p.items():
        if x.grad is not None:assert torch.isfinite(x.grad).all()
    assert not p['u_logvar'].grad[~p['u_mask']].any()


def test_mean_scale_units_and_q_is_always_detached():
    m,a,b,v,r,t=fixture();m.cfg=replace(m.cfg,center_local=False)
    q,p=latent_pair(m,a,v)
    # Zero means, identical variances => zero. Two-unit mean offsets => 2.
    for d in (q,p):
        for k,x in d.items():
            if k!='u_mask':d[k]=torch.zeros_like(x,requires_grad=True)
    assert distribution_match(m,q,p,v,'mean_scale')==0
    q['g_mean']=q['g_mean']+2.;q['u_mean']=q['u_mean']+2.
    torch.testing.assert_close(distribution_match(m,q,p,v,'mean_scale'),torch.tensor(2.))
    distribution_match(m,q,p,v,'mean_scale').backward()
    assert all(x.grad is None for k,x in q.items() if x.is_leaf)


@pytest.mark.parametrize('mode',['kl','mean_scale'])
def test_only_prior_heads_train_and_exact_optimizer_resume(mode):
    m,a,base,v,r,t=fixture();params=freeze_receiver(m);frozen=frozen_digests(m)
    m.decoder.forward=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Training called decoder'))
    b={'audio_features':a.clone().requires_grad_(),'motion':torch.rand_like(base,requires_grad=True),
       'b0':base,'valid':v,'channel_mask':torch.ones(2,52,dtype=torch.bool),'times':t,
       'emotion_id':torch.tensor([1,2]),'intensity_id':torch.tensor([1,2]),'intensity_valid':torch.ones(2,dtype=torch.bool)}
    opt=torch.optim.AdamW(params,lr=1e-4)
    def update():
        opt.zero_grad(set_to_none=True);loss,_=objective(m,b,r,mode);loss.backward();opt.step();return float(loss.detach())
    update()
    assert b['motion'].grad is None and base.grad is None
    assert not b['audio_features'].grad[...,:768].any()
    assert all(p.grad is None for part in (m.posterior,m.style,m.decoder) for p in part.parameters())
    for part in (m.prior,m.emotion_head,m.intensity_head):assert any(p.grad is not None and p.grad.abs().sum()>0 for p in part.parameters())
    saved=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(opt.state_dict());loss=update();expected=copy.deepcopy(m.state_dict())
    m.load_state_dict(saved);opt.load_state_dict(optimizer);replay=update()
    assert replay==loss and all(torch.equal(v,m.state_dict()[k]) for k,v in expected.items())
    assert frozen_digests(m)==frozen
    poison={**b,'audio_features':b['audio_features'].detach().clone()};poison['audio_features'][...,:768]=float('nan')
    with torch.no_grad():torch.testing.assert_close(objective(m,b,r,mode)[0],objective(m,poison,r,mode)[0],rtol=0,atol=0)


def test_resume_flag_excluded_but_scientific_settings_bound():
    a=SimpleNamespace(resume=False,matching='kl',epochs=8,seed=47)
    initial=scientific_args(a,'single');a.resume=True
    assert scientific_args(a,'single')==initial
    a.matching='mean_scale'
    assert scientific_args(a,'single')!=initial
