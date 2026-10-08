import pytest
import torch
from scripts.diagnose_expression_targets import ClipRidge, encode_inputs, frame_features
from tests.test_expression_response import fixture


def test_ridge_clip_weight_and_mask_nonfinite():
    r=ClipRidge(1,1)
    x=torch.tensor([[[1.],[float('nan')]],[[2.],[2.]]])
    y=torch.tensor([[[3.],[float('nan')]],[[4.],[4.]]])
    mask=torch.tensor([[True,False],[True,True]])
    r.add(x,y,mask)
    assert r.count==2
    torch.testing.assert_close(r.xx,torch.tensor([[5.]],dtype=torch.float64))
    torch.testing.assert_close(r.xy,torch.tensor([[11.]],dtype=torch.float64))
    with pytest.raises(ValueError):r.add(x,y,torch.zeros_like(mask))


def test_ridge_solves_affine_and_centered_heads():
    torch.manual_seed(123)
    x=torch.randn(12,9,3);mask=torch.ones(12,9,dtype=torch.bool)
    target=x@torch.tensor([[2.],[-1.],[.5]])+4
    r=ClipRidge(3,1,intercept=True);r.add(x,target,mask);w,n=r.solve(1e-8)
    torch.testing.assert_close(ClipRidge.predict(x,w,True),target,rtol=1e-5,atol=1e-5)
    assert n.gt(0).all()


def test_posterior_input_construction_and_effective_hidden_mapping():
    m,a,base,valid,refs,times=fixture()
    from dataclasses import replace
    m.cfg=replace(m.cfg,center_local=True)
    m.eval()
    channels=torch.ones(2,52,dtype=torch.bool)
    b={'motion':torch.rand_like(base),'b0':base.detach(),'valid':valid,
       'channel_mask':channels,'times':times,'audio_features':a}
    s=m.encode_style(refs)['code']
    q=m.motion_posterior(b['motion'],b['b0'],s,valid,channels,times)
    other=m.posterior(encode_inputs(m,b,s),valid)
    for k in q:torch.testing.assert_close(q[k],other[k],rtol=0,atol=0)
    h=m.prior.encoder((a[...,768:]-m.feature_mean)/m.feature_std,valid)
    p=m.audio_prior(a,valid);g,u=m.conditions(p,valid)
    f=frame_features(m,b,h)
    pred=f['hidden']@m.prior.local_head.weight[:m.cfg.local_dim].T
    torch.testing.assert_close(pred,u,rtol=1e-5,atol=1e-6)
