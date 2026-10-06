import pytest
import torch

from scripts.train_full_staged import balanced_flow_units, flow_vector_mse, mse, parser


def test_default_and_explicit_unit_scale_preserve_loss_and_gradient_bit_exact():
    x=torch.randn(2,5,4,requires_grad=True);y=torch.randn_like(x)
    mask=torch.rand(x.shape)>.2
    baseline=mse(x,y,mask);want=torch.autograd.grad(baseline,x,retain_graph=True)[0]
    for std in (None,torch.ones(4)):
        value=flow_vector_mse(x,y,mask,std)
        actual=torch.autograd.grad(value,x,retain_graph=True)[0]
        assert torch.equal(value,baseline) and torch.equal(actual,want)
    assert parser().parse_args(['--output','unused']).flow_vector_units=='scalar'


def test_relative_units_equal_analytic_error_and_exclude_missing_nan_values():
    x=torch.tensor([[[3.,float('nan')],[5.,3.]]],requires_grad=True)
    y=torch.tensor([[[1.,float('nan')],[1.,1.]]])
    observed=torch.tensor([[[True,False],[True,True]]])
    std=torch.tensor([2.,.5])
    loss=flow_vector_mse(x,y,observed,std)
    assert float(loss.detach())==pytest.approx((1+4+16)/3)
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad[0,0,1]==0
    assert std.grad is None


def test_changing_units_together_preserves_dimensionless_error():
    x=torch.randn(2,3,4);y=torch.randn_like(x);mask=torch.ones_like(x,dtype=torch.bool)
    std=torch.tensor([.2,.5,1.,2.]);change=torch.tensor([3.,7.,.5,2.])
    torch.testing.assert_close(flow_vector_mse(x,y,mask,std),
        flow_vector_mse(x*change,y*change,mask,std*change),rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('std',[torch.zeros(4),torch.tensor([1.,1.,-1.,1.]),torch.ones(3),torch.full((4,),float('nan'))])
def test_invalid_units_are_rejected(std):
    x=torch.zeros(2,3,4)
    with pytest.raises(ValueError):flow_vector_mse(x,x,torch.ones_like(x,dtype=torch.bool),std)


def test_balanced_units_preserve_relative_precision_with_mean_weight_one():
    std=torch.tensor([1e-4,.2,.8,1.])
    support=torch.tensor([True,True,True,False])
    units=balanced_flow_units(std,support)
    weights=units.double().square().reciprocal()
    assert float(weights[support].mean())==pytest.approx(1.,rel=2e-7)
    torch.testing.assert_close(weights[0]/weights[1],(std[1].double()/std[0].double()).square())
    x=torch.randn(2,3,4);y=torch.randn_like(x);mask=support.view(1,1,-1).expand_as(x)
    raw=flow_vector_mse(x,y,mask,std)
    torch.testing.assert_close(flow_vector_mse(x,y,mask,units),raw/std[support].square().reciprocal().mean(),rtol=1e-6,atol=1e-6)


def test_balanced_units_are_invariant_to_global_choice_of_residual_units():
    std=torch.tensor([.001,.2,3.])
    support=torch.ones(3,dtype=torch.bool)
    torch.testing.assert_close(balanced_flow_units(std,support),balanced_flow_units(std*10,support),rtol=1e-6,atol=1e-6)
