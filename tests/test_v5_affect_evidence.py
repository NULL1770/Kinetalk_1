import torch
from kinetalk_v5.affect_evidence import AffectEvidence,affect_loss,query_moment_loss


def sample():
    y=torch.rand(2,20,52,requires_grad=True)
    b=dict(motion=torch.rand_like(y),valid=torch.ones(2,20,dtype=torch.bool),
           channel_mask=torch.ones(2,52,dtype=torch.bool),dataset_id=torch.tensor([0,1]),
           emotion_id=torch.tensor([2,4]))
    return y,b


def test_frozen_real_observer_has_input_gradients_and_masks_unknown_hdtf():
    y,b=sample();obs=AffectEvidence().eval().requires_grad_(False)
    affect_loss(obs,y,b).backward()
    assert y.grad[0].abs().sum()>0
    assert y.grad[1].abs().sum()==0
    assert all(p.grad is None for p in obs.parameters())


def test_query_moment_losses_ignore_invalid_and_unobserved_coefficients():
    y,b=sample();b['valid'][:,-2:]=False;b['channel_mask'][:,51]=False
    scale=torch.ones(52)*.01
    a=query_moment_loss(y,b,scale)
    modified=b['motion'].clone();modified[:,-2:]=1000.;modified[:,:,51]=1000.
    z=query_moment_loss(y,{**b,'motion':modified},scale)
    assert torch.equal(a,z)
    a.backward();assert y.grad[:,-2:].abs().sum()==0
    assert y.grad[:,:,51].abs().sum()==0


def test_evidence_scale_floors_bound_almost_constant_channels():
    obs=AffectEvidence();obs.fit_scales(torch.ones(8,306))
    assert obs.scale[:204].min()>=.04
    assert obs.scale[204:].min()>=.015
