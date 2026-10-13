import torch
from tests.test_v5_factor_rebuild import setup
from scripts.pilot_v5_style_output import timing_loss,batch_donors


def test_style_timing_ignores_global_offset_and_has_valid_frame_gradients():
    _,model,b,_=setup()
    b['times']=torch.arange(b['valid'].shape[1],dtype=torch.float64)[None].expand(2,-1)/25
    truth=b['motion'].detach()
    y=truth.clone().requires_grad_(True)
    assert timing_loss(model,y,truth,b).item()==0.
    assert timing_loss(model,truth+.2,truth,b).item()<1e-10
    changed=(truth+torch.randn_like(truth)*.01).requires_grad_(True)
    loss=timing_loss(model,changed,truth,b);loss.backward()
    assert changed.grad.abs().sum()>0
    assert changed.grad[~b['valid']].abs().sum()==0


def test_style_training_donors_are_explicitly_different_people_or_skipped():
    p=torch.tensor([0,0,2,3]);ids,use=batch_donors(p)
    assert use.all();assert (p[ids]!=p).all()
    p=torch.tensor([1,1]);ids,use=batch_donors(p)
    assert not use.any();assert torch.equal(ids,torch.arange(2))
