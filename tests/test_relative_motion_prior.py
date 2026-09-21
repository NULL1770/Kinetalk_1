import torch
from torch import nn
from scripts.train_relative_motion_prior import conditions, prior_objective
from kinetalk_b0.models.dc_protected_temporal_flow import DCProtectedTemporalFlow


def example():
    torch.set_num_threads(1);torch.manual_seed(47)
    flow=DCProtectedTemporalFlow({'model':{'content_dim':8,'emotion_dim':8,'style_dim':8,
        'dit_dim':32,'dit_depth':1,'heads':4,'dropout':0.}})
    adapter=nn.Linear(28,8)
    valid=torch.ones(2,19,dtype=torch.bool);valid[0,:2]=False
    state=torch.randn(2,19,4)
    state=torch.where(valid[...,None],state,0.)
    state=torch.where(valid[...,None],state-state.sum(1,keepdim=True)/valid.sum(1)[:,None,None],0.)
    b={'valid':valid,'channel_mask':torch.ones(2,52,dtype=torch.bool),
        'relative_acoustic':torch.randn(2,19,28),'relative_state':state,'relative_delta':torch.zeros(2,19,9),
        'identity_code':torch.randn(2,8),'global_code':torch.randn(2,8),'intensity_value':torch.ones(2,1),
        'motion':torch.randn(2,19,52)}
    return flow,adapter,b


def test_conditions_ignore_gt_and_static_removes_all_frame_audio():
    f,a,b=example();out=conditions(b,a,f,'static')
    assert out[0]['h0'].count_nonzero()==0
    assert out[4].count_nonzero()==0
    assert out[5].count_nonzero()==0
    modified={**b,'motion':b['motion']*100,'relative_acoustic':b['relative_acoustic']*25}
    second=conditions(modified,a,f,'static')
    torch.testing.assert_close(out[3],second[3],rtol=0,atol=0)


def test_flow_training_finite_and_sample_mean_protected():
    f,a,b=example();noise=torch.randn(2,19,9)
    loss=prior_objective(f,a,b,torch.ones(9),'audio',noise,torch.tensor([.2,.8]),torch.randn_like(noise))
    loss.backward();assert any(p.grad is not None for p in f.parameters())
    q,i,e,local,state,_=conditions(b,a,f,'audio')
    y=f.decode(q,i,e,local,state,noise,4)
    torch.testing.assert_close(y.sum(1),torch.zeros(2,9),atol=2e-6,rtol=0)
