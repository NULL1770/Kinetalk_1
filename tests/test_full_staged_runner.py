import copy

import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_full_staged import base_forward, identity_pairs, targets, UpperFlow, upper_motion
from kinetalk_b0.models.slow_state_affect import compose_upper_face, UPPER_INDICES, readout_slow_state, masked_slow_state


def config():
    return {'data':{'motion_dim':52,'content_dim':8,'audio_dim':8,'neutral_output_indices':[17,18],
                    'emotion_classes':['neutral','angry'],'num_intensity_levels':4},
            'model':{'content_dim':8,'emotion_dim':8,'style_dim':8,'hidden_dim':8,'heads':2,
                     'dit_dim':8,'dit_depth':1,'dropout':0.,'residual_scale':.25,'affect_hidden_dim':8}}


def test_b0_training_has_gradients_and_padding_invariance():
    torch.set_num_threads(1);torch.manual_seed(10)
    system=NeutralAffectSystem(config()).eval();system.stage1.requires_grad_(True)
    x=torch.randn(2,12,8);mask=torch.ones(2,12,dtype=torch.bool);mask[0,9:]=False
    y=base_forward(system,x,mask,gradients=True)
    padded=base_forward(system,torch.nn.functional.pad(x,(0,0,0,4)),torch.nn.functional.pad(mask,(0,4)),gradients=True)
    torch.testing.assert_close(y['b0'],padded['b0'][:,:12],rtol=0,atol=0)
    y['b0'].square().mean().backward()
    grads=[p.grad for p in system.stage1.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0


def test_identity_epoch_uses_only_fit_disjoint_views():
    refs={7:{'valid':torch.ones(4,12,dtype=torch.bool)},19:{'valid':torch.ones(2,12,dtype=torch.bool)},99:{'valid':torch.ones(4,12,dtype=torch.bool)}}
    pairs=identity_pairs(refs,[7,19])
    assert len(pairs)==4
    for sid,a,b in pairs:
        assert sid in (7,19) and not set(a)&set(b)
        assert set(a)|set(b)==set(range(len(refs[sid]['valid'])))


def test_runner_flow_target_and_nonupper_contract():
    torch.set_num_threads(1);torch.manual_seed(8)
    b,t=2,20;valid=torch.ones(b,t,dtype=torch.bool);valid[0,5]=False
    anchor=torch.full((b,52),.2);scales=torch.linspace(.03,.12,52)
    q={'motion':anchor[:,None]+torch.randn(b,t,52)*.02,'valid':valid,'channel_mask':torch.ones(b,52,dtype=torch.bool),
       'anchors':anchor,'anchor_valid':torch.ones(b,52,dtype=torch.bool),'h0':torch.randn(b,t,8)}
    truth,innovation=targets(q,scales,8)
    identity={'code':torch.randn(b,8)};affect={'global':torch.randn(b,8),'intensity_value':torch.ones(b,1)}
    local=torch.randn(b,t,8,requires_grad=True);model=UpperFlow(config(),stride=8)
    noise=torch.randn(b,t,9);flowtime=torch.rand(b)
    loss=model.flow_loss(innovation,q,identity,affect,local,truth['state'],noise,flowtime)
    loss.backward();assert torch.isfinite(local.grad).all() and local.grad.abs().sum()>0
    residual=model.decode(q,identity,affect,local.detach(),truth['state'],noise,3)
    uv=upper_motion(q,scales,truth['state'],residual)
    base=torch.randn(b,t,52);generated=compose_upper_face(base,uv,valid)
    others=[i for i in range(52) if i not in UPPER_INDICES]
    assert torch.equal(generated[...,others],base[...,others])
    read,mask=readout_slow_state(generated,valid[...,None].expand_as(generated),anchor,scales)
    back=masked_slow_state(read,mask,stride=8)['state']
    torch.testing.assert_close(back,truth['state'],atol=2e-6,rtol=2e-5)
