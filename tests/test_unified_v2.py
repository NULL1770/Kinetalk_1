import inspect
import torch
from kinetalk_v2.unified_model import (UnifiedConfig,UnifiedExpressionModel,teacher_targets,
    FeasibleResidualComposer,relation_style_loss,supervision_losses)

torch.set_num_threads(2)

def setup():
    torch.manual_seed(47)
    cfg=UnifiedConfig(hidden=32,decoder_hidden=48,content_dim=16)
    model=UnifiedExpressionModel(cfg,torch.zeros(772),torch.ones(772),torch.full((52,),.1))
    b,t=2,17
    valid=torch.ones(b,t,dtype=torch.bool);valid[1,-3:]=False
    base=torch.full((b,t,52),.05)
    motion=base.clone();motion[:,5:9,0]=.8;motion[:,5:9,7]=.8
    batch={'audio_features':torch.randn(b,t,1540),'h0':torch.randn(b,t,16),
        'b0':base,'motion':motion,'valid':valid,'channel_mask':torch.tensor([[True]*51+[False]]*b),
        'times':torch.arange(t).double()[None].expand(b,-1)/25,'emotion_id':torch.tensor([5,-1]),
        'intensity_id':torch.tensor([2,-1]),'intensity_valid':torch.tensor([True,False])}
    refs={'motion':motion.flip(1),'valid':torch.ones_like(valid),'channel_mask':batch['channel_mask']}
    targets=teacher_targets(motion,base,valid,batch['channel_mask'],batch['times'])
    return model,batch,refs,targets

def test_observed_events_duration_and_gap_censoring():
    _,b,_,targets=setup()
    assert targets['onset'][0,5,0]
    assert targets['duration_valid'][0,5,0]
    assert targets['duration'][0,5,0]==1
    assert targets['active'][0,5:9,0].all()
    b['times'][:,7:]+=.04
    censored=teacher_targets(b['motion'],b['b0'],b['valid'],b['channel_mask'],b['times'])
    assert not censored['duration_valid'][0,5,0]
    assert not censored['onset_valid'][0,7,0]

def test_reconstruction_gradient_reaches_student_and_single_reference():
    model,b,refs,targets=setup()
    out=model.training_forward(b,refs,targets)
    (out['motion']-b['motion']).square().mean().backward()
    assert model.student.state.weight.grad.abs().sum()>0
    assert model.style.output.weight.grad.abs().sum()>0
    assert model.teacher.state.weight.grad is None
    assert torch.equal(out['motion'][~b['valid']],torch.zeros_like(out['motion'][~b['valid']]))

def test_teacher_targets_ignore_missing_channels_and_invalid_frames():
    _,b,_,target=setup()
    b['channel_mask'][:,23]=False;b['motion'][:, :,23]=float('nan')
    b['motion'][~b['valid']]=float('nan')
    target=teacher_targets(b['motion'],b['b0'],b['valid'],b['channel_mask'],b['times'])
    assert torch.isfinite(target['state']).all()
    assert not target['quality'][...,0].any()

def test_rollout_seed_and_no_query_target_interface():
    model,b,refs,_=setup();model.eval()
    assert 'motion' not in inspect.signature(model.predict).parameters
    with torch.no_grad():
        a=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,times=b['times'],seed=91)
        same=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,times=b['times'],seed=91)
        other=model.predict(b['audio_features'],b['h0'],b['b0'],b['valid'],refs,times=b['times'],seed=92)
    assert torch.equal(a['motion'],same['motion'])
    assert not torch.equal(a['schedule']['events'],other['schedule']['events'])
    assert not torch.equal(a['motion'],other['motion'])

def test_feasible_composer_preserves_null_directions_and_detaches_labels():
    proposal=torch.zeros(1,4,52,requires_grad=True)
    proposal.data[...,17]=.8;proposal.data[...,23]=.4
    jac=torch.zeros(1,4,1,52);jac[...,17]=1
    upper=torch.full((1,4,1),.2,requires_grad=True)
    constraints={'jacobian':jac,'lower':torch.zeros_like(upper),'upper':upper,'weight':torch.ones_like(upper)}
    composer=FeasibleResidualComposer()
    y=composer(proposal,constraints)
    assert (y[...,17]-.2).abs().max()<.001
    assert torch.equal(y[...,23],proposal[...,23])
    y.sum().backward();assert upper.grad is None
    assert torch.equal(composer(proposal),proposal)

def test_loss_masks_unknown_emotion_and_relations_are_not_constant_solution():
    model,b,refs,t=setup();out=model.training_forward(b,refs,t)
    parts=supervision_losses(model,out,t,b)
    assert torch.isfinite(parts['total'])
    parts['total'].backward()
    codes=torch.randn(4,64,requires_grad=True);speakers=torch.tensor([0,1,0,1])
    relation=relation_style_loss(codes,speakers);relation.backward()
    assert codes.grad.abs().sum()>0
    assert torch.isfinite(relation)
