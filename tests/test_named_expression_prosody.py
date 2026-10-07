import copy

import torch

from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts.train_full_staged import emotion_intensity_prototypes, native_expression_target


def test_named_temporal_contract_ignores_hubert_and_retains_measured_prosody():
    torch.set_num_threads(1);torch.manual_seed(33)
    mean=torch.randn(772);std=torch.rand(772)+.2
    student=SlowStateAffect(mean,std,hidden=8,global_dim=8,local_dim=64,
                           temporal_layout='expression-prosody').eval()
    x=torch.randn(2,12,1540,requires_grad=True);mask=torch.ones(2,12,dtype=torch.bool);mask[0,9:]=False
    output=student(x,mask,include_legacy_state=False)
    assert torch.equal(output['u_a'][...,:4],output['expression_state'])
    expected=torch.where(mask[...,None],(x[...,1536:]-mean[-4:])/std[-4:],0.)
    assert torch.equal(output['u_a'][...,4:8],expected)
    assert not output['u_a'][...,8:].count_nonzero()
    grad=torch.autograd.grad(output['u_a'].square().sum()+output['global'].square().sum(),x)[0]
    assert not grad[...,:768].count_nonzero() and not grad[~mask].count_nonzero()
    assert grad[...,1536:][mask].abs().sum()>0
    ck=dict(config={'model':{'audio_temporal_layout':'expression-prosody','audio_temporal_stride':16}},
            audio=copy.deepcopy(student.state_dict()),feature_stats={'mean':mean,'std':std})
    restored=SlowStateAffect.from_checkpoint(ck).eval()
    assert all(torch.equal(v,restored(x,mask,include_legacy_state=False)[k]) for k,v in output.items())


def test_expression_target_never_reads_mouth_gt_and_has_actual_temporal_variation():
    torch.manual_seed(31)
    mask=torch.ones(2,25,dtype=torch.bool);mask[0,20:]=False
    batch=dict(motion=torch.randn(2,25,52),valid=mask,channel_mask=torch.ones(2,52,dtype=torch.bool),
               anchors=torch.zeros(2,52),anchor_valid=torch.ones(2,52,dtype=torch.bool))
    scale=torch.ones(52)
    target=native_expression_target(batch,scale,8)
    changed=copy.deepcopy(batch);changed['motion'][...,14:41]=float('nan')
    other=native_expression_target(changed,scale,8)
    assert all(torch.equal(target[k],other[k]) for k in target)
    assert target['state'][mask].std(0).min()>0
    assert not target['state'][~mask].count_nonzero()


def test_label_prototypes_are_shared_per_emotion_intensity_and_preserve_missing_cells():
    codes=torch.tensor([[1.,3.],[3.,5.],[9.,2.],[8.,3.]])
    emotion=torch.tensor([0,0,1,1]);level=torch.tensor([0,0,1,2])
    proto,count=emotion_intensity_prototypes(codes,emotion,level,2,4)
    assert torch.equal(proto[0,0],torch.tensor([2.,4.]))
    assert count[0,0]==2 and count[0,1]==0
    assert not proto[0,1].count_nonzero()
    assert torch.equal(proto[emotion,level][0],proto[emotion,level][1])


def test_named_state_head_receives_direct_expression_gradient():
    torch.set_num_threads(1)
    student=SlowStateAffect(torch.zeros(772),torch.ones(772),hidden=8,temporal_layout='expression-prosody')
    valid=torch.ones(2,10,dtype=torch.bool)
    output=student(torch.randn(2,10,772),valid,include_legacy_state=False)
    (output['expression_state']-1).square().mean().backward()
    assert student.state_head.weight.grad.abs().sum()>0
    assert student.local_head.weight.grad is None
