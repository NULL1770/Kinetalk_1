import copy

import pytest
import torch

from kinetalk_b0.models.dit import ResidualDiT


def model(enabled=False):
    return ResidualDiT(52,8,64,8,12,1,3,0.,global_dropout=0.,style_dropout=0.,
                       expression_modulation=enabled).eval()


def inputs():
    valid=torch.ones(2,9,dtype=torch.bool);valid[0,7:]=False
    return dict(x_t=torch.randn(2,9,52),time=torch.rand(2),content=torch.randn(2,9,8),
                global_emotion=torch.randn(2,64),intensity=torch.rand(2,1),style=torch.randn(2,8),
                mask=valid,temporal_condition=torch.randn(2,9,64),condition_dropout=False)


def test_zero_start_preserves_shared_state_rng_and_outputs():
    torch.set_num_threads(1)
    torch.manual_seed(89);old=model();rng=torch.random.get_rng_state()
    torch.manual_seed(89);new=model(True)
    assert torch.equal(torch.random.get_rng_state(),rng)
    for key,value in old.state_dict().items():
        assert torch.equal(value,new.state_dict()[key])
    assert set(new.state_dict())-set(old.state_dict())=={
        'expression_modulation.weight','expression_modulation.bias'}
    assert not new.expression_modulation.weight.count_nonzero()
    assert not new.expression_modulation.bias.count_nonzero()
    args=inputs()
    assert torch.equal(old(**args),new(**args))


def test_expression_route_receives_flow_gradient_and_influences_mouth():
    torch.manual_seed(81);network=model(True);args=inputs()
    loss=network(**args)[...,14:41][args['mask']].square().mean()
    loss.backward()
    assert network.expression_modulation.weight.grad.abs().sum()>0
    with torch.no_grad():
        network.expression_modulation.weight.add_(-.1*network.expression_modulation.weight.grad)
    before=network(**args)
    changed=copy.deepcopy(args);changed['temporal_condition'][...,:4]+=1
    after=network(**changed)
    assert (before[...,14:41][args['mask']]-after[...,14:41][args['mask']]).abs().sum()>0


def test_receiver_reads_only_four_named_states_and_safely_clears_padding():
    torch.manual_seed(82);network=model(True);args=inputs();seen=[]
    handle=network.expression_modulation.register_forward_pre_hook(lambda module,x:seen.append(x[0].detach().clone()))
    args['temporal_condition'][~args['mask']]=float('nan')
    result=network(**args);handle.remove()
    assert torch.isfinite(result).all() and not result[~args['mask']].count_nonzero()
    assert seen[0].shape[-1]==4
    assert torch.equal(seen[0][args['mask']],args['temporal_condition'][...,:4][args['mask']])
    assert not seen[0][~args['mask']].count_nonzero()


def test_nonzero_receiver_restores_and_has_exact_inference_meaning():
    torch.manual_seed(83);network=model(True);args=inputs()
    with torch.no_grad():network.expression_modulation.weight.normal_(0,.01)
    restored=model(True);restored.load_state_dict(network.state_dict(),strict=True)
    assert torch.equal(network(**args),restored(**args))
    with pytest.raises(RuntimeError):model().load_state_dict(network.state_dict(),strict=True)


def test_receiver_refuses_missing_condition_and_observed_nan():
    network=model(True);args=inputs();args['temporal_condition']=None
    with pytest.raises(ValueError,match='named temporal'):network(**args)
    args=inputs();args['temporal_condition'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):network(**args)


def test_enabling_loaded_model_preserves_rng_and_shared_parameters():
    network=model();before=copy.deepcopy(network.state_dict());rng=torch.random.get_rng_state()
    network.set_expression_modulation(True)
    assert torch.equal(torch.random.get_rng_state(),rng)
    assert all(torch.equal(v,network.state_dict()[k]) for k,v in before.items())
    network.set_expression_modulation(False)
    assert set(network.state_dict())==set(before)
