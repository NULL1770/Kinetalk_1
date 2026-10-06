import pytest
import torch
from scripts.frozen_factor_diagnostic import (
    stratified_indices, original_batches, identity_variant, response_values,
)


def test_sample_cells_are_fixed_before_generation_and_batch_order_independent():
    ids=['z','a','b','c'];s=[0,0,1,1];e=[0,0,0,1];l=[0,0,0,3]
    assert stratified_indices(ids,s,e,l)==[1,2,3]
    order=[3,2,1,0]
    selected=stratified_indices([ids[i] for i in order],[s[i] for i in order],
        [e[i] for i in order],[l[i] for i in order])
    assert {ids[order[i]] for i in selected}=={'a','b','c'}
    assert original_batches([1,2,5],7,2)==[[0,1],[2,3],[4,5]]
    with pytest.raises(ValueError): original_batches([7],7,2)


def test_code_bias_interventions_preserve_the_other_path_by_identity():
    own={'code':torch.ones(1,3),'baseline':torch.ones(1,2)}
    other={'code':torch.full((1,3),2.),'baseline':torch.full((1,2),3.)}
    for mode in ['zero_code','code_other_1']:
        result=identity_variant(own,own,own,other,other,mode)
        assert result['baseline'] is own['baseline']
    for mode in ['zero_bias','bias_other_1']:
        result=identity_variant(own,own,own,other,other,mode)
        assert result['code'] is own['code']
    assert own['code'].sum()==3 and own['baseline'].sum()==2


def test_constant_bias_has_zero_dynamic_response_and_ignores_missing_frames():
    valid=torch.tensor([True,True,False,True,True])
    channels=torch.tensor([True,False])
    reference=torch.zeros(5,2)
    value=torch.tensor([[2.,float('nan')]]).repeat(5,1)
    value[2]=float('nan')
    assert response_values(value,reference,valid,channels,[0,1])==[2.,2.,0.,0.]
