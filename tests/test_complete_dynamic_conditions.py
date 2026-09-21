import pytest
import torch

from scripts.train_full_staged import complete_dynamic_conditions


def fixture():
    valid = torch.tensor([[True, False, True, True, False], [True, True, True, False, False]])
    feature = torch.arange(20, dtype=torch.float64).reshape(2,5,2).requires_grad_()
    h0 = feature.detach().clone() * .25
    h0[~valid] = float('nan')
    features = torch.where(valid[...,None], feature, float('nan'))
    q = {'valid':valid, 'h0':h0, 'audio_features':features,
         'global':torch.ones(2,3), 'identity':torch.ones(2,4)*2,
         'noise':torch.arange(90).reshape(2,5,9),
         'motion':torch.full((2,5,52),float('nan'))}
    output = {'state':features * 2, 'local':features * 3,
              'global':torch.full((2,3),9.), 'emotion_logits':torch.full((2,8),4.)}
    class Encoder:
        def __call__(self, supplied, supplied_valid):
            assert supplied is q['audio_features']
            assert supplied_valid is valid
            return output
    return q, output, Encoder(), feature


@pytest.mark.parametrize('mode', ['static','reverse'])
def test_all_frame_conditions_are_intervened_on_native_valid_positions(mode):
    q, original, encoder, _ = fixture()
    h0_before = q['h0'].clone(); input_before = q['audio_features'].clone()
    local_before = original['local'].clone(); state_before = original['state'].clone()
    actual_q, dynamic = complete_dynamic_conditions(q,encoder,mode)
    for actual, source in ((actual_q['h0'],h0_before),(dynamic['local'],local_before),(dynamic['state'],state_before)):
        for row, valid in enumerate(q['valid']):
            observed = source[row,valid]
            expected = observed.mean(0,keepdim=True).expand_as(observed) if mode=='static' else observed.flip(0)
            torch.testing.assert_close(actual[row,valid],expected)
        assert actual[~q['valid']].count_nonzero()==0
        assert torch.isfinite(actual).all()
    # Global context, identity, draw, audio source and target are untouched;
    # the class consumes the explicit h0/local/state outputs for generation.
    for name in ('global','identity','noise','valid','motion','audio_features'):
        assert actual_q[name] is q[name]
    for name in ('global','emotion_logits'):
        assert dynamic[name] is original[name]
    torch.testing.assert_close(q['h0'],h0_before,equal_nan=True)
    torch.testing.assert_close(q['audio_features'],input_before,equal_nan=True)
    torch.testing.assert_close(original['local'],local_before,equal_nan=True)
    torch.testing.assert_close(original['state'],state_before,equal_nan=True)


def test_static_pooling_remains_differentiable_and_ignores_invalid_nan_gradients():
    q, _, encoder, feature = fixture()
    _, static = complete_dynamic_conditions(q,encoder,'static')
    (static['local'].square().sum()+static['state'].square().sum()).backward()
    assert torch.isfinite(feature.grad).all()
    assert feature.grad[q['valid']].abs().sum()>0
    assert feature.grad[~q['valid']].count_nonzero()==0
    for row, valid in enumerate(q['valid']):
        torch.testing.assert_close(feature.grad[row,valid],feature.grad[row,valid][:1].expand_as(feature.grad[row,valid]))


def test_audio_mode_keeps_condition_tensors_unchanged_and_unknown_mode_rejected():
    q, output, encoder, _ = fixture()
    actual_q, dynamic = complete_dynamic_conditions(q,encoder,'audio')
    assert actual_q is not q
    assert all(actual_q[k] is value for k,value in q.items())
    assert dynamic is output
    with pytest.raises(ValueError,match='dynamic intervention'):
        complete_dynamic_conditions(q,encoder,'shift')


def test_static_erases_encoded_boundary_variation_not_just_input_audio():
    valid=torch.tensor([[True,True,True,True,False]])
    q={'audio_features':torch.ones(1,5,2),'h0':torch.ones(1,5,2),'valid':valid}
    def boundary_encoder(feature,mask):
        values=torch.tensor([1.,4.,4.,2.,float('nan')])[None,:,None].expand(1,5,3)
        return {'local':values,'state':values*2,'global':torch.zeros(1,2)}
    _,dynamic=complete_dynamic_conditions(q,boundary_encoder,'static')
    torch.testing.assert_close(dynamic['local'][valid],torch.full((4,3),2.75))
    torch.testing.assert_close(dynamic['state'][valid],torch.full((4,3),5.5))
