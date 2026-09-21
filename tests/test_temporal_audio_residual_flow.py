import pytest
import torch

from kinetalk_b0.models.audio_residual_flow import AudioResidualFlow
from kinetalk_b0.models.temporal_audio_residual_flow import TemporalAudioResidualFlow, correlated_native_noise


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs():
    torch.manual_seed(135)
    cfg = {'content_dim':8,'emotion_dim':8,'style_dim':8,'dit_dim':12,'dit_depth':1,'heads':3,'dropout':0.}
    valid = torch.ones(2, 8, dtype=torch.bool); valid[0, 3] = False; valid[1, 6:] = False
    q = {'valid':valid, 'h0':torch.randn(2, 8, 8), 'channel_mask':torch.ones(2, 52, dtype=torch.bool)}
    identity = {'code':torch.randn(2, 8)}
    affect = {'global':torch.randn(2, 8), 'intensity_value':torch.ones(2, 1)}
    local = torch.randn(2, 8, 8); state = torch.randn(2, 8, 4)
    noise = torch.randn(2, 8, 9); target = torch.randn(2, 8, 9)
    return cfg, q, identity, affect, local, state, noise, target


def test_ar_noise_resets_at_holes_and_preserves_supplied_rng():
    noise = torch.arange(1, 7, dtype=torch.float64)[None, :, None].expand(1, 6, 9).clone()
    valid = torch.tensor([[True, True, False, True, True, False]])
    noise[:, ~valid[0]] = float('nan')
    rng = torch.get_rng_state().clone()
    value = correlated_native_noise(noise, valid, rho=.6)
    torch.testing.assert_close(value[:, 0], noise[:, 0])
    torch.testing.assert_close(value[:, 1], .6 * noise[:, 0] + .8 * noise[:, 1])
    torch.testing.assert_close(value[:, 3], noise[:, 3])
    torch.testing.assert_close(value[:, 4], .6 * noise[:, 3] + .8 * noise[:, 4])
    assert value[~valid].count_nonzero() == 0
    assert torch.equal(rng, torch.get_rng_state())


def test_flow_and_decode_apply_same_noise_transform_and_keep_padding_zero():
    cfg, q, identity, affect, local, state, noise, target = inputs()
    model = TemporalAudioResidualFlow(cfg, stride=4)
    # A zero velocity field leaves the correlated starting draw unchanged.
    for parameter in model.parameters(): parameter.data.zero_()
    expected = correlated_native_noise(noise, q['valid'], rho=model.noise_rho)
    decoded = model.decode(q, identity, affect, local, state, noise, steps=3)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    actual_loss = model.flow_loss(target, q, identity, affect, local, state, noise, torch.zeros(2))
    expected_loss = (target[q['valid']] - expected[q['valid']]).square().mean()
    torch.testing.assert_close(actual_loss, expected_loss)
    assert decoded[~q['valid']].count_nonzero() == 0


def test_temporal_path_starts_exactly_zero_and_has_trainable_gradients():
    cfg, q, identity, affect, local, state, noise, target = inputs()
    model = TemporalAudioResidualFlow(cfg, stride=4)
    torch.testing.assert_close(model._temporal_state(noise, q['valid']),
                               torch.where(q['valid'][...,None], noise, 0.), rtol=0, atol=0)
    local.requires_grad_()
    loss = model.flow_loss(target, q, identity, affect, local, state, noise, torch.tensor([.2,.7]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.temporal_up.weight.grad.abs().sum() > 0
    assert local.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_temporal_mixing_is_neighbor_local_bounded_and_cannot_cross_a_hole():
    cfg, q, *_ = inputs(); model = TemporalAudioResidualFlow(cfg)
    for module in (model.temporal_down, model.temporal_conv, model.temporal_up):
        module.weight.data.fill_(1.)
    x = torch.zeros(1, 8, 9); valid = q['valid'][:1]
    changed = x.clone(); changed[:, 2] = 1000.
    first = model._temporal_state(x, valid); second = model._temporal_state(changed, valid)
    assert (second[:, 1] - first[:, 1]).abs().max() > 0
    torch.testing.assert_close(second[:, 4:], first[:, 4:], rtol=0, atol=0)
    assert second[:, 3].count_nonzero() == 0
    correction = second - torch.where(valid[...,None], changed, 0.)
    assert correction.abs().max() <= model.temporal_scale


def test_native_positions_and_generation_ignore_target_payloads_and_appended_padding():
    cfg, q, identity, affect, local, state, noise, _ = inputs()
    model = TemporalAudioResidualFlow(cfg).eval()
    p = model._positions(8, noise); longer = model._positions(12, noise)
    torch.testing.assert_close(p, longer[:8], rtol=0, atol=0)
    assert not torch.equal(p[0], p[4])
    expected = model.decode(q, identity, affect, local, state, noise, 2)
    dirty = {**q, 'motion':torch.full((2,8,52), float('nan')), 'channel_mask':torch.zeros_like(q['channel_mask'])}
    torch.testing.assert_close(model.decode(dirty, identity, affect, local, state, noise, 2), expected, rtol=0, atol=0)
    pad = lambda x: torch.nn.functional.pad(x, (0,0,0,4), value=float('nan'))
    padded_q = {**q, 'h0':pad(q['h0']), 'valid':torch.nn.functional.pad(q['valid'],(0,4),value=False)}
    actual = model.decode(padded_q,identity,affect,pad(local),pad(state),pad(noise),2)
    torch.testing.assert_close(actual[:,:8],expected,atol=2e-6,rtol=2e-6)
    assert actual[:,8:].count_nonzero()==0


def test_training_rejects_missing_upper_support_and_invalid_noise():
    cfg, q, identity, affect, local, state, noise, target = inputs()
    model = TemporalAudioResidualFlow(cfg)
    q['channel_mask'][:,41] = False
    with pytest.raises(ValueError,match='all nine'):
        model.flow_loss(target,q,identity,affect,local,state,noise,torch.zeros(2))
    noise[0,0,0] = float('nan')
    with pytest.raises(ValueError,match='noise'):
        model.decode(q,identity,affect,local,state,noise,2)
    with pytest.raises(ValueError,match='noise'):
        correlated_native_noise(noise[:,:0],q['valid'][:,:0])


def test_same_conditioned_state_with_zero_temporal_path_matches_parent_velocity():
    cfg, q, identity, affect, local, state, noise, _ = inputs()
    model = TemporalAudioResidualFlow(cfg)
    parent = AudioResidualFlow(cfg)
    parent.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('temporal_')})
    conditions = model._conditions(q['valid'],q['h0'],identity['code'],affect,local,state)
    x = torch.where(q['valid'][...,None],noise,0.)
    time = torch.zeros(2)
    torch.testing.assert_close(model._velocity_unrestricted(x,time,q['valid'],conditions),
                               parent._velocity_unrestricted(x,time,q['valid'],conditions),rtol=0,atol=0)
