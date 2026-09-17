import pytest
import torch

from kinetalk_b0.models.temporal_upper import TemporalUpperFlow, UPPER_INDICES, compose_upper_face


def fixture(*, use_state=False):
    torch.manual_seed(29)
    cfg = {'model': {'content_dim': 8, 'emotion_dim': 6, 'style_dim': 5,
                     'dit_dim': 12, 'dit_depth': 2, 'heads': 3, 'dropout': 0.}}
    model = TemporalUpperFlow(cfg, use_state=use_state)
    valid = torch.ones(2, 9, dtype=torch.bool); valid[0, [3, 7, 8]] = False
    content = torch.randn(2, 9, 8)
    identity = torch.randn(2, 5)
    affect = {'global': torch.randn(2, 6), 'intensity_value': torch.randn(2, 1)}
    local = torch.randn(2, 9, 6)
    state = torch.randn(2, 9, 4)
    noise = torch.randn(2, 9, 9)
    return model, valid, content, identity, affect, local, state, noise


def test_direct_arm_ignores_state_and_has_identical_parameter_structure():
    direct, valid, content, identity, affect, local, state, noise = fixture()
    soft, *_ = fixture(use_state=True)
    assert direct.state_dict().keys() == soft.state_dict().keys()
    assert all(torch.equal(value, soft.state_dict()[key]) for key, value in direct.state_dict().items())
    expected = direct.decode(valid, content, identity, affect, local, None, noise, steps=2)
    actual = direct.decode(valid, content, identity, affect, local, torch.full_like(state, float('nan')), noise, steps=2)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    actual.square().mean().backward()
    assert direct.renderer.state_projection.weight.grad is None


def test_each_layer_condition_gets_gradient_and_audio_time_changes_decode():
    model, valid, content, identity, affect, local, state, noise = fixture()
    local.requires_grad_()
    target = torch.randn_like(noise)
    loss = model.flow_loss(target, valid, content, identity, affect, local, state, noise, torch.tensor([.2, .8]))
    loss.backward()
    assert local.grad.abs().sum() > 0 and torch.isfinite(local.grad).all()
    for layer in model.renderer.frame_condition:
        assert layer.weight.grad.abs().sum() > 0 and torch.isfinite(layer.weight.grad).all()
    expected = model.decode(valid, content, identity, affect, local, None, noise, steps=3)
    changed = model.decode(valid, content, identity, affect, local.flip(1), None, noise, steps=3)
    assert not torch.allclose(expected, changed)


def test_soft_state_is_learned_condition_not_fixed_motion_lift():
    model, valid, content, identity, affect, local, state, noise = fixture(use_state=True)
    state.requires_grad_()
    low = model.decode(valid, content, identity, affect, local, state, noise, steps=3)
    high = model.decode(valid, content, identity, affect, local, state + 2., noise, steps=3)
    assert not torch.allclose(low, high)
    low.square().sum().backward()
    assert state.grad.abs().sum() > 0 and torch.isfinite(state.grad).all()
    assert model.renderer.state_projection.weight.grad.abs().sum() > 0
    with torch.no_grad(): model.renderer.state_projection.weight.zero_()
    zero = model.decode(valid, content, identity, affect, local, state, noise, steps=2)
    other = model.decode(valid, content, identity, affect, local, state + 100., noise, steps=2)
    torch.testing.assert_close(zero, other, atol=0, rtol=0)


def test_all_nine_dimensions_including_slow_group_means_remain_free():
    model, valid, content, identity, affect, local, state, noise = fixture()
    with torch.no_grad():
        model.renderer.output.weight.zero_()
        model.renderer.output.bias.copy_(torch.arange(1., 10.))
    out = model.decode(valid, content, identity, affect, local, state, torch.zeros_like(noise), steps=4)
    expected = torch.arange(1., 10.).expand_as(out)
    torch.testing.assert_close(out[valid], expected[valid], atol=0, rtol=0)
    assert out[valid].mean() > 1.  # No centering, Q-projection, clamp or amplitude cap.


def test_training_and_one_step_decode_use_identical_condition_velocity():
    model, valid, content, identity, affect, local, state, noise = fixture(use_state=True)
    conditions = model.prepare_conditions(valid, content, identity, affect, local, state)
    velocity = model.velocity(noise, torch.zeros(2), conditions)
    decoded = model.decode(valid, content, identity, affect, local, state, noise, steps=1)
    expected = torch.where(valid[..., None], noise + velocity, 0.)
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)
    target = torch.randn_like(noise)
    loss = model.flow_loss(target, valid, content, identity, affect, local, state, noise, torch.zeros(2))
    expected_loss = (velocity[valid] - (target - noise)[valid]).square().mean()
    torch.testing.assert_close(loss, expected_loss)


def test_invalid_nan_and_padding_do_not_change_observed_outputs_or_gradients():
    model, valid, content, identity, affect, local, state, noise = fixture(use_state=True)
    expected = model.decode(valid, content, identity, affect, local, state, noise, steps=2)
    mask = torch.cat((valid, torch.zeros(2, 5, dtype=torch.bool)), 1)
    values = []
    for x in (content, local, state, noise):
        x = x.masked_fill(~valid[..., None], float('nan'))
        values.append(torch.cat((x, torch.full((2, 5, x.shape[-1]), float('nan'))), 1).requires_grad_())
    actual = model.decode(mask, values[0], identity, affect, values[1], values[2], values[3], steps=2)
    torch.testing.assert_close(actual[:, :9][valid], expected[valid], atol=1e-6, rtol=1e-5)
    assert actual[~mask].count_nonzero() == 0
    actual.square().mean().backward()
    for x in values:
        assert torch.isfinite(x.grad).all() and x.grad[~mask].count_nonzero() == 0


def test_upper_composition_still_copies_other_43_exactly():
    model, valid, content, identity, affect, local, state, noise = fixture()
    upper = model.decode(valid, content, identity, affect, local, state, noise, steps=2)
    base = torch.randn(2, 9, 52)
    final = compose_upper_face(base, upper, valid)
    other = [i for i in range(52) if i not in UPPER_INDICES]
    assert torch.equal(final[..., other], base[..., other])
    assert torch.equal(final[~valid], base[~valid])


def test_observed_nonfinite_conditions_and_missing_soft_state_are_rejected():
    model, valid, content, identity, affect, local, state, noise = fixture(use_state=True)
    with pytest.raises(ValueError, match='state'):
        model.decode(valid, content, identity, affect, local, None, noise)
    content[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model.decode(valid, content, identity, affect, local, state, noise)
