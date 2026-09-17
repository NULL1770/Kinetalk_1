import pytest
import torch

from kinetalk_b0.models.slow_state_affect import (
    SlowStateAffect, UpperInnovationFlow, STATE_GROUPS, UPPER_INDICES, compose_upper_face,
    lift_slow_state, masked_slow_state, project_upper_innovation, readout_slow_state,
)


def test_partial_support_gaps_group_masks_and_padding():
    value = torch.arange(1., 11.).reshape(1, 5, 2)
    mask = torch.tensor([[[True, True], [False, False], [True, False],
                          [True, True], [True, False]]])
    clean = value.masked_fill(~mask, float('nan'))
    out = masked_slow_state(clean, mask, stride=3)
    torch.testing.assert_close(out['bin_state'], torch.tensor([[[1., 2.], [7., 8.], [13., 0.]]]))
    torch.testing.assert_close(out['state'][mask], value[mask])
    assert torch.equal(out['state_mask'], mask)
    torch.testing.assert_close(out['bin_count'], torch.tensor([[[4/3, 1.], [7/3, 1.], [1/3, 0.]]]))
    assert out['state'][~mask].count_nonzero() == 0
    padded = masked_slow_state(torch.cat([clean, torch.full((1, 5, 2), float('nan'))], 1),
                               torch.cat([mask, torch.zeros(1, 5, 2, dtype=torch.bool)], 1), stride=3)
    torch.testing.assert_close(out['state'], padded['state'][:, :5], rtol=1e-6, atol=1e-6)
    assert not padded['bin_mask'][:, 3:].any()


def test_readout_lift_roundtrip_is_signed_and_only_changes_declared_channels():
    state = torch.tensor([[[-2., 3., -.5, 1.2], [1.1, -4., .8, -1.]]])
    scales = torch.linspace(.02, 1., 52)
    lifted = lift_slow_state(state, scales)
    frame, mask = readout_slow_state(lifted, torch.ones_like(lifted, dtype=torch.bool),
                                    torch.zeros(1, 52), scales)
    torch.testing.assert_close(frame, state)
    assert mask.all()
    remaining = [i for i in range(52) if i not in UPPER_INDICES]
    assert lifted[..., remaining].count_nonzero() == 0
    assert (lifted[0, 0, list(STATE_GROUPS[0])] < 0).all()
    assert (lifted[0, 0, list(STATE_GROUPS[1])] > 0).all()


def test_target_missing_channels_are_masked_per_frame_group():
    x = torch.zeros(1, 3, 52)
    observed = torch.zeros_like(x, dtype=torch.bool)
    observed[0, 0, 43] = True
    observed[0, 1, 41] = True
    x[0, 0, 43] = -.2
    x[0, 1, 41] = .4
    x[~observed] = float('nan')
    anchor = torch.full((1, 52), float('nan'))
    anchor[0, [41, 43]] = 0.
    state, mask = readout_slow_state(x, observed, anchor, torch.full((52,), .1))
    torch.testing.assert_close(state, torch.tensor([[[-2., 0., 0., 0.], [0., 4., 0., 0.], [0., 0., 0., 0.]]]))
    assert torch.equal(mask, state != 0)
    lifted = lift_slow_state(state.masked_fill(~mask, float('nan')), torch.ones(52), mask)
    assert torch.isfinite(lifted).all()


def test_innovation_projection_idempotence_and_zero_coarse_group_state():
    torch.manual_seed(18)
    value = torch.randn(2, 35, 9, requires_grad=True)
    valid = torch.ones(2, 35, dtype=torch.bool)
    valid[0, [3, 8, 16, 17, 34]] = False
    projected = project_upper_innovation(value, valid)
    again = project_upper_innovation(projected, valid)
    torch.testing.assert_close(projected, again, atol=3e-7, rtol=1e-6)
    grouped = torch.stack([projected[..., list(g)].mean(-1) for g in ((2, 3, 4), (0, 1), (5, 7), (6, 8))], -1)
    slow = masked_slow_state(grouped, valid)['state']
    torch.testing.assert_close(slow, torch.zeros_like(slow), atol=2e-7, rtol=0)
    coarse = torch.where(valid[..., None], value, 0.) - projected
    torch.testing.assert_close((coarse * projected).sum(), torch.zeros(()), atol=1e-5, rtol=0)
    assert projected[~valid].count_nonzero() == 0
    projected.square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert value.grad[~valid].count_nonzero() == 0


def test_projection_preserves_asymmetry_and_orthogonal_fast_timing():
    value = torch.zeros(1, 4, 9)
    value[0, :, 0] = 2.
    value[0, :, 1] = -2.
    value[0, :, 2:5] = torch.tensor([1., -3., 3., -1.])[:, None]
    valid = torch.ones(1, 4, dtype=torch.bool)
    torch.testing.assert_close(project_upper_innovation(value, valid, stride=4), value, atol=1e-6, rtol=1e-6)
    poisoned = torch.cat([value, torch.full((1, 3, 9), float('nan'))], 1)
    padded_mask = torch.cat([valid, torch.zeros(1, 3, dtype=torch.bool)], 1)
    out = project_upper_innovation(poisoned, padded_mask, stride=4)
    torch.testing.assert_close(out[:, :4], value, atol=1e-6, rtol=1e-6)
    assert out[:, 4:].count_nonzero() == 0


def test_rank_deficient_single_observation_empty_feature_and_single_frame():
    value = torch.full((1, 39, 2), float('nan'), dtype=torch.float64)
    valid = torch.zeros_like(value, dtype=torch.bool)
    valid[0, 19, 0] = True
    value[0, 19, 0] = -3.7
    value.requires_grad_()
    out = masked_slow_state(value, valid)
    torch.testing.assert_close(out['state'][valid], value[valid], atol=1e-12, rtol=0)
    assert torch.isfinite(out['bin_state']).all()
    assert out['bin_state'][..., 1].count_nonzero() == 0
    assert out['state'][~valid].count_nonzero() == 0
    out['state'].sum().backward()
    torch.testing.assert_close(value.grad[valid], torch.ones_like(value.grad[valid]))
    assert value.grad[~valid].count_nonzero() == 0
    single = masked_slow_state(torch.tensor([[[2., -1.]]]), torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(single['state'], torch.tensor([[[2., -1.]]]))


def test_linear_curves_are_recovered_and_internal_knots_have_no_step():
    time = torch.arange(49, dtype=torch.float64)
    straight = (2. + .15 * time)[None, :, None]
    valid = torch.ones(1, 49, dtype=torch.bool)
    valid[0, 5:10] = False
    out = masked_slow_state(straight.masked_fill(~valid[..., None], float('nan')), valid)
    torch.testing.assert_close(out['state'][valid], straight[valid], atol=1e-12, rtol=0)
    # A piecewise-linear trajectory with deliberately different segment slopes
    # must retain its common knot value, without a sample-and-hold jump.
    bent = (1. + .1 * time - .15 * (time - 16).clamp_min(0.) + .08 * (time - 32).clamp_min(0.))[None, :, None]
    all_valid = torch.ones_like(valid)
    projected = masked_slow_state(bent, all_valid)['state']
    torch.testing.assert_close(projected, bent, atol=1e-12, rtol=0)
    for knot in (16, 32):
        left_slope = projected[0, knot - 1, 0] - projected[0, knot - 2, 0]
        right_slope = projected[0, knot + 2, 0] - projected[0, knot + 1, 0]
        torch.testing.assert_close(projected[0, knot - 1, 0] + left_slope, projected[0, knot, 0], atol=1e-12, rtol=0)
        torch.testing.assert_close(projected[0, knot + 1, 0] - right_slope, projected[0, knot, 0], atol=1e-12, rtol=0)


def test_composition_preserves_other_channels_exactly_and_gradients_cannot_leak():
    torch.manual_seed(2)
    base = torch.randn(2, 5, 52, requires_grad=True)
    upper = torch.randn(2, 5, 52, requires_grad=True)
    valid = torch.tensor([[True, True, False, True, False], [True] * 5])
    output = compose_upper_face(base, upper, valid)
    remaining = [i for i in range(52) if i not in UPPER_INDICES]
    assert torch.equal(output[..., remaining], base[..., remaining])
    assert torch.equal(output[~valid], base[~valid])
    assert torch.equal(output[..., list(UPPER_INDICES)][valid], upper[..., list(UPPER_INDICES)][valid])
    output.sum().backward()
    assert upper.grad[..., remaining].count_nonzero() == 0
    assert upper.grad[~valid].count_nonzero() == 0
    assert torch.equal(base.grad[..., remaining], torch.ones_like(base.grad[..., remaining]))


def fixture():
    torch.manual_seed(9)
    model = SlowStateAffect(torch.zeros(7), torch.ones(7), hidden=16, global_dim=8,
                            local_dim=6, stride=4)
    valid = torch.tensor([[True, True, False, True, True, True, False], [True] * 7])
    features = torch.randn(2, 7, 7)
    return model, features, valid


def test_model_zero_state_local_start_and_global_training():
    model, features, valid = fixture()
    output = model(features, valid)
    assert output['state'].count_nonzero() == 0
    assert output['local'].count_nonzero() == 0
    assert output['global'].shape == (2, 8)
    assert output['emotion_logits'].shape == (2, 8)
    assert output['intensity_logits'].shape == (2, 4)
    loss = (output['global'] - 1).square().mean() + (output['state'][valid] - 1).square().mean()
    loss.backward()
    assert model.state_head.weight.grad.abs().sum() > 0
    assert model.input.weight.grad.abs().sum() > 0


def test_model_padding_and_mask_poison_invariant_with_finite_gradients():
    model, features, valid = fixture()
    with torch.no_grad():
        model.local_head.weight.normal_(std=.1)
        model.state_head.weight.normal_(std=.1)
    original = model(features, valid)
    poisoned = features.masked_fill(~valid[..., None], float('nan')).requires_grad_()
    padded = torch.cat([poisoned, torch.full((2, 5, 7), float('nan'))], 1)
    mask = torch.cat([valid, torch.zeros(2, 5, dtype=torch.bool)], 1)
    output = model(padded, mask)
    for key in ('global', 'emotion_logits', 'intensity_logits'):
        torch.testing.assert_close(original[key], output[key], atol=1e-6, rtol=1e-5)
    for key in ('local', 'state'):
        torch.testing.assert_close(original[key], output[key][:, :7], atol=1e-6, rtol=1e-5)
        assert output[key][~mask].count_nonzero() == 0
    (output['state'].sum() + output['local'].sum()).backward()
    assert torch.isfinite(poisoned.grad).all()
    assert poisoned.grad[~valid].count_nonzero() == 0


def test_oracle_is_explicit_and_does_not_change_audio_predictions():
    model, features, valid = fixture()
    states = torch.tensor([-3., 2., -.7, 1.5]).expand(2, 7, 4).clone()
    states[~valid] = float('nan')
    output = model(features, valid, state_override=states)
    torch.testing.assert_close(output['state'][valid], torch.tensor([-3., 2., -.7, 1.5]).expand(int(valid.sum()), 4))
    assert output['predicted_state'].count_nonzero() == 0
    assert output['local'].count_nonzero() == 0
    assert output['state'][~valid].count_nonzero() == 0
    assert model(features, valid)['state'].count_nonzero() == 0
    # The audio mask is authoritative even if the supplied target mask has
    # extra entries; missing poisoned values never become observations.
    extra = model(features, valid, state_override=states,
                  state_override_mask=torch.ones_like(states, dtype=torch.bool))
    torch.testing.assert_close(extra['state'], output['state'], atol=0, rtol=0)


def test_invalid_observed_values_and_empty_audio_are_rejected():
    model, features, valid = fixture()
    features[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model(features, valid)
    with pytest.raises(ValueError, match='nonempty'):
        model(torch.zeros_like(features), torch.zeros_like(valid))
    with pytest.raises(ValueError, match='scales'):
        lift_slow_state(torch.zeros(1, 2, 4), torch.zeros(52))


def flow_fixture():
    torch.manual_seed(38)
    flow = UpperInnovationFlow({'model': {'content_dim': 8, 'emotion_dim': 6, 'style_dim': 5,
                                         'dit_dim': 12, 'dit_depth': 1, 'heads': 3}}, stride=4)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[0, [3, 7, 8]] = False
    content = torch.randn(2, 9, 8)
    identity = torch.randn(2, 5)
    global_affect = {'global': torch.randn(2, 6), 'intensity_value': torch.randn(2, 1)}
    local = torch.randn(2, 9, 6, requires_grad=True)
    state = torch.randn(2, 9, 4, requires_grad=True)
    noise = torch.randn(2, 9, 9)
    return flow, valid, content, identity, global_affect, local, state, noise


def test_upper_flow_train_gradients_are_finite_and_target_slow_component_is_excluded():
    flow, valid, content, identity, affect, local, state, noise = flow_fixture()
    target = torch.randn_like(noise)
    time = torch.tensor([.2, .8])
    loss = flow.flow_loss(target, valid, content, identity, affect, local, state, noise, time)
    assert torch.isfinite(loss) and loss > 0
    # Constant per-group shifts lie entirely in P and must not become an
    # innovation target or cause the innovation flow to cancel slow state.
    shift = lift_slow_state(torch.tensor([1., -2., .5, 3.]).expand(2, 9, 4), torch.ones(52))[..., list(UPPER_INDICES)]
    second = flow.flow_loss(target + shift, valid, content, identity, affect, local, state, noise, time)
    torch.testing.assert_close(loss, second, atol=1e-6, rtol=1e-6)
    loss.backward()
    assert state.grad.abs().sum() > 0 and torch.isfinite(state.grad).all()
    assert local.grad.abs().sum() > 0 and torch.isfinite(local.grad).all()
    assert flow.state_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())


def test_upper_flow_decode_preserves_q_and_is_conditioned_and_seeded():
    flow, valid, content, identity, affect, local, state, noise = flow_fixture()
    output = flow.decode(valid, content, identity, affect, local, state, noise, steps=3)
    assert torch.isfinite(output).all() and output[~valid].count_nonzero() == 0
    torch.testing.assert_close(output, project_upper_innovation(output, valid, stride=4), atol=3e-7, rtol=1e-6)
    again = flow.decode(valid, content, identity, affect, local, state, noise, steps=3)
    torch.testing.assert_close(output, again, atol=0, rtol=0)
    changed_seed = flow.decode(valid, content, identity, affect, local, state, -noise, steps=3)
    changed_state = flow.decode(valid, content, identity, affect, local, state + 2., noise, steps=3)
    assert not torch.allclose(output, changed_seed)
    assert not torch.allclose(output, changed_state)
    output.square().mean().backward()
    assert torch.isfinite(flow.renderer.output.weight.grad).all()
