import copy

import pytest
import torch
from torch.nn import functional as F

from kinetalk_b0.models.semantic_upper_flow import SemanticUpperFlow, semantic_intervention


def sample():
    torch.manual_seed(926)
    model = SemanticUpperFlow(semantic_dim=3, global_dim=4, identity_dim=2, hidden=16, depth=4)
    valid = torch.ones(2, 19, dtype=torch.bool); valid[0, -4:] = False
    semantic = torch.randn(2, 19, 3)
    global_condition = torch.randn(2, 4)
    identity = torch.randn(2, 2)
    noise = torch.randn(2, 19, 9)
    return model, valid, semantic, global_condition, identity, noise


def test_single_fm_matches_observed_velocity_mse_and_condition_gradients():
    model, valid, semantic, glob, identity, noise = sample()
    semantic.requires_grad_(); glob.requires_grad_(); identity.requires_grad_()
    target = torch.randn_like(noise); time = torch.tensor([.25, .7])
    loss = model.flow_loss(target, valid, semantic, glob, identity, noise, time)
    conditions = model.prepare_conditions(valid, semantic, glob, identity)
    f = time[:, None, None]
    pred = model.velocity((1-f)*noise+f*target, time, conditions)
    torch.testing.assert_close(loss, (pred-(target-noise))[valid].square().mean())
    loss.backward()
    for value in (semantic, glob, identity):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert torch.count_nonzero(semantic.grad[~valid]) == 0
    for projection in (model.semantic_projection, model.global_projection, model.identity_projection):
        assert projection.weight.grad.abs().sum() > 0


def test_loss_mask_only_changes_loss_support_not_native_field():
    model, valid, semantic, glob, identity, noise = sample()
    target = torch.randn_like(noise); time = torch.tensor([.25, .7])
    loss_mask = valid.clone(); loss_mask[:, 4:8] = False
    conditions = model.prepare_conditions(valid, semantic, glob, identity)
    f = time[:, None, None]
    velocity = model.velocity((1-f)*noise+f*target, time, conditions)
    expected = (velocity-(target-noise))[loss_mask].square().mean()
    actual = model.flow_loss(target, valid, semantic, glob, identity, noise, time, loss_mask=loss_mask)
    torch.testing.assert_close(expected, actual)
    bad = loss_mask.clone(); bad[0, -1] = True
    with pytest.raises(ValueError, match='loss_mask'):
        model.flow_loss(target, valid, semantic, glob, identity, noise, time, loss_mask=bad)
    with pytest.raises(ValueError, match='loss_mask'):
        model.flow_loss(target, valid, semantic, glob, identity, noise, time, loss_mask=torch.zeros_like(valid))


def test_joint_temporal_field_responds_to_neighbor_motion_and_each_condition():
    model, valid, semantic, glob, identity, noise = sample()
    conditions = model.prepare_conditions(valid, semantic, glob, identity)
    time = torch.tensor([.25, .7])
    base = model.velocity(noise, time, conditions)
    perturbed = noise.clone(); perturbed[0, 8] += 3
    result = model.velocity(perturbed, time, conditions)
    assert (base[0, 7]-result[0, 7]).abs().max() > 1e-7
    for new_sem, new_glob, new_id in ((semantic.flip(1), glob, identity),
                                     (semantic, glob+1, identity), (semantic, glob, identity+1)):
        changed = model.velocity(noise, time, model.prepare_conditions(valid, new_sem, new_glob, new_id))
        assert not torch.allclose(base[valid], changed[valid], rtol=1e-6, atol=1e-7)


def test_invalid_nan_padding_is_ignored_without_mutating_inputs():
    model, valid, semantic, glob, identity, noise = sample()
    target = torch.randn_like(noise); time = torch.tensor([.2, .5])
    original = model.flow_loss(target, valid, semantic, glob, identity, noise, time)
    for value in (semantic, noise, target):
        value[~valid] = float('nan')
    before = [x.clone() for x in (semantic, noise, target)]
    actual = model.flow_loss(target, valid, semantic, glob, identity, noise, time)
    torch.testing.assert_close(original, actual, rtol=0, atol=0)
    decoded = model.decode(valid, semantic, glob, identity, noise, steps=3)
    assert torch.isfinite(decoded).all() and torch.count_nonzero(decoded[~valid]) == 0
    for value, snapshot in zip((semantic, noise, target), before):
        torch.testing.assert_close(value, snapshot, equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize('use_seed', [False, True])
def test_appended_padding_does_not_change_valid_trajectory(use_seed):
    model, valid, semantic, glob, identity, noise = sample()
    extra = 7
    padded_valid = F.pad(valid, (0, extra), value=False)
    padded_semantic = F.pad(semantic, (0, 0, 0, extra), value=float('nan'))
    padded_noise = F.pad(noise, (0, 0, 0, extra), value=float('nan'))
    kwargs = {'seed': 421} if use_seed else {'noise': noise}
    padded_kwargs = {'seed': 421} if use_seed else {'noise': padded_noise}
    a = model.decode(valid, semantic, glob, identity, steps=3, **kwargs)
    b = model.decode(padded_valid, padded_semantic, glob, identity, steps=3, **padded_kwargs)
    torch.testing.assert_close(a, b[:, :valid.shape[1]], rtol=2e-6, atol=2e-6)
    assert torch.count_nonzero(b[~padded_valid]) == 0


def test_seeded_sampling_is_repeatable_and_does_not_consume_global_rng():
    model, valid, semantic, glob, identity, noise = sample()
    state = torch.random.get_rng_state().clone()
    a = model.decode(valid, semantic, glob, identity, seed=421, steps=3)
    b = model.decode(valid, semantic, glob, identity, seed=421, steps=3)
    c = model.decode(valid, semantic, glob, identity, seed=422, steps=3)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(a, b) and not torch.equal(a, c)
    restored = copy.deepcopy(model)
    assert torch.equal(a, restored.decode(valid, semantic, glob, identity, seed=421, steps=3))


def test_temporal_taps_do_not_jump_over_missing_runs():
    model, valid, semantic, glob, identity, noise = sample()
    valid[:, 8] = False
    conditions = model.prepare_conditions(valid, semantic, glob, identity)
    base = model.velocity(noise, torch.tensor([.2, .5]), conditions)
    changed_noise = noise.clone(); changed_noise[:, :8] += 100
    changed = model.velocity(changed_noise, torch.tensor([.2, .5]), conditions)
    torch.testing.assert_close(base[:, 9:], changed[:, 9:], rtol=0, atol=0)


def test_static_and_reverse_preserve_support_and_never_cross_gaps():
    valid = torch.tensor([[True, True, False, True, True, True, False]])
    sem = torch.tensor([[[1.], [3.], [float('nan')], [5.], [6.], [10.], [float('nan')]]])
    static = semantic_intervention(sem, valid, 'static')
    reverse = semantic_intervention(sem, valid, 'reverse')
    torch.testing.assert_close(static, torch.tensor([[[2.], [2.], [0.], [7.], [7.], [7.], [0.]]]))
    torch.testing.assert_close(reverse, torch.tensor([[[3.], [1.], [0.], [10.], [6.], [5.], [0.]]]))
    torch.testing.assert_close(semantic_intervention(reverse, valid, 'reverse'), semantic_intervention(sem, valid))


def test_position_is_native_not_normalized_by_batch_padding_and_length_is_explicit():
    model, valid, semantic, glob, identity, noise = sample()
    # A zero temporal input still distinguishes native positions.
    valid[:] = True; semantic[:] = 0; noise[:] = 0
    conditions = model.prepare_conditions(valid, semantic, glob, identity)
    velocity = model.velocity(noise, torch.zeros(2), conditions)
    assert not torch.allclose(velocity[:, 5], velocity[:, 6])
    assert 'length_projection.weight' in model.state_dict()


@pytest.mark.parametrize('kind', ['empty_row', 'semantic_nan', 'semantic_shape', 'identity_dtype', 'bad_time'])
def test_invalid_observed_contracts_rejected(kind):
    model, valid, semantic, glob, identity, noise = sample()
    time = torch.tensor([.2, .5])
    if kind == 'empty_row': valid[0] = False
    if kind == 'semantic_nan': semantic[0, 0, 0] = float('nan')
    if kind == 'semantic_shape': semantic = semantic[:, :, :2]
    if kind == 'identity_dtype': identity = identity.double()
    if kind == 'bad_time': time[0] = 1.1
    with pytest.raises(ValueError):
        model.flow_loss(noise, valid, semantic, glob, identity, noise, time)


@pytest.mark.parametrize('kwargs', [{}, {'seed': 1, 'noise': torch.zeros(2, 19, 9)},
                                    {'seed': 1, 'steps': 0}, {'seed': True}])
def test_sampling_requires_explicit_valid_randomness_and_steps(kwargs):
    model, valid, semantic, glob, identity, noise = sample()
    with pytest.raises(ValueError):
        model.decode(valid, semantic, glob, identity, **kwargs)


def test_default_model_is_small_and_does_not_apply_output_sigmoid_or_clip():
    model = SemanticUpperFlow(7)
    assert len(model.blocks) == 4 and model.hidden == 64
    assert sum(p.numel() for p in model.parameters()) < 200_000
    for parameter in model.parameters(): parameter.data.zero_()
    valid = torch.ones(1, 3, dtype=torch.bool)
    noise = torch.full((1, 3, 9), -3.)
    output = model.decode(valid, torch.zeros(1, 3, 7), torch.zeros(1, 65), torch.zeros(1, 128), noise, steps=2)
    torch.testing.assert_close(output, noise, rtol=0, atol=0)
