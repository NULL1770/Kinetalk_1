"""Contracts for the audio-only context and duration-conditioned process prior."""
import inspect

import pytest
import torch

from kinetalk_b0.models.motion_process_prior import MotionProcessPrior, initial_nll, joint_event_nll


def fixture():
    generator = torch.Generator().manual_seed(217)
    value = torch.randn(3, 17, 6, generator=generator, dtype=torch.float64)
    valid = torch.ones(3, 17, dtype=torch.bool)
    valid[0, 5:7] = False
    valid[1, -4:] = False
    valid[2, :2] = False
    valid[2, 3:15] = False
    value[~valid] = float('nan')
    return value, valid


def model():
    with torch.random.fork_rng():
        torch.manual_seed(821)
        return MotionProcessPrior(feature_dim=6, hidden=12).double()


def event_args(prior):
    value, valid = fixture()
    hidden, pooled = prior.forward_context(value, valid, local_enabled=True)
    return hidden[:, 2], pooled, torch.tensor([0, 2, 3]), torch.tensor([0., 1., -.5], dtype=value.dtype), \
        torch.tensor([.1, -.2, .3], dtype=value.dtype), torch.tensor([0, 8, 20])


def test_context_accepts_only_audio_mask_and_policy_and_does_not_mutate_inputs():
    assert list(inspect.signature(MotionProcessPrior.forward_context).parameters) == ['self', 'features', 'valid', 'local_enabled']
    value, valid = fixture()
    before, before_mask = value.clone(), valid.clone()
    prior = model()
    hidden, pooled = prior.forward_context(value, valid, True)
    assert hidden.shape == (3, 17, 12) and pooled.shape == (3, 12)
    assert torch.isfinite(hidden).all() and torch.isfinite(pooled).all()
    assert torch.count_nonzero(hidden[~valid]) == 0
    torch.testing.assert_close(value, before, atol=0, rtol=0, equal_nan=True)
    assert torch.equal(valid, before_mask)


@pytest.mark.parametrize('local_enabled', [False, True])
def test_invalid_payload_and_extra_padding_leave_observed_outputs_and_pool_unchanged(local_enabled):
    value, valid = fixture()
    prior = model()
    hidden, pooled = prior.forward_context(value, valid, local_enabled)
    alternate = torch.where(valid[..., None], value, torch.full_like(value, float('inf')))
    alternate = torch.cat((alternate, torch.full((3, 8, 6), float('nan'), dtype=value.dtype)), 1)
    new_valid = torch.cat((valid, torch.zeros(3, 8, dtype=torch.bool)), 1)
    h2, p2 = prior.forward_context(alternate, new_valid, local_enabled)
    torch.testing.assert_close(h2[:, :17], hidden, atol=2e-15, rtol=0)
    torch.testing.assert_close(p2, pooled, atol=2e-15, rtol=0)


def test_local_hidden_matches_separately_encoded_contiguous_runs_even_with_dilation():
    value, valid = fixture()
    prior = model()
    hidden, _ = prior.forward_context(value, valid, True)
    for row, begin, end in ((0, 0, 5), (0, 7, 17), (1, 0, 13), (2, 2, 3), (2, 15, 17)):
        isolated, _ = prior.forward_context(value[row:row+1, begin:end], torch.ones(1, end-begin, dtype=torch.bool), True)
        torch.testing.assert_close(hidden[row, begin:end], isolated[0], atol=2e-15, rtol=0)
    changed = value.clone()
    changed[0, 7:] *= 100
    modified, _ = prior.forward_context(changed, valid, True)
    torch.testing.assert_close(hidden[0, :5], modified[0, :5], atol=0, rtol=0)


def test_global_only_is_time_constant_across_gaps_edges_and_independent_of_audio_order():
    value, valid = fixture()
    prior = model()
    hidden, pooled = prior.forward_context(value, valid, False)
    reversed_value = value.clone()
    for row in range(3):
        observed = hidden[row, valid[row]]
        torch.testing.assert_close(observed, observed[:1].expand_as(observed), atol=0, rtol=0)
        torch.testing.assert_close(pooled[row], observed[0], atol=2e-15, rtol=0)
        reversed_value[row, valid[row]] = value[row, valid[row]].flip(0)
    reversed_hidden, reversed_pooled = prior.forward_context(reversed_value, valid, False)
    torch.testing.assert_close(reversed_hidden, hidden, atol=2e-15, rtol=0)
    torch.testing.assert_close(reversed_pooled, pooled, atol=2e-15, rtol=0)


def test_local_and_global_arms_share_exact_global_context_and_initial_distribution():
    value, valid = fixture()
    prior = model()
    local_hidden, local_pooled = prior.forward_context(value, valid, True)
    global_hidden, global_pooled = prior.forward_context(value, valid, False)
    assert torch.equal(local_pooled, global_pooled)
    local_initial = prior.initial_distribution(local_pooled)
    global_initial = prior.initial_distribution(global_pooled)
    for key in local_initial:
        assert torch.equal(local_initial[key], global_initial[key])
    assert not torch.equal(local_hidden[valid], global_hidden[valid])


def test_padding_never_receives_gradient_and_event_likelihood_trains_all_subsystems():
    prior = model()
    value, valid = fixture()
    value.requires_grad_()
    hidden, pooled = prior.forward_context(value, valid, True)
    distribution = prior.event_distribution(hidden[:, 2], pooled, torch.tensor([0, 1, 3]),
        torch.tensor([.4, -.8, 1.1], dtype=value.dtype), torch.tensor([.1, .2, -.2], dtype=value.dtype),
        torch.tensor([4, 8, 20]))
    initial = prior.initial_distribution(pooled)
    loss = joint_event_nll(distribution, torch.tensor([0, 2, 4]), torch.tensor([.2, -.1, .8], dtype=value.dtype)).mean()
    loss = loss + initial_nll(initial, torch.zeros_like(initial['loc'])).mean()
    loss.backward()
    assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(value.grad[~valid]) == 0
    assert value.grad[valid].abs().sum() > 0
    for name, parameter in prior.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_invalid_payload_does_not_change_parameter_gradients():
    value, valid = fixture()
    alternate = torch.where(valid[..., None], value, torch.full_like(value, float('inf')))
    left, right = model(), model()
    for prior, features in ((left, value), (right, alternate)):
        hidden, pooled = prior.forward_context(features, valid, True)
        (hidden.square().mean() + pooled.square().mean()).backward()
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)


def test_event_and_initial_nll_match_explicit_gaussian_joint_formula():
    logits = torch.tensor([[.2, -.1], [.7, -.2]], dtype=torch.float64)
    loc = torch.tensor([[.1, .6], [-.2, .3]], dtype=torch.float64)
    log_scale = torch.tensor([[-.2, .4], [.1, -.6]], dtype=torch.float64)
    duration = torch.tensor([1, 0])
    delta = torch.tensor([.3, -.8], dtype=torch.float64)
    out = {'duration_logits': logits, 'loc': loc, 'log_scale': log_scale}
    expected = []
    for i, j in enumerate(duration):
        gaussian = torch.distributions.Normal(loc[i, j], log_scale[i, j].exp())
        expected.append(-logits.log_softmax(-1)[i, j] - gaussian.log_prob(delta[i]))
    torch.testing.assert_close(joint_event_nll(out, duration, delta), torch.stack(expected), atol=1e-15, rtol=0)
    truth = torch.tensor([[.4, -.2], [0., 1.1]], dtype=torch.float64)
    expected_initial = -torch.distributions.Normal(loc, log_scale.exp()).log_prob(truth).sum(-1)
    torch.testing.assert_close(initial_nll(out, truth), expected_initial, atol=1e-15, rtol=0)


def test_distribution_shapes_and_log_standard_deviation_bounds():
    prior = model()
    with torch.no_grad():
        prior.event_head[-1].bias[14:] = torch.tensor([-100., 100., 0., -4., 4., -3., 2.])
        prior.initial_head[-1].bias[4:] = torch.tensor([-100., 100., -5., 5.])
    args = event_args(prior)
    out = prior.event_distribution(*args)
    assert all(v.shape == (3, 7) for v in out.values())
    initial = prior.initial_distribution(args[1])
    assert all(v.shape == (3, 4) for v in initial.values())
    for result in (out, initial):
        assert (result['log_scale'] >= -3).all() and (result['log_scale'] <= 2).all()


def test_explicit_generator_reproduces_event_and_initial_samples_without_global_rng_consumption():
    prior = model()
    args = event_args(prior)
    global_rng = torch.random.get_rng_state().clone()
    first = torch.Generator().manual_seed(731)
    second = torch.Generator().manual_seed(731)
    a = prior.sample_event(*args, generator=first)
    b = prior.sample_event(*args, generator=second)
    assert set(a) == {'duration_index', 'duration', 'delta'}
    for key in a:
        assert torch.equal(a[key], b[key])
        assert a[key].shape == (3,)
    assert torch.equal(a['duration'], prior.durations[a['duration_index']])
    ia, ib = prior.sample_initial(args[1], generator=first), prior.sample_initial(args[1], generator=second)
    assert torch.equal(ia, ib) and ia.shape == (3, 4)
    assert not a['delta'].requires_grad and not ia.requires_grad
    assert torch.equal(global_rng, torch.random.get_rng_state())
    c = prior.sample_event(*args, generator=torch.Generator().manual_seed(99))
    assert not torch.equal(a['delta'], c['delta'])


@pytest.mark.parametrize('corruption', ['nan', 'all_invalid', 'float_mask', 'bad_width', 'empty_time', 'bad_policy'])
def test_context_rejects_invalid_observation_contract(corruption):
    value, valid = fixture()
    policy = True
    if corruption == 'nan': value[0, 0] = float('nan')
    elif corruption == 'all_invalid': valid[1] = False
    elif corruption == 'float_mask': valid = valid.double()
    elif corruption == 'bad_width': value = value[..., :5]
    elif corruption == 'empty_time': value, valid = value[:, :0], valid[:, :0]
    elif corruption == 'bad_policy': policy = 1
    with pytest.raises(ValueError): model().forward_context(value, valid, policy)


@pytest.mark.parametrize('kwargs', [{'feature_dim': 0}, {'hidden': True}, {'groups': -1},
    {'durations': ()}, {'durations': (4, 4)}, {'durations': (8, 4)}, {'durations': (4., 8.)}])
def test_constructor_rejects_invalid_dimensions_or_duration_support(kwargs):
    with pytest.raises(ValueError): MotionProcessPrior(**kwargs)


@pytest.mark.parametrize('corruption', ['group', 'negative_duration', 'state_shape', 'state_nan'])
def test_event_rejects_invalid_explicit_history(corruption):
    prior = model()
    args = list(event_args(prior))
    if corruption == 'group': args[2] = torch.tensor([0, 4, 2])
    elif corruption == 'negative_duration': args[5] = torch.tensor([0, -4, 8])
    elif corruption == 'state_shape': args[3] = args[3][:2]
    elif corruption == 'state_nan': args[4][0] = float('nan')
    with pytest.raises(ValueError): prior.event_distribution(*args)
