"""Native clock preservation and exact categorical scoring contracts."""
import pytest
import torch

from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior, categorical_energy_score


def controller(**kwargs):
    return ClockedMotionPrior(torch.zeros(3), torch.ones(3),
                              torch.zeros(2), torch.ones(2), hidden=4, k=5, **kwargs)


def inputs(batch=2, horizon=32):
    return torch.randn(batch, horizon, 3), torch.ones(batch, horizon, dtype=torch.bool), torch.randn(batch, 2)


def test_zero_initial_logits_give_common_uniform_distribution():
    model = controller()
    x, mask, global_ = inputs()
    output = model(x, mask, global_)
    assert output.shape == (2, 5)
    assert torch.count_nonzero(output) == 0
    torch.testing.assert_close(output.softmax(-1), torch.full((2, 5), .2))


def test_statistics_are_frozen_copies_and_not_fitted_on_query():
    mean, std = torch.arange(3.).requires_grad_(), torch.ones(3, requires_grad=True)
    model = ClockedMotionPrior(mean, std, torch.zeros(2), torch.ones(2), hidden=4, k=5)
    with torch.no_grad():
        mean.add_(100)
        std.mul_(3)
    assert not model.feature_mean.requires_grad
    assert not model.feature_std.requires_grad
    assert 'feature_mean' not in dict(model.named_parameters())
    torch.testing.assert_close(model.feature_mean, torch.arange(3.))
    torch.testing.assert_close(model.feature_std, torch.ones(3))


def test_masked_nan_and_inf_are_ignored_before_arithmetic_and_in_backward():
    torch.manual_seed(345)
    model = controller()
    with torch.no_grad():
        model.output.weight.normal_()
    x, mask, global_ = inputs()
    mask[0, 3:8] = False
    mask[1, 16:] = False
    clean = x.clone()
    x[~mask] = float('nan')
    x[0, 4] = float('inf')
    x.requires_grad_()
    output = model(x, mask, global_)
    torch.testing.assert_close(output, model(clean, mask, global_))
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[~mask].count_nonzero() == 0
    assert x.grad[mask].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_empty_audio_row_and_empty_bins_are_safe_global_only_fallback():
    model = controller()
    x, mask, global_ = inputs()
    mask[0] = False
    mask[1, 8:] = False
    x[~mask] = float('nan')
    descriptor = model.temporal_descriptor(x, mask)
    assert descriptor[0].count_nonzero() == 0
    assert descriptor[1, 4:].count_nonzero() == 0
    assert torch.isfinite(model(x, mask, global_)).all()


def test_observations_keep_absolute_bin_clock_instead_of_tail_stretching():
    model = controller()
    # Isolate exact bin means from learned convolution effects.
    with torch.no_grad():
        model.input.weight.fill_(1.)
        model.input.bias.zero_()
        for block in model.blocks:
            block.conv.weight.zero_()
            block.bias.zero_()
    x = torch.ones(1, 32, 3)
    prefix_mask = torch.zeros(1, 32, dtype=torch.bool)
    prefix_mask[:, :8] = True
    shifted_mask = prefix_mask.roll(8, dims=1)
    first = model.temporal_descriptor(x, prefix_mask).reshape(1, 4, 4)
    second = model.temporal_descriptor(x, shifted_mask).reshape(1, 4, 4)
    assert first[:, 1:].count_nonzero() == 0
    assert second[:, [0, 2, 3]].count_nonzero() == 0
    torch.testing.assert_close(first[:, 0], second[:, 1])
    assert not torch.equal(first, second)


def test_temporal_order_survives_pooling_and_can_change_logits():
    torch.manual_seed(456)
    model = controller()
    with torch.no_grad():
        model.output.weight.normal_()
    x, mask, global_ = inputs()
    assert not torch.allclose(model(x, mask, global_), model(x.flip(1), mask, global_))


def test_nonmultiple_horizon_bins_use_fixed_floor_boundaries():
    model = controller(horizon=11)
    with torch.no_grad():
        model.input.weight.fill_(1/3)
        model.input.bias.zero_()
        for block in model.blocks:
            block.conv.weight.zero_()
            block.bias.zero_()
    x = torch.arange(11.).reshape(1, 11, 1).repeat(1, 1, 3)
    actual = model.temporal_descriptor(x, torch.ones(1, 11, dtype=torch.bool)).reshape(4, 4)
    expected = torch.stack([torch.nn.functional.silu(torch.arange(float(a), float(b))).mean()
                            for a, b in ((0, 2), (2, 5), (5, 8), (8, 11))])
    torch.testing.assert_close(actual, expected[:, None].expand(4, 4))


def test_controller_learns_output_then_acoustic_encoder_from_score():
    torch.manual_seed(789)
    model = controller()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    x, mask, global_ = inputs()
    coordinates = torch.arange(5.)[:, None]
    pairwise = torch.cdist(coordinates, coordinates)
    target = pairwise[[0, 4]]
    loss = categorical_energy_score(model(x, mask, global_).softmax(-1), target, pairwise).mean()
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.input.weight.grad.count_nonzero() == 0
    optimizer.step()
    optimizer.zero_grad()
    categorical_energy_score(model(x, mask, global_).softmax(-1), target, pairwise).mean().backward()
    assert model.input.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('kind', ['observed_nan', 'wrong_horizon', 'bad_mask', 'wrong_global', 'global_nan', 'dtype'])
def test_bad_forward_inputs_fail_closed(kind):
    model = controller()
    x, mask, global_ = inputs()
    if kind == 'observed_nan':
        x[0, 0] = float('nan')
    elif kind == 'wrong_horizon':
        x, mask = x[:, :31], mask[:, :31]
    elif kind == 'bad_mask':
        mask = mask.float()
    elif kind == 'wrong_global':
        global_ = global_[:, :1]
    elif kind == 'global_nan':
        global_[0, 0] = float('nan')
    elif kind == 'dtype':
        x = x.double()
    with pytest.raises(ValueError):
        model(x, mask, global_)


def test_score_matches_exact_enumeration_and_analytic_two_token_case():
    p = torch.tensor([[.25, .75], [1., 0.]], dtype=torch.float64)
    distance = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.float64)
    target = torch.tensor([[0., 2.], [1., 1.]], dtype=torch.float64)
    actual = categorical_energy_score(p, target, distance)
    expected = torch.tensor([sum(p[b, k]*target[b, k] for k in range(2))
                             -.5*sum(p[b, k]*p[b, j]*distance[k, j]
                                     for k in range(2) for j in range(2)) for b in range(2)])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([1.125, 1.], dtype=torch.float64))


def test_expected_score_minimum_matches_known_true_distribution():
    distance = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.float64)
    q = torch.tensor([.3, .7], dtype=torch.float64)
    def expected(p):
        return (categorical_energy_score(p.expand(2, -1), distance, distance)*q).sum()
    optimum = expected(q)
    for p in (torch.tensor([.1, .9], dtype=torch.float64), torch.tensor([.7, .3], dtype=torch.float64)):
        assert expected(p) > optimum


def test_score_gradient_matches_double_precision_gradcheck():
    logits = torch.tensor([[.2, -.3, 1.]], dtype=torch.float64, requires_grad=True)
    tokens = torch.tensor([[0., 0.], [1., 0.], [1., 2.]], dtype=torch.float64)
    target = torch.cdist(torch.tensor([[.4, .3]], dtype=torch.float64), tokens)
    pairwise = torch.cdist(tokens, tokens)
    assert torch.autograd.gradcheck(lambda z: categorical_energy_score(z.softmax(-1), target, pairwise), (logits,))


@pytest.mark.parametrize('kind', ['unnormalized', 'negative', 'nan', 'asymmetric', 'diagonal', 'shape', 'dtype'])
def test_invalid_score_inputs_are_rejected(kind):
    p = torch.tensor([[.25, .75]])
    target = torch.tensor([[0., 2.]])
    distance = torch.tensor([[0., 2.], [2., 0.]])
    if kind == 'unnormalized':
        p *= .5
    elif kind == 'negative':
        p = torch.tensor([[-.1, 1.1]])
    elif kind == 'nan':
        target[0, 0] = float('nan')
    elif kind == 'asymmetric':
        distance[0, 1] = 3.
    elif kind == 'diagonal':
        distance[0, 0] = 1.
    elif kind == 'shape':
        target = target[:, :1]
    elif kind == 'dtype':
        distance = distance.double()
    with pytest.raises(ValueError):
        categorical_energy_score(p, target, distance)
