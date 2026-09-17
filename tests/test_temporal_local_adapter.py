"""Temporal adapter contracts independent of any motion or global condition."""

import copy

import pytest
import torch

from kinetalk_b0.models.temporal_local_adapter import TemporalLocalAdapter


def fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(271)
    local = torch.randn(3, 11, 64, generator=generator, dtype=dtype)
    valid = torch.ones(3, 11, dtype=torch.bool)
    valid[0, 2:4] = False
    valid[1, -3:] = False
    valid[2, :-1] = False
    local[0, 0, 0] = -0.0
    local[~valid] = float('nan')
    return local, valid


def bitwise_equal(left, right):
    return torch.equal(left.contiguous().view(torch.uint8),
                       right.contiguous().view(torch.uint8))


def randomize_up(model):
    generator = torch.Generator().manual_seed(117)
    with torch.no_grad():
        model.up.weight.copy_(torch.randn(model.up.weight.shape,
                                         generator=generator,
                                         dtype=model.up.weight.dtype) * .1)
    return model


def test_constructor_preserves_random_stream_and_has_only_1024_parameters():
    before = torch.random.get_rng_state().clone()
    model = TemporalLocalAdapter()
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == 1024
    assert set(model.state_dict()) == {'down.weight', 'up.weight'}
    assert model.down.bias is None and model.up.bias is None
    assert torch.count_nonzero(model.up.weight) == 0
    other = TemporalLocalAdapter()
    for key, value in model.state_dict().items():
        assert bitwise_equal(value, other.state_dict()[key])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_zero_init_preserves_every_input_bit_without_mutation(dtype):
    local, valid = fixture(dtype)
    before = local.clone()
    result = TemporalLocalAdapter().to(dtype=dtype)(local, valid)
    assert bitwise_equal(result, local)
    assert bitwise_equal(local, before)
    assert torch.signbit(result[0, 0, 0])


@pytest.mark.parametrize('dtype,atol', [(torch.float32, 2e-7), (torch.float64, 1e-15)])
def test_trained_adapter_preserves_per_clip_mean_and_invalid_payloads(dtype, atol):
    local, valid = fixture(dtype)
    model = randomize_up(TemporalLocalAdapter().to(dtype=dtype))
    result = model(local, valid)
    assert bitwise_equal(result[~valid], local[~valid])
    assert torch.isfinite(result[valid]).all()
    assert not bitwise_equal(result[valid], local[valid])
    for row in range(len(local)):
        torch.testing.assert_close(result[row, valid[row]].mean(dim=0),
                                   local[row, valid[row]].mean(dim=0),
                                   atol=atol, rtol=0)
    assert bitwise_equal(result[2], local[2])


def test_invalid_payload_does_not_affect_observed_output_or_parameter_gradient():
    local, valid = fixture(torch.float64)
    local = local.requires_grad_()
    alternate = torch.where(valid[..., None], local.detach(),
                            torch.full_like(local, float('inf'))).requires_grad_()
    left = randomize_up(TemporalLocalAdapter().double())
    right = copy.deepcopy(left)
    a = left(local, valid)
    b = right(alternate, valid)
    assert bitwise_equal(a[valid], b[valid])
    assert bitwise_equal(b[~valid], alternate[~valid])
    a[valid].square().sum().backward()
    b[valid].square().sum().backward()
    assert torch.isfinite(local.grad).all()
    assert torch.count_nonzero(local.grad[~valid]) == 0
    torch.testing.assert_close(local.grad, alternate.grad, atol=0, rtol=0)
    for lp, rp in zip(left.parameters(), right.parameters()):
        assert torch.isfinite(lp.grad).all()
        torch.testing.assert_close(lp.grad, rp.grad, atol=0, rtol=0)


def test_first_step_updates_up_then_down_receives_finite_nonzero_gradient():
    local, valid = fixture(torch.float64)
    model = TemporalLocalAdapter().double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0)
    initial_down = model.down.weight.detach().clone()
    model(local, valid)[valid].square().mean().backward()
    assert torch.isfinite(model.up.weight.grad).all()
    assert model.up.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(model.down.weight.grad) == 0
    optimizer.step()
    assert torch.count_nonzero(model.up.weight) > 0
    assert bitwise_equal(model.down.weight, initial_down)
    optimizer.zero_grad(set_to_none=True)
    model(local, valid)[valid].square().mean().backward()
    assert torch.isfinite(model.down.weight.grad).all()
    assert model.down.weight.grad.abs().sum() > 0


def test_centers_full_clip_and_does_not_mix_batch_examples():
    local, valid = fixture(torch.float64)
    model = randomize_up(TemporalLocalAdapter().double())
    result = model(local, valid)
    for row in range(len(local)):
        alone = model(local[row:row + 1], valid[row:row + 1])
        torch.testing.assert_close(alone, result[row:row + 1],
                                   atol=1e-15, rtol=0, equal_nan=True)
    changed = local.clone()
    changed[0, -1] += 1
    updated = model(changed, valid)
    assert not bitwise_equal(updated[0, 0], result[0, 0])
    assert bitwise_equal(updated[1:], result[1:])


def test_feature_offset_does_not_change_temporal_correction():
    local, valid = fixture(torch.float64)
    model = randomize_up(TemporalLocalAdapter().double())
    shift = torch.linspace(-1, 1, 64, dtype=local.dtype).reshape(1, 1, 64)
    a = model(local, valid)
    b = model(local + shift, valid)
    torch.testing.assert_close((b - (local + shift))[valid],
                               (a - local)[valid], atol=1e-15, rtol=0)


def test_forward_backward_are_deterministic_without_rng_consumption():
    local, valid = fixture(torch.float64)
    model = randomize_up(TemporalLocalAdapter().double())
    before = torch.random.get_rng_state().clone()
    a = model(local, valid)
    a[valid].square().sum().backward()
    gradient = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    b = model(local, valid)
    b[valid].square().sum().backward()
    assert bitwise_equal(a, b)
    assert torch.equal(before, torch.random.get_rng_state())
    for name, parameter in model.named_parameters():
        assert bitwise_equal(parameter.grad, gradient[name])


@pytest.mark.parametrize('corruption', [
    'observed_nan', 'observed_inf', 'all_invalid', 'float_mask', 'bad_mask_shape',
    'bad_feature_dimension', 'bad_feature_rank', 'integer_features', 'empty_batch',
    'empty_time',
])
def test_rejects_invalid_observations_and_mask_contract(corruption):
    local, valid = fixture()
    if corruption == 'observed_nan':
        local[0, 0, 0] = float('nan')
    elif corruption == 'observed_inf':
        local[0, 0, 0] = float('inf')
    elif corruption == 'all_invalid':
        valid[0] = False
    elif corruption == 'float_mask':
        valid = valid.float()
    elif corruption == 'bad_mask_shape':
        valid = valid[:, :-1]
    elif corruption == 'bad_feature_dimension':
        local = local[..., :-1]
    elif corruption == 'bad_feature_rank':
        local = local[0]
    elif corruption == 'integer_features':
        local = torch.zeros_like(local, dtype=torch.int64)
    elif corruption == 'empty_batch':
        local, valid = local[:0], valid[:0]
    elif corruption == 'empty_time':
        local, valid = local[:, :0], valid[:, :0]
    with pytest.raises(ValueError, match='Finite observed local'):
        TemporalLocalAdapter()(local, valid)


@pytest.mark.parametrize('local_dim,rank', [(0, 8), (64, 0), (4, 8), (64, 1.5), (True, 1)])
def test_rejects_invalid_dimensions(local_dim, rank):
    with pytest.raises(ValueError, match='integer dimensions'):
        TemporalLocalAdapter(local_dim, rank)
