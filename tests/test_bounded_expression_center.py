import pytest
import torch

from kinetalk_b0.models.bounded_expression_center import BoundedExpressionCenter, compose_bounded_center
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face


def bit_equal(left, right):
    integer = torch.int64 if left.dtype == torch.float64 else torch.int32
    return torch.equal(left.contiguous().view(integer), right.contiguous().view(integer))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_head_zero_initialization_and_anchor_relative_extreme_bound(dtype):
    model = BoundedExpressionCenter(global_dim=3, identity_dim=2, hidden=5).to(dtype)
    global_code = torch.ones(2, 3, dtype=dtype)
    identity = torch.zeros(2, 2, dtype=dtype)
    anchor = torch.linspace(0, 1, 9, dtype=dtype)[None].expand(2, -1).clone()
    expected = anchor.clamp(.01, .99)
    torch.testing.assert_close(model(global_code, identity, anchor), expected)
    assert model.export_config() == dict(global_dim=3, identity_dim=2, hidden=5,
                                         max_logit_delta=6., anchor_eps=.01)
    with torch.no_grad():
        model.head.bias.copy_(torch.linspace(-1e5, 1e5, 9, dtype=dtype))
    output = model(global_code, identity, anchor)
    assert torch.isfinite(output).all() and ((output >= 0) & (output <= 1)).all()
    logits = torch.logit(expected)
    lower, upper = torch.sigmoid(logits-6), torch.sigmoid(logits+6)
    assert (output >= lower).all() and (output <= upper).all()


def test_head_gradients_are_finite_and_last_layer_learns_from_zero():
    torch.manual_seed(11)
    model = BoundedExpressionCenter(global_dim=3, identity_dim=2, hidden=5).double()
    global_code = torch.randn(4, 3, dtype=torch.float64)
    identity = torch.randn(4, 2, dtype=torch.float64)
    anchor = torch.full((4, 9), .2, dtype=torch.float64)
    (model(global_code, identity, anchor)-.65).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_composer_exact_center_and_constant_channel_scale_with_gaps(dtype):
    clock = torch.linspace(-1, 1, 7, dtype=dtype)[None, :, None]
    amplitude = torch.linspace(.05, .45, 9, dtype=dtype)[None, None]
    base = (.5 + clock*amplitude).expand(2, -1, -1).clone()
    valid = torch.tensor([[True, True, False, True, False, True, True],
                          [False, True, True, True, True, False, False]])
    base[~valid] = float('nan')
    base[:, :, 0] = torch.where(valid, torch.full_like(base[:, :, 0], .37), base[:, :, 0])
    center = torch.linspace(0, 1, 9, dtype=dtype)[None].expand(2, -1).clone()
    result = compose_bounded_center(base, center, valid)
    output = result['upper']
    assert torch.isfinite(output[valid]).all()
    assert ((output[valid] >= 0) & (output[valid] <= 1)).all()
    assert bit_equal(output[~valid], base[~valid])
    assert torch.equal(result['scale'][:, 0], torch.ones(2, dtype=dtype))
    assert torch.equal(result['scale'][:, -1], torch.zeros(2, dtype=dtype))
    for row in range(2):
        x, y = base[row, valid[row]], output[row, valid[row]]
        torch.testing.assert_close(y.mean(0), center[row], atol=2e-7 if dtype == torch.float32 else 2e-15, rtol=0)
        torch.testing.assert_close(result['original_mean'][row], x.mean(0))
        torch.testing.assert_close(y-center[row], (x-x.mean(0))*result['scale'][row],
                                   atol=2e-7 if dtype == torch.float32 else 2e-15, rtol=0)
    assert (result['scale'] >= 0).all() and (result['scale'] <= 1).all()


def test_padding_invariance_and_constant_sequence():
    base = torch.tensor([.1, .3, .8], dtype=torch.float64)[None, :, None].expand(1, 3, 9).clone()
    center = torch.full((1, 9), .25, dtype=torch.float64)
    first = compose_bounded_center(base, center, torch.ones(1, 3, dtype=torch.bool))
    padded = torch.full((1, 8, 9), float('nan'), dtype=torch.float64)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, [1, 4, 5]] = True
    padded[mask] = base[0]
    second = compose_bounded_center(padded, center, mask)
    assert bit_equal(first['upper'][0], second['upper'][mask])
    assert bit_equal(first['scale'], second['scale'])
    flat = torch.full((2, 13, 9), .37, dtype=torch.float64)
    wanted = torch.linspace(0, 1, 9, dtype=torch.float64)[None].expand(2, -1)
    result = compose_bounded_center(flat, wanted, torch.ones(2, 13, dtype=torch.bool))
    assert bit_equal(result['upper'], wanted[:, None].expand_as(flat))
    assert torch.equal(result['scale'], torch.ones_like(wanted))


def test_composer_gradients_finite_with_nan_padding_flat_channels_and_endpoints():
    base = torch.tensor([.1, .4, .8, float('nan')], dtype=torch.float64)[None, :, None].expand(1, 4, 9).clone()
    base[:, :3, 0] = .2
    base.requires_grad_()
    center = torch.linspace(0, 1, 9, dtype=torch.float64)[None].clone().requires_grad_()
    valid = torch.tensor([[True, True, True, False]])
    output = compose_bounded_center(base, center, valid)['upper']
    output[valid].square().mean().backward()
    assert torch.isfinite(base.grad).all() and torch.isfinite(center.grad).all()
    assert torch.equal(base.grad[~valid], torch.zeros_like(base.grad[~valid]))
    assert center.grad.abs().sum() > 0


@pytest.mark.parametrize('dtype,amplitude', [(torch.float32, 1e-44), (torch.float64, 1e-320)])
def test_subnormal_residuals_do_not_create_infinite_backward(dtype, amplitude):
    base = torch.tensor([0., amplitude], dtype=dtype)[None, :, None].expand(1, 2, 9).clone().requires_grad_()
    center = torch.zeros(1, 9, dtype=dtype, requires_grad=True)
    result = compose_bounded_center(base, center, torch.ones(1, 2, dtype=torch.bool))
    assert ((result['upper'] >= 0) & (result['upper'] <= 1)).all()
    result['upper'].sum().backward()
    assert torch.isfinite(base.grad).all() and torch.isfinite(center.grad).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_existing_full_composition_preserves_other43_and_all_padding(dtype):
    torch.manual_seed(17)
    base = torch.rand(3, 6, 52, dtype=dtype)
    valid = torch.tensor([[True, False, True, True, False, True],
                          [True, True, True, False, False, False],
                          [False, False, True, True, True, True]])
    base[~valid] = float('nan')
    upper = compose_bounded_center(base[..., list(UPPER_INDICES)],
                                   torch.full((3, 9), .15, dtype=dtype), valid)['upper']
    full = compose_upper_face(base, upper, valid)
    other = [i for i in range(52) if i not in UPPER_INDICES]
    assert bit_equal(base[..., other], full[..., other])
    assert bit_equal(base[~valid], full[~valid])
    assert bit_equal(upper[valid], full[..., list(UPPER_INDICES)][valid])


@pytest.mark.parametrize('kwargs', [dict(global_dim=0), dict(hidden=True),
    dict(max_logit_delta=0), dict(max_logit_delta=float('nan')), dict(anchor_eps=.5)])
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BoundedExpressionCenter(**kwargs)


@pytest.mark.parametrize('field', ['global', 'identity', 'anchor_nan', 'anchor_range', 'dtype'])
def test_head_rejects_invalid_conditions(field):
    model = BoundedExpressionCenter(global_dim=3, identity_dim=2)
    g, s, a = torch.zeros(2, 3), torch.zeros(2, 2), torch.full((2, 9), .2)
    if field == 'global': g[0, 0] = float('inf')
    elif field == 'identity': s = s[:, :1]
    elif field == 'anchor_nan': a[0, 0] = float('nan')
    elif field == 'anchor_range': a[0, 0] = -.1
    else: a = a.double()
    with pytest.raises(ValueError):
        model(g, s, a)


@pytest.mark.parametrize('field', ['base_nan', 'base_range', 'center_nan', 'center_range', 'empty', 'mask'])
def test_composer_rejects_invalid_observed_values(field):
    base = torch.full((2, 3, 9), .5)
    center = torch.full((2, 9), .4)
    valid = torch.ones(2, 3, dtype=torch.bool)
    if field == 'base_nan': base[0, 0, 0] = float('nan')
    elif field == 'base_range': base[0, 0, 0] = 1.1
    elif field == 'center_nan': center[0, 0] = float('nan')
    elif field == 'center_range': center[0, 0] = -.1
    elif field == 'empty': valid[0] = False
    else: valid = valid.float()
    with pytest.raises(ValueError):
        compose_bounded_center(base, center, valid)
