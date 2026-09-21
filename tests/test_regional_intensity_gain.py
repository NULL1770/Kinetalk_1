import pytest
import torch

from kinetalk_b0.models.mean_preserving_upper import UPPER_INDICES
from kinetalk_b0.models.regional_intensity_gain import (
    RegionalIntensityGainAdapter, apply_regional_gain, compose_regional_gain,
    regional_envelope,
)


def sample(dtype=torch.float64):
    generator = torch.Generator().manual_seed(20260921)
    prior = torch.randn(2, 9, 9, generator=generator, dtype=dtype)
    valid = torch.tensor([[True, True, True, False, False, True, True, True, True],
                          [False, True, True, True, True, False, False, True, True]])
    return prior, valid


def test_envelope_constant_zero_and_gap_safe_with_poisoned_padding():
    prior, valid = sample()
    prior[:] = 0
    prior[0, :3, :5] = 2
    prior[0, 5:, :5] = 5
    prior[0, 5:, 5:] = 3
    prior[~valid] = float('nan')
    value = regional_envelope(prior, valid, window=3)
    # A constant run remains constant after boundary-renormalized smoothing.
    torch.testing.assert_close(value[0, :3, 0], torch.full((3,), 2., dtype=value.dtype))
    torch.testing.assert_close(value[0, 5:, 0], torch.full((4,), 5., dtype=value.dtype))
    torch.testing.assert_close(value[0, 5:, 1], torch.full((4,), 3., dtype=value.dtype))
    assert torch.equal(value[~valid], torch.zeros_like(value[~valid]))


def test_envelope_rejects_even_window_and_cross_run_input_contract():
    prior, valid = sample()
    with pytest.raises(ValueError, match='odd'):
        regional_envelope(prior, valid, window=4)
    prior[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        regional_envelope(prior, valid)


def test_apply_gain_preserves_channel_means_and_invalid_frames():
    prior, valid = sample()
    prior[~valid] = float('nan')
    gain = torch.full((2, 9, 2), 2., dtype=prior.dtype)
    result = apply_regional_gain(prior, gain, valid)
    # Scaling about the per-channel mean keeps the mean unchanged even when
    # brow and eye gains differ.
    before = prior[0, valid[0]].mean(0)
    after = result[0, valid[0]].mean(0)
    torch.testing.assert_close(before, after)
    assert torch.equal(result[~valid], torch.zeros_like(result[~valid]))


def test_apply_gain_ignores_nonfinite_padding_gain():
    prior, valid = sample()
    prior[~valid] = float('nan')
    gain = torch.full((2, 9, 2), 2., dtype=prior.dtype)
    gain[~valid] = float('nan')
    result = apply_regional_gain(prior, gain, valid)
    assert torch.isfinite(result[valid]).all()
    assert torch.equal(result[~valid], torch.zeros_like(result[~valid]))


def test_zero_initialized_adapter_is_identity_and_composition_is_protected():
    prior, valid = sample(torch.float32)
    baseline = torch.randn(2, 9, 52, generator=torch.Generator().manual_seed(8))
    baseline[~valid] = float('nan')
    prior[~valid] = float('nan')
    baseline[..., list(UPPER_INDICES)] = prior
    adapter = RegionalIntensityGainAdapter()
    target = regional_envelope(prior, valid)
    out = adapter(prior, target, valid)
    torch.testing.assert_close(out['gain'][valid], torch.ones_like(out['gain'][valid]), atol=0, rtol=0)
    torch.testing.assert_close(out['dynamic_upper'][valid], prior[valid], atol=0, rtol=0)
    composed = compose_regional_gain(baseline, prior, out['gain'], valid)
    assert torch.equal(composed[valid], baseline[valid])
    other = [i for i in range(52) if i not in UPPER_INDICES]
    assert torch.equal(composed[..., other].view(torch.int32), baseline[..., other].view(torch.int32))


def test_gain_is_bounded_and_a_toy_target_has_a_finite_training_gradient():
    prior, valid = sample(torch.float32)
    adapter = RegionalIntensityGainAdapter(hidden=16, max_gain=3.)
    target = regional_envelope(prior, valid) * torch.tensor([2., .5])
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.05)
    initial = None
    for _ in range(30):
        out = adapter(prior, target, valid)
        loss = (torch.where(valid[..., None], out['gain'] - torch.tensor([2., .5]), 0.) ** 2).mean()
        if initial is None:
            initial = float(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    result = adapter(prior, target, valid)
    assert float(loss) < initial
    assert torch.isfinite(result['gain'][valid]).all()
    assert float(result['gain'][valid].max()) <= 3. + 1e-5
    assert float(result['gain'][valid].min()) >= 1 / 3. - 1e-5


def test_zero_rms_envelope_has_finite_gradient_including_mask_gaps():
    prior, valid = sample(torch.float64)
    prior.zero_()
    prior[~valid] = float('nan')
    prior.requires_grad_()
    envelope = regional_envelope(prior, valid, window=3)
    envelope.sum().backward()
    assert torch.isfinite(prior.grad).all()
    assert torch.equal(prior.grad, torch.zeros_like(prior.grad))
    assert torch.equal(envelope, torch.zeros_like(envelope))


def test_gain_one_composes_a_different_prior_and_protects_all_other_channels():
    prior, valid = sample()
    baseline = torch.randn(2, 9, 52, generator=torch.Generator().manual_seed(8), dtype=prior.dtype)
    baseline[~valid] = float('nan')
    prior[~valid] = float('nan')
    gain = torch.ones(2, 9, 2, dtype=prior.dtype, requires_grad=True)
    result = compose_regional_gain(baseline, prior, gain, valid)
    indices = list(UPPER_INDICES)
    for batch_index in range(len(prior)):
        before = baseline[batch_index, valid[batch_index]][:, indices]
        source = prior[batch_index, valid[batch_index]]
        expected = before.mean(0) + source - source.mean(0)
        actual = result[batch_index, valid[batch_index]][:, indices]
        torch.testing.assert_close(actual, expected)
        assert not torch.equal(actual, before)
        torch.testing.assert_close(actual.mean(0), before.mean(0))
    other = [i for i in range(52) if i not in UPPER_INDICES]
    assert torch.equal(result[..., other].view(torch.int64), baseline[..., other].view(torch.int64))
    assert torch.equal(result[~valid].view(torch.int64), baseline[~valid].view(torch.int64))
    result[valid][:, indices].square().sum().backward()
    assert torch.isfinite(gain.grad).all()
    assert gain.grad[valid].abs().sum() > 0


@pytest.mark.parametrize('loss_type', ['motion', 'envelope', 'composed_motion'])
def test_zero_head_receives_finite_nonzero_gradient_from_actual_output(loss_type):
    prior, valid = sample(torch.float64)
    prior[~valid] = float('nan')
    adapter = RegionalIntensityGainAdapter(hidden=12).double()
    desired_gain = torch.tensor([1.7, .65], dtype=prior.dtype).expand(2, 9, 2)
    desired_upper = apply_regional_gain(prior, desired_gain, valid).detach()
    desired_envelope = regional_envelope(desired_upper, valid)
    output = adapter(prior, desired_envelope, valid)
    if loss_type == 'motion':
        loss = (output['dynamic_upper'][valid] - desired_upper[valid]).square().mean()
    elif loss_type == 'envelope':
        predicted_envelope = regional_envelope(output['dynamic_upper'], valid)
        loss = (predicted_envelope[valid] - desired_envelope[valid]).square().mean()
    else:
        baseline = torch.zeros(2, 9, 52, dtype=prior.dtype)
        baseline[..., list(UPPER_INDICES)] = prior
        composed = compose_regional_gain(baseline, prior, output['gain'], valid)
        loss = (composed[valid][:, list(UPPER_INDICES)] - desired_upper[valid]).square().mean()
    loss.backward()
    for parameter in (adapter.output.weight, adapter.output.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_gain_derivative_matches_finite_differences_at_identity():
    prior, valid = sample(torch.float64)
    prior[~valid] = float('nan')
    gain = torch.ones(2, 9, 2, dtype=prior.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: apply_regional_gain(prior, value, valid), (gain,))
