import pytest
import torch

from kinetalk_b0.models.mean_preserving_upper import UPPER_INDICES, compose_mean_preserving_upper


OTHER = [index for index in range(52) if index not in UPPER_INDICES]
UPPER = list(UPPER_INDICES)


def inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(20260917)
    baseline = torch.randn(2, 7, 52, generator=generator, dtype=dtype)
    dynamic = torch.randn(2, 7, 9, generator=generator, dtype=dtype)
    valid = torch.tensor([[True, False, True, True, False, False, False],
                          [False, True, True, False, True, True, True]])
    return baseline, dynamic, valid


def test_replaces_original_dynamics_and_preserves_baseline_mean_without_clamp():
    baseline, dynamic, valid = inputs()
    dynamic = dynamic * 10 + 30
    actual = compose_mean_preserving_upper(baseline, dynamic, valid)
    for row in range(2):
        base = baseline[row, valid[row]][:, UPPER]
        movement = dynamic[row, valid[row]]
        result = actual[row, valid[row]][:, UPPER]
        torch.testing.assert_close(result.mean(0), base.mean(0), atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(result - result.mean(0), movement - movement.mean(0))
        torch.testing.assert_close(result, base.mean(0) + movement - movement.mean(0))
    assert actual[valid][:, UPPER].abs().max() > 1
    constant = compose_mean_preserving_upper(baseline, torch.full_like(dynamic, 100.), valid)
    for row in range(2):
        expected = baseline[row, valid[row]][:, UPPER].mean(0)
        torch.testing.assert_close(constant[row, valid[row]][:, UPPER], expected.expand(int(valid[row].sum()), -1))


def test_43_channels_and_invalid_frames_are_bitwise_copied_including_nan_padding():
    baseline, dynamic, valid = inputs(torch.float32)
    baseline[~valid] = float('nan')
    baseline[0, 1, 0] = -0.
    baseline[1, 0, 1] = float('inf')
    dynamic[~valid] = float('nan')
    before = baseline.clone()
    output = compose_mean_preserving_upper(baseline, dynamic, valid)
    assert torch.equal(output[..., OTHER].contiguous().view(torch.int32), baseline[..., OTHER].contiguous().view(torch.int32))
    assert torch.equal(output[~valid].view(torch.int32), baseline[~valid].view(torch.int32))
    assert torch.equal(baseline.view(torch.int32), before.view(torch.int32))
    assert output.data_ptr() != baseline.data_ptr()
    assert torch.isfinite(output[valid]).all()


def test_analytic_gradients_preserve_mean_and_remove_dynamic_constant_direction():
    baseline, dynamic, valid = inputs()
    baseline[~valid] = float('nan')
    dynamic[~valid] = float('nan')
    baseline.requires_grad_(); dynamic.requires_grad_()
    weights = torch.arange(baseline.numel(), dtype=baseline.dtype).reshape_as(baseline) / 100
    output = compose_mean_preserving_upper(baseline, dynamic, valid)
    (output[valid] * weights[valid]).sum().backward()
    expected_base = torch.zeros_like(baseline)
    expected_dynamic = torch.zeros_like(dynamic)
    for row in range(2):
        observed_weights = weights[row, valid[row]].clone()
        upper_weights = observed_weights[:, UPPER]
        expected_dynamic[row, valid[row]] = upper_weights - upper_weights.mean(0)
        observed_weights[:, UPPER] = upper_weights.mean(0)
        expected_base[row, valid[row]] = observed_weights
    torch.testing.assert_close(baseline.grad, expected_base)
    torch.testing.assert_close(dynamic.grad, expected_dynamic)
    assert torch.isfinite(baseline.grad).all() and torch.isfinite(dynamic.grad).all()
    assert baseline.grad[~valid].count_nonzero() == 0
    assert dynamic.grad[~valid].count_nonzero() == 0
    assert dynamic.grad[valid].abs().sum() > 0


def test_appended_nan_padding_and_separate_batch_lengths_leave_outputs_and_gradients_unchanged():
    baseline, dynamic, valid = inputs()
    baseline.requires_grad_(); dynamic.requires_grad_()
    expected = compose_mean_preserving_upper(baseline, dynamic, valid)
    expected[valid].square().sum().backward()
    padded_base = torch.cat([baseline.detach(), torch.full((2, 11, 52), float('nan'), dtype=baseline.dtype)], 1).requires_grad_()
    padded_dynamic = torch.cat([dynamic.detach(), torch.full((2, 11, 9), float('nan'), dtype=dynamic.dtype)], 1).requires_grad_()
    padded_valid = torch.cat([valid, torch.zeros(2, 11, dtype=torch.bool)], 1)
    actual = compose_mean_preserving_upper(padded_base, padded_dynamic, padded_valid)
    actual[padded_valid].square().sum().backward()
    torch.testing.assert_close(actual[:, :7][valid], expected[valid], atol=0, rtol=0)
    torch.testing.assert_close(padded_base.grad[:, :7], baseline.grad, atol=0, rtol=0)
    torch.testing.assert_close(padded_dynamic.grad[:, :7], dynamic.grad, atol=0, rtol=0)
    assert torch.isfinite(padded_base.grad).all() and torch.isfinite(padded_dynamic.grad).all()
    assert padded_base.grad[:, 7:].count_nonzero() == padded_dynamic.grad[:, 7:].count_nonzero() == 0
    for row in range(2):
        compact_base = baseline.detach()[row, valid[row]][None]
        compact_dynamic = dynamic.detach()[row, valid[row]][None]
        compact = compose_mean_preserving_upper(compact_base, compact_dynamic, torch.ones(compact_base.shape[:2], dtype=torch.bool))
        torch.testing.assert_close(compact[0], expected[row, valid[row]], atol=0, rtol=0)


def test_dynamic_offset_and_baseline_zero_mean_motion_have_no_effect():
    baseline, dynamic, valid = inputs()
    expected = compose_mean_preserving_upper(baseline, dynamic, valid)
    shifted_dynamic = dynamic + torch.arange(9, dtype=dynamic.dtype)[None, None]
    changed_baseline = baseline.clone()
    for row in range(2):
        frames = changed_baseline[row, valid[row]].clone()
        perturbation = torch.arange(len(frames), dtype=baseline.dtype)[:, None].expand(-1, 9)
        frames[:, UPPER] += perturbation - perturbation.mean(0)
        changed_baseline[row, valid[row]] = frames
    actual = compose_mean_preserving_upper(changed_baseline, shifted_dynamic, valid)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=1e-13)


def test_single_valid_frame_has_baseline_upper_and_zero_dynamic_gradient():
    baseline = torch.randn(1, 3, 52, dtype=torch.float64, requires_grad=True)
    dynamic = torch.randn(1, 3, 9, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[False, True, False]])
    output = compose_mean_preserving_upper(baseline, dynamic, valid)
    torch.testing.assert_close(output, baseline, atol=0, rtol=0)
    output[valid].sum().backward()
    assert dynamic.grad.count_nonzero() == 0


@pytest.mark.parametrize('source', ['baseline', 'dynamic'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_rejects_nonfinite_observed_values(source, bad):
    baseline, dynamic, valid = inputs()
    (baseline if source == 'baseline' else dynamic)[0, 0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        compose_mean_preserving_upper(baseline, dynamic, valid)


@pytest.mark.parametrize('case', ['empty_row', 'mask_dtype', 'mask_shape', 'dynamic_shape', 'dynamic_dtype', 'base_shape', 'base_dtype'])
def test_rejects_invalid_input_contract(case):
    baseline, dynamic, valid = inputs()
    if case == 'empty_row': valid[0] = False
    elif case == 'mask_dtype': valid = valid.float()
    elif case == 'mask_shape': valid = valid[:, :-1]
    elif case == 'dynamic_shape': dynamic = dynamic[..., :-1]
    elif case == 'dynamic_dtype': dynamic = dynamic.float()
    elif case == 'base_shape': baseline = baseline[..., :-1]
    elif case == 'base_dtype': baseline = baseline.long()
    with pytest.raises(ValueError):
        compose_mean_preserving_upper(baseline, dynamic, valid)
