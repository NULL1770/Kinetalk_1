import pytest
import torch

from kinetalk_b0.label_guided_intensity import (
    BROWS, EYES, fit_intensity_scales, regional_intensity, regional_window_activity,
)


def data():
    motion = torch.zeros(2, 4, 52)
    return motion, torch.ones_like(motion, dtype=torch.bool), torch.zeros(2, 52), torch.ones(52)


def test_equal_region_weight_preserves_constant_offset_and_no_neutral_zeroing():
    x, mask, anchor, scales = data()
    x[..., list(BROWS)] = 2
    x[..., list(EYES)] = 4
    intensity, valid = regional_intensity(x, mask, anchor, scales)
    torch.testing.assert_close(intensity, torch.full((2, 4, 1), 3.))
    assert valid.all()
    # Identity-independent coordinate translation changes neither deviation nor target.
    shifted, _ = regional_intensity(x + 10, mask, anchor + 10, scales)
    torch.testing.assert_close(shifted, intensity)
    changed, _ = regional_intensity(x + 1, mask, anchor, scales)
    torch.testing.assert_close(changed, intensity + 1)
    activity, activity_valid = regional_window_activity(x, mask, anchor, scales)
    assert torch.equal(activity, torch.zeros(2, 2)) and activity_valid.all()


def test_masked_poison_ignored_partial_groups_renormalize_and_missing_group_invalid():
    x, mask, anchor, scales = data()
    x[..., list(BROWS)] = 2; x[..., list(EYES)] = 4
    mask[0, 0, list(BROWS)[1:]] = False
    mask[0, 1, list(EYES)] = False
    mask[1, 3] = False
    clean, valid = regional_intensity(x, mask, anchor, scales)
    x[~mask] = float('nan')
    actual, poisoned_valid = regional_intensity(x, mask, anchor, scales)
    torch.testing.assert_close(actual, clean, rtol=0, atol=0)
    assert torch.equal(valid, poisoned_valid)
    assert actual[0, 0] == 3 and not valid[0, 1] and not valid[1, 3]
    assert actual[0, 1] == 0 and actual[1, 3] == 0
    assert torch.isfinite(regional_window_activity(x, mask, anchor, scales)[0]).all()


def test_observed_nonfinite_is_rejected_and_unobserved_anchor_poison_is_ignored():
    x, mask, anchor, scales = data()
    x[0, 0, 41] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite observed'):
        regional_intensity(x, mask, anchor, scales)
    mask[0, :, 41] = False; anchor[0, 41] = float('inf')
    assert torch.isfinite(regional_intensity(x, mask, anchor, scales)[0]).all()
    mask[0, 1, 41] = True
    with pytest.raises(ValueError, match='Nonfinite observed'):
        fit_intensity_scales(x, mask, anchor)


def test_fit_rms_anchor_deviation_floor_and_no_query_dependent_norm():
    x, mask, anchor, _ = data()
    x[:] = .4; anchor[:] = .1
    mask[..., 0] = False; x[..., 0] = float('nan')
    scales = fit_intensity_scales(x, mask, anchor)
    torch.testing.assert_close(scales[1:], torch.full((51,), .3))
    assert scales[0] == pytest.approx(.02)
    before = scales.clone()
    value, _ = regional_intensity(x, mask, anchor, scales)
    doubled, _ = regional_intensity(x * 2 - anchor[:, None], mask, anchor, scales)
    torch.testing.assert_close(value, torch.ones_like(value))
    torch.testing.assert_close(doubled, value * 2)
    assert torch.equal(scales, before)
    with pytest.raises(ValueError, match='floor'):
        fit_intensity_scales(x, mask, anchor, floor=.001)
    mask[..., list(EYES)] = False
    with pytest.raises(ValueError, match='entire intensity group'):
        fit_intensity_scales(x, mask, anchor)


def test_activity_ddof_zero_missing_channel_and_finite_constant_gradient():
    x, mask, anchor, scales = data()
    x[:] = torch.tensor([0., 2., 0., 2.])[None, :, None]
    mask[0, 2:] = False; mask[1, 1:, list(EYES)] = False
    activity, valid = regional_window_activity(x, mask, anchor, scales)
    assert activity[0].tolist() == [1., 1.]
    assert activity[1, 0] == 1 and not valid[1, 1] and activity[1, 1] == 0
    constant = torch.full_like(x, .3, requires_grad=True)
    activity, _ = regional_window_activity(constant, torch.ones_like(mask), anchor, scales)
    activity.sum().backward()
    assert torch.isfinite(constant.grad).all()


@pytest.mark.parametrize('case', ['motion', 'mask_shape', 'mask_type', 'anchor', 'scale_shape', 'scale_floor', 'scale_nan'])
def test_shape_and_scale_contract(case):
    x, mask, anchor, scales = data()
    if case == 'motion': x = x[..., :51]
    if case == 'mask_shape': mask = mask[..., 0]
    if case == 'mask_type': mask = mask.float()
    if case == 'anchor': anchor = anchor[:, None]
    if case == 'scale_shape': scales = scales[None]
    if case == 'scale_floor': scales[0] = 0
    if case == 'scale_nan': scales[0] = float('nan')
    with pytest.raises(ValueError): regional_intensity(x, mask, anchor, scales)
