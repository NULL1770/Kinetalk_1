import torch

from scripts.diagnose_native_dynamic_scales import lagged_common_support, feature_scale


def test_all_lags_keep_identical_native_support_without_wrapping():
    x = torch.arange(40.).reshape(1, 40, 1)
    valid = torch.ones(1, 40, dtype=torch.bool)
    valid[:, 18] = False
    reference = None
    for lag in (-8, -4, 0, 4, 8):
        shifted, mask, positions = lagged_common_support(x, valid, lag)
        if reference is None:
            reference = mask
        assert torch.equal(mask, reference)
        assert torch.equal(shifted[mask], x[:, positions+lag][mask])
        assert not mask[:, (positions-18).abs() <= 8].any()


def test_feature_scale_does_not_use_internal_holdout():
    x = torch.tensor([[[1.], [-1.]], [[2.], [-2.]], [[1000.], [-1000.]]])
    weight = torch.ones(3, 2)
    fit = torch.tensor([0, 1])
    before = feature_scale(x, weight, fit)
    x[2] = 100000.
    assert torch.equal(before, feature_scale(x, weight, fit))
    torch.testing.assert_close(before, torch.tensor([2.5**.5]))
