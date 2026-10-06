import numpy as np
import torch

from scripts.audit_b0_supervision import lag_profile


def test_lag_sign_detects_a_delayed_gt_without_changing_prediction():
    gen = torch.Generator().manual_seed(73)
    pred = torch.randn(80, generator=gen)
    target = torch.roll(pred, 2)
    original = pred.clone()
    valid = torch.ones(80, dtype=torch.bool)
    times = torch.arange(80, dtype=torch.float64) / 25
    rows, n = lag_profile(pred, target, valid, times)
    assert n == 69
    np.testing.assert_allclose(rows[:, 7], [1., 1.], atol=1e-12)
    assert np.argmax(rows[0]) == 7
    torch.testing.assert_close(pred, original)


def test_common_lag_window_excludes_invalid_values_and_clock_gaps():
    gen = torch.Generator().manual_seed(74)
    pred = torch.randn(80, generator=gen)
    target = pred.clone()
    valid = torch.ones(80, dtype=torch.bool)
    valid[30] = False
    pred[30] = float('nan')
    target[30] = float('nan')
    times = torch.arange(80, dtype=torch.float64) / 25
    times[50:] += 1
    rows, n = lag_profile(pred, target, valid, times)
    assert n == 46
    assert np.isfinite(rows).all()
    np.testing.assert_allclose(rows[:, 5], [1., 1.], atol=1e-12)
