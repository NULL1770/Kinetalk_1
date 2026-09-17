"""Observed-count decomposition handles unequal/missing frames and channels."""
import numpy as np
import torch

from scripts.audit_projection_mean_dynamic_tradeoff import exact_error_components, component_summary, paired_component_change


def test_exact_decomposition_with_nonconstant_mask_and_time_varying_B0():
    # Per-channel valid counts differ; missing entries may contain NaN.
    error = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]], [[-1., 1.], [2., 7.], [9., 9.]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True], [True, False], [False, True]], [[True, True], [True, False], [False, False]]])
    target = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2) / 10
    prediction = target + error
    prediction[~mask] = float('nan')
    base = torch.sin(torch.arange(12, dtype=torch.float64)).reshape(2, 3, 2)
    identity = torch.tensor([[.3, -.2], [1., 2.]], dtype=torch.float64)
    rows = exact_error_components(prediction, target, mask, [0, 1], base, identity)
    # Clip0 errors: c0=[1,3],c1=[2,6]: raw50,dynamic10,mean40,count4.
    # Clip1 errors: c0=[-1,2],c1=[1]: raw6,dynamic4.5,mean1.5,count3.
    np.testing.assert_allclose(rows, [[50., 10., 40., 4.], [6., 4.5, 1.5, 3.]], rtol=0, atol=1e-12)
    total = component_summary(rows)
    assert abs(total['raw_mse'] - (50 + 6) / 7) < 1e-12
    assert total['decomposition_abs_error'] < 1e-12
    baseline = rows.copy(); baseline[:, 0] += 2; baseline[:, 2] += 2
    change = paired_component_change(rows, baseline, ['a', 'b'], [0, 1], samples=100)
    assert abs(change['delta_candidate_minus_baseline']['raw_mse'] + 4 / 7) < 1e-12
    assert change['raw_delta_sum_error'] < 1e-12
