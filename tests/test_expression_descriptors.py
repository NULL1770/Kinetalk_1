import numpy as np
import pytest
import torch
from scripts.diagnose_expression_descriptors import (
    descriptors, feature_windows, reverse_control, summarize)
from scripts.diagnose_paired_predictability import ChannelRidge, score


def test_descriptors_match_direct_native_windows_and_reject_gaps():
    rng = np.random.default_rng(3)
    query, neutral = rng.normal(size=(24, 3)), rng.normal(size=(24, 3))
    mask = np.ones((24, 3), bool)
    mask[5, 0] = False
    mask[:, 2] = False
    query[~mask] = np.nan
    times = np.arange(24)*.04
    times[15:] += .04
    y, valid = descriptors(query, neutral, mask, times)
    expected = np.zeros_like(valid)
    for t in range(2, 22):
        for c in range(3):
            ok = mask[t-2:t+3, c].all() and np.allclose(np.diff(times[t-2:t+3]), .04)
            expected[t, c] = ok
            if ok:
                z = query[t-2:t+3, c]-neutral[t-2:t+3, c]
                np.testing.assert_allclose(y[t, c], [z.mean(), z.std()], atol=1e-12)
    expected &= (expected.sum(0) >= 2)[None]
    np.testing.assert_array_equal(valid, expected)
    assert not valid[:, 2].any() and np.isfinite(y).all()


def test_constant_offset_changes_mean_but_not_descriptor_dynamics():
    t = np.arange(30)*.04
    q = np.stack((np.sin(t*7), t*t), -1)
    m = np.ones_like(q, dtype=bool)
    a, am = descriptors(q, q*.2, m, t)
    b, bm = descriptors(q+np.array([2., -3.]), q*.2, m, t)
    np.testing.assert_array_equal(am, bm)
    for c in range(2):
        for k in range(2):
            av, bv = a[am[:, c], c, k], b[bm[:, c], c, k]
            np.testing.assert_allclose(av-av.mean(), bv-bv.mean(), atol=2e-12)


def test_feature_windows_keep_empty_native_slots_and_do_not_bridge():
    t = np.arange(13)*.04
    x = np.stack((t, 2*t), -1)
    valid = np.ones(13, bool)
    valid[6] = False
    x[6] = np.nan
    mean, support = feature_windows(x, valid, t)
    assert np.flatnonzero(support).tolist() == [2, 3, 9, 10]
    np.testing.assert_allclose(mean[support], x[support], atol=1e-12)
    assert np.isfinite(mean).all()


def test_single_center_is_not_a_temporal_target():
    x = np.arange(5., dtype=float)[:, None]
    y, valid = descriptors(x, np.zeros_like(x), np.ones_like(x, bool), np.arange(5)*.04)
    assert not valid.any() and y.shape == (5, 1, 2)


def test_reverse_uses_actual_length_and_same_support_without_gap_compression():
    x = torch.arange(18., dtype=torch.float64).reshape(2, 9, 1)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[0, 1] = False
    valid[1, 6:] = False
    mask = valid[..., None].expand(-1, -1, 2).clone()
    mask[0, 3, 1] = False
    reverse, common = reverse_control(x, valid, mask, [9, 6])
    torch.testing.assert_close(reverse[1, :6], x[1, :6].flip(0))
    assert not reverse[1, 6:].any()
    assert not common[0, 1].any() and not common[0, 7].any()
    assert not common[0, 3, 1] and not common[1, 6:].any()
    assert reverse[0, 2, 0] == x[0, 6, 0]


def test_supported_dynamic_target_can_generalize_and_reverse_fails():
    def sample(offset):
        t = np.arange(40)*.04
        q = ((t+offset)**2)[:, None]
        y, mask = descriptors(q, q*0, np.ones_like(q, bool), t)
        # Synthetic known predictors to independently test fit/score arithmetic.
        x = y[:, 0]
        return [torch.from_numpy(v[None]) for v in (x, y, mask, t)]
    x, y, m, t = sample(0.)
    fit = ChannelRidge(2, channels=1, targets=2)
    fit.add(x, y, m, t, 'centered')
    w, energy = fit.solve()
    held, hy, hm, ht = sample(.3)
    rev, common = reverse_control(held, hm.any(-1), hm, [40])
    forward, zero, _ = score(held, hy, common, ht, 'centered', w)
    backward, _, _ = score(rev, hy, common, ht, 'centered', w)
    assert (forward/zero).max() < .005
    assert (backward/zero).min() > 2
    saved = w.clone()
    score(held*99, hy, hm, ht, 'centered', w)
    torch.testing.assert_close(saved, w, rtol=0, atol=0)
    assert (energy > 0).all()


def test_summary_keeps_absolute_errors_and_null_empty_channels():
    error = np.ones((2, 52, 2))
    zero = error*2
    mask = np.zeros((2, 52), bool)
    mask[:, 14:41] = True
    report = summarize(error, zero, mask, np.ones((52, 2)))
    assert report['mouth']['local_std']['energy_reduction'] == .5
    assert report['eyes']['local_mean']['mse'] is None
    assert report['eyes']['local_mean']['clips'] == 0


@pytest.mark.parametrize('case', ['times', 'mask', 'observed_nan'])
def test_invalid_descriptor_inputs_fail(case):
    q = np.ones((10, 2))
    m = np.ones_like(q, bool)
    t = np.arange(10)*.04
    if case == 'times':
        t[5] = t[4]
    elif case == 'mask':
        m = m.astype(float)
    else:
        q[4, 0] = np.nan
    with pytest.raises(ValueError):
        descriptors(q, q*0, m, t)
