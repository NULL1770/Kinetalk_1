import pytest
import torch

from kinetalk_b0.models.slow_state_affect import UPPER_INDICES
from kinetalk_b0.models.temporal_motion_carrier import smooth_motion_carrier


UPPER = list(UPPER_INDICES)
OTHER = [i for i in range(52) if i not in UPPER_INDICES]


def sample(dtype=torch.float64):
    base = torch.randn(3, 13, 52, generator=torch.Generator().manual_seed(9022), dtype=dtype)
    valid = torch.tensor([
        [True] * 10 + [False] * 3,
        [False, True, True, True, False, True, True, True, True, True, False, False, True],
        [True] * 10 + [False] * 3,
    ])
    base[~valid] = float("nan")
    return base, valid


def bits(value):
    return value.contiguous().view(torch.int64 if value.dtype == torch.float64 else torch.int32)


def test_identity_keeps_raw_and_all_bits_and_has_identity_gradient():
    base, valid = sample()
    base.requires_grad_()
    out = smooth_motion_carrier(base, valid, window=1)
    assert out["raw"] is base
    assert out["smoothed"] is not base
    assert torch.equal(bits(out["smoothed"]), bits(base))
    out["smoothed"][valid].sum().backward()
    assert torch.equal(base.grad[valid], torch.ones_like(base.grad[valid]))
    assert torch.equal(base.grad[~valid], torch.zeros_like(base.grad[~valid]))


@pytest.mark.parametrize("window", [3, 5, 7, 9])
def test_mean_nonupper_and_invalid_protection(window):
    base, valid = sample()
    original_bits = bits(base).clone()
    out = smooth_motion_carrier(base, valid, window)["smoothed"]
    assert torch.equal(bits(base), original_bits)
    assert torch.equal(bits(out[..., OTHER]), bits(base[..., OTHER]))
    assert torch.equal(bits(out[~valid]), bits(base[~valid]))
    for row in range(len(base)):
        torch.testing.assert_close(out[row, valid[row]][:, UPPER].mean(0),
                                   base[row, valid[row]][:, UPPER].mean(0), atol=1e-12, rtol=1e-12)
    assert (out[valid][:, UPPER] - base[valid][:, UPPER]).abs().sum() > 0


def test_default_triangle_matches_independent_boundary_normalized_reference():
    base = torch.zeros(1, 12, 52, dtype=torch.float64)
    wave = torch.tensor([0., .2, 1.1, -.4, 0., 1.5, -.2, .3, .1, 0., .4, -.3], dtype=base.dtype)
    base[0, :, UPPER] = wave[:, None]
    valid = torch.ones(1, 12, dtype=torch.bool)
    weights = [1., 2., 3., 2., 1.]
    filtered = []
    for frame in range(len(wave)):
        numerator = denominator = 0.
        for offset, weight in zip(range(-2, 3), weights):
            if 0 <= frame + offset < len(wave):
                numerator = numerator + wave[frame + offset] * weight
                denominator += weight
        filtered.append(numerator / denominator)
    expected = torch.stack(filtered)
    expected += wave.mean() - expected.mean()
    out = smooth_motion_carrier(base, valid)["smoothed"]
    torch.testing.assert_close(out[0, :, UPPER], expected[:, None].expand(-1, 9), atol=1e-12, rtol=1e-12)


def test_gaps_are_independent_and_padding_cannot_change_observed_output():
    base, valid = sample()
    out = smooth_motion_carrier(base, valid)["smoothed"]
    changed = base.clone()
    changed[1, 5:10, UPPER] *= 100.
    other_out = smooth_motion_carrier(changed, valid)["smoothed"]
    torch.testing.assert_close(other_out[1, 1:4], out[1, 1:4], atol=0., rtol=0.)
    torch.testing.assert_close(other_out[1, 12:13], out[1, 12:13], atol=0., rtol=0.)
    solo = smooth_motion_carrier(base[1:2, 1:4], torch.ones(1, 3, dtype=torch.bool))["smoothed"]
    torch.testing.assert_close(solo[0], out[1, 1:4], atol=1e-12, rtol=1e-12)
    padded = torch.cat([base, torch.full((3, 7, 52), float("inf"), dtype=base.dtype)], dim=1)
    padded_valid = torch.cat([valid, torch.zeros(3, 7, dtype=torch.bool)], dim=1)
    padded_out = smooth_motion_carrier(padded, padded_valid)["smoothed"]
    torch.testing.assert_close(padded_out[:, :13][valid], out[valid], atol=0., rtol=0.)


def test_static_carrier_remains_static_without_range_clamping():
    base = torch.full((1, 8, 52), 2., dtype=torch.float64)
    valid = torch.ones(1, 8, dtype=torch.bool)
    out = smooth_motion_carrier(base, valid)["smoothed"]
    torch.testing.assert_close(out, base, atol=1e-12, rtol=1e-12)
    assert (out[..., UPPER] > 1.).all()


def test_smoothing_does_not_restore_lost_rms():
    base = torch.zeros(1, 40, 52, dtype=torch.float64)
    alternating = torch.tensor([-1., 1.] * 20, dtype=base.dtype)
    base[0, :, UPPER] = alternating[:, None]
    valid = torch.ones(1, 40, dtype=torch.bool)
    out = smooth_motion_carrier(base, valid)["smoothed"]
    assert out[..., UPPER].std() < .3 * base[..., UPPER].std()


def test_gradients_are_finite_with_nan_padding_and_match_finite_differences():
    base, valid = sample()
    base.requires_grad_()
    result = smooth_motion_carrier(base, valid)["smoothed"]
    result[valid].square().mean().backward()
    assert torch.isfinite(base.grad).all()
    assert base.grad[valid][:, UPPER].abs().sum() > 0
    assert torch.equal(base.grad[~valid], torch.zeros_like(base.grad[~valid]))
    short = torch.randn(1, 3, 52, dtype=torch.float64, requires_grad=True)
    short_valid = torch.ones(1, 3, dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x: smooth_motion_carrier(x, short_valid)["smoothed"],
                                   (short,), fast_mode=True)


@pytest.mark.parametrize("window", [0, 2, 4, 11, True, 5.0])
def test_invalid_windows_are_rejected(window):
    base, valid = sample()
    with pytest.raises(ValueError, match="window"):
        smooth_motion_carrier(base, valid, window)


def test_invalid_observation_contract_is_rejected():
    base, valid = sample()
    with pytest.raises(ValueError, match="observed frame"):
        smooth_motion_carrier(base, torch.zeros_like(valid))
    base[0, 0, 43] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        smooth_motion_carrier(base, valid)
