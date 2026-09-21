import pytest
import torch

from kinetalk_b0.models.audio_regional_envelope import (
    AudioRegionalEnvelope,
    compose_prior_with_envelope,
    intervene_envelope,
)
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES


def sample(dtype=torch.float32):
    g = torch.Generator().manual_seed(20260921)
    prior = torch.randn(2, 11, 52, generator=g, dtype=dtype)
    valid = torch.tensor([
        [True, True, True, True, False, False, True, True, True, True, True],
        [False, True, True, True, True, True, True, False, True, True, True],
    ])
    prior[~valid] = float("nan")
    return prior, valid


def test_audio_envelope_is_native_rate_nonnegative_and_masks_padding():
    g = torch.Generator().manual_seed(1)
    features = torch.randn(2, 11, 8, generator=g)
    valid = torch.tensor([[True, True, True, False, True, True, True, True, True, True, True],
                          [True, True, False, False, True, True, True, True, True, True, True]])
    features[~valid] = float("nan")
    model = AudioRegionalEnvelope(torch.zeros(8), torch.ones(8), hidden=12, stride=1)
    out = model(features, valid)
    assert out.shape == (2, 11, 2)
    assert torch.isfinite(out[valid]).all()
    assert (out[valid] >= 0).all()
    assert torch.equal(out[~valid], torch.zeros_like(out[~valid]))


def test_stride16_is_rejected_and_interventions_keep_native_shape():
    with pytest.raises(ValueError, match="stride-16"):
        AudioRegionalEnvelope(torch.zeros(4), torch.ones(4), stride=16)
    envelope = torch.arange(2 * 7 * 2, dtype=torch.float32).reshape(2, 7, 2)
    valid = torch.tensor([[True, True, True, False, True, True, True],
                          [True, True, False, False, True, True, True]])
    for mode in ("audio", "zero", "static", "reverse"):
        value = intervene_envelope(envelope, valid, mode)
        assert value.shape == envelope.shape
        assert torch.equal(value[~valid], torch.zeros_like(value[~valid]))
    static = intervene_envelope(envelope, valid, "static")
    for row in range(len(valid)):
        expected = envelope[row, valid[row]].mean(0)
        torch.testing.assert_close(static[row, valid[row]], expected.expand(valid[row].sum(), 2))


def test_composition_preserves_mean_and_nonupper_channels_for_all_controls():
    prior, valid = sample()
    base_upper = prior[..., list(UPPER_INDICES)]
    # A prescribed mean is supplied independently of the source trajectory.
    target_mean = (torch.where(valid[..., None], base_upper, 0.).sum(1) / valid.sum(1, keepdim=True).to(base_upper.dtype))
    controls = {
        "zero": torch.zeros(2, 11, 2),
        "static": torch.ones(2, 11, 2),
        "reverse": torch.linspace(.5, 1.5, 11).reshape(1, 11, 1).expand(2, 11, 2).flip(1),
    }
    for name, gain in controls.items():
        out = compose_prior_with_envelope(prior, gain, target_mean, valid)
        assert out.shape == prior.shape, name
        other = [i for i in range(52) if i not in UPPER_INDICES]
        assert torch.equal(out[..., other].view(torch.int32), prior[..., other].view(torch.int32)), name
        actual = (torch.where(valid[..., None], out[..., list(UPPER_INDICES)], 0.).sum(1) / valid.sum(1, keepdim=True).to(out.dtype))
        torch.testing.assert_close(actual, target_mean, atol=2e-6, rtol=2e-6)
        assert torch.equal(out[~valid].view(torch.int32), prior[~valid].view(torch.int32)), name


def test_nine_channel_composition_accepts_framewise_gain():
    prior, valid = sample()
    upper = prior[..., list(UPPER_INDICES)]
    mean = (torch.where(valid[..., None], upper, 0.).sum(1) / valid.sum(1, keepdim=True).to(upper.dtype))
    gain = torch.ones(2, 11, 2)
    result = compose_prior_with_envelope(upper, gain, mean, valid)
    torch.testing.assert_close(result[valid], upper[valid], atol=2e-6, rtol=2e-6)
