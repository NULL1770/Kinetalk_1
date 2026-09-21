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
    assert torch.equal(result[valid], upper[valid])


@pytest.mark.parametrize("width", [9, 52])
def test_unit_gain_preserves_exact_prior_and_has_nonzero_gradient(width):
    prior, valid = sample(torch.float64)
    upper = prior[..., list(UPPER_INDICES)]
    mean = torch.where(valid[..., None], upper, 0.).sum(1) / valid.sum(1, keepdim=True)
    source = prior if width == 52 else upper
    gain = torch.ones(2, 11, 2, dtype=torch.float64, requires_grad=True)
    result = compose_prior_with_envelope(source, gain, mean, valid)
    assert torch.equal(result.view(torch.int64), source.view(torch.int64))
    result[valid].square().sum().backward()
    assert torch.isfinite(gain.grad).all()
    assert gain.grad[valid].abs().sum() > 0
    assert torch.equal(gain.grad[~valid], torch.zeros_like(gain.grad[~valid]))


@pytest.mark.parametrize("mean_width", [9, 52])
def test_composition_rejects_framewise_mean(mean_width):
    prior, valid = sample()
    with pytest.raises(ValueError, match="static"):
        compose_prior_with_envelope(prior, torch.ones(2, 11, 2),
                                   torch.zeros(2, 11, mean_width), valid)


def test_composition_restores_independent_static_mean():
    prior, valid = sample(torch.float64)
    mean = torch.linspace(-.1, .3, 9, dtype=torch.float64)
    gain = torch.linspace(.25, 2., 22, dtype=torch.float64).reshape(1, 11, 2).expand(2, 11, 2)
    result = compose_prior_with_envelope(prior, gain, mean, valid)
    actual = torch.where(valid[..., None], result[..., list(UPPER_INDICES)], 0.).sum(1) / valid.sum(1, keepdim=True)
    torch.testing.assert_close(actual, mean.expand_as(actual), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("stride", [1, 4])
def test_poisoned_padding_has_finite_model_and_feature_gradients(stride):
    torch.manual_seed(13)
    model = AudioRegionalEnvelope(torch.zeros(8), torch.ones(8), hidden=12, stride=stride)
    # Exercise upstream gradients as after training has moved the zero head.
    with torch.no_grad():
        model.head.weight.normal_(std=.2)
    features = torch.randn(2, 11, 8)
    _, valid = sample()
    features[~valid] = float("nan")
    features.requires_grad_()
    output = model(features, valid)
    output[valid].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.input.weight.grad.abs().sum() > 0
    assert torch.isfinite(features.grad).all()
    assert features.grad[valid].abs().sum() > 0
    assert torch.equal(features.grad[~valid], torch.zeros_like(features.grad[~valid]))


@pytest.mark.parametrize("stride", [1, 4])
def test_grouped_temporal_processing_matches_independent_runs_and_padding(stride):
    torch.manual_seed(14)
    model = AudioRegionalEnvelope(torch.zeros(5), torch.ones(5), hidden=12, stride=stride).double().eval()
    with torch.no_grad():
        model.head.weight.normal_(std=.2)
    features = torch.randn(3, 12, 5, dtype=torch.float64)
    valid = torch.tensor([
        [True] * 8 + [False] * 4,
        [True] * 8 + [False] * 4,
        [True, True, False, True, True, True, True, True, True, True, True, False],
    ])
    features[~valid] = float("nan")
    result = model(features, valid)
    for row, left, right in [(0, 0, 8), (1, 0, 8), (2, 0, 2), (2, 3, 11)]:
        single = model(features[row:row + 1, left:right], torch.ones(1, right - left, dtype=torch.bool))
        torch.testing.assert_close(result[row, left:right], single[0], atol=1e-12, rtol=1e-12)
    padded_features = torch.cat((features, torch.full((3, 9, 5), float("nan"), dtype=features.dtype)), dim=1)
    padded_valid = torch.cat((valid, torch.zeros(3, 9, dtype=torch.bool)), dim=1)
    padded_result = model(padded_features, padded_valid)
    torch.testing.assert_close(result[valid], padded_result[:, :12][valid], atol=1e-12, rtol=1e-12)
