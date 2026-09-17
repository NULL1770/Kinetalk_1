from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_conditioned_dynamic import EmotionConditionedDynamicPredictor


def _case():
    torch.manual_seed(41)
    features = torch.randn(3, 13, 8)
    valid = torch.ones(3, 13, dtype=torch.bool)
    valid[1, 9:] = False
    valid[2] = False
    probabilities = F.one_hot(torch.tensor([0, 1, 2]), 3).float()
    model = EmotionConditionedDynamicPredictor(8, 3, 12)
    return model, features, probabilities, valid


def test_initial_conditions_match_then_receive_gradient_and_change_timing():
    model, features, probabilities, valid = _case()
    uniform = torch.full_like(probabilities, 1 / 3)
    original = model(features, probabilities, valid)
    torch.testing.assert_close(original, model(features, uniform, valid), atol=0, rtol=0)
    count = valid.sum(1, keepdim=True).clamp_min(1)[..., None]
    centered = original - original.sum(1, keepdim=True) / count
    target = features[..., :1]
    ((centered - target).square()[valid]).mean().backward()
    assert model.condition.weight.grad.abs().sum() > 0
    assert model.input.weight.grad.abs().sum() > 0
    torch.optim.SGD(model.parameters(), lr=.1).step()
    first = model(features, probabilities, valid)
    changed = model(features, probabilities.roll(1, -1), valid)
    # Different conditions must change temporal variation, not only a bias
    # which would disappear during downstream clip centering.
    difference = first - changed
    assert difference[0, :, 0].std() > 1e-6


def test_padding_length_and_nan_values_do_not_change_valid_predictions():
    model, features, probabilities, valid = _case()
    with torch.no_grad():
        model.condition.weight.normal_(0, .1)
    original = model(features, probabilities, valid)
    corrupt = features.clone()
    corrupt[~valid] = float("nan")
    actual = model(corrupt, probabilities, valid)
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
    extended = F.pad(corrupt, (0, 0, 0, 7), value=float("nan"))
    extended_valid = F.pad(valid, (0, 7), value=False)
    longer = model(extended, probabilities, extended_valid)
    torch.testing.assert_close(longer[:, :13], original, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(longer).all()
    assert not longer[~extended_valid].any()
    # A short clip alone must also match its padded, mixed-length batch view.
    alone = model(features[1:2, :9], probabilities[1:2], valid[1:2, :9])
    torch.testing.assert_close(alone, original[1:2, :9], atol=1e-6, rtol=1e-5)


def test_forward_does_not_mutate_inputs_or_force_neutral_field_to_zero():
    model, features, probabilities, valid = _case()
    before = [x.clone() for x in (features, probabilities, valid)]
    output = model(features, probabilities, valid)
    assert output.shape == (3, 13, 1)
    assert output[0].std() > 0  # class zero has no special gate in this module
    for actual, expected in zip((features, probabilities, valid), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_linear_readout_preserves_centered_dynamics_under_large_bias():
    model, features, probabilities, valid = _case()
    model = model.double()
    features, probabilities = features.double(), probabilities.double()
    observed = valid[0:1]
    predictions, gradients = [], []
    for bias in (0., 100.):
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.output.bias.fill_(bias)
        raw = model(features[0:1], probabilities[0:1], observed)
        centered = raw - raw.mean(1, keepdim=True)
        centered.square().mean().backward()
        predictions.append(centered.detach())
        gradients.append(model.output.weight.grad.clone())
    torch.testing.assert_close(predictions[0], predictions[1], atol=1e-12, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-12, rtol=0)
    assert gradients[1].abs().sum() > 0


def test_invalid_dimensions_and_mask_fail_clearly():
    with pytest.raises(ValueError, match="dimensions"):
        EmotionConditionedDynamicPredictor(hidden_dim=0)
    model, features, probabilities, valid = _case()
    with pytest.raises(ValueError, match="probabilities"):
        model(features, probabilities[:, :2], valid)
    with pytest.raises(ValueError, match="floating point"):
        model(features, probabilities.long(), valid)
    with pytest.raises(ValueError, match="valid"):
        model(features, probabilities, valid.float())
