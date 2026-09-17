import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinetalk_b0.emotion_ray import (
    NUISANCE_CHANNELS_52,
    direction_from_probs,
    fit_emotion_rays,
    project_ray_field,
)


def _toy(channels=3):
    # Speaker 0 has twice as many happy clips but must not get twice the weight.
    speaker = torch.tensor([0, 0, 0, 1, 1, 9])
    emotion = torch.tensor([0, 1, 1, 0, 1, 2])
    x = torch.zeros(6, 5, channels)
    x[:3, :, -1] = 3
    x[3:5, :, -1] = -2
    x[1:3, :, 0] += 2
    x[4, :, 1] += 2
    valid = torch.ones(6, 5, dtype=torch.bool)
    observed = torch.ones(6, channels, dtype=torch.bool)
    return x, valid, observed, speaker, emotion, torch.arange(5)


def test_fit_is_neutral_anchored_equal_speaker_weight_and_unit_direction():
    state = fit_emotion_rays(*_toy(), num_emotions=3)
    torch.testing.assert_close(state["rays"][1], torch.tensor([2**-0.5, 2**-0.5, 0]))
    assert state["active"].tolist() == [False, True, False]
    assert state["counts"]["clips"].tolist() == [2, 3, 0]
    assert state["counts"]["speakers"].tolist() == [2, 2, 0]
    assert state["anchor_speaker_ids"].tolist() == [0, 1]
    torch.testing.assert_close(state["neutral_anchors"][:, -1], torch.tensor([3., -2.]))


def test_heldout_motion_masks_and_labels_cannot_poison_fit():
    args = list(_toy())
    expected = fit_emotion_rays(*args, num_emotions=3)
    args[0][-1] = float("nan")
    args[1][-1] = False
    args[2][-1] = False
    args[3][-1] = -999
    args[4][-1] = 999
    actual = fit_emotion_rays(*args, num_emotions=3)
    for name in ("rays", "active", "neutral_anchors", "observed_channels"):
        torch.testing.assert_close(actual[name], expected[name])


def test_shared_channel_mask_and_nuisance_exclusion():
    x, valid, observed, speaker, emotion, train = _toy(52)
    x[1:3, :, 14] += 2
    x[4, :, 14] += 2
    x[1:3, :, 15] = 100
    observed[0, 15] = False
    x[0, :, 15] = float("nan")
    state = fit_emotion_rays(x, valid, observed, speaker, emotion, train, 3)
    assert not state["observed_channels"][list(NUISANCE_CHANNELS_52)].any()
    assert not state["observed_channels"][15]
    assert state["rays"][1, 14] == 1
    assert state["rays"][1].count_nonzero() == 1
    kept = fit_emotion_rays(x, valid, observed, speaker, emotion, train, 3, exclude_nuisance=False)
    assert kept["rays"][1, 0] > 0


def test_signed_projection_centers_valid_frames_and_weights_partial_bins():
    x = torch.tensor([[[1., 7.], [2., 7.], [3., 7.], [4., 7.], [5., 7.], [float("nan"), 0]]])
    valid = torch.tensor([[True, True, True, True, True, False]])
    ray = torch.tensor([[1., 0.]])
    scalar, weight, bins = project_ray_field(x, valid, ray, stride=2)
    torch.testing.assert_close(weight, torch.tensor([[2., 2., 1.]]))
    torch.testing.assert_close(scalar[..., 0], torch.tensor([[-1.5, .5, 2.]]))
    torch.testing.assert_close((scalar[..., 0] * weight).sum(1), torch.zeros(1))
    assert torch.equal(bins[..., 1], torch.zeros(1, 3))
    neg, _, _ = project_ray_field(x, valid, -ray, stride=2)
    torch.testing.assert_close(neg, -scalar)


def test_neutral_direction_never_forces_expression_and_missing_mass_is_explicit():
    state = fit_emotion_rays(*_toy(), num_emotions=3)
    neutral = direction_from_probs(torch.tensor([[1., 0, 0]]), state)
    assert not neutral.any()
    scalar, _, _ = project_ray_field(torch.randn(1, 5, 3), torch.ones(1, 5, dtype=torch.bool), neutral)
    assert not scalar.any()
    with pytest.raises(ValueError, match="no fitted emotion ray"):
        direction_from_probs(torch.tensor([[0., .8, .2]]), state)
    direction, info = direction_from_probs(torch.tensor([[0., .8, .2], [1., 0, 0]]), state, return_info=True)
    torch.testing.assert_close(direction[0], state["rays"][1])
    torch.testing.assert_close(info["missing_mass"], torch.tensor([.2, 0]))
    torch.testing.assert_close(info["neutral_mass"], torch.tensor([0., 1]))
    assert not direction[1].any()


def test_missing_training_neutral_cannot_borrow_heldout_neutral():
    args = list(_toy())
    args[-1] = torch.tensor([0, 1, 2, 4])  # Speaker 1's neutral is held out.
    with pytest.raises(ValueError, match="lack training neutral"):
        fit_emotion_rays(*args, num_emotions=3)
    state = fit_emotion_rays(*args, num_emotions=3, missing_neutral="skip")
    assert state["skipped_speaker_ids"] == [1]
    torch.testing.assert_close(state["rays"][1], torch.tensor([1., 0, 0]))


def test_padding_and_empty_projection_are_zero_and_never_nan():
    x = torch.full((2, 5, 3), float("nan"))
    x[0, :2] = torch.arange(6.).reshape(2, 3)
    valid = torch.tensor([[True, True, False, False, False], [False] * 5])
    scalar, weight, bins = project_ray_field(x, valid, torch.tensor([[1., 0, 0], [0., 1, 0]]), stride=2)
    assert torch.isfinite(scalar).all() and torch.isfinite(bins).all()
    assert torch.equal(scalar[weight == 0], torch.zeros_like(scalar[weight == 0]))
    assert torch.equal(bins[weight == 0], torch.zeros_like(bins[weight == 0]))


def test_probabilities_are_validated_and_opposite_rays_can_cancel():
    state = fit_emotion_rays(*_toy(), num_emotions=3)
    state["rays"][2] = -state["rays"][1]
    state["active"][2] = True
    direction = direction_from_probs(torch.tensor([[0., .5, .5]]), state)
    assert not direction.any()
    with pytest.raises(ValueError, match="sum to one"):
        direction_from_probs(torch.tensor([[0., .5, .2]]), state)
