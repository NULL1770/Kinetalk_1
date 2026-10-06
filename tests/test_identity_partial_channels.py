"""Identity pooling must ignore channels absent from a reference view."""

import torch

from kinetalk_b0.models.neutral_affect import NeutralIdentityEncoder


def test_partial_channel_pooling_does_not_count_zero_or_nan_padding():
    encoder = NeutralIdentityEncoder(motion_dim=4, style_dim=5, hidden_dim=7).eval()
    residual = torch.tensor(
        [[
            [[2.0, 4.0, float("nan"), 10.0],
             [2.0, 4.0, float("nan"), 10.0]],
            [[float("nan"), 6.0, float("nan"), 12.0],
             [float("nan"), 6.0, float("nan"), 12.0]],
        ]]
    )
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    channels = torch.tensor(
        [[[True, True, False, True],
          [False, True, False, True]]]
    )

    output = encoder(residual, valid, channels)

    torch.testing.assert_close(
        output["neutral_mean"], torch.tensor([[2.0, 5.0, 0.0, 11.0]])
    )
    assert output["observed_channels"].tolist() == [[True, True, False, True]]
    assert torch.isfinite(output["code"]).all()
    assert torch.isfinite(output["per_reference"]).all()


def test_missing_channel_value_cannot_change_identity_code():
    encoder = NeutralIdentityEncoder(motion_dim=3, style_dim=4, hidden_dim=6).eval()
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    channels = torch.tensor(
        [[[True, True, False],
          [True, True, False]]]
    )
    first = torch.tensor(
        [[[[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
          [[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]]]]
    )
    second = first.clone()
    second[..., 2] = 1000.0

    a = encoder(first, valid, channels)
    b = encoder(second, valid, channels)

    torch.testing.assert_close(a["code"], b["code"], rtol=0, atol=0)
    torch.testing.assert_close(a["neutral_mean"], b["neutral_mean"], rtol=0, atol=0)
    torch.testing.assert_close(a["per_reference"], b["per_reference"], rtol=0, atol=0)


def test_single_reference_channel_mask_is_supported():
    encoder = NeutralIdentityEncoder(motion_dim=3, style_dim=4, hidden_dim=6).eval()
    residual = torch.tensor(
        [[[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]]
    )
    valid = torch.ones(1, 2, dtype=torch.bool)
    channels = torch.tensor([[True, False, True]])
    output = encoder(residual, valid, channels)

    torch.testing.assert_close(output["neutral_mean"], torch.tensor([[1.0, 0.0, 0.0]]))
    assert output["observed_channels"].tolist() == [[True, False, True]]
