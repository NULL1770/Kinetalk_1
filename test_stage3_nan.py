import torch

from kinetalk_b0.models.encoders import AudioEmotionDistributionEncoder


def test_stage3_audio_encoder_has_finite_outputs_and_gradients():
    encoder = AudioEmotionDistributionEncoder(69, 8, 16, 2, 3, 4, 0.0)
    x = torch.randn(2, 12, 69, requires_grad=True)
    y = encoder(x, torch.ones(2, 12, dtype=torch.bool))
    loss = y["global"].square().mean() + y["intensity_value"].square().mean()
    loss.backward()
    assert all(torch.isfinite(value).all() for value in y.values())
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    test_stage3_audio_encoder_has_finite_outputs_and_gradients()
    print("stage3 numerical stability tests passed")
