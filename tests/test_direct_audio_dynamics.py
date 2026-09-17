import copy

import torch

from scripts.train_audio_conditioned_flow_probe import configure_adaptation
from scripts.train_direct_audio_dynamics import (
    DirectAudioEncoder, direct_affect, feature_statistics, generated_losses,
    interpolate_local, training_loss,
)
from tests.test_audio_conditioned_flow_probe import fixture


def test_encoder_mask_padding_and_interpolation_are_stable():
    torch.manual_seed(3)
    features = torch.randn(2, 4, 11)
    weight = torch.tensor([[4., 4., 2., 0.], [4., 4., 4., 4.]])
    mean, std = feature_statistics(features, weight)
    encoder = DirectAudioEncoder(mean, std, output_dim=7, hidden=8).eval()
    with torch.no_grad():
        encoder.output.weight.normal_(); encoder.output.bias.normal_()
    a = encoder(features, weight)
    padded = torch.cat([features, torch.randn(2, 3, 11) * 100], 1)
    padded_weight = torch.cat([weight, torch.zeros(2, 3)], 1)
    b = encoder(padded, padded_weight)[:, :4]
    # cuDNN/CPU convolution kernels can differ by one float32 ulp when the
    # padded tensor shape changes; the semantic output remains invariant.
    torch.testing.assert_close(a, b, rtol=0, atol=2e-7)
    valid = torch.ones(2, 16, dtype=torch.bool)
    local = interpolate_local(a, weight, valid, 4)
    assert local.shape == (2, 16, 7) and torch.isfinite(local).all()


def test_dynamics_loss_backpropagates_to_direct_encoder_and_renderer():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        system, batch, _, weight = fixture()
        # Undo the narrow old-probe setup and open the renderer used here.
        system = copy.deepcopy(system)
        for p in system.parameters(): p.requires_grad_(False)
        system.renderer.requires_grad_(True); system.eval()
        features = torch.randn(2, 2, 13)
        encoder = DirectAudioEncoder(*feature_statistics(features, weight), output_dim=64, hidden=8).eval()
        scales = {'dynamic': torch.full((52,), .1)}
        noise = torch.randn_like(batch['q']['motion'])
        time = torch.tensor([0., .7])
        total, losses = training_loss(system, encoder, batch, features, weight, scales, noise, time, 'dynamics')
        assert set(losses) == {'flow', 'centered_upper'} and torch.isfinite(total)
        total.backward()
        assert encoder.output.weight.grad is not None and encoder.output.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.renderer.parameters())
    finally:
        torch.set_num_threads(prior)


def test_generated_loss_ignores_padding_and_reverse_zero_modes_differ():
    system, batch, _, weight = fixture()
    prediction = batch['q']['motion'].clone()
    scales = {'dynamic': torch.ones(52)}
    assert generated_losses(prediction, batch, scales)['centered_upper'] == 0
    local = torch.randn(2, 2, 64)
    full = direct_affect(system, batch, local, weight, 'full')['local']
    zero = direct_affect(system, batch, local, weight, 'zero')['local']
    reverse = direct_affect(system, batch, local, weight, 'reverse')['local']
    assert zero.abs().sum() == 0 and not torch.equal(full, reverse)
