import torch

from kinetalk_b0.models.centered_upper_flow import CenteredUpperFlow, center_valid, fit_dynamic_statistics


CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 12, 'dit_depth': 1, 'heads': 3, 'dropout': 0.}}


def test_prior_expected_amplitude_and_time_correlation():
    torch.manual_seed(71)
    valid = torch.ones(3000, 24, dtype=torch.bool)
    valid[:, 20:] = False
    white = torch.randn(3000, 24, 9)
    model = CenteredUpperFlow(CFG, torch.full((9,), .9))
    smooth = model.prior(white, valid)
    torch.testing.assert_close(smooth.sum(1), torch.zeros(3000, 9), atol=2e-5, rtol=0)
    assert abs(float(smooth[valid].square().mean())-1) < .035
    independent = CenteredUpperFlow(CFG, torch.zeros(9)).prior(white, valid)
    assert (smooth[:, 1:20]-smooth[:, :19]).square().mean() < (independent[:, 1:20]-independent[:, :19]).square().mean()*.4
    padded = model.prior(torch.cat((white, torch.randn(3000, 7, 9)), 1), torch.cat((valid, torch.zeros(3000, 7, dtype=torch.bool)), 1))
    torch.testing.assert_close(padded[:, :24], smooth, atol=1e-6, rtol=1e-6)


def test_decode_is_centered_and_temporal_condition_receives_gradient():
    torch.manual_seed(72)
    model = CenteredUpperFlow(CFG, torch.full((9,), .8))
    valid = torch.ones(2, 12, dtype=torch.bool); valid[0, -3:] = False
    content = torch.randn(2, 12, 4)
    identity = torch.randn(2, 2)
    affect = {'global': torch.randn(2, 3), 'intensity_value': torch.randn(2, 1)}
    local = torch.randn(2, 12, 3, requires_grad=True)
    noise = model.prior(torch.randn(2, 12, 9), valid)
    target = torch.randn(2, 12, 9)
    loss = model.flow_loss(target, valid, content, identity, affect, local, None, noise, torch.tensor([.2, .6]))
    loss.backward()
    assert torch.isfinite(local.grad).all() and local.grad.abs().sum() > 0
    decoded = model.decode(valid, content, identity, affect, local, None, noise, steps=3)
    torch.testing.assert_close(decoded.sum(1), torch.zeros(2, 9), atol=3e-6, rtol=0)
    assert not torch.equal(decoded, model.decode(valid, content, identity, affect, local.flip(1), None, noise, steps=3))


def test_fit_uses_dynamics_and_ignores_padding_and_constant_offsets():
    torch.manual_seed(73)
    valid = torch.ones(4, 16, dtype=torch.bool); valid[:, -4:] = False
    motion = torch.randn(4, 16, 9)
    a, b = fit_dynamic_statistics(motion, valid)
    moved = motion+torch.randn(4, 1, 9)*10
    moved[~valid] = float('nan')
    c, d = fit_dynamic_statistics(moved, valid)
    torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b, d, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(center_valid(moved, valid)).all()
