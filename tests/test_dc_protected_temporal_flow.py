import pytest
import torch

from kinetalk_b0.models.dc_protected_temporal_flow import (
    DCProtectedTemporalFlow,
    project_temporal_dc,
)
from kinetalk_b0.models.temporal_audio_residual_flow import correlated_native_noise


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs():
    torch.manual_seed(137)
    cfg = {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8,
           'dit_dim': 12, 'dit_depth': 1, 'heads': 3, 'dropout': 0.}
    valid = torch.ones(2, 13, dtype=torch.bool)
    valid[0, 3] = False
    valid[1, 9:] = False
    q = {'valid': valid, 'h0': torch.randn(2, 13, 8),
         'channel_mask': torch.ones(2, 52, dtype=torch.bool)}
    identity = {'code': torch.randn(2, 8)}
    affect = {'global': torch.randn(2, 8), 'intensity_value': torch.ones(2, 1)}
    local = torch.randn(2, 13, 8)
    state = torch.randn(2, 13, 4)
    noise = torch.randn(2, 13, 9)
    target = torch.randn(2, 13, 9)
    return cfg, q, identity, affect, local, state, noise, target


def assert_zero_dc(value, valid, atol=3e-7):
    assert value[~valid].count_nonzero() == 0
    mean = value.double().sum(1) / valid.sum(1)[:, None]
    torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0, atol=atol)


def test_projection_idempotent_masked_differentiable_and_offset_invariant():
    _, q, *_, noise, _ = inputs()
    valid = q['valid']
    value = noise.double().requires_grad_()
    dirty = torch.where(valid[..., None], value, float('nan'))
    projected = project_temporal_dc(dirty, valid)
    assert_zero_dc(projected, valid, atol=1e-14)
    torch.testing.assert_close(project_temporal_dc(projected, valid), projected, atol=1e-14, rtol=0)
    offset = torch.arange(18, dtype=torch.float64).reshape(2, 1, 9)
    torch.testing.assert_close(project_temporal_dc(dirty + offset, valid), projected, atol=3e-15, rtol=0)
    projected.square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert value.grad[~valid].count_nonzero() == 0
    assert value.grad[valid].abs().sum() > 0


def test_projection_preserves_nonconstant_slow_motion_and_amplitude():
    # A linear variation spanning the entire clip is a low-frequency state,
    # not a fast innovation.  Removing only DC retains it exactly.
    ramp = torch.linspace(-3., 3., 129, dtype=torch.float64)[None, :, None].expand(1, 129, 9)
    valid = torch.ones(1, 129, dtype=torch.bool)
    actual = project_temporal_dc(ramp + 7., valid)
    torch.testing.assert_close(actual, ramp, rtol=0, atol=1e-14)
    assert actual.abs().max() > 1  # No range clamp or amplitude normalization.
    single = project_temporal_dc(torch.ones(1, 1, 9), torch.ones(1, 1, dtype=torch.bool))
    assert single.count_nonzero() == 0


def test_start_noise_correlated_once_same_in_loss_and_decode_and_no_rng_consumed():
    cfg, q, identity, affect, local, state, noise, target = inputs()
    model = DCProtectedTemporalFlow(cfg).eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    expected_start = project_temporal_dc(correlated_native_noise(noise, q['valid'], rho=model.noise_rho), q['valid'])
    rng = torch.get_rng_state().clone()
    first = model.decode(q, identity, affect, local, state, noise, 3)
    second = model.decode(q, identity, affect, local, state, noise, 3)
    torch.testing.assert_close(first, expected_start, atol=2e-7, rtol=0)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    actual_loss = model.flow_loss(target, q, identity, affect, local, state, noise, torch.zeros(2))
    expected_velocity = project_temporal_dc(project_temporal_dc(target, q['valid']) - expected_start, q['valid'])
    torch.testing.assert_close(actual_loss, expected_velocity[q['valid']].square().mean())
    other = model.decode(q, identity, affect, local, state, noise + torch.randn_like(noise), 3)
    assert not torch.equal(first, other)
    assert_zero_dc(other, q['valid'])


def test_training_and_every_decode_state_velocity_stay_zero_dc_with_gradients():
    cfg, q, identity, affect, local, state, noise, target = inputs()
    model = DCProtectedTemporalFlow(cfg)
    local.requires_grad_()
    seen = []
    handle = model.renderer.register_forward_pre_hook(lambda _, args, kwargs: seen.append(args[0].detach().clone()), with_kwargs=True)
    loss = model.flow_loss(target, q, identity, affect, local, state, noise, torch.tensor([.2, .7]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.temporal_up.weight.grad.abs().sum() > 0
    assert local.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        decoded = model.decode(q, identity, affect, local, state, noise, 5)
    handle.remove()
    # The temporal up path starts at zero, so renderer inputs here are the
    # actual flow states; all training/intermediate Euler inputs obey DC.
    assert len(seen) == 6
    for value in seen + [decoded]:
        assert_zero_dc(value, q['valid'])
    conditions = model._conditions(q['valid'], q['h0'], identity['code'], affect, local, state)
    velocity = model._velocity_unrestricted(project_temporal_dc(noise, q['valid']), torch.zeros(2), q['valid'], conditions)
    assert_zero_dc(velocity, q['valid'])


def test_decode_ignores_targets_missing_channels_and_appended_padding():
    cfg, q, identity, affect, local, state, noise, _ = inputs()
    model = DCProtectedTemporalFlow(cfg).eval()
    expected = model.decode(q, identity, affect, local, state, noise, 2)
    dirty = {**q, 'motion': torch.full((2, 13, 52), float('nan')),
             'channel_mask': torch.zeros_like(q['channel_mask'])}
    torch.testing.assert_close(model.decode(dirty, identity, affect, local, state, noise, 2), expected, rtol=0, atol=0)
    pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, 4), value=float('nan'))
    padded_q = {**q, 'h0': pad(q['h0']), 'valid': torch.nn.functional.pad(q['valid'], (0, 4), value=False)}
    actual = model.decode(padded_q, identity, affect, pad(local), pad(state), pad(noise), 2)
    torch.testing.assert_close(actual[:, :13], expected, atol=2e-6, rtol=2e-6)
    assert_zero_dc(actual, padded_q['valid'])


def test_fixed_support_invalid_observations_and_budgets_fail_closed():
    cfg, q, identity, affect, local, state, noise, target = inputs()
    model = DCProtectedTemporalFlow(cfg)
    q['channel_mask'][:, 41] = False
    with pytest.raises(ValueError, match='all nine'):
        model.flow_loss(target, q, identity, affect, local, state, noise, torch.zeros(2))
    for budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='integer'):
            model.decode(q, identity, affect, local, state, noise, budget)
    for bad_valid in (torch.zeros_like(q['valid']), q['valid'].float(), q['valid'][:, :2]):
        with pytest.raises(ValueError, match='valid'):
            project_temporal_dc(noise, bad_valid)
    noise[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='noise'):
        model.decode(q, identity, affect, local, state, noise, 2)
    with pytest.raises(ValueError, match='finite'):
        project_temporal_dc(noise, q['valid'])


def test_architecture_metadata_can_be_reconstructed_and_state_dict_roundtrips():
    cfg, q, identity, affect, local, state, noise, _ = inputs()
    model = DCProtectedTemporalFlow(cfg, stride=8, noise_rho=.6, position_scale=.03)
    assert model.architecture_config['projection'] == 'native_valid_clip_channel_dc'
    assert model.architecture_config['architecture'] == 'dc_protected_temporal_flow_v1'
    copy = DCProtectedTemporalFlow(cfg, stride=model.stride, **model.temporal_config)
    copy.load_state_dict(model.state_dict())
    torch.testing.assert_close(copy.decode(q, identity, affect, local, state, noise, 3),
                               model.decode(q, identity, affect, local, state, noise, 3), rtol=0, atol=0)
