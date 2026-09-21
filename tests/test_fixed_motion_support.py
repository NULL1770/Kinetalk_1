"""Missing motion channels cannot drive supported outputs through the ODE."""
import copy

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.audio_residual_flow import AudioResidualFlow


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(61)
    yield
    torch.set_num_threads(previous)


def config():
    return {
        'data': {'content_dim': 8, 'motion_dim': 6, 'neutral_output_indices': [2, 3],
                 'emotion_classes': ['neutral', 'happy', 'sad'], 'num_intensity_levels': 3,
                 'audio_emotion_dim': 5},
        'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8, 'hidden_dim': 8,
                  'heads': 2, 'dropout': 0., 'dit_dim': 8, 'dit_depth': 1,
                  'residual_scale': .25, 'affect_stride': 4, 'affect_rank': 3,
                  'global_condition_dropout': 0., 'style_condition_dropout': 0.,
                  'motion_support': [True, True, True, True, True, False]}}


def inputs(system):
    valid = torch.tensor([[True, True, False, True, True], [True, True, True, True, False]])
    content = torch.randn(2, 5, 8)
    identity = system.encode_identity(torch.randn(2, 2, 5, 6))
    affect = system.encode_audio(torch.randn(2, 5, 5), valid)
    return valid, content, identity, affect


def test_flow_cleans_missing_target_noise_and_baseline_before_channel_mixing():
    system = NeutralAffectSystem(config()).eval()
    valid, content, identity, affect = inputs(system)
    channels = torch.ones(2, 6, dtype=torch.bool)
    channels[0, 4] = False  # Train per-item missingness; fixed support remains unchanged.
    observed = valid[..., None] & channels[:, None] & system.motion_support
    motion, noise = torch.randn(2, 5, 6), torch.randn(2, 5, 6)
    base = system.base(content, valid)
    time = torch.tensor([.25, .75])
    first = system.flow(motion, content, valid, identity, affect, noise=noise, time=time,
                        base=base, observation_mask=channels)
    changed_motion = motion.masked_fill(~observed, float('nan')).requires_grad_()
    changed_noise = noise.masked_fill(~observed, float('nan')).requires_grad_()
    changed_base = {**base, 'b0': base['b0'].masked_fill(~observed, float('nan'))}
    changed_identity = {**identity, 'baseline': identity['baseline'].clone()}
    changed_identity['baseline'][:, 5] = float('nan')
    second = system.flow(changed_motion, content, valid, changed_identity, affect,
                         noise=changed_noise, time=time, base=changed_base, observation_mask=channels)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for key in ('prediction', 'velocity_target', 'target_residual', 'x_t', 'motion', 'b0'):
        assert second[key][~observed].count_nonzero() == 0
    loss = (second['prediction'][observed] - second['velocity_target'][observed]).square().mean()
    loss.backward()
    assert torch.isfinite(changed_motion.grad).all() and torch.isfinite(changed_noise.grad).all()
    assert changed_motion.grad[~observed].count_nonzero() == 0
    assert changed_noise.grad[~observed].count_nonzero() == 0
    assert system.renderer.motion_input.weight.grad[:, 5].count_nonzero() == 0
    assert system.renderer.output.weight.grad[5].count_nonzero() == 0


def test_decode_fixed_support_projects_every_step_and_never_uses_query_gt():
    system = NeutralAffectSystem(config()).eval()
    valid, content, identity, affect = inputs(system)
    noise = torch.randn(2, 5, 6)
    with torch.no_grad():
        first = system.generate(content, valid, identity, affect, noise, steps=4)
        # Even an arbitrary learned output in an untrained channel cannot
        # re-enter motion_input on the next solver step.
        system.renderer.output.weight[5].fill_(1e8)
        system.renderer.output.bias[5] = 1e8
        noise[:, :, 5] = float('nan')
        seen = []
        hook = system.renderer.register_forward_pre_hook(lambda _module, args: seen.append(args[0].clone()))
        second = system.generate(content, valid, identity, affect, noise, steps=4)
        hook.remove()
    assert len(seen) == 4
    support = valid[..., None] & system.motion_support
    assert all(state[~support].count_nonzero() == 0 for state in seen)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        assert second[key][~support].count_nonzero() == 0


def test_fixed_support_covers_identity_teacher_and_b0_without_new_state_keys():
    cfg = config()
    system = NeutralAffectSystem(cfg).eval()
    refs, residual = torch.randn(2, 2, 5, 6), torch.randn(2, 5, 6)
    identity = system.encode_identity(refs)
    teacher = system.encode_motion(residual)
    refs[..., 5] = float('nan'); residual[..., 5] = float('nan')
    for key, expected in identity.items():
        torch.testing.assert_close(system.encode_identity(refs)[key], expected, rtol=0, atol=0)
    for key, expected in teacher.items():
        torch.testing.assert_close(system.encode_motion(residual)[key], expected, rtol=0, atol=0)
    assert system.base(torch.randn(2, 5, 8))['b0'][..., 5].count_nonzero() == 0
    assert identity['baseline'][..., 5].count_nonzero() == 0
    assert 'motion_support' not in system.state_dict()
    restored = NeutralAffectSystem(copy.deepcopy(cfg))
    restored.load_state_dict(system.state_dict(), strict=True)
    assert torch.equal(restored.motion_support, system.motion_support)
    old_cfg = copy.deepcopy(cfg); del old_cfg['model']['motion_support']
    legacy = NeutralAffectSystem(old_cfg)
    legacy.load_state_dict(system.state_dict(), strict=True)
    assert legacy.motion_support.all()
    with pytest.raises(ValueError, match='Boolean'):
        restored.set_motion_support([1, 1, 1, 1, 1, 0])


def test_observed_nonfinite_motion_and_noise_fail_fast():
    system = NeutralAffectSystem(config()).eval()
    valid, content, identity, affect = inputs(system)
    motion, noise = torch.randn(2, 5, 6), torch.randn(2, 5, 6)
    motion[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        system.flow(motion, content, valid, identity, affect, noise=noise)
    motion[0, 0, 0] = 0.; noise[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        system.flow(motion, content, valid, identity, affect, noise=noise)
    with pytest.raises(ValueError, match='finite'):
        system.generate(content, valid, identity, affect, noise)


def test_residual_support_protects_deterministic_channels_across_noise_seeds():
    cfg = config()
    cfg['model']['residual_support'] = [True, True, False, False, True, False]
    system = NeutralAffectSystem(cfg).eval()
    valid, content, identity, affect = inputs(system)
    base = system.base(content, valid)
    identity = {**identity, 'baseline': torch.tensor([[.02, .02, .02, .02, .02, 0.]]).expand(2, -1)}
    protected = [2, 3]
    expected = torch.where(valid[..., None], base['b0'] + identity['baseline'][:, None], 0.)
    values = []
    for seed in (42, 123, 2026):
        noise = torch.randn(2, 5, 6, generator=torch.Generator().manual_seed(seed))
        generated = system.generate(content, valid, identity, affect, noise, steps=4, base=base)
        torch.testing.assert_close(generated['motion'][..., protected], expected[..., protected], rtol=0, atol=0)
        assert generated['residual'][..., protected].count_nonzero() == 0
        values.append(generated['motion'])
    assert (values[0][..., 0] - values[1][..., 0]).abs().max() > 1e-6
    target = torch.randn(2, 5, 6)
    noise = torch.randn_like(target)
    output = system.flow(target, content, valid, identity, affect, noise=noise,
                         time=torch.tensor([.3, .7]), base=base)
    for key in ('target_residual', 'velocity_target', 'prediction', 'x_t', 'residual'):
        assert output[key][..., protected].count_nonzero() == 0
    torch.testing.assert_close(output['motion'][..., protected], expected[..., protected], rtol=0, atol=0)
    score_mask = output['observation_mask']
    (output['prediction'][score_mask]-output['velocity_target'][score_mask]).square().mean().backward()
    assert system.renderer.output.weight.grad[[0, 1, 4]].abs().sum() > 0
    assert system.renderer.output.weight.grad[protected].count_nonzero() == 0
    assert system.renderer.motion_input.weight.grad[:, protected].count_nonzero() == 0
    restored = NeutralAffectSystem(cfg)
    restored.load_state_dict(system.state_dict(), strict=True)
    assert torch.equal(restored.residual_support, system.residual_support)
    with pytest.raises(ValueError, match='subset'):
        restored.set_residual_support([True, True, False, False, True, True])


def test_upper_flow_checks_training_observations_but_decode_ignores_gt_mask():
    model = AudioResidualFlow(config(), stride=2).eval()
    valid = torch.tensor([[True, True, False, True]])
    q = {'valid': valid, 'h0': torch.randn(1, 4, 8), 'channel_mask': torch.ones(1, 52, dtype=torch.bool)}
    identity = {'code': torch.randn(1, 8)}
    affect = {'global': torch.randn(1, 8), 'intensity_value': torch.ones(1, 1)}
    local, state = torch.randn(1, 4, 8), torch.randn(1, 4, 4)
    target, noise, time = torch.randn(1, 4, 9), torch.randn(1, 4, 9), torch.tensor([.5])
    target[:, 2] = float('nan'); noise[:, 2] = float('nan')
    loss = model.flow_loss(target, q, identity, affect, local, state, noise, time)
    assert torch.isfinite(loss)
    first = model.decode(q, identity, affect, local, state, noise, 3)
    q['channel_mask'][:, 41] = False
    with pytest.raises(ValueError, match='all nine observed'):
        model.flow_loss(target, q, identity, affect, local, state, noise, time)
    second = model.decode(q, identity, affect, local, state, noise, 3)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first[:, 2].count_nonzero() == 0
