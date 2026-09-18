"""Contracts for native ordered compression and a noise-owned latent prior."""
import copy

import pytest
import torch
from torch.nn import functional as F

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE


@pytest.fixture(autouse=True, scope='module')
def small_cpu_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def ae_case():
    torch.manual_seed(3029)
    ae = ContinuousUpperAE(hidden=24, latent_dim=8, depth=2)
    valid = torch.arange(13)[None] < torch.tensor([13, 7])[:, None]
    motion = torch.randn(2, 13, 9)
    return ae, motion, valid


def flow_case():
    torch.manual_seed(4729)
    model = ContinuousLatentFlow(context_dim=7, audio_dim=6, latent_dim=8, hidden=24, depth=3)
    valid = torch.arange(5)[None] < torch.tensor([5, 3])[:, None]
    frame_valid = (torch.arange(25)[None] < torch.tensor([23, 11])[:, None]).reshape(2, 5, 5)
    context = torch.randn(2, 7)
    audio = torch.randn(2, 5, 5, 6)
    noise = torch.randn(2, 5, 8)
    time = torch.tensor([.2, .7])
    return model, valid, frame_valid, context, audio, noise, time


def test_ae_packs_native_values_in_order_and_explicitly_masks_partial_tail():
    ae, motion, valid = ae_case()
    motion[~valid] = float('nan')
    captured = []
    handle = ae.encoder_input.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    z, latent_valid = ae.encode(motion, valid)
    handle.remove()
    assert z.shape == (2, 3, 8)
    assert torch.equal(latent_valid, torch.tensor([[True, True, True], [True, True, False]]))
    packed = captured[0]
    torch.testing.assert_close(packed[0, 0, :45], motion[0, :5].flatten())
    torch.testing.assert_close(packed[0, 2, :27], motion[0, 10:13].flatten())
    assert torch.count_nonzero(packed[0, 2, 27:45]) == 0
    torch.testing.assert_close(packed[0, 2, 45:], torch.tensor([1., 1., 1., 0., 0.]))
    assert torch.count_nonzero(z[~latent_valid]) == 0
    shuffled = motion.clone(); shuffled[:, :5] = shuffled[:, :5].flip(1)
    assert not torch.allclose(ae.encode(shuffled, valid)[0], z)


def test_ae_padding_values_and_appended_padding_cannot_change_observed_output():
    ae, motion, valid = ae_case()
    original = ae(motion, valid)
    motion[~valid] = float('nan')
    padded_motion = F.pad(motion, (0, 0, 0, 12), value=float('nan'))
    padded_valid = F.pad(valid, (0, 12), value=False)
    reconstructed = ae(padded_motion, padded_valid)
    torch.testing.assert_close(original, reconstructed[:, :13], rtol=2e-6, atol=2e-6)
    assert torch.isfinite(reconstructed).all()
    assert torch.count_nonzero(reconstructed[~padded_valid]) == 0
    z, _ = ae.encode(padded_motion, padded_valid)
    z[:, 3:] = float('nan')
    torch.testing.assert_close(reconstructed, ae.decode(z, padded_valid), rtol=0, atol=0)


@pytest.mark.parametrize('bad_mask', [torch.tensor([[True, False, True]]),
                                    torch.zeros(1, 3, dtype=torch.bool),
                                    torch.ones(1, 3)])
def test_ae_rejects_empty_non_boolean_and_gap_masks(bad_mask):
    ae = ContinuousUpperAE(hidden=8, depth=1)
    with pytest.raises(ValueError, match='prefix|Boolean'):
        ae(torch.zeros(1, 3, 9), bad_mask)


def test_ae_preserves_gradients_and_does_not_bound_raw_residuals():
    ae, motion, valid = ae_case()
    motion.requires_grad_()
    output = ae(motion, valid)
    (output[valid] - motion[valid]).square().mean().backward()
    assert torch.isfinite(motion.grad).all() and motion.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(motion.grad[~valid]) == 0
    for parameter in ae.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in ae.parameters():
        parameter.data.zero_()
    ae.decoder_output.bias.data.fill_(3.)
    assert torch.equal(ae(motion.detach(), valid)[valid], torch.full_like(output[valid], 3.))


def test_ae_small_trajectory_overfit_keeps_native_peak():
    torch.manual_seed(548)
    ae = ContinuousUpperAE(latent_dim=16, hidden=32, depth=2)
    valid = torch.ones(2, 20, dtype=torch.bool)
    t = torch.linspace(0, 2 * torch.pi, 20)
    motion = torch.stack((t.sin(), (t + .4).cos()))[..., None].repeat(1, 1, 9)
    motion[:, 9, 0] += 1.8  # a one-frame peak inside a five-frame block
    optimizer = torch.optim.Adam(ae.parameters(), lr=.004)
    initial = (ae(motion, valid) - motion).square().mean().item()
    for _ in range(180):
        optimizer.zero_grad()
        loss = (ae(motion, valid) - motion).square().mean()
        loss.backward(); optimizer.step()
    reconstructed = ae(motion, valid).detach()
    final = (reconstructed - motion).square().mean().item()
    assert final < initial * .015 and final < .008
    assert (reconstructed[:, 9, 0] - motion[:, 9, 0]).abs().max() < .25
    # This is implementation capacity only, not generalization or naturalness.


def test_flow_loss_is_exact_masked_fm_and_has_audio_context_gradients():
    model, valid, frame_valid, context, audio, noise, time = flow_case()
    context.requires_grad_(); audio.requires_grad_()
    target = torch.randn_like(noise)
    fraction = time[:, None, None]
    velocity = model.velocity((1-fraction)*noise + fraction*target, time, valid, context, audio,
                              audio_frame_valid=frame_valid)
    expected = (velocity-(target-noise))[valid].square().mean()
    loss = model.flow_loss(target, valid, context, audio, noise, time, audio_frame_valid=frame_valid)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert context.grad.abs().sum() > 0 and audio.grad[frame_valid].abs().sum() > 0
    assert torch.count_nonzero(audio.grad[~frame_valid]) == 0
    assert torch.isfinite(audio.grad).all()
    assert model.audio_frame_projection[0].weight.grad.abs().sum() > 0
    assert model.context_projection.weight.grad.abs().sum() > 0


def test_flow_no_audio_branch_is_invariant_and_has_no_audio_gradient():
    model, valid, frame_valid, context, audio, noise, time = flow_case()
    audio.requires_grad_()
    first = model.velocity(noise, time, valid, context, audio, False, audio_frame_valid=frame_valid)
    second = model.velocity(noise, time, valid, context, audio * 100 + 900, False,
                            audio_frame_valid=frame_valid)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.square().sum().backward()
    assert audio.grad is None
    assert model.audio_frame_projection[0].weight.grad is None


def test_flow_tail_nan_and_appended_padding_have_no_effect():
    model, valid, frame_valid, context, audio, noise, time = flow_case()
    original = model.sample(valid, context, audio, noise, steps=3, audio_frame_valid=frame_valid)
    audio[~frame_valid] = float('nan'); noise[~valid] = float('nan')
    pad_valid = F.pad(valid, (0, 3), value=False)
    pad_frames = F.pad(frame_valid, (0, 0, 0, 3), value=False)
    pad_audio = F.pad(audio, (0, 0, 0, 0, 0, 3), value=float('nan'))
    pad_noise = F.pad(noise, (0, 0, 0, 3), value=float('nan'))
    result = model.sample(pad_valid, context, pad_audio, pad_noise, steps=3, audio_frame_valid=pad_frames)
    torch.testing.assert_close(original, result[:, :5], rtol=2e-6, atol=2e-6)
    assert torch.isfinite(result).all() and torch.count_nonzero(result[~pad_valid]) == 0


def test_flow_is_joint_over_time_and_sampling_leaves_rng_and_input_unchanged():
    model, valid, frame_valid, context, audio, noise, time = flow_case()
    state = torch.random.get_rng_state().clone()
    before = noise.clone()
    a = model.sample(valid, context, audio, noise, steps=3, audio_frame_valid=frame_valid)
    b = model.sample(valid, context, audio, noise, steps=3, audio_frame_valid=frame_valid)
    assert torch.equal(a, b) and torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(before, noise)
    perturbed = noise.clone(); perturbed[:, 1] += 2
    base = model.velocity(noise, time, valid, context, audio, audio_frame_valid=frame_valid)
    changed = model.velocity(perturbed, time, valid, context, audio, audio_frame_valid=frame_valid)
    assert (base[:, 0]-changed[:, 0]).abs().max() > 1e-7


@pytest.mark.parametrize('kind', ['gap', 'empty', 'time', 'audio_nan', 'tail_gap', 'tail_mismatch',
                                 'target_nan', 'context_nan', 'use_audio'])
def test_flow_rejects_invalid_observed_inputs(kind):
    model, valid, frames, context, audio, noise, time = flow_case()
    target = torch.randn_like(noise); use_audio = True
    if kind == 'gap': valid[0, 1] = False
    if kind == 'empty': valid[0] = False
    if kind == 'time': time[0] = 1.01
    if kind == 'audio_nan': audio[0, 0, 0, 0] = float('nan')
    if kind == 'tail_gap': frames[0, 0, 0] = False
    if kind == 'tail_mismatch': frames[0, -1] = False
    if kind == 'target_nan': target[0, 0, 0] = float('nan')
    if kind == 'context_nan': context[0, 0] = float('nan')
    if kind == 'use_audio': use_audio = 1
    with pytest.raises(ValueError):
        model.flow_loss(target, valid, context, audio, noise, time, use_audio, audio_frame_valid=frames)


def test_configs_and_states_roundtrip_and_flow_is_not_bounded():
    ae, motion, valid = ae_case()
    restored_ae = ContinuousUpperAE(**ae.config); restored_ae.load_state_dict(copy.deepcopy(ae.state_dict()))
    torch.testing.assert_close(ae(motion, valid), restored_ae(motion, valid), rtol=0, atol=0)
    model, valid, frames, context, audio, noise, _ = flow_case()
    restored = ContinuousLatentFlow(**model.config); restored.load_state_dict(copy.deepcopy(model.state_dict()))
    a = model.sample(valid, context, audio, noise, steps=2, audio_frame_valid=frames)
    torch.testing.assert_close(a, restored.sample(valid, context, audio, noise, steps=2, audio_frame_valid=frames), rtol=0, atol=0)
    for parameter in model.parameters(): parameter.data.zero_()
    noise.fill_(-3.)
    result = model.sample(valid, context, audio, noise, steps=2, audio_frame_valid=frames)
    assert torch.equal(result[valid], noise[valid])
    with pytest.raises(ValueError, match='steps'):
        model.sample(valid, context, audio, noise, steps=0, audio_frame_valid=frames)
