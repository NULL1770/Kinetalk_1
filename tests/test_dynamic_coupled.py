"""Gradients and preservation for matched renderer/coupled adaptation."""

from copy import deepcopy

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_dynamic_coupled import (
    GLOBAL_KEYS, configure_trainable, coupled_objective, optimize_coupled, teacher_probability,
)


@pytest.fixture(autouse=True)
def seed():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(57)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def setup():
    cfg = {'data': {'content_dim': 8, 'motion_dim': 6, 'neutral_output_indices': [2, 3],
                    'emotion_classes': ['neutral', 'happy'], 'num_intensity_levels': 3, 'audio_emotion_dim': 5},
           'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8, 'hidden_dim': 8,
                     'heads': 2, 'dropout': 0., 'dit_dim': 8, 'dit_depth': 1,
                     'residual_scale': .25, 'affect_stride': 4, 'affect_rank': 3,
                     'audio_control_refiner': True}}
    system = NeutralAffectSystem(cfg).eval()
    query = {'audio': torch.randn(4, 17, 5), 'content': torch.randn(4, 17, 8),
             'motion': torch.randn(4, 17, 6)*.1, 'valid': torch.ones(4, 17, dtype=torch.bool),
             'channel_mask': torch.ones(4, 6, dtype=torch.bool)}
    query['valid'][2, 13:] = False
    with torch.no_grad():
        base = system.base(query['content'], query['valid'])
        query['residual'] = query['motion'] - base['b0']
        identity = system.encode_identity(torch.randn(4, 2, 17, 6)*.03)
    scales = {'controls': torch.ones(3), 'global': torch.ones(8)}
    return system, query, base, identity, scales


def objective(setup, probability, seed=82):
    system, query, base, identity, scales = setup
    return coupled_objective(system, query, base, identity, scales, torch.Generator().manual_seed(seed), probability)


@pytest.mark.parametrize('mode', ['renderer', 'coupled'])
def test_global_paths_frozen_and_teacher_grad_only_for_coupled_flow(setup, mode):
    system, query, base, identity, scales = setup
    allowed = configure_trainable(system, mode)
    audio_before = system.encode_audio(query['audio'], query['valid'])
    teacher_before = system.encode_motion(query['residual']-identity['baseline'][:, None], query['valid'])
    frozen = {name: tensor.clone() for name, tensor in system.state_dict().items() if name not in allowed}
    loss, _, random = objective(setup, 1.)
    assert random[0].sum() == len(query['motion'])-1  # Preserve one audio item.
    loss.backward()
    assert system.audio_encoder.control_head.weight.grad.abs().sum() > 0
    assert system.audio_encoder.control_refiner.output.weight.grad.abs().sum() > 0
    assert system.renderer.output.weight.grad.abs().sum() > 0
    teacher_grad = system.motion_teacher.control_head.weight.grad
    assert teacher_grad is None if mode == 'renderer' else teacher_grad.abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    audio_after = system.encode_audio(query['audio'], query['valid'])
    teacher_after = system.encode_motion(query['residual']-identity['baseline'][:, None], query['valid'])
    for key in GLOBAL_KEYS:
        torch.testing.assert_close(audio_before[key], audio_after[key], atol=0, rtol=0)
        torch.testing.assert_close(teacher_before[key], teacher_after[key], atol=0, rtol=0)
    for name, tensor in frozen.items():
        torch.testing.assert_close(tensor, system.state_dict()[name], atol=0, rtol=0)
    assert all(p.grad is None for name, p in system.named_parameters() if name not in allowed)


def test_alignment_detaches_teacher_and_no_teacher_batch_skips_momentum(setup):
    system = setup[0]
    configure_trainable(system, 'coupled')
    params = [p for p in system.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.1)
    loss, _, random = objective(setup, 1.)
    optimize_coupled(loss, optimizer, params, system, bool(random[0].any()))
    teacher = {name: tensor.clone() for name, tensor in system.motion_teacher.control_head.state_dict().items()}
    # The teacher has optimizer momentum from the first update. It still must
    # not move when the next batch's alignment sees detached teacher targets.
    loss, _, random = objective(setup, 0.)
    assert not random[0].any()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert system.motion_teacher.control_head.weight.grad.count_nonzero() == 0
    loss, _, random = objective(setup, 0.)
    optimize_coupled(loss, optimizer, params, system, False)
    for name, tensor in teacher.items():
        torch.testing.assert_close(tensor, system.motion_teacher.control_head.state_dict()[name], atol=0, rtol=0)
    assert all(p.grad is None for p in system.motion_teacher.control_head.parameters())


def test_both_arms_share_initial_loss_and_noise_but_teacher_updates_differ(setup):
    renderer = setup
    coupled = (deepcopy(setup[0]), *setup[1:])
    configure_trainable(renderer[0], 'renderer')
    configure_trainable(coupled[0], 'coupled')
    first, first_parts, first_random = objective(renderer, .5)
    second, second_parts, second_random = objective(coupled, .5)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first_parts == second_parts
    for left, right in zip(first_random, second_random):
        assert torch.equal(left, right)
    initial = {name: value.clone() for name, value in coupled[0].motion_teacher.control_head.state_dict().items()}
    for state, loss, random in ((renderer, first, first_random), (coupled, second, second_random)):
        system = state[0]
        params = [p for p in system.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=.001)
        optimize_coupled(loss, optimizer, params, system, bool(random[0].any()))
    assert any(not torch.equal(value, coupled[0].motion_teacher.control_head.state_dict()[name]) for name, value in initial.items())
    for name, value in initial.items():
        torch.testing.assert_close(value, renderer[0].motion_teacher.control_head.state_dict()[name], atol=0, rtol=0)


def test_teacher_schedule_is_fixed_half_then_tenth():
    assert teacher_probability(0, 1600) == .5
    assert teacher_probability(799, 1600) == .5
    assert teacher_probability(800, 1600) == .1
    assert teacher_probability(1599, 1600) == .1
