import copy

import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts.train_full_staged import audio_parameter_groups, audio_affect, parser
from tests.test_full_staged_runner import config


def test_frozen_receiver_still_trains_student_through_flow_and_never_updates_weights():
    torch.set_num_threads(1)
    torch.manual_seed(104)
    system = NeutralAffectSystem(config()).eval().requires_grad_(False)
    audio = SlowStateAffect(torch.zeros(10), torch.ones(10), global_dim=8,
                            hidden=16, local_dim=8, num_emotions=8, num_levels=4).eval()
    renderer_before = copy.deepcopy(system.renderer.state_dict())
    audio_before = copy.deepcopy(audio.state_dict())
    groups = audio_parameter_groups(system, audio, paper_data=True, freeze_renderer=True)
    optimizer = torch.optim.AdamW(groups)
    valid = torch.ones(2, 7, dtype=torch.bool)
    features = torch.randn(2, 7, 10)
    affect = audio_affect(audio, features, valid)
    baseline = {'b0': torch.zeros(2, 7, 52), 'h0': torch.randn(2, 7, 8)}
    identity = {'code': torch.randn(2, 8), 'baseline': torch.zeros(2, 52)}
    out = system.flow(torch.randn(2, 7, 52), torch.randn(2, 7, 8), valid,
                      identity, affect, noise=torch.randn(2, 7, 52),
                      time=torch.tensor([.2, .7]), base=baseline,
                      observation_mask=torch.ones(2, 52, dtype=torch.bool))
    loss = (out['prediction'] - out['velocity_target']).square().mean()
    loss.backward()
    assert all(p.grad is None for p in system.parameters())
    for head in (audio.global_head, audio.local_head):
        assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    optimizer.step()
    for k, value in renderer_before.items():
        torch.testing.assert_close(system.renderer.state_dict()[k], value, rtol=0, atol=0)
    assert any(not torch.equal(audio.state_dict()[k], v) for k, v in audio_before.items())


def test_freeze_only_changes_optimizer_membership_and_preserves_rng_mode_and_forward():
    torch.manual_seed(105)
    system = NeutralAffectSystem(config()).eval()
    audio = torch.nn.Linear(8, 8).eval()
    state = copy.deepcopy(system.state_dict())
    before_rng = torch.random.get_rng_state().clone()
    frozen = audio_parameter_groups(system, audio, paper_data=True, freeze_renderer=True)
    torch.testing.assert_close(before_rng, torch.random.get_rng_state(), rtol=0, atol=0)
    assert not system.renderer.training
    assert len(frozen) == 1 and frozen[0]['lr'] == 1e-4
    joint = audio_parameter_groups(system, audio, paper_data=True)
    assert [g['lr'] for g in joint] == [1e-4, 5e-5]
    assert set(map(id, joint[1]['params'])) == set(map(id, system.renderer.parameters()))
    for k, value in state.items():
        torch.testing.assert_close(system.state_dict()[k], value, rtol=0, atol=0)
    assert parser().parse_args(['--output', 'tmp']).freeze_renderer_audio is False
