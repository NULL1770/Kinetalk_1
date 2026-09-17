"""The audio-local adapter must preserve the frozen teacher's coordinates."""
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kinetalk_b0.models.aligned_audio_local import AlignedAudioLocal
from kinetalk_b0.models.neutral_affect import LowRateAffectEncoder, NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(812)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return {
        'data': {'content_dim': 8, 'motion_dim': 6, 'neutral_output_indices': [2, 3],
                 'emotion_classes': ['neutral', 'happy', 'sad'], 'num_intensity_levels': 3,
                 'audio_emotion_dim': 5},
        'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8, 'hidden_dim': 8,
                  'heads': 2, 'dropout': 0., 'dit_dim': 8, 'dit_depth': 1,
                  'residual_scale': .25, 'affect_stride': 4, 'affect_rank': 3}}


class FixedFrameLogits(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', value)

    def forward(self, hidden):
        assert hidden.shape[:2] == self.value.shape[:2]
        return self.value


def test_defaults_are_teacher_rank_stride_and_no_global_or_renderer_projection():
    model = AlignedAudioLocal(torch.zeros(1540), torch.ones(1540))
    assert model.rank == 8 and model.stride == 4
    assert model.input.weight.shape == (128, 1540)
    assert [block.conv.dilation[0] for block in model.blocks] == [1, 2, 4, 8]
    output = model(torch.randn(2, 11, 1540), torch.ones(2, 11, dtype=torch.bool))
    assert set(output) == {'controls', 'control_mask', 'control_weight'}
    assert output['controls'].shape == (2, 3, 8)
    assert output['controls'].count_nonzero() == 0
    assert not any('global' in key or 'projection' in key for key in model.state_dict())


def test_binning_tanh_partial_bins_and_weighted_centering_equal_teacher(cfg):
    teacher = LowRateAffectEncoder(cfg, 5, motion=True).eval()
    adapter = AlignedAudioLocal(torch.zeros(5), torch.ones(5), hidden=8, rank=3).eval()
    logits = torch.randn(2, 15, 3) * 2
    teacher.control_head = FixedFrameLogits(logits)
    adapter.control_head = FixedFrameLogits(logits)
    valid = torch.tensor([[True, False, True, True, False, False, False, False,
                           True, False, False, False, True, True, False],
                          [True] * 13 + [False, False]])
    features = torch.randn(2, 15, 5).masked_fill(~valid[..., None], float('nan'))
    expected, actual = teacher(features, valid), adapter(features, valid)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(actual['control_weight'][0], torch.tensor([3., 0., 1., 2.]))
    assert actual['controls'][~actual['control_mask']].count_nonzero() == 0
    torch.testing.assert_close((actual['controls'] * actual['control_weight'][..., None]).sum(1),
                               torch.zeros(2, 3), rtol=0, atol=1e-6)


def test_learned_adapter_padding_gaps_and_long_companion_do_not_change_local(cfg):
    system = NeutralAffectSystem(cfg).eval().requires_grad_(False)
    adapter = AlignedAudioLocal(torch.randn(5), torch.rand(5) + .2, hidden=8, rank=3).eval()
    with torch.no_grad():
        adapter.control_head.weight.normal_(std=.2)
        adapter.control_head.bias.normal_(std=.1)
    features = torch.randn(1, 13, 5)
    valid = torch.ones(1, 13, dtype=torch.bool)
    valid[:, 4:8] = False
    features[~valid] = float('nan')
    expected = system.project_affect(adapter(features, valid), valid)
    padded_features = F.pad(features, (0, 0, 0, 18), value=float('nan'))
    padded_valid = F.pad(valid, (0, 18), value=False)
    mixed_features = torch.cat((padded_features, torch.randn_like(padded_features)), 0)
    mixed_valid = torch.cat((padded_valid, torch.ones_like(padded_valid)), 0)
    actual = system.project_affect(adapter(mixed_features, mixed_valid), mixed_valid)
    for key in ('controls', 'control_mask', 'control_weight'):
        torch.testing.assert_close(actual[key][:1, :expected[key].shape[1]], expected[key],
                                   rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual['local'][:1, :13], expected['local'], rtol=2e-6, atol=2e-6)
    assert actual['local'][~mixed_valid].count_nonzero() == 0
    assert actual['controls'][~actual['control_mask']].count_nonzero() == 0
    mixed_features.requires_grad_()
    system.project_affect(adapter(mixed_features, mixed_valid), mixed_valid)['local'].square().sum().backward()
    assert torch.isfinite(mixed_features.grad).all()
    assert mixed_features.grad[~mixed_valid].count_nonzero() == 0
    assert mixed_features.grad[mixed_valid].abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())


def test_warm_start_copies_only_trunk_and_leaves_source_and_new_head_independent():
    mean, std = torch.randn(5), torch.rand(5) + .2
    slow = SlowStateAffect(mean, std, hidden=8, global_dim=6, local_dim=6).eval()
    source_before = deepcopy(slow.state_dict())
    adapter = AlignedAudioLocal(mean, std, hidden=8, rank=3)
    with torch.no_grad():
        adapter.control_head.weight.fill_(.9)
        adapter.control_head.bias.fill_(.9)
    assert adapter.initialize_from_slow(slow) is adapter
    for name in ('input', 'blocks'):
        for key, value in getattr(slow, name).state_dict().items():
            copied = getattr(adapter, name).state_dict()[key]
            torch.testing.assert_close(copied, value, rtol=0, atol=0)
            assert copied.data_ptr() != value.data_ptr()
    assert adapter.control_head.weight.count_nonzero() == 0
    assert adapter.control_head.bias.count_nonzero() == 0
    for key, value in source_before.items():
        torch.testing.assert_close(slow.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('incompatibility', ['statistics', 'hidden', 'dilation'])
def test_incompatible_warm_start_is_rejected_without_partial_mutation(incompatibility):
    adapter = AlignedAudioLocal(torch.zeros(5), torch.ones(5), hidden=8)
    slow = SlowStateAffect(torch.zeros(5), torch.ones(5), hidden=12 if incompatibility == 'hidden' else 8)
    if incompatibility == 'statistics':
        slow.feature_std[0] = 2.
    if incompatibility == 'dilation':
        slow.blocks[-1].conv.dilation = (7,)
    before = deepcopy(adapter.state_dict())
    with pytest.raises(ValueError, match='Warm start'):
        adapter.initialize_from_slow(slow)
    for key, value in before.items():
        torch.testing.assert_close(adapter.state_dict()[key], value, rtol=0, atol=0)


def test_adapter_training_through_frozen_projection_preserves_global_and_identity(cfg):
    system = NeutralAffectSystem(cfg).eval().requires_grad_(False)
    protected = deepcopy(system.state_dict())
    adapter = AlignedAudioLocal(torch.zeros(5), torch.ones(5), hidden=8, rank=3).eval()
    features, audio = torch.randn(2, 19, 5), torch.randn(2, 19, 5)
    valid = torch.ones(2, 19, dtype=torch.bool)
    valid[1, 15:] = False
    references = torch.randn(2, 2, 11, 6)
    frozen_global = system.encode_audio(audio, valid)
    identity_before = system.encode_identity(references)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    target = torch.randn(2, 19, 8)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        merged = {**frozen_global, **adapter(features, valid)}
        output = system.project_affect(merged, valid)
        for key in ('global', 'emotion_logits', 'intensity_logits', 'intensity_value'):
            assert output[key] is frozen_global[key]
        F.mse_loss(output['local'][valid], target[valid]).backward()
        assert adapter.control_head.weight.grad.abs().sum() > 0
        if step == 0:
            assert adapter.input.weight.grad.count_nonzero() == 0
        else:
            assert adapter.input.weight.grad.abs().sum() > 0
            assert adapter.blocks[0].conv.weight.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in system.parameters())
        optimizer.step()
    after = system.encode_audio(audio, valid)
    for key in ('global', 'emotion_logits', 'intensity_logits', 'intensity_value'):
        torch.testing.assert_close(after[key], frozen_global[key], rtol=0, atol=0)
    for key, value in system.encode_identity(references).items():
        torch.testing.assert_close(value, identity_before[key], rtol=0, atol=0)
    for key, value in protected.items():
        torch.testing.assert_close(system.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('problem', ['empty', 'shape', 'not_bool', 'observed_nan', 'integer'])
def test_invalid_observation_contract_is_rejected(problem):
    adapter = AlignedAudioLocal(torch.zeros(5), torch.ones(5), hidden=8)
    features, valid = torch.randn(2, 9, 5), torch.ones(2, 9, dtype=torch.bool)
    if problem == 'empty':
        valid[0] = False
    elif problem == 'shape':
        valid = valid[:, :-1]
    elif problem == 'not_bool':
        valid = valid.float()
    elif problem == 'observed_nan':
        features[0, 0, 0] = float('nan')
    elif problem == 'integer':
        features = features.long()
    with pytest.raises(ValueError):
        adapter(features, valid)
