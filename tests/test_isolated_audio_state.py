import inspect

import pytest
import torch
from torch.nn import functional as F

from kinetalk_b0.models.isolated_audio_state import IsolatedAudioState
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, lift_slow_state


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(*, trained=True):
    torch.manual_seed(52)
    model = IsolatedAudioState(torch.randn(12), torch.rand(12) + .5,
                               torch.rand(52) + .1, torch.tensor([.2, .4, .7, 1.1]),
                               global_dim=5, identity_dim=7, hidden=16, stride=4)
    if trained:
        torch.nn.init.normal_(model.mean_branch[-1].weight, std=.1)
        torch.nn.init.normal_(model.state_branch.head.weight, std=.1)
    features = torch.randn(2, 19, 12)
    valid = torch.ones(2, 19, dtype=torch.bool)
    valid[0, [0, 7, 8, 18]] = False
    valid[1, 15:] = False
    features[~valid] = float('nan')
    args = (features, valid, torch.randn(2, 5), torch.randn(2, 7), torch.randn(2, 52))
    return model, args


def test_zero_initialization_and_no_target_or_label_api():
    model, args = example(trained=False)
    result = model(*args)
    torch.testing.assert_close(result['mean'], args[-1][:, list(UPPER_INDICES)], rtol=0, atol=0)
    assert result['state'].count_nonzero() == 0
    torch.testing.assert_close(result['upper'][args[1]], result['mean'][:, None].expand(-1, 19, -1)[args[1]])
    names = list(inspect.signature(model.forward).parameters)
    assert names == ['features', 'valid', 'global_code', 'identity_code', 'anchor', 'mode']
    with pytest.raises(TypeError):
        model(*args, query_target=torch.zeros(2, 19, 52))


def test_state_is_centered_and_lift_cannot_change_mean():
    model, args = example()
    valid = args[1]
    result = model(*args)
    assert result['state'].abs().max() > 0
    for name in ('state', 'upper', 'local'):
        assert result[name][~valid].count_nonzero() == 0
        assert torch.isfinite(result[name]).all()
    counts = valid.sum(1, keepdim=True)
    torch.testing.assert_close(result['state'].sum(1) / counts, torch.zeros(2, 4), atol=2e-7, rtol=0)
    torch.testing.assert_close(result['upper'].sum(1) / counts, result['mean'], atol=3e-7, rtol=0)
    delta = lift_slow_state(result['state'], model.channel_scales)[..., list(UPPER_INDICES)]
    torch.testing.assert_close(result['upper'][valid], (result['mean'][:, None] + delta)[valid])


def test_padding_and_invalid_payloads_cannot_change_valid_predictions():
    model, args = example()
    baseline = model(*args)
    features, valid, *conditions = args
    dirty = features.clone()
    dirty[~valid] = 1e30
    replacement = model(dirty, valid, *conditions)
    padded = model(F.pad(features, (0, 0, 0, 23), value=float('nan')),
                   F.pad(valid, (0, 23), value=False), *conditions)
    for name in ('mean', 'state', 'upper', 'local'):
        torch.testing.assert_close(replacement[name], baseline[name], rtol=0, atol=0)
        value = padded[name] if name == 'mean' else padded[name][:, :19]
        torch.testing.assert_close(value, baseline[name], rtol=0, atol=0)
        if name != 'mean':
            assert padded[name][:, 19:].count_nonzero() == 0


def test_static_and_reverse_are_postencoding_interventions_with_fixed_mean():
    model, args = example()
    audio, static, reverse = [model(*args, mode=mode) for mode in ('audio', 'static', 'reverse')]
    valid = args[1]
    torch.testing.assert_close(static['mean'], audio['mean'], rtol=0, atol=0)
    torch.testing.assert_close(reverse['mean'], audio['mean'], rtol=0, atol=0)
    assert static['state'].count_nonzero() == 0
    for batch in range(len(valid)):
        positions = valid[batch].nonzero().flatten()
        torch.testing.assert_close(static['upper'][batch, positions],
                                   audio['mean'][batch].expand(len(positions), -1), rtol=0, atol=0)
        torch.testing.assert_close(static['local'][batch, positions],
                                   static['local'][batch, positions[0]].expand(len(positions), -1), rtol=0, atol=0)
        for name in ('state', 'local'):
            torch.testing.assert_close(reverse[name][batch, positions], audio[name][batch, positions.flip(0)],
                                       rtol=0, atol=0)


@pytest.mark.parametrize('objective', ['mean', 'state'])
def test_branch_gradients_and_frozen_external_conditions_are_isolated(objective):
    model, args = example()
    mean_parameters = list(model.mean_branch.parameters())
    state_parameters = list(model.state_branch.parameters())
    assert not {id(value) for value in mean_parameters} & {id(value) for value in state_parameters}
    assert {id(value) for value in model.parameters()} == {id(value) for value in mean_parameters + state_parameters}
    features, valid, *conditions = args
    features.requires_grad_()
    for value in conditions:
        value.requires_grad_()
    output = model(features, valid, *conditions)
    output[objective].square().mean().backward()
    active, frozen = ((mean_parameters, state_parameters) if objective == 'mean'
                      else (state_parameters, mean_parameters))
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in active)
    assert all(value.grad is None for value in frozen)
    assert all(value.grad is None for value in conditions)
    assert features.grad[~valid].count_nonzero() == 0
    assert torch.isfinite(features.grad).all()


def test_reload_retains_train_statistics_and_architecture():
    model, args = example()
    state = model.state_dict()
    copy = IsolatedAudioState(state['feature_mean'], state['feature_std'], state['channel_scales'],
                              state['state_dynamic_scales'], **model.export_config())
    copy.load_state_dict(state)
    for name, expected in model(*args).items():
        torch.testing.assert_close(copy(*args)[name], expected, rtol=0, atol=0)
    assert all(not value.requires_grad for value in model.buffers())


def test_input_contract_rejects_observed_nan_empty_samples_and_invalid_statistics():
    model, args = example()
    features, valid, global_code, identity_code, anchor = args
    bad = features.clone()
    bad[0, 1, 0] = float('nan')
    with pytest.raises(ValueError, match='Observed features'):
        model(bad, valid, global_code, identity_code, anchor)
    bad_mask = valid.clone()
    bad_mask[0] = False
    with pytest.raises(ValueError, match='observations'):
        model(features, bad_mask, global_code, identity_code, anchor)
    with pytest.raises(ValueError, match='mode'):
        model(*args, mode='oracle')
    bad_anchor = anchor.clone()
    bad_anchor[0, 41] = float('inf')
    with pytest.raises(ValueError, match='anchor'):
        model(features, valid, global_code, identity_code, bad_anchor)
    # Unused non-upper channels are outside this predictor's support.
    unused_anchor = anchor.clone()
    unused_anchor[:, 51] = float('nan')
    torch.testing.assert_close(model(features, valid, global_code, identity_code, unused_anchor)['upper'],
                               model(*args)['upper'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='state_dynamic_scales'):
        IsolatedAudioState(torch.zeros(12), torch.ones(12), torch.ones(52), torch.zeros(4))
