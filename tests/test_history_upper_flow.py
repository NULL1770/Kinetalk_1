from unittest.mock import patch

import pytest
import torch

from kinetalk_b0.models.history_upper_flow import HistoryUpperFlow
from kinetalk_b0.models.temporal_upper import TemporalUpperFlow


CFG = {'model': {'content_dim': 8, 'emotion_dim': 6, 'style_dim': 5,
                 'dit_dim': 12, 'dit_depth': 2, 'heads': 3, 'dropout': 0.}}


def fixture(*, use_history=True, frames=16):
    torch.manual_seed(82)
    model = HistoryUpperFlow(CFG, use_history=use_history, history_hidden=12)
    valid = torch.ones(2, frames, dtype=torch.bool); valid[0, -3:] = False
    content = torch.randn(2, frames, 8)
    identity = torch.randn(2, 5)
    affect = {'global': torch.randn(2, 6), 'intensity_value': torch.randn(2, 1)}
    local = torch.randn(2, frames, 6)
    noise = torch.randn(2, frames, 9)
    history = torch.randn(2, 8, 9)
    history_valid = torch.ones(2, 8, dtype=torch.bool); history_valid[0, [0, 1, 4]] = False
    return model, valid, content, identity, affect, local, noise, history, history_valid


def test_matched_initialization_and_no_history_ignores_every_supplied_history_value():
    model, valid, content, identity, affect, local, noise, history, mask = fixture(use_history=False)
    enabled, *_ = fixture()
    assert model.state_dict().keys() == enabled.state_dict().keys()
    assert all(torch.equal(v, enabled.state_dict()[k]) for k, v in model.state_dict().items())
    expected = model.decode_history(valid, content, identity, affect, local, noise, steps=2)
    for supplied, supplied_mask in [(history, mask), (torch.full_like(history, float('nan')), None),
                                     ('not a tensor', object())]:
        actual = model.decode_history(valid, content, identity, affect, local, noise,
                                      history=supplied, history_valid=supplied_mask, steps=2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    target = torch.randn_like(noise)
    time = torch.tensor([.2, .6])
    plain_loss = model.flow_loss_history(target, valid, content, identity, affect, local, noise, time)
    ignored_loss = model.flow_loss_history(target, valid, content, identity, affect, local, noise, time,
                                           history=object(), history_valid='ignored')
    torch.testing.assert_close(ignored_loss, plain_loss, atol=0, rtol=0)
    expected[valid].square().sum().backward()
    for name, parameter in model.named_parameters():
        if name.startswith('history_'):
            assert parameter.grad is None


@pytest.mark.parametrize('frames', [12, 16])
def test_empty_history_is_legal_and_equals_unconditioned_temporal_flow(frames):
    model, valid, content, identity, affect, local, noise, history, mask = fixture(frames=frames)
    parent = TemporalUpperFlow(CFG, use_state=False)
    parent.renderer.load_state_dict(model.renderer.state_dict())
    expected = parent.decode(valid, content, identity, affect, local, None, noise, steps=2)
    for value, history_mask in [(None, None), (history[:, :0], mask[:, :0]),
                                (torch.full_like(history, float('nan')), torch.zeros_like(mask))]:
        output = model.decode_history(valid, content, identity, affect, local, noise,
                                      history=value, history_valid=history_mask, steps=2)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_prefix_values_and_chronological_order_change_output_and_receive_gradients():
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    mask[:] = True
    history.requires_grad_(); local.requires_grad_()
    output = model.decode_history(valid, content, identity, affect, local, noise,
                                  history=history, history_valid=mask, steps=2)
    changed = model.decode_history(valid, content, identity, affect, local, noise,
                                   history=history + 2., history_valid=mask, steps=2)
    reversed_output = model.decode_history(valid, content, identity, affect, local, noise,
                                           history=history.flip(1), history_valid=mask, steps=2)
    assert not torch.allclose(output, changed)
    assert not torch.allclose(output, reversed_output)
    output[valid].square().sum().backward()
    assert torch.isfinite(history.grad).all() and (history.grad.abs().sum((0, 2)) > 0).all()
    assert not torch.allclose(history.grad[:, 0], history.grad[:, -1])
    assert local.grad[valid].abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name.startswith('history_'):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


def test_masked_nan_history_and_empty_batch_row_are_finite_with_zero_padding_gradients():
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    mask[1] = False
    expected = model.decode_history(valid, content, identity, affect, local, noise,
                                    history=history, history_valid=mask, steps=2)
    history[~mask] = float('nan'); history.requires_grad_()
    actual = model.decode_history(valid, content, identity, affect, local, noise,
                                  history=history, history_valid=mask, steps=2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all() and actual[~valid].count_nonzero() == 0
    actual[valid].square().sum().backward()
    assert torch.isfinite(history.grad).all() and history.grad[~mask].count_nonzero() == 0
    empty = model.decode_history(valid, content, identity, affect, local, noise, steps=2)
    torch.testing.assert_close(actual[1], empty[1], atol=0, rtol=0)


def test_left_padding_does_not_change_right_aligned_history_but_missing_native_gap_does():
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    history = history[:, -4:]; mask = torch.ones(2, 4, dtype=torch.bool)
    compact = model.prepare_history_conditions(valid, content, identity, affect, local,
                                               history=history, history_valid=mask)['local']
    padded_history = torch.cat((torch.full_like(history, float('nan')), history), 1)
    padded_mask = torch.cat((torch.zeros_like(mask), mask), 1)
    padded = model.prepare_history_conditions(valid, content, identity, affect, local,
                                              history=padded_history, history_valid=padded_mask)['local']
    torch.testing.assert_close(compact, padded, atol=0, rtol=0)
    # The same four observations one frame older are a different native clock,
    # even though masked GRU updates still see exactly four observed values.
    older_history = torch.cat((history, torch.full_like(history[:, :1], float('nan'))), 1)
    older_mask = torch.cat((mask, torch.zeros_like(mask[:, :1])), 1)
    older = model.prepare_history_conditions(valid, content, identity, affect, local,
                                             history=older_history, history_valid=older_mask)['local']
    assert not torch.allclose(compact, older, atol=1e-8, rtol=1e-7)


def test_training_and_one_step_decode_use_same_conditions_without_target_history_leak():
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    conditions = model.prepare_history_conditions(valid, content, identity, affect, local,
                                                  history=history, history_valid=mask)
    velocity = model.velocity(noise, torch.zeros(2), conditions)
    output = model.decode_history(valid, content, identity, affect, local, noise,
                                  history=history, history_valid=mask, steps=1)
    torch.testing.assert_close(output, torch.where(valid[..., None], noise + velocity, 0.), atol=0, rtol=0)
    for target in (torch.randn_like(noise), torch.randn_like(noise) * 100):
        with patch.object(model, 'velocity', wraps=model.velocity) as call:
            loss = model.flow_loss_history(target, valid, content, identity, affect, local, noise,
                                           torch.zeros(2), history=history, history_valid=mask)
        noisy_input, _, used_conditions = call.call_args.args
        torch.testing.assert_close(noisy_input[valid], noise[valid], atol=0, rtol=0)
        torch.testing.assert_close(used_conditions['local'], conditions['local'], atol=0, rtol=0)
        torch.testing.assert_close(loss, (velocity[valid] - (target - noise)[valid]).square().mean())


def test_generated_prefix_can_drive_next_chunk_without_targets_or_implicit_detach():
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    valid[:] = True
    first = model.decode_history(valid, content, identity, affect, local, noise, steps=1)
    first.retain_grad()
    second = model.decode_history(valid, content, identity, affect, local, noise + 1,
                                   history=first[:, -8:], history_valid=valid[:, -8:], steps=1)
    second.square().mean().backward()
    assert first.grad is not None and first.grad[:, -8:].abs().sum() > 0
    assert first.grad[:, :-8].count_nonzero() == 0
    # A scheduled-sampling runner owns the stop-gradient operation explicitly.
    detached = model.prepare_history_conditions(valid, content, identity, affect, local,
                                                 history=first[:, -8:].detach(), history_valid=valid[:, -8:])
    assert torch.isfinite(detached['local']).all()


def test_current_chunk_nan_padding_does_not_poison_training_or_history_gradients():
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    target = torch.randn_like(noise)
    values = []
    for value in (content, local, noise, target):
        values.append(value.masked_fill(~valid[..., None], float('nan')).requires_grad_())
    history = history.masked_fill(~mask[..., None], float('nan')).requires_grad_()
    loss = model.flow_loss_history(values[3], valid, values[0], identity, affect, values[1], values[2],
                                   torch.tensor([.2, .6]), history=history, history_valid=mask)
    loss.backward()
    assert torch.isfinite(loss)
    for value in values:
        assert torch.isfinite(value.grad).all() and value.grad[~valid].count_nonzero() == 0
    assert torch.isfinite(history.grad).all() and history.grad[~mask].count_nonzero() == 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_chunk_output_is_not_centered_clamped_or_forced_back_to_zero():
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    with torch.no_grad():
        model.renderer.output.weight.zero_()
        model.renderer.output.bias.copy_(torch.arange(1., 10.))
    output = model.decode_history(valid, content, identity, affect, local, torch.zeros_like(noise),
                                  history=history, history_valid=mask, steps=2)
    torch.testing.assert_close(output[valid], torch.arange(1., 10.).expand(int(valid.sum()), -1), atol=0, rtol=0)


@pytest.mark.parametrize('case', ['too_long', 'missing_mask', 'missing_history', 'wrong_batch',
                                  'wrong_width', 'wrong_dtype', 'mask_dtype', 'mask_shape', 'nan', 'inf'])
def test_rejects_invalid_observed_history_contract(case):
    model, valid, content, identity, affect, local, noise, history, mask = fixture()
    if case == 'too_long': history = torch.cat((history, history[:, :1]), 1); mask = torch.ones(2, 9, dtype=torch.bool)
    elif case == 'missing_mask': mask = None
    elif case == 'missing_history': history = None
    elif case == 'wrong_batch': history = history[:1]; mask = mask[:1]
    elif case == 'wrong_width': history = history[..., :-1]
    elif case == 'wrong_dtype': history = history.double()
    elif case == 'mask_dtype': mask = mask.float()
    elif case == 'mask_shape': mask = mask[:, :-1]
    elif case == 'nan': history[1, 0, 0] = float('nan')
    elif case == 'inf': history[1, 0, 0] = float('inf')
    with pytest.raises(ValueError, match='history'):
        model.decode_history(valid, content, identity, affect, local, noise,
                              history=history, history_valid=mask, steps=1)


@pytest.mark.parametrize('kwargs', [{'history_frames': 9}, {'history_frames': 0},
                                    {'history_frames': True}, {'use_history': 1}, {'history_hidden': 0}])
def test_rejects_invalid_constructor_contract(kwargs):
    with pytest.raises(ValueError):
        HistoryUpperFlow(CFG, **kwargs)


def test_custom_history_limit_rejects_longer_prefix_even_below_eight():
    _, valid, content, identity, affect, local, noise, history, mask = fixture()
    model = HistoryUpperFlow(CFG, history_frames=4, history_hidden=12)
    with pytest.raises(ValueError, match='history'):
        model.decode_history(valid, content, identity, affect, local, noise,
                              history=history[:, -5:], history_valid=mask[:, -5:], steps=1)
