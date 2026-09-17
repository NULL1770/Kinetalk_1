from unittest.mock import patch

import pytest
import torch

from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from kinetalk_b0.models.history_upper_flow import HistoryUpperFlow
from kinetalk_b0.models.temporal_upper import TemporalUpperFlow


CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 8, 'dit_depth': 2, 'heads': 2, 'dropout': 0.}}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(142)
    model = PrefixUpperFlow(CFG)
    valid = torch.ones(2, 24, dtype=torch.bool); valid[0, [0, 2, 22, 23]] = False
    known_mask = torch.zeros_like(valid); known_mask[:, :8] = valid[:, :8]
    h0 = torch.randn(2, 24, 4)
    identity = torch.randn(2, 2)
    affect = {'global': torch.randn(2, 3), 'intensity_value': torch.randn(2, 1)}
    local = torch.randn(2, 24, 3)
    known = torch.randn(2, 24, 9)
    noise = torch.randn_like(known)
    return model, valid, h0, identity, affect, local, noise, known, known_mask


def test_zero_mask_embedding_and_empty_prefix_match_warmstarted_backbone_exactly():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    base = TemporalUpperFlow(CFG, use_state=False)
    base.renderer.load_state_dict(model.renderer.state_dict())
    assert model.known_embedding.weight.count_nonzero() == 0
    mask[:] = False; known[:] = float('nan')
    expected = base.decode(valid, h0, identity, affect, local, None, noise, steps=3)
    actual = model.decode_prefix(valid, h0, identity, affect, local, noise, known=known, known_mask=mask, steps=3)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    target = torch.randn_like(noise); time = torch.tensor([.2, .7])
    parent_loss = base.flow_loss(target, valid, h0, identity, affect, local, None, noise, time)
    actual_loss = model.flow_loss_prefix(target, valid, h0, identity, affect, local, noise, time, known=known, known_mask=mask)
    torch.testing.assert_close(actual_loss, parent_loss, atol=0, rtol=0)


def test_every_solver_input_and_final_output_clamp_known_values_exactly():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    noise[mask | ~valid] = float('nan')
    def velocity(x, time, conditions):
        torch.testing.assert_close(x[mask], known[mask], atol=0, rtol=0)
        assert x[~valid].count_nonzero() == 0
        return torch.full_like(x, 100.)
    with patch.object(model, 'velocity', side_effect=velocity) as call:
        output = model.decode_prefix(valid, h0, identity, affect, local, noise, known=known, known_mask=mask, steps=4)
    assert call.call_count == 4
    torch.testing.assert_close(output[mask], known[mask], atol=0, rtol=0)
    torch.testing.assert_close(output[valid & ~mask], noise[valid & ~mask] + 100.)
    assert output[~valid].count_nonzero() == 0


def test_nonknown_payload_is_never_read_and_target_prefix_does_not_condition_training():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    reference = model.decode_prefix(valid, h0, identity, affect, local, noise, known=known, known_mask=mask, steps=2)
    known[~mask] = float('nan')
    output = model.decode_prefix(valid, h0, identity, affect, local, noise, known=known, known_mask=mask, steps=2)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    targets = [torch.randn_like(noise), torch.randn_like(noise) * 100]
    captured = []
    for target in targets:
        target[mask | ~valid] = float('nan')
        with patch.object(model, 'velocity', wraps=model.velocity) as call:
            model.flow_loss_prefix(target, valid, h0, identity, affect, local, noise, torch.zeros(2), known=known, known_mask=mask)
        captured.append(call.call_args.args)
    torch.testing.assert_close(captured[0][0], captured[1][0], atol=0, rtol=0)
    torch.testing.assert_close(captured[0][0][mask], known[mask], atol=0, rtol=0)
    for key in captured[0][2]:
        if torch.is_tensor(captured[0][2][key]):
            torch.testing.assert_close(captured[0][2][key], captured[1][2][key], atol=0, rtol=0)


def test_flow_loss_excludes_prefix_and_invalid_frames_from_loss_and_denominator():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    target = torch.ones_like(noise) * 3
    noise[:] = 1
    target[mask | ~valid] = noise[mask | ~valid] = float('nan')
    def velocity(x, time, conditions):
        torch.testing.assert_close(x[mask], known[mask], atol=0, rtol=0)
        return torch.where(mask[..., None], torch.full_like(x, 100000), torch.zeros_like(x))
    with patch.object(model, 'velocity', side_effect=velocity):
        loss = model.flow_loss_prefix(target, valid, h0, identity, affect, local, noise,
                                      torch.tensor([.3, .8]), known=known, known_mask=mask)
    assert loss.item() == 4.


def test_prefix_values_and_order_affect_unknown_outputs_through_shared_attention():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    expected = model.decode_prefix(valid, h0, identity, affect, local, noise, known=known, known_mask=mask, steps=2)
    changed = known.clone(); changed[mask] += 3
    varied = model.decode_prefix(valid, h0, identity, affect, local, noise, known=changed, known_mask=mask, steps=2)
    reverse = known.clone()
    for row in range(len(mask)):
        indices = mask[row].nonzero(as_tuple=True)[0]
        reverse[row, indices] = reverse[row, indices.flip(0)]
    reversed_output = model.decode_prefix(valid, h0, identity, affect, local, noise, known=reverse, known_mask=mask, steps=2)
    unknown = valid & ~mask
    assert not torch.allclose(expected[unknown], varied[unknown], atol=1e-8, rtol=1e-7)
    assert not torch.allclose(expected[unknown], reversed_output[unknown], atol=1e-8, rtol=1e-7)


def test_nan_padding_and_known_gradients_are_finite_and_current_gt_remains_target_only():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    unknown = valid & ~mask
    known = known.masked_fill(~mask[..., None], float('nan')).requires_grad_()
    noise = noise.masked_fill(~unknown[..., None], float('nan')).requires_grad_()
    target = torch.randn_like(noise).masked_fill(~unknown[..., None], float('nan')).requires_grad_()
    h0 = h0.masked_fill(~valid[..., None], float('nan')).requires_grad_()
    local = local.masked_fill(~valid[..., None], float('nan')).requires_grad_()
    loss = model.flow_loss_prefix(target, valid, h0, identity, affect, local, noise,
                                  torch.tensor([.3, .8]), known=known, known_mask=mask)
    loss.backward()
    for value, used in [(known, mask), (noise, unknown), (target, unknown), (h0, valid), (local, valid)]:
        assert torch.isfinite(value.grad).all()
        assert value.grad[~used].count_nonzero() == 0
        assert value.grad[used].abs().sum() > 0
    assert model.known_embedding.weight.grad is not None
    assert torch.isfinite(model.known_embedding.weight.grad).all() and model.known_embedding.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_appended_invalid_padding_preserves_native_positions_and_prefix():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    expected = model.decode_prefix(valid, h0, identity, affect, local, noise, known=known, known_mask=mask, steps=2)
    def padded(x):
        return torch.cat([x, torch.full((len(x), 5, x.shape[-1]), float('nan'))], 1)
    padded_valid = torch.cat([valid, torch.zeros(2, 5, dtype=torch.bool)], 1)
    padded_mask = torch.cat([mask, torch.zeros(2, 5, dtype=torch.bool)], 1)
    actual = model.decode_prefix(padded_valid, padded(h0), identity, affect, padded(local), padded(noise),
                                 known=padded(known), known_mask=padded_mask, steps=2)
    torch.testing.assert_close(actual[:, :24], expected, atol=1e-6, rtol=1e-5)
    assert actual[:, 24:].count_nonzero() == 0


def test_conditions_are_fresh_and_source_values_are_not_mutated():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    local.requires_grad_()
    with torch.no_grad(): model.known_embedding.weight[1].fill_(2.)
    original_local = local.detach().clone(); original_global = affect['global'].clone()
    conditions, _, _ = model.prepare_prefix_conditions(valid, h0, identity, affect, local,
                                                       known=known, known_mask=mask)
    torch.testing.assert_close(local, original_local, atol=0, rtol=0)
    assert conditions['global'].data_ptr() != affect['global'].data_ptr()
    conditions['global'].zero_()
    torch.testing.assert_close(affect['global'], original_global, atol=0, rtol=0)
    conditions['local'].sum().backward()
    assert local.grad[valid].abs().sum() > 0


def test_fixed_eight_left_padding_keeps_twenty_four_positions_with_empty_first_prefix():
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    valid[:, :8] = False; mask[:] = False
    for value in (h0, local, noise, known):
        value[:, :8] = float('nan')
    base = TemporalUpperFlow(CFG, use_state=False)
    base.renderer.load_state_dict(model.renderer.state_dict())
    expected = base.decode(valid, h0, identity, affect, local, None, noise, steps=2)
    with patch.object(model, 'velocity', wraps=model.velocity) as call:
        output = model.decode_prefix(valid, h0, identity, affect, local, noise,
                                      known=known, known_mask=mask, steps=2)
    assert all(args.args[0].shape == (2, 24, 9) for args in call.call_args_list)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert output[:, :8].count_nonzero() == 0
    # Empty prefix uses current positions 8..23 in both arms. Removing those
    # padded tokens would instead change the inherited positional coordinates.


def test_history_backbone_warmstart_and_upper_only_updates_leave_local_frozen():
    torch.manual_seed(197)
    source = HistoryUpperFlow(CFG, use_history=False, history_hidden=8)
    warmstart = {key: value for key, value in source.state_dict().items() if not key.startswith('history_')}
    models = []
    for _ in range(2):
        torch.manual_seed(198)
        model = PrefixUpperFlow(CFG)
        loaded = model.load_state_dict(warmstart, strict=False)
        assert loaded.missing_keys == ['known_embedding.weight'] and loaded.unexpected_keys == []
        models.append(model)
    assert all(torch.equal(value, models[1].state_dict()[key]) for key, value in models[0].state_dict().items())
    model, valid, h0, identity, affect, _, noise, known, mask = fixture()
    model.load_state_dict(models[0].state_dict())
    frozen_local = torch.nn.Linear(7, 3).eval().requires_grad_(False)
    frozen_before = {key: value.clone() for key, value in frozen_local.state_dict().items()}
    local = frozen_local(torch.randn(2, 24, 7))
    before_upper = model.renderer.output.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.flow_loss_prefix(torch.randn_like(noise), valid, h0, identity, affect, local, noise,
                                  torch.tensor([.2, .7]), known=known, known_mask=mask)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    assert not torch.equal(before_upper, model.renderer.output.weight)
    assert all(parameter.grad is None for parameter in frozen_local.parameters())
    assert all(torch.equal(value, frozen_before[key]) for key, value in frozen_local.state_dict().items())


@pytest.mark.parametrize('case', ['mask_dtype', 'mask_shape', 'known_invalid', 'no_unknown',
                                  'known_shape', 'known_dtype', 'known_nan', 'noise_nan', 'target_nan'])
def test_invalid_contracts_are_rejected(case):
    model, valid, h0, identity, affect, local, noise, known, mask = fixture()
    target = torch.randn_like(noise)
    if case == 'mask_dtype': mask = mask.float()
    elif case == 'mask_shape': mask = mask[:, :-1]
    elif case == 'known_invalid': mask[0, 0] = True
    elif case == 'no_unknown': mask[1] = valid[1]
    elif case == 'known_shape': known = known[..., :-1]
    elif case == 'known_dtype': known = known.double()
    elif case == 'known_nan': known[1, 0, 0] = float('nan')
    elif case == 'noise_nan': noise[1, 10, 0] = float('nan')
    elif case == 'target_nan': target[1, 10, 0] = float('nan')
    with pytest.raises(ValueError):
        model.flow_loss_prefix(target, valid, h0, identity, affect, local, noise,
                                torch.tensor([.3, .8]), known=known, known_mask=mask)
