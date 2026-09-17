"""Synthetic upper-audio adaptation and offline DC composition contracts."""
import copy

import pytest
import torch

from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts import train_audio_prefix_adaptation as runner


UPPER = list(runner.p.CC)
OTHER = list(runner.p.r.NOT_UPPER)
CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def audio_fixture(frames=29):
    torch.manual_seed(508)
    model = SlowStateAffect(torch.tensor([.1, -.2, .4, 1., -1.]),
                            torch.tensor([.3, 2., 1., .7, 3.]),
                            hidden=8, global_dim=3, local_dim=3,
                            num_emotions=4, num_levels=2, stride=4).eval()
    # Simulate the learned source head; a zero-initialized head would hide
    # numerical mismatches and prevent first-step trunk gradients.
    with torch.no_grad():
        model.local_head.weight.normal_(std=.1)
        model.local_head.bias.normal_(std=.05)
    valid = torch.ones(2, frames, dtype=torch.bool)
    valid[0, 3:5] = False
    valid[1, :2] = False
    valid[1, -4:] = False
    features = torch.randn(2, frames, 5)
    features[~valid] = float('nan')
    return model, features, valid


@pytest.mark.parametrize('trainable', [False, True])
def test_configure_local_changes_only_allowed_gradient_flags_and_never_weights(trainable):
    model, _, _ = audio_fixture()
    model.train().requires_grad_(True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = runner.configure_local(model, trainable)
    assert result is model and not model.training
    for name, parameter in model.named_parameters():
        allowed = name.startswith(('input.', 'blocks.', 'local_head.'))
        assert parameter.requires_grad is (trainable and allowed)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert not model.feature_mean.requires_grad and not model.feature_std.requires_grad
    runner.configure_local(model, False)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_specialized_native_forward_is_exactly_equal_and_never_executes_other_heads():
    model, features, valid = audio_fixture()
    expected = model(features, valid)['local']
    before = features.clone()

    def forbidden(module, args):
        raise AssertionError('Unused global/state head executed')

    hooks = [getattr(model, name).register_forward_pre_hook(forbidden) for name in
             ('global_head', 'emotion_classifier', 'intensity_classifier', 'state_head')]
    try:
        actual = runner.local_features(model, features, valid)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all() and actual[~valid].count_nonzero() == 0
    torch.testing.assert_close(features, before, equal_nan=True, atol=0, rtol=0)


def test_padding_payload_is_irrelevant_to_features_and_parameter_gradients():
    model, features, valid = audio_fixture()
    left = runner.configure_local(model, True)
    right = runner.configure_local(copy.deepcopy(model), True)
    clean = torch.where(valid[..., None], features, torch.full_like(features, 1e8)).requires_grad_()
    dirty = features.clone().requires_grad_()
    a = runner.local_features(left, dirty, valid)
    b = runner.local_features(right, clean, valid)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().sum().backward(); b.square().sum().backward()
    assert torch.isfinite(dirty.grad).all() and dirty.grad[~valid].count_nonzero() == 0
    assert dirty.grad[valid].abs().sum() > 0
    torch.testing.assert_close(dirty.grad, clean.grad, atol=0, rtol=0)
    for (name, lp), (_, rp) in zip(left.named_parameters(), right.named_parameters()):
        if lp.requires_grad:
            assert lp.grad is not None and torch.isfinite(lp.grad).all()
            torch.testing.assert_close(lp.grad, rp.grad, atol=0, rtol=0)
        else:
            assert lp.grad is None and rp.grad is None


@pytest.mark.parametrize('trainable', [False, True])
def test_upper_flow_backward_updates_only_enabled_local_path(trainable):
    model, features, valid = audio_fixture(frames=96)
    model = runner.configure_local(model, trainable)
    local_before = {key: value.clone() for key, value in model.state_dict().items()}
    native = runner.local_features(model, features, valid)
    b = {'valid': valid, 'motion': torch.randn(2, 96, 52), 'static_upper': torch.randn(2, 9),
         'h0': torch.randn(2, 96, 4), 'audio_global': torch.randn(2, 3),
         'audio_intensity': torch.randn(2, 1)}
    identity = {'code': torch.randn(2, 2)}
    upper = PrefixUpperFlow(CFG).eval()
    parameters = list(upper.parameters())+[parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    loss, _ = runner.context.context_batch(upper, b, identity, native, torch.ones(9),
                                           torch.Generator().manual_seed(73), 'chunk_teacher')
    assert torch.isfinite(loss)
    loss.backward()
    for name in ('input', 'blocks', 'local_head'):
        grads = [parameter.grad for parameter in getattr(model, name).parameters()]
        if trainable:
            assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
            assert sum(float(grad.abs().sum()) for grad in grads) > 0
        else:
            assert all(grad is None for grad in grads)
    for name, parameter in model.named_parameters():
        if not name.startswith(('input.', 'blocks.', 'local_head.')):
            assert parameter.grad is None
    optimizer.step()
    allowed_changes = []
    for name, value in model.state_dict().items():
        allowed = trainable and name.startswith(('input.', 'blocks.', 'local_head.'))
        if allowed:
            allowed_changes.append(not torch.equal(value, local_before[name]))
        else:
            torch.testing.assert_close(value, local_before[name], atol=0, rtol=0)
    assert any(allowed_changes) is trainable


@pytest.mark.parametrize('corruption', ['observed_nan', 'empty_sequence', 'float_mask'])
def test_native_feature_contract_rejects_bad_observations(corruption):
    model, features, valid = audio_fixture()
    if corruption == 'observed_nan':
        features[0, 0, 0] = float('nan')
    elif corruption == 'empty_sequence':
        valid[0].fill_(False)
    else:
        valid = valid.float()
    with pytest.raises(ValueError, match='Finite observed native'):
        runner.local_features(model, features, valid)


def dc_fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(149)
    baseline = torch.randn(2, 9, 52, generator=generator, dtype=dtype)
    raw = torch.randn(2, 9, 9, generator=generator, dtype=dtype)*3+5
    static = torch.randn(2, 9, generator=generator, dtype=dtype)
    valid = torch.tensor([[True, True, False, True, True, True, False, False, False],
                          [False, True, True, False, True, True, True, True, False]])
    baseline[~valid] = float('nan'); raw[~valid] = float('nan')
    return baseline, raw, static, valid


def test_dc_sets_deployable_static_mean_and_preserves_all_centered_motion_and_adjacent_steps():
    baseline, raw, static, valid = dc_fixture()
    result = runner.compose_dc(baseline, raw, static, valid)
    output_upper = result[..., UPPER]
    for row in range(2):
        movement = raw[row, valid[row]]
        output = output_upper[row, valid[row]]
        torch.testing.assert_close(output.mean(0), static[row], atol=2e-15, rtol=1e-13)
        torch.testing.assert_close(output-output.mean(0), movement-movement.mean(0), atol=2e-15, rtol=1e-13)
        expected = static[row]+movement-movement.mean(0)
        torch.testing.assert_close(output, expected, atol=2e-15, rtol=1e-13)
        torch.testing.assert_close(output.std(0, correction=0), movement.std(0, correction=0))
    pairs = valid[:, 1:] & valid[:, :-1]
    torch.testing.assert_close((output_upper[:, 1:]-output_upper[:, :-1])[pairs],
                                (raw[:, 1:]-raw[:, :-1])[pairs], atol=2e-15, rtol=1e-13)
    assert output_upper[valid].abs().max() > 1, 'Composition must not clip dynamics'


def test_dc_preserves_all_43_and_invalid_bits_without_mutating_inputs():
    baseline, raw, static, valid = dc_fixture(torch.float32)
    baseline[0, 2, 0] = -0.
    baseline[1, 0, 1] = float('inf')
    before = [value.clone() for value in (baseline, raw, static, valid)]
    result = runner.compose_dc(baseline, raw, static, valid)
    assert torch.equal(result[..., OTHER].contiguous().view(torch.int32), baseline[..., OTHER].contiguous().view(torch.int32))
    assert torch.equal(result[~valid].contiguous().view(torch.int32), baseline[~valid].contiguous().view(torch.int32))
    for previous, actual in zip(before, (baseline, raw, static, valid)):
        assert torch.equal(previous.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8))
    assert result.data_ptr() != baseline.data_ptr()


def test_dc_ignores_previous_upper_mean_and_raw_constant_offsets_but_is_offline_not_causal():
    baseline, raw, static, valid = dc_fixture()
    expected = runner.compose_dc(baseline, raw, static, valid)
    changed_base = baseline.clone()
    changed_base[..., UPPER] = 700.
    # Restore invalid entries because those intentionally retain the baseline.
    changed_base[~valid] = baseline[~valid]
    shifted = raw+torch.arange(9, dtype=raw.dtype)[None, None]
    result = runner.compose_dc(changed_base, shifted, static, valid)
    torch.testing.assert_close(result[valid], expected[valid], atol=3e-15, rtol=1e-13)
    future = raw.clone()
    future[0, 5] += 2.
    changed = runner.compose_dc(baseline, future, static, valid)
    # Changing only a late valid prediction shifts earlier predictions through
    # the whole-window mean, demonstrating the declared noncausal operation.
    shift = -2./int(valid[0].sum())
    torch.testing.assert_close(changed[0, 0, UPPER]-expected[0, 0, UPPER], torch.full((9,), shift, dtype=raw.dtype))


def test_dc_gradient_is_finite_and_flows_to_raw_motion_and_static_mean_with_nan_padding():
    baseline, raw, static, valid = dc_fixture()
    raw.requires_grad_(); static.requires_grad_()
    result = runner.compose_dc(baseline, raw, static, valid)
    weights = torch.arange(18, dtype=raw.dtype).reshape(2, 9)
    loss = (result[..., UPPER][valid].square()).sum()+(result[:, 1, UPPER]*weights).sum()
    loss.backward()
    assert torch.isfinite(raw.grad).all() and torch.isfinite(static.grad).all()
    assert raw.grad[~valid].count_nonzero() == 0
    assert raw.grad[valid].abs().sum() > 0 and static.grad.abs().sum() > 0


@pytest.mark.parametrize('static_change', ['shape', 'nan'])
def test_dc_rejects_invalid_deployable_static_mean(static_change):
    baseline, raw, static, valid = dc_fixture()
    if static_change == 'shape':
        static = static[:, :8]
    else:
        static[0, 0] = float('nan')
    with pytest.raises(ValueError, match='Finite deployable static'):
        runner.compose_dc(baseline, raw, static, valid)
