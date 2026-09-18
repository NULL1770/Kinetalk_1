"""Bounded-prior correction and audio-only native-bin contracts."""
import numpy as np
import pytest
import torch

from kinetalk_b0.models.joint_token_audio import BoundedTokenResidual, unit_audio_descriptor


def test_zero_initial_logits_preserve_arbitrary_base_prior_exactly():
    model = BoundedTokenResidual(393)
    x = torch.randn(4, 393)
    residual = model(x)
    prior = torch.randn(4, 128)
    assert residual.shape == prior.shape
    assert residual.count_nonzero() == 0
    assert torch.equal(prior+residual, prior)
    assert torch.equal((prior+residual).softmax(-1), prior.softmax(-1))
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in model.modules())


def test_logits_remain_bounded_after_nonzero_large_weight_updates():
    model = BoundedTokenResidual(9, hidden=12, k=7)
    with torch.no_grad():
        model.output.weight.normal_(std=30)
        model.output.bias.normal_(std=20)
    output = model(torch.randn(2, 3, 9)*100)
    assert output.shape == (2, 3, 7)
    assert torch.isfinite(output).all()
    assert output.abs().max() <= 1
    assert output.abs().max() > .9


def test_output_learns_first_step_and_hidden_learns_after_output_moves():
    torch.manual_seed(852)
    model = BoundedTokenResidual(5, hidden=8, k=3)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    features = torch.randn(6, 5)
    target = torch.tensor([0, 1, 2, 1, 0, 2])
    loss = torch.nn.functional.cross_entropy(model(features), target)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.input.weight.grad.count_nonzero() == 0
    opt.step()
    opt.zero_grad()
    torch.nn.functional.cross_entropy(model(features), target).backward()
    assert model.input.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('shape', [(393,), (3, 393), (2, 4, 393)])
def test_token_correction_accepts_vector_or_batched_last_dimension(shape):
    model = BoundedTokenResidual(393, hidden=4, k=8)
    assert model(torch.zeros(shape)).shape == (*shape[:-1], 8)


@pytest.mark.parametrize('kwargs', [dict(input_dim=0), dict(input_dim=3, hidden=-1),
                                   dict(input_dim=3, k=True), dict(input_dim=3.5)])
def test_bad_model_dimensions_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundedTokenResidual(**kwargs)


@pytest.mark.parametrize('x', [torch.ones(4, 4), torch.tensor([[float('nan')]*3]),
                               torch.zeros(0, 3), torch.ones(2, 3, dtype=torch.long)])
def test_bad_model_inputs_rejected(x):
    with pytest.raises(ValueError):
        BoundedTokenResidual(3)(x)


def audio_fixture(length=40):
    frame = np.arange(length, dtype=np.float32)[:, None]
    channel = np.arange(64, dtype=np.float32)[None]*.001
    return frame+channel, np.ones(length, dtype=bool)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_four_bins_use_native_fixed_boundaries_and_preserve_order(backend):
    local, valid = audio_fixture()
    if backend == 'torch':
        local, valid = torch.from_numpy(local), torch.from_numpy(valid)
    descriptor = unit_audio_descriptor(local, valid, 4)
    array = descriptor.numpy() if backend == 'torch' else descriptor
    expected = np.concatenate([np.arange(left, left+8).mean()+np.arange(64)*.001
                               for left in (4, 12, 20, 28)])
    assert array.shape == (256,)
    np.testing.assert_allclose(array, expected, rtol=1e-6, atol=1e-6)
    assert array.dtype == np.float32


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_internal_invalid_nan_bins_zero_without_time_compression(backend):
    local, valid = audio_fixture(32)
    valid[8:16] = False
    valid[18:21] = False
    local[~valid] = np.nan
    if backend == 'torch':
        local, valid = torch.from_numpy(local), torch.from_numpy(valid)
    descriptor = unit_audio_descriptor(local, valid, 0)
    array = descriptor.numpy() if backend == 'torch' else descriptor
    assert np.isfinite(array).all()
    assert (array[64:128] == 0).all()
    expected = np.mean([16, 17, 21, 22, 23])+np.arange(64)*.001
    np.testing.assert_allclose(array[128:192], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(array[192:], 27.5+np.arange(64)*.001, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_short_clip_keeps_fixed_clock_and_beyond_tail_bins_zero(backend):
    local, valid = audio_fixture(10)
    if backend == 'torch':
        local, valid = torch.from_numpy(local), torch.from_numpy(valid)
    descriptor = unit_audio_descriptor(local, valid, 0)
    array = descriptor.numpy() if backend == 'torch' else descriptor
    np.testing.assert_allclose(array[:64], 3.5+np.arange(64)*.001, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(array[64:128], 8.5+np.arange(64)*.001, rtol=1e-6, atol=1e-6)
    assert (array[128:] == 0).all()
    after_clip = unit_audio_descriptor(local, valid, 100)
    if backend == 'torch':
        assert after_clip.count_nonzero() == 0
    else:
        assert np.count_nonzero(after_clip) == 0


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_all_invalid_poison_returns_zero_audio_without_auxiliary_inputs(backend):
    local = np.full((32, 64), np.nan, dtype=np.float64)
    valid = np.zeros(32, dtype=bool)
    if backend == 'torch':
        local, valid = torch.from_numpy(local), torch.from_numpy(valid)
    descriptor = unit_audio_descriptor(local, valid, 0)
    if backend == 'torch':
        assert torch.equal(descriptor, torch.zeros(256, dtype=torch.float64))
    else:
        assert np.array_equal(descriptor, np.zeros(256, dtype=np.float64))


def test_nonmultiple_duration_uses_floor_boundaries_not_resampling():
    local, valid = audio_fixture(30)
    actual = unit_audio_descriptor(local, valid, np.int64(3), duration=np.int64(11))
    expected = np.concatenate([local[a:b].mean(0) for a, b in ((3, 5), (5, 8), (8, 11), (11, 14))])
    np.testing.assert_array_equal(actual, expected)


def test_torch_descriptor_gradients_only_touch_valid_in_horizon_audio():
    array, mask = audio_fixture(48)
    mask[10:12] = False
    array[~mask] = np.nan
    local = torch.tensor(array, requires_grad=True)
    valid = torch.tensor(mask)
    unit_audio_descriptor(local, valid, 8).sum().backward()
    assert torch.isfinite(local.grad).all()
    assert local.grad[~valid].count_nonzero() == 0
    assert local.grad[:8].count_nonzero() == 0
    assert local.grad[40:].count_nonzero() == 0
    assert (local.grad[12:16] > 0).all()


@pytest.mark.parametrize('kind', ['negative_start', 'bool_start', 'short_duration',
                                  'shape', 'mask_dtype', 'nan_observed', 'mixed_types'])
def test_bad_descriptor_contract_rejected(kind):
    local, valid = audio_fixture()
    start, duration = 0, 32
    if kind == 'negative_start':
        start = -1
    elif kind == 'bool_start':
        start = True
    elif kind == 'short_duration':
        duration = 3
    elif kind == 'shape':
        local = local[:, :63]
    elif kind == 'mask_dtype':
        valid = valid.astype(np.float32)
    elif kind == 'nan_observed':
        local[0, 0] = np.nan
    else:
        local = torch.from_numpy(local)
    with pytest.raises(ValueError):
        unit_audio_descriptor(local, valid, start, duration)
