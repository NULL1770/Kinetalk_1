import copy

import pytest
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow
from kinetalk_b0.models.prior_audio_adapter import BoundedPriorAudioAdapter


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def batch():
    generator = torch.Generator().manual_seed(931)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    frames = torch.tensor([[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]],
                           [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]], dtype=torch.bool)
    return dict(noisy=torch.randn(2, 3, 4, generator=generator),
                time=torch.tensor([.2, .7]), valid=valid,
                context=torch.randn(2, 7, generator=generator),
                audio_blocks=torch.randn(2, 3, 5, 6, generator=generator),
                audio_frame_valid=frames)


def model():
    torch.manual_seed(729)
    return BoundedPriorAudioAdapter(ContinuousLatentFlow(7, audio_dim=6, latent_dim=4,
                                    hidden=12, depth=2), hidden=12, depth=3, audio_hidden=5)


def excite(adapter):
    with torch.no_grad():
        adapter.output.weight.normal_(std=.2)
        adapter.output.bias.normal_(std=.2)


def test_zero_initialization_and_disabled_path_are_exact_prior():
    adapter = model(); values = batch()
    prior = adapter.prior.velocity(**values, use_audio=False)
    assert torch.equal(adapter.velocity(**values, use_audio=True), prior)
    assert torch.count_nonzero(adapter.residual(**values)) == 0
    excite(adapter)
    assert torch.equal(adapter.velocity(**values, use_audio=False), prior)
    sample_args = {key: values[key] for key in ('valid', 'context', 'audio_blocks', 'audio_frame_valid')}
    expected = adapter.prior.sample(**sample_args, noise=values['noisy'], steps=5, use_audio=False)
    assert torch.equal(adapter.sample(**sample_args, noise=values['noisy'], steps=5, use_audio=False), expected)


def test_only_adapter_receives_gradients_and_prior_never_changes():
    adapter = model(); values = batch()
    before = {key: value.clone() for key, value in adapter.prior.state_dict().items()}
    optimizer = torch.optim.AdamW(adapter.adapter_parameters(), lr=.001)
    for _ in range(3):
        adapter.train()
        assert adapter.training and not adapter.prior.training
        assert all(not parameter.requires_grad for parameter in adapter.prior.parameters())
        loss = adapter.flow_loss(values['noisy'] * .4, values['valid'], values['context'],
                                 values['audio_blocks'], values['noisy'], values['time'],
                                 audio_frame_valid=values['audio_frame_valid'])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        assert all(parameter.grad is None for parameter in adapter.prior.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in adapter.audio_frame_projection.parameters())
    assert all(torch.equal(before[key], value) for key, value in adapter.prior.state_dict().items())
    assert not torch.equal(adapter.velocity(**values), adapter.velocity(**values, use_audio=False))
    assert {id(p) for p in adapter.adapter_parameters()}.isdisjoint({id(p) for p in adapter.prior.parameters()})
    adapter.eval(); assert not adapter.training and not adapter.prior.training


def test_fixed_bound_and_invalid_latent_positions_are_zero():
    adapter = model(); excite(adapter); values = batch()
    with torch.no_grad():
        adapter.output.weight.mul_(1e5)
    correction = adapter.residual(**values)
    assert correction.abs().max() <= adapter.max_delta
    assert torch.count_nonzero(correction[~values['valid']]) == 0


def test_padding_nan_tail_values_are_ignored_and_valid_order_matters():
    adapter = model(); excite(adapter); values = batch()
    expected = adapter.residual(**values)
    padded = dict(values)
    padded['audio_blocks'] = torch.where(values['audio_frame_valid'][..., None],
                                          values['audio_blocks'], float('nan'))
    padded['noisy'] = torch.where(values['valid'][..., None], values['noisy'], float('nan'))
    assert torch.equal(adapter.residual(**padded), expected)
    changed = dict(values); changed['audio_blocks'] = values['audio_blocks'].clone()
    changed['audio_blocks'][0, 0] = changed['audio_blocks'][0, 0].flip(0)
    assert not torch.equal(adapter.residual(**changed), expected)


def test_sample_uses_only_explicit_noise_and_no_internal_randomness():
    adapter = model(); excite(adapter); values = batch()
    arguments = {key: values[key] for key in ('valid', 'context', 'audio_blocks', 'audio_frame_valid')}
    first = adapter.sample(**arguments, noise=values['noisy'], steps=4)
    torch.randn(51)
    adapter.train()
    second = adapter.sample(**arguments, noise=values['noisy'], steps=4)
    assert torch.equal(first, second)
    assert not torch.equal(first, adapter.sample(**arguments, noise=values['noisy'] + .1, steps=4))


def test_state_dict_roundtrip_retains_bound_config_and_same_output():
    original = model(); excite(original); values = batch()
    restored = model(); restored.load_state_dict(copy.deepcopy(original.state_dict()))
    assert restored.config == original.config
    assert torch.equal(original.velocity(**values), restored.velocity(**values))
    assert all(not p.requires_grad for p in restored.prior.parameters())


@pytest.mark.parametrize('field', ['audio_nan', 'tail_hole', 'latent_hole', 'tail_support', 'context', 'time'])
def test_invalid_observed_inputs_are_rejected(field):
    adapter = model(); values = batch()
    if field == 'audio_nan': values['audio_blocks'][0, 0, 0, 0] = float('nan')
    elif field == 'tail_hole': values['audio_frame_valid'][0, 0, 1] = False
    elif field == 'latent_hole': values['valid'][0, 1] = False
    elif field == 'tail_support': values['audio_frame_valid'][0, -1] = False
    elif field == 'context': values['context'][0, 0] = float('inf')
    elif field == 'time': values['time'][0] = 1.1
    with pytest.raises(ValueError): adapter.velocity(**values)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_invalid_bound_rejected(value):
    with pytest.raises(ValueError): BoundedPriorAudioAdapter(ContinuousLatentFlow(7), max_delta=value)


def test_no_target_or_motion_mask_in_generation_api():
    adapter = model(); values = batch()
    with pytest.raises(TypeError): adapter.velocity(**values, target=torch.zeros(2, 3, 9))
    with pytest.raises(TypeError): adapter.velocity(**values, motion_mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError): adapter.velocity(**values, use_audio=1)


def test_disabled_flow_loss_is_exact_prior_loss():
    adapter = model(); excite(adapter); values = batch()
    args = (values['noisy']*.3, values['valid'], values['context'], values['audio_blocks'],
            values['noisy'], values['time'])
    options = dict(use_audio=False, audio_frame_valid=values['audio_frame_valid'])
    assert torch.equal(adapter.flow_loss(*args, **options), adapter.prior.flow_loss(*args, **options))
