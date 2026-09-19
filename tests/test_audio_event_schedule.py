import inspect

import pytest
import torch

from kinetalk_b0.models.audio_event_schedule import AudioEventSchedulePredictor, DURATIONS


def fixture():
    torch.manual_seed(12)
    m = AudioEventSchedulePredictor(6, 3, hidden=10).double()
    x = torch.randn(2, 21, 6, dtype=torch.float64)
    c = torch.randn(2, 3, dtype=torch.float64)
    v = torch.ones(2, 21, dtype=torch.bool); v[0, 6:9] = False; v[1, -2:] = False
    x[~v] = float('nan')
    return m, x, c, v


def test_audio_only_signature_and_shapes():
    assert list(inspect.signature(AudioEventSchedulePredictor.forward).parameters) == ['self', 'features', 'context', 'valid']
    m, x, c, v = fixture(); out = m(x, c, v)
    assert set(out) == {'onset_logits', 'duration_logits'}
    assert out['onset_logits'].shape == (2, 21, 4)
    assert out['duration_logits'].shape == (2, 21, 4, 7)
    for value in out.values():
        assert torch.isfinite(value).all()
        assert not value[~v].any()


def test_constant_audio_is_constant_at_all_edges_and_gaps():
    m, x, c, v = fixture(); x[:] = torch.randn(2, 1, 6, dtype=x.dtype)
    out = m(x, c, v)
    for value in out.values():
        for row in range(2):
            supported = value[row, v[row]]
            torch.testing.assert_close(supported, supported[:1].expand_as(supported), atol=1e-14, rtol=0)


def test_gap_separates_runs_and_padding_does_not_change_observed_output():
    m, x, c, v = fixture(); out = m(x, c, v)
    changed = x.clone(); changed[0, 9:] *= 100
    other = m(changed, c, v)
    for key in out: torch.testing.assert_close(out[key][0, :6], other[key][0, :6], rtol=0, atol=0)
    isolated = m(x[:1, 9:], c[:1], v[:1, 9:])
    for key in out: torch.testing.assert_close(out[key][0, 9:], isolated[key][0], atol=1e-14, rtol=0)
    pad = torch.cat((x, torch.full((2, 4, 6), float('inf'), dtype=x.dtype)), 1)
    pv = torch.cat((v, torch.zeros(2, 4, dtype=torch.bool)), 1)
    padded = m(pad, c, pv)
    for key in out: torch.testing.assert_close(out[key], padded[key][:, :21], atol=1e-14, rtol=0)


def test_invalid_features_have_zero_gradient_and_finite_parameter_gradients():
    m, x, c, v = fixture(); x.requires_grad_()
    out = m(x, c, v); sum(value.square().mean() for value in out.values()).backward()
    assert torch.isfinite(x.grad).all() and not x.grad[~v].any()
    for parameter in m.parameters(): assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_explicit_rng_sampling_is_reproducible_and_gap_safe():
    m, x, c, v = fixture()
    with torch.no_grad(): m.onset_head.weight.zero_(); m.onset_head.bias.fill_(5.)
    state = torch.random.get_rng_state().clone()
    a = m.sample_schedule(x, c, v, generator=torch.Generator().manual_seed(18))
    b = m.sample_schedule(x, c, v, generator=torch.Generator().manual_seed(18))
    assert a['condition'].shape == (2, 21, 12)
    for key in a: assert torch.equal(a[key], b[key])
    assert torch.equal(state, torch.random.get_rng_state())
    assert not a['condition'][~v].any()
    assert set(a['sampled_duration'][a['onset']].tolist()) <= set(DURATIONS)
    assert torch.isfinite(a['condition']).all()


def test_invalid_input_contract_is_rejected():
    m, x, c, v = fixture(); x[0, 0] = float('nan')
    with pytest.raises(ValueError): m(x, c, v)
    with pytest.raises(TypeError): m.sample_schedule(x, c, v)
