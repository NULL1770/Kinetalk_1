"""Native context equivalence, causal GT history and padding contracts."""
import copy

import pytest
import torch
from torch import nn

from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from kinetalk_b0.models.slow_state_affect import lift_slow_state
from scripts import native_context_runtime as runtime
from scripts import train_context_mechanism as context


CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def flow_fixture(frames=96):
    gen = torch.Generator().manual_seed(902)
    valid = torch.ones(2, frames, dtype=torch.bool)
    valid[0, 4:6] = False
    valid[1, :2] = False
    valid[1, -3:] = False
    b = {'valid': valid, 'motion': torch.randn(2, frames, 52, generator=gen),
         'static_upper': torch.randn(2, 9, generator=gen),
         'h0': torch.randn(2, frames, 4, generator=gen),
         'audio_global': torch.randn(2, 3, generator=gen),
         'audio_intensity': torch.ones(2, 1),
         'channel_mask': torch.ones(2, 52, dtype=torch.bool)}
    native = torch.randn(2, frames, 3, generator=gen)
    ident = {'code': torch.randn(2, 2, generator=gen)}
    scales = torch.arange(1, 10).float()*.2
    noise = torch.randn(2, frames, 9, generator=gen)
    time = torch.rand(2, generator=gen)
    return b, ident, native, scales, noise, time


def test_native96_loss_and_gradients_equal_locked_context_batch():
    b, ident, native, scales, _, _ = flow_fixture()
    upper = PrefixUpperFlow(CFG).eval()
    other = copy.deepcopy(upper)
    a_native = native.clone().requires_grad_()
    b_native = native.clone().requires_grad_()
    previous, (noise, time) = context.context_batch(upper, b, ident, a_native, scales,
        torch.Generator().manual_seed(761), 'chunk_teacher')
    current = runtime.variable_context_loss(other, b, ident, b_native, scales, noise, time)
    torch.testing.assert_close(current, previous, atol=0, rtol=0)
    previous.backward(); current.backward()
    torch.testing.assert_close(a_native.grad, b_native.grad, atol=0, rtol=0)
    for left, right in zip(upper.parameters(), other.parameters()):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_tail_gap_padding_poison_does_not_affect_loss_or_gradients():
    b, ident, native, scales, noise, time = flow_fixture(37)
    # Interior invalid chunk and a three-frame final tail exercise skipping,
    # native gaps, and a tail noise layout padded to exactly24 slots.
    b['valid'][0, 16:32] = False
    clean = copy.deepcopy(b)
    dirty = copy.deepcopy(b)
    dirty_native = native.clone()
    dirty_noise = noise.clone()
    for key in ('motion', 'h0'):
        dirty[key][~b['valid']] = float('nan')
        clean[key][~b['valid']] = 0.
    dirty_native[~b['valid']] = float('nan')
    clean_native = torch.where(b['valid'][..., None], native, 0.)
    dirty_noise[~b['valid']] = float('nan')
    clean_noise = torch.where(b['valid'][..., None], noise, 0.)
    left = PrefixUpperFlow(CFG).eval()
    right = copy.deepcopy(left)
    dirty_native.requires_grad_(); clean_native.requires_grad_()
    a = runtime.variable_context_loss(left, dirty, ident, dirty_native, scales, dirty_noise, time)
    z = runtime.variable_context_loss(right, clean, ident, clean_native, scales, clean_noise, time)
    assert torch.isfinite(a)
    torch.testing.assert_close(a, z, atol=0, rtol=0)
    a.backward(); z.backward()
    torch.testing.assert_close(dirty_native.grad, clean_native.grad, atol=0, rtol=0)
    assert dirty_native.grad[~b['valid']].count_nonzero() == 0
    for lp, rp in zip(left.parameters(), right.parameters()):
        if lp.grad is not None:
            torch.testing.assert_close(lp.grad, rp.grad, atol=0, rtol=0)


def test_appending_invalid_padding_is_loss_and_gradient_invariant():
    b, ident, native, scales, noise, time = flow_fixture(37)
    padded = copy.deepcopy(b)
    for key in ('motion', 'h0'):
        padded[key] = torch.nn.functional.pad(b[key], (0, 0, 0, 27), value=float('nan'))
    padded['valid'] = torch.nn.functional.pad(b['valid'], (0, 27), value=False)
    native_pad = torch.nn.functional.pad(native, (0, 0, 0, 27), value=float('nan'))
    noise_pad = torch.nn.functional.pad(noise, (0, 0, 0, 27), value=float('nan'))
    upper = PrefixUpperFlow(CFG).eval()
    current = runtime.variable_context_loss(upper, b, ident, native, scales, noise, time)
    longer = runtime.variable_context_loss(upper, padded, ident, native_pad, scales, noise_pad, time)
    torch.testing.assert_close(current, longer, atol=0, rtol=0)


def test_unknown_supervision_and_known_prefix_use_exact_native_positions(monkeypatch):
    b, ident, native, scales, noise, time = flow_fixture(37)
    b['valid'].fill_(True)
    b['valid'][0, 12:15] = False
    target = runtime.p.h.normalized_target(b, scales)
    calls = []

    class Receiver:
        def flow_loss_prefix(self, supervision, valid, h0, identity, affect, local, supplied_noise, supplied_time, *, known, known_mask):
            index = len(calls)*16
            expected = torch.zeros_like(known)
            expected_mask = torch.zeros_like(known_mask)
            if index:
                expected[:, :8] = target[:, index-8:index]
                expected_mask[:, :8] = b['valid'][:, index-8:index]
            torch.testing.assert_close(known, torch.where(expected_mask[..., None], expected, 0.), atol=0, rtol=0)
            assert torch.equal(known_mask, expected_mask)
            assert not known.requires_grad
            assert (valid & ~known_mask)[:, :8].count_nonzero() == 0
            tail = min(16, 37-index)
            torch.testing.assert_close(supplied_noise[:, 8:8+tail], noise[:, index:index+tail], atol=0, rtol=0)
            assert supplied_noise.shape == (2, 24, 9)
            torch.testing.assert_close(supplied_time, time, atol=0, rtol=0)
            assert supervision[~(valid & ~known_mask)].count_nonzero() == 0
            calls.append(index)
            return torch.tensor(float(index+1))

    result = runtime.variable_context_loss(Receiver(), b, ident, native, scales, noise, time)
    counts = [int(b['valid'][:, start:start+16].sum()) for start in (0, 16, 32)]
    assert float(result) == pytest.approx(sum(v*c for v, c in zip((1, 17, 33), counts))/sum(counts))
    assert calls == [0, 16, 32]


def test_paired_noise_uses_common_full_draw_and_native_aligned_center_slices():
    lengths, starts = [37, 125, 170], [0, 14, 37]
    one = torch.Generator().manual_seed(881)
    two = torch.Generator().manual_seed(881)
    full, tf, df = runtime.paired_noise(lengths, starts, one, 'full')
    center, tc, dc = runtime.paired_noise(lengths, starts, two, 'center')
    assert full.shape == (3, 176, 9) and center.shape == (3, 96, 9)
    assert df['draw_frames'] == dc['draw_frames'] == 176
    torch.testing.assert_close(df['full_noise'], dc['full_noise'], atol=0, rtol=0)
    torch.testing.assert_close(tf, tc, atol=0, rtol=0)
    assert torch.equal(one.get_state(), two.get_state())
    for row, start in enumerate(starts):
        torch.testing.assert_close(center[row], full[row, start:start+96], atol=0, rtol=0)
    assert center[0, 37:].count_nonzero() > 0, 'Padding gets paired noise; mask excludes it from FM.'


def test_paired_noise_native96_matches_existing_random_draw_order():
    a = torch.Generator().manual_seed(752)
    b = torch.Generator().manual_seed(752)
    noise, time, _ = runtime.paired_noise([96, 96], [0, 0], a, 'center')
    expected_noise = torch.randn(2, 96, 9, generator=b)
    expected_time = torch.rand(2, generator=b)
    torch.testing.assert_close(noise, expected_noise, atol=0, rtol=0)
    torch.testing.assert_close(time, expected_time, atol=0, rtol=0)
    assert torch.equal(a.get_state(), b.get_state())


@pytest.mark.parametrize('lengths,starts,mode', [([0], [0], 'full'), ([40], [1], 'center'),
    ([120], [25], 'full'), ([120], [-1], 'full'), ([120.], [0], 'full'),
    ([120], [0], 'other'), ([120, 160], [0], 'full')])
def test_paired_noise_rejects_bad_metadata_before_advancing_rng(lengths, starts, mode):
    gen = torch.Generator().manual_seed(312)
    before = gen.get_state().clone()
    with pytest.raises(ValueError):
        runtime.paired_noise(lengths, starts, gen, mode)
    assert torch.equal(gen.get_state(), before)


class FakeSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def stage1(self, content, valid):
        assert content.shape[:2] == valid.shape and content.shape[-1] == 768
        assert not torch.is_grad_enabled() and torch.isfinite(content).all()
        self.calls.append((content.shape, valid.clone()))
        return {'b0': content[..., :52]*.1, 'h0': content[..., :4]*.2}


class FakeAudio(nn.Module):
    def forward(self, features, valid):
        assert features.shape == (*valid.shape, 1540) and not torch.is_grad_enabled()
        clean = torch.where(valid[..., None], features, 0.)
        return {'global': clean[..., :3].sum(1)/valid.sum(1)[:, None],
                'intensity_value': clean[..., 3:4].sum(1)/valid.sum(1)[:, None],
                'state': clean[..., 4:8], 'local': torch.full((*valid.shape, 64), 900.)}


def encoding_fixture():
    gen = torch.Generator().manual_seed(724)
    valid = torch.ones(2, 37, dtype=torch.bool)
    valid[0, 5:8] = False
    valid[0, 33:] = False
    b = {'valid': valid, 'content': torch.randn(2, 37, 768, generator=gen),
         'audio_features': torch.randn(2, 37, 1540, generator=gen),
         'anchors': torch.randn(2, 52, generator=gen), 'anchor_valid': torch.ones(2, 52, dtype=torch.bool),
         'motion': object(), 'emotion_id': object(), 'prefix_local': object(),
         'native_lengths': torch.tensor([33, 37]), 'crop_start': [0, 5]}
    b['content'][~valid] = float('nan')
    b['audio_features'][~valid] = float('nan')
    return FakeSystem().eval(), FakeAudio().eval(), b, torch.arange(1, 53).float()*.02


def test_encoding_uses_true_length_audio_only_origin_and_preserves_input_fields():
    system, audio, b, scales = encoding_fixture()
    before = {key: value.clone() for key, value in b.items() if torch.is_tensor(value)}
    result = runtime.encode_full_conditions(system, audio, b, scales)
    assert result is not b and 'static_upper' not in b and 'h0' not in b
    assert sorted(shape[1] for shape, _ in system.calls) == [33, 37]
    assert result['motion'] is b['motion'] and 'prefix_local' not in result
    assert 'prefix_local' in b
    for key, old in before.items():
        torch.testing.assert_close(b[key], old, atol=0, rtol=0, equal_nan=True)
    state = torch.where(b['valid'][..., None], b['audio_features'][..., 4:8], 0.)
    mean = state.sum(1, keepdim=True)/b['valid'].sum(1)[:, None, None]
    expected = b['anchors'][:, None]+lift_slow_state(mean, scales)
    torch.testing.assert_close(result['static_upper'], expected[:, 0, runtime.p.CC], atol=0, rtol=0)
    assert result['b0'][~b['valid']].count_nonzero() == 0
    assert result['h0'][~b['valid']].count_nonzero() == 0
    for key in ('b0', 'h0', 'static_upper', 'audio_global', 'audio_intensity'):
        assert not result[key].requires_grad and torch.isfinite(result[key]).all()
    b['motion'] = 'different target must be irrelevant'
    b['emotion_id'] = -1000
    again = runtime.encode_full_conditions(system, audio, b, scales)
    for key in ('b0', 'h0', 'static_upper', 'audio_global', 'audio_intensity'):
        torch.testing.assert_close(result[key], again[key], atol=0, rtol=0)


@pytest.mark.parametrize('bad', ['train_mode', 'unfrozen', 'anchor_mask', 'scale', 'observed_nan'])
def test_encoding_rejects_unfrozen_or_invalid_conditions(bad):
    system, audio, b, scales = encoding_fixture()
    if bad == 'train_mode':
        audio.train()
    elif bad == 'unfrozen':
        audio.register_parameter('unexpected', nn.Parameter(torch.tensor(1.)))
    elif bad == 'anchor_mask':
        b['anchor_valid'][0, runtime.p.CC[0]] = False
    elif bad == 'scale':
        scales[0] = 0.
    else:
        b['content'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        runtime.encode_full_conditions(system, audio, b, scales)


def test_unsupervised_channels_are_not_read_and_upper_observation_loss_is_rejected():
    b, ident, native, scales, noise, time = flow_fixture(21)
    upper = PrefixUpperFlow(CFG).eval()
    expected = runtime.variable_context_loss(upper, b, ident, native, scales, noise, time)
    b['motion'][..., list(runtime.p.r.NOT_UPPER)] = float('nan')
    actual = runtime.variable_context_loss(upper, b, ident, native, scales, noise, time)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    b['channel_mask'][0, runtime.p.CC[0]] = False
    with pytest.raises(ValueError, match='nine supervised'):
        runtime.variable_context_loss(upper, b, ident, native, scales, noise, time)
