"""Synthetic prefix-window and rollout checks; no persisted dataset reads."""
import copy

import pytest
import torch
from torch import nn

from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from scripts import train_prefix_upper as runner


CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(frames=40, batch=2):
    torch.manual_seed(209)
    upper = PrefixUpperFlow(CFG).eval()
    clock = torch.arange(frames).float()[None, :, None]
    b = {'valid': torch.ones(batch, frames, dtype=torch.bool),
         'h0': clock.expand(batch, -1, 4).clone(),
         'audio_global': torch.randn(batch, 3), 'audio_intensity': torch.randn(batch, 1),
         'motion': torch.randn(batch, frames, 52), 'static_upper': torch.randn(batch, 9),
         'emotion_id': torch.arange(batch).remainder(2), 'speaker_id': torch.arange(batch),
         'channel_mask': torch.ones(batch, 52, dtype=torch.bool),
         'times': torch.arange(frames, dtype=torch.float64)[None].expand(batch, -1) / 25,
         'b0': torch.zeros(batch, frames, 52), 'clip_id': ['synthetic_'+str(i) for i in range(batch)]}
    local = (clock + 100).expand(batch, -1, 3).clone()
    source = (clock + 200).expand(batch, -1, 9).clone()
    noise = torch.randn(batch, frames, 9)
    identity = {'code': torch.randn(batch, 2), 'baseline': torch.zeros(batch, 52)}
    return upper, b, identity, local, source, noise


class ExplicitPrefixDecoder:
    def __init__(self): self.calls = []

    def decode_prefix(self, valid, h0, identity, affect, local, noise, *, known, known_mask, steps):
        self.calls.append({'valid': valid.clone(), 'known': known.clone(), 'known_mask': known_mask.clone(),
                           'noise': noise.clone(), 'h0': h0.clone(), 'local': local.clone()})
        clean = torch.where(known_mask[..., None], known, 0.)
        mean = clean.sum(1) / known_mask.sum(1)[:, None].clamp_min(1)
        generated = noise + mean[:, None]
        return torch.where(known_mask[..., None], known, torch.where(valid[..., None], generated, 0.))


def test_window_has_fixed_slots_and_native_clock_for_first_middle_and_tail_chunks():
    _, b, identity, local, source, _ = fixture(frames=35, batch=3)
    conditions, known, mask, index = runner.window_batch(b, identity, local, source,
                                                        torch.arange(3), torch.tensor([0, 16, 32]))
    valid, h0, _, _, native = conditions
    assert valid.shape == mask.shape == index.shape == (3, 24)
    assert known.shape == (3, 24, 9) and h0.shape == (3, 24, 4)
    assert not valid[0, :8].any() and valid[0, 8:].all()
    assert valid[1].all() and mask[1, :8].all()
    assert valid[2, :11].all() and not valid[2, 11:].any()
    assert mask[:, 8:].count_nonzero() == known[:, 8:].count_nonzero() == 0
    for row, start in enumerate([0, 16, 32]):
        current_count = min(16, 35-start)
        torch.testing.assert_close(h0[row, 8:8+current_count, 0], torch.arange(start, start+current_count).float())
        torch.testing.assert_close(native[row, 8:8+current_count, 0], torch.arange(start, start+current_count).float()+100)
    torch.testing.assert_close(known[1, :8, 0], torch.arange(8., 16.)+200)
    torch.testing.assert_close(known[2, :8, 0], torch.arange(24., 32.)+200)


def test_source_current_future_and_missing_history_payload_cannot_change_known_values_or_gradients():
    _, b, identity, local, source, _ = fixture()
    b['valid'][1, [9, 11]] = False
    ids, starts = torch.tensor([0, 1]), torch.tensor([0, 16])
    _, expected, expected_mask, _ = runner.window_batch(b, identity, local, source, ids, starts)
    polluted = source.clone(); polluted[0] = float('nan'); polluted[1, 16:] = float('nan')
    polluted[1, [9, 11]] = float('nan'); polluted.requires_grad_()
    _, actual, mask, _ = runner.window_batch(b, identity, local, polluted, ids, starts)
    assert torch.equal(mask, expected_mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(polluted.grad).all()
    allowed = torch.zeros_like(b['valid']); allowed[1, 8:16] = b['valid'][1, 8:16]
    assert polluted.grad[~allowed].count_nonzero() == 0
    assert polluted.grad[allowed].abs().sum() > 0


def test_early_partial_prefix_and_internal_gap_keep_fixed_positions():
    _, b, identity, local, source, _ = fixture(batch=1)
    b['valid'][0, 1] = False
    conditions, known, mask, _ = runner.window_batch(b, identity, local, source, torch.tensor([0]), torch.tensor([3]))
    valid = conditions[0]
    assert not valid[0, :5].any()
    assert mask[0].nonzero(as_tuple=True)[0].tolist() == [5, 7]
    assert known[0, 5, 0] == 200 and known[0, 7, 0] == 202
    assert known[0, 6].count_nonzero() == 0
    assert conditions[1][0, 8, 0] == 3


def test_reverse_prefix_only_reverses_observed_motion_not_audio_clock_or_mask():
    _, b, identity, local, source, _ = fixture(batch=1)
    b['valid'][0, [10, 13]] = False
    ids, starts = torch.tensor([0]), torch.tensor([16])
    conditions, known, mask, _ = runner.window_batch(b, identity, local, source, ids, starts)
    reverse_conditions, reversed_known, reversed_mask, _ = runner.window_batch(b, identity, local, source, ids, starts, reverse=True)
    assert torch.equal(mask, reversed_mask)
    torch.testing.assert_close(reversed_known[mask], known[mask].flip(0), atol=0, rtol=0)
    torch.testing.assert_close(reverse_conditions[1], conditions[1], atol=0, rtol=0)
    torch.testing.assert_close(reverse_conditions[-1], conditions[-1], atol=0, rtol=0)


def test_empty_prefix_ignores_source_and_keeps_current_at_slots_eight_to_twenty_three():
    _, b, identity, local, _, _ = fixture(batch=1)
    conditions, known, mask, _ = runner.window_batch(b, identity, local, None, torch.tensor([0]), torch.tensor([16]), empty=True)
    assert not conditions[0][0, :8].any() and conditions[0][0, 8:].all()
    assert known.count_nonzero() == mask.count_nonzero() == 0
    assert conditions[1][0, 8, 0] == 16
    assert conditions[1][0, :8].count_nonzero() == 0


def test_generated_rollout_ignores_query_motion_emotion_and_other_target_fields():
    upper, b, identity, local, _, noise = fixture()
    expected = runner.rollout(upper, b, identity, local, noise, steps=2)
    changed = copy.deepcopy(b)
    changed['motion'].fill_(float('nan')); changed['emotion_id'].fill_(1000); changed['static_upper'].fill_(float('nan'))
    actual = runner.rollout(upper, changed, identity, local, noise, steps=2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not actual.requires_grad


def test_oracle_uses_only_strict_past_and_never_current_or_future_target():
    _, b, identity, local, target, noise = fixture(frames=48, batch=1)
    decoder = ExplicitPrefixDecoder()
    expected = runner.rollout(decoder, b, identity, local, noise, steps=1, mode='oracle_history', oracle_target=target)
    assert len(decoder.calls) == 3
    for call, start in zip(decoder.calls, [0, 16, 32]):
        if start:
            torch.testing.assert_close(call['known'][:, :8], target[:, start-8:start], atol=0, rtol=0)
        assert call['known'][:, 8:].count_nonzero() == 0
    changed = target.clone(); changed[:, 16:32] += 100
    actual = runner.rollout(ExplicitPrefixDecoder(), b, identity, local, noise, steps=1,
                            mode='oracle_history', oracle_target=changed)
    torch.testing.assert_close(actual[:, :32], expected[:, :32], atol=0, rtol=0)
    assert not torch.equal(actual[:, 32:], expected[:, 32:])
    changed = target.clone(); changed[:, 32:] = float('nan')
    final = runner.rollout(ExplicitPrefixDecoder(), b, identity, local, noise, steps=1,
                           mode='oracle_history', oracle_target=changed)
    torch.testing.assert_close(final, expected, atol=0, rtol=0)


def test_no_prefix_rollout_is_exactly_equal_for_all_history_interventions():
    upper, b, identity, local, _, noise = fixture()
    expected = runner.rollout(upper, b, identity, local, noise, steps=2, use_prefix=False)
    for mode in ('empty', 'reverse_history', 'oracle_history'):
        kwargs = {'oracle_target': torch.full_like(noise, float('nan'))} if mode == 'oracle_history' else {}
        actual = runner.rollout(upper, b, identity, local, noise, steps=2, use_prefix=False, mode=mode, **kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mode_and_oracle_argument_contracts_are_explicit():
    upper, b, identity, local, _, noise = fixture()
    with pytest.raises(ValueError, match='Unknown'):
        runner.rollout(upper, b, identity, local, noise, mode='typo')
    with pytest.raises(ValueError, match='explicit'):
        runner.rollout(upper, b, identity, local, noise, mode='oracle_history')
    with pytest.raises(ValueError, match='reject'):
        runner.rollout(upper, b, identity, local, noise, oracle_target=noise)


def test_rollout_skips_whole_empty_chunks_and_handles_tail_and_nan_padding():
    upper, b, identity, local, _, noise = fixture(frames=37)
    b['valid'][0, :32] = False
    b['valid'][1, 16:32] = False
    b['valid'][1, -1] = False
    for value in (b['h0'], local, noise): value[~b['valid']] = float('nan')
    decoder = ExplicitPrefixDecoder()
    output = runner.rollout(decoder, b, identity, local, noise, steps=1)
    assert len(decoder.calls) == 2
    assert [len(call['valid']) for call in decoder.calls] == [1, 2]
    assert all(call['valid'].shape[1] == 24 for call in decoder.calls)
    assert not decoder.calls[-1]['known_mask'].any()
    assert torch.isfinite(output).all() and output[~b['valid']].count_nonzero() == 0
    actual = runner.rollout(upper, b, identity, local, noise, steps=1)
    assert torch.isfinite(actual).all() and actual[~b['valid']].count_nonzero() == 0


def test_patch_loss_reads_only_current_supervision_and_detaches_source_history():
    upper, b, identity, local, source, _ = fixture()
    source.requires_grad_()
    target = torch.randn(2, 40, 9, requires_grad=True)
    local.requires_grad_()
    ids, starts = torch.tensor([0, 1]), torch.tensor([0, 16])
    noise = torch.randn(2, 24, 9)
    loss = runner.patch_loss(upper, b, identity, local, target, source, ids, starts,
                             noise, torch.tensor([.2, .6]))
    loss.backward()
    assert source.grad is None
    current = torch.zeros_like(b['valid']); current[0, :16] = True; current[1, 16:32] = True
    assert torch.isfinite(target.grad).all() and target.grad[~current].count_nonzero() == 0
    assert target.grad[current].abs().sum() > 0
    assert local.grad is not None and torch.isfinite(local.grad).all()


class UnknownMeanLoss(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(1.)); self.calls = []

    def flow_loss_prefix(self, target, valid, h0, identity, affect, local, noise, times, *, known, known_mask):
        unknown = valid & ~known_mask
        self.calls.append((target.clone(), unknown.clone(), known.clone()))
        return self.weight * target[unknown].square().mean()


def test_patch_loss_normalizes_by_unknown_valid_frames_with_tail_and_gaps():
    _, b, identity, local, source, _ = fixture(frames=35)
    b['valid'][0, 18:22] = False
    b['valid'][1, 33] = False
    target = torch.full((2, 35, 9), float('nan'))
    target[0, 16:32] = 2; target[1, 32:] = 4
    target[~b['valid']] = float('nan')
    model = UnknownMeanLoss()
    loss = runner.patch_loss(model, b, identity, local, target, source, torch.tensor([0, 1]),
                             torch.tensor([16, 32]), torch.zeros(2, 24, 9), torch.tensor([.3, .8]))
    # 12 current frames of value2 and two current frames of value4. Known eight
    # frames in each row and invalid tail/gaps do not dilute the denominator.
    assert loss.item() == pytest.approx((12*4 + 2*16) / 14)
    saved_target, unknown, _ = model.calls[0]
    assert int(unknown.sum()) == 14 and unknown[:, :8].count_nonzero() == 0
    assert torch.isfinite(saved_target).all()


def test_patch_loss_no_prefix_is_independent_of_any_source_values():
    upper, b, identity, local, source, _ = fixture()
    target = torch.randn_like(source); noise = torch.randn(2, 24, 9)
    ids, starts, times = torch.tensor([0, 1]), torch.tensor([0, 16]), torch.tensor([.3, .8])
    first = runner.patch_loss(upper, b, identity, local, target, source, ids, starts, noise, times, empty=True)
    second = runner.patch_loss(upper, b, identity, local, target, torch.full_like(source, float('nan')),
                               ids, starts, noise, times, empty=True)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_pilot_evaluation_keeps_nonupper_as_unscored_zero_placeholders_and_marks_oracle():
    upper, b, identity, local, _, _ = fixture(frames=20, batch=1)
    report, curves = runner.evaluate(upper, b, identity, local, torch.ones(9), steps=1, use_prefix=True)
    assert report['nonupper_scored'] is False and report['test_loaded'] is False
    assert 'distribution' not in report
    for key, pred in curves['predictions'].items():
        assert pred[..., list(runner.r.NOT_UPPER)].count_nonzero() == 0
        assert set(report['modes'][key]['metrics']) == {'brows', 'eyes_expression'}
        assert report['modes'][key]['oracle_target_history'] == key.endswith('/oracle_history')
    assert curves['noise_seeds'] == list(runner.r.SEEDS)
