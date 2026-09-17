"""Synthetic runner contract checks; no stored training or evaluation data."""
import copy

import pytest
import torch
from torch import nn

from kinetalk_b0.models.history_upper_flow import HistoryUpperFlow
from scripts import train_history_upper as runner


CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(*, frames=40, batch=2, use_history=True):
    torch.manual_seed(106)
    upper = HistoryUpperFlow(CFG, use_history=use_history, history_hidden=8).eval()
    valid = torch.ones(batch, frames, dtype=torch.bool)
    b = {'valid': valid, 'h0': torch.randn(batch, frames, 4),
         'audio_global': torch.randn(batch, 3), 'audio_intensity': torch.randn(batch, 1),
         'motion': torch.randn(batch, frames, 52), 'static_upper': torch.randn(batch, 9),
         'emotion_id': torch.arange(batch), 'channel_mask': torch.ones(batch, 52, dtype=torch.bool)}
    identity = {'code': torch.randn(batch, 2)}
    local = torch.randn(batch, frames, 3)
    noise = torch.randn(batch, frames, 9)
    return upper, b, identity, local, noise


class HistoryRecordingDecoder:
    """Make each past-window effect explicit, independent of random weights."""

    def __init__(self):
        self.calls = []

    def decode_history(self, valid, h0, identity, affect, local, noise, *, history, history_valid, steps):
        self.calls.append({'valid': valid.clone(), 'history': history.clone(),
                           'history_valid': history_valid.clone(), 'identity': identity.clone()})
        clean = torch.where(history_valid[..., None], history, 0.)
        mean = clean.sum(1) / history_valid.sum(1)[:, None].clamp_min(1)
        return torch.where(valid[..., None], noise + mean[:, None], 0.)


def test_full_rollout_is_independent_of_query_motion_and_emotion_labels():
    upper, b, identity, local, noise = fixture()
    expected = runner.rollout(upper, b, identity, local, noise, steps=2)
    changed = copy.deepcopy(b)
    changed['motion'].fill_(float('nan'))
    changed['emotion_id'].fill_(10000)
    changed['static_upper'].fill_(float('nan'))
    actual = runner.rollout(upper, changed, identity, local, noise, steps=2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not actual.requires_grad


def test_oracle_is_explicit_past_only_and_future_target_changes_affect_later_chunks_only():
    _, b, identity, local, noise = fixture(frames=48, batch=1)
    target = torch.arange(48.)[None, :, None].expand(1, -1, 9).clone()
    decoder = HistoryRecordingDecoder()
    expected = runner.rollout(decoder, b, identity, local, noise, steps=1,
                              mode='oracle_history', oracle_target=target)
    assert len(decoder.calls) == 3
    for call, start in zip(decoder.calls, [0, 16, 32]):
        torch.testing.assert_close(call['history'], target[:, max(0, start-8):start], atol=0, rtol=0)
        assert call['history'].shape[1] <= 8
    changed = target.clone(); changed[:, 16:32] += 100
    actual = runner.rollout(HistoryRecordingDecoder(), b, identity, local, noise, steps=1,
                            mode='oracle_history', oracle_target=changed)
    torch.testing.assert_close(actual[:, :32], expected[:, :32], atol=0, rtol=0)
    assert not torch.equal(actual[:, 32:], expected[:, 32:])
    # The final chunk has no succeeding consumer and cannot change any oracle
    # output, which also rules out implicit current-chunk target conditioning.
    final_only = target.clone(); final_only[:, 32:] += 10000
    final_result = runner.rollout(HistoryRecordingDecoder(), b, identity, local, noise, steps=1,
                                  mode='oracle_history', oracle_target=final_only)
    torch.testing.assert_close(final_result, expected, atol=0, rtol=0)


def test_deploy_rejects_explicit_target_and_oracle_never_falls_back_to_query_motion():
    upper, b, identity, local, noise = fixture()
    for mode in ('full', 'empty', 'reverse_history', 'static', 'reverse'):
        with pytest.raises(ValueError, match='Deployable'):
            runner.rollout(upper, b, identity, local, noise, mode=mode, oracle_target=noise)
    with pytest.raises(ValueError, match='explicit target'):
        runner.rollout(upper, b, identity, local, noise, mode='oracle_history')


def test_no_history_full_empty_reverse_history_and_oracle_are_exactly_equivalent():
    upper, b, identity, local, noise = fixture(use_history=False)
    expected = runner.rollout(upper, b, identity, local, noise, steps=2)
    for mode in ('empty', 'reverse_history', 'oracle_history'):
        kwargs = {'oracle_target': torch.full_like(noise, float('nan'))} if mode == 'oracle_history' else {}
        actual = runner.rollout(upper, b, identity, local, noise, steps=2, mode=mode, **kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_normalized_target_retains_offsets_and_does_not_use_future_query_mean():
    _, b, _, _, _ = fixture(frames=40)
    scales = torch.arange(1., 10.) / 10
    target = runner.normalized_target(b, scales)
    expected = (b['motion'][..., runner.CC] - b['static_upper'][:, None]) / scales
    torch.testing.assert_close(target, expected, atol=0, rtol=0)
    changed = copy.deepcopy(b)
    changed['motion'][:, 24:, runner.CC] += 100
    after = runner.normalized_target(changed, scales)
    torch.testing.assert_close(after[:, :24], target[:, :24], atol=0, rtol=0)
    assert not torch.equal(after[:, 24:], target[:, 24:])
    assert not torch.allclose(target.mean(1), torch.zeros_like(target.mean(1)))
    changed = copy.deepcopy(b)
    changed['valid'][:, -3:] = False
    changed['motion'][:, -3:] = float('nan')
    masked = runner.normalized_target(changed, scales)
    assert torch.isfinite(masked).all() and masked[:, -3:].count_nonzero() == 0


def test_history_at_limits_native_window_and_reverses_observed_order_without_moving_mask():
    source = torch.arange(40.)[None, :, None].expand(2, -1, 9).clone()
    valid = torch.ones(2, 40, dtype=torch.bool); valid[1, [9, 12, 14]] = False
    ids = torch.tensor([1, 0])
    history, mask = runner.history_at(source, valid, ids, 16)
    torch.testing.assert_close(history, source[ids, 8:16], atol=0, rtol=0)
    assert torch.equal(mask, valid[ids, 8:16])
    reversed_history, reversed_mask = runner.history_at(source, valid, ids, 16, mode='reverse_history')
    assert torch.equal(reversed_mask, mask)
    assert torch.equal(reversed_history[~mask], history[~mask])
    for row in range(2):
        torch.testing.assert_close(reversed_history[row, mask[row]], history[row, mask[row]].flip(0), atol=0, rtol=0)
    early, _ = runner.history_at(source, valid, ids, 3)
    assert early.shape == (2, 3, 9)
    empty, empty_mask = runner.history_at(source, valid, ids, 16, mode='empty')
    assert empty.shape == (2, 0, 9) and empty_mask.shape == (2, 0)


def test_rollout_skips_empty_current_chunks_and_uses_no_unobserved_history():
    _, b, identity, local, noise = fixture(frames=48)
    b['valid'][:] = False
    b['valid'][0, 32:40] = True
    b['valid'][1, :16] = True
    b['valid'][1, 32:] = True
    noise[~b['valid']] = float('nan')
    decoder = HistoryRecordingDecoder()
    output = runner.rollout(decoder, b, identity, local, noise, steps=1)
    assert len(decoder.calls) == 2  # The complete middle chunk is skipped.
    assert [len(call['valid']) for call in decoder.calls] == [1, 2]
    assert decoder.calls[0]['history'].shape[1] == 0
    assert not decoder.calls[1]['history_valid'].any()
    assert torch.isfinite(output).all() and output[~b['valid']].count_nonzero() == 0
    torch.testing.assert_close(output[b['valid']], noise[b['valid']], atol=0, rtol=0)


def test_actual_flow_handles_empty_chunks_invalid_history_and_trailing_partial_chunk():
    upper, b, identity, local, noise = fixture(frames=37)
    b['valid'][0, :32] = False
    b['valid'][1, 16:32] = False
    for value in (b['h0'], local, noise):
        value[~b['valid']] = float('nan')
    output = runner.rollout(upper, b, identity, local, noise, steps=1)
    assert output.shape == noise.shape
    assert torch.isfinite(output).all() and output[~b['valid']].count_nonzero() == 0


class RecordingLoss(nn.Module):
    def __init__(self, *, use_history=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.calls = []
        self.use_history = use_history

    def flow_loss_history(self, target, valid, content, identity, affect, native, noise, times,
                          *, history, history_valid):
        self.calls.append({'history': history, 'history_valid': history_valid.clone(),
                           'target': target, 'valid': valid.clone(), 'identity': identity.clone()})
        if not self.use_history:
            return self.weight * target[valid].square().mean()
        mean = torch.where(history_valid[..., None], history, 0.).sum(1)
        mean = mean / history_valid.sum(1)[:, None].clamp_min(1)
        pred = self.weight * native[..., :1] + mean[:, None]
        return (pred[valid] - target[valid]).square().mean()


def training_inputs(b, local, *, requires_grad=False):
    frames = b['valid'].shape[1]
    chunks = (frames + runner.CHUNK - 1) // runner.CHUNK
    target = torch.randn(len(local), frames, 9)
    generated = torch.randn_like(target).requires_grad_(requires_grad)
    noise = torch.randn_like(target)
    times = torch.rand(len(local), chunks)
    draws = torch.rand(len(local), chunks)
    return target, generated, noise, times, draws


def test_scheduled_training_uses_per_chunk_decisions_and_detaches_generated_history():
    _, b, identity, local, _ = fixture(frames=48)
    local.requires_grad_()
    target, generated, noise, times, draws = training_inputs(b, local, requires_grad=True)
    draws[:] = torch.tensor([[.1, .1, .9], [.9, .9, .1]])
    model = RecordingLoss(use_history=True)
    loss, used, available = runner.chunk_training_loss(model, b, identity, local, target, generated,
                                                     noise, times, draws, .5)
    assert (used, available) == (2, 4)
    for i, call in enumerate(model.calls):
        start = i * runner.CHUNK
        left = max(0, start - runner.HISTORY)
        teacher = draws[:, i] < .5
        expected = torch.where(teacher[:, None, None], target[:, left:start], generated.detach()[:, left:start])
        torch.testing.assert_close(call['history'], expected, atol=0, rtol=0)
        assert not call['history'].requires_grad
    loss.backward()
    assert generated.grad is None
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad)
    assert torch.isfinite(local.grad).all() and local.grad.abs().sum() > 0


@pytest.mark.parametrize('probability', [0., 1.])
def test_scheduled_probability_endpoints_select_only_the_requested_source(probability):
    _, b, identity, local, _ = fixture(frames=35)
    target, generated, noise, times, draws = training_inputs(b, local)
    draws.fill_(.5)
    model = RecordingLoss()
    _, used, available = runner.chunk_training_loss(model, b, identity, local, target, generated,
                                                  noise, times, draws, probability)
    assert used == (available if probability == 1. else 0)
    source = target if probability == 1. else generated
    for i, call in enumerate(model.calls):
        start = i * runner.CHUNK
        torch.testing.assert_close(call['history'], source[:, max(0, start-8):start], atol=0, rtol=0)


def test_training_loss_is_valid_frame_weighted_with_empty_and_partial_chunks():
    _, b, identity, local, _ = fixture(frames=35, batch=3)
    b['valid'][0, 16:32] = False
    b['valid'][1, :16] = False
    b['valid'][2, :32] = False
    target, generated, noise, times, draws = training_inputs(b, local)
    target[~b['valid']] = float('nan')
    model = RecordingLoss()
    loss, _, _ = runner.chunk_training_loss(model, b, identity, local, target, generated,
                                           noise, times, draws, .5)
    torch.testing.assert_close(loss, target[b['valid']].square().mean())
    assert [int(call['valid'].sum()) for call in model.calls] == [16, 16, 9]
    assert [len(call['valid']) for call in model.calls] == [1, 1, 3]
    loss.backward()
    torch.testing.assert_close(model.weight.grad, target[b['valid']].square().mean())


def test_no_history_training_ignores_scheduled_source_and_generated_values():
    upper, b, identity, local, _ = fixture(use_history=False)
    target, generated, noise, times, draws = training_inputs(b, local)
    first, _, _ = runner.chunk_training_loss(upper, b, identity, local, target, generated,
                                            noise, times, draws, 0.)
    second, _, _ = runner.chunk_training_loss(upper, b, identity, local, target,
                                             torch.full_like(generated, float('nan')),
                                             noise, times, draws, 1.)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    second.backward()
    assert all(p.grad is None for name, p in upper.named_parameters() if name.startswith('history_'))


def test_teacher_schedule_is_monotone_and_last_four_epochs_are_generated_only():
    probabilities = [runner.teacher_probability(epoch, 12) for epoch in range(12)]
    assert probabilities[0] == .75
    assert all(0 <= value <= .75 for value in probabilities)
    assert all(a >= b for a, b in zip(probabilities, probabilities[1:]))
    assert probabilities[-4:] == [0., 0., 0., 0.]


def test_boundary_metrics_select_the_actual_chunk_seam_and_exclude_mask_gaps():
    valid = torch.ones(2, 34, dtype=torch.bool)
    valid[1, 15] = False  # Exclude the seam 15 -> 16 for row 1.
    pred = torch.zeros(2, 34, 52)
    target = torch.zeros_like(pred)
    pred[:, 16:, runner.CC] += 2
    pred[:, 32:, runner.CC] += 4
    target[:, 16:, runner.CC] += 1
    target[:, 32:, runner.CC] += 3
    result = runner.boundary_report(pred, target, valid)
    for region in result.values():
        seam = region['chunk_boundary']
        assert seam['pairs'] == 3
        assert seam['pred_rms'] == pytest.approx(((2**2 + 4**2 + 4**2) / 3)**.5)
        assert seam['gt_rms'] == pytest.approx(((1**2 + 3**2 + 3**2) / 3)**.5)
        assert seam['displacement_mse'] == pytest.approx(1.)
        inside = region['inside_chunk']
        assert inside['pairs'] == 61
        assert inside['pred_rms'] == inside['gt_rms'] == inside['displacement_mse'] == 0.


def test_boundary_metrics_return_none_when_no_valid_seam_pairs_exist():
    valid = torch.ones(1, 12, dtype=torch.bool)
    result = runner.boundary_report(torch.zeros(1, 12, 52), torch.zeros(1, 12, 52), valid)
    for region in result.values():
        assert region['chunk_boundary'] is None
        assert region['inside_chunk']['pairs'] == 11
