"""Synthetic context-mechanism integration checks; no persisted data access."""
import copy

import pytest
import torch
from torch import nn

from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow
from scripts import train_context_mechanism as runner


CFG = {'model': {'content_dim': 4, 'emotion_dim': 3, 'style_dim': 2,
                 'dit_dim': 8, 'dit_depth': 1, 'heads': 2, 'dropout': 0.}}
ARMS = ('chunk_empty', 'chunk_teacher', 'whole')


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(382)
    valid = torch.ones(3, 96, dtype=torch.bool)
    valid[0, 11:13] = False
    valid[1, :16] = False
    valid[1, 21:25] = False
    valid[1, 32:48] = False
    valid[1, 90:] = False
    valid[2, :89] = False
    clock = torch.arange(96).float()[None, :, None]
    b = {'valid': valid, 'h0': clock.expand(3, -1, 4).clone(),
         'audio_global': torch.randn(3, 3), 'audio_intensity': torch.randn(3, 1),
         'motion': torch.randn(3, 96, 52), 'static_upper': torch.randn(3, 9),
         'emotion_id': torch.tensor([0, 1, 2]), 'speaker_id': torch.tensor([3, 5, 7])}
    identity = {'code': torch.stack((torch.arange(3).float(), torch.ones(3)), -1),
                'baseline': torch.zeros(3, 52)}
    native = torch.randn(3, 96, 3)
    scales = torch.linspace(.5, 2., 9)
    noise = torch.randn(3, 96, 9)
    for value in (b['motion'], b['h0'], native, noise):
        value[~valid] = float('nan')
    return b, identity, native, scales, noise


class RecordingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.loss_calls = []
        self.decode_calls = []

    def flow_loss_prefix(self, target, valid, h0, identity, affect, local, noise, time, *, known, known_mask):
        self.loss_calls.append({'target': target.clone(), 'valid': valid.clone(), 'h0': h0.clone(),
                               'identity': identity.clone(), 'local': local.clone(),
                               'noise': noise.clone(), 'time': time.clone(),
                               'known': known.clone(), 'known_mask': known_mask.clone()})
        selected = valid & ~known_mask
        return self.weight * target[selected].square().mean()

    def decode_prefix(self, valid, h0, identity, affect, local, noise, *, known, known_mask, steps):
        self.decode_calls.append({'valid': valid.clone(), 'h0': h0.clone(), 'identity': identity.clone(),
                                  'local': local.clone(), 'noise': noise.clone(), 'known': known.clone(),
                                  'known_mask': known_mask.clone()})
        mean = known.sum(1)/known_mask.sum(1)[:, None].clamp_min(1)
        generated = torch.where(valid[..., None], noise, 0.)+mean[:, None]
        return torch.where(known_mask[..., None], known, torch.where(valid[..., None], generated, 0.))


def execute(model, b, identity, native, scales, arm, seed=512):
    return runner.context_batch(model, b, identity, native, scales,
                                torch.Generator().manual_seed(seed), arm)


@pytest.mark.parametrize('arm', ARMS)
def test_all_arms_weight_each_valid_current_frame_once_with_internal_gaps(arm):
    b, identity, native, scales, _ = fixture()
    model = RecordingFlow()
    loss, _ = execute(model, b, identity, native, scales, arm)
    target = runner.p.h.normalized_target(b, scales)
    torch.testing.assert_close(loss, target[b['valid']].square().mean())
    assert sum(int((call['valid'] & ~call['known_mask']).sum()) for call in model.loss_calls) == int(b['valid'].sum())
    assert len(model.loss_calls) == (1 if arm == 'whole' else 6)
    loss.backward()
    torch.testing.assert_close(model.weight.grad, target[b['valid']].square().mean())


def test_common_native_noise_and_per_clip_time_draws_match_for_all_three_arms():
    b, identity, native, scales, _ = fixture()
    outputs, states = [], []
    for arm in ARMS:
        generator = torch.Generator().manual_seed(117)
        model = RecordingFlow()
        loss, randoms = runner.context_batch(model, b, identity, native, scales, generator, arm)
        assert [tuple(value.shape) for value in randoms] == [(3, 96, 9), (3,)]
        noise, times = randoms
        for index, call in enumerate(model.loss_calls):
            ids = call['identity'][:, 0].long()
            start, count = (0, 96) if arm == 'whole' else (index*16, 16)
            selected = call['valid'][:, 8:8+count]
            torch.testing.assert_close(call['noise'][:, 8:8+count][selected], noise[ids, start:start+count][selected], atol=0, rtol=0)
            assert call['noise'][:, :8].count_nonzero() == 0
            torch.testing.assert_close(call['time'], times[ids], atol=0, rtol=0)
        outputs.append(randoms); states.append(generator.get_state())
    for other in outputs[1:]:
        for left, right in zip(outputs[0], other):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert all(torch.equal(states[0], state) for state in states[1:])


def test_teacher_training_supplies_detached_strictly_previous_observed_targets_only():
    b, identity, native, scales, _ = fixture()
    b['motion'].requires_grad_()
    model = RecordingFlow()
    loss, _ = execute(model, b, identity, native, scales, 'chunk_teacher')
    target = runner.p.h.normalized_target(b, scales)
    for start, call in zip(range(0, 96, 16), model.loss_calls):
        ids = call['identity'][:, 0].long()
        assert not call['known'].requires_grad
        assert not call['known_mask'][:, 8:].any() and not call['known'][:, 8:].any()
        for row, sample in enumerate(ids):
            expected_mask = torch.zeros(24, dtype=torch.bool)
            expected_known = torch.zeros(24, 9)
            if start:
                expected_mask[:8] = b['valid'][sample, start-8:start]
                expected_known[:8] = torch.where(expected_mask[:8, None], target[sample, start-8:start], 0.)
            assert torch.equal(call['known_mask'][row], expected_mask)
            torch.testing.assert_close(call['known'][row], expected_known, atol=0, rtol=0)
    loss.backward()
    assert torch.isfinite(b['motion'].grad).all()
    assert b['motion'].grad[~b['valid']].count_nonzero() == 0


@pytest.mark.parametrize('arm', ('chunk_empty', 'whole'))
def test_no_teacher_arms_have_zero_known_motion_and_eight_invalid_initial_slots(arm):
    b, identity, native, scales, _ = fixture()
    model = RecordingFlow()
    execute(model, b, identity, native, scales, arm)
    for call in model.loss_calls:
        assert not call['known_mask'].any() and not call['known'].any()
        assert not call['valid'][:, :8].any()
        assert not call['h0'][:, :8].any() and not call['local'][:, :8].any()
    if arm == 'whole':
        call = model.loss_calls[0]
        assert call['valid'].shape == (3, 104)
        assert torch.equal(call['valid'][:, 8:], b['valid'])
        torch.testing.assert_close(call['h0'][:, 8:][b['valid']], b['h0'][b['valid']], atol=0, rtol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_deploy_decode_does_not_read_query_motion_emotion_or_static_target_fields(arm):
    b, identity, native, _, noise = fixture()
    expected = runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1, arm=arm)
    changed = copy.deepcopy(b)
    changed['motion'].fill_(float('nan'))
    changed['emotion_id'].fill_(999)
    changed['static_upper'].fill_(float('nan'))
    actual = runner.decode_context(RecordingFlow(), changed, identity, native, noise, steps=1, arm=arm)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (3, 96, 9) and torch.isfinite(actual).all()
    assert not actual.requires_grad and not actual[~b['valid']].any()


@pytest.mark.parametrize('arm', ('chunk_empty', 'whole'))
def test_no_motion_history_decode_is_invariant_to_history_interventions(arm):
    b, identity, native, _, noise = fixture()
    full = runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1, arm=arm)
    for mode in ('empty', 'reverse_history', 'oracle_history', 'oracle_reverse_history'):
        kwargs = {'oracle_target': torch.full_like(noise, float('nan'))} if mode.startswith('oracle_') else {}
        value = runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1, mode=mode, arm=arm, **kwargs)
        torch.testing.assert_close(value, full, atol=0, rtol=0)


def test_teacher_arm_oracle_condition_reads_only_valid_strict_past_at_decode():
    b, identity, native, _, noise = fixture()
    target = torch.arange(96).float()[None, :, None].expand(3, -1, 9).clone()
    target[~b['valid']] = float('nan')
    model = RecordingFlow()
    expected = runner.decode_context(model, b, identity, native, noise, steps=1,
                                    arm='chunk_teacher', mode='oracle_history', oracle_target=target)
    for start, call in zip(range(0, 96, 16), model.decode_calls):
        assert not call['known_mask'][:, 8:].any() and not call['known'][:, 8:].any()
        ids = call['identity'][:, 0].long()
        if start:
            good = call['known_mask'][:, :8]
            torch.testing.assert_close(call['known'][:, :8][good], target[ids, start-8:start][good], atol=0, rtol=0)
    changed = target.clone(); changed[:, 80:] = float('nan')
    result = runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1,
                                  arm='chunk_teacher', mode='oracle_history', oracle_target=changed)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_oracle_reverse_reverses_only_observed_past_motion_leaving_audio_clock_and_masks_fixed():
    b, identity, native, _, noise = fixture()
    target = torch.arange(96).float()[None, :, None].expand(3, -1, 9).clone()
    target[~b['valid']] = float('nan')
    models = {mode: RecordingFlow() for mode in ('oracle_history', 'oracle_reverse_history')}
    for mode, model in models.items():
        runner.decode_context(model, b, identity, native, noise, steps=1,
                              arm='chunk_teacher', mode=mode, oracle_target=target)
    for ordinary, reverse in zip(models['oracle_history'].decode_calls, models['oracle_reverse_history'].decode_calls):
        assert torch.equal(ordinary['known_mask'], reverse['known_mask'])
        assert torch.equal(ordinary['valid'], reverse['valid'])
        torch.testing.assert_close(ordinary['h0'], reverse['h0'], atol=0, rtol=0)
        torch.testing.assert_close(ordinary['local'], reverse['local'], atol=0, rtol=0)
        for row in range(len(ordinary['valid'])):
            good = ordinary['known_mask'][row]
            torch.testing.assert_close(reverse['known'][row, good], ordinary['known'][row, good].flip(0), atol=0, rtol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_minimum_audio_only_condition_mapping_is_sufficient_for_normal_decode(arm):
    b, identity, native, _, noise = fixture()
    small = {key: b[key] for key in ('valid', 'h0', 'audio_global', 'audio_intensity')}
    expected = runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1, arm=arm)
    actual = runner.decode_context(RecordingFlow(), small, identity, native, noise, steps=1, arm=arm)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_train_batch_is_rejected_without_consuming_random_stream():
    b, identity, native, scales, _ = fixture()
    b['valid'][0].fill_(False)
    generator = torch.Generator().manual_seed(77); before = generator.get_state()
    with pytest.raises(ValueError, match='observed 96-frame'):
        runner.context_batch(RecordingFlow(), b, identity, native, scales, generator, 'whole')
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize('arm', ARMS)
def test_deploy_modes_reject_extra_oracle_target_and_unknown_modes(arm):
    b, identity, native, _, noise = fixture()
    with pytest.raises(ValueError, match='rejects GT'):
        runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1,
                              arm=arm, oracle_target=noise)
    with pytest.raises(ValueError, match='Unknown'):
        runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1, arm=arm, mode='typo')
    if arm == 'chunk_teacher':
        for mode in ('oracle_history', 'oracle_reverse_history'):
            with pytest.raises(ValueError, match='explicit target'):
                runner.decode_context(RecordingFlow(), b, identity, native, noise, steps=1, arm=arm, mode=mode)


@pytest.mark.parametrize('arm', ARMS)
def test_acoustic_interventions_preserve_native_valid_mask_and_change_observed_conditions(arm):
    b, identity, native, _, noise = fixture()
    models = {mode: RecordingFlow() for mode in ('full', 'static', 'reverse')}
    for mode, model in models.items():
        runner.decode_context(model, b, identity, native, noise, steps=1, arm=arm, mode=mode)
    for mode in ('static', 'reverse'):
        changed = False
        for baseline, value in zip(models['full'].decode_calls, models[mode].decode_calls):
            assert torch.equal(baseline['valid'], value['valid'])
            valid = value['valid']
            assert torch.isfinite(value['h0'][valid]).all() and torch.isfinite(value['local'][valid]).all()
            changed |= not torch.equal(baseline['h0'][valid], value['h0'][valid])
            changed |= not torch.equal(baseline['local'][valid], value['local'][valid])
        assert changed


@pytest.mark.parametrize('arm', ARMS)
def test_real_model_backward_and_decode_are_finite_with_nan_padding(arm):
    b, identity, native, scales, noise = fixture()
    model = PrefixUpperFlow(CFG).eval()
    loss, _ = execute(model, b, identity, native, scales, arm)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
    assert sum(float(gradient.abs().sum()) for gradient in grads) > 0
    decoded = runner.decode_context(model, b, identity, native, noise, steps=1, arm=arm)
    assert torch.isfinite(decoded).all() and not decoded[~b['valid']].any()
