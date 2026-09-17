"""Formal prefix evaluation contracts using synthetic clips only."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import evaluate_prefix_formal as e


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ExplicitDecoder(nn.Module):
    """Deterministic row-local decoder, retaining supplied known slots exactly."""
    def decode_prefix(self, valid, h0, identity, affect, local, noise, *, known, known_mask, steps):
        past = torch.where(known_mask[..., None], known, 0.)
        mean = past.sum(1) / known_mask.sum(1)[:, None].clamp_min(1)
        current = noise * .08 + mean[:, None] * .2 + local[..., :1] * .1
        return torch.where(known_mask[..., None], known, torch.where(valid[..., None], current, 0.))


class MotionTeacher(nn.Module):
    def encode_motion(self, residual, valid):
        assert torch.isfinite(residual).all()
        return {'emotion_logits': residual.new_tensor([[1., 0.]]).expand(len(valid), -1)}


def fixture(count=5, frames=40, batch_size=2):
    generator = torch.Generator().manual_seed(441)
    clock = torch.arange(frames, dtype=torch.float64)[None].expand(count, -1) / 25
    motion = .5 + .1 * torch.randn(count, frames, 52, generator=generator)
    valid = torch.ones(count, frames, dtype=torch.bool)
    valid[0, -3:] = False
    if count > 1:
        valid[1, 9] = False
    if count > 2:
        valid[2, 15] = False
    q = {'motion': motion, 'valid': valid,
         'h0': torch.randn(count, frames, 4, generator=generator),
         'prefix_local': torch.randn(count, frames, 3, generator=generator),
         'audio_global': torch.randn(count, 3, generator=generator),
         'audio_intensity': torch.rand(count, 1, generator=generator),
         'static_upper': torch.full((count, 9), .5),
         'clip_id': [f'synthetic_{i}' for i in range(count)], 'times': clock,
         'channel_mask': torch.ones(count, 52, dtype=torch.bool),
         'b0': torch.zeros_like(motion), 'emotion_id': (torch.arange(count) >= 2).long(),
         'speaker_id': torch.arange(count)}
    identities = {i: {'code': torch.ones(1, 2) * i, 'baseline': torch.zeros(1, 52)} for i in range(count)}
    bases = {key: copy.deepcopy(q[key]) for key in ('clip_id', 'times', 'valid', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    bases['target'] = motion.clone()
    bases['predictions'] = {}
    for seed in e.p.r.SEEDS:
        value = motion + seed / 100000.
        value[..., 0] = -0.
        value[~valid] = float('nan')
        bases['predictions'][f'{seed}/base'] = value
    return ExplicitDecoder().eval(), MotionTeacher().eval(), q, identities, torch.ones(9), bases, SimpleNamespace(device='cpu', batch_size=batch_size)


def run(data, *, use_prefix=True):
    return e.evaluate_formal(*data, steps=1, use_prefix=use_prefix)


def same_bits(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def test_global_noise_arrays_are_drawn_before_slicing_and_shared_by_modes(monkeypatch):
    data = fixture()
    count, frames = data[2]['valid'].shape
    real_randn, real_rollout = torch.randn, e.p.rollout
    draws, calls = [], []

    def randn(*shape, **kwargs):
        value = real_randn(*shape, **kwargs)
        draws.append((tuple(value.shape), kwargs['generator'].initial_seed(), value.clone()))
        return value

    def rollout(upper, b, identity, local, noise, **kwargs):
        assert len(draws) == 3
        calls.append((kwargs['mode'], list(b['clip_id']), noise.clone()))
        return real_rollout(upper, b, identity, local, noise, **kwargs)

    monkeypatch.setattr(e.torch, 'randn', randn)
    monkeypatch.setattr(e.p, 'rollout', rollout)
    run(data)
    assert [(shape, seed) for shape, seed, _ in draws] == [((count, frames, 9), seed) for seed in e.p.r.SEEDS]
    batches = (count + data[-1].batch_size - 1) // data[-1].batch_size
    assert len(calls) == 8 * batches
    offset = 0
    for seed, modes in ((42, e.MODES), (123, ('full',)), (2026, ('full',))):
        expected = next(value for _, value_seed, value in draws if value_seed == seed)
        for mode in modes:
            selected = calls[offset:offset+batches]
            assert all(name == mode for name, _, _ in selected)
            assert sum((ids for _, ids, _ in selected), []) == data[2]['clip_id']
            torch.testing.assert_close(torch.cat([value for _, _, value in selected]), expected, atol=0, rtol=0)
            offset += batches


def test_batch_size_does_not_change_predictions_or_aggregate_metrics():
    first = fixture(batch_size=2)
    second = copy.deepcopy(first)
    second[-1].batch_size = 5
    first_report, first_curves = run(first)
    second_report, second_curves = run(second)
    assert first_report == second_report
    for key in first_curves['predictions']:
        assert same_bits(first_curves['predictions'][key], second_curves['predictions'][key])
    assert first_report['modes']['42/full']['generated_emotion_accuracy_nonindependent'] == .4
    prediction = first_curves['predictions']['42/full'][..., list(e.p.GROUPS['brows'])]
    target = first[2]['motion'][..., list(e.p.GROUPS['brows'])]
    mask = first[2]['valid'][..., None].expand_as(prediction)
    expected = float((prediction.double()[mask]-target.double()[mask]).square().mean())
    actual = first_report['modes']['42/full']['populations']['all']['brows']['raw_mse']
    assert actual == pytest.approx(expected)


def test_all_modes_copy_nonupper_and_invalid_baseline_bit_exactly():
    data = fixture()
    report, curves = run(data)
    assert report['nonupper_exact'] and report['invalid_baseline_exact']
    for key, pred in curves['predictions'].items():
        base = data[5]['predictions'][key.split('/')[0]+'/base']
        assert same_bits(pred[..., list(e.p.r.NOT_UPPER)], base[..., list(e.p.r.NOT_UPPER)])
        assert same_bits(pred[~data[2]['valid']], base[~data[2]['valid']])
        assert torch.signbit(pred[..., 0][data[2]['valid']]).all()
        assert torch.isnan(pred[~data[2]['valid']]).all()
    json.dumps(report, allow_nan=False)


def test_oracle_is_separate_from_all_deployment_aggregation():
    report, curves = run(fixture())
    assert len(curves['predictions']) == 8 and len(report['modes']) == 7
    assert '42/oracle_history' not in report['modes']
    assert '42/oracle_history' not in report['deployable_interventions_seed42']
    assert 'single_seed_interventions' not in report['distribution']
    diagnostic = report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']
    assert set(diagnostic['modes']) == {'42/oracle_history'}
    assert diagnostic['modes']['42/oracle_history']['oracle_target_history'] is True
    assert curves['oracle_prediction_keys'] == ['42/oracle_history']
    assert report['test_loaded'] is False and report['default_replaced'] is False
    for mode in ('empty', 'reverse_history', 'static', 'reverse'):
        detail = report['modes']['42/'+mode]['actual_prefix_continuation']
        assert detail['scored'] is False and detail['metrics'] is None and detail['note']


def test_deployable_generation_ignores_query_targets_but_oracle_reads_strict_past():
    data = fixture()
    _, original = run(data)
    changed = copy.deepcopy(data)
    changed[2]['motion'] += 2.
    changed[2]['emotion_id'] = 1-changed[2]['emotion_id']
    changed[5]['target'] = changed[2]['motion'].clone()
    changed[5]['emotion_id'] = changed[2]['emotion_id'].clone()
    _, modified = run(changed)
    for key, value in original['predictions'].items():
        if key != '42/oracle_history':
            assert same_bits(value, modified['predictions'][key])
    oracle = '42/oracle_history'
    assert same_bits(original['predictions'][oracle][:, :16], modified['predictions'][oracle][:, :16])
    assert not same_bits(original['predictions'][oracle][:, 16:], modified['predictions'][oracle][:, 16:])


def test_supplied_prefix_metrics_use_gt_for_oracle_and_generated_past_for_full():
    data = fixture()
    report, curves = run(data)
    q = data[2]
    for mode in ('full', 'oracle_history'):
        records = report['modes'] if mode == 'full' else report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']
        pred = curves['predictions']['42/'+mode]
        past = pred if mode == 'full' else q['motion']
        expected = e.actual_prefix_continuation(pred, q['motion'], q['valid'], q['channel_mask'], past)
        detail = records['42/'+mode]['actual_prefix_continuation']
        assert detail['metrics'] == expected and detail['scored']
        assert detail['source'] == ('generated_past' if mode == 'full' else 'tracked_reference_GT')
    oracle = report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']['42/oracle_history']
    assert oracle['actual_prefix_continuation']['metrics']['brows']['rms'] != oracle['boundaries']['brows']['chunk_boundary']['pred_rms']


def test_actual_prefix_scoring_distinguishes_oracle_endpoint_from_stitched_seam():
    target = torch.arange(40).float()[None, :, None].expand(1, -1, 52).clone()
    pred = target + 5
    valid = torch.ones(1, 40, dtype=torch.bool)
    channels = torch.ones(1, 52, dtype=torch.bool)
    full = e.actual_prefix_continuation(pred, target, valid, channels, pred)
    oracle = e.actual_prefix_continuation(pred, target, valid, channels, target)
    assert full['brows']['rms'] == 1 and full['brows']['displacement_mse'] == 0
    assert oracle['brows']['rms'] == 6 and oracle['brows']['displacement_mse'] == 25
    assert oracle['brows']['reference_rms'] == 1


def test_actual_prefix_scoring_respects_gaps_channels_and_empty_pair_coverage():
    target = torch.arange(40).float()[None, :, None].expand(1, -1, 52).clone()
    valid = torch.ones(1, 40, dtype=torch.bool)
    channels = torch.ones(1, 52, dtype=torch.bool)
    valid[:, 15] = False
    channels[:, e.p.GROUPS['brows'][0]] = False
    result = e.actual_prefix_continuation(target, target, valid, channels, target)
    assert result['brows']['observed_channel_pairs'] == 4
    assert result['eyes_expression']['observed_channel_pairs'] == 4
    valid[:, 31] = False
    result = e.actual_prefix_continuation(target, target, valid, channels, target)
    for group in result.values():
        assert group == {'observed_channel_pairs': 0, 'rms': None, 'reference_rms': None, 'displacement_mse': None}


def test_no_prefix_history_modes_equal_and_have_no_supplied_endpoint_score():
    report, curves = run(fixture(), use_prefix=False)
    assert report['no_prefix_history_interventions_equal']
    for mode in ('empty', 'reverse_history', 'oracle_history'):
        assert same_bits(curves['predictions']['42/full'], curves['predictions']['42/'+mode])
    for records in (report['modes'], report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']):
        for row in records.values():
            assert row['actual_prefix_continuation']['scored'] is False
            assert row['actual_prefix_continuation']['metrics'] is None


def test_same_gt_endpoint_is_marked_as_output_diagnostic_even_for_no_prefix():
    data = fixture()
    report, curves = run(data, use_prefix=False)
    row = report['modes']['42/full']
    assert row['actual_prefix_continuation']['scored'] is False
    diagnostic = row['same_gt_endpoint_output_diagnostic']
    assert diagnostic['GT_was_supplied_as_history'] is False
    expected = e.actual_prefix_continuation(curves['predictions']['42/full'], data[2]['motion'],
                                            data[2]['valid'], data[2]['channel_mask'], data[2]['motion'])
    assert diagnostic['metrics'] == expected
    assert report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']['42/oracle_history']['same_gt_endpoint_output_diagnostic']['GT_was_supplied_as_history'] is False
    report, _ = run(data)
    oracle = report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']['42/oracle_history']
    assert oracle['same_gt_endpoint_output_diagnostic']['GT_was_supplied_as_history'] is True
    assert oracle['same_gt_endpoint_output_diagnostic']['metrics'] == oracle['actual_prefix_continuation']['metrics']


def test_chunk_profiles_report_raw_mean_first_native_frames_and_within_block_speed():
    target = torch.arange(35).float()[None, :, None].expand(2, -1, 52).clone()
    pred = target + 3
    valid = torch.ones(2, 35, dtype=torch.bool)
    valid[1, 16:] = False
    valid[0, 16] = False
    channel_mask = torch.ones(2, 52, dtype=torch.bool)
    pred[~valid] = float('nan')
    profile = e.chunk_diagnostics(pred, target, valid, channel_mask)
    assert profile['common_clip_coverage']['clip_indices'] == [0]
    assert profile['common_clip_coverage']['clip_count'] == 1
    rows = profile['all_observed']['brows']
    assert [row['start_frame'] for row in rows] == [0, 16, 32]
    assert rows[0]['reference_raw_mean'] == 7.5
    assert rows[0]['prediction_raw_mean'] == 10.5
    for row in rows:
        assert row['raw_mse'] == 9 and row['mean_error_rms'] == 3
        assert row['within_chunk_displacement']['prediction_rms'] == 1
        assert row['within_chunk_displacement']['reference_rms'] == 1
        assert row['within_chunk_displacement']['displacement_mse'] == 0
        assert row['early_frame_errors']['first_2_native_frames']['raw_mse'] == 9
        assert row['early_frame_errors']['first_4_native_frames']['raw_mse'] == 9
    # Native frame 16 is invalid: the first-two score uses only frame17 and
    # never compresses the clock to pull frame18 into the early window.
    assert rows[1]['early_frame_errors']['first_2_native_frames']['observed_values'] == 5
    assert rows[1]['early_frame_errors']['first_4_native_frames']['observed_values'] == 15
    assert rows[1]['within_chunk_displacement']['observed_channel_pairs'] == 14*5
    assert rows[-1]['early_frame_errors']['first_4_native_frames']['observed_values'] == 3*5
    json.dumps(profile, allow_nan=False)


def test_chunk_profiles_with_no_common_coverage_use_null_metrics():
    target = torch.zeros(1, 35, 52)
    valid = torch.ones(1, 35, dtype=torch.bool)
    valid[:, 16:] = False
    channel_mask = torch.ones(1, 52, dtype=torch.bool)
    result = e.chunk_diagnostics(target, target, valid, channel_mask)
    assert result['common_clip_coverage']['clip_count'] == 0
    for name, rows in result['common_clip_coverage']['profiles'].items():
        assert len(rows) == 3
        for row in rows:
            assert row['observed_values'] == 0
            assert row['raw_mse'] is None and row['prediction_raw_mean'] is None
            assert row['within_chunk_displacement']['prediction_rms'] is None
            assert row['early_frame_errors']['first_2_native_frames']['raw_mse'] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('mutation,match', [
    ('missing_seed', 'baseline'), ('wrong_metadata', 'metadata'), ('wrong_order', 'order'),
    ('bad_clock', 'clock'), ('bad_scales', 'scales'), ('unobserved_upper', 'observed upper'),
    ('duplicate_id', 'unique IDs'), ('bad_baseline', 'baseline'), ('train_mode', 'eval mode'),
])
def test_invalid_evaluation_inputs_fail_before_generation(mutation, match, monkeypatch):
    data = list(fixture())
    if mutation == 'missing_seed': del data[5]['predictions']['123/base']
    elif mutation == 'wrong_metadata': data[5]['target'][0, 0, 0] += 1
    elif mutation == 'wrong_order': data[5]['clip_id'].reverse()
    elif mutation == 'bad_clock': data[2]['times'] = data[2]['times'] * 2
    elif mutation == 'bad_scales': data[4][0] = 0
    elif mutation == 'unobserved_upper': data[2]['channel_mask'][0, e.p.CC[0]] = False
    elif mutation == 'duplicate_id': data[2]['clip_id'][1] = data[2]['clip_id'][0]
    elif mutation == 'bad_baseline': data[5]['predictions']['42/base'][0, 0, 0] = float('nan')
    elif mutation == 'train_mode': data[0].train()

    def fail_if_called(*args, **kwargs):
        pytest.fail('Generation must not run for invalid evaluation inputs')

    monkeypatch.setattr(e.p, 'rollout', fail_if_called)
    with pytest.raises(ValueError, match=match):
        run(data)


def test_history_intervention_first_chunk_invariant_is_enforced(monkeypatch):
    data = fixture()
    real_rollout = e.p.rollout

    def bad_rollout(*args, **kwargs):
        value = real_rollout(*args, **kwargs)
        if kwargs['mode'] == 'empty': value[:, 0] += 1
        return value

    monkeypatch.setattr(e.p, 'rollout', bad_rollout)
    with pytest.raises(RuntimeError, match='first chunk'):
        run(data)
