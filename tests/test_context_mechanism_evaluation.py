"""Synthetic evaluator checks; no saved train/development/test data are read."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import evaluate_context_mechanism as e


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Teacher(nn.Module):
    def encode_motion(self, values, valid):
        assert torch.isfinite(values).all()
        return {'emotion_logits': values.new_tensor([[1., 0.]]).expand(len(valid), -1)}


def fixture(count=5, frames=40, batch_size=2, smoke=False):
    generator = torch.Generator().manual_seed(130)
    valid = torch.ones(count, frames, dtype=torch.bool)
    valid[0, -3:] = False
    if count > 1: valid[1, 9] = False
    q = {'motion': .5 + .1 * torch.randn(count, frames, 52, generator=generator),
         'valid': valid, 'h0': torch.randn(count, frames, 4, generator=generator),
         'prefix_local': torch.randn(count, frames, 3, generator=generator),
         'audio_global': torch.randn(count, 3, generator=generator),
         'audio_intensity': torch.rand(count, 1, generator=generator),
         'static_upper': torch.full((count, 9), .5),
         'clip_id': [f'synthetic_{i}' for i in range(count)],
         'times': torch.arange(frames, dtype=torch.float64)[None].expand(count, -1)/25,
         'channel_mask': torch.ones(count, 52, dtype=torch.bool),
         'b0': torch.zeros(count, frames, 52), 'emotion_id': (torch.arange(count) >= 2).long(),
         'speaker_id': torch.arange(count)}
    identities = {i: {'code': torch.ones(1, 2)*i, 'baseline': torch.zeros(1, 52)} for i in range(count)}
    bases = {key: copy.deepcopy(q[key]) for key in ('clip_id', 'times', 'valid', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    bases.update(target=q['motion'].clone(), predictions={})
    for seed in e.p.r.SEEDS:
        value = q['motion'] + seed/100000.
        value[..., 0] = -0.
        value[~valid] = float('nan')
        bases['predictions'][f'{seed}/base'] = value
    return (nn.Identity().eval(), Teacher().eval(), q, identities, torch.ones(9), bases,
            SimpleNamespace(device='cpu', batch_size=batch_size, smoke=smoke))


def rollout(upper, b, identity, local, noise, *, steps, mode='full', arm, oracle_target=None):
    assert 'motion' not in b and 'emotion_id' not in b and 'b0' not in b
    assert (oracle_target is not None) == (arm == 'chunk_teacher' and mode in e.ORACLE_MODES)
    native = local
    if mode == 'static': native = local.mean(1, keepdim=True).expand_as(local)
    elif mode == 'reverse': native = local.flip(1)
    generated = .1*noise + .1*native[..., :1]
    if arm == 'whole':
        generated = generated + .1*noise.mean(1, keepdim=True)
    if arm == 'chunk_teacher' and mode != 'empty':
        for start in range(16, noise.shape[1], 16):
            source = oracle_target if mode in e.ORACLE_MODES else generated
            past = source[:, start-8:start]
            if mode in ('reverse_history', 'oracle_reverse_history'): past = past.flip(1)
            generated[:, start:start+16] += .2*past[:, -1:]
    return torch.where(b['valid'][..., None], generated, 0.)


def run(data, arm='chunk_teacher', rollout_fn=rollout):
    return e.evaluate(*data, steps=1, arm=arm, rollout_fn=rollout_fn)


def without_time(report):
    result = copy.deepcopy(report)
    result.pop('evaluation_seconds')
    result.pop('mode_seconds_including_scoring')
    return result


@pytest.mark.parametrize('arm', e.ARMS)
def test_all_arms_have_all_seeds_separate_oracle_and_finite_time(arm):
    report, curves = run(fixture(), arm)
    assert report['schema'] == curves['schema'] == e.SCHEMA
    assert report['arm'] == curves['arm'] == arm
    assert curves['noise_seeds'] == [42, 123, 2026]
    assert len(curves['predictions']) == 9 and len(report['modes']) == 7
    for mode in e.ORACLE_MODES:
        assert '42/'+mode not in report['modes']
        assert '42/'+mode not in report['deployable_interventions_seed42']
    assert 'single_seed_interventions' not in report['distribution']
    diagnostic = report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']
    assert set(diagnostic['modes']) == set(diagnostic['paired_metrics_vs_full']) == {'42/'+mode for mode in e.ORACLE_MODES}
    assert diagnostic['active_target_conditioning'] == (arm == 'chunk_teacher')
    assert report['evaluation_seconds'] > 0
    assert set(report['mode_seconds_including_scoring']) == set(curves['predictions'])
    assert all(value > 0 for value in report['mode_seconds_including_scoring'].values())
    assert report['test_loaded'] is report['default_replaced'] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('arm', e.ARMS)
def test_all_modes_preserve_frozen_channels_and_invalid_nan_and_negative_zero(arm):
    data = fixture()
    report, curves = run(data, arm)
    assert report['nonupper_exact'] and report['invalid_baseline_exact']
    for key, value in curves['predictions'].items():
        baseline = data[5]['predictions'][key.split('/')[0]+'/base']
        assert e._same_bits(value[..., list(e.p.r.NOT_UPPER)], baseline[..., list(e.p.r.NOT_UPPER)])
        assert e._same_bits(value[~data[2]['valid']], baseline[~data[2]['valid']])
        assert torch.signbit(value[..., 0][data[2]['valid']]).all()


def test_all_global_noise_is_drawn_first_shared_modes_and_stable_across_batches(monkeypatch):
    data = fixture()
    real_randn = torch.randn
    draws, calls = [], []

    def randn(*shape, **kwargs):
        value = real_randn(*shape, **kwargs)
        draws.append((kwargs['generator'].initial_seed(), value.clone()))
        return value

    def spy(upper, b, identity, local, noise, **kwargs):
        assert len(draws) == 3
        calls.append((kwargs['mode'], list(b['clip_id']), noise.clone()))
        return rollout(upper, b, identity, local, noise, **kwargs)

    monkeypatch.setattr(e.torch, 'randn', randn)
    run(data, rollout_fn=spy)
    assert [seed for seed, _ in draws] == [42, 123, 2026]
    assert all(value.shape == (5, 40, 9) for _, value in draws)
    cursor = 0
    for seed, modes in ((42, e.MODES), (123, ('full',)), (2026, ('full',))):
        expected = next(value for value_seed, value in draws if value_seed == seed)
        for mode in modes:
            selected = calls[cursor:cursor+3]
            assert [name for name, _, _ in selected] == [mode]*3
            assert sum([ids for _, ids, _ in selected], []) == data[2]['clip_id']
            torch.testing.assert_close(torch.cat([value for _, _, value in selected]), expected, atol=0, rtol=0)
            cursor += 3


@pytest.mark.parametrize('arm', e.ARMS)
def test_batch_size_invariance_and_correct_global_accuracy(arm):
    data = fixture()
    first_report, first_curves = run(data, arm)
    data[-1].batch_size = 5
    second_report, second_curves = run(data, arm)
    assert without_time(first_report) == without_time(second_report)
    for key, value in first_curves['predictions'].items():
        assert e._same_bits(value, second_curves['predictions'][key])
    assert first_report['modes']['42/full']['generated_emotion_accuracy_nonindependent'] == .4


@pytest.mark.parametrize('arm', ('chunk_empty', 'whole'))
def test_history_free_arms_never_receive_gt_and_all_history_interventions_are_noops(arm):
    report, curves = run(fixture(), arm)
    assert report['history_free_arm_interventions_equal'] is True
    assert curves['oracle_has_target_input'] is False
    for mode in ('empty', 'reverse_history') + e.ORACLE_MODES:
        assert e._same_bits(curves['predictions']['42/'+mode], curves['predictions']['42/full'])
    for row in list(report['modes'].values()) + list(report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes'].values()):
        assert row['actual_prefix_continuation']['scored'] is False
        assert row['actual_prefix_continuation']['metrics'] is None
        assert row['same_gt_endpoint_output_diagnostic']['GT_was_supplied_as_history'] is False
        assert row['oracle_target_history'] is False
        assert row['boundaries_are_decoder_stitches'] == (arm != 'whole')
        if arm == 'whole': assert 'not decoder seams' in row['boundary_definition']


def test_teacher_arm_scores_generated_and_oracle_actual_endpoints_separately():
    data = fixture()
    report, curves = run(data)
    full = report['modes']['42/full']
    oracle = report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']['42/oracle_history']
    assert full['actual_prefix_continuation']['source'] == 'generated_past'
    assert oracle['actual_prefix_continuation']['source'] == 'tracked_reference_GT'
    assert oracle['actual_prefix_continuation']['metrics'] == oracle['same_gt_endpoint_output_diagnostic']['metrics']
    assert oracle['same_gt_endpoint_output_diagnostic']['GT_was_supplied_as_history']
    assert not full['same_gt_endpoint_output_diagnostic']['GT_was_supplied_as_history']
    for mode, row in (('full', full), ('oracle_history', oracle)):
        pred = curves['predictions']['42/'+mode]
        expected = e.actual_prefix_continuation(pred, data[2]['motion'], data[2]['valid'], data[2]['channel_mask'],
                                                data[2]['motion'] if mode == 'oracle_history' else pred)
        assert row['actual_prefix_continuation']['metrics'] == expected


@pytest.mark.parametrize('arm', e.ARMS)
def test_query_target_changes_cannot_change_any_deployment_prediction(arm):
    data = fixture()
    _, expected = run(data, arm)
    modified = copy.deepcopy(data)
    modified[2]['motion'] += 10
    modified[2]['emotion_id'] = 1-modified[2]['emotion_id']
    modified[5]['target'] = modified[2]['motion'].clone()
    modified[5]['emotion_id'] = modified[2]['emotion_id'].clone()
    _, actual = run(modified, arm)
    for key in expected['predictions']:
        if arm == 'chunk_teacher' and key in {'42/'+mode for mode in e.ORACLE_MODES}:
            assert e._same_bits(expected['predictions'][key][:, :16], actual['predictions'][key][:, :16])
            assert not e._same_bits(expected['predictions'][key][:, 16:], actual['predictions'][key][:, 16:])
        else:
            assert e._same_bits(expected['predictions'][key], actual['predictions'][key])


def test_smoke_retains_all_three_global_seeds_and_uses_all_supplied_rows():
    report, curves = run(fixture(count=32, frames=20, batch_size=7, smoke=True), 'whole')
    assert report['smoke'] is True and report['clips'] == 32
    assert len(curves['clip_id']) == 32 and curves['noise_seeds'] == [42, 123, 2026]


@pytest.mark.parametrize('arm,frame,error', [('chunk_teacher', 0, 'first chunk'),
                                           ('chunk_empty', 16, 'History-free'), ('whole', 16, 'History-free')])
def test_history_invariants_are_enforced(arm, frame, error):
    def corrupt(*args, **kwargs):
        value = rollout(*args, **kwargs)
        if kwargs['mode'] == 'empty': value[:, frame] += 1
        return value

    with pytest.raises(RuntimeError, match=error):
        run(fixture(), arm, corrupt)


def test_input_contract_rejects_unknown_arm_bad_rollout_and_mismatched_metadata():
    data = fixture()
    with pytest.raises(ValueError, match='arm'): run(data, 'unknown')
    with pytest.raises(ValueError, match='callable'): run(data, rollout_fn=None)
    data[5]['times'] = data[5]['times'] + .04
    with pytest.raises(ValueError, match='metadata'): run(data)


def test_complete_trajectory_shape_is_required():
    def truncated(*args, **kwargs): return rollout(*args, **kwargs)[:, :16]

    with pytest.raises(ValueError, match='complete'):
        run(fixture(), rollout_fn=truncated)


def test_reversed_gt_oracle_scores_actual_reversed_endpoint_separately_from_same_gt():
    data = fixture()
    report, curves = run(data)
    diagnostics = report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']
    row = diagnostics['modes']['42/oracle_reverse_history']
    assert row['oracle_target_history'] and row['same_gt_endpoint_output_diagnostic']['GT_was_supplied_as_history']
    assert row['actual_prefix_continuation']['scored'] is True
    assert row['actual_prefix_continuation']['source'] == 'tracked_reference_GT_reversed'
    pred = curves['predictions']['42/oracle_reverse_history']
    endpoint = e.supplied_history_endpoints(pred, data[2]['motion'], data[2]['valid'], 'oracle_reverse_history')
    expected = e.actual_prefix_continuation(pred, data[2]['motion'], data[2]['valid'], data[2]['channel_mask'], endpoint)
    assert row['actual_prefix_continuation']['metrics'] == expected
    assert row['actual_prefix_continuation']['metrics'] != row['same_gt_endpoint_output_diagnostic']['metrics']
    assert not e._same_bits(curves['predictions']['42/oracle_reverse_history'], curves['predictions']['42/oracle_history'])
    assert e._same_bits(curves['predictions']['42/oracle_reverse_history'][:, :16], curves['predictions']['42/full'][:, :16])


def test_reversed_endpoint_uses_first_valid_slot_not_first_native_slot_without_gap_filling():
    pred = torch.arange(40).float()[None, :, None].expand(3, -1, 52).clone()
    target = pred + 100
    valid = torch.ones(3, 40, dtype=torch.bool)
    valid[0, 8:11] = False
    valid[1, 15] = False
    valid[2, 16] = False
    saved = valid.clone()
    for mode, source in (('reverse_history', pred), ('oracle_reverse_history', target)):
        supplied = e.supplied_history_endpoints(pred, target, valid, mode)
        assert torch.equal(valid, saved)
        # The reverse decoder reverses only valid slots 11..15; last slot15
        # receives original slot11. Rows1/2 have no adjacent boundary pair.
        torch.testing.assert_close(supplied[0, 15], source[0, 11], atol=0, rtol=0)
        torch.testing.assert_close(supplied[1:, 15], source[1:, 15], atol=0, rtol=0)
        torch.testing.assert_close(supplied[:, 31], source[:, 24], atol=0, rtol=0)
        unchanged = torch.ones_like(valid)
        unchanged[0, 15] = False; unchanged[:, 31] = False
        torch.testing.assert_close(supplied[unchanged], source[unchanged], atol=0, rtol=0)
    assert torch.equal(pred[:, 15, 0], torch.full((3,), 15.))
    assert torch.equal(target[:, 15, 0], torch.full((3,), 115.))


def test_reversed_oracle_boundary_endpoint_never_reads_current_or_future_gt():
    pred = torch.zeros(1, 40, 52)
    target = torch.arange(40).float()[None, :, None].expand(1, -1, 52).clone()
    valid = torch.ones(1, 40, dtype=torch.bool)
    expected = e.supplied_history_endpoints(pred, target, valid, 'oracle_reverse_history')
    changed = target.clone(); changed[:, 16:] = float('nan')
    actual = e.supplied_history_endpoints(pred, changed, valid, 'oracle_reverse_history')
    torch.testing.assert_close(actual[:, 15], expected[:, 15], atol=0, rtol=0)
    # A later start32 can legitimately read earlier frames24..31; changing
    # frame32 onward must not affect its supplied endpoint.
    changed = target.clone(); changed[:, 32:] = float('nan')
    actual = e.supplied_history_endpoints(pred, changed, valid, 'oracle_reverse_history')
    torch.testing.assert_close(actual[:, [15, 31]], expected[:, [15, 31]], atol=0, rtol=0)
    assert e.supplied_history_endpoints(pred, changed, valid, 'reverse_history').count_nonzero() == 0


def test_all_teacher_prefix_modes_except_empty_have_actual_endpoint_metrics():
    report, _ = run(fixture())
    records = {**report['modes'], **report['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']}
    for key, row in records.items():
        assert row['actual_prefix_continuation']['scored'] == (key != '42/empty')
    assert records['42/reverse_history']['actual_prefix_continuation']['source'] == 'generated_past_reversed'
    for mode in ('static', 'reverse'):
        assert records['42/'+mode]['actual_prefix_continuation']['source'] == 'generated_past'
