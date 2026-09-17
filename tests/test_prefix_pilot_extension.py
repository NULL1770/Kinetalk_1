"""Continuation is measured against the actual supplied past endpoint."""
import copy

import pytest
import torch

from scripts import extend_prefix_pilot as extension


UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]


def clip(frames=33, batch=1):
    target = torch.zeros(batch, frames, 52, dtype=torch.float64)
    target[:, 16:] = 1.
    if frames > 32: target[:, 32:] = 2.
    valid = torch.ones(batch, frames, dtype=torch.bool)
    mask = torch.ones(batch, 52, dtype=torch.bool)
    return target, valid, mask


def test_continuation_uses_actual_supplied_gt_prefix_not_previous_oracle_output():
    target, valid, channels = clip()
    pred = torch.zeros_like(target)
    pred[:, 16, UPPER] = 1.5
    pred[:, 32, UPPER] = 2.5
    pred[:, [15, 31], :] = 1000.  # Unrelated previous oracle chunk endpoints.
    actual = extension.continuation_report(pred, target, valid, channels, target)
    stitched = extension.continuation_report(pred, target, valid, channels, pred)
    for name in ('brows', 'eyes_expression'):
        assert actual[name]['observed_channel_pairs'] == (10 if name == 'brows' else 8)
        assert actual[name]['rms'] == 1.5
        assert actual[name]['reference_rms'] == 1.
        assert actual[name]['displacement_mse'] == .25
        assert stitched[name]['rms'] > 900
    pred[:, [15, 31], :] = float('nan')
    again = extension.continuation_report(pred, target, valid, channels, target)
    assert again == actual


def test_generated_continuation_uses_generated_past_and_matches_stitched_steps():
    target, valid, channels = clip()
    pred = target.clone()
    pred[:, 15, UPPER] = 2.; pred[:, 16, UPPER] = 2.5
    pred[:, 31, UPPER] = 4.; pred[:, 32, UPPER] = 4.5
    result = extension.continuation_report(pred, target, valid, channels, pred)
    for row in result.values():
        assert row['rms'] == .5
        assert row['reference_rms'] == 1.
        assert row['displacement_mse'] == .25


def test_invalid_preceding_frame_and_unobserved_channels_are_excluded_without_gap_bridging():
    target, valid, channels = clip(batch=2)
    pred = target.clone(); supplied = target.clone()
    pred[:, 16, UPPER] += .5; pred[:, 32, UPPER] += .5
    valid[0, 15] = False
    valid[1, 32] = False
    pred[0, 16] = float('nan'); supplied[0, 15] = float('nan')
    pred[1, 32] = float('nan')
    channels[:, 41] = False
    pred[..., 41] = target[..., 41] = supplied[..., 41] = float('nan')
    result = extension.continuation_report(pred, target, valid, channels, supplied)
    for row in result.values():
        assert row['observed_channel_pairs'] == 8  # Two valid seams x four channels.
        assert row['rms'] == 1.5 and row['reference_rms'] == 1.
        assert row['displacement_mse'] == .25


def gate_fixture():
    target, valid, channels = clip()
    metadata = {'target': target, 'valid': valid, 'channel_mask': channels,
                'times': torch.arange(33, dtype=torch.float64)[None]/25,
                'clip_id': ['synthetic'], 'b0': torch.zeros_like(target),
                'emotion_id': torch.tensor([1]), 'speaker_id': torch.tensor([7])}
    no_prefix = target.clone(); oracle = target.clone()
    for frame in (16, 32):
        no_prefix[:, frame, UPPER] += 1.
        oracle[:, frame, UPPER] += .25
    oracle[:, [15, 31], :] = 1000.  # Must not become the supplied GT endpoint.
    generated = target.clone(); generated[:, 15, UPPER] = -9.; generated[:, 31, UPPER] = -8.
    curves = {'no_prefix': {**copy.deepcopy(metadata), 'predictions': {'42/full': no_prefix}},
              'teacher_prefix': {**copy.deepcopy(metadata), 'predictions': {'42/full': generated, '42/oracle_history': oracle}}}
    def metrics(value): return {'metrics': {name: {'centered_mse': value} for name in ('brows', 'eyes_expression')}}
    reports = {'no_prefix': {'modes': {'42/full': metrics(2.)}},
               'teacher_prefix': {'modes': {'42/full': metrics(999.), '42/oracle_history': metrics(1.)}}}
    return curves, reports


def test_corrected_gate_compares_oracle_to_gt_and_no_prefix_to_same_gt_without_mixing_full():
    curves, reports = gate_fixture()
    result = extension.corrected_gate(curves, reports)
    assert result['passed'] is True
    for name in ('brows', 'eyes_expression'):
        assert result['checks'][name]['oracle_boundary_rms_ratio'] == 1.25
        assert result['checks'][name]['boundary_error_ratio_to_no_prefix'] == .0625
        assert result['checks'][name]['centered_mse_not_worse']
        assert result['teacher_to_actual_supplied_gt_prefix'][name]['displacement_mse'] == .0625
        assert result['no_prefix_to_same_gt_DIAGNOSTIC_NOT_INPUT'][name]['displacement_mse'] == 1.
        assert result['generated_to_actual_supplied_generated_prefix'][name]['rms'] == 10.
    assert result['post_result_metric_semantics_correction'] is True
    assert result['threshold_values_unchanged'] is True
    # A bad deployment rollout is reported separately; it cannot be silently
    # substituted for the oracle endpoint being checked as receiver evidence.
    assert 'generalization success' in result['scope']


def test_corrected_gate_does_not_round_a_ratio_above_two_into_passing():
    curves, reports = gate_fixture()
    for frame, oracle, no_prefix in ((16, 2.021, 5.), (32, 3.021, 6.)):
        curves['teacher_prefix']['predictions']['42/oracle_history'][:, frame, UPPER] = oracle
        curves['no_prefix']['predictions']['42/full'][:, frame, UPPER] = no_prefix
    result = extension.corrected_gate(curves, reports)
    assert result['passed'] is False
    for row in result['checks'].values():
        assert row['oracle_boundary_rms_ratio'] == pytest.approx(2.021)
        assert row['boundary_error_ratio_to_no_prefix'] < .5


def test_corrected_gate_uses_oracle_centered_mse_and_requires_both_regions():
    curves, reports = gate_fixture()
    reports['teacher_prefix']['modes']['42/full']['metrics']['brows']['centered_mse'] = 0.
    reports['teacher_prefix']['modes']['42/oracle_history']['metrics']['brows']['centered_mse'] = 3.
    result = extension.corrected_gate(curves, reports)
    assert result['passed'] is False
    assert not result['checks']['brows']['centered_mse_not_worse']
    assert result['checks']['eyes_expression']['centered_mse_not_worse']


def test_corrected_gate_rejects_mismatched_reference_metadata():
    curves, reports = gate_fixture()
    curves['teacher_prefix']['target'][0, 0, 41] = 999.
    with pytest.raises(ValueError, match='metadata'):
        extension.corrected_gate(curves, reports)
