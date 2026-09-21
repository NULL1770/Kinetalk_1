import json

import pytest
import torch

from scripts.mouth_protection import MOUTH, mouth_metrics, protection_report


def sample():
    target = torch.zeros(4, 9, 52, dtype=torch.float64)
    target[..., list(MOUTH)] = target.new_tensor([.125, .375, .5, .75, .5, .375, .125, .5, .25])[None, :, None]
    valid = torch.ones(4, 9, dtype=torch.bool)
    channels = torch.ones(4, 52, dtype=torch.bool)
    emotion = torch.tensor([0, 0, 1, 2])
    return target, valid, channels, emotion


def test_metrics_separate_mean_offset_from_timing_without_clamping():
    target, valid, channels, _ = sample()
    prediction = target + 1.
    metrics = mouth_metrics(prediction, target, valid, channels)
    assert metrics['valid']
    assert metrics['raw_mse'] == pytest.approx(1.)
    assert metrics['centered_mse'] == pytest.approx(0., abs=1e-28)
    assert metrics['centered_correlation'] == pytest.approx(1.)
    assert metrics['centered_r2'] == pytest.approx(1.)
    assert metrics['frame_displacement_mse'] == pytest.approx(0., abs=1e-28)
    assert metrics['outside_fraction'] == 1.
    assert metrics['prediction_range']['max'] > 1.
    assert metrics['observed_values'] == 4 * 9 * 27
    assert metrics['adjacent_pairs'] == 4 * 8 * 27
    assert metrics['per_channel_observations'] == [36] * 27


def test_channel_and_frame_missingness_ignore_payload_and_do_not_bridge_gaps():
    target, valid, channels, _ = sample()
    valid[:, 3] = False
    channels[:, 15] = False
    prediction = target.clone()
    target[:, 3] = float('nan'); prediction[:, 3] = float('inf')
    target[:, :, 15] = float('inf'); prediction[:, :, 15] = float('nan')
    # Unknown non-mouth output never enters this mouth-specific diagnostic.
    prediction[:, :, 51] = float('nan')
    metrics = mouth_metrics(prediction, target, valid, channels)
    assert metrics['valid']
    assert metrics['raw_mse'] == metrics['frame_displacement_mse'] == 0.
    assert metrics['observed_values'] == 4 * 8 * 26
    assert metrics['adjacent_pairs'] == 4 * 6 * 26
    assert metrics['per_channel_observations'][1] == 0
    assert metrics == mouth_metrics(prediction, target, valid, channels[:, None].expand_as(target))


def test_stage_gate_accepts_same_base_and_smaller_mean_error():
    target, valid, channels, emotion = sample()
    base = target + .25
    same = protection_report(base, base, target, valid, channels, emotion)
    assert same['passed']
    # Binary-exact offsets retain zero finite-difference error.
    better = protection_report(target + .125, base, target, valid, channels, emotion)
    assert better['passed']
    assert not better['failures']
    json.dumps(better, allow_nan=False)


def test_temporal_damage_fails_even_when_raw_mse_improves():
    target, valid, channels, emotion = sample()
    base = target + .25
    prediction = target.clone()
    prediction[:, ::2, list(MOUTH)] += .05
    report = protection_report(prediction, base, target, valid, channels, emotion)
    assert report['groups']['overall']['prediction']['raw_mse'] < report['groups']['overall']['base']['raw_mse']
    assert not report['passed']
    assert 'overall/velocity_preserved' in report['failures']
    assert 'neutral/velocity_preserved' in report['failures']
    assert 'nonneutral/velocity_preserved' in report['failures']


def test_neutral_raw_degradation_is_detected_with_identical_timing():
    target, valid, channels, emotion = sample()
    base = target + .125
    prediction = base.clone(); prediction[emotion == 0] += .125
    report = protection_report(prediction, base, target, valid, channels, emotion)
    assert not report['passed']
    assert 'neutral/raw_mse_preserved' in report['failures']
    assert report['groups']['neutral']['checks']['correlation_preserved']


def test_reverse_motion_fails_correlation_check():
    target, valid, channels, emotion = sample()
    prediction = target.flip(1)
    report = protection_report(prediction, target, target, valid, channels, emotion)
    assert 'overall/correlation_preserved' in report['failures']


@pytest.mark.parametrize('problem', ['nonfinite_prediction', 'nonfinite_target', 'constant', 'missing_neutral', 'no_adjacent', 'no_channels'])
def test_undefined_or_nonfinite_metrics_fail_closed_and_are_json_safe(problem):
    target, valid, channels, emotion = sample()
    prediction = target.clone()
    if problem == 'nonfinite_prediction': prediction[0, 2, 17] = float('nan')
    elif problem == 'nonfinite_target': target[0, 2, 17] = float('inf')
    elif problem == 'constant': prediction[..., list(MOUTH)] = .5
    elif problem == 'missing_neutral': emotion[:] = 1
    elif problem == 'no_adjacent': valid[:, 1::2] = False
    elif problem == 'no_channels': channels[:, list(MOUTH)] = False
    report = protection_report(prediction, target, target, valid, channels, emotion)
    assert not report['passed']
    assert any('finite_defined_metrics' in reason for reason in report['failures'])
    json.dumps(report, allow_nan=False)


def test_invalid_contract_rejected():
    target, valid, channels, emotion = sample()
    with pytest.raises(ValueError, match='Boolean'):
        mouth_metrics(target, target, valid.float(), channels)
    with pytest.raises(ValueError, match='channel_mask'):
        mouth_metrics(target, target, valid, channels[0])
    with pytest.raises(ValueError, match='emotion'):
        protection_report(target, target, target, valid, channels, emotion.float())


def test_constant_decimal_stream_is_undefined_not_rounding_motion():
    target, valid, channels, emotion = sample()
    prediction = torch.full_like(target, .1)
    report = protection_report(prediction, target, target, valid, channels, emotion)
    metrics = report['groups']['overall']['prediction']
    assert not metrics['valid']
    assert metrics['centered_correlation'] is None
    assert 'undefined_prediction_temporal_variance' in metrics['reasons']


def test_finite_inputs_that_overflow_statistics_fail_closed():
    target, valid, channels, emotion = sample()
    prediction = target.clone()
    prediction[:, ::2, list(MOUTH)] = 1e308
    prediction[:, 1::2, list(MOUTH)] = -1e308
    report = protection_report(prediction, target, target, valid, channels, emotion)
    assert not report['passed']
    assert any('nonfinite_statistic' in reason for reason in report['groups']['overall']['prediction']['reasons'])
    json.dumps(report, allow_nan=False)
