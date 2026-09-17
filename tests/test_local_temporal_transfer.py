import torch
import pytest

from scripts.audit_local_temporal_transfer import (
    analyze_curves, finish_metrics, pool, sufficient_statistics,
)


def test_internal_gaps_never_become_velocity_pairs():
    pred = torch.tensor([[[0.], [1.], [100.], [3.], [4.]]])
    valid = torch.tensor([[True, True, False, True, True]])
    row = sufficient_statistics(pred, pred, valid)[0]
    assert row['pair_count'] == 2
    assert row['pred_velocity_energy'] == 2
    assert finish_metrics(row)['frame_displacement_mse'] == 0


def test_pool_centers_clips_separately_and_weights_observed_values():
    pred = torch.tensor([[[0.], [2.], [100.]], [[8.], [10.], [12.]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    rows = sufficient_statistics(pred, pred, valid)
    metric = finish_metrics(pool(rows, [0, 1]))
    assert metric['observed_values'] == 5
    assert metric['prediction_centered_rms'] == pytest.approx((10 / 5) ** .5)
    assert metric['centered_correlation'] == pytest.approx(1.)


def test_degenerate_target_returns_null_not_artificial_huge_ratio():
    pred = torch.arange(4.).reshape(1, 4, 1)
    rows = sufficient_statistics(pred, torch.zeros_like(pred), torch.ones(1, 4, dtype=torch.bool))
    metric = finish_metrics(rows[0])
    assert metric['rms_ratio'] is None
    assert metric['centered_correlation'] is None
    assert metric['velocity_rms_ratio'] is None


def tiny_curves():
    pred = torch.full((1, 4, 52), 2.)
    pred[0, :, 41] = torch.tensor([1., 2., 3., 4.])
    target = pred.clone()
    return {'predictions': {'42/full': pred}, 'target': target,
            'valid': torch.ones(1, 4, dtype=torch.bool),
            'channel_mask': torch.ones(1, 52, dtype=torch.bool)}


META = [{'clip_id': 'a', 'sentence_id': 'sentence', 'speaker': 'person', 'emotion': 'angry'}]


def test_clamping_reveals_saturation_without_mutating_saved_curves():
    curves = tiny_curves()
    original = curves['predictions']['42/full'].clone()
    report = analyze_curves(curves, META, require_local_static=False)
    retention = report['groups']['all']['clamp_retention']['full/brows']
    assert retention['pooled_centered_rms_retained'] == 0
    assert retention['clips_below_half_raw_rms']['fraction'] == 1
    assert torch.equal(original, curves['predictions']['42/full'])
    assert report['local_static_available'] is False
    assert report['groups']['all']['full_vs_local_static'] == {}


def test_missing_local_static_is_not_substituted_by_joint_static():
    curves = tiny_curves()
    curves['predictions']['42/static'] = curves['predictions']['42/full'].clone()
    with pytest.raises(ValueError, match='local_static'):
        analyze_curves(curves, META, require_local_static=True)


def test_local_static_equal_predictions_have_zero_benefit_fraction():
    curves = tiny_curves()
    curves['predictions']['42/local_static'] = curves['predictions']['42/full'].clone()
    report = analyze_curves(curves, META, require_local_static=True)
    result = report['groups']['all']['full_vs_local_static']['brows/raw']['centered_mse']
    assert result['full_better']['fraction'] == 0
    assert result['equal']['fraction'] == 1
