import numpy as np

from scripts.regional_receiver_metrics import (
    evaluate_receiver_metrics, fit_activity_thresholds, paired_clustered_ci,
)


def test_constant_sequence_has_none_correlation_and_json_safe_metrics():
    pred = np.ones((1, 20, 2)); target = np.ones((1, 20, 2)); valid = np.ones((1, 20), bool)
    thresholds = fit_activity_thresholds(target, valid)
    out = evaluate_receiver_metrics(pred, target, valid, thresholds)
    assert out["aggregate"]["brow"]["pearson"] is None
    assert out["aggregate"]["eye"]["xcorr_best_lag_frames"] is None
    assert out["per_clip"][0]["regions"]["brow"]["missed_active_frame_rate"] == 0.0


def test_known_delayed_burst_reports_native_clock_delay():
    n = 50
    target = np.zeros((1, n, 2)); pred = np.zeros_like(target)
    target[0, 20:30, 0] = 1.; pred[0, 23:33, 0] = 1.
    target[0, 10:20, 1] = 1.; pred[0, 10:20, 1] = 1.
    valid = np.ones((1, n), bool)
    out = evaluate_receiver_metrics(pred, target, valid, fit_activity_thresholds(target, valid))
    brow = out["per_clip"][0]["regions"]["brow"]
    assert brow["xcorr_best_lag_frames"] == 3
    assert brow["dominant_peak_abs_delay_frames"] == 3
    assert brow["dominant_peak_abs_delay_seconds"] == 3 / 25


def test_gaps_do_not_join_runs_and_padding_is_invariant():
    n = 40
    target = np.zeros((1, n, 2)); pred = np.zeros_like(target)
    target[0, 2:10, 0] = pred[0, 2:10, 0] = 1.
    target[0, 25:34, 0] = pred[0, 25:34, 0] = 1.
    valid = np.zeros((1, n), bool); valid[0, 2:10] = True; valid[0, 25:34] = True
    out1 = evaluate_receiver_metrics(pred, target, valid, fit_activity_thresholds(target, valid))
    pred2, target2, valid2 = np.concatenate([pred, np.zeros((1, 10, 2))], 1), np.concatenate([target, np.zeros((1, 10, 2))], 1), np.concatenate([valid, np.zeros((1, 10), bool)], 1)
    out2 = evaluate_receiver_metrics(pred2, target2, valid2, fit_activity_thresholds(target, valid))
    assert out1["aggregate"]["brow"]["pearson"] == out2["aggregate"]["brow"]["pearson"]
    assert out1["aggregate"]["brow"]["missed_active_frame_rate"] == out2["aggregate"]["brow"]["missed_active_frame_rate"]


def test_missing_score_cannot_be_fixed_by_prediction_rescaling():
    target = np.zeros((1, 20, 2)); target[0, 5:15, 0] = 1.
    valid = np.ones((1, 20), bool); thresholds = fit_activity_thresholds(target, valid)
    low = np.zeros_like(target); low[0, 5:15, 0] = .05
    high = np.zeros_like(target); high[0, 5:15, 0] = .5
    a = evaluate_receiver_metrics(low, target, valid, thresholds)
    b = evaluate_receiver_metrics(high, target, valid, thresholds)
    assert a["aggregate"]["brow"]["missed_active_frame_rate"] == 1.
    assert b["aggregate"]["brow"]["missed_active_frame_rate"] == 0.
    assert a["thresholds"] == b["thresholds"]


def test_cluster_ci_is_json_safe_and_groups_sentences():
    out = paired_clustered_ci([-1., -2., 2., 2.], ["a", "a", "b", "b"], bootstrap=200)
    assert out["clusters"] == 2 and isinstance(out["sentence_cluster_95ci"], list)
