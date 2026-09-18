"""Joint/free-rollout metric contracts, including gaps and correlated channels."""
import copy
import json

import numpy as np
import pytest

from scripts.joint_motion_metrics import score_clip, summarize


def _wave(frames=65):
    t = np.arange(frames, dtype=np.float64)
    return .5 + .2 * np.sin(t[:, None] * (.11 + np.arange(9)[None] * .025))


def test_exact_rollout_has_zero_scores_and_preserves_group_energy():
    target = _wave()
    row = score_clip(np.repeat(target[None], 3, axis=0), target, np.ones(len(target), bool),
                     np.linspace(.1, .3, 9), [[0, 16, 32, 65]] * 3)
    assert row['joint_fair_es'] == {'raw': 0., 'centered': 0.}
    assert row['group_fair_es'] == {'raw': [0.] * 4, 'centered': [0.] * 4}
    assert row['variogram']['aggregate'] == pytest.approx(0., abs=1e-28)
    assert row['covariance_distance']['centered'] == pytest.approx(0., abs=1e-15)
    assert row['covariance_distance']['velocity'] == pytest.approx(0., abs=1e-15)
    np.testing.assert_allclose(row['rms_ratio'], 1.)
    np.testing.assert_allclose(row['clamp_rms_retention'], 1.)
    assert row['raw_oob'] == [0.] * 4
    assert row['speed']['seam']['count'] == 6
    assert row['speed']['within']['count'] == 3 * 64 - 6
    json.dumps(row, allow_nan=False)
    json.dumps(summarize([row]), allow_nan=False)


def test_fair_es_uses_unbiased_distinct_sample_spread_and_all_samples():
    target = np.zeros((1, 9))
    samples = np.stack((target, np.ones_like(target)))
    # E distance=.5; fair spread=1/(2*(2-1))=.5. The biased K^2
    # normalization would incorrectly give .25, and best-of-N would hide the
    # change when both draws below are moved away from the target.
    row = score_clip(samples, target, np.array([True]), np.ones(9))
    assert row['joint_fair_es']['raw'] == 0.
    samples[0] = 1.
    assert score_clip(samples, target, np.array([True]), np.ones(9))['joint_fair_es']['raw'] == 1.


def test_group_order_retains_left_right_channel_differences():
    target = np.zeros((4, 9)); samples = np.zeros((2, 4, 9))
    samples[..., 2] = 1.
    row = score_clip(samples, target, np.ones(4, bool), np.ones(9))
    np.testing.assert_allclose(row['group_fair_es']['raw'], [1 / np.sqrt(3), 0, 0, 0])
    assert row['joint_fair_es']['raw'] == pytest.approx(1 / 3)


def test_marginally_identical_channel_shuffle_changes_joint_scores():
    # Four joint samples are perfectly correlated across channels. Reassigning
    # channel 1's samples preserves every channel's entire trajectory marginal,
    # but destroys its relationship to the other eight channels.
    base = np.sin(np.arange(48) * .21) * .2
    signs = np.array([-1., -1., 1., 1.])
    coupled = .5 + signs[:, None, None] * np.repeat(base[None, :, None], 9, axis=2)
    shuffled = coupled.copy(); shuffled[..., 1] = coupled[[0, 2, 1, 3], :, 1]
    target = coupled[0]
    for channel in range(9):
        np.testing.assert_array_equal(np.sort(coupled[..., channel], axis=0),
                                      np.sort(shuffled[..., channel], axis=0))
    a = score_clip(coupled, target, np.ones(48, bool), np.ones(9))
    b = score_clip(shuffled, target, np.ones(48, bool), np.ones(9))
    assert b['covariance_distance']['centered'] > a['covariance_distance']['centered'] + 1e-3
    assert b['covariance_distance']['velocity'] > a['covariance_distance']['velocity'] + 1e-5
    assert abs(b['joint_fair_es']['centered'] - a['joint_fair_es']['centered']) > 1e-3
    # Channelwise variograms cannot detect this sample-wise joint mismatch.
    assert b['variogram']['aggregate'] == pytest.approx(a['variogram']['aggregate'], abs=1e-14)


def test_padding_and_invalid_payload_do_not_change_any_score():
    target = _wave(38)
    samples = np.stack((target + .02, target - .01, target + .04))
    valid = np.ones(38, bool); valid[12:16] = False
    boundaries = [[0, 8, 16, 25, 38]] * 3
    first = score_clip(samples, target, valid, np.ones(9), boundaries)
    padded = score_clip(np.pad(samples, ((0, 0), (0, 40), (0, 0)), constant_values=np.nan),
                        np.pad(target, ((0, 40), (0, 0)), constant_values=np.nan),
                        np.pad(valid, (0, 40)), np.ones(9), boundaries)
    assert padded == first
    samples[:, ~valid] = np.nan; target[~valid] = np.inf
    assert score_clip(samples, target, valid, np.ones(9), boundaries) == first


def test_run_offsets_and_gap_jumps_do_not_become_dynamic_energy_or_speed():
    target = np.zeros((12, 9)); target[7:] = 100
    valid = np.array([True] * 5 + [False] * 2 + [True] * 5)
    samples = np.repeat(target[None], 2, axis=0)
    samples[:, :5] += 20; samples[:, 7:] -= 50
    row = score_clip(samples, target, valid, np.ones(9), [[5, 7], [5, 7]])
    assert row['joint_fair_es']['centered'] == 0
    assert row['joint_fair_es']['raw'] > 1
    assert row['pooled']['prediction_energy'] == [0.] * 4
    assert row['pooled']['target_energy'] == [0.] * 4
    assert row['variogram']['aggregate'] == 0
    assert row['variogram']['by_lag']['4']['pairs'] == 2
    assert row['variogram']['by_lag']['16']['score'] is None
    assert row['covariance_distance'] == {'centered': 0., 'velocity': 0.}
    assert row['speed']['all']['sum_squares'] == 0
    assert row['speed']['all']['count'] == 16
    assert row['speed']['seam']['count'] == 0
    assert row['rms_ratio'] == [None] * 4
    assert row['clamp_rms_retention'] == [1.] * 4


def test_seams_use_declared_new_start_and_never_gap_or_tail():
    target = np.repeat(np.arange(8, dtype=float)[:, None], 9, axis=1)
    target[4:] += 5
    valid = np.ones(8, bool); valid[6] = False
    samples = np.repeat(target[None], 2, axis=0)
    row = score_clip(samples, target, valid, np.ones(9), [[0, 4, 6, 7, 8], [2, 8]])
    # First sample seam at 4 has magnitude 6; second at 2 magnitude 1.
    assert row['speed']['seam']['sum_squares'] == 37.
    assert row['speed']['seam']['count'] == 2
    assert row['speed']['within']['count'] == 8
    assert row['speed']['reference_seam'] == row['speed']['seam']
    no_boundaries = score_clip(samples, target, valid, np.ones(9))
    assert no_boundaries['speed']['seam'] is None
    assert no_boundaries['speed']['within'] is None
    assert no_boundaries['speed']['all'] == row['speed']['all']


def test_raw_bounds_and_clamp_retention_do_not_modify_primary_scores():
    target = np.repeat(np.array([0., 1., 0., 1.])[:, None], 9, axis=1)
    samples = np.repeat((target * 3 - 1)[None], 2, axis=0)
    row = score_clip(samples, target, np.ones(4, bool), np.ones(9))
    assert row['raw_oob'] == [1.] * 4
    np.testing.assert_allclose(row['clamp_rms_retention'], 1 / 3)
    assert row['joint_fair_es']['raw'] == 1.
    np.testing.assert_allclose(row['rms_ratio'], 3.)


def test_summary_pools_energy_instead_of_averaging_near_static_ratios():
    wave = np.repeat(np.array([-1., 1.])[:, None], 9, axis=1)
    a = score_clip(np.repeat((wave * .1)[None], 2, axis=0), wave * 1e-6,
                   np.ones(2, bool), np.ones(9), [[1], [1]])
    b = score_clip(np.repeat(wave[None], 2, axis=0), wave,
                   np.ones(2, bool), np.ones(9), [[1], [1]])
    summary = summarize([a, b])
    np.testing.assert_allclose(summary['rms_ratio'], np.sqrt(1.01 / (1 + 1e-12)))
    assert a['rms_ratio'][0] > 99999
    assert summary['joint_fair_es']['raw'] == pytest.approx(
        (a['joint_fair_es']['raw'] + b['joint_fair_es']['raw']) / 2)
    assert summary['speed']['seam']['mean_clip_p95'] == pytest.approx(1.1)
    assert summary['speed']['seam']['rms'] == pytest.approx(np.sqrt(2.02))
    assert 'p95' not in summary['speed']['seam']


def test_singleton_runs_have_no_temporal_support_and_json_remains_finite():
    row = score_clip(np.zeros((2, 3, 9)), np.zeros((3, 9)), np.array([True, False, True]),
                     np.ones(9), [[1, 2], [1, 2]])
    assert row['variogram']['aggregate'] is None
    assert row['covariance_distance']['velocity'] is None
    assert row['speed']['all']['rms'] is None
    summary = summarize([row])
    assert summary['variogram']['supported_clips'] == 0
    assert summary['speed']['all']['clips_with_transitions'] == 0
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize('bad', [None, [1.] * 8, [0.] * 9, [float('nan')] * 9])
def test_bad_scales_rejected_without_fitting_a_replacement(bad):
    with pytest.raises(ValueError):
        score_clip(np.zeros((2, 4, 9)), np.zeros((4, 9)), np.ones(4, bool), bad)


@pytest.mark.parametrize('boundaries', [[[1]], [[1, 1], []], [[2, 1], []],
                                      [[1.5], []], [[True], []], [[-1], []], [[5], []]])
def test_ambiguous_or_out_of_clock_boundaries_are_rejected(boundaries):
    with pytest.raises(ValueError):
        score_clip(np.zeros((2, 4, 9)), np.zeros((4, 9)), np.ones(4, bool), np.ones(9), boundaries)


def test_nonfinite_observations_unfair_sample_count_and_mixed_units_are_rejected():
    with pytest.raises(ValueError):
        score_clip(np.zeros((1, 4, 9)), np.zeros((4, 9)), np.ones(4, bool), np.ones(9))
    with pytest.raises(ValueError):
        score_clip(np.full((2, 4, 9), np.nan), np.zeros((4, 9)), np.ones(4, bool), np.ones(9))
    with pytest.raises(ValueError):
        score_clip(np.zeros((2, 4, 9)), np.zeros((4, 9)), np.ones(4), np.ones(9))
    row = score_clip(np.zeros((2, 4, 9)), np.zeros((4, 9)), np.ones(4, bool), np.ones(9))
    other = copy.deepcopy(row); other['scales'][0] = 2.
    with pytest.raises(ValueError):
        summarize([row, other])
    with pytest.raises(ValueError):
        summarize([])
