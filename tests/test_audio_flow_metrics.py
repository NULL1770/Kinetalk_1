import math

import pytest
import torch

from scripts.audio_flow_metrics import (
    adjacent_variogram_score, audit_audio_flow_samples, paired_condition_response,
    trajectory_energy_score, velocity_statistics,
)


def test_whole_trajectory_energy_score_has_exact_fair_and_empirical_values():
    predictions = torch.tensor([[[[0.], [0.]]], [[[1.], [1.]]], [[[2.], [2.]]]])
    target = torch.zeros(1, 2, 1)
    observed = torch.ones_like(target, dtype=torch.bool)
    score = trajectory_energy_score(predictions, target, observed)
    assert score['target_distance']['mean'] == pytest.approx(1)
    assert score['fair_pair_spread_term']['mean'] == pytest.approx(2 / 3)
    assert score['fair']['mean'] == pytest.approx(1 / 3)
    assert score['empirical']['mean'] == pytest.approx(5 / 9)
    # If samples are identical at the observed target, both ES estimates are zero.
    perfect = trajectory_energy_score(torch.zeros_like(predictions), target, observed)
    assert perfect['fair']['mean'] == 0
    assert perfect['empirical']['mean'] == 0


def test_energy_is_joint_trajectory_score_and_ignores_unobserved_nans():
    predictions = torch.tensor([[[[0.], [2.], [float('nan')]]],
                                [[[2.], [0.], [float('nan')]]],
                                [[[0.], [2.], [float('nan')]]]])
    target = torch.tensor([[[0.], [0.], [float('nan')]]])
    observed = torch.tensor([[[True], [True], [False]]])
    score = trajectory_energy_score(predictions, target, observed)
    # Each complete target distance is sqrt((0^2+2^2)/2), not mean |frame error|=1.
    assert score['target_distance']['mean'] == pytest.approx(math.sqrt(2))
    assert score['fair']['mean'] == pytest.approx(math.sqrt(2) - 2 / 3)


def test_variogram_fair_finite_k_correction_and_gap_mask():
    # The only scored adjacent pair is frames 0-1; large values across the invalid
    # middle frame and the isolated last frame must not affect the result.
    predictions = torch.tensor([[[[0.], [0.], [float('nan')], [100.]]],
                                [[[0.], [1.], [float('nan')], [-100.]]],
                                [[[0.], [2.], [float('nan')], [200.]]]])
    target = torch.tensor([[[0.], [1.], [float('nan')], [0.]]])
    observed = torch.tensor([[[True], [True], [False], [True]]])
    score = adjacent_variogram_score(predictions, target, observed, power=1)
    assert score['per_clip_observed_pairs'] == [1]
    assert score['empirical']['mean'] == 0
    assert score['finite_k_correction']['mean'] == pytest.approx(1 / 3)
    assert score['fair']['mean'] == pytest.approx(-1 / 3)


def test_variogram_no_valid_adjacency_returns_none_not_nan():
    target = torch.zeros(1, 3, 1)
    observed = torch.tensor([[[True], [False], [True]]])
    score = adjacent_variogram_score(torch.zeros(3, 1, 3, 1), target, observed)
    assert score['fair']['clips'] == 0
    assert score['fair']['mean'] is None


def test_velocity_uses_seconds_and_rejects_invalid_scored_timestamps():
    target = torch.tensor([[[0.], [1.], [2.]]])
    predictions = target.unsqueeze(0).repeat(3, 1, 1, 1)
    observed = torch.ones_like(target, dtype=torch.bool)
    times = torch.tensor([[0., .5, 1.]])
    out = velocity_statistics(predictions, target, observed, times)
    assert out['observed_pairs'] == 2
    assert out['decomposition']['target_rms'] == pytest.approx(2)
    assert out['decomposition']['single_sample_mse'] == 0
    with pytest.raises(ValueError, match='increasing'):
        velocity_statistics(predictions, target, observed, torch.tensor([[0., 0., 1.]]))


def test_paired_condition_response_uses_same_seed():
    right = torch.tensor([[[[0.], [0.]]], [[[10.], [10.]]], [[[20.], [20.]]]])
    left = right + 2
    observed = torch.ones(1, 2, 1, dtype=torch.bool)
    response = paired_condition_response(left, right, observed)
    assert response['paired_response_rms'] == 2
    assert response['response_seed_std'] == 0


def _bundle():
    seeds = [71, 13, 104, 96]
    target = torch.arange(3, dtype=torch.double)[None, :, None].expand(2, 3, 52).clone() * .02
    reference = {'q': {'motion': target, 'valid': torch.ones(2, 3, dtype=torch.bool),
        'channel_mask': torch.ones(2, 52, dtype=torch.bool), 'emotion_id': torch.tensor([0, 1]),
        'speaker_id': torch.tensor([2, 3]), 'times': torch.tensor([[0., .04, .08], [0., .04, .08]])},
        'base': {'b0': torch.zeros_like(target)}, 'identity': {'baseline': torch.zeros(2, 52)}}
    curves = {'noise_seeds': seeds, 'decode_steps': 7, 'motion': {str(seed): {
        'full': target + i * .01, 'zero': target * 0, 'reverse': target.flip(1) + i * .01}
        for i, seed in enumerate(seeds)}}
    return seeds, curves, reference


def test_audit_arbitrary_predeclared_seeds_scores_all_regions_and_centering():
    seeds, curves, reference = _bundle()
    out = audit_audio_flow_samples(curves, reference, expected_seeds=seeds)
    assert out['noise_seeds'] == seeds
    assert out['sample_count'] == 4
    raw = out['scores']['raw_motion']['full']['nonneutral']['brows']
    centered = out['scores']['centered_residual']['full']['nonneutral']['brows']
    assert raw['error_decomposition']['seed_variance'] > 0
    assert centered['error_decomposition']['seed_variance'] < 1e-30
    assert centered['trajectory_energy_score']['fair']['mean'] == pytest.approx(0, abs=1e-16)
    assert 'full_minus_reverse' in out['paired_condition_response']['centered_residual']
    assert out['population_clip_indices']['neutral'] == [0]
    assert set(out['scores']['raw_motion']['full']['all']) == {'upper_expression', 'brows', 'eyes_expression', 'mouth', 'jaw17'}


def test_audit_rejects_changed_seed_order_and_duplicate_or_too_few_seeds():
    seeds, curves, reference = _bundle()
    with pytest.raises(ValueError, match='order'):
        audit_audio_flow_samples(curves, reference, expected_seeds=seeds[::-1])
    with pytest.raises(ValueError, match='distinct'):
        audit_audio_flow_samples(curves, reference, expected_seeds=[1, 1, 2])
    with pytest.raises(ValueError, match='three'):
        audit_audio_flow_samples(curves, reference, expected_seeds=[1, 2])
