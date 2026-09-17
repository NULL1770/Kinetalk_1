import copy

import pytest
import torch

from scripts.audit_temporal_repair import (
    center, paired_metrics, validate_matched, trajectory_energy_score, adjacent_variogram_score, metadata_equal,
)


def test_model_b0_recomputation_is_separate_from_exact_target_binding():
    a={'clip_id':['a'],'target':torch.zeros(1,2,52),'b0':torch.zeros(1,2,52),
       'valid':torch.ones(1,2,dtype=torch.bool),'channel_mask':torch.ones(1,52,dtype=torch.bool),
       'times':torch.tensor([[0.,.04]])}
    b=copy.deepcopy(a);b['b0']+=.0001
    with pytest.raises(ValueError):metadata_equal(a,b)
    metadata_equal(a,b,compare_b0=False)
    b['target']+=.0001
    with pytest.raises(ValueError):metadata_equal(a,b,compare_b0=False)


def test_fair_energy_three_seed_formula_and_invalid_padding():
    target = torch.zeros(1, 3, 1)
    mask = torch.tensor([[[True], [True], [False]]])
    values = torch.tensor([0., 1., 2.])[:, None, None, None].expand(3, 1, 3, 1).clone()
    values[:, :, -1] = float('nan'); target[:, -1] = float('nan')
    score = trajectory_energy_score(values, target, mask)
    # mean distances=1, unordered pair sum=4; fair subtracts 4/(3*2).
    assert score['fair']['mean'] == pytest.approx(1/3)
    assert score['empirical']['mean'] == pytest.approx(5/9)
    torch.testing.assert_close(center(values, mask[None])[..., :2, :], torch.zeros(3, 1, 2, 1))


def test_fair_variogram_excludes_gaps_and_has_finite_sample_correction():
    target = torch.tensor([[[0.], [1.], [float('nan')], [10.]]])
    mask = torch.tensor([[[True], [True], [False], [True]]])
    samples = torch.tensor([[[[0.], [0.], [float('nan')], [8.]]],
                            [[[0.], [1.], [float('nan')], [9.]]],
                            [[[0.], [4.], [float('nan')], [10.]]]])
    score = adjacent_variogram_score(samples, target, mask, power=.5)
    assert score['per_clip_observed_pairs'] == [1]
    assert score['empirical']['mean'] == pytest.approx(0.)
    assert score['fair']['mean'] == pytest.approx(-1/3)


def test_matched_audit_rejects_rng_or_initial_state_drift():
    def run(phase):
        state = {'weight': torch.ones(2)}
        return {'recipe': {'phase': phase, 'epochs': 1},
                'initial': {k: copy.deepcopy(state) for k in ('upper', 'local', 'adapter')},
                'final': {'adapter': copy.deepcopy(state)}, 'binding': {'initial': phase + '_hash'},
                'epochs': [{'epoch': 1, 'batches': 2, 'total_steps': 2, 'batch_noise_time_sha256': 'same'}]}
    a, b = run('direct'), run('soft')
    assert validate_matched(a, b)['initial_upper_local_adapter_equal_exactly']
    b['epochs'][0]['batch_noise_time_sha256'] = 'different'
    with pytest.raises(ValueError, match='RNG'): validate_matched(a, b)
    b = run('soft'); b['initial']['local']['weight'][0] += .001
    with pytest.raises(ValueError, match='tensor changed'): validate_matched(a, b)


def test_centered_paired_scores_do_not_mistake_offset_for_dynamic_change():
    truth = torch.tensor([[[0.], [1.], [0.], [-1.]]])
    mask = torch.ones_like(truth, dtype=torch.bool)
    result = paired_metrics(truth + 3., truth, mask)
    assert result['raw_mse'] == pytest.approx(9.)
    assert result['centered_mse'] == pytest.approx(0.)
    assert result['centered_r2'] == pytest.approx(1.)
    assert result['centered_correlation'] == pytest.approx(1.)
