"""Same-clock diagnosis must not hide generated error with a second U map."""
import numpy as np
import pytest
import torch

from kinetalk_b0.predictable_motion import bin_centered_frames
from scripts.audit_projection_delivery import delivery_diagnostic, mean_noise_statistics, residual_bins
from scripts.audit_predictable_motion_predictions import clip_statistics, paired_summary


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    torch.manual_seed(42)
    target = torch.randn(6, 12, 52) * .1
    valid = torch.ones(6, 12, dtype=torch.bool); valid[0, 10:] = False
    b0 = torch.full_like(target, .2); identity = torch.full((6, 52), .3)
    q = {'motion': target + b0 + identity[:, None], 'valid': valid, 'motion_valid': valid,
         'channel_mask': torch.ones(6, 52, dtype=torch.bool), 'emotion_id': torch.tensor([0, 1] * 3),
         'speaker_id': torch.tensor([4, 4, 5, 5, 6, 6]), 'sentence_id': ['a', 'b'] * 3}
    split = {'q': q, 'base': {'b0': b0}, 'identity': {'baseline': identity},
             'affect': {'emotion_logits': torch.tensor([[-20., 20.]] * 6)}}
    bins, weight = bin_centered_frames(target, valid)
    feature = torch.zeros(6, 3, 2)
    bundle = {'motion_bins': bins, 'weight': weight,
              'features': {'content': feature, 'middle': feature, 'prosody': feature}}
    basis = torch.zeros(52, 8); basis[torch.arange(5, 13), torch.arange(8)] = 1
    head = {'basis': basis, 'channels': torch.arange(52), 'feature_std': torch.ones(6),
            'target_scale': torch.ones(8), 'linear.weight': torch.zeros(8, 6)}
    curves = {'motion': {str(seed): {'full': q['motion'].clone(), 'zero': b0 + identity[:, None],
                                   'oracle': q['motion'].clone()} for seed in (42, 123, 2026)}}
    return split, bundle, head, curves


def test_generated_residuals_preserve_directions_outside_teacher_basis():
    split, bundle, head, curves = fixture()
    report = delivery_diagnostic(split, bundle, curves, head, ['neutral', 'happy'], {'a': 4, 'b': 5, 'c': 6}, samples=50)
    scores = report['scores']['pooled/nonneutral']
    # U excludes all brows; generated motion is exactly correct in that region.
    assert scores['motion_projection_oracle_gated']['brows']['r2_against_zero'] == pytest.approx(0)
    assert scores['generated_oracle']['brows']['r2_against_zero'] == pytest.approx(1)
    assert scores['generated_zero']['brows']['r2_against_zero'] == pytest.approx(0)
    assert scores['audio_gated_native']['brows']['r2_against_zero'] == pytest.approx(0)
    assert report['clock']['bins'] == 3


def test_generated_binning_rejects_a_different_partial_bin_clock():
    split, bundle, _, _ = fixture()
    actual = residual_bins(split['q']['motion'], split, bundle)
    torch.testing.assert_close(actual, bundle['motion_bins'], rtol=1e-5, atol=1e-7)
    bundle['weight'] = bundle['weight'].clone(); bundle['weight'][0, -1] += 1
    with pytest.raises(ValueError, match='clock'):
        residual_bins(split['q']['motion'], split, bundle)


def test_three_noise_aggregation_pairs_exactly_with_deterministic_target():
    torch.manual_seed(238)
    target = torch.randn(7, 24, 52, dtype=torch.float64)
    weights = torch.full((7, 24), 4., dtype=torch.float64)
    weights[0, -1] = 1
    reference = clip_statistics(torch.zeros_like(target), target, weights, list(range(41, 46)))
    rows = [clip_statistics(torch.randn(target.shape, generator=torch.Generator().manual_seed(seed),
                                       dtype=torch.float64), target, weights, list(range(41, 46)))
            for seed in (42, 123, 2026)]
    averaged = mean_noise_statistics(rows, reference)
    np.testing.assert_array_equal(averaged[:, [1, 3, 6]], reference[:, [1, 3, 6]])
    np.testing.assert_allclose(averaged[:, [0, 2, 4, 5]], np.stack(rows).mean(0)[:, [0, 2, 4, 5]], rtol=0, atol=0)
    result = paired_summary(averaged, reference, ['a', 'b', 'c', 'd', 'a', 'b', 'e'], list(range(7)), samples=100)
    assert result['bootstrap_valid_samples'] == 100
    corrupted = [row.copy() for row in rows]; corrupted[1][0, 1] += 1e-9
    with pytest.raises(ValueError, match='different target'):
        mean_noise_statistics(corrupted, reference)
