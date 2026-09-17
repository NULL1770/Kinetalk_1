import torch
import pytest
from scripts.audit_multiseed_stochasticity import audit, center, decompose


def test_exact_decomposition_separates_bias_and_variance():
    # A common bias is ensemble error; symmetric seed differences are variance.
    target = torch.tensor([[1., 3.], [2., 5.]], dtype=torch.double)
    predictions = torch.stack([target + 1, target + 3, target + 5])
    observed = torch.tensor([[True, True], [False, True]])
    out = decompose(predictions, target, observed)
    assert out['ensemble_mean_mse'] == pytest.approx(9)
    assert out['seed_variance'] == pytest.approx(8 / 3)
    assert out['single_sample_mse'] == pytest.approx(35 / 3)
    assert out['decomposition_absolute_error'] < 1e-12


def test_unobserved_nonfinite_entries_do_not_change_decomposition():
    target = torch.tensor([[1., float('nan')], [2., 5.]])
    observed = torch.tensor([[True, False], [True, True]])
    predictions = torch.stack([target + 1, target - 1])
    out = decompose(predictions, target, observed)
    assert out['seed_variance'] == pytest.approx(1)
    assert out['ensemble_mean_mse'] == pytest.approx(0)
    assert torch.equal(center(predictions, observed.expand_as(predictions))[:, 0, 1], torch.zeros(2))


def test_centering_removes_static_seed_offsets_but_retains_motion_variation():
    target = torch.zeros(1, 3, 1)
    observed = torch.ones_like(target, dtype=torch.bool)
    predictions = torch.tensor([[[[1.], [1.], [1.]]], [[[4.], [4.], [4.]]]])
    assert decompose(predictions, target, observed)['seed_variance'] > 0
    centered = center(predictions, observed.expand_as(predictions))
    assert decompose(centered, target, observed)['seed_variance'] == 0


def test_audit_rejects_non_protocol():
    with pytest.raises(ValueError):
        audit({'noise_seeds': [1], 'decode_steps': 1, 'motion': {}}, {})
