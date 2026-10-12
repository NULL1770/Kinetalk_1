import torch
from scripts.audit_v5_state_coordinates import decomposition


def test_mse_decomposition_separates_posture_bias_from_temporal_error():
    target = torch.tensor([[0., 2.], [1., 4.], [2., 6.], [100., 100.]])
    prediction = torch.tensor([[3., 6.], [4., 4.], [5., 2.], [-100., -100.]])
    quality = torch.ones_like(target, dtype=torch.bool)
    quality[-1] = False
    parts = decomposition(prediction, target, quality)
    total = torch.tensor(parts['bias_mse']) + torch.tensor(parts['centered_mse'])
    assert torch.allclose(total, (prediction[:3]-target[:3]).square().mean(0))
    assert parts['bias_mse'][0] == 9.
    assert parts['centered_mse'][0] == 0.
    assert parts['bias_mse'][1] == 0.
    assert parts['centered_mse'][1] > 0.
    assert parts['valid_count'] == [3, 3]


def test_decomposition_never_imputes_missing_state_axes_or_invalid_frames():
    target = torch.tensor([[1., float('nan')], [3., float('nan')], [float('nan'), float('nan')]])
    prediction = torch.tensor([[2., float('nan')], [4., float('nan')], [float('nan'), float('nan')]])
    quality = torch.tensor([[True, False], [True, False], [False, False]])
    parts = decomposition(prediction, target, quality)
    assert parts['valid_count'] == [2, 0]
    assert parts['bias_mse'] == [1., 0.]
    assert parts['centered_mse'] == [0., 0.]
    assert parts['target_centered_variance'] == [1., 0.]
