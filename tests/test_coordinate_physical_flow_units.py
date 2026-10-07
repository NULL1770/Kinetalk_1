"""Separate standardized numerical coordinates from the physical flow metric."""
import pytest
import torch

from scripts.train_full_staged import coordinate_flow_error_std, flow_vector_mse


def test_default_units_preserve_exact_tensor_and_loss():
    std = torch.tensor([.01, .2, .8])
    prediction = torch.tensor([[[.3, .2, .1]]], requires_grad=True)
    target = torch.tensor([[[.5, .1, -.2]]])
    observed = torch.ones_like(prediction, dtype=torch.bool)
    rng = torch.get_rng_state().clone()
    selected = coordinate_flow_error_std(std)
    assert selected is std and torch.equal(rng, torch.get_rng_state())
    old = flow_vector_mse(prediction, target, observed, std)
    new = flow_vector_mse(prediction, target, observed, selected)
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, prediction)[0], torch.autograd.grad(new, prediction)[0])


def test_physical_metric_has_original_loss_and_analytical_gradient():
    std = torch.tensor([1e-4, .2, 1.])
    raw_output = torch.tensor([[[.3, .2, -.1], [.5, .6, .7]]], requires_grad=True)
    mean = torch.tensor([.1, -.2, .4])
    prediction = mean + std * raw_output
    target = torch.tensor([[[.2, .3, .5], [float('nan'), .1, float('nan')]]])
    observed = torch.tensor([[[True, True, True], [False, True, False]]])
    before = prediction.detach().clone()
    rng = torch.get_rng_state().clone()
    selected = coordinate_flow_error_std(std, mode='physical')
    assert selected is None
    loss = flow_vector_mse(prediction, target, observed, selected)
    reference = ((prediction[observed]-target[observed])**2).mean()
    assert torch.equal(loss, reference)
    gradient = torch.autograd.grad(loss, raw_output)[0]
    clean_error = torch.where(observed, prediction-target, 0.)
    expected = 2 * clean_error * std / observed.sum()
    torch.testing.assert_close(gradient, expected, rtol=1e-6, atol=1e-9)
    assert not gradient[~observed].any() and torch.isfinite(gradient).all()
    assert torch.equal(prediction.detach(), before) and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('std,mode', [(torch.tensor([0., 1.]), 'physical'),
    (torch.tensor([float('nan')]), 'coordinate'), (torch.ones(2,2), 'physical'),
    (torch.ones(2), 'unknown')])
def test_invalid_error_unit_contract_rejected(std, mode):
    with pytest.raises(ValueError):
        coordinate_flow_error_std(std, mode=mode)
