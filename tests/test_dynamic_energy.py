import torch

from kinetalk_b0.dynamic_energy import fit_control_probe, projected_energy, upper_indices, upper_l1_field


def test_upper_field_is_centered_and_masked():
    residual = torch.zeros(2, 5, 52)
    residual[:, :, 0] = torch.arange(5)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1, -1] = False
    field, weight = upper_l1_field(residual, valid, stride=2)
    assert field.shape == (2, 3, 1)
    assert weight.shape == (2, 3)
    assert field[1, -1].eq(0).all()
    means = (field.squeeze(-1) * weight).sum(1) / weight.sum(1)
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-6, rtol=0)


def test_control_probe_recovers_known_direction():
    torch.manual_seed(3)
    controls = torch.randn(4, 6, 3)
    weight = torch.ones(4, 6)
    direction = torch.tensor([.5, -.25, .75])
    target = projected_energy(controls, direction).unsqueeze(-1)
    fitted = fit_control_probe(controls, target, weight, ridge=1e-6)
    torch.testing.assert_close(fitted, direction, atol=2e-4, rtol=2e-4)


def test_upper_indices_are_bounded_and_unique():
    result = upper_indices(10)
    assert result.tolist() == list(range(10))
    assert len(set(result.tolist())) == len(result)

