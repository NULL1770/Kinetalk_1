import pytest
import torch
from kinetalk_v5.feasible_update import project_descent_step


def test_already_feasible_and_zero_gradients_preserve_proposed_step():
    d = torch.tensor([-2., -1.])
    assert torch.equal(project_descent_step(d, torch.tensor([1., 0.]), torch.tensor([0., 1.])), d)
    assert torch.equal(project_descent_step(d, torch.zeros(2), torch.zeros(2)), d)


def test_projection_satisfies_both_conflicting_objectives_and_keeps_orthogonal_freedom():
    d = torch.tensor([2., 1., 4.])
    a, b = torch.tensor([1., 0., 0.]), torch.tensor([-1., 1., 0.])
    x = project_descent_step(d, a, b)
    assert (a @ x) <= 1e-6 and (b @ x) <= 1e-6
    assert torch.allclose(x, torch.tensor([0., 0., 4.]), atol=1e-6)
    assert torch.allclose(project_descent_step(d, 100*a, .001*b), x, atol=1e-6)


def test_opposed_objectives_preserve_shared_tangent_and_reject_bad_vectors():
    a = torch.tensor([1., 0.])
    assert torch.equal(project_descent_step(torch.tensor([2., 3.]), a, -a), torch.tensor([0., 3.]))
    with pytest.raises(ValueError, match='flat vectors'):
        project_descent_step(torch.ones(2), torch.ones(3), torch.ones(2))
