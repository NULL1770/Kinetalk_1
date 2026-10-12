import torch
from kinetalk_v4.geometry_support import clean_geometry_jacobian, geometry_support


def test_unobserved_numerical_noise_does_not_invalidate_geometry():
    jac = torch.tensor([[3., 6.148e-7], [2., 2.908e-6]])
    channels = torch.tensor([[True, False]])
    cleaned = clean_geometry_jacobian(jac)
    assert geometry_support(cleaned, channels).all()
    assert torch.equal(cleaned[:, 0], jac[:, 0])
    assert cleaned[:, 1].eq(0).all()


def test_substantive_missing_dependency_is_still_masked():
    jac = torch.tensor([[3., .1], [2., 2.908e-6]])
    channels = torch.tensor([[True, False], [True, True]])
    assert torch.equal(geometry_support(jac, channels), torch.tensor([[False, True], [True, True]]))


def test_cleaned_geometry_error_is_bounded_by_asset_tolerance():
    torch.manual_seed(47)
    jac = torch.randn(4, 52)
    jac[:, :20] *= 1e-6
    x = torch.rand(80, 52)
    clean = clean_geometry_jacobian(jac)
    assert ((x @ jac.T) - (x @ clean.T)).abs().max() < 52e-4
