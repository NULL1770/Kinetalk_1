import numpy as np

from scripts.audit_semantic_code_subspace import semantic_projection, error_parts, probabilities


def test_fixed_semantic_projection_preserves_both_probability_heads():
    rng = np.random.default_rng(92)
    ew, iw = rng.normal(size=(8, 64)), rng.normal(size=(4, 64))
    p, _, rank = semantic_projection(ew, iw)
    assert rank == 10
    code, mean = rng.normal(size=(20, 64)), rng.normal(size=64)
    projected = mean + (code - mean) @ p
    for weight in (ew, iw):
        bias = rng.normal(size=len(weight))
        np.testing.assert_allclose(probabilities(code, weight, bias),
                                   probabilities(projected, weight, bias), atol=1e-12)


def test_projection_error_decomposition_with_redundant_head_directions():
    ew = np.array([[1., 2., 3.], [-1., -2., -3.]])
    iw = 2 * ew
    p, _, rank = semantic_projection(ew, iw)
    assert rank == 1
    row = error_parts(np.eye(3), np.zeros((3, 3)), p)
    np.testing.assert_allclose(row['mse'], row['head_rowspace_mse'] + row['other_directions_mse'])
    np.testing.assert_allclose(row['head_rowspace_error_fraction'], 1/3)
