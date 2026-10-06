import numpy as np

from scripts.audit_affect_generalization import error_partition


def test_unequal_group_sizes_preserve_exact_error_identity():
    errors = np.array([[1., 2.], [3., 4.], [5., 6.], [-1., -2.]])
    labels = np.array(['a', 'a', 'a', 'b'])
    out = error_partition(errors, labels)
    np.testing.assert_allclose(out['mse'], (errors ** 2).mean())
    np.testing.assert_allclose(out['mse'], out['group_bias_mse'] + out['centered_error_mse'])
    assert out['groups']['a']['clips'] == 3


def test_pure_group_offsets_have_no_within_group_error():
    errors = np.array([[1., 2.], [1., 2.], [-3., -4.]])
    out = error_partition(errors, [0, 0, 1])
    assert out['centered_error_mse'] == 0
    np.testing.assert_allclose(out['group_bias_fraction'], 1., rtol=0, atol=1e-14)
