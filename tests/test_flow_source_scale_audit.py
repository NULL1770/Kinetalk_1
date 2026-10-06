import numpy as np
import pytest
from scripts.audit_flow_source_scale import ChannelMoments


def test_train_moments_ignore_missing_values_and_match_population_statistics():
    x = np.array([[[1., 2., np.nan], [3., np.nan, np.inf]], [[5., 6., np.nan], [7., 8., np.nan]]])
    mask = np.isfinite(x)
    moments = ChannelMoments(3)
    for value, observed in zip(x, mask):
        moments.add(value, observed)
    row = moments.result(.25)
    np.testing.assert_allclose(row['residual_mean'], [4., 16/3, 0.])
    np.testing.assert_allclose(row['residual_std'][:2], [np.std([1., 3., 5., 7.]), np.std([2., 6., 8.])])
    assert row['count'] == [4, 3, 0] and row['normalized_source_std'][2] == 1.


def test_constant_channel_keeps_nonsingular_source_and_invalid_observations_fail():
    moments = ChannelMoments(2)
    moments.add(np.array([[1., 2.], [1., 2.]]), np.ones((2, 2), bool))
    assert moments.result(.25)['normalized_source_std'] == [1e-4, 1e-4]
    with pytest.raises(ValueError):
        moments.add(np.array([[np.nan, 0.]]), np.ones((1, 2), bool))
