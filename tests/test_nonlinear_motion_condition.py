import numpy as np
from scripts.probe_nonlinear_motion_condition import fit, predict


def test_predictor_can_fit_nonlinearity_and_static_input_has_no_time_pattern():
    rng = np.random.default_rng(25)
    x = rng.normal(size=(100, 4))
    y = np.repeat((x[:, :1]**2), 8, axis=1)
    checkpoint = fit(x, y, np.ones(100), seed=42, steps=300, device='cpu')
    prediction = predict(checkpoint, x)
    assert np.mean((prediction-y)**2) < np.mean((y-y.mean(0))**2) * .5
    constant = np.repeat(x[:1], 30, axis=0)
    value = predict(checkpoint, constant)
    np.testing.assert_array_equal(value, np.repeat(value[:1], 30, axis=0))
