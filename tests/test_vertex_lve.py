import numpy as np

from scripts.evaluate_vertex_lve import evaluate


def test_vertex_lve_zero_and_known_delta():
    neutral = np.zeros((2, 3))
    deltas = np.zeros((52, 2, 3))
    deltas[0, 0, 0] = 1.
    target = np.zeros((2, 52)); target[0, 0] = 1.
    pred = target.copy()
    lip = np.array([True, False])
    result = evaluate(pred, target, np.array([True, True]), np.ones(52, bool), neutral, deltas,
                      lip_mask=lip, eye_forehead_mask=lip)
    assert result["vertex_lve_sqrt_mean"] == 0.
    pred[1, 0] = 1.
    result = evaluate(pred, target, np.array([True, True]), np.ones(52, bool), neutral, deltas,
                      lip_mask=lip, eye_forehead_mask=lip)
    assert result["vertex_lve_sqrt_mean"] > 0.


def test_vertex_lve_accepts_time_varying_channel_mask():
    neutral = np.zeros((1, 3))
    deltas = np.zeros((52, 1, 3))
    deltas[0, 0, 0] = 1.
    target = np.zeros((2, 52)); target[1, 0] = 1.
    pred = target.copy(); pred[0, 0] = 1.
    # Frame 0 is unobserved for channel 0, so its arbitrary prediction must
    # not contribute to the vertex error.
    mask = np.ones((2, 52), dtype=bool); mask[0, 0] = False
    result = evaluate(pred, target, np.ones(2, bool), mask, neutral, deltas,
                      lip_mask=np.ones(1, bool), eye_forehead_mask=np.ones(1, bool))
    assert result["facediffuser_mve_mean"] == 0.
