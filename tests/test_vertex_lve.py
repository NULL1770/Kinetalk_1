import numpy as np
import pytest

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


def test_vertex_paper_metrics_units_masks_and_fdd_are_declared():
    neutral = np.zeros((2, 3))
    deltas = np.zeros((52, 2, 3))
    deltas[0, 0, 0] = 1.0
    deltas[1, 1, 1] = 1.0
    target = np.zeros((3, 52))
    target[:, 0] = [0.0, 0.5, 1.0]
    target[:, 1] = [0.0, 0.25, 0.5]
    prediction = target.copy()
    prediction[1, 0] += 0.1
    result = evaluate(
        prediction, target, np.ones(3, bool), np.ones(52, bool), neutral, deltas,
        lip_mask=np.array([True, False]),
        expression_mask=np.array([False, True]),
        coordinate_scale_to_mm=1000.0,
        coordinate_unit="m",
    )
    assert result["lve_mean_mm_mean"] == pytest.approx(100.0 / 3.0)
    assert result["eve_mean_mm_mean"] == 0.0
    assert result["vertex_fdd_status"] == "computed"
    assert result["reported_distance_unit"] == "mm"
    assert result["reported_fdd_unit"] == "mm^2"
    assert len(result["rig_identity"]["rig_sha256"]) == 64


def test_vertex_evaluator_allows_nonfinite_unobserved_coefficients():
    neutral = np.zeros((1, 3))
    deltas = np.zeros((52, 1, 3))
    target = np.zeros((2, 52))
    prediction = target.copy()
    target[0, 2] = np.nan
    prediction[0, 2] = np.inf
    mask = np.ones((2, 52), dtype=bool)
    mask[0, 2] = False
    result = evaluate(prediction, target, np.ones(2, bool), mask, neutral, deltas,
                      lip_mask=np.ones(1, bool), expression_mask=np.ones(1, bool))
    assert result["facediffuser_mve_mean"] == 0.0


def test_fdd_requires_all_declared_train_support_not_unobserved_tongue():
    support=np.array([True]*51+[False]);neutral=np.zeros((1,3));deltas=np.zeros((52,1,3))
    deltas[0,0,0]=1.
    target=np.zeros((3,52));target[:,0]=[0,.5,1];pred=target.copy();pred[:,0]*=.5
    kwargs=dict(lip_mask=np.ones(1,bool),expression_mask=np.ones(1,bool))
    missing=evaluate(pred,target,np.ones(3,bool),support,neutral,deltas,**kwargs)
    assert missing['vertex_fdd_absolute_mm2_mean'] is None
    result=evaluate(pred,target,np.ones(3,bool),support,neutral,deltas,coefficient_support=support,**kwargs)
    assert result['vertex_fdd_absolute_mm2_mean']>0 and result['fdd_complete_frames']==3
