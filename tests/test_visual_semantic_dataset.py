import numpy as np
from scripts.build_visual_semantic_dataset import align_visual


def test_only_bracketed_observations_no_extrapolation():
    t=np.arange(12)/25;v=np.ones(12,bool)
    y,m=align_visual(np.array([0,.2,.4]),np.ones(3,bool),np.array([[0.],[1.],[0.]]),t,v)
    assert m[:11].all() and not m[11]
    np.testing.assert_allclose(y[:6,0],np.arange(6)/5,atol=1e-7)


def test_missing_visual_does_not_become_interpolation_or_static():
    t=np.arange(11)/25;v=np.ones(11,bool)
    y,m=align_visual(np.array([0,.2,.4]),np.array([True,False,True]),np.array([[0.],[np.nan],[1.]]),t,v)
    np.testing.assert_array_equal(np.flatnonzero(m),[0,10])


def test_native_gap_is_not_bridged():
    t=np.arange(6)/25;v=np.ones(6,bool);v[2]=False
    y,m=align_visual(np.array([0,.2]),np.ones(2,bool),np.array([[0.],[1.]]),t,v)
    np.testing.assert_array_equal(np.flatnonzero(m),[0,5])
