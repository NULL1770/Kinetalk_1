import numpy as np
import pytest
from scripts.diagnose_paired_coordinates import coordinate_terms, aggregate


def test_nonorthogonal_coordinates_keep_cross_term():
    n=np.arange(6.)[:,None]*np.ones((1,52));c=np.zeros_like(n);x=3*n
    terms=coordinate_terms(x,n,c,np.ones_like(x,dtype=bool),np.arange(6)*.04)
    for kind in ('position','centered','displacement'):
        a=terms[kind]
        np.testing.assert_allclose(a[0],a[1]+a[2]+a[3])
        assert (a[3]>0).all()
    assert np.all(terms['adjacent']==5)


def test_invalid_channels_gaps_and_nan_are_not_observations():
    x=np.zeros((5,52));n=x.copy();c=x.copy();mask=np.ones_like(x,dtype=bool)
    mask[2]=False;mask[:,51]=False
    x[2]=np.nan;n[:,51]=np.nan;x[:,17]=[0,1,np.nan,3,4]
    r=coordinate_terms(x,n,c,mask,np.arange(5)*.04)
    assert r['adjacent'][17]==2 and r['observed'][51]==0
    assert r['displacement'][0,17]==1
    a=aggregate([dict(terms=r)])
    assert a['clips']==1
    bad=mask.copy();bad[2,0]=True
    with pytest.raises(ValueError):coordinate_terms(x,n,c,bad,np.arange(5)*.04)


def test_centering_removes_static_bias_without_retiming():
    c=np.zeros((6,52));n=c+.3;x=n+.7
    t=np.array([0,.04,.08,.20,.24,.28]);mask=np.ones_like(x,dtype=bool)
    r=coordinate_terms(x,n,c,mask,t)
    np.testing.assert_allclose(r['mean'],np.asarray([1,.7,.3])[:,None]*np.ones((1,52)))
    np.testing.assert_allclose(r['centered'],0,atol=1e-30)
    assert (r['adjacent']==4).all()


def test_opposing_components_can_exceed_total_energy():
    c=np.zeros((4,52));n=np.ones_like(c);x=c+.1
    r=coordinate_terms(x,n,c,np.ones_like(x,dtype=bool),np.arange(4)*.04)
    assert (r['position'][1]>r['position'][0]).all()
    assert (r['position'][3]<0).all()
    np.testing.assert_allclose(r['position'][0],.01)


@pytest.mark.parametrize('times', [[0,.04,.03], [0,.04,np.nan], [[0],[.04],[.08]]])
def test_reject_invalid_native_time(times):
    x=np.zeros((3,52))
    with pytest.raises(ValueError,match='time axis'):
        coordinate_terms(x,x,x,np.ones_like(x,dtype=bool),times)
