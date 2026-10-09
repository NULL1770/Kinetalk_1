import numpy as np
import pytest
from scripts.audit_paired_target_reliability import window_moments, rank_corr, target_metrics


@pytest.mark.parametrize('width',[1,5,11])
def test_strict_windows_agree_with_independent_loop_and_reject_clock_gaps(width):
    rng=np.random.default_rng(3);x=rng.normal(size=(31,3));mask=np.ones_like(x,bool)
    mask[6:8,1]=False;mask[:,2]=False;x[~mask]=np.nan
    times=np.arange(31)*.04;times[15:]+=.08
    mean,std,valid=window_moments(x,mask,times,width)
    for i in range(len(x)):
        for c in range(3):
            lo,hi=i-width//2,i+width//2+1
            allowed=lo>=0 and hi<=len(x) and mask[lo:hi,c].all() and np.allclose(np.diff(times[lo:hi]),.04)
            assert bool(valid[i,c])==allowed
            if allowed:
                np.testing.assert_allclose(mean[i,c],x[lo:hi,c].mean(),atol=1e-12)
                np.testing.assert_allclose(std[i,c],x[lo:hi,c].std(),atol=1e-7)
            else:assert mean[i,c]==0 and std[i,c]==0


def test_static_posture_is_not_dynamic_energy():
    times=np.arange(25)*.04;x=np.full((25,2),3.);n=np.ones_like(x);mask=np.ones_like(x,bool)
    r=target_metrics(x,n,mask,times)
    assert r['centered_energy']==0 and r['displacement_energy']==0
    for w in (1,5,11):
        s=r['w'+str(w)]
        assert s['centered_energy_ratio'] is None and s['shift_error_ratio'] is None
        assert s['shift_error']==0


def test_zero_neutral_velocity_has_zero_shift_sensitivity_even_for_expressive_query():
    times=np.arange(45)*.04;x=np.sin(times*3)[:,None];n=np.ones_like(x)*.2;m=np.ones_like(x,bool)
    r=target_metrics(x,n,m,times)
    for width in (1,5,11):
        z=r['w'+str(width)]
        assert z['shift_error']==0 and z['shift_displacement_error']==0
        assert z['shift_error_ratio']==0 and z['shift_local_std_error']==0


def test_known_linear_teacher_shift_magnitude_and_missing_support():
    times=np.arange(25)*.04;n=np.arange(25.)[:,None]*.1;x=2*n;mask=np.ones_like(x,bool)
    r=target_metrics(x,n,mask,times)
    np.testing.assert_allclose(r['w1']['shift_error'],.01,atol=1e-15)
    assert r['w1']['shift_observations']==23
    assert r['w5']['observations']==21 and r['w5']['shift_observations']==19
    assert r['w11']['observations']==15 and r['w11']['shift_observations']==13
    assert r['w1']['shift_displacement_error']<1e-28


def test_fast_jitter_attenuates_more_than_slow_expression_and_offset_is_irrelevant():
    times=np.arange(101)*.04;m=np.ones((101,1),bool)
    slow=np.sin(np.arange(101)*.07)[:,None];fast=((-1.)**np.arange(101))[:,None]
    a=target_metrics(slow,np.zeros_like(slow),m,times)
    b=target_metrics(fast,np.zeros_like(fast),m,times)
    assert a['w5']['centered_energy_ratio']>.9
    assert b['w5']['centered_energy_ratio']<.05
    offset=target_metrics(slow+20,np.zeros_like(slow),m,times)
    np.testing.assert_allclose(a['w5']['centered_energy_ratio'],offset['w5']['centered_energy_ratio'],atol=1e-12)


def test_mask_and_native_gap_do_not_become_observations():
    times=np.arange(20)*.04;times[10:]+=.04
    x=np.ones((20,2));m=np.ones_like(x,bool);m[:,1]=False;m[5]=False;x[~m]=np.nan
    r=target_metrics(x,np.zeros_like(x),m,times)
    assert r['native_observations']==19
    assert r['adjacent_observations']==16
    assert r['w5']['observations']==6+1  # [0:5], [6:10] too short, [10:20]
    assert r['w11']['observations']==0
    assert r['w11']['centered_energy_ratio'] is None


def test_tied_rank_correlation_uses_average_ranks():
    assert rank_corr([1,1,1],[1,2,3]) is None
    # Centered average ranks dot product=4, each squared norm=4.5.
    np.testing.assert_allclose(rank_corr([1,1,2,3],[1,2,3,3]),8/9)
    assert rank_corr([1,2,3],[3,2,1])<-0.99


@pytest.mark.parametrize('bad',['time','observed_nan','mask','even_window'])
def test_invalid_inputs_rejected(bad):
    x=np.zeros((7,2));m=np.ones_like(x,bool);times=np.arange(7)*.04;w=5
    if bad=='time':times[3]=times[2]
    elif bad=='observed_nan':x[0,0]=np.nan
    elif bad=='mask':m=m.astype(float)
    else:w=4
    with pytest.raises(ValueError):window_moments(x,m,times,w)
