import numpy as np
import pytest
from scripts.evaluate_paper_coefficients import coefficient_metrics, cluster_interval, UPPER

def inputs():
    y=np.zeros((6,52));y[:,UPPER]=np.arange(6)[:,None]/10
    return y.copy()[None],y,np.ones_like(y,dtype=bool),np.arange(6)/25

def test_perfect_and_signed_gap():
    x,y,m,t=inputs();s=coefficient_metrics(x,y,m,t)
    assert s['upper_mae']==0 and s['upper_std_absolute_gap']==0
    x[:]=0;s=coefficient_metrics(x,y,m,t)
    assert s['upper_std_signed_gt_minus_pred']==pytest.approx(np.std(np.arange(6)/10))

def test_reversal_retains_amplitude_but_changes_alignment():
    x,y,m,t=inputs();s=coefficient_metrics(x[:,::-1],y,m,t)
    assert s['upper_std_absolute_gap']==pytest.approx(0,abs=1e-15)
    assert s['upper_mae']>0 and s['upper_velocity_mae_per_second']>0

def test_target_fills_do_not_enter_scores_or_cross_gap_velocity():
    x,y,m,t=inputs();m[2:4]=False;x[:,2:4]=999;y[2:4]=-999
    s=coefficient_metrics(x,y,m,t)
    assert s['upper_mae']==0 and s['upper_velocity_mae_per_second']==0

def test_draw_scores_not_ensemble_mean():
    x,y,m,t=inputs();s=coefficient_metrics(np.concatenate((x+.1,x-.1)),y,m,t)
    assert s['all52_mae']==pytest.approx(.1)
    assert s['upper_pairwise_rms_diversity']==pytest.approx(.2)

def test_missing_and_invalid_observed():
    x,y,m,t=inputs();m[:]=False
    assert coefficient_metrics(x,y,m,t)['upper_mae'] is None
    m[1,1]=True;x[0,1,1]=np.nan
    with pytest.raises(ValueError):coefficient_metrics(x,y,m,t)

def test_cluster_bootstrap_equal_clip_mean():
    s=cluster_interval([1,3,8],['a','a','b'])
    assert s['mean']==4 and s['ci95']==[2,8]
