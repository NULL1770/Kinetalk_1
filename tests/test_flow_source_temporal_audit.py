import numpy as np
from scripts.audit_flow_source_temporal import clip_lag_moments, summarize_lags


def test_clip_means_missing_frames_and_padding_do_not_create_lag_signal():
    x=np.array([[1.,0.],[2.,1.],[np.nan,np.nan],[4.,1.],[5.,0.]])
    mask=np.isfinite(x)
    moments=clip_lag_moments(x,mask)
    shifted=np.where(mask,x+1000.,np.nan)
    np.testing.assert_allclose(moments,clip_lag_moments(shifted,mask),atol=1e-12)
    np.testing.assert_allclose(moments,clip_lag_moments(np.pad(x,((0,3),(0,0)),constant_values=np.nan),np.pad(mask,((0,3),(0,0)))))
    assert moments[0].tolist()==[2,2]
    np.testing.assert_allclose(moments[4],moments[2]+moments[3]-2*moments[1])


def test_constant_channel_rho_zero_and_nondegenerate_bound():
    row=summarize_lags(clip_lag_moments(np.ones((5,2)),np.ones((5,2),bool)))
    assert row['source_rho']==[0.,0.]
    moments=np.array([[3.],[3.],[3.],[3.],[0.]])
    assert summarize_lags(moments)['source_rho']==[.9999]
