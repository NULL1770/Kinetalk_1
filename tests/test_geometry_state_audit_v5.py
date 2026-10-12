import numpy as np
from scripts.audit_v5_geometry import geometry_error


def test_geometry_bias_plus_centered_error_recovers_total_mse_with_missing_axes():
    target=np.array([[1.,2.,0.,np.nan],[2.,4.,0.,np.nan],[999.,999.,999.,np.nan]])
    pred=np.array([[4.,4.,0.,np.nan],[5.,2.,0.,np.nan],[-999.,-999.,-999.,np.nan]])
    mask=np.array([[True,True,True,False],[True,True,True,False],[False,False,False,False]])
    out=geometry_error(pred,target,mask)
    width=out['width_mm'];lift=out['corner_lift_mm']
    assert width['bias']==3. and width['MSE']==9. and width['centered_MSE']==0.
    assert lift['bias']==0. and lift['MSE']==4. and lift['centered_MSE']==4.
    assert out['corner_depth_mm']['n']==0 and out['corner_depth_mm']['MAE'] is None
    assert width['MSE']==width['bias']**2+width['centered_MSE']
