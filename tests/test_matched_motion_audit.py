import numpy as np
import pytest
import torch
from scripts.audit_matched_motion_curves import longest_valid_span, region_values
from scripts.audit_motion_noise_attribution import decompose, values
from scripts.diagnose_temporal_condition import temporal_variant

def test_native_interval_preserves_gaps_and_earliest_tie():
    assert longest_valid_span([False, True, True, False, True, True]) == (1, 3)
    assert longest_valid_span([True, False, True, True, True]) == (2, 5)
    with pytest.raises(ValueError):
        longest_valid_span([False, False])

def test_region_decomposes_constant_bias_without_fabricating_dynamics():
    target=torch.arange(6.).reshape(-1,1).expand(-1,52).clone()
    pred=target+2
    valid=torch.tensor([True,True,False,True,True,True])
    channel=torch.ones(52,dtype=torch.bool)
    pred[2]=float('nan')
    row=region_values(pred,target,valid,channel,[17])
    np.testing.assert_allclose(row[:3],[4,4,0],atol=1e-6)
    np.testing.assert_allclose(row[5:7],[1,0],atol=1e-6)
    np.testing.assert_allclose(row[3],row[4],atol=1e-6)
    np.testing.assert_allclose(row[7],row[8],atol=1e-6)

def test_unsupported_channels_do_not_change_region_error():
    target=torch.arange(6.).reshape(-1,1).expand(-1,52).clone()
    pred=target.clone(); pred[:,18]=float('nan')
    channel=torch.ones(52,dtype=torch.bool); channel[18]=False
    row=region_values(pred,target,torch.ones(6,dtype=torch.bool),channel,[17,18])
    np.testing.assert_allclose(row[:3],[0,0,0],atol=1e-6)
    np.testing.assert_allclose(row[6],0,atol=1e-6)

def test_three_draw_decomposition_exactly_separates_noise_and_bias():
    target=torch.tensor([[1.,2.],[2.,4.],[3.,6.]])
    draws=torch.stack([target-1,target,target+1])
    error,bias,variance,fraction=decompose(draws,target)
    np.testing.assert_allclose([error,bias,variance,fraction],[2/3,0,2/3,1],rtol=1e-6)

def test_noise_attribution_excludes_invalid_frames_and_channel_values():
    target=torch.arange(6.).reshape(-1,1).expand(-1,52).clone()
    draws=torch.stack([target-1,target,target+1])
    valid=torch.tensor([True,True,False,True,True,True])
    channel=torch.ones(52,dtype=torch.bool); channel[18]=False
    draws[:,2]=float('nan'); draws[:,:,18]=float('nan')
    row=values(draws,target,target,valid,channel,[17,18])
    np.testing.assert_allclose(row[:4],[2/3,0,2/3,1],rtol=1e-6)
    np.testing.assert_allclose(row[4:8],[0,0,0,0],atol=1e-7)

def test_temporal_order_controls_preserve_valid_frame_distribution():
    x=torch.arange(12.).reshape(1,6,2)
    valid=torch.tensor([[False,True,True,False,True,True]])
    original=x.clone()
    for mode in ('reverse','shuffle'):
        out=temporal_variant(x,valid,mode,torch.Generator().manual_seed(2))
        torch.testing.assert_close(out[valid].sort(0).values,x[valid].sort(0).values)
        assert (out[~valid]==0).all()
    static=temporal_variant(x,valid,'static')
    torch.testing.assert_close(static[valid],x[valid].mean(0).expand(4,2))
    assert (temporal_variant(x,valid,'zero')==0).all()
    torch.testing.assert_close(x,original)
