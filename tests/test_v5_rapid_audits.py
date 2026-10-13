import pytest
import torch
from scripts.audit_v5_style_lips import pattern_scores
from scripts.audit_v5_semantic_control import native_statistics


def test_style_scale_preserves_time_pattern_but_reports_size_change():
    x=torch.tensor([0.,1.,2.,3.,2.,1.,0.,1.])
    v=torch.ones(8,dtype=torch.bool);t=torch.arange(8,dtype=torch.float64)/25
    s=pattern_scores(x,2*x+4,v,t)
    assert s['aperture_pattern_correlation']==pytest.approx(1.,abs=1e-6)
    assert s['aperture_velocity_cosine']==pytest.approx(1.,abs=1e-6)
    assert s['low_aperture_rank_f1']==1.
    assert s['aperture_amplitude_ratio']==pytest.approx(2.)


def test_dynamic_statistics_do_not_count_timestamp_gap_as_velocity():
    y=torch.tensor([[0.],[1.],[2.],[200.],[201.],[202.]])
    valid=torch.ones(6,dtype=torch.bool)
    times=torch.tensor([0.,.04,.08,1.,1.04,1.08],dtype=torch.float64)
    s=native_statistics(y,valid,times,torch.tensor([True]))
    assert s['speed']==1.
    assert s['acceleration']==0.
