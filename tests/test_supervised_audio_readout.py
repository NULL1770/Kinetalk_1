import torch
from scripts.train_supervised_audio_readout import SupervisedAudioReadout
from kinetalk_b0.models.relative_audio_timing import relative_features, masked_center
from kinetalk_b0.models.slow_state_affect import masked_slow_state


def test_linear_projection_matches_inference_with_gaps_and_padding():
    torch.manual_seed(27); torch.set_num_threads(1)
    x=torch.randn(2, 41, 1540); x[..., -1]=1
    valid=torch.ones(2,41,dtype=torch.bool); valid[0,10:13]=False; valid[1,32:]=False
    x[~valid]=float('nan')
    mean=torch.zeros(1540); std=torch.ones(1540); coefficient=torch.randn(1540,4)*.002
    model=SupervisedAudioReadout(mean,std,coefficient,torch.ones(52))
    clean=relative_features(x,valid,mean,std)
    proj=masked_slow_state(torch.eye(41)[None].expand(2,-1,-1),valid)['state']
    expected=masked_center(proj@clean,valid)@coefficient
    out=model(x,valid)
    torch.testing.assert_close(out['state'],expected,atol=1e-6,rtol=1e-5)
    assert torch.isfinite(out['state']).all()
    assert out['state'][~valid].count_nonzero()==0
    assert model(x,valid,'static')['state'].count_nonzero()==0
    assert out['delta'][...,14:41].count_nonzero()==0
