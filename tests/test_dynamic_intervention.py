import torch

from scripts.evaluate_dynamic_intervention import _response


def _curves():
    b,t,d=2,6,52
    valid=torch.ones(b,t,dtype=torch.bool); cm=torch.ones(b,d,dtype=torch.bool)
    target=torch.zeros(b,t,d); b0=torch.zeros_like(target)
    full=torch.zeros_like(target); full[:,:,0]=torch.arange(t)[None]
    zero=torch.zeros_like(target); mean=torch.zeros_like(target); rev=full.flip(1)
    out={"target":target,"b0":b0,"valid":valid,"channel_mask":cm,"times":torch.arange(t)[None].repeat(b,1)*.04}
    for s in ("audio","teacher"):
        out.update({f"{s}_full":full,f"{s}_zero":zero,f"{s}_mean":mean,f"{s}_reverse":rev})
    return out

def test_response_detects_dynamic_intervention_and_regions():
    result=_response(_curves(),"audio")
    assert result["full_vs_zero_mse"] > 0
    assert result["full_vs_mean_mse"] > 0
    assert result["full_vs_reverse_mse"] > result["full_vs_zero_mse"]
    assert result["full_upper_response_mse"] > 0
    assert result["full_brows_response_mse"] == 0
    assert len(result["per_clip"]) == 2

def test_response_uses_valid_channel_mask():
    c=_curves(); c["channel_mask"][:,0]=False
    result=_response(c,"audio")
    assert result["full_vs_zero_mse"] == 0
    assert result["full_upper_response_mse"] == 0
