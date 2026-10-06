import copy
import pytest
import torch
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from tests.test_full_staged_runner import config
from scripts.train_full_staged import load_warm_system, parser

def test_adapter_is_opt_in_zero_start_and_preserves_rng_and_output():
    torch.set_num_threads(1); torch.manual_seed(91)
    system=NeutralAffectSystem(config()).eval()
    assert system.renderer.motion_temporal is None
    state=copy.deepcopy(system.state_dict())
    before=torch.random.get_rng_state()
    system.renderer.set_temporal_adapter(True)
    torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)
    load_warm_system(system,state,allow_zero_temporal_adapter=True)
    baseline=NeutralAffectSystem(config()).eval(); baseline.load_state_dict(state,strict=True)
    args=(torch.randn(2,9,52),torch.tensor([.2,.7]),torch.randn(2,9,8),
          torch.randn(2,8),torch.randn(2,1),torch.randn(2,8),torch.ones(2,9,dtype=torch.bool))
    torch.testing.assert_close(system.renderer(*args),baseline.renderer(*args),rtol=0,atol=0)
    assert parser().parse_args(['--output','tmp']).renderer_temporal_adapter is False

def test_new_branch_gets_finite_gradient_and_masks_padding():
    torch.manual_seed(92)
    system=NeutralAffectSystem(config()).eval(); r=system.renderer; r.set_temporal_adapter(True)
    x=torch.randn(1,8,52); content=torch.randn(1,8,8)
    valid=torch.tensor([[True,True,True,True,True,False,False,False]])
    args=(torch.tensor([.4]),content,torch.randn(1,8),torch.randn(1,1),torch.randn(1,8),valid)
    pred=r(x,*args)
    pred[valid].square().mean().backward()
    grad=r.motion_temporal.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    with torch.no_grad():
        r.motion_temporal.weight.normal_(std=.03)
    original=r(x,*args)
    changed_x=x.clone(); changed_x[:,5:]=1000
    changed_content=content.clone(); changed_content[:,5:]=-1000
    changed=r(changed_x,args[0],changed_content,*args[2:])
    torch.testing.assert_close(original[valid],changed[valid],rtol=0,atol=0)
    assert (changed[~valid]==0).all()

def test_warm_load_allows_only_explicit_zero_adapter_and_roundtrips():
    system=NeutralAffectSystem(config()).eval(); state=copy.deepcopy(system.state_dict())
    system.renderer.set_temporal_adapter(True)
    with pytest.raises(RuntimeError):
        load_warm_system(system,state)
    broken={k:v for k,v in state.items() if k!='renderer.output.bias'}
    with pytest.raises(ValueError,match='mismatch'):
        load_warm_system(system,broken,allow_zero_temporal_adapter=True)
    load_warm_system(system,state,allow_zero_temporal_adapter=True)
    cfg=config(); cfg['model']['renderer_temporal_adapter']=True
    restored=NeutralAffectSystem(cfg).eval(); restored.load_state_dict(system.state_dict(),strict=True)
    torch.testing.assert_close(restored.renderer.motion_temporal.weight,system.renderer.motion_temporal.weight)
