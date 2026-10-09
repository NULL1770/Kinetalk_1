"""The new head must be an identity at start and cannot retrain old pathways."""
import copy
from dataclasses import replace
import torch
from kinetalk_b0.models.expression_response import ExpressionResponse, ResponseConfig
from scripts.train_native_affine_response import candidate, optimizer, objective, frozen_digest
from scripts.train_reference_response import candidate as statistical
from tests.test_reference_response import fixture


def setup():
    parent, _, b, r = fixture()
    parent = statistical(parent, 'statistics', 'factorized').eval()
    return parent, candidate(parent, 'bounded_residual').eval(), b, r


def test_zero_gain_preserves_every_parent_tensor_and_predictions_exactly():
    parent, m, b, r = setup()
    assert set(m.state_dict())-set(parent.state_dict()) == {
        'decoder.residual_gain.weight', 'decoder.residual_gain.bias'}
    assert all(torch.equal(v, m.state_dict()[n]) for n,v in parent.state_dict().items())
    with torch.no_grad():
        assert torch.equal(parent.predict(b['audio_features'], b['b0'], b['valid'], r),
                           m.predict(b['audio_features'], b['b0'], b['valid'], r))


def test_branch_constructor_preserves_legacy_rng_and_common_initialization():
    cfg = ResponseConfig(hidden=8, decoder_hidden=8)
    torch.manual_seed(13)
    a = ExpressionResponse(cfg, torch.zeros(772), torch.ones(772), torch.ones(52))
    rng = torch.random.get_rng_state()
    torch.manual_seed(13)
    b = ExpressionResponse(replace(cfg,response_head='bounded_residual'), a.feature_mean,a.feature_std,a.scales)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items())


def test_gain_bounds_apply_to_residual_not_base_or_posture_and_all_channels_remain_open():
    parent, m, b, r = setup(); valid=b['valid']
    with torch.no_grad():
        # A nonzero posture lets this test detect accidental offset scaling.
        parent.decoder.bias.bias.fill_(.4);m.decoder.bias.bias.copy_(parent.decoder.bias.bias)
        p=m.audio_prior(b['audio_features'],valid);s=m.encode_style(r)['code']
        old=parent.decode(b['b0'],p,s,valid)
        offset=m.decoder.bias(s.chunk(2,-1)[0])[:,None]*m.scales
        residual=old-b['b0']-offset
        for bias,gain in [(-1000.,.5),(1000.,1.5)]:
            m.decoder.residual_gain.bias.fill_(bias)
            new=m.decode(b['b0'],p,s,valid)
            expected=b['b0']+offset+gain*residual
            torch.testing.assert_close(new[valid],expected[valid],rtol=2e-5,atol=2e-6)
            assert not new[~valid].count_nonzero()
            assert ((new-old)[valid].abs().sum(0)>0).all()


def test_new_gain_input_is_content_free_and_masked_but_output_is_not_claimed_timing_invariant():
    _,m,b,r=setup();v=b['valid'].clone();v[0,3]=False
    with torch.no_grad():
        p=m.audio_prior(b['audio_features'],b['valid']);s=m.encode_style(r)['code'];g,u=m.conditions(p,v)
        dirty=u.clone();dirty[~v]=float('nan')
        seen=[]
        hook=m.decoder.residual_gain.register_forward_pre_hook(lambda mod,args:seen.append(args[0].clone()))
        base=b['b0'].clone();base[~v]=float('nan')
        first=m.decoder(base,g,dirty,s,v,m.scales)
        other=base.clone();other[v]=torch.randn_like(other[v])*3
        second=m.decoder(other,g,dirty,s,v,m.scales);hook.remove()
        assert torch.equal(seen[0],seen[1]) and not seen[0][~v].count_nonzero()
        assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_training_updates_only_gain_and_does_not_read_teacher_or_send_motion_gradients():
    _,m,b,r=setup();params,opt=optimizer(m,1e-3);digest=frozen_digest(m)
    b['b0']=b['b0'].clone().requires_grad_();b['audio_features']=b['audio_features'].clone().requires_grad_()
    m.motion_posterior=lambda *args: (_ for _ in ()).throw(AssertionError('teacher called'))
    loss,_=objective(m,b,r);loss.backward()
    assert b['b0'].grad is None and b['audio_features'].grad is None
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('decoder.residual_gain.'))
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    opt.step();assert frozen_digest(m)==digest


def test_checkpoint_optimizer_resume_and_hubert_nan_isolation():
    _,m,b,r=setup();_,opt=optimizer(m,1e-3)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);objective(m,b,r)[0].backward();opt.step()
    state=copy.deepcopy(m.state_dict());os=copy.deepcopy(opt.state_dict())
    restored=ExpressionResponse(ResponseConfig(**m.checkpoint_config()),m.feature_mean,m.feature_std,m.scales).eval()
    restored.load_state_dict(state,strict=True);_,other=optimizer(restored,1e-3);other.load_state_dict(os)
    for model,optim in [(m,opt),(restored,other)]:
        optim.zero_grad(set_to_none=True);objective(model,b,r)[0].backward();optim.step()
    assert all(torch.equal(v,restored.state_dict()[n]) for n,v in m.state_dict().items())
    with torch.no_grad():
        audio=b['audio_features'].clone();audio[...,:768]=float('nan')
        assert torch.equal(m.predict(audio,b['b0'],b['valid'],r),m.predict(b['audio_features'],b['b0'],b['valid'],r))
