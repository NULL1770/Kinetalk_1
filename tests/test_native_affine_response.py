import copy
from dataclasses import replace
import pytest
import torch
from kinetalk_b0.models.expression_response import ExpressionResponse, ResponseConfig, clean
from scripts.train_native_affine_response import candidate, optimizer, objective, frozen_digest
from scripts.train_reference_response import candidate as statistical
from tests.test_reference_response import fixture


def setup():
    parent,_,b,r=fixture()
    parent=statistical(parent,'statistics','factorized').eval()
    return parent,candidate(parent).eval(),b,r


def test_legacy_default_and_replacement_drop_obsolete_decoder_parameters():
    parent,m,_,_=setup()
    assert parent.cfg.response_head=='residual' and m.cfg.response_head=='native_affine'
    assert not hasattr(m.decoder,'modulations') and not hasattr(m.decoder,'input')
    assert frozen_digest(parent)==frozen_digest(m)
    for n,v in parent.state_dict().items():
        if not n.startswith('decoder.') or n.startswith('decoder.bias.'):
            assert torch.equal(v,m.state_dict()[n])


def test_global_positive_gain_preserves_native_content_increment_and_peak():
    _,m,b,r=setup();v=b['valid']
    with torch.no_grad():
        m.decoder.gain[-1].bias.fill_(1000.)
        p=m.audio_prior(b['audio_features'],v);s=m.encode_style(r)['code'];g,u=m.conditions(p,v)
        gain,expression,_=m.decoder.components(g,u,s,v)
        assert (gain>=.25).all() and (gain<=4.000001).all() and gain.shape==(len(v),52)
        x=torch.arange(v.shape[1],dtype=b['b0'].dtype)[None,:,None].expand_as(b['b0'])
        y=m.decode(x,p,s,v);z=m.decode(x+.7,p,s,v)
        torch.testing.assert_close((z-y)[v],(.7*gain[:,None].expand_as(x))[v],rtol=2e-5,atol=2e-6)
        m.decoder.output.weight.zero_();m.decoder.output.bias.zero_();m.decoder.bias.weight.zero_();m.decoder.bias.bias.zero_()
        y=m.decode(x,p,s,v)
        assert torch.equal(y.argmax(1),clean(x,v).argmax(1))
        assert not y[~v].count_nonzero()


def test_expression_branch_cannot_read_query_content_and_invalid_values_are_isolated():
    _,m,b,r=setup();v=b['valid'].clone();v[0,3]=False
    with torch.no_grad():
        p=m.audio_prior(b['audio_features'],b['valid']);s=m.encode_style(r)['code'];g,u=m.conditions(p,v)
        first=m.decoder.components(g,u,s,v)
        dirty=u.clone();dirty[~v]=float('nan')
        second=m.decoder.components(g,dirty,s,v)
        for x,y in zip(first,second):torch.testing.assert_close(x,y,rtol=0,atol=0)
        base=b['b0'].clone();base[~v]=float('nan')
        y=m.decoder(base,g,dirty,s,v,m.scales)
        assert torch.isfinite(y).all() and not y[~v].count_nonzero()
        # Only the direct affine term can respond to an arbitrary B0 change.
        other=base.clone();other[v]=torch.randn_like(other[v])*3
        z=m.decoder(other,g,dirty,s,v,m.scales)
        torch.testing.assert_close((z-y)[v],((other-base)*first[0][:,None])[v],rtol=2e-5,atol=2e-6)


def test_motion_training_updates_only_new_response_and_never_calls_teacher():
    _,m,b,r=setup();params,opt=optimizer(m,1e-3);before=frozen_digest(m)
    b['b0']=b['b0'].clone().requires_grad_();b['audio_features']=b['audio_features'].clone().requires_grad_()
    def forbidden(*a,**kw):raise AssertionError('Teacher called by response training')
    m.motion_posterior=forbidden
    initial={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
    loss,_=objective(m,b,r);loss.backward()
    assert b['b0'].grad is None and b['audio_features'].grad is None
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('decoder.') or n.startswith('decoder.bias.'))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    opt.step();assert frozen_digest(m)==before
    assert any(not torch.equal(v,dict(m.named_parameters())[n]) for n,v in initial.items())


def test_strict_checkpoint_prediction_roundtrip_and_hubert_nan_isolation():
    _,m,b,r=setup()
    with torch.no_grad():
        m.decoder.gain[-1].bias.normal_();m.decoder.output.weight.normal_(std=.1)
        y=m.predict(b['audio_features'],b['b0'],b['valid'],r)
        restored=ExpressionResponse(ResponseConfig(**m.checkpoint_config()),m.feature_mean,m.feature_std,m.scales).eval()
        restored.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(y,restored.predict(b['audio_features'],b['b0'],b['valid'],r),rtol=0,atol=0)
        audio=b['audio_features'].clone();audio[...,:768]=float('nan')
        torch.testing.assert_close(y,m.predict(audio,b['b0'],b['valid'],r),rtol=0,atol=0)


def test_adam_state_resume_matches_uninterrupted_next_update():
    _,m,b,r=setup();_,opt=optimizer(m,1e-3)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss,_=objective(m,b,r);loss.backward();opt.step()
    ck=copy.deepcopy(m.state_dict());os=copy.deepcopy(opt.state_dict())
    restored=ExpressionResponse(ResponseConfig(**m.checkpoint_config()),m.feature_mean,m.feature_std,m.scales).eval()
    restored.load_state_dict(ck,strict=True);_,second=optimizer(restored,1e-3);second.load_state_dict(os)
    for model,optim in [(m,opt),(restored,second)]:
        optim.zero_grad(set_to_none=True);loss,_=objective(model,b,r);loss.backward();optim.step()
    for n,v in m.state_dict().items():assert torch.equal(v,restored.state_dict()[n]),n


def test_invalid_new_head_and_incompatible_reference_config_are_rejected():
    for cfg in [ResponseConfig(response_head='unknown'),ResponseConfig(response_head='native_affine')]:
        with pytest.raises(ValueError):ExpressionResponse(cfg,torch.zeros(772),torch.ones(772),torch.ones(52))
