from dataclasses import replace
import pytest
import torch
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig
from scripts.train_reference_response import candidate,reference_optimizer,objective,mask_reference_gradients,mask_posture_gradients,protected_receiver_digest,frozen_digests
from tests.test_reference_response import fixture


def setup():
    parent,_,b,r=fixture();m=candidate(parent,'statistics','factorized')
    b['emotion_id']=torch.tensor([0,1]);return parent,m,b,r


def test_zero_reference_factorization_preserves_initial_prediction_and_roundtrip():
    parent,m,b,r=setup();old=candidate(parent,'statistics');old.load_state_dict(m.state_dict());m.eval();old.eval()
    with torch.no_grad():
        y=m.predict(b['audio_features'],b['b0'],b['valid'],r)
        torch.testing.assert_close(y,old.predict(b['audio_features'],b['b0'],b['valid'],r),rtol=1e-6,atol=2e-7)
        restored=ExpressionResponse(ResponseConfig(**m.checkpoint_config()),m.feature_mean,m.feature_std,m.scales).eval()
        restored.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(y,restored.predict(b['audio_features'],b['b0'],b['valid'],r),rtol=0,atol=0)
        b['audio_features'][...,:768]=float('nan')
        torch.testing.assert_close(y,m.predict(b['audio_features'],b['b0'],b['valid'],r),rtol=0,atol=0)


@pytest.mark.parametrize('scope',['joint','style_only'])
def test_neutral_posture_routing_keeps_forward_values_and_motion_gradient_boundary(scope):
    _,m,b,r=setup();params,opt=reference_optimizer(m,scope,1e-3)
    m.eval();frozen=frozen_digests(m);protected=protected_receiver_digest(m)
    with torch.no_grad():
        p=m.audio_prior(b['audio_features'],b['valid']);s=m.encode_style(r)['code']
        a=m.decode(b['b0'],p,s,b['valid']);z=m.decode(b['b0'],p,s,b['valid'],posture_grad_mask=b['emotion_id'].eq(0))
        torch.testing.assert_close(a,z,rtol=0,atol=0)
    # All emotional queries cannot teach the neutral offset, but they still
    # teach dynamic reference modulation and full receiver when requested.
    b['emotion_id'].fill_(1);loss,_=objective(m,b,r,neutral_id=0);loss.backward();mask_reference_gradients(m,scope);mask_posture_gradients(m,b,0)
    assert m.decoder.bias.weight.grad is None and m.decoder.bias.bias.grad is None
    assert all(p.grad is None for p in m.style.posture.parameters())
    assert any(mod.weight.grad[:,-4:].abs().sum()>0 for mod in m.decoder.modulations)
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith(('style.','decoder.')))
    opt.step();assert frozen_digests(m)==frozen
    if scope=='style_only':assert protected_receiver_digest(m)==protected
    opt.zero_grad(set_to_none=True);loss,_=objective(m,b,r,neutral_id=0);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.style.response.parameters())
    mask_reference_gradients(m,scope);mask_posture_gradients(m,b,0);opt.step()
    opt.zero_grad(set_to_none=True);b['emotion_id'].fill_(0)
    loss,_=objective(m,b,r,neutral_id=0);loss.backward()
    assert m.decoder.bias.weight.grad.abs().sum()>0
    mask_reference_gradients(m,scope);opt.step();opt.zero_grad(set_to_none=True)
    loss,_=objective(m,b,r,neutral_id=0);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.style.posture.parameters())
    mask_reference_gradients(m,scope);opt.step();opt.zero_grad(set_to_none=True)
    snapshot={n:p.clone() for n,p in m.named_parameters() if n.startswith(('style.posture.','decoder.bias.'))}
    b['emotion_id'].fill_(1);loss,_=objective(m,b,r,neutral_id=0);loss.backward()
    mask_reference_gradients(m,scope);mask_posture_gradients(m,b,0);opt.step()
    assert all(torch.equal(v,dict(m.named_parameters())[n]) for n,v in snapshot.items())


def test_large_reference_does_not_saturate_affect_gain_and_padding_is_clean():
    _,m,b,r=setup();d=m.decoder;m.eval()
    with torch.no_grad():
        first=d.modulations[0];first.weight.zero_();first.bias.zero_()
        first.weight[:m.cfg.decoder_hidden,0]=.25
        first.weight[:m.cfg.decoder_hidden,m.cfg.global_dim+m.cfg.local_dim:]=1000.
        g=torch.zeros(2,m.cfg.global_dim);u=torch.zeros(2,9,m.cfg.local_dim);style=torch.ones(2,m.cfg.style_dim)
        valid=b['valid'].clone();valid[0,-2:]=False;seen=[]
        handle=d.blocks[0].register_forward_pre_hook(lambda _,args:seen.append(args[0].clone()))
        d(b['b0'],g,u,style,valid,m.scales);g[:,0]=1.;d(b['b0'],g,u,style,valid,m.scales);handle.remove()
        # The style gate is saturated, yet the affect gain retains sensitivity.
        assert (seen[1]-seen[0])[valid].abs().max()>1e-5
        assert not seen[1][~valid].count_nonzero()


def test_incompatible_configuration_refused():
    with pytest.raises(ValueError,match='statistical'):
        ExpressionResponse(ResponseConfig(style_modulation='factorized'),torch.zeros(772),torch.ones(772),torch.ones(52))
