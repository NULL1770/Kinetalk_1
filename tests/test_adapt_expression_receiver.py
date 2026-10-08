import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from tests.test_expression_response import fixture
from scripts.adapt_expression_receiver import (
    freeze_conditions, frozen_digest, objective, inherited_budget, scientific_args)
from kinetalk_b0.models.expression_response import motion_objective


def example():
    m, a, base, valid, refs, times = fixture()
    m.cfg = replace(m.cfg, center_local=True)
    b = dict(audio_features=a.requires_grad_(), b0=base, valid=valid,
             motion=torch.rand_like(base, requires_grad=True), times=times,
             channel_mask=torch.ones(2,52,dtype=torch.bool))
    return m, b, refs


@pytest.mark.parametrize('mode', ['mixed','deploy'])
def test_only_decoder_changes_and_sampling_optimizer_replay(mode):
    m, b, refs = example()
    params = freeze_conditions(m)
    frozen = frozen_digest(m)
    original = copy.deepcopy(m.state_dict())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    rng = torch.Generator().manual_seed(71)
    def step():
        opt.zero_grad(set_to_none=True)
        loss, _ = objective(m,b,refs,mode,rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step()
        return float(loss.detach())
    step()
    assert frozen_digest(m) == frozen
    assert b['audio_features'].grad is None and b['b0'].grad is None and b['motion'].grad is None
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('decoder.'))
    assert (m.decoder.output.weight.grad.abs().sum(1) > 0).all()  # all mouth channels open
    assert any(not torch.equal(t,m.state_dict()[n]) for n,t in original.items() if n.startswith('decoder.'))
    saved, optim, state = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict()), rng.get_state()
    expected_loss = step()
    expected = copy.deepcopy(m.state_dict())
    m.load_state_dict(saved); opt.load_state_dict(optim); rng.set_state(state)
    assert step() == expected_loss
    assert all(torch.equal(t,m.state_dict()[n]) for n,t in expected.items())


@pytest.mark.parametrize('mode', ['mixed','deploy'])
def test_original_objective_coefficients_and_content_isolation(mode):
    m, b, refs = example(); freeze_conditions(m)
    gen = lambda: torch.Generator().manual_seed(31)
    total, logs = objective(m,b,refs,mode,gen())
    with torch.no_grad():
        style = m.encode_style(refs)['code']
        p = m.audio_prior(b['audio_features'],b['valid'])
        pred = m.decode(b['b0'],p,style,b['valid'])
        loss, _ = motion_objective(pred,b['motion'],b['valid'],b['channel_mask'],b['times'],m.scales)
        expected = 1.5 * loss
        if mode == 'mixed':
            q = m.motion_posterior(b['motion'],b['b0'],style,b['valid'],b['channel_mask'],b['times'])
            y = m.decode(b['b0'],q,style,b['valid'],sample=True,generator=gen())
            qloss,_ = motion_objective(y,b['motion'],b['valid'],b['channel_mask'],b['times'],m.scales)
            expected = qloss + .5 * loss
    torch.testing.assert_close(total,expected,rtol=0,atol=0)
    poisoned = dict(b,audio_features=b['audio_features'].detach().clone())
    poisoned['audio_features'][...,:768] = float('nan')
    torch.testing.assert_close(total,objective(m,poisoned,refs,mode,gen())[0],rtol=0,atol=0)


def test_deploy_never_invokes_teacher_or_labels_and_masks_loss():
    m,b,refs = example(); freeze_conditions(m)
    def forbidden(*args,**kwargs):
        raise AssertionError('Deploy objective accessed teacher/labels')
    m.motion_posterior = forbidden
    m.emotion_head.forward = forbidden; m.intensity_head.forward = forbidden
    b['channel_mask'][:,51] = False
    total,_ = objective(m,b,refs,'deploy')
    changed = b['motion'].detach().clone()
    changed[~b['valid']] = float('nan'); changed[:,:,51] = float('nan')
    actual,_ = objective(m,dict(b,motion=changed),refs,'deploy')
    torch.testing.assert_close(actual,total,rtol=0,atol=0)
    total.backward()
    assert torch.isfinite(m.decoder.output.weight.grad).all()
    with pytest.raises(ValueError): objective(m,b,refs,'unknown')
    with pytest.raises(ValueError): objective(m,b,{k:x[:,:1] for k,x in refs.items()},'deploy')


def test_budget_includes_analytic_parent_and_resume_binds_scientific_args():
    p = dict(epoch=0,step=0,analytic_fit=dict(parent_epochs=24,parent_updates=16368,
        analytic_fit_passes=1,analytic_fit_clips=10903))
    budget = inherited_budget(p)
    assert budget == dict(parent_epochs=24,parent_updates=16368,
                          inherited_analytic_fit_passes=1,inherited_analytic_fit_clips=10903)
    assert inherited_budget(dict(epoch=8,step=5456,**budget))['parent_updates'] == 21824
    a = SimpleNamespace(mode='mixed',resume=False,epochs=8,seed=47)
    original = scientific_args(a); a.resume=True
    assert scientific_args(a) == original
    a.mode='deploy'
    assert scientific_args(a) != original
