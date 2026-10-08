from dataclasses import replace
import pytest
import torch
from tests.test_expression_response import fixture
from scripts.calibrate_expression_heads import replace_means
from scripts.diagnose_expression_targets import frame_features
from kinetalk_b0.models.expression_response import masked_pool


@pytest.mark.parametrize('mode',['g','u','gu'])
def test_mean_slices_native_gaps_covariance_and_content_isolation(mode):
    m,a,base,valid,refs,times=fixture();m.cfg=replace(m.cfg,center_local=True);m.eval()
    valid[0,3:5]=False
    a[~valid]=float('nan')
    fitted={'global':torch.randn(m.cfg.hidden+1,m.cfg.global_dim),
            'hidden':torch.randn(m.cfg.hidden,m.cfg.local_dim)}
    before={k:v.clone() for k,v in m.state_dict().items()}
    prior=m.audio_prior(a,valid)
    receipt=replace_means(m,fitted,mode)
    assert receipt['passed'] and receipt['local_bias_preserved']
    after=m.audio_prior(a,valid)
    for k in ('g_logvar','u_logvar','u_mask'):assert torch.equal(prior[k],after[k])
    assert torch.equal(before['prior.local_head.bias'],m.prior.local_head.bias)
    h=m.prior.encoder((a[...,768:]-m.feature_mean)/m.feature_std,valid)
    g,u=m.conditions(after,valid)
    if 'g' in mode:torch.testing.assert_close(g,masked_pool(h,valid)@fitted['global'][:-1]+fitted['global'][-1])
    else:assert torch.equal(after['g_mean'],prior['g_mean'])
    if 'u' in mode:
        b={'audio_features':a,'b0':base,'valid':valid}
        torch.testing.assert_close(u,frame_features(m,b,h)['hidden']@fitted['hidden'],atol=2e-6,rtol=1e-5)
    else:assert torch.equal(after['u_mean'],prior['u_mean'])
    poison=a.clone();poison[...,:768]=float('nan')
    for k,v in after.items():assert torch.equal(v,m.audio_prior(poison,valid)[k])


def test_invalid_matrix_fails_before_any_mutation():
    m,*_=fixture();m.cfg=replace(m.cfg,center_local=True)
    before={k:v.clone() for k,v in m.state_dict().items()}
    bad={'global':torch.zeros(m.cfg.hidden+1,m.cfg.global_dim),
         'hidden':torch.full((m.cfg.hidden,m.cfg.local_dim),float('nan'))}
    with pytest.raises(ValueError):replace_means(m,bad,'gu')
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
