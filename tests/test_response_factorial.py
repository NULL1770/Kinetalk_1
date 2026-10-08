"""Phase43 transforms preserve clocks, gradient roles, and the legacy default."""
from dataclasses import replace
import io
import pytest
import torch
from tests.test_expression_response import fixture
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig,masked_pool
from scripts.train_expression_response import training_reference_views,repeated_query


def centered_fixture():
    m,a,b,v,r,t=fixture()
    m.cfg=replace(m.cfg,center_local=True)
    return m,a,b,v,r,t


@pytest.mark.parametrize('sample',[False,True])
def test_centering_is_shift_invariant_masked_and_replayable(sample):
    m,a,b,v,r,t=centered_fixture();v[0,3:5]=False
    p=m.audio_prior(a,v)
    shifted={k:x.clone() for k,x in p.items()}
    shifted['u_mean']=shifted['u_mean']+torch.tensor([2.,-1.,3.,.5])
    shifted['u_mean'][~p['u_mask']]=float('nan')
    rng=lambda:torch.Generator().manual_seed(27)
    g,u=m.conditions(p,v,sample,rng());gg,uu=m.conditions(shifted,v,sample,rng())
    torch.testing.assert_close(g,gg,rtol=0,atol=0)
    torch.testing.assert_close(u,uu,rtol=0,atol=1e-6)
    torch.testing.assert_close(masked_pool(u,v),torch.zeros_like(g[:,:4]),rtol=0,atol=2e-7)
    assert not u[~v].any()
    # Appending masked frames cannot change the native timeline or centering.
    vp=torch.cat((v,torch.zeros(2,4,dtype=torch.bool)),1)
    _,up=m.conditions(p,vp,sample,rng())
    torch.testing.assert_close(u,up[:,:v.shape[1]],rtol=0,atol=2e-7)
    assert not up[:,v.shape[1]:].any()
    # Projection removes an offset, never changes adjacent differences/timing.
    m.cfg=replace(m.cfg,center_local=False)
    _,raw=m.conditions(p,v,sample,rng())
    adjacent=v[:,1:]&v[:,:-1]
    torch.testing.assert_close((u[:,1:]-u[:,:-1])[adjacent],
                              (raw[:,1:]-raw[:,:-1])[adjacent],rtol=1e-5,atol=2e-7)


def test_centered_model_keeps_global_and_temporal_gradients_and_no_content():
    m,a,b,v,r,t=centered_fixture();s=m.encode_style(r)['code']
    p=m.audio_prior(a,v);p['g_mean'].retain_grad();p['u_mean'].retain_grad()
    y=m.decode(b,p,s,v);(y*torch.randn_like(y)).sum().backward()
    assert p['g_mean'].grad.abs().sum()>0 and p['u_mean'].grad.abs().sum()>0
    assert b.grad is None
    # A uniform local shift is a null direction of the generation transform.
    torch.testing.assert_close(p['u_mean'].grad.sum(1),torch.zeros(2,4),rtol=0,atol=1e-7)
    a[...,:768]=float('nan')
    torch.testing.assert_close(y,m.predict(a,b,v,r),rtol=0,atol=0)


def test_default_initialization_and_centered_checkpoint_roundtrip():
    m,a,b,v,r,t=fixture();state=torch.get_rng_state()
    cfg=replace(m.cfg,center_local=True)
    torch.manual_seed(19)
    other=ExpressionResponse(cfg,m.feature_mean,m.feature_std,m.scales)
    for k,x in m.state_dict().items():assert torch.equal(x,other.state_dict()[k])
    other.load_state_dict(m.state_dict());y=other.predict(a,b,v,r)
    f=io.BytesIO();torch.save({'config':other.checkpoint_config(),'model':other.state_dict()},f);f.seek(0)
    ck=torch.load(f,weights_only=True)
    restored=ExpressionResponse(ResponseConfig(**ck['config']),m.feature_mean,m.feature_std,m.scales)
    restored.load_state_dict(ck['model'])
    torch.testing.assert_close(y,restored.predict(a,b,v,r),rtol=0,atol=0)
    legacy=m.checkpoint_config();legacy.pop('center_local')
    assert not ResponseConfig(**legacy).center_local


@pytest.mark.parametrize('step',[0,1,48,49])
def test_mixed_refs_match_deployment_and_single_views_without_query_reordering(step):
    m,a,b,v,refs,t=fixture();rng=torch.get_rng_state()
    mixed=training_reference_views(refs,'mixed',step)
    assert torch.equal(rng,torch.get_rng_state())
    style=m.encode_style(mixed)['code']
    both=m.encode_style(refs)['code']
    single=m.encode_style({k:x[:,step%2:step%2+1] for k,x in refs.items()})['code']
    torch.testing.assert_close(style[0::2],both,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(style[1::2],single,rtol=1e-5,atol=1e-6)
    assert not mixed['valid'][1::2,1].any() and not mixed['channel_mask'][1::2,1].any()
    for k in ('motion','b0'):mixed[k][1::2,1]=float('nan')
    torch.testing.assert_close(style,m.encode_style(mixed)['code'],rtol=0,atol=0)
    q=repeated_query({'speaker_id':torch.tensor([10,20]),'clip_id':['a','b']})
    assert q['speaker_id'].tolist()==[10,10,20,20] and q['clip_id']==['a','a','b','b']
    legacy=training_reference_views(refs)
    for k,x in refs.items():assert torch.equal(legacy[k],x.flatten(0,1).unsqueeze(1))
    flipped={k:x.flip(1) for k,x in refs.items()}
    flipcode=m.encode_style(training_reference_views(flipped,'mixed',1-step%2))['code']
    torch.testing.assert_close(style,flipcode,rtol=1e-5,atol=1e-6)


def test_invalid_reference_mode_rejected():
    refs=fixture()[4]
    with pytest.raises(ValueError):training_reference_views(refs,'unexpected')
    with pytest.raises(ValueError):training_reference_views({k:x[:,:1] for k,x in refs.items()},'mixed')
