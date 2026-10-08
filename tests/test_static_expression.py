import inspect
from types import SimpleNamespace
import pytest
import torch
from kinetalk_b0.models.static_expression import (
    neutral_reference_mean, receiver_features, fit_static_correction, StaticExpressionCorrection)


def test_reference_equal_weight_order_padding_and_missing_channel():
    m = torch.zeros(1, 2, 4, 52)
    m[:, 0, :2] = 2
    m[:, 1] = 4
    v = torch.tensor([[[True, True, False, False], [True]*4]])
    c = torch.ones(1, 2, 52, dtype=torch.bool)
    c[:, 1, 3] = False
    m[0, 0, 2:] = float('nan')
    m[0, 1, :, 3] = float('nan')
    r = dict(motion=m, b0=torch.zeros_like(m), valid=v, channel_mask=c)
    a = neutral_reference_mean(r, torch.ones(52))
    assert a[0, 0] == 3 and a[0, 3] == 2
    b = neutral_reference_mean({k:x.flip(1) for k,x in r.items()}, torch.ones(52))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    r['motion'][0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError): neutral_reference_mean(r, torch.ones(52))


def test_fit_generalizes_affine_target_and_preserves_raw_dynamics():
    rng = torch.Generator().manual_seed(41)
    x = torch.randn(240, 5, generator=rng)
    w = torch.randn(5, 52, generator=rng)
    y = x@w+.3
    support = torch.ones_like(y,dtype=torch.bool)
    support[:, -1] = False
    y[:, -1] = float('nan')
    fitted = fit_static_correction(x,y,support,'latent',1e-6)
    query = torch.randn(15, 5, generator=rng)
    torch.testing.assert_close(fitted.offset(query)[:,:51], (query@w+.3)[:,:51], atol=2e-5, rtol=1e-4)
    assert fitted.offset(query)[:,-1].eq(0).all()
    pred = torch.randn(15, 8, 52, generator=rng)
    mask = torch.ones(15,8,dtype=torch.bool); mask[:,3]=False
    corrected = fitted(pred,query,mask,torch.ones(52))
    adjacent = mask[:,1:] & mask[:,:-1]
    torch.testing.assert_close((corrected[:,1:]-corrected[:,:-1])[adjacent],
        (pred[:,1:]-pred[:,:-1])[adjacent],atol=2e-6,rtol=1e-5)
    assert corrected[~mask].eq(0).all()


def test_features_are_detached_and_no_query_gt_api():
    model = SimpleNamespace(scales=torch.ones(52), emotion_head=torch.nn.Linear(32,8))
    prior = {'g_mean':torch.randn(2,32,requires_grad=True)}
    style = torch.randn(2,64,requires_grad=True)
    refs = dict(motion=torch.randn(2,2,4,52,requires_grad=True),b0=torch.zeros(2,2,4,52),
                valid=torch.ones(2,2,4,dtype=torch.bool),channel_mask=torch.ones(2,2,52,dtype=torch.bool))
    for mode,dim in [('latent',96),('reference',564)]:
        x = receiver_features(model,prior,style,refs,mode)
        assert x.shape==(2,dim) and not x.requires_grad
    assert list(inspect.signature(StaticExpressionCorrection.predict).parameters)==[
        'self','model','audio','base','valid','refs']


def test_saved_correction_round_trip(tmp_path):
    x=torch.randn(15,4); y=torch.randn(15,52); support=torch.ones_like(y,dtype=torch.bool)
    a=fit_static_correction(x,y,support,'latent')
    p=tmp_path/'static.pt';torch.save(dict(mode=a.mode,state=a.state_dict()),p)
    ck=torch.load(p,weights_only=True);b=StaticExpressionCorrection(ck['mode'],**ck['state'])
    torch.testing.assert_close(a.offset(x),b.offset(x),rtol=0,atol=0)
