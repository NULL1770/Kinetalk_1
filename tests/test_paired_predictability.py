import numpy as np
import pytest
import torch
from scripts.diagnose_paired_predictability import (
    ChannelRidge, coordinate_targets, score, summarize, views)


def fixture():
    torch.manual_seed(7)
    x = torch.randn(3, 8, 3, dtype=torch.float64)
    y = torch.randn(3, 8, 4, 3, dtype=torch.float64)
    mask = torch.ones(3, 8, 4, dtype=torch.bool)
    mask[0, 3:] = False  # unequal lengths must not weight this clip less
    mask[1, 2:4, 1:3] = False  # channel-specific event mask
    mask[:, :, 3] = False
    times = torch.arange(8, dtype=torch.float64)[None].expand(3,-1)*.04
    return x, y, mask, times


@pytest.mark.parametrize('kind', ['centered', 'displacement'])
def test_grouped_ridge_agrees_with_independent_channel_numpy(kind):
    x, y, mask, times = fixture()
    ridge = ChannelRidge(3, channels=4)
    ridge.add(x,y,mask,times,kind)
    w, energy = ridge.solve()
    for c in range(3):
        xx, xy, yy, count = np.zeros((3,3)), np.zeros((3,3)), np.zeros(3), 0
        for j in range(3):
            valid = mask[j,:,c].numpy()
            a,b = x[j].numpy(),y[j,:,c].numpy()
            if kind == 'centered':
                a,b = a[valid],b[valid]
                a,b = a-a.mean(0),b-b.mean(0)
            else:
                valid = valid[1:] & valid[:-1]
                a,b = np.diff(a,axis=0)[valid],np.diff(b,axis=0)[valid]
            if len(a):
                xx += a.T@a/len(a);xy += a.T@b/len(a);yy += (b*b).mean(0);count += 1
        xx/=count;xy/=count
        scale = np.sqrt(np.maximum(np.diag(xx),1e-8))
        expected = np.linalg.solve(xx/scale[:,None]/scale[None]+.001*np.eye(3),xy/scale[:,None])/scale[:,None]
        np.testing.assert_allclose(w[c],expected,atol=1e-10)
        np.testing.assert_allclose(energy[c],yy/count,atol=1e-10)
    assert ridge.count[3] == 0
    assert not w[3].any() and not energy[3].any()


def test_gaps_masks_and_nan_never_bridge():
    x=torch.tensor([0.,1.,99.,3.,4.,10.])[None,:,None]
    y=x[:,:,None].expand(1,6,2,3).clone()
    mask=torch.ones(1,6,2,dtype=torch.bool);mask[:,2]=False;mask[:,:,1]=False
    x[:,2]=float('nan');y[:,2]=float('nan')
    times=torch.tensor([[0.,.04,.08,.12,.16,.28]])
    groups=list(views(x,y,mask,times,'displacement'))
    assert len(groups)==1
    ids,a,b,valid,count=groups[0]
    assert ids.tolist()==[0] and count.tolist()==[2]
    torch.testing.assert_close(a[valid],torch.ones(2,1,dtype=torch.float64))
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    mask[:,2,0]=True
    with pytest.raises(ValueError,match='Nonfinite observed'):
        list(views(x,y,mask,times,'centered'))


def test_each_channel_centers_on_its_own_observed_frames():
    x=torch.arange(5.)[None,:,None]
    y=x[:,:,None].expand(1,5,2,3)+torch.tensor([2.,9.])[None,None,:,None]
    mask=torch.ones(1,5,2,dtype=torch.bool);mask[:,2:,1]=False
    times=torch.arange(5.)[None]*.04
    groups=list(views(x,y,mask,times,'centered'))
    assert len(groups)==2
    for ids,a,b,valid,count in groups:
        assert torch.allclose(a.sum(1),torch.zeros_like(a.sum(1)))
        assert torch.allclose(b.sum(1),torch.zeros_like(b.sum(1)))
        torch.testing.assert_close(a[...,None].expand_as(b),b)


def test_physical_components_sum_with_nonorthogonal_values():
    x=torch.tensor([[[.2,.9],[.4,.6]]]);n=x+.7;b=x-.1
    mask=torch.ones_like(x,dtype=torch.bool)
    y=coordinate_targets(x,n,b,mask)
    torch.testing.assert_close(y[...,0]+y[...,1],y[...,2],rtol=1e-13,atol=1e-13)
    assert (y[...,0].abs()>y[...,2].abs()).all()


def test_score_does_not_refit_or_change_normalization_and_excludes_missing_channels():
    x,y,mask,times=fixture()
    fit=ChannelRidge(3,channels=4);fit.add(x,y,mask,times,'centered')
    weight,energy=fit.solve();before=(weight.clone(),energy.clone(),fit.xx.clone())
    error,zero,support=score(x*10,y+100,mask,times,'centered',weight)
    assert error.shape==(3,4,3) and not support[:,3].any()
    assert torch.isfinite(error).all() and torch.isfinite(zero).all()
    for a,b in zip(before,(weight,energy,fit.xx)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_perfect_predictor_and_wrong_predictor_have_expected_errors():
    x=torch.arange(5.,dtype=torch.float64)[None,:,None]
    y=x[:,:,None].expand(1,5,2,3).clone()
    mask=torch.ones(1,5,2,dtype=torch.bool);t=x[:,:,0]*.04
    w=torch.ones(2,1,3,dtype=torch.float64)
    for kind in ('centered','displacement'):
        error,zero,_=score(x,y,mask,t,kind,w)
        assert error.max()==0 and zero.min()>0
        wrong,_,_=score(x,y,mask,t,kind,-w)
        torch.testing.assert_close(wrong,4*zero)


def test_normalized_summary_does_not_inflate_constant_channels():
    e=np.ones((2,52,3));z=2*e;s=np.ones((2,52),bool);s[:,51]=False
    energy=np.ones((52,3));energy[0]=0
    r=summarize(e,z,s,energy)['all51']['expression_difference']
    assert r['mse']==1 and r['zero_mse']==2 and r['energy_reduction']==.5
    assert r['train_energy_normalized_mse']==1 and r['normalized_active_channels']==50


@pytest.mark.parametrize('case',['bad_time','nan_observed','wrong_mask'])
def test_invalid_native_inputs_rejected(case):
    x,y,mask,times=fixture();times=times.clone()
    if case=='bad_time':times[0,1]=-.1
    elif case=='nan_observed':y[0,1,0,0]=float('nan')
    else:mask=mask.float()
    with pytest.raises(ValueError):list(views(x,y,mask,times,'displacement'))
