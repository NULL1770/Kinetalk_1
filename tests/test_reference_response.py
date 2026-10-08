import pytest
import torch
from kinetalk_b0.models.expression_response import ExpressionResponse,ResponseConfig,reference_statistics
from scripts.train_reference_response import (candidate,freeze_expression,frozen_digests,objective,
    support_pool,choose_supports)

torch.set_num_threads(2)


def fixture(mode='statistics'):
    torch.manual_seed(71)
    old=ExpressionResponse(ResponseConfig(hidden=16,decoder_hidden=24,global_dim=8,local_dim=4,style_dim=8),
                           torch.zeros(772),torch.ones(772),torch.ones(52)*.2)
    model=candidate(old,mode)
    b=dict(audio_features=torch.rand(2,9,1540),b0=torch.rand(2,9,52),motion=torch.rand(2,9,52),
           valid=torch.ones(2,9,dtype=torch.bool),channel_mask=torch.ones(2,52,dtype=torch.bool),
           times=torch.arange(9,dtype=torch.float64)[None].expand(2,-1)*.04)
    r=dict(motion=torch.rand(2,2,7,52),b0=torch.rand(2,2,7,52),valid=torch.ones(2,2,7,dtype=torch.bool),
           channel_mask=torch.ones(2,2,52,dtype=torch.bool))
    return old,model,b,r


def test_partition_posture_changes_only_static_output_and_response_can_change_dynamics():
    _,m,b,r=fixture();m.eval()
    with torch.no_grad():
        m.decoder.bias.weight.normal_(0,.1)
        for mod in m.decoder.modulations:mod.weight[:,-4:].normal_(0,.1)
        style=m.encode_style(r)['code'];p=m.audio_prior(b['audio_features'],b['valid'])
        y=m.decode(b['b0'],p,style,b['valid'])
        posture=style.clone();posture[:,:4]+=.5
        difference=m.decode(b['b0'],p,posture,b['valid'])-y
        assert difference.abs().sum()>0
        torch.testing.assert_close(difference[:,1:],difference[:,:1].expand(-1,8,-1),rtol=1e-5,atol=1e-7)
        response=style.clone();response[:,4:]+=.5
        difference=m.decode(b['b0'],p,response,b['valid'])-y
        assert (difference[:,1:]-difference[:,:-1]).abs().max()>1e-6


def test_statistics_padding_missing_channel_and_absent_reference_do_not_change_result():
    _,m,b,r=fixture()
    r['channel_mask'][:,:,51]=False
    expected=m.encode_style(r)['code']
    r['motion'][:,:,:,51]=float('nan');r['b0'][:,:,:,51]=float('nan')
    for k in ('motion','b0'):
        r[k]=torch.cat((r[k],torch.full((2,2,3,52),float('nan'))),2)
    r['valid']=torch.cat((r['valid'],torch.zeros(2,2,3,dtype=torch.bool)),2)
    for k,v in list(r.items()):
        r[k]=torch.cat((v,torch.zeros_like(v[:,:1]) if v.dtype==torch.bool else torch.full_like(v[:,:1],float('nan'))),1)
    torch.testing.assert_close(m.encode_style(r)['code'],expected,rtol=1e-5,atol=1e-6)
    reversed_refs={k:v.flip(1) for k,v in r.items()}
    torch.testing.assert_close(m.encode_style(reversed_refs)['code'],expected,rtol=1e-5,atol=1e-6)
    r['motion'][0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):m.encode_style(r)


@pytest.mark.parametrize('mode',['temporal','statistics'])
def test_only_reference_and_decoder_receive_motion_gradients_and_content_nan_is_ignored(mode):
    old,m,b,r=fixture(mode)
    before=frozen_digests(old);assert frozen_digests(m)==before
    params=freeze_expression(m);optimizer=torch.optim.AdamW(params,lr=1e-3)
    loss,_=objective(m,b,r);loss.backward()
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith(('style.','decoder.')))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.decoder.parameters())
    optimizer.step();optimizer.zero_grad();loss,_=objective(m,b,r);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.style.parameters())
    assert frozen_digests(m)==before
    with torch.no_grad():
        y=m.predict(b['audio_features'],b['b0'],b['valid'],r)
        b['audio_features'][...,:768]=float('nan')
        torch.testing.assert_close(y,m.predict(b['audio_features'],b['b0'],b['valid'],r),rtol=0,atol=0)
        restored=ExpressionResponse(ResponseConfig(**m.checkpoint_config()),m.feature_mean,m.feature_std,m.scales)
        restored.load_state_dict(m.state_dict(),strict=True);freeze_expression(restored)
        torch.testing.assert_close(y,restored.predict(b['audio_features'],b['b0'],b['valid'],r),rtol=0,atol=0)


def test_support_selection_excludes_target_sentence_identity_and_internal_holds():
    q=dict(emotion_id=[0]*10,speaker_id=[0]*5+[1]*5,
           sentence_id=['a','a','b','c','d']*2,clip_id=[str(i) for i in range(10)])
    pool=support_pool(q,[0,1,2,3,5,6,7,8],0)
    rng=torch.Generator().manual_seed(71);state=rng.get_state()
    pairs=choose_supports(q,pool,[0,5],rng)
    for i,p in zip([0,5],pairs.tolist()):
        assert len(set(p))==2
        assert all(q['sentence_id'][j]!=q['sentence_id'][i] and q['speaker_id'][j]==q['speaker_id'][i] and j not in [4,9] for j in p)
    rng.set_state(state);assert torch.equal(pairs,choose_supports(q,pool,[0,5],rng))
    with pytest.raises(ValueError,match='independent'):choose_supports(q,{0:[0,1,2]},[0],rng)
