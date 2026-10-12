from dataclasses import replace
import pytest
import torch
from kinetalk_v4.acoustic_geometry import geometry_audio_context, load_zero_geometry_extension
from kinetalk_v4.unified_model_v4 import KineTalkV4Model
from tests.test_unified_v4 import fixture


def upgraded(mode):
    old,cfg,valid,base,motion,audio,refs,jac,off=fixture()
    old.eval()
    with torch.no_grad():
        old.decoder.output.weight.normal_(std=.02)
    new=KineTalkV4Model(replace(cfg,geometry_audio_mode=mode),old.feature_mean,
        old.feature_std,old.scales,jac,off).eval()
    assert load_zero_geometry_extension(new,old.state_dict())
    return old,new,cfg,valid,base,motion,audio,refs


@pytest.mark.parametrize('mode',['coarse','hubert'])
def test_zero_geometry_extension_preserves_source_motion_state_and_events(mode):
    old,new,cfg,valid,base,motion,audio,refs=upgraded(mode)
    content=torch.randn(2,19,cfg.content_dim)
    with torch.no_grad():
        first=old.predict(audio,content,base,valid,refs,seed=91)
        second=new.predict(audio,content,base,valid,refs,seed=91)
    for name in ('motion','geometry_target','proposal','events'):
        assert torch.equal(first[name],second[name])
    assert torch.equal(first['student']['state'],second['student']['state'])
    assert sum(p.numel() for p in new.parameters())>sum(p.numel() for p in old.parameters())


def test_hubert_context_excludes_emotion_features_and_handles_nan_padding():
    audio=torch.randn(1,5,1540);content=torch.randn(1,5,12)
    valid=torch.tensor([[True,True,True,False,False]])
    audio[:,3:]=float('nan');content[:,3:]=float('nan')
    context=geometry_audio_context('hubert',audio,content,valid,torch.zeros(772),torch.ones(772))
    changed=audio.clone();changed[... ,768:1536]=100.
    second=geometry_audio_context('hubert',changed,content,valid,torch.zeros(772),torch.ones(772))
    assert torch.equal(context,second)
    assert torch.isfinite(context).all()
    assert torch.count_nonzero(context[:,3:])==0
    with pytest.raises(ValueError,match='full 1540D'):
        geometry_audio_context('hubert',audio[...,768:],content,valid,torch.zeros(772),torch.ones(772))


@pytest.mark.parametrize('mode',['coarse','hubert'])
def test_geometry_receiver_has_geometry_only_native_audio_effect(mode):
    old,new,cfg,valid,base,motion,audio,refs=upgraded(mode)
    content=torch.randn(2,19,cfg.content_dim)
    with torch.no_grad():
        new.decoder.geometry_audio.output.weight.normal_(std=.2)
    changed=audio.clone();changed[..., :768]=torch.randn_like(changed[..., :768])
    a=new.predict(audio,content,base,valid,refs,seed=91)
    b=new.predict(changed,content,base,valid,refs,seed=91)
    assert torch.equal(a['events'],b['events'])
    assert torch.equal(a['proposal'],b['proposal'])
    assert torch.equal(a['student']['state'],b['student']['state'])
    if mode=='coarse':
        assert torch.equal(a['motion'],b['motion'])
    else:
        assert (a['geometry_target'][valid]-b['geometry_target'][valid]).abs().sum()>0
        assert (a['motion'][valid]-b['motion'][valid]).abs().sum()>0
    a['geometry_target'][valid].square().mean().backward()
    assert new.decoder.geometry_audio.output.weight.grad.abs().sum()>0
    assert torch.isfinite(new.decoder.geometry_audio.output.weight.grad).all()


def test_geometry_extension_rejects_unrelated_missing_source_parameters():
    old,new,cfg,valid,base,motion,audio,refs=upgraded('hubert')
    state=dict(old.state_dict());state.pop('student.state.weight')
    with pytest.raises(ValueError,match='Checkpoint mismatch'):
        load_zero_geometry_extension(new,state)
