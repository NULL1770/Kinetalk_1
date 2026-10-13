"""Actual decoder gradients, strict migrations and no privileged deployment."""
import torch
import pytest
from kinetalk_v5.affect_trend import V5AffectTrend
from tests.test_v5_factor_rebuild import setup


def trend_setup():
    _,old,b,refs=setup()
    return old,V5AffectTrend.from_rebuild(old.checkpoint()).eval(),b,refs


def test_zero_start_equals_rebuild_including_sampling_and_semantics():
    old,model,b,refs=trend_setup()
    args=(b['audio_features'],b['h0'],b['b0'],b['valid'],refs)
    with torch.no_grad():
        a=old.predict(*args,seed=91); z=model.predict(*args,seed=91)
    assert torch.equal(a['motion'],z['motion'])
    for k in ('emotion_logits','intensity_logits','state'):
        assert torch.equal(a['prediction'][k],z['prediction'][k])


def test_teacher_anchor_does_not_follow_the_trained_mean_adapter():
    _,model,b,refs=trend_setup(); before=model.visual_teacher(b)
    with torch.no_grad(): model.audio_trend.network[-1].bias.fill_(.5)
    after=model.visual_teacher(b)
    assert torch.equal(before['state'],after['state'])
    p=model.affect_audio(b['audio_features'],b['valid'])
    assert p['trend_correction'][b['valid']].abs().sum()>0
    assert torch.equal(p['trend_correction'][~b['valid']],torch.zeros_like(p['trend_correction'][~b['valid']]))


def test_real_motion_reconstruction_reaches_trend_without_decoder_drift():
    _,model,b,refs=trend_setup(); model.requires_grad_(False);model.audio_trend.requires_grad_(True)
    p=model.affect_audio(b['audio_features'],b['valid'])
    y=model.decode(b,p,model.encode_style(refs),mode='modal')['motion']
    (y-b['motion']).square()[b['valid']].mean().backward()
    assert model.audio_trend.network[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('audio_trend.'))


def test_deployment_uses_audio_only_and_roundtrip_is_strict():
    _,model,b,refs=trend_setup()
    with torch.no_grad(): model.audio_trend.network[-1].weight.normal_(std=.01)
    restored=V5AffectTrend.from_checkpoint(model.checkpoint()).eval()
    model.visual_teacher=lambda _:(_ for _ in ()).throw(AssertionError('privileged input at deployment'))
    args=(b['audio_features'],b['h0'],b['b0'],b['valid'],refs)
    with torch.no_grad(): assert torch.equal(model.predict(*args)['motion'],restored.predict(*args)['motion'])
    bad=model.checkpoint();bad['model'].pop('core.student.state.bias')
    with pytest.raises(RuntimeError): V5AffectTrend.from_checkpoint(bad)


def test_rebuild_extension_does_not_ignore_arbitrary_missing_keys():
    old,_,_,_=trend_setup();ck=old.checkpoint();ck['model'].pop('reference_context.output.bias')
    with pytest.raises(RuntimeError): V5AffectTrend.from_rebuild(ck)
