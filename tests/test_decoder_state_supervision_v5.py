"""Check that geometry-state supervision reaches the deployed conditioning path."""
from dataclasses import replace
import torch
from kinetalk_v2.unified_model import teacher_targets
from kinetalk_v4.unified_model_v4 import KineTalkV4Model, v4_losses
from tests.test_unified_v4 import fixture


def batch_fixture():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    times = torch.arange(19, dtype=torch.float64)[None].expand(2, -1) / 25
    channels = torch.ones(2, 52, dtype=torch.bool)
    batch = dict(audio_features=audio, h0=torch.randn(2, 19, cfg.content_dim), b0=base,
        motion=motion, valid=valid, times=times, channel_mask=channels,
        emotion_id=torch.tensor([1, 2]), intensity_id=torch.tensor([1, 2]),
        intensity_valid=torch.ones(2, dtype=torch.bool))
    targets = teacher_targets(motion, base, valid, channels, times)
    return model, cfg, refs, batch, targets, jac, off


def test_decoder_coordinate_loss_reaches_emotion_map_and_intensity():
    model, cfg, refs, batch, targets, jac, off = batch_fixture()
    model.cfg = replace(cfg, state_supervision_space='decoder', teacher_guidance_mode='none')
    with torch.no_grad():
        model.emotion_to_state.weight.fill_(.3)
    out = model.training_forward(batch, refs, targets)
    loss = v4_losses(model, out, targets, batch)['state_direct']
    loss.backward()
    assert model.emotion_to_state.weight.grad.abs().sum() > 0
    assert model.student.intensity.weight.grad.abs().sum() > 0
    assert model.student.state.weight.grad.abs().sum() > 0
    assert model.teacher.state.weight.grad is None
    for parameter in model.student.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_raw_loss_and_teacher_anchor_do_not_transform_the_motion_teacher():
    model, cfg, refs, batch, targets, jac, off = batch_fixture()
    out = model.training_forward(batch, refs, targets)
    original = v4_losses(model, out, targets, batch)
    out['conditioned']['state'] = out['conditioned']['state'] + 10.
    legacy = v4_losses(model, out, targets, batch)
    assert torch.equal(original['state_direct'], legacy['state_direct'])
    assert torch.equal(original['teacher_anchor'], legacy['teacher_anchor'])
    model.cfg = replace(cfg, state_supervision_space='decoder')
    corrected = v4_losses(model, out, targets, batch)
    assert corrected['state_direct'] > original['state_direct']
    assert torch.equal(corrected['teacher_anchor'], original['teacher_anchor'])


def test_coordinate_protocol_changes_no_parameters_or_initial_predictions():
    old, cfg, refs, batch, targets, jac, off = batch_fixture()
    new = KineTalkV4Model(replace(cfg, state_supervision_space='decoder'), old.feature_mean,
        old.feature_std, old.scales, jac, off)
    new.load_state_dict(old.state_dict(), strict=True)
    old.eval(); new.eval()
    inputs = (batch['audio_features'], batch['h0'], batch['b0'], batch['valid'], refs)
    with torch.no_grad():
        a = old.predict(*inputs, times=batch['times'], seed=47)
        b = new.predict(*inputs, times=batch['times'], seed=47)
    for key in ('motion', 'events', 'geometry_target'):
        assert torch.equal(a[key], b[key])
    assert set(old.state_dict()) == set(new.state_dict())
