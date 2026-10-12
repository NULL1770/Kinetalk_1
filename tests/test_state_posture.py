import torch
import pytest
from dataclasses import replace
from kinetalk_v4.state_posture import AudioPostureAdapter, posture_supervision, load_zero_posture_extension
from tests.test_unified_v4 import fixture
from kinetalk_v4.unified_model_v4 import KineTalkV4Model, v4_losses, teacher_targets


def test_posture_adapter_is_exact_zero_start():
    torch.manual_seed(4)
    head = AudioPostureAdapter(6, 3)
    x = torch.randn(2, 6)
    assert torch.equal(head(x), torch.zeros(2, 3))


def test_teacher_guidance_is_only_applied_on_teacher_advantage():
    target = torch.tensor([[[1., 2.], [1., 2.]], [[1., 2.], [1., 2.]]])
    student = torch.tensor([[[2., 2.], [2., 2.]], [[1.5, 5.], [1.5, 5.]]])
    teacher = torch.tensor([[[1., 3.], [1., 3.]], [[4., 2.], [4., 2.]]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    quality = torch.ones_like(target, dtype=torch.bool)
    out = posture_supervision(student, teacher, target, valid, quality)
    assert out['teacher_accepted_axes'].item() == 2
    assert out['supported_posture_axes'].item() == 4
    assert out['teacher_advantage_fraction'].item() == .5
    assert out['posture_guidance'].item() == 2.5
    assert out['posture_direct'].item() == 2.5625


def test_posture_masks_nan_padding_missing_axes_and_detaches_teacher():
    student = torch.tensor([[[2., float('nan')], [2., float('nan')],
                              [float('nan'), float('nan')]]], requires_grad=True)
    teacher = torch.tensor([[[1., float('nan')], [1., float('nan')],
                              [float('nan'), float('nan')]]], requires_grad=True)
    target = teacher.detach().clone()
    valid = torch.tensor([[True, True, False]])
    quality = torch.tensor([[[True, False], [True, False], [False, False]]])
    out = posture_supervision(student, teacher, target, valid, quality)
    assert out['supported_posture_axes'].item() == 1
    assert out['posture_direct'].item() == 1.
    assert out['posture_guidance'].item() == 1.
    (out['posture_direct'] + out['posture_guidance']).backward()
    assert torch.isfinite(student.grad).all()
    assert torch.equal(student.grad[0, :, 1], torch.zeros(3))
    assert torch.equal(student.grad[0, 2], torch.zeros(2))
    assert teacher.grad is None
    unsupported = posture_supervision(student, teacher, target, valid,
        torch.zeros_like(quality))
    assert unsupported['posture_direct'].item() == 0.
    assert unsupported['posture_guidance'].item() == 0.


def test_checkpoint_extension_preserves_motion_schedule_and_centered_trajectory():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    model.eval()
    with torch.no_grad():
        model.decoder.output.weight.normal_(std=.02)
    upgraded = KineTalkV4Model(replace(cfg, use_posture_adapter=True),
        model.feature_mean, model.feature_std, model.scales, jac, off).eval()
    assert load_zero_posture_extension(upgraded, model.state_dict())
    content = torch.randn(2, 19, cfg.content_dim)
    with torch.no_grad():
        first = model.predict(audio, content, base, valid, refs, seed=8)
        second = upgraded.predict(audio, content, base, valid, refs, seed=8)
    assert torch.equal(first['student']['state'], second['student']['state'])
    assert torch.equal(first['motion'], second['motion'])
    assert torch.equal(first['events'], second['events'])
    with torch.no_grad():
        upgraded.student.posture_adapter.output.bias.fill_(.25)
        shifted = upgraded.audio_state(audio, valid)['state']
    assert torch.allclose(shifted[valid]-second['student']['state'][valid],
                          torch.full_like(shifted[valid], .25), atol=1e-6)
    assert torch.equal(shifted[~valid], torch.zeros_like(shifted[~valid]))
    # A clip-constant correction does not independently invent a dynamic path.
    assert torch.allclose(shifted[0, 1:]-shifted[0, :-1],
        second['student']['state'][0, 1:]-second['student']['state'][0, :-1], atol=1e-6)


def test_checkpoint_extension_rejects_missing_old_weights_or_nonzero_new_head():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    upgraded = KineTalkV4Model(replace(cfg, use_posture_adapter=True),
        model.feature_mean, model.feature_std, model.scales, jac, off)
    broken = dict(model.state_dict()); broken.pop('student.state.weight')
    with pytest.raises(ValueError, match='Checkpoint mismatch'):
        load_zero_posture_extension(upgraded, broken)
    with torch.no_grad():
        upgraded.student.posture_adapter.output.bias.fill_(1.)
    with pytest.raises(ValueError, match='exactly at zero'):
        load_zero_posture_extension(upgraded, model.state_dict())


def test_posture_mode_removes_framewise_teacher_guidance_without_removing_teacher_anchor():
    model, cfg, valid, base, motion, audio, refs, jac, off = fixture()
    times = torch.arange(19, dtype=torch.float64)[None].expand(2, -1) / 25
    batch = dict(audio_features=audio, h0=torch.randn(2, 19, cfg.content_dim),
        b0=base, motion=motion, valid=valid, times=times,
        channel_mask=torch.ones(2, 52, dtype=torch.bool), emotion_id=torch.tensor([1, 2]),
        intensity_id=torch.tensor([2, 1]), intensity_valid=torch.tensor([True, True]))
    targets = teacher_targets(motion, base, valid, batch['channel_mask'], times)
    out = model.training_forward(batch, refs, targets)
    frame = v4_losses(model, out, targets, batch)
    model.cfg = replace(cfg, teacher_guidance_mode='posture')
    posture = v4_losses(model, out, targets, batch)
    assert torch.allclose(frame['total']-posture['total'], .5*frame['guidance'], atol=1e-6)
    assert torch.equal(frame['teacher_anchor'], posture['teacher_anchor'])
    assert torch.equal(frame['position'], posture['position'])
    posture['total'].backward()
    assert model.teacher.state.weight.grad.abs().sum() > 0
    assert model.student.state.weight.grad.abs().sum() > 0
