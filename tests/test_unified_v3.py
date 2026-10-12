import torch
from kinetalk_v3.unified_model_v3 import (
    UnifiedV3Config, UnifiedExpressionV3Model, teacher_targets,
    v3_supervision_losses,
)

torch.set_num_threads(2)


def fixture():
    torch.manual_seed(47)
    cfg = UnifiedV3Config(content_dim=16, hidden=32, decoder_hidden=48)
    model = UnifiedExpressionV3Model(cfg, torch.zeros(772), torch.ones(772), torch.ones(52) * .1)
    b, t = 2, 17
    valid = torch.ones(b, t, dtype=torch.bool); valid[1, -3:] = False
    base = torch.full((b, t, 52), .05); motion = base.clone()
    motion[:, 5:9, 0] = .8; motion[:, 5:9, 7] = .8
    batch = {'audio_features': torch.randn(b, t, 1540), 'h0': torch.randn(b, t, 16),
             'b0': base, 'motion': motion, 'valid': valid,
             'channel_mask': torch.tensor([[True] * 51 + [False]] * b),
             'times': torch.arange(t).double()[None].expand(b, -1) / 25,
             'emotion_id': torch.tensor([5, -1]), 'intensity_id': torch.tensor([2, -1]),
             'intensity_valid': torch.tensor([True, False])}
    refs = {'motion': motion.flip(1), 'valid': torch.ones_like(valid),
            'channel_mask': batch['channel_mask']}
    targets = teacher_targets(motion, base, valid, batch['channel_mask'], batch['times'])
    return model, batch, refs, targets


def test_v3_student_has_no_text_and_dual_rate_paths():
    model, batch, refs, targets = fixture()
    out = model.training_forward(batch, refs, targets)
    assert out['motion'].shape == batch['motion'].shape
    assert out['student']['state_slow'].shape == batch['motion'].shape[:2] + (10,)
    assert out['articulatory_residual'].shape == batch['motion'].shape
    assert torch.equal(out['motion'][~batch['valid']], torch.zeros_like(out['motion'][~batch['valid']]))


def test_v3_teacher_student_gradient_and_rollout():
    model, batch, refs, targets = fixture()
    out = model.training_forward(batch, refs, targets)
    loss = v3_supervision_losses(model, out, targets, batch)['total'] + (out['motion'] - batch['motion']).square().mean()
    loss.backward()
    assert model.student.state.weight.grad.abs().sum() > 0
    assert model.articulatory.coeff.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        a = model.predict(batch['audio_features'], batch['h0'], batch['b0'], batch['valid'], refs,
                          times=batch['times'], seed=91)
        b = model.predict(batch['audio_features'], batch['h0'], batch['b0'], batch['valid'], refs,
                          times=batch['times'], seed=92)
    assert not torch.equal(a['schedule']['events'], b['schedule']['events'])


def test_v3_emotion_override_reaches_state_and_events():
    model, batch, refs, _ = fixture(); model.eval()
    happy = torch.zeros(2, 8); happy[:, 5] = 1
    with torch.no_grad():
        a = model.predict(batch['audio_features'], batch['h0'], batch['b0'], batch['valid'], refs,
                          times=batch['times'], seed=47)
        b = model.predict(batch['audio_features'], batch['h0'], batch['b0'], batch['valid'], refs,
                          times=batch['times'], seed=47, emotion=happy)
    assert (a['student']['state'] - b['student']['state']).abs().sum() > 0
    assert (a['motion'] - b['motion']).abs().sum() > 0
